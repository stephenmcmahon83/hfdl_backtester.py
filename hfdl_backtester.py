import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HF Data Library Intraday Backtester", layout="wide")
st.title("📈 HF Data Library — Intraday Backtesting System")


class HFDataClient:
    BASE_URL = "https://api.hfdatalibrary.com/v1"

    def __init__(self, api_key: str, cache_dir: str = "/tmp/hf_cache"):
        self.api_key = (api_key or "").strip()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "HFDL-Streamlit-Backtester/1.0"})

    def request_safe(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        auth: bool = True,
        stream: bool = False,
        full_url: Optional[str] = None,
    ) -> Tuple[Optional[requests.Response], Optional[str]]:
        """
        Never raises RuntimeError so Streamlit Cloud never redacts the error message.
        Returns (response, error_string).
        """
        url = full_url or f"{self.BASE_URL}{endpoint}"
        headers = {"X-API-Key": self.api_key} if (auth and not full_url) else {}

        try:
            for attempt in range(3):
                resp = self.session.get(url, headers=headers, params=params, stream=stream, timeout=60)

                if not resp.content and "Content-Type" not in resp.headers:
                    return None, f"Empty response with no Content-Type from {url}. Check proxy/VPN."

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 10))
                    time.sleep(min(retry_after, 20))
                    continue

                if resp.status_code != 200:
                    try:
                        detail = json.dumps(resp.json())
                    except Exception:
                        detail = resp.text[:500]
                    return None, f"HTTP {resp.status_code} on {url} — Server message: {detail}"

                return resp, None

            return None, f"HTTP 429 Rate Limit exceeded after 3 retries on {url}."
        except Exception as exc:
            return None, f"Network/Request Exception on {url}: {type(exc).__name__}: {exc}"

    def list_symbols(self) -> Tuple[pd.DataFrame, Optional[str]]:
        resp, err = self.request_safe("/symbols", auth=False)
        if err or resp is None:
            return pd.DataFrame(), err
        return pd.DataFrame(resp.json().get("symbols", [])), None

    def get_symbol_meta(self, ticker: str) -> Tuple[Dict, Optional[str]]:
        resp, err = self.request_safe(f"/symbols/{ticker.upper()}", auth=False)
        if err or resp is None:
            return {}, err
        return resp.json(), None

    def test_auth(self, ticker: str = "AAPL") -> Tuple[bool, str]:
        """Tests the API key quickly using the lightweight /download-token endpoint."""
        resp, err = self.request_safe(
            f"/download-token/{ticker.upper()}",
            params={"timeframe": "daily", "format": "parquet", "version": "clean"},
            auth=True,
        )
        if err:
            return False, err
        return True, f"Authenticated successfully! RateLimit-Remaining: {resp.headers.get('X-RateLimit-Remaining', 'N/A')}"

    def get_bars(
        self,
        ticker: str,
        timeframe: str = "1min",
        version: str = "clean",
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        ticker = ticker.upper()
        file_path = self.cache_dir / f"{ticker}_{timeframe}_{version}.parquet"

        if file_path.exists() and file_path.stat().st_size > 0:
            try:
                return pd.read_parquet(file_path), None
            except Exception:
                file_path.unlink(missing_ok=True)

        if timeframe == "1min":
            resp, err = self.request_safe(f"/bars/{ticker}", params={"version": version}, auth=True, stream=True)
            if err or resp is None:
                return None, err
            with open(file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            return pd.read_parquet(file_path), None

        # Two-step signed URL flow for 5min, 15min, 30min, hourly, daily
        token_resp, err = self.request_safe(
            f"/download-token/{ticker}",
            params={"timeframe": timeframe, "format": "parquet", "version": version},
            auth=True,
        )
        if err or token_resp is None:
            return None, err

        signed_url = token_resp.json().get("url", "")
        if signed_url.startswith("/"):
            signed_url = f"https://api.hfdatalibrary.com{signed_url}"

        dl_resp, dl_err = self.request_safe("", auth=False, stream=True, full_url=signed_url)
        if dl_err or dl_resp is None:
            return None, dl_err

        with open(file_path, "wb") as f:
            for chunk in dl_resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        return pd.read_parquet(file_path), None


def run_intraday_backtest(
    df_raw: pd.DataFrame,
    start_date: str,
    end_date: str,
    orb_minutes: int = 15,
    vol_mult: float = 1.25,
    base_capital: float = 20000.0,
    commission_per_trade: float = 1.0,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
    df = df_raw.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = next((c for c in df.columns if c.lower() in ("timestamp", "datetime", "date", "time")), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]

    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)

    df = df.loc[str(start_date):str(end_date)]
    df = df.between_time("09:30", "16:00").copy()
    if df.empty:
        return df, pd.Series(dtype=float), {}

    df["session_date"] = df.index.normalize()

    # Session VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.groupby(df["session_date"]).cumsum()
    cum_vol = df["volume"].groupby(df["session_date"]).cumsum().replace(0, np.nan)
    df["vwap"] = cum_pv / cum_vol

    # Opening Range High & Low
    orb_end = (pd.Timestamp("2000-01-01 09:30:00") + pd.Timedelta(minutes=orb_minutes)).strftime("%H:%M")
    orb_bars = df.between_time("09:30", orb_end)
    orb_high = orb_bars.groupby("session_date")["high"].max().rename("orb_high")
    orb_low = orb_bars.groupby("session_date")["low"].min().rename("orb_low")
    df = df.join(orb_high, on="session_date").join(orb_low, on="session_date")

    df["vol_ma20"] = df.groupby("session_date")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())

    bar_times = df.index.strftime("%H:%M")
    can_enter = (bar_times > orb_end) & (bar_times < "15:55")
    eod_flat = bar_times >= "15:55"

    l_ent = (can_enter & (df["close"] > df["orb_high"]) & (df["close"] > df["vwap"]) & (df["volume"] > vol_mult * df["vol_ma20"])).to_numpy()
    s_ent = (can_enter & (df["close"] < df["orb_low"]) & (df["close"] < df["vwap"]) & (df["volume"] > vol_mult * df["vol_ma20"])).to_numpy()
    l_ext = ((df["close"] < df["vwap"]) | eod_flat).to_numpy()
    s_ext = ((df["close"] > df["vwap"]) | eod_flat).to_numpy()

    sig = np.zeros(len(df), dtype=np.float64)
    curr = 0.0
    for i in range(len(df)):
        if curr == 1.0 and l_ext[i]:
            curr = 0.0
        elif curr == -1.0 and s_ext[i]:
            curr = 0.0
        if curr == 0.0:
            if l_ent[i]:
                curr = 1.0
            elif s_ent[i]:
                curr = -1.0
        sig[i] = curr

    df["signal"] = sig
    df["position"] = df.groupby("session_date")["signal"].shift(1).fillna(0.0)
    df["bar_ret"] = df.groupby("session_date")["close"].pct_change().fillna(0.0)
    df["trade_count"] = df.groupby("session_date")["position"].diff().fillna(df["position"]).abs()

    # Daily P&L Simulation: $20,000 cap per position (profits not reinvested; losses reduce active capital)
    daily_gross_ret = df.groupby("session_date").apply(lambda g: (1.0 + g["position"] * g["bar_ret"]).prod() - 1.0)
    daily_trades = df.groupby("session_date")["trade_count"].sum()

    equity = base_capital
    equity_history = {}
    total_trades = int(daily_trades.sum())

    for dt, d_ret in daily_gross_ret.items():
        pos_size = min(base_capital, max(equity, 0.0))
        day_pnl = pos_size * d_ret - (daily_trades.loc[dt] * commission_per_trade)
        equity += day_pnl
        equity_history[dt] = equity

    equity_series = pd.Series(equity_history)
    daily_net_pct = equity_series.pct_change().fillna((equity_series.iloc[0] - base_capital) / base_capital)

    total_return_pct = ((equity - base_capital) / base_capital) * 100.0
    n_days = len(equity_series)
    n_years = max(n_days / 252.0, 1.0 / 252.0)
    ann_return_pct = (((equity / base_capital) ** (1.0 / n_years)) - 1.0) * 100.0 if equity > 0 else -100.0

    peak = equity_series.cummax()
    dd_pct = ((equity_series - peak) / peak) * 100.0
    max_dd_pct = float(dd_pct.min()) if len(dd_pct) else 0.0

    std_d = daily_net_pct.std()
    sharpe = float((daily_net_pct.mean() / std_d) * np.sqrt(252.0)) if std_d > 0 else 0.0

    active_days = daily_net_pct[daily_net_pct != 0]
    win_rate = float((active_days > 0).mean() * 100.0) if len(active_days) else 0.0

    metrics = {
        "Ending Account Value ($)": round(equity, 2),
        "Total Net Return (%)": round(total_return_pct, 2),
        "Annualized Return (%)": round(ann_return_pct, 2),
        "Annualized Sharpe": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd_pct, 2),
        "Win Rate (Active Days %)": round(win_rate, 2),
        "Total Trades Executed": total_trades,
        "Trading Days Tested": n_days,
    }
    return df, equity_series, metrics


# =====================================================================
# SIDEBAR CONTROLS
# =====================================================================
default_key = ""
try:
    if "HFDL_API_KEY" in st.secrets:
        default_key = st.secrets["HFDL_API_KEY"]
except Exception:
    pass
if not default_key:
    default_key = os.environ.get("HFDL_API_KEY", "")

with st.sidebar:
    st.header("1. API Key Setup")
    api_key = st.text_input(
        "Paste X-API-Key",
        value=default_key,
        type="password",
        placeholder="Enter your HF Data Library key...",
    )

    st.header("2. Market & Timeframe")
    pub_client = HFDataClient(api_key="")
    symbols_df, sym_err = pub_client.list_symbols()
    tickers = symbols_df["ticker"].tolist() if not symbols_df.empty else ["AAPL", "NVDA", "MSFT", "SPY", "TSLA"]
    default_idx = tickers.index("AAPL") if "AAPL" in tickers else 0

    ticker = st.selectbox("Ticker", options=tickers, index=default_idx)
    timeframe = st.selectbox("Timeframe", options=["1min", "5min", "15min", "30min"], index=0)
    version = st.selectbox("Version", options=["clean", "raw"], index=0)

    st.header("3. Strategy & Sizing")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-02"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2026-10-10"))
    orb_minutes = st.slider("Opening Range (mins)", 5, 60, 15, step=5)
    vol_mult = st.slider("Volume Filter Multiplier", 0.5, 3.0, 1.25, step=0.05)
    base_capital = st.number_input("Position Size ($)", min_value=1000.0, value=20000.0, step=1000.0)
    commission = st.number_input("Commission ($ / trade)", min_value=0.0, value=1.0, step=0.25)

    run_btn = st.button("🚀 Test Key & Run Backtest", type="primary", use_container_width=True)

# =====================================================================
# MAIN DIAGNOSTIC & BACKTEST VIEW
# =====================================================================
if sym_err:
    st.error(f"**Public Endpoint (`/symbols`) Error:** {sym_err}")
else:
    st.caption(f"✅ Connected to `api.hfdatalibrary.com` — {len(tickers)} tickers available.")

if not api_key or api_key == "your-key-here":
    st.info("👈 Paste your **HF Data Library `X-API-Key`** in the left sidebar and click **Test Key & Run Backtest**.")
    st.stop()

if run_btn:
    client = HFDataClient(api_key=api_key)

    with st.status("Step 1: Verifying API Key & Account Profile...", expanded=True) as status:
        ok, auth_msg = client.test_auth(ticker)
        if not ok:
            status.update(label="API Key Verification Failed", state="error")
            st.error(f"**API Diagnostic Result:**\n\n`{auth_msg}`")
            if "401" in auth_msg:
                st.warning("👉 **Fix for 401:** Your API key is missing, mistyped, or expired (keys expire every 30 days).")
            elif "403" in auth_msg:
                st.warning(
                    "👉 **Fix for 403:** Your API key is valid, but your HF Data Library account is missing a **verified email** "
                    "or a **completed profile** (Institution, Country, and Role must be saved on hfdatalibrary.com)."
                )
            st.stop()
        st.write(f"✅ {auth_msg}")

        status.update(label=f"Step 2: Downloading {ticker} ({timeframe}, {version}) Parquet...", state="running")
        bars_df, dl_err = client.get_bars(ticker, timeframe=timeframe, version=version)
        if dl_err or bars_df is None:
            status.update(label="Download Failed", state="error")
            st.error(f"**Download Error:** `{dl_err}`")
            st.stop()
        st.write(f"✅ Loaded {len(bars_df):,} bars for {ticker}.")

        status.update(label="Step 3: Running Intraday ORB + VWAP Backtest...", state="running")
        bt_df, eq_curve, summary = run_intraday_backtest(
            bars_df,
            start_date=str(start_date),
            end_date=str(end_date),
            orb_minutes=orb_minutes,
            vol_mult=vol_mult,
            base_capital=base_capital,
            commission_per_trade=commission,
        )
        status.update(label="Backtest Complete!", state="complete", expanded=False)

    if not summary:
        st.warning("No regular trading hours (09:30–16:00 ET) bars found in the selected date range.")
    else:
        st.subheader(f"Performance Metrics — {ticker} ({timeframe})")
        cols = st.columns(4)
        for i, (k, v) in enumerate(summary.items()):
            cols[i % 4].metric(k, f"{v:,}" if isinstance(v, (int, float)) else v)

        st.subheader("Account Equity Curve ($)")
        st.line_chart(eq_curve.rename("Account Equity ($)"))

        with st.expander("View Recent Intraday Bars & Signals"):
            st.dataframe(bt_df[["open", "high", "low", "close", "volume", "vwap", "orb_high", "orb_low", "position"]].tail(100))
