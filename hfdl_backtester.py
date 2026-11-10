import os
import html
import json
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HF Data Library — Semi-HFT QQQ & VXX Tournament", layout="wide")
st.title("⚡ Semi-High-Frequency Intraday Strategy Tournament (QQQ & VXX)")


# =====================================================================
# 1. API CLIENT
# =====================================================================
class HFDataClient:
    BASE_URL = "https://api.hfdatalibrary.com/v1"

    def __init__(self, api_key: str, cache_dir: str = "/tmp/hf_cache"):
        self.api_key = (api_key or "").strip()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "HFDL-SemiHFT-Tournament/4.0"})

    def request_safe(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        auth: bool = True,
        stream: bool = False,
        full_url: Optional[str] = None,
    ) -> Tuple[Optional[requests.Response], Optional[str]]:
        url = full_url or f"{self.BASE_URL}{endpoint}"
        headers = {"X-API-Key": self.api_key} if (auth and not full_url) else {}

        try:
            for _ in range(3):
                resp = self.session.get(url, headers=headers, params=params, stream=stream, timeout=60)
                if not resp.content and "Content-Type" not in resp.headers:
                    return None, f"Empty response with no Content-Type from {url}."
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 10))
                    time.sleep(min(retry_after, 20))
                    continue
                if resp.status_code != 200:
                    try:
                        detail = json.dumps(resp.json())
                    except Exception:
                        detail = resp.text[:500]
                    return None, f"HTTP {resp.status_code} on {url} — {detail}"
                return resp, None
            return None, "HTTP 429 Rate Limit exceeded."
        except Exception as exc:
            return None, f"Exception on {url}: {exc}"

    def list_symbols(self) -> Tuple[pd.DataFrame, Optional[str]]:
        resp, err = self.request_safe("/symbols", auth=False)
        if err or resp is None:
            return pd.DataFrame(), err
        return pd.DataFrame(resp.json().get("symbols", [])), None

    def test_auth(self, ticker: str = "AAPL") -> Tuple[bool, str]:
        resp, err = self.request_safe(
            f"/download-token/{ticker.upper()}",
            params={"timeframe": "daily", "format": "parquet", "version": "clean"},
            auth=True,
        )
        if err:
            return False, err
        return True, f"Authenticated! RateLimit-Remaining: {resp.headers.get('X-RateLimit-Remaining', 'N/A')}"

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


# =====================================================================
# 2. DATA PREPARATION & FEATURE ENGINEERING
# =====================================================================
def prepare_semi_hft_features(df_raw: pd.DataFrame, start_date: str, end_date: str, is_vxx: bool = False) -> pd.DataFrame:
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
        return df

    df["date"] = df.index.date
    df["time"] = df.index.strftime("%H:%M")
    df["year"] = df.index.year

    if is_vxx:
        df = df[df["date"] != pd.to_datetime("2024-08-05").date()].copy()

    # Session VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.groupby(df["date"]).cumsum()
    cum_vol = df["volume"].groupby(df["date"]).cumsum().replace(0, np.nan)
    df["vwap"] = cum_pv / cum_vol

    # Intraday returns & Z-score
    df["ret_1m"] = df.groupby("date")["close"].pct_change(1)
    df["ret_2m"] = df.groupby("date")["close"].pct_change(2)
    df["ret_3m"] = df.groupby("date")["close"].pct_change(3)
    df["ret_15m"] = df.groupby("date")["close"].pct_change(15)

    g = df.groupby("date")["close"]
    df["ma_15"] = g.transform(lambda x: x.rolling(15, min_periods=10).mean())
    df["std_15"] = g.transform(lambda x: x.rolling(15, min_periods=10).std())
    df["z_15"] = (df["close"] - df["ma_15"]) / df["std_15"].replace(0, np.nan)

    df["day_open"] = df.groupby("date")["open"].transform("first")
    df["ret_from_open"] = (df["close"] - df["day_open"]) / df["day_open"]

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["ibs"] = (df["close"] - df["low"]) / rng

    # 15-min Opening Range
    or15 = df[df["time"] <= "09:44"].groupby("date").agg(
        or15_high=("high", "max"),
        or15_low=("low", "min")
    )
    df["or15_high"] = df["date"].map(or15["or15_high"])
    df["or15_low"] = df["date"].map(or15["or15_low"])

    # 30-min Opening Range for breakdown filtering
    or30 = df[df["time"] <= "09:59"].groupby("date").agg(
        or30_low=("low", "min"),
        or30_close=("close", "last")
    )
    df["or30_low"] = df["date"].map(or30["or30_low"])
    df["or30_ret"] = (df["date"].map(or30["or30_close"]) - df["day_open"]) / df["day_open"]

    # Consecutive 1-min Up/Down Streaks
    dirs = np.where(df["ret_1m"] > 0, 1, np.where(df["ret_1m"] < 0, -1, 0))
    dates = df["date"].values
    u_st, d_st = np.zeros(len(df), dtype=int), np.zeros(len(df), dtype=int)
    cu, cd = 0, 0
    for i in range(len(df)):
        if i == 0 or dates[i] != dates[i-1]:
            cu, cd = 0, 0
        if dirs[i] == 1:
            cu += 1; cd = 0
        elif dirs[i] == -1:
            cd += 1; cu = 0
        else:
            cu = 0; cd = 0
        u_st[i] = cu; d_st[i] = cd
    df["u_streak"] = u_st
    df["d_streak"] = d_st
    return df


# =====================================================================
# 3. SEMI-HFT BACKTESTING ENGINE ($20K SIZE, $1 COMM, BAR-CLOSE EXEC)
# =====================================================================
def run_semi_hft_engine(
    df: pd.DataFrame,
    ticker: str,
    long_sig: pd.Series,
    short_sig: pd.Series,
    exit_l: pd.Series,
    exit_s: pd.Series,
    max_hold_bars: int,
    stop_loss_pct: float,
    profit_target_pct: float,
    fixed_trade_size: float = 20000.0,
    commission_per_trade: float = 1.0,
) -> pd.DataFrame:
    trades = []
    dates = df["date"].values
    years = df["year"].values
    times = df.index.strftime("%Y-%m-%d %H:%M:%S").values
    time_strs = df["time"].values
    closes = df["close"].values
    l_arr, s_arr = long_sig.values, short_sig.values
    el_arr, es_arr = exit_l.values, exit_s.values

    n = len(df)
    in_pos = 0
    entry_px, entry_idx, entry_time, shares = 0.0, 0, None, 0

    for i in range(n):
        is_eod = (i == n - 1) or (dates[i+1] != dates[i]) or (time_strs[i] >= "15:58")
        if in_pos != 0:
            bars_held = i - entry_idx
            cur_px = closes[i]
            ret = (cur_px - entry_px) / entry_px if in_pos == 1 else (entry_px - cur_px) / entry_px
            reason = None
            if is_eod:
                reason = "EOD Close"
            elif ret <= -stop_loss_pct:
                reason = "Stop Loss"
            elif ret >= profit_target_pct:
                reason = "Profit Target"
            elif (in_pos == 1 and el_arr[i]) or (in_pos == -1 and es_arr[i]):
                reason = "Regime Reversal"
            elif bars_held >= max_hold_bars:
                reason = f"Time Exit ({max_hold_bars}m)"

            if reason is not None:
                gross = (cur_px - entry_px) * shares if in_pos == 1 else (entry_px - cur_px) * shares
                trades.append({
                    "Ticker": ticker,
                    "Date": dates[i],
                    "Year": years[i],
                    "Entry Time": entry_time,
                    "Exit Time": times[i],
                    "Direction": "LONG" if in_pos == 1 else "SHORT",
                    "Shares": shares,
                    "Entry Price": round(entry_px, 3),
                    "Exit Price": round(cur_px, 3),
                    "Bars Held": bars_held,
                    "Exit Reason": reason,
                    "Gross PnL ($)": round(gross, 2),
                    "Net PnL ($)": round(gross - commission_per_trade, 2)
                })
                in_pos = 0

        if in_pos == 0 and not is_eod and ("09:45" <= time_strs[i] <= "15:40"):
            if l_arr[i]:
                in_pos = 1
                entry_px = closes[i]
                entry_idx = i
                entry_time = times[i]
                shares = max(1, int(round(fixed_trade_size / entry_px)))
            elif s_arr[i]:
                in_pos = -1
                entry_px = closes[i]
                entry_idx = i
                entry_time = times[i]
                shares = max(1, int(round(fixed_trade_size / entry_px)))
    return pd.DataFrame(trades)


# =====================================================================
# 4. STREAMLIT UI & STRATEGY SELECTION
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
    api_key = st.text_input("Paste X-API-Key", value=default_key, type="password")

    st.header("2. Market & Parameters")
    pub_client = HFDataClient(api_key="")
    symbols_df, sym_err = pub_client.list_symbols()
    tickers = symbols_df["ticker"].tolist() if not symbols_df.empty else ["QQQ", "VXX", "AAPL", "SPY"]
    default_idx = tickers.index("QQQ") if "QQQ" in tickers else 0

    ticker = st.selectbox("Ticker", options=tickers, index=default_idx)
    version = st.selectbox("Version", options=["clean", "raw"], index=0)

    fixed_trade_size = st.number_input("Fixed Trade Size ($)", value=20000.0, step=1000.0)
    commission = st.number_input("Commission per Trade ($)", value=1.0, step=0.25)

    st.header("3. Strategy Configuration")
    strategy_mode = st.selectbox(
        "Strategy Engine",
        options=[
            "QQQ Multi-Signal Engine (Long + Short, 120m Hold)",
            "QQQ Active 45m Scalper (Long + Short)",
            "QQQ Core Sub-Signal 1 Only (VWAP + 3-Bar Dip Long)",
            "VXX Volatility Crush & Cascade Engine"
        ]
    )

    start_date = st.date_input("Start Date", value=pd.to_datetime("2022-03-07"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2026-10-30"))

    run_btn = st.button("🚀 Run Semi-HFT Backtest", type="primary", use_container_width=True)

if sym_err:
    st.error(f"**Public Endpoint Error:** {sym_err}")

if not api_key or api_key == "your-key-here":
    st.info("👈 Paste your **HF Data Library `X-API-Key`** in the left sidebar and click **Run Semi-HFT Backtest**.")
    st.stop()

if run_btn:
    client = HFDataClient(api_key=api_key)

    with st.status(f"Downloading and backtesting {ticker}...", expanded=True) as status:
        ok, auth_msg = client.test_auth(ticker)
        if not ok:
            status.update(label="API Key Verification Failed", state="error")
            st.error(auth_msg)
            st.stop()

        bars_df, dl_err = client.get_bars(ticker, timeframe="1min", version=version)
        if dl_err or bars_df is None:
            status.update(label="Download Failed", state="error")
            st.error(dl_err)
            st.stop()

        is_vxx = "VXX" in ticker.upper()
        df_feat = prepare_semi_hft_features(bars_df, str(start_date), str(end_date), is_vxx=is_vxx)

        if df_feat.empty:
            status.update(label="No data in range", state="error")
            st.warning("No data found for the selected date range.")
            st.stop()

        # Define signal arrays based on strategy mode
        if is_vxx or "VXX" in strategy_mode:
            max_hold = 90
            pt = 0.015
            sl = 0.014
            l_sig = (
                (df_feat["time"] >= "09:45") & (df_feat["time"] <= "15:00") &
                (df_feat["close"] > df_feat["vwap"]) & (df_feat["close"] > df_feat["or15_high"]) &
                (df_feat["ret_from_open"] >= 0.0120) & (df_feat["ret_15m"] >= 0.0080) & (df_feat["ret_2m"] <= -0.0010)
            )
            s_sig = (
                (df_feat["time"] >= "09:45") & (df_feat["time"] <= "15:35") & (
                    ((df_feat["close"] < df_feat["vwap"]) & (df_feat["close"] < df_feat["or15_low"]) & (df_feat["ret_from_open"] <= -0.0050) & (df_feat["ret_2m"] >= 0.0010)) |
                    ((df_feat["close"] < df_feat["vwap"]) & (df_feat["ret_from_open"] <= -0.0060) & (df_feat["ret_15m"] <= -0.0080) & (df_feat["ret_1m"] >= 0.0)) |
                    ((df_feat["ret_from_open"] <= -0.0020) & (df_feat["z_15"] >= 1.8) & (df_feat["ibs"] >= 0.75))
                )
            )
            ex_l = (df_feat["close"] < df_feat["vwap"] * 0.995)
            ex_s = (df_feat["close"] > df_feat["vwap"] * 1.005) & (df_feat["ret_15m"] > 0.0060)
        else:
            if "Core Sub-Signal 1" in strategy_mode:
                max_hold = 120
                pt = 0.0075
                sl = 0.0080
                l_sig = (df_feat["close"] > df_feat["vwap"]) & (df_feat["ret_from_open"] >= 0.0015) & ((df_feat["d_streak"] >= 3) | ((df_feat["ret_3m"] <= -0.0010) & (df_feat["ibs"] <= 0.15)))
                s_sig = pd.Series(False, index=df_feat.index)
            elif "45m" in strategy_mode:
                max_hold = 45
                pt = 0.0045
                sl = 0.0055
                l_sig = (
                    ((df_feat["close"] > df_feat["vwap"]) & (df_feat["ret_from_open"] >= 0.0020) & (df_feat["ret_15m"] >= 0.0015) & (df_feat["ret_2m"] <= -0.0004) & (df_feat["ibs"] <= 0.25)) |
                    ((df_feat["close"] > df_feat["vwap"]) & (df_feat["close"] > df_feat["or15_high"]) & (df_feat["ret_15m"] >= 0.0040) & (df_feat["ret_1m"] <= 0.0)) |
                    ((df_feat["time"] >= "11:00") & (df_feat["ret_from_open"] >= -0.0030) & (df_feat["z_15"] <= -2.2) & (df_feat["ibs"] <= 0.05))
                )
                s_sig = (df_feat["close"] < df_feat["vwap"]) & (df_feat["close"] < df_feat["or15_low"]) & (df_feat["ret_from_open"] <= -0.0040) & (df_feat["ret_15m"] <= -0.0020) & (df_feat["ret_2m"] >= 0.0004) & (df_feat["ibs"] >= 0.75)
            else:
                max_hold = 120
                pt = 0.0075
                sl = 0.0080
                l_sig = (
                    ((df_feat["close"] > df_feat["vwap"]) & (df_feat["ret_from_open"] >= 0.0015) & ((df_feat["d_streak"] >= 3) | ((df_feat["ret_3m"] <= -0.0010) & (df_feat["ibs"] <= 0.15)))) |
                    ((df_feat["close"] > df_feat["or15_high"]) & (df_feat["close"] > df_feat["vwap"]) & (df_feat["ret_15m"] >= 0.0025) & (df_feat["d_streak"] >= 1) & (df_feat["ibs"] <= 0.25)) |
                    ((df_feat["d_streak"] >= 4) & (df_feat["ibs"] <= 0.05) & (df_feat["ret_from_open"] >= -0.0040))
                )
                s_sig = (df_feat["close"] < df_feat["vwap"]) & (df_feat["close"] < df_feat["or15_low"]) & (df_feat["ret_from_open"] <= -0.0035) & (df_feat["ret_15m"] <= -0.0015) & (df_feat["u_streak"] >= 2) & (df_feat["ibs"] >= 0.80)

            ex_l = (df_feat["close"] < df_feat["vwap"] * 0.9970) & (df_feat["ret_15m"] <= -0.0035)
            ex_s = (df_feat["close"] > df_feat["vwap"] * 1.0030) & (df_feat["ret_15m"] >= 0.0035)

        trades_df = run_semi_hft_engine(
            df_feat, ticker, l_sig, s_sig, ex_l, ex_s,
            max_hold_bars=max_hold, stop_loss_pct=sl, profit_target_pct=pt,
            fixed_trade_size=fixed_trade_size, commission_per_trade=commission
        )
        status.update(label="Backtest Complete!", state="complete", expanded=False)

    if trades_df.empty:
        st.warning("No trades generated with the current strategy parameters and date filter.")
    else:
        trades_df["Cumulative Net PnL ($)"] = trades_df["Net PnL ($)"].cumsum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Net PnL", f"${trades_df['Net PnL ($)'].sum():,.2f}")
        col2.metric("Total Trades", f"{len(trades_df):,}")
        col3.metric("Win Rate", f"{(trades_df['Net PnL ($)'] > 0).mean()*100:.2f}%")
        col4.metric("Total Commissions Paid", f"${len(trades_df)*commission:,.2f}")

        st.subheader("📅 Annualized Performance Breakdown")
        ann = trades_df.groupby("Year").agg(
            Trades=("Net PnL ($)", "count"),
            Win_Rate=("Net PnL ($)", lambda x: f"{(x > 0).mean()*100:.2f}%"),
            Gross_PnL=("Gross PnL ($)", lambda x: f"${x.sum():,.2f}"),
            Net_PnL=("Net PnL ($)", lambda x: f"${x.sum():,.2f}"),
            Return_on_Allocation=("Net PnL ($)", lambda x: f"{(x.sum() / fixed_trade_size)*100:+.2f}%")
        ).reset_index()
        st.dataframe(ann, use_container_width=True)

        st.subheader("📈 Cumulative Net PnL Curve")
        st.line_chart(trades_df.set_index("Exit Time")["Cumulative Net PnL ($)"])

        st.subheader("📋 Master Trade Log")
        st.dataframe(trades_df, use_container_width=True)
