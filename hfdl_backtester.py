import os
import html
import json
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HF Data Library — 10-Strategy Intraday Tournament", layout="wide")
st.title("📈 HF Data Library — 10-Strategy Intraday Backtest Tournament")


# =====================================================================
# 1. API CLIENT (SAFE ERROR HANDLING FOR STREAMLIT CLOUD)
# =====================================================================
class HFDataClient:
    BASE_URL = "https://api.hfdatalibrary.com/v1"

    def __init__(self, api_key: str, cache_dir: str = "/tmp/hf_cache"):
        self.api_key = (api_key or "").strip()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "HFDL-Tournament-Backtester/2.0"})

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

    def test_auth(self, ticker: str = "AAPL") -> Tuple[bool, str]:
        resp, err = self.request_safe(
            f"/download-token/{ticker.upper()}",
            params={"timeframe": "daily", "format": "parquet", "version": "clean"},
            auth=True,
        )
        if err:
            return False, err
        rem = resp.headers.get("X-RateLimit-Remaining", "N/A")
        return True, f"Authenticated! RateLimit-Remaining: {rem}"

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
# 2. DATA PREPARATION & MISSING DATA AUDIT
# =====================================================================
EXPECTED_BARS_PER_DAY = {
    "1min": 390,   # 09:30 to 15:59 inclusive (or 391 if 16:00 is included)
    "5min": 78,
    "15min": 26,
    "30min": 13,
}


def prepare_and_audit_data(
    df_raw: pd.DataFrame,
    timeframe: str,
    start_date: str,
    end_date: str,
    drop_incomplete_days: bool = True,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """
    Normalizes timestamps to US/Eastern, records exact dataset start dates,
    checks for missing intraday bars and NaN prices, and optionally removes incomplete sessions.
    """
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

    # Full dataset boundaries before date slicing
    raw_first_ts = df.index.min() if not df.empty else None
    raw_last_ts = df.index.max() if not df.empty else None

    # Slice user-selected window and filter to Regular Trading Hours (09:30 - 15:59 ET)
    df = df.loc[str(start_date):str(end_date)]
    df = df.between_time("09:30", "15:59").copy()

    if df.empty:
        return df, {}, pd.DataFrame()

    # Check and clean NaN values in OHLCV
    ohlc_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    nan_rows_count = int(df[ohlc_cols].isna().any(axis=1).sum())
    if nan_rows_count > 0:
        df = df.dropna(subset=ohlc_cols)

    # Remove duplicate timestamps if any exist
    dup_timestamps = int(df.index.duplicated().sum())
    if dup_timestamps > 0:
        df = df[~df.index.duplicated(keep="first")]

    df["session_date"] = df.index.normalize()

    # Audit expected bars per session
    expected_bars = EXPECTED_BARS_PER_DAY.get(timeframe, 390)
    freq_map = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min"}
    pd_freq = freq_map.get(timeframe, "1min")

    session_audit_rows = []
    for session_dt, group in df.groupby("session_date"):
        observed = len(group)
        missing_count = max(0, expected_bars - observed)
        # Calculate largest gap in minutes
        if len(group) > 1:
            diffs = group.index.to_series().diff().dt.total_seconds().div(60.0)
            max_gap_mins = float(diffs.max())
        else:
            max_gap_mins = 390.0

        session_audit_rows.append({
            "Session Date": session_dt.strftime("%Y-%m-%d"),
            "First Bar (ET)": group.index.min().strftime("%H:%M"),
            "Last Bar (ET)": group.index.max().strftime("%H:%M"),
            "Observed Bars": observed,
            "Expected Bars": expected_bars,
            "Missing Bars": missing_count,
            "Completeness (%)": round((observed / expected_bars) * 100.0, 1),
            "Max Gap (Mins)": round(max_gap_mins, 1),
            "Complete Session": observed >= expected_bars,
        })

    audit_df = pd.DataFrame(session_audit_rows)
    total_sessions_before = len(audit_df)
    incomplete_sessions_df = audit_df[~audit_df["Complete Session"]].copy()
    incomplete_days_count = len(incomplete_sessions_df)
    total_missing_bars = int(audit_df["Missing Bars"].sum())

    if drop_incomplete_days and incomplete_days_count > 0:
        valid_dates = pd.to_datetime(audit_df.loc[audit_df["Complete Session"], "Session Date"])
        df = df[df["session_date"].isin(valid_dates)].copy()

    bt_first_ts = df.index.min() if not df.empty else None
    bt_last_ts = df.index.max() if not df.empty else None

    audit_summary = {
        "raw_first_ts": raw_first_ts.strftime("%Y-%m-%d %H:%M:%S ET") if raw_first_ts is not None else "N/A",
        "raw_last_ts": raw_last_ts.strftime("%Y-%m-%d %H:%M:%S ET") if raw_last_ts is not None else "N/A",
        "bt_first_ts": bt_first_ts.strftime("%Y-%m-%d %H:%M:%S ET") if bt_first_ts is not None else "N/A",
        "bt_last_ts": bt_last_ts.strftime("%Y-%m-%d %H:%M:%S ET") if bt_last_ts is not None else "N/A",
        "total_sessions_found": total_sessions_before,
        "complete_sessions": total_sessions_before - incomplete_days_count,
        "incomplete_sessions": incomplete_days_count,
        "sessions_tested": int(df["session_date"].nunique()) if not df.empty else 0,
        "total_missing_bars": total_missing_bars,
        "nan_rows_removed": nan_rows_count,
        "duplicate_timestamps_removed": dup_timestamps,
        "expected_bars_per_day": expected_bars,
    }

    return df, audit_summary, incomplete_sessions_df


# =====================================================================
# 3. COMMON INDICATORS & NEXT-BAR OPEN EXECUTION ENGINE
# =====================================================================
def precompute_indicators(df: pd.DataFrame, x_lookback: int = 10, orb_minutes: int = 15) -> pd.DataFrame:
    """Computes shared intraday indicators grouped strictly by session_date so overnight gaps never pollute signals."""
    df = df.copy()
    g = df.groupby("session_date")

    # Bar-to-bar Open-to-Open return for execution at Next Bar's Open
    # When a signal fires at Bar[t] Close, we enter at Bar[t+1] Open and earn Open[t+1] -> Open[t+2]
    # On the final bar of the session, we exit at Close[t]
    next_open = g["open"].shift(-1)
    df["exec_bar_ret"] = (next_open / df["open"]) - 1.0
    df["exec_bar_ret"] = df["exec_bar_ret"].fillna((df["close"] / df["open"]) - 1.0)

    # Session VWAP and Intraday Standard Deviation Bands
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.groupby(df["session_date"]).cumsum()
    cum_vol = g["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = cum_pv / cum_vol

    sq_diff = ((tp - df["vwap"]) ** 2) * df["volume"]
    vwap_var = sq_diff.groupby(df["session_date"]).cumsum() / cum_vol
    df["vwap_std"] = np.sqrt(vwap_var.clip(lower=0.0))

    # Consecutive Lower Closes & Consecutive Higher Closes (within each session)
    is_lower_close = df["close"] < g["close"].shift(1)
    is_higher_close = df["close"] > g["close"].shift(1)

    # Prior X-Bars Highest Close, Lowest Close, Highest High, Lowest Low
    df["highest_close_x"] = g["close"].transform(lambda s: s.rolling(x_lookback, min_periods=2).max())
    df["lowest_close_x"] = g["close"].transform(lambda s: s.rolling(x_lookback, min_periods=2).min())
    df["donchian_high_x"] = g["high"].transform(lambda s: s.shift(1).rolling(x_lookback, min_periods=3).max())
    df["donchian_low_x"] = g["low"].transform(lambda s: s.shift(1).rolling(x_lookback, min_periods=3).min())

    # 5 consecutive lower closes & 3 consecutive lower closes
    df["down_5_streak"] = is_lower_close.groupby(df["session_date"]).transform(
        lambda s: s.rolling(5, min_periods=5).sum() == 5
    )
    df["down_3_streak"] = is_lower_close.groupby(df["session_date"]).transform(
        lambda s: s.rolling(3, min_periods=3).sum() == 3
    )
    df["up_5_streak"] = is_higher_close.groupby(df["session_date"]).transform(
        lambda s: s.rolling(5, min_periods=5).sum() == 5
    )

    # Intraday RSI(2) and EMA(9), EMA(21), SMA(20)
    delta = g["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.groupby(df["session_date"]).transform(lambda s: s.ewm(alpha=1 / 2, adjust=False).mean())
    avg_loss = loss.groupby(df["session_date"]).transform(lambda s: s.ewm(alpha=1 / 2, adjust=False).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi2"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    df["ema9"] = g["close"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["ema21"] = g["close"].transform(lambda s: s.ewm(span=21, adjust=False).mean())
    df["sma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df["std20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=5).std())
    df["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())

    # Internal Bar Strength (IBS) = (Close - Low) / (High - Low)
    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["ibs"] = ((df["close"] - df["low"]) / hl_range).fillna(0.5)
    df["prev_high"] = g["high"].shift(1)

    # Opening Range High / Low
    orb_end = (pd.Timestamp("2000-01-01 09:30:00") + pd.Timedelta(minutes=orb_minutes)).strftime("%H:%M")
    orb_slice = df.between_time("09:30", orb_end)
    orb_h = orb_slice.groupby("session_date")["high"].max().rename("orb_high")
    orb_l = orb_slice.groupby("session_date")["low"].min().rename("orb_low")
    df = df.join(orb_h, on="session_date").join(orb_l, on="session_date")

    # Session Opening Drive Return (first 15 mins)
    first_open = g["open"].transform("first")
    orb_close_map = orb_slice.groupby("session_date")["close"].last().rename("orb_close")
    df = df.join(orb_close_map, on="session_date")
    df["open_drive_ret"] = (df["orb_close"] / first_open) - 1.0

    df["bar_time"] = df.index.strftime("%H:%M")
    df["orb_end_time"] = orb_end
    return df


def simulate_state_machine(
    df: pd.DataFrame,
    long_entry: np.ndarray,
    long_exit: np.ndarray,
    short_entry: Optional[np.ndarray] = None,
    short_exit: Optional[np.ndarray] = None,
    max_hold_bars: int = 0,
    base_capital: float = 20000.0,
    commission_per_trade: float = 1.0,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Executes entries at Next Bar's Open (Open[t+1]) when signal triggers at Bar[t] Close.
    Enforces mandatory EOD exit at 15:55 ET.
    Applies fixed $20,000 position cap (gains are not reinvested; losses reduce active capital).
    """
    n = len(df)
    if short_entry is None:
        short_entry = np.zeros(n, dtype=bool)
    if short_exit is None:
        short_exit = np.zeros(n, dtype=bool)

    eod_flat = (df["bar_time"] >= "15:55").to_numpy()
    can_enter = (df["bar_time"] >= "09:35") & (df["bar_time"] < "15:50")
    can_enter_arr = can_enter.to_numpy()

    l_ent = long_entry & can_enter_arr
    s_ent = short_entry & can_enter_arr
    l_ext = long_exit | eod_flat
    s_ext = short_exit | eod_flat

    target_pos = np.zeros(n, dtype=np.float64)
    curr = 0.0
    bars_held = 0

    for i in range(n):
        if curr == 1.0:
            bars_held += 1
            if l_ext[i] or (max_hold_bars > 0 and bars_held >= max_hold_bars):
                curr = 0.0
                bars_held = 0
        elif curr == -1.0:
            bars_held += 1
            if s_ext[i] or (max_hold_bars > 0 and bars_held >= max_hold_bars):
                curr = 0.0
                bars_held = 0

        if curr == 0.0:
            if l_ent[i]:
                curr = 1.0
                bars_held = 0
            elif s_ent[i]:
                curr = -1.0
                bars_held = 0

        target_pos[i] = curr

    # Shift by 1 bar inside each session so signal at Bar[t] Close enters at Bar[t+1] Open
    pos_series = pd.Series(target_pos, index=df.index).groupby(df["session_date"]).shift(1).fillna(0.0)
    trade_units = pos_series.groupby(df["session_date"]).diff().fillna(pos_series).abs()

    strat_bar_ret = pos_series * df["exec_bar_ret"]
    temp = pd.DataFrame({
        "session_date": df["session_date"],
        "strat_ret": strat_bar_ret,
        "trades": trade_units,
    })

    daily_gross_ret = temp.groupby("session_date")["strat_ret"].apply(lambda r: (1.0 + r).prod() - 1.0)
    daily_trades = temp.groupby("session_date")["trades"].sum()

    equity = base_capital
    eq_hist = {}
    winning_days = 0
    active_days = 0
    gross_profits = 0.0
    gross_losses = 0.0

    for dt, d_ret in daily_gross_ret.items():
        n_tr = daily_trades.loc[dt]
        if d_ret != 0.0 or n_tr > 0:
            active_days += 1
        pos_dollars = min(base_capital, max(equity, 0.0))
        day_net_pnl = (pos_dollars * d_ret) - (n_tr * commission_per_trade)
        equity += day_net_pnl
        eq_hist[dt] = equity

        if day_net_pnl > 0:
            winning_days += 1
            gross_profits += day_net_pnl
        elif day_net_pnl < 0:
            gross_losses += abs(day_net_pnl)

    eq_series = pd.Series(eq_hist)
    if eq_series.empty:
        return eq_series, {}

    daily_pct = eq_series.pct_change().fillna((eq_series.iloc[0] - base_capital) / base_capital)
    total_ret_pct = ((equity - base_capital) / base_capital) * 100.0
    n_days = len(eq_series)
    n_years = max(n_days / 252.0, 1.0 / 252.0)
    ann_ret_pct = (((equity / base_capital) ** (1.0 / n_years)) - 1.0) * 100.0 if equity > 0 else -100.0

    peak = eq_series.cummax()
    dd_pct = ((eq_series - peak) / peak) * 100.0
    max_dd_pct = float(dd_pct.min()) if len(dd_pct) else 0.0

    std_d = daily_pct.std()
    sharpe = float((daily_pct.mean() / std_d) * np.sqrt(252.0)) if std_d > 0 else 0.0
    win_rate = (winning_days / active_days * 100.0) if active_days > 0 else 0.0
    pf = (gross_profits / gross_losses) if gross_losses > 0 else (99.9 if gross_profits > 0 else 0.0)

    metrics = {
        "Annualized Return (%)": round(ann_ret_pct, 2),
        "Total Net Return (%)": round(total_ret_pct, 2),
        "Ending Equity ($)": round(equity, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd_pct, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Profit Factor": round(min(pf, 99.9), 2),
        "Total Trades": int(daily_trades.sum()),
    }
    return eq_series, metrics


# =====================================================================
# 4. THE 10 INTRADAY STRATEGIES (WITH HOVER TOOLTIP RULES)
# =====================================================================
def run_all_10_strategies(
    df_ind: pd.DataFrame,
    x_lookback: int = 10,
    orb_minutes: int = 15,
    base_capital: float = 20000.0,
    commission: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    strategies_spec = [
        {
            "name": f"1. 5 Lower Closes -> Exit Highest Close ({x_lookback} Bars)",
            "rule": (
                f"ENTRY: 5 consecutive lower closes inside the session; buy Next Bar's Open. "
                f"EXIT: Exit at Next Bar's Open once Close >= Highest Close of the prior {x_lookback} bars, or at 15:55 ET."
            ),
            "l_ent": df_ind["down_5_streak"].to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["highest_close_x"]).to_numpy(),
            "s_ent": None,
            "s_ext": None,
            "max_hold": 0,
        },
        {
            "name": f"2. 3 Lower Closes Above VWAP -> Exit Highest Close ({x_lookback} Bars)",
            "rule": (
                f"ENTRY: 3 consecutive lower closes while price remains above Session VWAP (uptrend pullback); buy Next Bar's Open. "
                f"EXIT: Close >= Highest Close of the prior {x_lookback} bars, Close < VWAP - 1.5 SD, or 15:55 ET."
            ),
            "l_ent": (df_ind["down_3_streak"] & (df_ind["close"] > df_ind["vwap"])).to_numpy(),
            "l_ext": ((df_ind["close"] >= df_ind["highest_close_x"]) | (df_ind["close"] < df_ind["vwap"] - 1.5 * df_ind["vwap_std"])).to_numpy(),
            "s_ent": None,
            "s_ext": None,
            "max_hold": 0,
        },
        {
            "name": f"3. 5 Consecutive Bars Reversal (Long & Short, {x_lookback}-Bar Exit)",
            "rule": (
                f"LONG ENTRY: 5 consecutive lower closes -> buy Next Bar's Open; exit at Highest Close of prior {x_lookback} bars. "
                f"SHORT ENTRY: 5 consecutive higher closes -> short Next Bar's Open; cover at Lowest Close of prior {x_lookback} bars."
            ),
            "l_ent": df_ind["down_5_streak"].to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["highest_close_x"]).to_numpy(),
            "s_ent": df_ind["up_5_streak"].to_numpy(),
            "s_ext": (df_ind["close"] <= df_ind["lowest_close_x"]).to_numpy(),
            "max_hold": 0,
        },
        {
            "name": f"4. Opening Range Breakout ({orb_minutes}m) + Session VWAP Trend",
            "rule": (
                f"LONG ENTRY: After first {orb_minutes} mins, Close > Opening Range High, Close > VWAP, and Volume > 1.25x 20-bar MA -> buy Next Bar's Open. "
                f"SHORT ENTRY: Symmetric breakdown below Opening Range Low & VWAP. EXIT: Cross back through Session VWAP or 15:55 ET."
            ),
            "l_ent": ((df_ind["bar_time"] > df_ind["orb_end_time"]) & (df_ind["close"] > df_ind["orb_high"]) & (df_ind["close"] > df_ind["vwap"]) & (df_ind["volume"] > 1.25 * df_ind["vol_ma20"])).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["vwap"]).to_numpy(),
            "s_ent": ((df_ind["bar_time"] > df_ind["orb_end_time"]) & (df_ind["close"] < df_ind["orb_low"]) & (df_ind["close"] < df_ind["vwap"]) & (df_ind["volume"] > 1.25 * df_ind["vol_ma20"])).to_numpy(),
            "s_ext": (df_ind["close"] > df_ind["vwap"]).to_numpy(),
            "max_hold": 0,
        },
        {
            "name": "5. Session VWAP 2-SD Mean Reversion Band Bounce",
            "rule": (
                "LONG ENTRY: Close drops below Session VWAP minus 2.0 Intraday Standard Deviations -> buy Next Bar's Open. "
                "SHORT ENTRY: Close rises above Session VWAP plus 2.0 Intraday Standard Deviations -> short Next Bar's Open. "
                "EXIT: Price reverts back to touch Session VWAP or 15:55 ET."
            ),
            "l_ent": (df_ind["close"] < (df_ind["vwap"] - 2.0 * df_ind["vwap_std"])).to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["vwap"]).to_numpy(),
            "s_ent": (df_ind["close"] > (df_ind["vwap"] + 2.0 * df_ind["vwap_std"])).to_numpy(),
            "s_ext": (df_ind["close"] <= df_ind["vwap"]).to_numpy(),
            "max_hold": 0,
        },
        {
            "name": f"6. Connors Intraday RSI(2) < 5 Dip Buyer -> {x_lookback}-Bar High Exit",
            "rule": (
                f"ENTRY: 2-period Intraday RSI drops below 5 while price is above Session VWAP -> buy Next Bar's Open. "
                f"EXIT: RSI(2) > 70 or Close >= Highest Close of prior {x_lookback} bars, or 15:55 ET."
            ),
            "l_ent": ((df_ind["rsi2"] < 5.0) & (df_ind["close"] > df_ind["vwap"])).to_numpy(),
            "l_ext": ((df_ind["rsi2"] > 70.0) | (df_ind["close"] >= df_ind["highest_close_x"])).to_numpy(),
            "s_ent": None,
            "s_ext": None,
            "max_hold": 0,
        },
        {
            "name": f"7. Donchian Channel ({x_lookback}-Bar) Volatility Breakout",
            "rule": (
                f"LONG ENTRY: Close breaks above prior {x_lookback}-bar Highest High with Volume > 1.2x 20-bar average -> buy Next Bar's Open. "
                f"EXIT: Close falls below prior {x_lookback}-bar Lowest Close or 15:55 ET."
            ),
            "l_ent": ((df_ind["close"] > df_ind["donchian_high_x"]) & (df_ind["volume"] > 1.2 * df_ind["vol_ma20"])).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["lowest_close_x"]).to_numpy(),
            "s_ent": None,
            "s_ext": None,
            "max_hold": 0,
        },
        {
            "name": "8. Intraday Internal Bar Strength (IBS < 0.10) Pullback",
            "rule": (
                "ENTRY: Bar closes in the bottom 10% of its High-Low range (IBS < 0.10) while above 20-bar SMA -> buy Next Bar's Open. "
                "EXIT: Close exceeds prior bar's High (Close > High[t-1]) or 15:55 ET."
            ),
            "l_ent": ((df_ind["ibs"] < 0.10) & (df_ind["close"] > df_ind["sma20"])).to_numpy(),
            "l_ext": (df_ind["close"] > df_ind["prev_high"]).to_numpy(),
            "s_ent": None,
            "s_ext": None,
            "max_hold": 0,
        },
        {
            "name": "9. Bollinger Band (20, 2) Squeeze & Volume Expansion",
            "rule": (
                "LONG ENTRY: Close breaks above Upper Bollinger Band (20-SMA + 2 SD) on Volume > 1.5x 20-bar MA -> buy Next Bar's Open. "
                "EXIT: Close crosses back below 20-bar SMA or 15:55 ET."
            ),
            "l_ent": ((df_ind["close"] > (df_ind["sma20"] + 2.0 * df_ind["std20"])) & (df_ind["volume"] > 1.5 * df_ind["vol_ma20"])).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["sma20"]).to_numpy(),
            "s_ent": None,
            "s_ext": None,
            "max_hold": 0,
        },
        {
            "name": "10. EMA(9/21) Momentum Cross + VWAP Confirmation",
            "rule": (
                "LONG ENTRY: 9-bar EMA crosses above 21-bar EMA while Close > Session VWAP -> buy Next Bar's Open. "
                "SHORT ENTRY: 9-bar EMA crosses below 21-bar EMA while Close < Session VWAP -> short Next Bar's Open. "
                "EXIT: Opposite EMA crossover or 15:55 ET."
            ),
            "l_ent": ((df_ind["ema9"] > df_ind["ema21"]) & (df_ind["ema9"].shift(1) <= df_ind["ema21"].shift(1)) & (df_ind["close"] > df_ind["vwap"])).to_numpy(),
            "l_ext": (df_ind["ema9"] < df_ind["ema21"]).to_numpy(),
            "s_ent": ((df_ind["ema9"] < df_ind["ema21"]) & (df_ind["ema9"].shift(1) >= df_ind["ema21"].shift(1)) & (df_ind["close"] < df_ind["vwap"])).to_numpy(),
            "s_ext": (df_ind["ema9"] > df_ind["ema21"]).to_numpy(),
            "max_hold": 0,
        },
    ]

    rows = []
    eq_curves = {}

    for spec in strategies_spec:
        eq, m = simulate_state_machine(
            df_ind,
            long_entry=spec["l_ent"],
            long_exit=spec["l_ext"],
            short_entry=spec["s_ent"],
            short_exit=spec["s_ext"],
            max_hold_bars=spec["max_hold"],
            base_capital=base_capital,
            commission_per_trade=commission,
        )
        if m:
            row = {"Strategy": spec["name"], "Rule Tooltip": spec["rule"]}
            row.update(m)
            rows.append(row)
            eq_curves[spec["name"]] = eq

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        res_df = res_df.sort_values(by="Annualized Return (%)", ascending=False).reset_index(drop=True)
        res_df.index = res_df.index + 1  # Rank 1..10
        res_df.index.name = "Rank"

    return res_df, eq_curves


# =====================================================================
# 5. HTML TOURNAMENT TABLE WITH HOVER TOOLTIPS & GREEN/RED HIGHLIGHTING
# =====================================================================
def render_tournament_html_table(df: pd.DataFrame) -> str:
    """
    Renders an interactive HTML table where:
    - Best strategy (top row by Annualized Return %) is highlighted green.
    - Worst strategy (bottom row by Annualized Return %) is highlighted red.
    - Hovering the mouse over any strategy row displays a rich tooltip with the exact trading rules.
    """
    best_idx = df["Annualized Return (%)"].idxmax()
    worst_idx = df["Annualized Return (%)"].idxmin()

    display_cols = [
        "Strategy",
        "Annualized Return (%)",
        "Total Net Return (%)",
        "Ending Equity ($)",
        "Sharpe Ratio",
        "Max Drawdown (%)",
        "Win Rate (%)",
        "Profit Factor",
        "Total Trades",
    ]

    css = """
    <style>
    .tourn-wrap {
        overflow-x: auto;
        margin-top: 8px;
        margin-bottom: 18px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    table.tourn-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        border: 1px solid #d0d7de;
        border-radius: 8px;
    }
    table.tourn-table th {
        background-color: #1f2937;
        color: #ffffff;
        padding: 10px 12px;
        text-align: right;
        font-weight: 600;
        border-bottom: 2px solid #374151;
    }
    table.tourn-table th:first-child,
    table.tourn-table th:nth-child(2) {
        text-align: left;
    }
    table.tourn-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #e5e7eb;
        text-align: right;
        color: #111827;
    }
    table.tourn-table td:first-child,
    table.tourn-table td:nth-child(2) {
        text-align: left;
        font-weight: 600;
    }
    tr.row-normal {
        background-color: #ffffff;
        transition: background-color 0.15s ease;
    }
    tr.row-normal:hover {
        background-color: #f3f4f6;
    }
    tr.row-best {
        background-color: #d1fae5 !important;
        border-left: 5px solid #059669;
    }
    tr.row-best td {
        color: #065f46 !important;
        font-weight: 700;
    }
    tr.row-worst {
        background-color: #fee2e2 !important;
        border-left: 5px solid #dc2626;
    }
    tr.row-worst td {
        color: #991b1b !important;
        font-weight: 700;
    }
    .strat-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        border-bottom: 1px dashed #6b7280;
    }
    .strat-tooltip .tooltiptext {
        visibility: hidden;
        width: 420px;
        background-color: #111827;
        color: #f9fafb;
        text-align: left;
        border-radius: 6px;
        padding: 10px 12px;
        position: absolute;
        z-index: 9999;
        top: 125%;
        left: 0;
        opacity: 0;
        transition: opacity 0.18s ease-in-out;
        font-weight: 400;
        font-size: 13px;
        line-height: 1.45;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
        pointer-events: none;
    }
    .strat-tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """

    html_out = [css, '<div class="tourn-wrap"><table class="tourn-table"><thead><tr><th>Rank</th>']
    for col in display_cols:
        html_out.append(f"<th>{html.escape(col)}</th>")
    html_out.append("</tr></thead><tbody>")

    for rank_idx, row in df.iterrows():
        if rank_idx == best_idx:
            row_cls = "row-best"
            badge = "🏆 "
        elif rank_idx == worst_idx:
            row_cls = "row-worst"
            badge = "⚠️ "
        else:
            row_cls = "row-normal"
            badge = ""

        rule_escaped = html.escape(str(row["Rule Tooltip"]))
        strat_escaped = html.escape(str(row["Strategy"]))

        html_out.append(f'<tr class="{row_cls}" title="{rule_escaped}">')
        html_out.append(f"<td>#{rank_idx}</td>")
        html_out.append(
            f'<td><div class="strat-tooltip">{badge}{strat_escaped}'
            f'<span class="tooltiptext"><strong>Strategy Rules:</strong><br>{rule_escaped}</span>'
            f"</div></td>"
        )
        html_out.append(f'<td>{row["Annualized Return (%)"]:+.2f}%</td>')
        html_out.append(f'<td>{row["Total Net Return (%)"]:+.2f}%</td>')
        html_out.append(f'<td>${row["Ending Equity ($)"]:,.2f}</td>')
        html_out.append(f'<td>{row["Sharpe Ratio"]:.2f}</td>')
        html_out.append(f'<td>{row["Max Drawdown (%)"]:.2f}%</td>')
        html_out.append(f'<td>{row["Win Rate (%)"]:.1f}%</td>')
        html_out.append(f'<td>{row["Profit Factor"]:.2f}</td>')
        html_out.append(f'<td>{int(row["Total Trades"]):,}</td>')
        html_out.append("</tr>")

    html_out.append("</tbody></table></div>")
    return "".join(html_out)


# =====================================================================
# 6. SIDEBAR CONTROLS & EXECUTION
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

    st.header("2. Market & Data Integrity")
    pub_client = HFDataClient(api_key="")
    symbols_df, sym_err = pub_client.list_symbols()
    tickers = symbols_df["ticker"].tolist() if not symbols_df.empty else ["AAPL", "NVDA", "MSFT", "SPY", "TSLA"]
    default_idx = tickers.index("AAPL") if "AAPL" in tickers else 0

    ticker = st.selectbox("Ticker", options=tickers, index=default_idx)
    timeframe = st.selectbox("Timeframe", options=["1min", "5min", "15min", "30min"], index=0)
    version = st.selectbox("Data Version", options=["clean", "raw"], index=0)
    drop_incomplete = st.checkbox(
        "Omit Trading Days with Missing Intraday Bars (09:30–16:00)",
        value=True,
        help="Automatically removes any trading day that has fewer than the expected number of regular-session bars.",
    )

    st.header("3. Tournament Parameters")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-02"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2026-10-10"))
    x_lookback = st.slider(
        "Prior X-Bars Lookback (for Highest Close Exit & Donchian)",
        min_value=3,
        max_value=50,
        value=10,
        step=1,
    )
    orb_minutes = st.slider("Opening Range Window (mins)", min_value=5, max_value=60, value=15, step=5)
    base_capital = st.number_input("Position Size ($ Cap)", min_value=1000.0, value=20000.0, step=1000.0)
    commission = st.number_input("Commission ($ / trade)", min_value=0.0, value=1.0, step=0.25)

    run_btn = st.button("🚀 Run 10-Strategy Tournament", type="primary", use_container_width=True)

if sym_err:
    st.error(f"**Public Endpoint (`/symbols`) Error:** {sym_err}")
else:
    st.caption(f"✅ Connected to `api.hfdatalibrary.com` — {len(tickers)} tickers available.")

if not api_key or api_key == "your-key-here":
    st.info("👈 Paste your **HF Data Library `X-API-Key`** in the left sidebar and click **Run 10-Strategy Tournament**.")
    st.stop()

if run_btn:
    client = HFDataClient(api_key=api_key)

    with st.status("Running API Verification, Missing-Data Audit & 10-Strategy Tournament...", expanded=True) as status:
        ok, auth_msg = client.test_auth(ticker)
        if not ok:
            status.update(label="API Key Verification Failed", state="error")
            st.error(f"**API Diagnostic Result:**\n\n`{auth_msg}`")
            st.stop()
        st.write(f"✅ {auth_msg}")

        status.update(label=f"Downloading/Loading {ticker} ({timeframe}, {version}) Parquet...", state="running")
        bars_df, dl_err = client.get_bars(ticker, timeframe=timeframe, version=version)
        if dl_err or bars_df is None:
            status.update(label="Download Failed", state="error")
            st.error(f"**Download Error:** `{dl_err}`")
            st.stop()

        status.update(label="Auditing time series for missing intraday bars...", state="running")
        clean_df, audit_info, incomplete_df = prepare_and_audit_data(
            bars_df,
            timeframe=timeframe,
            start_date=str(start_date),
            end_date=str(end_date),
            drop_incomplete_days=drop_incomplete,
        )

        if clean_df.empty:
            status.update(label="No Valid Trading Sessions Found", state="error")
            st.warning(
                "No valid trading sessions remained after filtering. Try unchecking 'Omit Trading Days with Missing Intraday Bars' or widening the date range."
            )
            st.stop()

        status.update(label="Computing indicators & simulating 10 intraday strategies...", state="running")
        df_ind = precompute_indicators(clean_df, x_lookback=x_lookback, orb_minutes=orb_minutes)
        tourn_df, eq_curves = run_all_10_strategies(
            df_ind,
            x_lookback=x_lookback,
            orb_minutes=orb_minutes,
            base_capital=base_capital,
            commission=commission,
        )
        status.update(label="Tournament Complete!", state="complete", expanded=False)

    # -----------------------------------------------------------------
    # SECTION A: DATA START & MISSING-DATA INTEGRITY REPORT
    # -----------------------------------------------------------------
    st.subheader(f"🗓️ Data Coverage & Missing-Bar Integrity Check — {ticker} ({timeframe})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backtest Data Begins (ET)", audit_info["bt_first_ts"])
    c2.metric("Backtest Data Ends (ET)", audit_info["bt_last_ts"])
    c3.metric(
        "Sessions Tested / Total Found",
        f"{audit_info['sessions_tested']} / {audit_info['total_sessions_found']}",
        delta=f"-{audit_info['incomplete_sessions']} incomplete omitted" if drop_incomplete and audit_info["incomplete_sessions"] > 0 else "All sessions kept",
    )
    c4.metric(
        "Missing Intraday Bars Detected",
        f"{audit_info['total_missing_bars']:,}",
        help=f"Expected {audit_info['expected_bars_per_day']} regular-session bars per day for {timeframe} data.",
    )

    st.caption(
        f"**Full Ticker Archive Range on Server:** Begins `{audit_info['raw_first_ts']}` and ends `{audit_info['raw_last_ts']}` | "
        f"**NaN Rows Cleaned:** `{audit_info['nan_rows_removed']}` | "
        f"**Duplicate Timestamps Removed:** `{audit_info['duplicate_timestamps_removed']}`"
    )

    if not incomplete_df.empty:
        with st.expander(f"🔍 Inspect {len(incomplete_df)} Flagged Sessions with Missing Data (09:30–16:00 ET)"):
            st.dataframe(incomplete_df, use_container_width=True)

    # -----------------------------------------------------------------
    # SECTION B: 10-STRATEGY TOURNAMENT TABLE (GREEN/RED + TOOLTIPS)
    # -----------------------------------------------------------------
    st.subheader("🏆 10-Strategy Intraday Tournament Results")
    st.caption(
        "💡 **Hover your mouse over any Strategy name** to view its exact entry and exit rules. "
        "The **#1 Best Strategy is highlighted Green**, and the **#10 Worst Strategy is highlighted Red**. "
        f"All entries execute at **Next Bar's Open** (`Open[t+1]`) with a **${base_capital:,.0f}** non-compounding position cap and **${commission:.2f}** commission/trade."
    )

    html_table = render_tournament_html_table(tourn_df)
    st.markdown(html_table, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # SECTION C: EQUITY CURVES COMPARISON
    # -----------------------------------------------------------------
    st.subheader("📊 Strategy Equity Curves ($ Account Value)")
    if eq_curves:
        eq_df = pd.DataFrame(eq_curves)
        st.line_chart(eq_df)
