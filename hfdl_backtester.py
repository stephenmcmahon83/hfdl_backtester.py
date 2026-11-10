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

st.set_page_config(page_title="HF Data Library — 15-Strategy Intraday Tournament", layout="wide")
st.title("📈 HF Data Library — 15-Strategy Intraday Backtest Tournament")


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
        self.session.headers.update({"User-Agent": "HFDL-Tournament-Backtester/3.6"})

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
# 2. DATA PREPARATION & MISSING-BAR AUDIT
# =====================================================================
EXPECTED_BARS_PER_DAY = {
    "1min": 390,
    "5min": 78,
    "15min": 26,
    "30min": 13,
}


def prepare_and_audit_data(
    df_raw: pd.DataFrame,
    timeframe: str,
    start_date: str,
    end_date: str,
    min_completeness_pct: float = 95.0,
    ffill_quiet_minutes: bool = True,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
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

    raw_first_ts = df.index.min() if not df.empty else None
    raw_last_ts = df.index.max() if not df.empty else None

    df = df.loc[str(start_date):str(end_date)]
    df = df.between_time("09:30", "15:59").copy()

    if df.empty:
        return df, {}, pd.DataFrame()

    ohlc_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    nan_rows_count = int(df[ohlc_cols].isna().any(axis=1).sum())
    if nan_rows_count > 0:
        df = df.dropna(subset=ohlc_cols)

    dup_timestamps = int(df.index.duplicated().sum())
    if dup_timestamps > 0:
        df = df[~df.index.duplicated(keep="first")]

    df["session_date"] = df.index.normalize()

    expected_bars = EXPECTED_BARS_PER_DAY.get(timeframe, 390)
    freq_map = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min"}
    pd_freq = freq_map.get(timeframe, "1min")

    counts = df.groupby("session_date").size().rename("Observed Bars")
    first_bars = df.groupby("session_date").apply(lambda g: g.index.min().strftime("%H:%M")).rename("First Bar (ET)")
    last_bars = df.groupby("session_date").apply(lambda g: g.index.max().strftime("%H:%M")).rename("Last Bar (ET)")

    audit_df = pd.concat([first_bars, last_bars, counts], axis=1).reset_index()
    audit_df["Session Date"] = audit_df["session_date"].dt.strftime("%Y-%m-%d")
    audit_df["Expected Bars"] = expected_bars
    audit_df["Missing Bars"] = (expected_bars - audit_df["Observed Bars"]).clip(lower=0)
    audit_df["Completeness (%)"] = ((audit_df["Observed Bars"] / expected_bars) * 100.0).round(1)
    audit_df["Meets Threshold"] = audit_df["Completeness (%)"] >= min_completeness_pct

    total_sessions_found = len(audit_df)
    flagged_df = audit_df[~audit_df["Meets Threshold"]].copy()
    dropped_sessions_count = len(flagged_df)
    total_missing_bars = int(audit_df["Missing Bars"].sum())

    valid_session_dates = audit_df.loc[audit_df["Meets Threshold"], "session_date"]
    df = df[df["session_date"].isin(valid_session_dates)].copy()

    if not df.empty and ffill_quiet_minutes:
        unique_days = df["session_date"].unique()
        intraday_offsets = pd.timedelta_range(start="09:30:00", end="15:59:00", freq=pd_freq)
        full_idx = pd.DatetimeIndex(
            np.add.outer(unique_days, intraday_offsets).ravel()
        )
        df = df.reindex(full_idx)
        df["session_date"] = df.index.normalize()
        df["close"] = df.groupby("session_date")["close"].ffill().bfill()
        for col in ["open", "high", "low"]:
            df[col] = df[col].fillna(df["close"])
        df["volume"] = df["volume"].fillna(0.0)

    bt_first_ts = df.index.min() if not df.empty else None
    bt_last_ts = df.index.max() if not df.empty else None
    cal_years = max((bt_last_ts - bt_first_ts).days / 365.25, 1.0 / 252.0) if (bt_first_ts and bt_last_ts) else 0.0

    audit_summary = {
        "raw_first_ts": raw_first_ts.strftime("%Y-%m-%d %H:%M:%S ET") if raw_first_ts is not None else "N/A",
        "raw_last_ts": raw_last_ts.strftime("%Y-%m-%d %H:%M:%S ET") if raw_last_ts is not None else "N/A",
        "bt_first_ts": bt_first_ts.strftime("%Y-%m-%d %H:%M:%S ET") if bt_first_ts is not None else "N/A",
        "bt_last_ts": bt_last_ts.strftime("%Y-%m-%d %H:%M:%S ET") if bt_last_ts is not None else "N/A",
        "calendar_years": round(cal_years, 2),
        "total_sessions_found": total_sessions_found,
        "sessions_tested": int(df["session_date"].nunique()) if not df.empty else 0,
        "dropped_sessions": dropped_sessions_count,
        "exact_390_sessions": int((audit_df["Observed Bars"] >= expected_bars).sum()),
        "total_missing_bars": total_missing_bars,
        "nan_rows_removed": nan_rows_count,
        "duplicate_timestamps_removed": dup_timestamps,
        "expected_bars_per_day": expected_bars,
    }

    display_cols = [
        "Session Date", "First Bar (ET)", "Last Bar (ET)",
        "Observed Bars", "Expected Bars", "Missing Bars", "Completeness (%)", "Meets Threshold"
    ]
    return df, audit_summary, flagged_df[display_cols] if not flagged_df.empty else pd.DataFrame()


# =====================================================================
# 3. ADVANCED INDICATOR PRECOMPUTATION
# =====================================================================
def precompute_indicators(df: pd.DataFrame, x_lookback: int = 10, orb_minutes: int = 15) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("session_date")

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.groupby(df["session_date"]).cumsum()
    cum_vol = g["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = (cum_pv / cum_vol).groupby(df["session_date"]).ffill().fillna(df["close"])

    sq_diff = ((tp - df["vwap"]) ** 2) * df["volume"]
    vwap_var = sq_diff.groupby(df["session_date"]).cumsum() / cum_vol
    df["vwap_std"] = np.sqrt(vwap_var.clip(lower=0.0)).fillna(0.0)

    prev_close = g["close"].shift(1)
    is_lower_close = df["close"] < prev_close
    is_higher_close = df["close"] > prev_close

    df["down_5_streak"] = is_lower_close.groupby(df["session_date"]).transform(lambda s: s.rolling(5, min_periods=5).sum() == 5)
    df["down_3_streak"] = is_lower_close.groupby(df["session_date"]).transform(lambda s: s.rolling(3, min_periods=3).sum() == 3)
    df["up_5_streak"] = is_higher_close.groupby(df["session_date"]).transform(lambda s: s.rolling(5, min_periods=5).sum() == 5)

    df["highest_close_x"] = g["close"].transform(lambda s: s.rolling(x_lookback, min_periods=2).max())
    df["lowest_close_x"] = g["close"].transform(lambda s: s.rolling(x_lookback, min_periods=2).min())
    df["donchian_high_x"] = g["high"].transform(lambda s: s.shift(1).rolling(x_lookback, min_periods=3).max())
    df["donchian_low_x"] = g["low"].transform(lambda s: s.shift(1).rolling(x_lookback, min_periods=3).min())

    df["sma50"] = g["close"].transform(lambda s: s.rolling(50, min_periods=10).mean())
    df["sma200"] = g["close"].transform(lambda s: s.rolling(200, min_periods=20).mean())
    
    rolling_min_252 = g["low"].transform(lambda s: s.rolling(252, min_periods=20).min())
    df["near_capitulation_low"] = df["low"] <= (rolling_min_252 * 1.015)

    recent_high_20 = g["high"].transform(lambda s: s.rolling(20, min_periods=5).max())
    df["pulled_back_from_high"] = (df["close"] < recent_high_20 * 0.985) & (df["close"] > df["sma50"])

    delta = g["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.groupby(df["session_date"]).transform(lambda s: s.ewm(alpha=0.5, adjust=False).mean())
    avg_loss = loss.groupby(df["session_date"]).transform(lambda s: s.ewm(alpha=0.5, adjust=False).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi2"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    df["ema9"] = g["close"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["ema21"] = g["close"].transform(lambda s: s.ewm(span=21, adjust=False).mean())
    df["prev_ema9"] = g["ema9"].shift(1)
    df["prev_ema21"] = g["ema21"].shift(1)

    df["sma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df["std20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=5).std())
    df["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())

    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["ibs"] = ((df["close"] - df["low"]) / hl_range).fillna(0.5)
    df["prev_high"] = g["high"].shift(1)

    orb_end = (pd.Timestamp("2000-01-01 09:30:00") + pd.Timedelta(minutes=orb_minutes)).strftime("%H:%M")
    orb_slice = df.between_time("09:30", orb_end)
    orb_h = orb_slice.groupby("session_date")["high"].max().rename("orb_high")
    orb_l = orb_slice.groupby("session_date")["low"].min().rename("orb_low")
    df = df.join(orb_h, on="session_date").join(orb_l, on="session_date")

    # Opening Range Drive / Return calculation for Strategy #15
    first_open = g["open"].transform("first")
    orb_close_map = orb_slice.groupby("session_date")["close"].last().rename("orb_close")
    df = df.join(orb_close_map, on="session_date")
    df["open_drive_ret"] = (df["orb_close"] / first_open) - 1.0

    df["bar_time"] = df.index.strftime("%H:%M")
    df["orb_end_time"] = orb_end
    return df


def simulate_round_trip_trades(
    df: pd.DataFrame,
    long_entry: np.ndarray,
    long_exit: np.ndarray,
    short_entry: Optional[np.ndarray] = None,
    short_exit: Optional[np.ndarray] = None,
    base_capital: float = 20000.0,
    commission_per_order: float = 1.0,
) -> Tuple[pd.Series, Dict[str, float]]:
    n = len(df)
    if short_entry is None:
        short_entry = np.zeros(n, dtype=bool)
    if short_exit is None:
        short_exit = np.zeros(n, dtype=bool)

    opens = df["open"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    session_dates = df["session_date"].to_numpy()
    bar_times = df["bar_time"].to_numpy()

    can_enter = (bar_times >= "09:35") & (bar_times < "15:50")
    eod_flat_signal = bar_times >= "15:54"

    l_ent = long_entry & can_enter
    s_ent = short_entry & can_enter
    l_ext = long_exit | eod_flat_signal
    s_ext = short_exit | eod_flat_signal

    equity = base_capital
    peak_equity = base_capital
    max_dd_pct = 0.0

    trade_pnls = []
    daily_pnl_map = {}

    unique_sessions = pd.unique(session_dates)
    for s_dt in unique_sessions:
        daily_pnl_map[pd.Timestamp(s_dt)] = 0.0

    pos = 0
    entry_price = 0.0
    pos_dollars = 0.0

    for i in range(n - 1):
        curr_session = session_dates[i]
        next_session = session_dates[i + 1]
        is_last_bar_of_session = curr_session != next_session

        if pos != 0 and is_last_bar_of_session:
            exit_price = closes[i]
            raw_ret = (exit_price / entry_price - 1.0) if pos == 1 else (entry_price / exit_price - 1.0)
            net_pnl = (pos_dollars * raw_ret) - (2.0 * commission_per_order)
            equity += net_pnl
            trade_pnls.append(net_pnl)
            daily_pnl_map[pd.Timestamp(curr_session)] += net_pnl

            if equity > peak_equity:
                peak_equity = equity
            dd = ((equity - peak_equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
            if dd < max_dd_pct:
                max_dd_pct = dd

            pos = 0
            continue

        if pos == 1 and l_ext[i]:
            exit_price = opens[i + 1]
            raw_ret = (exit_price / entry_price) - 1.0
            net_pnl = (pos_dollars * raw_ret) - (2.0 * commission_per_order)
            equity += net_pnl
            trade_pnls.append(net_pnl)
            daily_pnl_map[pd.Timestamp(curr_session)] += net_pnl

            if equity > peak_equity:
                peak_equity = equity
            dd = ((equity - peak_equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
            if dd < max_dd_pct:
                max_dd_pct = dd

            pos = 0

        elif pos == -1 and s_ext[i]:
            exit_price = opens[i + 1]
            raw_ret = (entry_price / exit_price) - 1.0
            net_pnl = (pos_dollars * raw_ret) - (2.0 * commission_per_order)
            equity += net_pnl
            trade_pnls.append(net_pnl)
            daily_pnl_map[pd.Timestamp(curr_session)] += net_pnl

            if equity > peak_equity:
                peak_equity = equity
            dd = ((equity - peak_equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
            if dd < max_dd_pct:
                max_dd_pct = dd

            pos = 0

        if pos == 0 and not is_last_bar_of_session and equity > 0:
            if l_ent[i]:
                pos = 1
                entry_price = opens[i + 1]
                pos_dollars = min(base_capital, max(equity, 0.0))
            elif s_ent[i]:
                pos = -1
                entry_price = opens[i + 1]
                pos_dollars = min(base_capital, max(equity, 0.0))

    daily_pnl_series = pd.Series(daily_pnl_map).sort_index()
    eq_series = base_capital + daily_pnl_series.cumsum()

    full_eq = pd.concat([pd.Series([base_capital]), eq_series])
    eod_peaks = full_eq.cummax()
    eod_dd = float((((full_eq - eod_peaks) / eod_peaks) * 100.0).min())
    max_dd_pct = min(max_dd_pct, eod_dd)

    if len(daily_pnl_series) >= 2:
        cal_days = (daily_pnl_series.index[-1] - daily_pnl_series.index[0]).days
        cal_years = max(cal_days / 365.25, 1.0 / 12.0)
    else:
        cal_years = 1.0

    total_ret_pct = ((equity - base_capital) / base_capital) * 100.0
    ann_ret_pct = total_ret_pct / cal_years

    daily_rets_on_base = daily_pnl_series / base_capital
    std_d = daily_rets_on_base.std()
    sharpe = float((daily_rets_on_base.mean() / std_d) * np.sqrt(252.0)) if std_d > 0 else 0.0

    trades_arr = np.array(trade_pnls, dtype=np.float64)
    n_trades = len(trades_arr)
    if n_trades > 0:
        wins = trades_arr[trades_arr > 0]
        losses = trades_arr[trades_arr < 0]
        win_rate = (len(wins) / n_trades) * 100.0
        gross_win = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        pf = (gross_win / gross_loss) if gross_loss > 0 else (99.9 if gross_win > 0 else 0.0)
    else:
        win_rate = 0.0
        pf = 0.0

    metrics = {
        "Annualized Return (%)": round(ann_ret_pct, 2),
        "Total Net Return (%)": round(total_ret_pct, 2),
        "Ending Equity ($)": round(equity, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd_pct, 2),
        "Trade Win Rate (%)": round(win_rate, 1),
        "Profit Factor": round(min(pf, 99.9), 2),
        "Round-Trip Trades": int(n_trades),
    }
    return eq_series, metrics


# =====================================================================
# 4. THE 15 STRATEGIES
# =====================================================================
def run_all_15_strategies(
    df_ind: pd.DataFrame,
    x_lookback: int = 10,
    orb_minutes: int = 15,
    base_capital: float = 20000.0,
    commission: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    strategies_spec = [
        {
            "name": f"1. 5 Lower Closes -> Exit Highest Close ({x_lookback} Bars)",
            "rule": f"ENTRY: 5 consecutive lower closes; buy Next Bar's Open. EXIT: Close >= Highest Close of prior {x_lookback} bars or 15:55 ET.",
            "l_ent": df_ind["down_5_streak"].to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["highest_close_x"]).to_numpy(),
        },
        {
            "name": f"2. 3 Lower Closes Above VWAP -> Exit Highest Close ({x_lookback} Bars)",
            "rule": f"ENTRY: 3 consecutive lower closes above VWAP; buy Next Bar's Open. EXIT: Close >= Highest Close of prior {x_lookback} bars or VWAP - 1.5 SD.",
            "l_ent": (df_ind["down_3_streak"] & (df_ind["close"] > df_ind["vwap"])).to_numpy(),
            "l_ext": ((df_ind["close"] >= df_ind["highest_close_x"]) | (df_ind["close"] < df_ind["vwap"] - 1.5 * df_ind["vwap_std"])).to_numpy(),
        },
        {
            "name": f"3. 5 Consecutive Bars Reversal (Long & Short)",
            "rule": "LONG ENTRY: 5 lower closes -> buy Next Open. SHORT ENTRY: 5 higher closes -> short Next Open. Exit on reversal signals.",
            "l_ent": df_ind["down_5_streak"].to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["highest_close_x"]).to_numpy(),
            "s_ent": df_ind["up_5_streak"].to_numpy(),
            "s_ext": (df_ind["close"] <= df_ind["lowest_close_x"]).to_numpy(),
        },
        {
            "name": f"4. Opening Range Breakout ({orb_minutes}m) + Session VWAP Trend",
            "rule": f"LONG ENTRY: After {orb_minutes}m, Close > ORB High & VWAP & Volume > 1.25x MA. EXIT: Cross back through VWAP.",
            "l_ent": ((df_ind["bar_time"] > df_ind["orb_end_time"]) & (df_ind["close"] > df_ind["orb_high"]) & (df_ind["close"] > df_ind["vw
