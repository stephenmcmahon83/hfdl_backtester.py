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
        self.session.headers.update({"User-Agent": "HFDL-Tournament-Backtester/3.7"})

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
            "l_ent": (
                (df_ind["bar_time"] > df_ind["orb_end_time"]) &
                (df_ind["close"] > df_ind["orb_high"]) &
                (df_ind["close"] > df_ind["vwap"]) &
                (df_ind["volume"] > 1.25 * df_ind["vol_ma20"])
            ).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["vwap"]).to_numpy(),
            "s_ent": (
                (df_ind["bar_time"] > df_ind["orb_end_time"]) &
                (df_ind["close"] < df_ind["orb_low"]) &
                (df_ind["close"] < df_ind["vwap"]) &
                (df_ind["volume"] > 1.25 * df_ind["vol_ma20"])
            ).to_numpy(),
            "s_ext": (df_ind["close"] > df_ind["vwap"]).to_numpy(),
        },
        {
            "name": "5. Session VWAP 2-SD Mean Reversion Band Bounce",
            "rule": "LONG ENTRY: Close < VWAP - 2.0 SD. SHORT ENTRY: Close > VWAP + 2.0 SD. EXIT: Touch VWAP.",
            "l_ent": (df_ind["close"] < (df_ind["vwap"] - 2.0 * df_ind["vwap_std"])).to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["vwap"]).to_numpy(),
            "s_ent": (df_ind["close"] > (df_ind["vwap"] + 2.0 * df_ind["vwap_std"])).to_numpy(),
            "s_ext": (df_ind["close"] <= df_ind["vwap"]).to_numpy(),
        },
        {
            "name": f"6. Connors Intraday RSI(2) < 5 Dip Buyer -> {x_lookback}-Bar High Exit",
            "rule": f"ENTRY: RSI(2) < 5 while above VWAP -> buy Next Open. EXIT: RSI(2) > 70 or Highest Close of prior {x_lookback} bars.",
            "l_ent": ((df_ind["rsi2"] < 5.0) & (df_ind["close"] > df_ind["vwap"])).to_numpy(),
            "l_ext": ((df_ind["rsi2"] > 70.0) | (df_ind["close"] >= df_ind["highest_close_x"])).to_numpy(),
        },
        {
            "name": f"7. Donchian Channel ({x_lookback}-Bar) Volatility Breakout",
            "rule": f"LONG ENTRY: Close > prior {x_lookback}-bar Highest High on Volume > 1.2x MA. EXIT: Close < Lowest Close.",
            "l_ent": ((df_ind["close"] > df_ind["donchian_high_x"]) & (df_ind["volume"] > 1.2 * df_ind["vol_ma20"])).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["lowest_close_x"]).to_numpy(),
        },
        {
            "name": "8. Intraday Internal Bar Strength (IBS < 0.10) Pullback",
            "rule": "ENTRY: IBS < 0.10 (close in bottom 10% of range) while above 20 SMA -> buy Next Open. EXIT: Close > prior bar High.",
            "l_ent": ((df_ind["ibs"] < 0.10) & (df_ind["close"] > df_ind["sma20"])).to_numpy(),
            "l_ext": (df_ind["close"] > df_ind["prev_high"]).to_numpy(),
        },
        {
            "name": "9. Bollinger Band (20, 2) Squeeze & Volume Expansion",
            "rule": "LONG ENTRY: Close breaks Upper BB on Volume > 1.5x MA. EXIT: Close < 20 SMA.",
            "l_ent": ((df_ind["close"] > (df_ind["sma20"] + 2.0 * df_ind["std20"])) & (df_ind["volume"] > 1.5 * df_ind["vol_ma20"])).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["sma20"]).to_numpy(),
        },
        {
            "name": "10. EMA(9/21) Momentum Cross + VWAP Confirmation",
            "rule": "LONG ENTRY: 9 EMA crosses above 21 EMA while Close > VWAP. SHORT ENTRY: 9 EMA crosses below 21 EMA while Close < VWAP.",
            "l_ent": ((df_ind["ema9"] > df_ind["ema21"]) & (df_ind["prev_ema9"] <= df_ind["prev_ema21"]) & (df_ind["close"] > df_ind["vwap"])).to_numpy(),
            "l_ext": (df_ind["ema9"] < df_ind["ema21"]).to_numpy(),
            "s_ent": ((df_ind["ema9"] < df_ind["ema21"]) & (df_ind["prev_ema9"] >= df_ind["prev_ema21"]) & (df_ind["close"] < df_ind["vwap"])).to_numpy(),
            "s_ext": (df_ind["ema9"] > df_ind["ema21"]).to_numpy(),
        },
        {
            "name": "11. 12-Month Low Capitulation Mean-Reversion Bounce",
            "rule": "ENTRY: Price touches rolling 12-month low capitulation zone (within 1.5% of 252-session low) with RSI(2) < 10 -> buy Next Open. EXIT: Revert to 20 SMA or 15:55 ET.",
            "l_ent": (df_ind["near_capitulation_low"] & (df_ind["rsi2"] < 10.0)).to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["sma20"]).to_numpy(),
        },
        {
            "name": "12. Multi-Timeframe Trend + Lower-High Pullback Entry",
            "rule": "ENTRY: Price pulls back from 20-bar local high while remaining above 50 SMA (uptrend filter) and RSI(2) < 15 -> buy Next Open. EXIT: Close > recent local high or 15:55 ET.",
            "l_ent": (df_ind["pulled_back_from_high"] & (df_ind["rsi2"] < 15.0)).to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["highest_close_x"]).to_numpy(),
        },
        {
            "name": "13. 200-SMA Macro Trend Filter + Intraday VWAP Dip Buyer",
            "rule": "ENTRY: Close > 200-bar SMA (macro bull trend) and Close drops below Session VWAP - 1.0 SD -> buy Next Open. EXIT: Touch VWAP + 0.5 SD or 15:55 ET.",
            "l_ent": ((df_ind["close"] > df_ind["sma200"]) & (df_ind["close"] < df_ind["vwap"] - 1.0 * df_ind["vwap_std"])).to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["vwap"] + 0.5 * df_ind["vwap_std"]).to_numpy(),
        },
        {
            "name": "14. Volume Shock Spike (3x Volume MA) Breakout",
            "rule": "ENTRY: Intraday bar volume spikes to > 3.0x 20-bar volume MA and Close > prior bar High -> buy Next Open. EXIT: Close < 9 EMA or 15:55 ET.",
            "l_ent": ((df_ind["volume"] > 3.0 * df_ind["vol_ma20"]) & (df_ind["close"] > df_ind["prev_high"])).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["ema9"]).to_numpy(),
        },
        {
            "name": "15. Opening Drive Momentum Continuation (First 15m > 1%)",
            "rule": "ENTRY: Opening 15-minute drive return > +1.0% and Close > VWAP at bar 16 -> buy Next Open. EXIT: Close crosses below VWAP or 15:55 ET.",
            "l_ent": (
                (df_ind["bar_time"] == (pd.Timestamp("2000-01-01 09:30:00") + pd.Timedelta(minutes=orb_minutes)).strftime("%H:%M")) &
                (df_ind["open_drive_ret"] > 0.01) &
                (df_ind["close"] > df_ind["vwap"])
            ).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["vwap"]).to_numpy(),
        },
    ]

    rows = []
    eq_curves = {}

    for spec in strategies_spec:
        s_ent = spec.get("s_ent", None)
        s_ext = spec.get("s_ext", None)
        eq, m = simulate_round_trip_trades(
            df_ind,
            long_entry=spec["l_ent"],
            long_exit=spec["l_ext"],
            short_entry=s_ent,
            short_exit=s_ext,
            base_capital=base_capital,
            commission_per_order=commission,
        )
        if m:
            row = {"Strategy": spec["name"], "Rule Tooltip": spec["rule"]}
            row.update(m)
            rows.append(row)
            eq_curves[spec["name"]] = eq

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        res_df = res_df.sort_values(by="Annualized Return (%)", ascending=False).reset_index(drop=True)
        res_df.index = res_df.index + 1
        res_df.index.name = "Rank"

    return res_df, eq_curves


# =====================================================================
# 5. HTML TOURNAMENT TABLE WITH HOVER TOOLTIPS & GREEN/RED HIGHLIGHTING
# =====================================================================
def render_tournament_html_table(df: pd.DataFrame) -> str:
    best_idx = df["Annualized Return (%)"].idxmax()
    worst_idx = df["Annualized Return (%)"].idxmin()

    display_cols = [
        "Strategy",
        "Annualized Return (%)",
        "Total Net Return (%)",
        "Ending Equity ($)",
        "Sharpe Ratio",
        "Max Drawdown (%)",
        "Trade Win Rate (%)",
        "Profit Factor",
        "Round-Trip Trades",
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
        html_out.append(f'<td>{row["Trade Win Rate (%)"]:.1f}%</td>')
        html_out.append(f'<td>{row["Profit Factor"]:.2f}</td>')
        html_out.append(f'<td>{int(row["Round-Trip Trades"]):,}</td>')
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

    min_completeness = st.slider(
        "Minimum Session Completeness (%)",
        min_value=50.0,
        max_value=100.0,
        value=95.0,
        step=1.0,
        help="Keeps normal trading days while dropping broken or half-day sessions.",
    )
    ffill_quiet = st.checkbox(
        "Forward-Fill Quiet 0-Volume Minutes",
        value=True,
        help="Pads quiet minutes with prior close and Volume=0 to keep rolling indicators aligned.",
    )

    st.header("3. Tournament Parameters")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-02"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2026-10-10"))
    x_lookback = st.slider("Prior X-Bars Lookback", 3, 50, 10, 1)
    orb_minutes = st.slider("Opening Range Window (mins)", 5, 60, 15, 5)
    base_capital = st.number_input("Position Size ($ Cap)", 1000.0, 20000.0, 1000.0)
    commission = st.number_input("Commission ($ / order)", 0.0, 1.0, 0.25)

    run_btn = st.button("🚀 Run 15-Strategy Tournament", type="primary", use_container_width=True)

if sym_err:
    st.error(f"**Public Endpoint Error:** {sym_err}")
else:
    st.caption(f"✅ Connected to `api.hfdatalibrary.com` — {len(tickers)} tickers available.")

if not api_key or api_key == "your-key-here":
    st.info("👈 Paste your **HF Data Library `X-API-Key`** in the left sidebar and click **Run 15-Strategy Tournament**.")
    st.stop()

if run_btn:
    client = HFDataClient(api_key=api_key)

    with st.status("Running API Verification, Missing-Data Audit & 15-Strategy Tournament...", expanded=True) as status:
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
        clean_df, audit_info, flagged_df = prepare_and_audit_data(
            bars_df,
            timeframe=timeframe,
            start_date=str(start_date),
            end_date=str(end_date),
            min_completeness_pct=min_completeness,
            ffill_quiet_minutes=ffill_quiet,
        )

        if clean_df.empty:
            status.update(label="No Valid Trading Sessions Found", state="error")
            st.warning("No valid sessions remained. Lower the 'Minimum Session Completeness (%)' slider.")
            st.stop()

        status.update(label="Computing indicators & simulating 15 intraday strategies...", state="running")
        df_ind = precompute_indicators(clean_df, x_lookback=x_lookback, orb_minutes=orb_minutes)
        tourn_df, eq_curves = run_all_15_strategies(
            df_ind,
            x_lookback=x_lookback,
            orb_minutes=orb_minutes,
            base_capital=base_capital,
            commission=commission,
        )
        status.update(label="Tournament Complete!", state="complete", expanded=False)

    st.subheader(f"🗓️ Data Coverage & Missing-Bar Integrity Check — {ticker} ({timeframe})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backtest Data Begins (ET)", audit_info["bt_first_ts"])
    c2.metric("Backtest Data Ends (ET)", audit_info["bt_last_ts"])
    c3.metric(
        "Sessions Tested / Total Found",
        f"{audit_info['sessions_tested']:,} / {audit_info['total_sessions_found']:,}",
        delta=f"{audit_info['calendar_years']} Calendar Yrs ({audit_info['dropped_sessions']} low-coverage days omitted)",
    )
    c4.metric("Quiet/Missing Bars Detected", f"{audit_info['total_missing_bars']:,}")

    st.caption(
        f"**Full Ticker Archive Range on Server:** Begins `{audit_info['raw_first_ts']}` and ends `{audit_info['raw_last_ts']}` | "
        f"**Calendar Span Tested:** `{audit_info['calendar_years']} years` | "
        f"**NaN Rows Cleaned:** `{audit_info['nan_rows_removed']}`"
    )

    if not flagged_df.empty:
        with st.expander(f"🔍 Inspect {len(flagged_df)} Omitted Sessions Below {min_completeness:.0f}% Bar Completeness"):
            st.dataframe(flagged_df, use_container_width=True)

    st.subheader("🏆 15-Strategy Intraday Tournament Results")
    st.caption(
        "💡 **Hover your mouse over any Strategy name** to view its exact entry and exit rules. "
        "The **#1 Best Strategy is highlighted Green**, and the **#15 Worst Strategy is highlighted Red**. "
        f"All entries/exits execute at **Next Bar's Open** with a **${base_capital:,.0f}** non-compounding cap "
        f"and **${commission:.2f}** order commission (**${commission * 2:.2f}** per round trip)."
    )

    html_table = render_tournament_html_table(tourn_df)
    st.markdown(html_table, unsafe_allow_html=True)

    st.subheader("📊 Strategy Equity Curves ($ Account Value)")
    if eq_curves:
        eq_df = pd.DataFrame(eq_curves)
        st.line_chart(eq_df)
