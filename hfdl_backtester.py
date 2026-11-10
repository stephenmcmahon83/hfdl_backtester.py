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

st.set_page_config(page_title="HF Data Library — 15-Strategy Semi-HFT Tournament", layout="wide")
st.title("📈 HF Data Library — 15-Strategy Semi-HFT Tournament & Data Audit")


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
        self.session.headers.update({"User-Agent": "HFDL-SemiHFT-Tournament/5.2"})

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
# 2. DATA VALIDATOR & MISSING-BAR INTEGRITY AUDIT
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
    is_vxx: bool = False,
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

    if is_vxx and "2024-08-05" in df.index.strftime("%Y-%m-%d"):
        df = df[df.index.strftime("%Y-%m-%d") != "2024-08-05"].copy()

    ohlc_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    nan_rows_count = int(df[ohlc_cols].isna().any(axis=1).sum())
    if nan_rows_count > 0:
        df = df.dropna(subset=ohlc_cols)

    df["session_date"] = df.index.normalize()
    expected_bars = EXPECTED_BARS_PER_DAY.get(timeframe, 390)
    pd_freq = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min"}.get(timeframe, "1min")

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
    total_missing_bars = int(audit_df["Missing Bars"].sum())

    valid_session_dates = audit_df.loc[audit_df["Meets Threshold"], "session_date"]
    df = df[df["session_date"].isin(valid_session_dates)].copy()

    if not df.empty and ffill_quiet_minutes:
        unique_days = df["session_date"].unique()
        intraday_offsets = pd.timedelta_range(start="09:30:00", end="15:59:00", freq=pd_freq)
        full_idx = pd.DatetimeIndex(np.add.outer(unique_days, intraday_offsets).ravel())
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
        "dropped_sessions": len(flagged_df),
        "total_missing_bars": total_missing_bars,
        "nan_rows_removed": nan_rows_count,
    }

    display_cols = [
        "Session Date", "First Bar (ET)", "Last Bar (ET)",
        "Observed Bars", "Expected Bars", "Missing Bars", "Completeness (%)", "Meets Threshold"
    ]
    return df, audit_summary, flagged_df[display_cols] if not flagged_df.empty else pd.DataFrame()


# =====================================================================
# 3. SEMI-HFT & INTRADAY FEATURE PRECOMPUTATION
# =====================================================================
def precompute_indicators(df: pd.DataFrame, x_lookback: int = 10, orb_minutes: int = 15) -> pd.DataFrame:
    df = df.copy()
    # Ensure session_date column exists securely
    if "session_date" not in df.columns:
        df["session_date"] = df.index.normalize()

    g = df.groupby("session_date")

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.groupby("session_date").cumsum()
    cum_vol = g["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = (cum_pv / cum_vol).groupby("session_date").ffill().fillna(df["close"])

    # Returns & Z-Score
    df["ret_1m"] = g["close"].pct_change(1)
    df["ret_2m"] = g["close"].pct_change(2)
    df["ret_3m"] = g["close"].pct_change(3)
    df["ret_15m"] = g["close"].pct_change(15)

    df["ma_15"] = g["close"].transform(lambda x: x.rolling(15, min_periods=10).mean())
    df["std_15"] = g["close"].transform(lambda x: x.rolling(15, min_periods=10).std())
    df["z_15"] = (df["close"] - df["ma_15"]) / df["std_15"].replace(0, np.nan)

    df["day_open"] = g["open"].transform("first")
    df["ret_from_open"] = (df["close"] - df["day_open"]) / df["day_open"]

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["ibs"] = (df["close"] - df["low"]) / rng

    # Opening Range High / Low (15m and 30m)
    orb_end = (pd.Timestamp("2000-01-01 09:30:00") + pd.Timedelta(minutes=orb_minutes)).strftime("%H:%M")
    orb_slice = df.between_time("09:30", orb_end)
    orb_h = orb_slice.groupby("session_date")["high"].max().rename("or15_high")
    orb_l = orb_slice.groupby("session_date")["low"].min().rename("or15_low")
    df = df.join(orb_h, on="session_date").join(orb_l, on="session_date")

    or30_slice = df.between_time("09:30", "09:59")
    or30_l = or30_slice.groupby("session_date")["low"].min().rename("or30_low")
    or30_c = or30_slice.groupby("session_date")["close"].last().rename("or30_close")
    df = df.join(or30_l, on="session_date").join(or30_c, on="session_date")
    df["or30_ret"] = (df["or30_close"] - df["day_open"]) / df["day_open"]

    # Consecutive Up/Down Streaks
    prev_close = g["close"].shift(1)
    is_lower = df["close"] < prev_close
    is_higher = df["close"] > prev_close
    df["down_5_streak"] = is_lower.groupby("session_date").transform(lambda s: s.rolling(5, min_periods=5).sum() == 5)
    df["down_3_streak"] = is_lower.groupby("session_date").transform(lambda s: s.rolling(3, min_periods=3).sum() == 3)
    df["up_5_streak"] = is_higher.groupby("session_date").transform(lambda s: s.rolling(5, min_periods=5).sum() == 5)

    dirs = np.where(df["ret_1m"] > 0, 1, np.where(df["ret_1m"] < 0, -1, 0))
    dates = df["session_date"].values
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

    df["highest_close_x"] = g["close"].transform(lambda s: s.rolling(x_lookback, min_periods=2).max())
    df["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df["bar_time"] = df.index.strftime("%H:%M")
    return df


def simulate_round_trip_trades(
    df: pd.DataFrame,
    long_entry: np.ndarray,
    long_exit: np.ndarray,
    short_entry: Optional[np.ndarray] = None,
    short_exit: Optional[np.ndarray] = None,
    max_hold_bars: int = 120,
    stop_loss_pct: float = 0.008,
    profit_target_pct: float = 0.0075,
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

    can_enter = (bar_times >= "09:45") & (bar_times <= "15:40")
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
    for s_dt in pd.unique(session_dates):
        daily_pnl_map[pd.Timestamp(s_dt)] = 0.0

    pos = 0
    entry_price = 0.0
    entry_idx = 0
    pos_dollars = 0.0

    for i in range(n - 1):
        curr_session = session_dates[i]
        next_session = session_dates[i + 1]
        is_last_bar_of_session = curr_session != next_session

        if pos != 0:
            bars_held = i - entry_idx
            cur_px = closes[i]
            ret = (cur_px - entry_price) / entry_price if pos == 1 else (entry_price - cur_px) / entry_price

            reason = None
            if is_last_bar_of_session or eod_flat_signal[i]:
                reason = "EOD"
            elif ret <= -stop_loss_pct:
                reason = "SL"
            elif ret >= profit_target_pct:
                reason = "PT"
            elif (pos == 1 and l_ext[i]) or (pos == -1 and s_ext[i]):
                reason = "Signal"
            elif bars_held >= max_hold_bars:
                reason = "Time"

            if reason is not None:
                net_pnl = (pos_dollars * ret) - (2.0 * commission_per_order)
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

        if pos == 0 and not is_last_bar_of_session and equity > 0:
            if l_ent[i]:
                pos = 1
                entry_price = opens[i + 1]
                entry_idx = i + 1
                pos_dollars = min(base_capital, max(equity, 0.0))
            elif s_ent[i]:
                pos = -1
                entry_price = opens[i + 1]
                entry_idx = i + 1
                pos_dollars = min(base_capital, max(equity, 0.0))

    daily_pnl_series = pd.Series(daily_pnl_map).sort_index()
    eq_series = base_capital + daily_pnl_series.cumsum()

    full_eq = pd.concat([pd.Series([base_capital]), eq_series])
    eod_peaks = full_eq.cummax()
    eod_dd = float((((full_eq - eod_peaks) / eod_peaks) * 100.0).min())
    max_dd_pct = min(max_dd_pct, eod_dd)

    cal_years = max((daily_pnl_series.index[-1] - daily_pnl_series.index[0]).days / 365.25, 1.0 / 12.0) if len(daily_pnl_series) >= 2 else 1.0
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
