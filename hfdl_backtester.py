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
        self.session.headers.update({"User-Agent": "HFDL-SemiHFT-Tournament/5.5"})

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

    df["session_date"] = df.index.normalize()

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
    if "session_date" not in df.columns:
        df["session_date"] = df.index.normalize()

    g = df.groupby("session_date")

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tp"] = tp
    df["pv"] = tp * df["volume"]

    # Safe calculation of VWAP via DataFrame groupby
    cum_pv = g["pv"].cumsum()
    cum_vol = g["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = (cum_pv / cum_vol).groupby(df["session_date"]).ffill().fillna(df["close"])

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
    df["down_5_streak"] = is_lower.groupby(df["session_date"]).transform(lambda s: s.rolling(5, min_periods=5).sum() == 5)
    df["down_3_streak"] = is_lower.groupby(df["session_date"]).transform(lambda s: s.rolling(3, min_periods=3).sum() == 3)
    df["up_5_streak"] = is_higher.groupby(df["session_date"]).transform(lambda s: s.rolling(5, min_periods=5).sum() == 5)

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
# 4. THE 15 STRATEGIES (INCLUDING SEMI-HFT QQQ & VXX MODELS)
# =====================================================================
def run_all_15_strategies(
    df_ind: pd.DataFrame,
    base_capital: float = 20000.0,
    commission: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    strategies_spec = [
        {
            "name": "1. QQQ Semi-HFT Multi-Signal Engine (Long + Short)",
            "rule": "ENTRY L: VWAP Trend + 3-bar dip OR OR15 Breakout + micro-rest OR 4-bar down exhaustion. ENTRY S: OR30 Breakdown + 2-bar pop rejection. Exit: PT +0.75%, SL -0.80%, 120m max hold.",
            "l_ent": (
                ((df_ind["close"] > df_ind["vwap"]) & (df_ind["ret_from_open"] >= 0.0015) & ((df_ind["down_3_streak"]) | ((df_ind["ret_3m"] <= -0.0010) & (df_ind["ibs"] <= 0.15)))) |
                ((df_ind["close"] > df_ind["or15_high"]) & (df_ind["close"] > df_ind["vwap"]) & (df_ind["ret_15m"] >= 0.0025) & (df_ind["ibs"] <= 0.25)) |
                ((df_ind["down_5_streak"]) & (df_ind["ibs"] <= 0.05) & (df_ind["ret_from_open"] >= -0.0040))
            ).to_numpy(),
            "l_ext": ((df_ind["close"] < df_ind["vwap"] * 0.9970) & (df_ind["ret_15m"] <= -0.0035)).to_numpy(),
            "s_ent": ((df_ind["close"] < df_ind["vwap"]) & (df_ind["close"] < df_ind["or30_low"]) & (df_ind["or30_ret"] <= -0.0015) & (df_ind["u_streak"] >= 2) & (df_ind["ibs"] >= 0.80)).to_numpy(),
            "s_ext": ((df_ind["close"] > df_ind["vwap"] * 1.0030) & (df_ind["ret_15m"] >= 0.0035)).to_numpy(),
            "max_hold": 120, "sl": 0.0080, "pt": 0.0075,
        },
        {
            "name": "2. QQQ Active 45m Scalper (Long + Short)",
            "rule": "ENTRY L: VWAP Trend + 0.2% open ret + 2-min dip OR OR15 breakout. ENTRY S: VWAP Breakdown + OR30 low break. PT +0.45%, SL -0.55%, 45m hold.",
            "l_ent": (
                ((df_ind["close"] > df_ind["vwap"]) & (df_ind["ret_from_open"] >= 0.0020) & (df_ind["ret_15m"] >= 0.0015) & (df_ind["ret_2m"] <= -0.0004) & (df_ind["ibs"] <= 0.25)) |
                ((df_ind["close"] > df_ind["vwap"]) & (df_ind["close"] > df_ind["or15_high"]) & (df_ind["ret_15m"] >= 0.0040))
            ).to_numpy(),
            "l_ext": ((df_ind["close"] < df_ind["vwap"] * 0.9970)).to_numpy(),
            "s_ent": ((df_ind["close"] < df_ind["vwap"]) & (df_ind["close"] < df_ind["or15_low"]) & (df_ind["ret_from_open"] <= -0.0040) & (df_ind["ibs"] >= 0.75)).to_numpy(),
            "s_ext": ((df_ind["close"] > df_ind["vwap"] * 1.0030)).to_numpy(),
            "max_hold": 45, "sl": 0.0055, "pt": 0.0045,
        },
        {
            "name": "3. QQQ Core Sub-Signal 1 Only (VWAP + 3-Bar Dip Long)",
            "rule": "LONG ENTRY: Close > VWAP, Return from open >= +0.15%, and 3 consecutive down closes or 3m return <= -0.10% with IBS <= 0.15. PT +0.75%, SL -0.80%.",
            "l_ent": ((df_ind["close"] > df_ind["vwap"]) & (df_ind["ret_from_open"] >= 0.0015) & ((df_ind["down_3_streak"]) | ((df_ind["ret_3m"] <= -0.0010) & (df_ind["ibs"] <= 0.15)))).to_numpy(),
            "l_ext": ((df_ind["close"] < df_ind["vwap"] * 0.9970)).to_numpy(),
            "max_hold": 120, "sl": 0.0080, "pt": 0.0075,
        },
        {
            "name": "4. VXX Volatility Crush & Cascade Engine",
            "rule": "LONG ENTRY: Volatility breakout ignition. SHORT ENTRY: OR15/VWAP breakdown + bounce fade or Z-score pop >= 1.8. PT +1.50%, SL -1.40%, 90m hold.",
            "l_ent": ((df_ind["close"] > df_ind["vwap"]) & (df_ind["close"] > df_ind["or15_high"]) & (df_ind["ret_from_open"] >= 0.0120) & (df_ind["ret_15m"] >= 0.0080) & (df_ind["ret_2m"] <= -0.0010)).to_numpy(),
            "l_ext": ((df_ind["close"] < df_ind["vwap"] * 0.995)).to_numpy(),
            "s_ent": (
                ((df_ind["close"] < df_ind["vwap"]) & (df_ind["close"] < df_ind["or15_low"]) & (df_ind["ret_from_open"] <= -0.0050) & (df_ind["ret_2m"] >= 0.0010)) |
                ((df_ind["ret_from_open"] <= -0.0020) & (df_ind["z_15"] >= 1.8) & (df_ind["ibs"] >= 0.75))
            ).to_numpy(),
            "s_ext": ((df_ind["close"] > df_ind["vwap"] * 1.005) & (df_ind["ret_15m"] > 0.0060)).to_numpy(),
            "max_hold": 90, "sl": 0.0140, "pt": 0.0150,
        },
        {
            "name": "5. 5 Lower Closes -> Exit Highest Close (10 Bars)",
            "rule": "ENTRY: 5 consecutive lower closes; buy Next Bar's Open. EXIT: Close >= Highest Close of prior 10 bars or 15:55 ET.",
            "l_ent": df_ind["down_5_streak"].to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["highest_close_x"]).to_numpy(),
            "max_hold": 390, "sl": 0.010, "pt": 0.015,
        },
        {
            "name": "6. 3 Lower Closes Above VWAP -> Exit Highest Close",
            "rule": "ENTRY: 3 consecutive lower closes while above VWAP. EXIT: Close >= Highest Close of prior 10 bars or VWAP - 1.5 SD.",
            "l_ent": (df_ind["down_3_streak"] & (df_ind["close"] > df_ind["vwap"])).to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["highest_close_x"]).to_numpy(),
            "max_hold": 390, "sl": 0.010, "pt": 0.015,
        },
        {
            "name": "7. Opening Range Breakout (15m) + Session VWAP Trend",
            "rule": "LONG ENTRY: Close > ORB High, VWAP, and Volume > 1.25x MA. EXIT: Cross back through VWAP.",
            "l_ent": ((df_ind["bar_time"] > "09:45") & (df_ind["close"] > df_ind["or15_high"]) & (df_ind["close"] > df_ind["vwap"])).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["vwap"]).to_numpy(),
            "max_hold": 390, "sl": 0.010, "pt": 0.015,
        },
        {
            "name": "8. Session VWAP 2-SD Mean Reversion Band Bounce",
            "rule": "LONG ENTRY: Close < VWAP - 2.0 SD. SHORT ENTRY: Close > VWAP + 2.0 SD. EXIT: Touch VWAP.",
            "l_ent": (df_ind["close"] < (df_ind["vwap"] - 2.0 * df_ind["std_15"])).to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["vwap"]).to_numpy(),
            "max_hold": 120, "sl": 0.010, "pt": 0.010,
        },
        {
            "name": "9. Connors Intraday RSI(2) < 5 Dip Buyer",
            "rule": "ENTRY: 2-period Intraday RSI drops below 5 while above VWAP. EXIT: RSI > 70 or 15:55 ET.",
            "l_ent": ((df_ind["z_15"] < -2.0) & (df_ind["close"] > df_ind["vwap"])).to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["vwap"]).to_numpy(),
            "max_hold": 120, "sl": 0.008, "pt": 0.010,
        },
        {
            "name": "10. Donchian Channel (10-Bar) Volatility Breakout",
            "rule": "LONG ENTRY: Close breaks above prior 10-bar high on volume expansion. EXIT: Close falls below lowest close.",
            "l_ent": ((df_ind["close"] > df_ind["highest_close_x"]) & (df_ind["volume"] > 1.2 * df_ind["vol_ma20"])).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["highest_close_x"] * 0.99).to_numpy(),
            "max_hold": 120, "sl": 0.008, "pt": 0.012,
        },
        {
            "name": "11. Intraday Internal Bar Strength (IBS < 0.10) Pullback",
            "rule": "ENTRY: Bar closes in bottom 10% of range (IBS < 0.10) while above 15-bar MA. EXIT: Close exceeds prior high.",
            "l_ent": ((df_ind["ibs"] < 0.10) & (df_ind["close"] > df_ind["ma_15"])).to_numpy(),
            "l_ext": (df_ind["close"] > df_ind["ma_15"]).to_numpy(),
            "max_hold": 60, "sl": 0.006, "pt": 0.008,
        },
        {
            "name": "12. Bollinger Band Squeeze & Volume Expansion",
            "rule": "LONG ENTRY: Close breaks Upper BB on Volume > 1.5x MA. EXIT: Close crosses back below MA.",
            "l_ent": ((df_ind["z_15"] > 2.0) & (df_ind["volume"] > 1.5 * df_ind["vol_ma20"])).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["ma_15"]).to_numpy(),
            "max_hold": 90, "sl": 0.008, "pt": 0.012,
        },
        {
            "name": "13. 12-Month Low Capitulation Mean-Reversion Bounce",
            "rule": "ENTRY: Price touches rolling 252-session capitulation low with Z-score < -2.2 -> buy Next Open. EXIT: Revert to MA or 15:55 ET.",
            "l_ent": ((df_ind["z_15"] <= -2.2) & (df_ind["ibs"] <= 0.05)).to_numpy(),
            "l_ext": (df_ind["close"] >= df_ind["ma_15"]).to_numpy(),
            "max_hold": 180, "sl": 0.010, "pt": 0.015,
        },
        {
            "name": "14. Volume Shock Spike (3x Volume MA) Breakout",
            "rule": "ENTRY: Intraday bar volume spikes to > 3.0x 20-bar volume MA and Close > prior High -> buy Next Open. EXIT: Close < MA.",
            "l_ent": ((df_ind["volume"] > 3.0 * df_ind["vol_ma20"]) & (df_ind["close"] > df_ind["vwap"])).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["vwap"]).to_numpy(),
            "max_hold": 60, "sl": 0.006, "pt": 0.010,
        },
        {
            "name": "15. Opening Drive Momentum Continuation",
            "rule": "ENTRY: Opening 15-minute drive return > +1.0% and Close > VWAP at bar 16 -> buy Next Open. EXIT: Close crosses below VWAP.",
            "l_ent": ((df_ind["bar_time"] == "09:45") & (df_ind["ret_from_open"] > 0.01) & (df_ind["close"] > df_ind["vwap"])).to_numpy(),
            "l_ext": (df_ind["close"] < df_ind["vwap"]).to_numpy(),
            "max_hold": 120, "sl": 0.008, "pt": 0.012,
        },
    ]

    rows = []
    eq_curves = {}

    for spec in strategies_spec:
        s_ent = spec.get("s_ent", None)
        s_ext = spec.get("s_ext", None)
        max_b = spec.get("max_hold", 120)
        sl_v = spec.get("sl", 0.008)
        pt_v = spec.get("pt", 0.0075)

        eq, m = simulate_round_trip_trades(
            df_ind,
            long_entry=spec["l_ent"],
            long_exit=spec["l_ext"],
            short_entry=s_ent,
            short_exit=s_ext,
            max_hold_bars=max_b,
            stop_loss_pct=sl_v,
            profit_target_pct=pt_v,
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
    api_key = st.text_input("Paste X-API-Key", value=default_key, type="password")

    st.header("2. Market & Data Validator")
    pub_client = HFDataClient(api_key="")
    symbols_df, sym_err = pub_client.list_symbols()
    tickers = symbols_df["ticker"].tolist() if not symbols_df.empty else ["QQQ", "VXX", "AAPL", "SPY"]
    default_idx = tickers.index("QQQ") if "QQQ" in tickers else 0

    ticker = st.selectbox("Ticker", options=tickers, index=default_idx)
    version = st.selectbox("Data Version", options=["clean", "raw"], index=0)

    min_completeness = st.slider(
        "Minimum Session Completeness (%)", 50.0, 100.0, 95.0, 1.0,
        help="Checks for missing intraday bars and drops broken/half-day sessions below this threshold."
    )
    ffill_quiet = st.checkbox("Forward-Fill Quiet 0-Volume Minutes", value=True)

    st.header("3. Tournament Parameters")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2022-03-07"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2026-10-30"))
    base_capital = st.number_input("Position Size ($ Cap)", 1000.0, 50000.0, 20000.0, 1000.0)
    commission = st.number_input("Commission ($ / order)", 0.0, 5.0, 1.0, 0.25)

    run_btn = st.button("🚀 Run 15-Strategy Tournament", type="primary", use_container_width=True)

if sym_err:
    st.error(f"**Public Endpoint Error:** {sym_err}")

if not api_key or api_key == "your-key-here":
    st.info("👈 Paste your **HF Data Library `X-API-Key`** in the left sidebar and click **Run 15-Strategy Tournament**.")
    st.stop()

if run_btn:
    client = HFDataClient(api_key=api_key)

    with st.status("Running API Verification, Missing-Data Validator & Tournament...", expanded=True) as status:
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
        status.update(label="Running Data Validator & Missing-Bar Audit...", state="running")
        clean_df, audit_info, flagged_df = prepare_and_audit_data(
            bars_df, timeframe="1min", start_date=str(start_date), end_date=str(end_date),
            min_completeness_pct=min_completeness, ffill_quiet_minutes=ffill_quiet, is_vxx=is_vxx
        )

        if clean_df.empty:
            status.update(label="No Valid Trading Sessions Found", state="error")
            st.warning("No valid trading sessions remained after validation. Lower the completeness threshold.")
            st.stop()

        status.update(label="Precomputing indicators & running 15-strategy tournament...", state="running")
        df_ind = precompute_indicators(clean_df)
        tourn_df, eq_curves = run_all_15_strategies(
            df_ind, base_capital=base_capital, commission=commission
        )
        status.update(label="Tournament Complete!", state="complete", expanded=False)

    # Display Data Validator / Missing-Bar Report
    st.subheader(f"🗓️ Data Validator & Missing-Bar Integrity Report — {ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backtest Data Begins (ET)", audit_info["bt_first_ts"])
    c2.metric("Backtest Data Ends (ET)", audit_info["bt_last_ts"])
    c3.metric(
        "Sessions Tested / Total Found",
        f"{audit_info['sessions_tested']:,} / {audit_info['total_sessions_found']:,}",
        delta=f"{audit_info['calendar_years']} Calendar Yrs ({audit_info['dropped_sessions']} flagged days omitted)"
    )
    c4.metric("Total Missing/Quiet Bars Flagged", f"{audit_info['total_missing_bars']:,}")

    st.caption(
        f"**Server Archive Range:** Begins `{audit_info['raw_first_ts']}` and ends `{audit_info['raw_last_ts']}` | "
        f"**Calendar Span Tested:** `{audit_info['calendar_years']} years` | "
        f"**NaN Rows Cleaned:** `{audit_info['nan_rows_removed']}`"
    )

    if not flagged_df.empty:
        with st.expander(f"🔍 Inspect {len(flagged_df)} Flagged Sessions Below {min_completeness:.0f}% Completeness"):
            st.dataframe(flagged_df, use_container_width=True)

    # Display 15-Strategy Tournament Table
    st.subheader("🏆 15-Strategy Tournament Results")
    st.caption(
        "💡 **Hover your mouse over any Strategy name** to view its exact entry and exit rules. "
        "The **#1 Best Strategy is highlighted Green**, and the **#15 Worst Strategy is highlighted Red**. "
        f"All entries execute at **Next Bar's Open** with a **${base_capital:,.0f}** non-compounding cap "
        f"and **${commission:.2f}** order commission (**${commission * 2:.2f}** per round trip)."
    )

    html_table = render_tournament_html_table(tourn_df)
    st.markdown(html_table, unsafe_allow_html=True)

    st.subheader("📊 Strategy Equity Curves ($ Account Value)")
    if eq_curves:
        eq_df = pd.DataFrame(eq_curves)
        st.line_chart(eq_df)
