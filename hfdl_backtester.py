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
        full_idx = pd.DatetimeIndex(np.add.outer(unique_days, intraday_offsets).
