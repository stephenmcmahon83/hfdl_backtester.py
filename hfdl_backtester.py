import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
import requests
import numpy as np
import pandas as pd


class HFDataClient:
    """
    Client for https://api.hfdatalibrary.com/v1
    Caches whole-ticker Parquet files locally and validates freshness via /symbols/{ticker}.
    """
    BASE_URL = "https://api.hfdatalibrary.com/v1"

    def __init__(self, api_key: str, cache_dir: str = "./hf_cache"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def _request(
        self,
        endpoint: str,
        params: Optional[ dict ] = None,
        auth: bool = True,
        stream: bool = False,
        full_url: Optional[str] = None,
    ) -> requests.Response:
        url = full_url or f"{self.BASE_URL}{endpoint}"
        headers = {"X-API-Key": self.api_key} if (auth and not full_url) else {}

        for attempt in range(3):
            resp = self.session.get(url, headers=headers, params=params, stream=stream, timeout=60)

            # Verify response came from the genuine API edge (per documentation)
            if not resp.content and "Content-Type" not in resp.headers:
                raise RuntimeError(
                    "Empty response with no Content-Type. Check for a VPN, proxy, or TLS appliance blocking the request."
                )

            # Respect rate limits automatically
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 15))
                print(f"[Rate Limit 429] Waiting {retry_after}s before retrying...")
                time.sleep(retry_after)
                continue

            if resp.status_code != 200:
                try:
                    err_detail = resp.json()
                except Exception:
                    err_detail = resp.text
                raise RuntimeError(f"HTTP {resp.status_code} on {url}: {err_detail}")

            return resp

        raise RuntimeError(f"Max retries exceeded for {url}")

    def list_symbols(self) -> pd.DataFrame:
        """Fetch all available symbols and metadata (public endpoint, no API key required)."""
        resp = self._request("/symbols", auth=False)
        data = resp.json()
        return pd.DataFrame(data.get("symbols", []))

    def get_symbol_meta(self, ticker: str) -> Dict:
        """Fetch size_bytes and last_modified per version for a single ticker (public)."""
        resp = self._request(f"/symbols/{ticker.upper()}", auth=False)
        return resp.json()

    def _is_cache_fresh(self, file_path: Path, meta_path: Path, remote_meta: Dict) -> bool:
        if not file_path.exists() or not meta_path.exists():
            return False
        try:
            local_meta = json.loads(meta_path.read_text())
            return (
                local_meta.get("last_modified") == remote_meta.get("last_modified")
                and file_path.stat().st_size == remote_meta.get("size_bytes")
            )
        except Exception:
            return False

    def get_1min_bars(self, ticker: str, version: str = "clean", force_refresh: bool = False) -> pd.DataFrame:
        """Download and cache complete 1-minute OHLCV history via /bars/{ticker}."""
        ticker = ticker.upper()
        file_path = self.cache_dir / f"{ticker}_bars_1min_{version}.parquet"
        meta_path = self.cache_dir / f"{ticker}_bars_1min_{version}.meta.json"

        remote_info = self.get_symbol_meta(ticker)["versions"].get(version, {})

        if not force_refresh and self._is_cache_fresh(file_path, meta_path, remote_info):
            return pd.read_parquet(file_path)

        print(f"Downloading 1-min Parquet for {ticker} ({version})...")
        resp = self._request(f"/bars/{ticker}", params={"version": version}, auth=True, stream=True)
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

        meta_path.write_text(json.dumps(remote_info))
        return pd.read_parquet(file_path)

    def get_resampled_bars(
        self,
        ticker: str,
        timeframe: str = "5min",
        version: str = "clean",
        fmt: str = "parquet",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Two-step signed URL flow (/download-token/{ticker} -> /download/{ticker}?token=...)
        Supports: 1min, 5min, 15min, 30min, hourly, daily, weekly, monthly.
        """
        if timeframe == "1min" and fmt == "parquet":
            return self.get_1min_bars(ticker, version=version, force_refresh=force_refresh)

        ticker = ticker.upper()
        ext = "parquet" if fmt == "parquet" else "csv"
        file_path = self.cache_dir / f"{ticker}_bars_{timeframe}_{version}.{ext}"

        if file_path.exists() and not force_refresh:
            return pd.read_parquet(file_path) if fmt == "parquet" else pd.read_csv(file_path)

        # Step 1: Request signed URL with API key
        token_resp = self._request(
            f"/download-token/{ticker}",
            params={"timeframe": timeframe, "format": fmt, "version": version},
            auth=True,
        ).json()

        signed_url = token_resp["url"]
        if signed_url.startswith("/"):
            signed_url = f"https://api.hfdatalibrary.com{signed_url}"

        # Step 2: Fetch file within 10 minutes (no API key header required)
        print(f"Downloading {timeframe} {fmt} for {ticker} via signed URL...")
        dl_resp = self._request("", auth=False, stream=True, full_url=signed_url)
        with open(file_path, "wb") as f:
            for chunk in dl_resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

        return pd.read_parquet(file_path) if fmt == "parquet" else pd.read_csv(file_path)

    def get_variables(self, ticker: str, version: str = "clean") -> pd.DataFrame:
        """Download daily academic variables via /variables/{ticker}."""
        ticker = ticker.upper()
        file_path = self.cache_dir / f"{ticker}_variables_{version}.parquet"
        if not file_path.exists():
            resp = self._request(f"/variables/{ticker}", params={"version": version}, auth=True, stream=True)
            with open(file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        return pd.read_parquet(file_path)

    def get_quality(self, ticker: str, version: str = "clean") -> pd.DataFrame:
        """Download daily data quality metrics via /quality/{ticker}."""
        ticker = ticker.upper()
        file_path = self.cache_dir / f"{ticker}_quality_{version}.parquet"
        if not file_path.exists():
            resp = self._request(f"/quality/{ticker}", params={"version": version}, auth=True, stream=True)
            with open(file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        return pd.read_parquet(file_path)


class IntradayBacktester:
    """
    Vectorized intraday backtesting engine with session-anchored indicators,
    data-quality filtering, realistic slippage/commission, and mandatory EOD flattening.
    """

    def __init__(
        self,
        df_bars: pd.DataFrame,
        df_quality: Optional[pd.DataFrame] = None,
        df_variables: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cost_bps: float = 2.0,  # 2 bps (0.02%) per side for spread + slippage
    ):
        self.cost_rate = cost_bps / 10_000.0
        self.df = self._prepare_data(df_bars, df_quality, df_variables, start_date, end_date)

    @staticmethod
    def _prepare_data(
        df: pd.DataFrame,
        df_quality: Optional[pd.DataFrame],
        df_variables: Optional[pd.DataFrame],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pd.DataFrame:
        df = df.copy()

        # Normalize datetime index and column names
        if not isinstance(df.index, pd.DatetimeIndex):
            time_col = next((c for c in df.columns if c.lower() in ("timestamp", "datetime", "date", "time")), None)
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.set_index(time_col)
            else:
                df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        df.columns = [c.lower() for c in df.columns]

        # Ensure US Eastern Time for accurate session boundaries
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)

        # Slice date range locally (as recommended by API docs)
        if start_date:
            df = df.loc[start_date:]
        if end_date:
            df = df.loc[:end_date]

        # Filter to Regular Trading Hours (09:30 to 16:00 ET)
        df = df.between_time("09:30", "16:00").copy()
        df["session_date"] = df.index.normalize()

        # Filter out low-quality days if /quality/{ticker} is provided
        if df_quality is not None and not df_quality.empty:
            q = df_quality.copy()
            q.columns = [c.lower() for c in q.columns]
            if not isinstance(q.index, pd.DatetimeIndex):
                q.index = pd.to_datetime(q.index)
            if q.index.tz is not None:
                q.index = q.index.tz_convert("America/New_York").tz_localize(None)
            q["session_date"] = q.index.normalize()

            if "gap_rate" in q.columns:
                valid_days = q.loc[q["gap_rate"] <= 0.05, "session_date"]
                df = df[df["session_date"].isin(valid_days)]

        # Merge lagged daily /variables/{ticker} (shifted 1 day to avoid look-ahead bias)
        if df_variables is not None and not df_variables.empty:
            v = df_variables.copy()
            v.columns = [f"var_{c.lower()}" for c in v.columns]
            if not isinstance(v.index, pd.DatetimeIndex):
                v.index = pd.to_datetime(v.index)
            if v.index.tz is not None:
                v.index = v.index.tz_convert("America/New_York").tz_localize(None)
            v = v.sort_index().shift(1)  # Crucial: only use yesterday's known values
            v["session_date"] = v.index.normalize()
            df = df.reset_index().merge(v, on="session_date", how="left").set_index("index")
            df.index.name = "timestamp"

        return df

    def run_orb_vwap_strategy(
        self,
        orb_minutes: int = 15,
        vol_mult: float = 1.25,
        eod_exit_time: str = "15:55",
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Intraday Opening Range Breakout (ORB) + Session-Anchored VWAP Strategy:
        1. Defines the Opening Range High/Low during the first `orb_minutes` (e.g., 09:30-09:45).
        2. Computes session-anchored VWAP and standard deviation bands.
        3. Enters Long when price closes above ORB High, stays above Session VWAP, and volume exceeds
           `vol_mult` * 20-bar average volume. Enters Short on the symmetric breakdown.
        4. Exits if price crosses back through Session VWAP or at `eod_exit_time` (15:55 ET) so zero
           inventory is held overnight.
        """
        df = self.df.copy()
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        pv = tp * df["volume"]

        # Session-Anchored VWAP (resets at 09:30 every day)
        cum_pv = pv.groupby(df["session_date"]).cumsum()
        cum_vol = df["volume"].groupby(df["session_date"]).cumsum().replace(0, np.nan)
        df["vwap"] = cum_pv / cum_vol

        # Opening Range High & Low per trading session
        orb_end_time = (pd.Timestamp("2000-01-01 09:30:00") + pd.Timedelta(minutes=orb_minutes)).strftime("%H:%M")
        orb_slice = df.between_time("09:30", orb_end_time)
        orb_high = orb_slice.groupby("session_date")["high"].max().rename("orb_high")
        orb_low = orb_slice.groupby("session_date")["low"].min().rename("orb_low")

        df = df.join(orb_high, on="session_date").join(orb_low, on="session_date")

        # Relative volume confirmation
        df["vol_ma20"] = df.groupby("session_date")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())

        # Time masks
        bar_times = df.index.strftime("%H:%M")
        in_trading_window = (bar_times > orb_end_time) & (bar_times < eod_exit_time)
        is_eod_flatten = bar_times >= eod_exit_time

        # Entry & Exit Logic (state machine per session to hold until stop/EOD)
        long_entry = in_trading_window & (df["close"] > df["orb_high"]) & (df["close"] > df["vwap"]) & (df["volume"] > vol_mult * df["vol_ma20"])
        short_entry = in_trading_window & (df["close"] < df["orb_low"]) & (df["close"] < df["vwap"]) & (df["volume"] > vol_mult * df["vol_ma20"])

        long_exit = (df["close"] < df["vwap"]) | is_eod_flatten
        short_exit = (df["close"] > df["vwap"]) | is_eod_flatten

        positions = np.zeros(len(df), dtype=np.float64)
        closes = df["close"].to_numpy()
        vwaps = df["vwap"].to_numpy()
        l_ent = long_entry.to_numpy()
        s_ent = short_entry.to_numpy()
        l_ext = long_exit.to_numpy()
        s_ext = short_exit.to_numpy()

        curr_pos = 0.0
        for i in range(len(df)):
            if curr_pos == 1.0 and l_ext[i]:
                curr_pos = 0.0
            elif curr_pos == -1.0 and s_ext[i]:
                curr_pos = 0.0

            if curr_pos == 0.0:
                if l_ent[i]:
                    curr_pos = 1.0
                elif s_ent[i]:
                    curr_pos = -1.0
            positions[i] = curr_pos

        df["signal_pos"] = positions
        # Shift position by 1 bar: signal on bar t close executes on bar t+1
        df["position"] = df.groupby("session_date")["signal_pos"].shift(1).fillna(0.0)

        # Bar returns (zeroed across overnight boundary since position is flat at EOD)
        df["bar_ret"] = df.groupby("session_date")["close"].pct_change().fillna(0.0)
        df["turnover"] = df.groupby("session_date")["position"].diff().abs().fillna(df["position"].abs())
        df["cost"] = df["turnover"] * self.cost_rate

        df["gross_ret"] = df["position"] * df["bar_ret"]
        df["net_ret"] = df["gross_ret"] - df["cost"]

        metrics = self.compute_metrics(df)
        return df, metrics

    @staticmethod
    def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
        # Aggregate to daily net returns for clean annualized Sharpe/Sortino
        daily_ret = df.groupby("session_date")["net_ret"].apply(lambda r: (1.0 + r).prod() - 1.0)
        daily_gross = df.groupby("session_date")["gross_ret"].apply(lambda r: (1.0 + r).prod() - 1.0)

        equity = (1.0 + daily_ret).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min()) if len(drawdown) else 0.0

        ann_factor = 252.0
        mean_d = daily_ret.mean()
        std_d = daily_ret.std()
        downside_std = daily_ret[daily_ret < 0].std()

        sharpe = float((mean_d / std_d) * np.sqrt(ann_factor)) if std_d > 0 else 0.0
        sortino = float((mean_d / downside_std) * np.sqrt(ann_factor)) if downside_std > 0 else 0.0
        total_ret = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
        n_years = max(len(daily_ret) / ann_factor, 1 / ann_factor)
        cagr = float((equity.iloc[-1] ** (1.0 / n_years)) - 1.0) if len(equity) and equity.iloc[-1] > 0 else 0.0
        calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

        active_days = daily_ret[daily_ret != 0]
        win_rate = float((active_days > 0).mean()) if len(active_days) else 0.0
        gains = active_days[active_days > 0].sum()
        losses = abs(active_days[active_days < 0].sum())
        profit_factor = float(gains / losses) if losses > 0 else np.inf

        return {
            "Cumulative Net Return (%)": round(total_ret * 100, 2),
            "Cumulative Gross Return (%)": round(float(((1.0 + daily_gross).prod() - 1.0) * 100), 2),
            "CAGR (%)": round(cagr * 100, 2),
            "Annualized Sharpe Ratio": round(sharpe, 2),
            "Annualized Sortino Ratio": round(sortino, 2),
            "Calmar Ratio": round(calmar, 2),
            "Max Drawdown (%)": round(max_dd * 100, 2),
            "Active Day Win Rate (%)": round(win_rate * 100, 2),
            "Daily Profit Factor": round(profit_factor, 2),
            "Trading Days Analyzed": int(len(daily_ret)),
        }


if __name__ == "__main__":
    API_KEY = os.environ.get("HFDL_API_KEY", "your-key-here")
    client = HFDataClient(api_key=API_KEY)

    # 1. Inspect available symbols (No API key needed)
    symbols_df = client.list_symbols()
    print(f"Available tickers on HF Data Library: {len(symbols_df)}")

    # 2. Download/load cached data for AAPL
    ticker = "AAPL"
    # Use 5-minute bars via the 2-step signed URL flow to filter 1-min microstructure noise,
    # or call client.get_1min_bars(ticker) for raw 1-minute resolution:
    bars_df = client.get_1min_bars(ticker, version="clean")
    quality_df = client.get_quality(ticker, version="clean")
    variables_df = client.get_variables(ticker, version="clean")

    # 3. Run backtest on 2024–2026 slice with 2 bps transaction cost per side
    bt = IntradayBacktester(
        df_bars=bars_df,
        df_quality=quality_df,
        df_variables=variables_df,
        start_date="2024-01-02",
        end_date="2026-10-10",
        cost_bps=2.0,
    )

    results_df, summary = bt.run_orb_vwap_strategy(orb_minutes=15, vol_mult=1.25)
    print(f"\n--- {ticker} Intraday Backtest Summary ---")
    for k, v in summary.items():
        print(f"{k:30s}: {v}")
