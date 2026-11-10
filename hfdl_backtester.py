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

    def test_auth(self, ticker: str = "AAPL") -> Tuple
