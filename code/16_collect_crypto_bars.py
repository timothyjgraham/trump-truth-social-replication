#!/usr/bin/env python3
"""
16_collect_crypto_bars.py
=========================

Collect 5-minute bars for BTC-USD and ETH-USD across the same 73-day window
the equity pipeline uses, via the Coinbase Exchange public candles API.

Why Coinbase, not yfinance:
  yfinance only ships 5-minute crypto data for the most recent 60 days,
  which leaves the first ~27 days of the Jan-26 → Apr-9 window un-served.
  Coinbase's public REST endpoint
    https://api.exchange.coinbase.com/products/<id>/candles
  returns up to 300 candles per call, free, no auth, well-behaved on
  rate limits — and goes back years.

We fetch the full window, paginating in 24-hour chunks (288 5-minute
candles per chunk). The output schema matches the equity 5-minute parquets
(`Open, High, Low, Close, Volume, ret, logret, dollar_volume, is_rth=False`)
so script 03's `enrich` function can be called on it without modification.

Then we run that enrich step in-process to populate
data/interim/signals_5m/{BTC,ETH}-USD.parquet.

Outputs:
  data/raw/minute_bars_5m/BTC-USD.parquet
  data/raw/minute_bars_5m/ETH-USD.parquet
  data/interim/signals_5m/BTC-USD.parquet
  data/interim/signals_5m/ETH-USD.parquet
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _paths import MINUTE_BARS_5M, SIGNALS_5M, ensure_dirs

# Re-use the canonical signal-builder
import importlib.util
_path = __import__("pathlib").Path(__file__).resolve().parent / "03_build_signals.py"
spec = importlib.util.spec_from_file_location("_sig", _path)
_sig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_sig)
enrich = _sig.enrich


COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
GRANULARITY = 300   # seconds per candle = 5 min
CHUNK_HOURS = 24    # candles per request: 24h / 5m = 288 (max 300)

# Match the equity pipeline window exactly.
START_UTC = datetime(2026, 1, 26, tzinfo=timezone.utc)
END_UTC   = datetime(2026, 4, 22, tzinfo=timezone.utc)

PRODUCTS = ["BTC-USD", "ETH-USD"]


def fetch_chunk(product: str, start: datetime, end: datetime,
                retries: int = 3) -> list[list[float]]:
    params = dict(granularity=GRANULARITY,
                  start=start.isoformat(),
                  end=end.isoformat())
    for attempt in range(retries):
        try:
            r = requests.get(COINBASE_URL.format(product=product),
                             params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return []


def collect_one(product: str) -> pd.DataFrame:
    """Paginate Coinbase 5-min candles across the study window."""
    rows: list[list[float]] = []
    cursor = START_UTC
    while cursor < END_UTC:
        chunk_end = min(cursor + timedelta(hours=CHUNK_HOURS), END_UTC)
        data = fetch_chunk(product, cursor, chunk_end)
        if data:
            rows.extend(data)
        cursor = chunk_end
        time.sleep(0.15)   # polite pause; Coinbase rate-limit is 10 req/s
    if not rows:
        raise RuntimeError(f"{product}: empty response across full window")

    # Coinbase candle format: [time, low, high, open, close, volume]
    df = pd.DataFrame(rows, columns=["time", "Low", "High", "Open", "Close", "Volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = (df.drop_duplicates(subset="time")
            .sort_values("time")
            .set_index("time"))
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    # Match equity-bar derived columns
    close = df["Close"].values
    df["ret"] = np.concatenate([[np.nan], close[1:] / close[:-1] - 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.full(len(close), np.nan)
        logret[1:] = np.log(close[1:] / close[:-1])
    df["logret"] = logret
    df["dollar_volume"] = df["Close"] * df["Volume"]

    # Crypto trades 24/7; the equity convention `is_rth` is meaningless,
    # but downstream code reads the column, so we set the flags explicitly.
    df["is_rth"] = False
    df["is_weekday"] = df.index.weekday < 5
    df["is_extended"] = False
    return df


def main() -> None:
    ensure_dirs()
    print(f"window: {START_UTC.date()} → {END_UTC.date()}  "
          f"granularity={GRANULARITY}s")

    for prod in PRODUCTS:
        print(f"\n=== {prod} ===")
        df = collect_one(prod)
        out_raw = MINUTE_BARS_5M / f"{prod}.parquet"
        df.to_parquet(out_raw)
        print(f"  raw: {len(df):,} bars  "
              f"{df.index.min()} → {df.index.max()}")
        print(f"  wrote {out_raw}")

        sig = enrich(df)
        out_sig = SIGNALS_5M / f"{prod}.parquet"
        sig.to_parquet(out_sig)
        print(f"  signals: vpin_z non-null = {sig['vpin_z'].notna().sum():,};  "
              f"OFI_bvc non-null = {sig['OFI_bvc'].notna().sum():,}")
        print(f"  wrote {out_sig}")


if __name__ == "__main__":
    main()
