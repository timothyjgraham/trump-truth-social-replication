#!/usr/bin/env python3
"""
collect_minute_bars.py

Collects multi-resolution intraday OHLCV for the order-flow event study.

Sources & resolutions:
  - yfinance 5m bars: last 60 days, RTH + extended hours  (workhorse)
  - yfinance 30m bars: last 60 days, RTH + extended hours (corroboration)
  - yfinance 1m bars:  last 7 days,  RTH only             (spot-check resolution)

Tickers:
  DJT  - Trump Media & Technology Group (retail-dominated, small float)
  VXX  - VIX short-term futures ETF (volatility proxy)
  SPY  - S&P 500 ETF (broad equity benchmark)
  QQQ  - NASDAQ 100 ETF (tech benchmark)
  XLE  - Energy Select Sector SPDR (energy tweet exposure)
  USO  - US Oil Fund (direct crude exposure)
  GLD  - Gold SPDR (flight-to-safety proxy)
  UUP  - US Dollar Bullish ETF (dollar reaction)
  XLF  - Financials Select Sector SPDR (for fed/rates topic)
  XLK  - Technology Select Sector SPDR (tech reaction)

Outputs parquet files per (ticker, resolution) in
  /sessions/sleepy-gifted-ptolemy/work/data/minute_bars/<resolution>/<TICKER>.parquet

The bars are kept in UTC with an 'is_rth' flag (US Eastern 9:30-16:00).
A single manifest.json summarises what was collected.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

WORK = Path("/sessions/sleepy-gifted-ptolemy/work")
OUT = WORK / "data" / "minute_bars"
OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["DJT", "VXX", "SPY", "QQQ", "XLE", "USO", "GLD", "UUP", "XLF", "XLK"]

# (resolution_label, yf_interval, yf_period, prepost)
RESOLUTIONS = [
    ("5m", "5m", "60d", True),
    ("30m", "30m", "60d", True),
    ("1m", "1m", "7d", False),
]


def flatten_cols(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance returns a MultiIndex when threads=True; flatten to simple cols."""
    if isinstance(df.columns, pd.MultiIndex):
        # rows: Open High Low Close Adj Close Volume; level 1 is ticker
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df


def mark_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each bar as Regular Trading Hours (US Eastern 9:30-16:00, Mon-Fri)."""
    et = df.index.tz_convert("America/New_York")
    h = et.hour + et.minute / 60.0
    df["is_rth"] = (et.weekday < 5) & (h >= 9.5) & (h < 16.0)
    df["is_weekday"] = et.weekday < 5
    df["is_extended"] = (et.weekday < 5) & (((h >= 4.0) & (h < 9.5)) | ((h >= 16.0) & (h < 20.0)))
    return df


manifest = {"collected_at": pd.Timestamp.utcnow().isoformat(), "resolutions": {}}

for label, interval, period, prepost in RESOLUTIONS:
    sub = OUT / label
    sub.mkdir(exist_ok=True)
    print(f"\n=== Resolution {label} (interval={interval}, period={period}, prepost={prepost}) ===")
    res_manifest = {}
    for tkr in TICKERS:
        try:
            df = yf.download(
                tkr,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
                prepost=prepost,
            )
            if df is None or len(df) == 0:
                print(f"  {tkr}: EMPTY")
                continue
            df = flatten_cols(df, tkr)
            # Ensure Volume present and integer-ish; coerce types
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]
            df = df.dropna(subset=["Close"])
            df = mark_rth(df)
            # Derived columns — vectorised, no apply()
            import numpy as np
            close = df["Close"].astype(float).values
            vol = df["Volume"].astype(float).values
            logret = np.full(len(close), np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                logret[1:] = np.log(close[1:] / close[:-1])
            df["ret"] = np.concatenate([[np.nan], close[1:] / close[:-1] - 1])
            df["logret"] = logret
            df["dollar_volume"] = close * vol

            path = sub / f"{tkr}.parquet"
            df.to_parquet(path)
            n = len(df)
            rth_n = int(df["is_rth"].sum())
            print(f"  {tkr:5s}: {n:6d} bars  ({rth_n} RTH)  {df.index.min()} -> {df.index.max()}")
            res_manifest[tkr] = {
                "rows": n,
                "rth_rows": rth_n,
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "path": str(path),
            }
            time.sleep(0.25)  # be polite
        except Exception as e:
            print(f"  {tkr}: ERROR {e}")
            res_manifest[tkr] = {"error": str(e)}
    manifest["resolutions"][label] = res_manifest

with (OUT / "manifest.json").open("w") as f:
    json.dump(manifest, f, indent=2, default=str)

print(f"\nManifest written to {OUT / 'manifest.json'}")
