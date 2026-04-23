#!/usr/bin/env python3
"""
02_collect_minute_bars.py
=========================

Collect 5-minute OHLCV bars (regular + extended hours) for the analysis tickers
via yfinance, for the most recent 60 days.

Tickers
  DJT  Trump Media & Technology Group     (retail-dominated, small float)
  VXX  VIX short-term futures ETF         (volatility proxy)
  SPY  S&P 500 ETF                        (broad equity benchmark)
  QQQ  NASDAQ 100 ETF                     (tech benchmark)
  XLE  Energy Select Sector SPDR          (energy headline finding)
  USO  US Oil Fund                        (direct crude exposure)
  GLD  Gold SPDR                          (flight-to-safety proxy)
  UUP  US Dollar Bullish ETF              (dollar reaction)
  XLF  Financials Select Sector SPDR      (rates / banks)
  XLK  Technology Select Sector SPDR      (tech reaction)

Output:
  data/raw/minute_bars_5m/<TICKER>.parquet      (UTC index, OHLCV + RTH flags)
  data/raw/minute_bars_5m/manifest.json         (collection summary)

NB. yfinance only ships ~60 days of 5-minute history; re-running this on a
date past the original study window will yield a different set of bars and
breaks bit-for-bit reproducibility. The cached parquets shipped with this
repo are the ones the paper uses.
"""

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from _paths import MINUTE_BARS_5M, ensure_dirs

TICKERS = ["DJT", "VXX", "SPY", "QQQ", "XLE", "USO", "GLD", "UUP", "XLF", "XLK"]
INTERVAL = "5m"
PERIOD = "60d"
PREPOST = True  # include extended hours


def flatten_cols(df, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df


def mark_rth(df):
    et = df.index.tz_convert("America/New_York")
    h = et.hour + et.minute / 60.0
    df["is_rth"] = (et.weekday < 5) & (h >= 9.5) & (h < 16.0)
    df["is_weekday"] = et.weekday < 5
    df["is_extended"] = (et.weekday < 5) & (((h >= 4.0) & (h < 9.5)) | ((h >= 16.0) & (h < 20.0)))
    return df


def main():
    ensure_dirs()
    manifest = {"collected_at": datetime.now(timezone.utc).isoformat(),
                "interval": INTERVAL, "period": PERIOD, "prepost": PREPOST,
                "tickers": {}}
    print(f"=== minute bars  interval={INTERVAL}  period={PERIOD}  prepost={PREPOST} ===")
    for tkr in TICKERS:
        try:
            df = yf.download(tkr, period=PERIOD, interval=INTERVAL,
                             progress=False, auto_adjust=False,
                             threads=False, prepost=PREPOST)
            if df is None or len(df) == 0:
                print(f"  {tkr}: EMPTY")
                continue
            df = flatten_cols(df, tkr)[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")].dropna(subset=["Close"])
            df = mark_rth(df)
            close = df["Close"].astype(float).values
            vol = df["Volume"].astype(float).values
            logret = np.full(len(close), np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                logret[1:] = np.log(close[1:] / close[:-1])
            df["ret"] = np.concatenate([[np.nan], close[1:] / close[:-1] - 1])
            df["logret"] = logret
            df["dollar_volume"] = close * vol

            path = MINUTE_BARS_5M / f"{tkr}.parquet"
            df.to_parquet(path)
            n, rth = len(df), int(df["is_rth"].sum())
            print(f"  {tkr:5s}: {n:6d} bars  ({rth} RTH)  "
                  f"{df.index.min()} -> {df.index.max()}")
            manifest["tickers"][tkr] = {
                "rows": n, "rth_rows": rth,
                "start": str(df.index.min()), "end": str(df.index.max()),
                "path": str(path),
            }
            time.sleep(0.25)
        except Exception as e:
            print(f"  {tkr}: ERROR  {e}")
            manifest["tickers"][tkr] = {"error": str(e)}

    with (MINUTE_BARS_5M / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nmanifest -> {MINUTE_BARS_5M/'manifest.json'}")


if __name__ == "__main__":
    main()
