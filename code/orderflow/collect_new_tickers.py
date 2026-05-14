#!/usr/bin/env python3
"""
Pull 5-minute bars for individual-name tickers to enable friends-vs-self analysis.

Tickers: TSLA, NVDA, MSFT, AAPL, META, XOM, CVX, JPM, GS
Rationale (per memo §2.1):
  - TSLA: Musk/Tesla posts
  - NVDA, MSFT, AAPL, META: Big Tech posts
  - XOM, CVX: Big Oil (complements USO/XLE)
  - JPM, GS: Big Banks

Note: yfinance 5m data only reaches ~60 calendar days back from today.
Main overlap with Truth posts is the most recent ~45 trading days.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

WORK = Path("/sessions/sleepy-gifted-ptolemy/work")
OUT = WORK / "data" / "minute_bars"
OUT.mkdir(parents=True, exist_ok=True)

NEW_TICKERS = ["TSLA", "NVDA", "MSFT", "AAPL", "META", "XOM", "CVX", "JPM", "GS"]

RESOLUTIONS = [("5m", "5m", "60d", True), ("30m", "30m", "60d", True)]


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


manifest = {"collected_at": pd.Timestamp.utcnow().isoformat(), "resolutions": {}}
for label, interval, period, prepost in RESOLUTIONS:
    sub = OUT / label
    sub.mkdir(exist_ok=True)
    print(f"\n=== {label} (interval={interval}, period={period}, prepost={prepost}) ===")
    res_manifest = {}
    for tkr in NEW_TICKERS:
        try:
            df = yf.download(tkr, period=period, interval=interval, progress=False,
                             auto_adjust=False, threads=False, prepost=prepost)
            if df is None or len(df) == 0:
                print(f"  {tkr}: EMPTY"); continue
            df = flatten_cols(df, tkr)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]
            df = df.dropna(subset=["Close"])
            df = mark_rth(df)
            close = df["Close"].astype(float).values
            vol = df["Volume"].astype(float).values
            logret = np.full(len(close), np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                logret[1:] = np.log(close[1:] / close[:-1])
            df["ret"] = np.concatenate([[np.nan], close[1:] / close[:-1] - 1])
            df["logret"] = logret
            df["dollar_volume"] = close * vol
            df["dP"] = df["Close"].astype(float).diff()

            path = sub / f"{tkr}.parquet"
            df.to_parquet(path)
            n, rth_n = len(df), int(df["is_rth"].sum())
            print(f"  {tkr:5s}: {n:6d} bars  ({rth_n} RTH)  {df.index.min()} -> {df.index.max()}")
            res_manifest[tkr] = {"rows": n, "rth_rows": rth_n,
                                  "start": str(df.index.min()), "end": str(df.index.max()),
                                  "path": str(path)}
            time.sleep(0.25)
        except Exception as e:
            print(f"  {tkr}: ERROR {e}")
            res_manifest[tkr] = {"error": str(e)}
    manifest["resolutions"][label] = res_manifest

with (OUT / "manifest_new.json").open("w") as f:
    json.dump(manifest, f, indent=2, default=str)
print(f"\nManifest: {OUT/'manifest_new.json'}")
