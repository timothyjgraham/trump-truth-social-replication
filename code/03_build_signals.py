#!/usr/bin/env python3
"""
03_build_signals.py
===================

Build per-bar order-flow signals from the cached 5-minute OHLCV bars.

Signals (indexed against `data/raw/minute_bars_5m/<TICKER>.parquet`):

  1. Bulk Volume Classification (BVC) — Easley, López de Prado, O'Hara (2012):
        V_B_t = V_t * Phi(dP_t / sigma_t)
        V_S_t = V_t - V_B_t
     where sigma_t is a 250-bar rolling std of close-to-close price changes,
     SHIFTED by 1 bar so the baseline never absorbs the bar being tested.

  2. Tick-rule signed volume:
        signed_vol_tick_t = sign(Close_t - Close_{t-1}) * V_t

  3. Order Flow Imbalance, BVC variant:
        OFI_bvc_t = (V_B_t - V_S_t) / V_t      ∈ [-1, +1]

  4. VPIN (Easley et al., 2012), 50-bar window in bar-time:
        VPIN_t = sum_{t-49..t} |V_B - V_S| / sum_{t-49..t} V

  5. vpin_z: VPIN z-scored against a 250-bar shifted baseline.

  6. Abnormal-volume z-score (vol_z) and dollar-volume z-score (dvol_z),
     same 250-bar shifted baseline.

  7. Kyle's lambda (rolling 100-bar OLS slope of logret on signed sqrt
     dollar volume) and its z-score.

All rolling baselines use shift(1) — strict pre-event, no look-ahead.

Output:
  data/interim/signals_5m/<TICKER>.parquet
  data/interim/signals_5m/manifest.json
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from _paths import MINUTE_BARS_5M, SIGNALS_5M, ensure_dirs

BASELINE_N = 250          # bars used for abnormal-volume / sigma baselines
VPIN_N = 50               # VPIN window
KYLE_WIN = 100            # Kyle's lambda regression window


def bvc(dP, V, sigma):
    frac = np.full_like(V, 0.5, dtype=float)
    mask = np.isfinite(sigma) & (sigma > 0) & np.isfinite(dP)
    frac[mask] = norm.cdf(dP[mask] / sigma[mask])
    V_B = V * frac
    return V_B, V - V_B


def rolling_shifted_std(x, n):
    return x.shift(1).rolling(n, min_periods=max(20, n // 5)).std()


def rolling_shifted_mean(x, n):
    return x.shift(1).rolling(n, min_periods=max(20, n // 5)).mean()


def kyle_lambda_rolling(logret, signed_sqrt_dv, win):
    n = len(logret)
    out = np.full(n, np.nan)
    if n < win + 2:
        return out
    x, y = signed_sqrt_dv, logret
    cumx = np.concatenate([[0], np.cumsum(np.nan_to_num(x))])
    cumy = np.concatenate([[0], np.cumsum(np.nan_to_num(y))])
    cumxx = np.concatenate([[0], np.cumsum(np.nan_to_num(x * x))])
    cumxy = np.concatenate([[0], np.cumsum(np.nan_to_num(x * y))])
    valid = (~np.isnan(x)) & (~np.isnan(y))
    cumvalid = np.concatenate([[0], np.cumsum(valid.astype(int))])
    for t in range(win + 1, n):
        lo, hi = t - win, t
        m = cumvalid[hi] - cumvalid[lo]
        if m < win // 2:
            continue
        sx = cumx[hi] - cumx[lo]
        sy = cumy[hi] - cumy[lo]
        sxx = cumxx[hi] - cumxx[lo]
        sxy = cumxy[hi] - cumxy[lo]
        denom = sxx - (sx * sx) / m
        if denom <= 0:
            continue
        out[t] = (sxy - (sx * sy) / m) / denom
    return out


def enrich(df):
    df = df.copy()
    close = df["Close"].astype(float).values
    vol = df["Volume"].astype(float).values
    dv = df["dollar_volume"].astype(float).values
    logret = df["logret"].astype(float).values
    dP = np.concatenate([[np.nan], np.diff(close)])
    df["dP"] = dP
    sigma_dP = rolling_shifted_std(pd.Series(dP, index=df.index), BASELINE_N).values
    df["sigma_dP_250"] = sigma_dP

    V_B, V_S = bvc(dP, vol, sigma_dP)
    df["V_buy_bvc"] = V_B
    df["V_sell_bvc"] = V_S
    df["OFI_bvc"] = np.where(vol > 0, (V_B - V_S) / vol, 0.0)

    sign_t = np.sign(np.concatenate([[0], np.diff(close)]))
    df["signed_vol_tick"] = sign_t * vol
    df["signed_dv_tick"] = sign_t * dv

    v_s = pd.Series(vol, index=df.index)
    dv_s = pd.Series(dv, index=df.index)
    mu_V = rolling_shifted_mean(v_s, BASELINE_N).values
    sd_V = rolling_shifted_std(v_s, BASELINE_N).values
    mu_DV = rolling_shifted_mean(dv_s, BASELINE_N).values
    sd_DV = rolling_shifted_std(dv_s, BASELINE_N).values
    df["vol_z"] = np.where(sd_V > 0, (vol - mu_V) / sd_V, np.nan)
    df["dvol_z"] = np.where(sd_DV > 0, (dv - mu_DV) / sd_DV, np.nan)

    V_abs = np.abs(V_B - V_S)
    V_abs_sum = pd.Series(V_abs, index=df.index).rolling(VPIN_N, min_periods=VPIN_N // 2).sum().values
    V_sum = pd.Series(vol, index=df.index).rolling(VPIN_N, min_periods=VPIN_N // 2).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        df["vpin_50"] = np.where(V_sum > 0, V_abs_sum / V_sum, np.nan)

    vpin_s = pd.Series(df["vpin_50"].values, index=df.index)
    mu_vp = rolling_shifted_mean(vpin_s, BASELINE_N).values
    sd_vp = rolling_shifted_std(vpin_s, BASELINE_N).values
    df["vpin_z"] = np.where(sd_vp > 0, (df["vpin_50"].values - mu_vp) / sd_vp, np.nan)

    s_sqrt_dv = sign_t * np.sqrt(np.abs(dv))
    lam = kyle_lambda_rolling(logret, s_sqrt_dv, KYLE_WIN)
    df["kyle_lambda_100"] = lam
    lam_s = pd.Series(lam, index=df.index)
    mu_lam = rolling_shifted_mean(lam_s, BASELINE_N).values
    sd_lam = rolling_shifted_std(lam_s, BASELINE_N).values
    with np.errstate(divide="ignore", invalid="ignore"):
        df["kyle_z"] = np.where(sd_lam > 0, (lam - mu_lam) / sd_lam, np.nan)
    return df


def main():
    ensure_dirs()
    manifest = {"tickers": {}}
    print("=== building signals (5m) ===")
    for p in sorted(MINUTE_BARS_5M.glob("*.parquet")):
        tkr = p.stem
        df = pd.read_parquet(p)
        out = enrich(df)
        path = SIGNALS_5M / f"{tkr}.parquet"
        out.to_parquet(path)
        any_ok = int(out["vol_z"].notna().sum())
        vp_ok = int(out["vpin_50"].notna().sum())
        ky_ok = int(out["kyle_lambda_100"].notna().sum())
        print(f"  {tkr:5s}: {len(out):6d} bars; vol_z={any_ok}, vpin={vp_ok}, kyle={ky_ok}")
        manifest["tickers"][tkr] = {
            "rows": len(out), "vol_z_non_null": any_ok,
            "vpin_non_null": vp_ok, "kyle_non_null": ky_ok,
            "path": str(path),
        }
    with (SIGNALS_5M / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nmanifest -> {SIGNALS_5M/'manifest.json'}")


if __name__ == "__main__":
    main()
