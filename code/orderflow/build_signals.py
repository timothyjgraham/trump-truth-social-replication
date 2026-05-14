#!/usr/bin/env python3
"""
build_signals.py

Builds per-bar order-flow signals from the minute-bar parquet files
collected by collect_minute_bars.py.

Signals computed:

  1. Bulk Volume Classification (BVC) — Easley, López de Prado, O'Hara (2012).
       V_B_t = V_t * Phi( dP_t / sigma_dP )
       V_S_t = V_t - V_B_t
     where Phi is the standard-normal CDF and sigma_dP is the rolling std of
     close-to-close price changes (pre-event baseline: 250 bars, shifted).

  2. Signed volume via tick rule on bar close (approximation):
       signed_vol_t = sign(Close_t - Close_{t-1}) * V_t

  3. VPIN (volume-synchronised PIN), windowed in bar-time:
       VPIN_t = sum_{t-n+1..t} |V_B - V_S| / sum_{t-n+1..t} V
     computed with n = 50-bar window.

  4. Abnormal-volume z-score, with strict pre-event baseline:
       z_V_t = (V_t - mean_{t-250..t-1}(V)) / std_{t-250..t-1}(V)

  5. Abnormal dollar-volume z-score (same form, $ volume).

  6. Kyle's lambda — rolling regression of log-return on signed sqrt dollar volume:
       r_t = lambda * S_t + eps,   S_t = sign(r_t) * sqrt(|$volume_t|)
     Window = 100 bars, shifted by 1. We store lambda per bar as local price impact.

  7. Order Flow Imbalance (OFI) — running imbalance of BVC buy vs sell volume:
       OFI_t = (V_B_t - V_S_t) / V_t              (normalised to [-1, +1])

Input:  /sessions/sleepy-gifted-ptolemy/work/data/minute_bars/<resolution>/<TICKER>.parquet
Output: /sessions/sleepy-gifted-ptolemy/work/data/signals/<resolution>/<TICKER>.parquet

All rolling statistics use shift(1) to guarantee strictly pre-event baselines
(no look-ahead). This matches the methodological correction flagged in
QA_report.md §1.7 and applied in market_deep_analysis_v2.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

WORK = Path("/sessions/sleepy-gifted-ptolemy/work")
BARS = WORK / "data" / "minute_bars"
OUT = WORK / "data" / "signals"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE_N = 250          # bars used for abnormal-volume / sigma baselines
VPIN_N = 50               # VPIN window (50-bucket convention)
KYLE_WIN = 100            # Kyle's lambda rolling regression window


def bvc(dP: np.ndarray, V: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bulk Volume Classification — standard-normal CDF variant.

    For each bar t:
        buy_frac_t = Phi( dP_t / sigma_t )
        V_B_t = V_t * buy_frac_t
        V_S_t = V_t - V_B_t
    """
    # Guard against zero or nan sigma: fall back to 0.5 (no information)
    frac = np.full_like(V, 0.5, dtype=float)
    mask = np.isfinite(sigma) & (sigma > 0) & np.isfinite(dP)
    frac[mask] = norm.cdf(dP[mask] / sigma[mask])
    V_B = V * frac
    V_S = V - V_B
    return V_B, V_S


def rolling_shifted_std(x: pd.Series, n: int) -> pd.Series:
    return x.shift(1).rolling(n, min_periods=max(20, n // 5)).std()


def rolling_shifted_mean(x: pd.Series, n: int) -> pd.Series:
    return x.shift(1).rolling(n, min_periods=max(20, n // 5)).mean()


def kyle_lambda_rolling(logret: np.ndarray, signed_sqrt_dv: np.ndarray, win: int) -> np.ndarray:
    """Rolling-window Kyle's lambda — OLS slope of logret on signed sqrt dollar volume.

    Uses the moving sums identity  beta = (sum(x*y) - N*mean(x)*mean(y)) /
                                          (sum(x^2) - N*mean(x)^2),
    implemented on a lag-1 window so no look-ahead at the event bar.
    """
    n = len(logret)
    out = np.full(n, np.nan)
    if n < win + 2:
        return out
    x = signed_sqrt_dv
    y = logret
    # Pad-away NaNs: treat both as 0 in cumulative sums? Safer: skip if any NaN in window.
    xx = x * x
    xy = x * y
    cumx = np.concatenate([[0], np.cumsum(np.nan_to_num(x))])
    cumy = np.concatenate([[0], np.cumsum(np.nan_to_num(y))])
    cumxx = np.concatenate([[0], np.cumsum(np.nan_to_num(xx))])
    cumxy = np.concatenate([[0], np.cumsum(np.nan_to_num(xy))])
    valid = (~np.isnan(x)) & (~np.isnan(y))
    cumvalid = np.concatenate([[0], np.cumsum(valid.astype(int))])
    # Evaluate at index t using window [t-win, t-1] (strict lag-1, exclusive of t)
    for t in range(win + 1, n):
        lo = t - win
        hi = t  # exclusive
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


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Types
    close = df["Close"].astype(float).values
    vol = df["Volume"].astype(float).values
    dv = df["dollar_volume"].astype(float).values
    logret = df["logret"].astype(float).values
    # Price change in dollars
    dP = np.concatenate([[np.nan], np.diff(close)])
    df["dP"] = dP

    # Rolling sigma of dP (shift-1, 250-bar)
    sigma_dP = rolling_shifted_std(pd.Series(dP, index=df.index), BASELINE_N).values
    df["sigma_dP_250"] = sigma_dP

    # BVC
    V_B, V_S = bvc(dP, vol, sigma_dP)
    df["V_buy_bvc"] = V_B
    df["V_sell_bvc"] = V_S
    df["OFI_bvc"] = np.where(vol > 0, (V_B - V_S) / vol, 0.0)

    # Tick-rule signed volume (approx)
    sign_t = np.sign(np.concatenate([[0], np.diff(close)]))
    df["signed_vol_tick"] = sign_t * vol
    df["signed_dv_tick"] = sign_t * dv

    # Abnormal volume z-scores (shift-1 baseline)
    v_s = pd.Series(vol, index=df.index)
    dv_s = pd.Series(dv, index=df.index)
    mu_V = rolling_shifted_mean(v_s, BASELINE_N).values
    sd_V = rolling_shifted_std(v_s, BASELINE_N).values
    mu_DV = rolling_shifted_mean(dv_s, BASELINE_N).values
    sd_DV = rolling_shifted_std(dv_s, BASELINE_N).values
    df["vol_z"] = np.where(sd_V > 0, (vol - mu_V) / sd_V, np.nan)
    df["dvol_z"] = np.where(sd_DV > 0, (dv - mu_DV) / sd_DV, np.nan)

    # Windowed VPIN (rolling 50-bar)
    V_abs = np.abs(V_B - V_S)
    V_abs_sum = pd.Series(V_abs, index=df.index).rolling(VPIN_N, min_periods=VPIN_N // 2).sum().values
    V_sum = pd.Series(vol, index=df.index).rolling(VPIN_N, min_periods=VPIN_N // 2).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        df["vpin_50"] = np.where(V_sum > 0, V_abs_sum / V_sum, np.nan)

    # VPIN z-score vs rolling baseline
    vpin_s = pd.Series(df["vpin_50"].values, index=df.index)
    mu_vp = rolling_shifted_mean(vpin_s, BASELINE_N).values
    sd_vp = rolling_shifted_std(vpin_s, BASELINE_N).values
    df["vpin_z"] = np.where(sd_vp > 0, (df["vpin_50"].values - mu_vp) / sd_vp, np.nan)

    # Kyle's lambda — rolling regression
    # Signed sqrt dollar volume using tick-rule sign
    s_sqrt_dv = sign_t * np.sqrt(np.abs(dv))
    lam = kyle_lambda_rolling(logret, s_sqrt_dv, KYLE_WIN)
    df["kyle_lambda_100"] = lam
    # Kyle lambda z-score vs its own rolling baseline (shifted)
    lam_s = pd.Series(lam, index=df.index)
    mu_lam = rolling_shifted_mean(lam_s, BASELINE_N).values
    sd_lam = rolling_shifted_std(lam_s, BASELINE_N).values
    with np.errstate(divide="ignore", invalid="ignore"):
        df["kyle_z"] = np.where(sd_lam > 0, (lam - mu_lam) / sd_lam, np.nan)

    return df


def main():
    manifest = {"resolutions": {}}
    for res_dir in sorted(BARS.iterdir()):
        if not res_dir.is_dir():
            continue
        res = res_dir.name
        out_dir = OUT / res
        out_dir.mkdir(exist_ok=True)
        print(f"\n=== Resolution {res} ===")
        res_m = {}
        for p in sorted(res_dir.glob("*.parquet")):
            tkr = p.stem
            df = pd.read_parquet(p)
            out = enrich(df)
            out_path = out_dir / f"{tkr}.parquet"
            out.to_parquet(out_path)
            # Quick sanity on a row
            any_ok = out["vol_z"].notna().sum()
            vpin_ok = out["vpin_50"].notna().sum()
            kyle_ok = out["kyle_lambda_100"].notna().sum()
            print(f"  {tkr:5s}: {len(out):6d} bars; vol_z ok={any_ok}, vpin ok={vpin_ok}, kyle ok={kyle_ok}")
            res_m[tkr] = {
                "rows": len(out),
                "vol_z_non_null": int(any_ok),
                "vpin_non_null": int(vpin_ok),
                "kyle_non_null": int(kyle_ok),
                "path": str(out_path),
            }
        manifest["resolutions"][res] = res_m

    import json
    with (OUT / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nSignals manifest written to {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
