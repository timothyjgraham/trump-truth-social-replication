#!/usr/bin/env python3
"""
07_time_shift_test.py
=====================

+24h time-shift falsification.

For every oil-themed post (`topic_energy_oil == True`), shift created_at
forward by exactly 24 hours and re-run the same pre-event order-flow
calculation on USO and XLE. Apply the same |pre-CAR z| < 1.5 initiated
filter as the headline pipeline.

Logic:
  If the headline finding is causal (pre-event order flow tied to the
  posts themselves), the shifted timestamps should produce a near-zero
  signal that is no longer significantly different from zero.
  If instead the result is driven by daily seasonality or some other
  intraday pattern, the shifted version will look similar.

Outputs:
  data/results/phase2b_timeshift.json
"""

import json

import numpy as np
import pandas as pd

from _paths import POSTS_PARQUET, MINUTE_BARS_5M, SIGNALS_5M, TIMESHIFT_JSON, ensure_dirs

PRE_BARS = 6   # 30 min at 5m
POST_BARS = 6
INITIATED_Z_CUT = 1.5
SHIFT_HOURS = 24
N_BOOT = 2000
RNG = np.random.default_rng(20260422)


def load_pair(symbol):
    bars = pd.read_parquet(MINUTE_BARS_5M / f"{symbol}.parquet")
    sigs = pd.read_parquet(SIGNALS_5M / f"{symbol}.parquet")
    bars.index = pd.to_datetime(bars.index, utc=True)
    sigs.index = pd.to_datetime(sigs.index, utc=True)
    return bars, sigs


def event_pre_stats(bars, sigs, post_ts, pre_bars=PRE_BARS, post_bars=POST_BARS):
    idx = bars.index
    pos = idx.searchsorted(post_ts, side="right") - 1
    if pos < 0 or pos >= len(idx):
        return None
    if pos - pre_bars < 0 or pos + post_bars >= len(idx):
        return None
    pre_slice = slice(pos - pre_bars, pos)
    pre = sigs.iloc[pre_slice]
    if pre.empty:
        return None
    out = {col + "_pre_mean": float(pre[col].mean()) if col in pre.columns else np.nan
           for col in ("OFI_bvc", "vpin_z", "signed_vol_tick")}
    close = bars["Close"].astype(float)
    log_close = np.log(close)
    ret = log_close.diff()
    pre_ret_cum = ret.iloc[pre_slice].sum()
    pre_ret_std = ret.iloc[max(0, pos - 288):pos].std()
    if pre_ret_std and pre_ret_std > 0:
        z = pre_ret_cum / (pre_ret_std * np.sqrt(pre_bars))
    else:
        z = np.nan
    out["pre_car_z"] = float(z) if not np.isnan(z) else np.nan
    return out


def boot_p(x, n=N_BOOT, rng=RNG):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, (np.nan, np.nan)
    obs = float(x.mean())
    centred = x - obs
    bm = np.empty(n)
    for i in range(n):
        bm[i] = rng.choice(centred, size=len(x), replace=True).mean()
    p = float((np.abs(bm) >= abs(obs)).mean())
    ci_lo, ci_hi = float(np.percentile(bm + obs, 2.5)), float(np.percentile(bm + obs, 97.5))
    return obs, p, (ci_lo, ci_hi)


def main():
    ensure_dirs()
    posts = pd.read_parquet(POSTS_PARQUET)
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    oil = posts[posts["topic_energy_oil"] == True].copy()
    print(f"oil-themed posts in study window: {len(oil)}")
    oil["shifted_ts"] = oil["created_at"] + pd.Timedelta(hours=SHIFT_HOURS)

    results = {
        "method": f"+{SHIFT_HOURS}h time-shift falsification",
        "shift_hours": SHIFT_HOURS,
        "pre_bars_5m": PRE_BARS,
        "post_bars_5m": POST_BARS,
        "initiated_z_cut": INITIATED_Z_CUT,
        "n_oil_posts_input": int(len(oil)),
        "assets": {},
    }

    for sym in ("USO", "XLE"):
        print(f"\n=== {sym} ===")
        bars, sigs = load_pair(sym)
        rows_real, rows_shift = [], []
        for ts_real, ts_shift in zip(oil["created_at"], oil["shifted_ts"]):
            r = event_pre_stats(bars, sigs, ts_real)
            if r is not None:
                rows_real.append(r)
            r = event_pre_stats(bars, sigs, ts_shift)
            if r is not None:
                rows_shift.append(r)
        rdf = pd.DataFrame(rows_real)
        sdf = pd.DataFrame(rows_shift)
        rdf_init = rdf[rdf["pre_car_z"].abs() < INITIATED_Z_CUT].copy()
        sdf_init = sdf[sdf["pre_car_z"].abs() < INITIATED_Z_CUT].copy()
        print(f"  real:    matched {len(rdf):>4d}  initiated {len(rdf_init):>4d}")
        print(f"  shifted: matched {len(sdf):>4d}  initiated {len(sdf_init):>4d}")

        block = {"real_reproduced": {}, "shifted": {}}
        for col in ("OFI_bvc", "vpin_z", "signed_vol_tick"):
            for label, frame in (("real_reproduced", rdf_init), ("shifted", sdf_init)):
                obs, p, ci = boot_p(frame[col + "_pre_mean"].values)
                block[label][col] = {"n": int(len(frame)),
                                     "pre_mean": obs,
                                     "boot_p_two_sided": p,
                                     "boot_ci95": ci}
                print(f"  {label:18s} {col:18s}  n={len(frame):>4d}  "
                      f"mean={obs:>+12.5g}  p={p:.3f}")
        results["assets"][sym] = block

    with TIMESHIFT_JSON.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {TIMESHIFT_JSON}")


if __name__ == "__main__":
    main()
