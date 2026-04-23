#!/usr/bin/env python3
"""
05_placebo_and_sensitivity.py
=============================

Three robustness checks on the headline event study:

  (1) MATCHED-PLACEBO. Generate 5,000 random timestamps that match the joint
      distribution of (hour-of-day-ET, weekday) of the real posts. Run the same
      pre/post pipeline on those fake posts. A real signal should NOT appear in
      the placebo set; if it does it is an intraday-pattern artefact.

  (2) WINDOW SENSITIVITY. Replicate the headline pipeline with pre/post
      windows of (3,3), (6,6), (12,12) bars to confirm findings survive.

  (3) BENJAMINI-HOCHBERG FDR. Across the joint table of
      (topic × asset × signal × {pre, post}) tests, apply BH-FDR correction
      and identify what survives at q = 0.05.

Inputs:
  data/interim/signals_5m/<TICKER>.parquet
  data/raw/posts_60d.parquet
  data/results/orderflow_event_study_5m.json   (from stage 04)

Outputs:
  data/results/orderflow_placebo_5m.json
  data/results/orderflow_sensitivity_5m.json
  data/results/orderflow_fdr_5m.json
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

from _paths import (
    SIGNALS_5M, POSTS_PARQUET, EVENT_STUDY_5M_JSON,
    PLACEBO_5M_JSON, SENSITIVITY_5M_JSON, FDR_5M_JSON, RESULTS_DIR, ensure_dirs,
)

TOPICS = [
    "tariff_trade", "iran_military", "energy_oil", "market_economy", "djt_media",
    "musk_tesla", "big_tech", "big_oil_companies", "big_banks",
]
ASSETS = [
    "DJT", "VXX", "SPY", "QQQ", "XLE", "USO", "GLD", "UUP", "XLF", "XLK",
]

SIGNAL_COLS = [
    "logret", "vol_z", "dvol_z", "signed_vol_tick", "OFI_bvc",
    "vpin_50", "vpin_z", "kyle_lambda_100", "kyle_z", "dollar_volume",
]


def load_signals():
    out = {}
    for a in ASSETS:
        p = SIGNALS_5M / f"{a}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        out[a] = df
    return out


def nearest_bar_index(bar_index, ts):
    pos = bar_index.searchsorted(ts, side="right") - 1
    return int(pos) if pos >= 0 else -1


def run_one(posts, signals, topics, pre_bars, post_bars,
            initiated_pre_bars, initiated_sigma_bars, initiated_threshold,
            label):
    res = {"label": label, "n_posts": len(posts), "config": {
        "pre_bars": pre_bars, "post_bars": post_bars,
        "initiated_pre_bars": initiated_pre_bars,
        "initiated_sigma_bars": initiated_sigma_bars,
        "initiated_threshold": initiated_threshold,
    }, "results": {}}

    pre_sigmas = {}
    for a, df in signals.items():
        roll = df["logret"].rolling(initiated_pre_bars).sum()
        sig = roll.shift(1).rolling(initiated_sigma_bars,
                                    min_periods=initiated_sigma_bars // 5).std() / np.sqrt(initiated_pre_bars)
        pre_sigmas[a] = sig

    for topic in topics:
        topic_col = f"topic_{topic}"
        sub = posts[posts[topic_col]] if topic_col in posts.columns else posts
        if len(sub) == 0:
            continue
        topic_out = {}
        for asset in ASSETS:
            if asset not in signals:
                continue
            df = signals[asset]
            bar_idx = df.index
            sigma_per_bar = pre_sigmas[asset]
            windows_init = []
            for ts in sub["created_at"]:
                t_idx = nearest_bar_index(bar_idx, ts)
                if t_idx < pre_bars + initiated_pre_bars or t_idx >= len(df) - post_bars:
                    continue
                sig = float(sigma_per_bar.iloc[t_idx]) if t_idx < len(sigma_per_bar) else np.nan
                lo, hi = t_idx - pre_bars, t_idx + post_bars + 1
                w = df.iloc[lo:hi].copy()
                w["offset"] = np.arange(-pre_bars, post_bars + 1)
                pre = w[w["offset"].between(-pre_bars, -1)]
                if len(pre) == 0 or not np.isfinite(sig) or sig <= 0:
                    continue
                car_pre = pre["logret"].sum()
                if abs(car_pre / (sig * np.sqrt(pre_bars))) >= initiated_threshold:
                    continue
                windows_init.append(w)

            if len(windows_init) < 5:
                topic_out[asset] = {"n": len(windows_init), "insufficient": True}
                continue

            n = len(windows_init)
            offsets = windows_init[0]["offset"].values
            out = {"n": n, "offsets": offsets.tolist(), "signals": {}}
            pre_mask = (offsets >= -pre_bars) & (offsets < 0)
            post_mask = (offsets > 0) & (offsets <= post_bars)
            w_len = windows_init[0].shape[0]
            for col in SIGNAL_COLS:
                mat = np.full((n, w_len), np.nan)
                for i, w in enumerate(windows_init):
                    if col in w.columns:
                        mat[i, :] = w[col].values
                agg = "sum" if col in ("signed_vol_tick",) else "mean"
                if agg == "sum":
                    pre_vals = np.nansum(mat[:, pre_mask], axis=1)
                    post_vals = np.nansum(mat[:, post_mask], axis=1)
                else:
                    pre_vals = np.nanmean(mat[:, pre_mask], axis=1)
                    post_vals = np.nanmean(mat[:, post_mask], axis=1)
                pre_vals = pre_vals[np.isfinite(pre_vals)]
                post_vals = post_vals[np.isfinite(post_vals)]
                if len(pre_vals) < 5 or len(post_vals) < 5:
                    continue
                t_pre, p_pre = stats.ttest_1samp(pre_vals, 0.0)
                t_post, p_post = stats.ttest_1samp(post_vals, 0.0)
                out["signals"][col] = {
                    "pre_mean": float(np.nanmean(pre_vals)),
                    "pre_p": float(p_pre) if np.isfinite(p_pre) else None,
                    "post_mean": float(np.nanmean(post_vals)),
                    "post_p": float(p_post) if np.isfinite(p_post) else None,
                    "n_pre": int(len(pre_vals)),
                    "n_post": int(len(post_vals)),
                }
            topic_out[asset] = out
        res["results"][topic] = topic_out
    return res


def generate_placebos(real_posts, n_placebo, signal_index, rng):
    ts = pd.to_datetime(real_posts["created_at"], utc=True).dt.tz_convert("America/New_York")
    real_hw = pd.DataFrame({"hour": ts.dt.hour, "weekday": ts.dt.weekday})
    et = signal_index.tz_convert("America/New_York")
    bar_df = pd.DataFrame({"ts_utc": signal_index, "hour": et.hour,
                           "weekday": et.weekday}).reset_index(drop=True)
    placebos = []
    groups = {(h, w): g.index.values for (h, w), g in bar_df.groupby(["hour", "weekday"])}
    per_post = max(1, n_placebo // max(len(real_hw), 1) + 1)
    for _, row in real_hw.iterrows():
        key = (int(row["hour"]), int(row["weekday"]))
        idxs = groups.get(key)
        if idxs is None or len(idxs) == 0:
            continue
        for _ in range(per_post):
            placebos.append(bar_df.loc[int(rng.choice(idxs)), "ts_utc"])
        if len(placebos) >= n_placebo:
            break
    placebos = placebos[:n_placebo]
    return pd.DataFrame({"created_at": placebos})


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    mask = np.isfinite(p)
    pvals = p[mask]
    n = len(pvals)
    if n == 0:
        return np.full_like(p, np.nan)
    order = np.argsort(pvals)
    ranks = np.argsort(order)
    sorted_p = pvals[order]
    adj = sorted_p * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.full(len(p), np.nan)
    out[mask] = adj[ranks]
    return out


def main():
    ensure_dirs()
    posts = pd.read_parquet(POSTS_PARQUET)
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    signals = load_signals()
    rng = np.random.default_rng(2026)

    # 1. Matched placebo
    print("\n=== matched placebo  (n=5000, hour-and-weekday matched) ===")
    spy_idx = signals["SPY"].index
    placebo = generate_placebos(posts, 5000, spy_idx, rng)
    for t in TOPICS:
        placebo[f"topic_{t}"] = False
    placebo["topic_placebo"] = True
    placebo_res = run_one(
        placebo, signals, ["placebo"],
        pre_bars=6, post_bars=6,
        initiated_pre_bars=24, initiated_sigma_bars=250, initiated_threshold=1.5,
        label="placebo_5m",
    )
    with PLACEBO_5M_JSON.open("w") as f:
        json.dump(placebo_res, f, default=str)
    print(f"  -> {PLACEBO_5M_JSON}")

    # 2. Window sensitivity
    print("\n=== window sensitivity  (3,3) (6,6) (12,12) ===")
    sens = {}
    for pre, post in [(3, 3), (6, 6), (12, 12)]:
        label = f"w_{pre}_{post}"
        print(f"  {label}")
        sens[label] = run_one(
            posts, signals, TOPICS,
            pre_bars=pre, post_bars=post,
            initiated_pre_bars=24, initiated_sigma_bars=250, initiated_threshold=1.5,
            label=label,
        )
    with SENSITIVITY_5M_JSON.open("w") as f:
        json.dump(sens, f, default=str)
    print(f"  -> {SENSITIVITY_5M_JSON}")

    # 3. BH-FDR
    print("\n=== BH-FDR across the joint test table ===")
    with EVENT_STUDY_5M_JSON.open() as f:
        main_res = json.load(f)
    rows = []
    for topic, assets in main_res["results"].items():
        for asset, sub in assets.items():
            ini = sub.get("initiated") if isinstance(sub, dict) else None
            if ini is None:
                continue
            for col, s in ini.get("signals", {}).items():
                rows.append({"topic": topic, "asset": asset, "signal": col, "window": "pre",
                             "mean": s["pre_mean"], "p": s.get("pre_boot_p") or s.get("pre_p"),
                             "n": ini["n"]})
                rows.append({"topic": topic, "asset": asset, "signal": col, "window": "post",
                             "mean": s["post_mean"], "p": s.get("post_boot_p") or s.get("post_p"),
                             "n": ini["n"]})
    fdr_df = pd.DataFrame(rows)
    fdr_df["p_fdr"] = bh_fdr(fdr_df["p"].values)
    fdr_df.to_parquet(RESULTS_DIR / "orderflow_fdr_5m.parquet")

    LEVEL_SIGNALS = {"kyle_lambda_100", "vpin_50", "dollar_volume"}
    keep = fdr_df[~fdr_df["signal"].isin(LEVEL_SIGNALS)].copy()
    keep["p_fdr_trim"] = bh_fdr(keep["p"].values)
    survivors = keep[keep["p_fdr_trim"] < 0.05].sort_values(["p_fdr_trim", "topic", "asset"])
    print(f"  total tests:          {len(keep)}")
    print(f"  survivors at q=0.05:  {len(survivors)}")
    if len(survivors) > 0:
        print(survivors.head(20).to_string())

    with FDR_5M_JSON.open("w") as f:
        json.dump({
            "n_tests_total": len(keep),
            "n_survivors_fdr_005": int(len(survivors)),
            "survivors": survivors.to_dict(orient="records"),
        }, f, default=str, indent=2)
    print(f"  -> {FDR_5M_JSON}")


if __name__ == "__main__":
    main()
