#!/usr/bin/env python3
"""
04_event_study.py
=================

Pre/post order-flow event study around Trump Truth Social posts.

For each (topic, asset) pair we:
  1. Select posts flagged with the topic (`topic_<name> == True` in the
     posts_60d.parquet dataframe shipped under data/raw/).
  2. Snap each post to the bar containing its `created_at` (UTC).
  3. Build a symmetric ±6 bar window (i.e. ±30 minutes at 5-minute resolution).
  4. Apply the "initiated/reactive" classifier:
        |sum(pre-window logret) / (sigma_per_bar * sqrt(pre_bars))|  <  1.5
     where sigma_per_bar is a 250-bar rolling std of the 24-bar logret sum,
     shifted by 1 bar (no look-ahead). Posts that fail the filter look like
     reactions to a market move that already happened.
  5. For each surviving event, compute pre and post means / sums of the
     signal columns and bootstrap a two-sided p-value for mean != 0.

Output:
  data/results/orderflow_event_study_5m.json
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

from _paths import POSTS_PARQUET, SIGNALS_5M, EVENT_STUDY_5M_JSON, ensure_dirs

CFG = {
    "resolution": "5m",
    "bar_minutes": 5,
    "pre_bars": 6,
    "post_bars": 6,
    "initiated_pre_bars": 24,        # 2-hour pre window for the classifier
    "initiated_sigma_bars": 250,     # baseline length for abnormal-return sigma
    "initiated_threshold": 1.5,
    "topics": [
        "tariff_trade", "iran_military", "energy_oil", "market_economy", "djt_media",
        "musk_tesla", "big_tech", "big_oil_companies", "big_banks",
    ],
    "assets": [
        "DJT", "VXX", "SPY", "QQQ", "XLE", "USO", "GLD", "UUP", "XLF", "XLK",
    ],
    "bootstrap_draws": 5000,
    "rng_seed": 42,
}

SIGNAL_COLS = [
    "logret", "vol_z", "dvol_z", "signed_vol_tick", "OFI_bvc",
    "vpin_50", "vpin_z", "kyle_lambda_100", "dollar_volume",
]


def nearest_bar_index(bar_index, ts):
    pos = bar_index.searchsorted(ts, side="right") - 1
    return int(pos) if pos >= 0 else -1


def gather_window(signals, t_idx, pre, post):
    lo, hi = t_idx - pre, t_idx + post + 1
    if lo < 0 or hi > len(signals):
        return None
    w = signals.iloc[lo:hi].copy()
    w["offset"] = np.arange(-pre, post + 1)
    return w


def initiated_filter(window, pre_bars, sigma, threshold):
    pre = window[window["offset"].between(-pre_bars, -1)]
    if len(pre) == 0 or not np.isfinite(sigma) or sigma <= 0:
        return False
    car_pre = pre["logret"].sum()
    return abs(car_pre / (sigma * np.sqrt(pre_bars))) < threshold


def bootstrap_p(values, draws, rng):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan
    obs = v.mean()
    v0 = v - obs
    sims = rng.choice(v0, size=(draws, len(v)), replace=True).mean(axis=1)
    return float(2 * min((sims >= obs).mean(), (sims <= obs).mean()))


def run_study(cfg, posts):
    rng = np.random.default_rng(cfg["rng_seed"])
    out = {"config": cfg, "results": {}}

    signals = {}
    for a in cfg["assets"]:
        p = SIGNALS_5M / f"{a}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        signals[a] = df

    pre_bars = cfg["initiated_pre_bars"]
    sigma_bars = cfg["initiated_sigma_bars"]
    pre_sigmas = {}
    for a, df in signals.items():
        roll = df["logret"].rolling(pre_bars).sum()
        sig = roll.shift(1).rolling(sigma_bars, min_periods=sigma_bars // 5).std() / np.sqrt(pre_bars)
        pre_sigmas[a] = sig

    for topic in cfg["topics"]:
        topic_col = f"topic_{topic}"
        if topic_col not in posts.columns:
            print(f"[skip] {topic_col} not in posts dataframe")
            continue
        sub = posts[posts[topic_col]].copy()
        if len(sub) == 0:
            continue
        print(f"[{cfg['resolution']}] {topic}  n_posts={len(sub)}")
        topic_out = {}
        for asset in cfg["assets"]:
            if asset not in signals:
                continue
            df = signals[asset]
            bar_idx = df.index
            sigma_per_bar = pre_sigmas[asset]
            windows_all, windows_init = [], []
            for ts in sub["created_at"]:
                t_idx = nearest_bar_index(bar_idx, ts)
                if t_idx < cfg["pre_bars"] + pre_bars or t_idx >= len(df) - cfg["post_bars"]:
                    continue
                sig = float(sigma_per_bar.iloc[t_idx]) if t_idx < len(sigma_per_bar) else np.nan
                w = gather_window(df, t_idx, cfg["pre_bars"], cfg["post_bars"])
                if w is None:
                    continue
                windows_all.append(w)
                if initiated_filter(w, cfg["pre_bars"], sig, cfg["initiated_threshold"]):
                    windows_init.append(w)

            def summarise(windows):
                if len(windows) < 5:
                    return None
                w_len = windows[0].shape[0]
                offsets = windows[0]["offset"].values
                res = {"n": len(windows), "offsets": offsets.tolist(), "signals": {}}
                pre_mask = (offsets >= -cfg["pre_bars"]) & (offsets < 0)
                post_mask = (offsets > 0) & (offsets <= cfg["post_bars"])
                for col in SIGNAL_COLS:
                    mat = np.full((len(windows), w_len), np.nan)
                    for i, w in enumerate(windows):
                        mat[i, :] = w[col].values
                    by_offset_mean = np.nanmean(mat, axis=0)
                    by_offset_se = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(len(windows))
                    agg = "sum" if col in ("signed_vol_tick",) else "mean"
                    if agg == "sum":
                        pre_vals = np.nansum(mat[:, pre_mask], axis=1)
                        post_vals = np.nansum(mat[:, post_mask], axis=1)
                    else:
                        pre_vals = np.nanmean(mat[:, pre_mask], axis=1)
                        post_vals = np.nanmean(mat[:, post_mask], axis=1)
                    t_pre, p_pre = stats.ttest_1samp(pre_vals[np.isfinite(pre_vals)], 0.0)
                    t_post, p_post = stats.ttest_1samp(post_vals[np.isfinite(post_vals)], 0.0)
                    boot_pre = bootstrap_p(pre_vals, cfg["bootstrap_draws"], rng)
                    boot_post = bootstrap_p(post_vals, cfg["bootstrap_draws"], rng)
                    res["signals"][col] = {
                        "by_offset_mean": by_offset_mean.tolist(),
                        "by_offset_se": by_offset_se.tolist(),
                        "pre_mean": float(np.nanmean(pre_vals)),
                        "pre_std": float(np.nanstd(pre_vals, ddof=1)),
                        "pre_t": float(t_pre) if np.isfinite(t_pre) else None,
                        "pre_p": float(p_pre) if np.isfinite(p_pre) else None,
                        "pre_boot_p": boot_pre if np.isfinite(boot_pre) else None,
                        "post_mean": float(np.nanmean(post_vals)),
                        "post_std": float(np.nanstd(post_vals, ddof=1)),
                        "post_t": float(t_post) if np.isfinite(t_post) else None,
                        "post_p": float(p_post) if np.isfinite(p_post) else None,
                        "post_boot_p": boot_post if np.isfinite(boot_post) else None,
                    }
                return res

            topic_out[asset] = {"all": summarise(windows_all),
                                "initiated": summarise(windows_init)}
        out["results"][topic] = topic_out
    return out


def main():
    ensure_dirs()
    posts = pd.read_parquet(POSTS_PARQUET)
    if "created_at" not in posts.columns:
        posts = posts.reset_index()
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    print(f"loaded {len(posts)} posts from {POSTS_PARQUET}")

    res = run_study(CFG, posts)
    with EVENT_STUDY_5M_JSON.open("w") as f:
        json.dump(res, f, default=str)
    print(f"wrote {EVENT_STUDY_5M_JSON}")


if __name__ == "__main__":
    main()
