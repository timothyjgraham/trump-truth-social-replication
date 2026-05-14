#!/usr/bin/env python3
"""
event_study.py

Pre/post order-flow event study around Trump Truth Social posts.

For each (topic, asset) pair:
  - select posts in the topic that fall within a bar-resolution match of asset data
  - align a symmetric window around each post, in bar units
  - compute mean signal at each offset, and headline pre-CAR / post-CAR aggregates
  - bootstrap p-values (null = random timestamps) and t-tests
  - apply |pre-z(price)| < 1.5 initiated filter per (post, asset), following v2 methodology

Resolution: 5m bars (headline); 1m bars (DJT-specific ultra-high-resolution spot-check).
Window:     [-6, +6] bars at 5m resolution  (i.e. -30 min / +30 min around the post)
            [-30, +30] bars at 1m resolution (i.e. -30 min / +30 min around the post)

Signals analysed at each offset (mean across events):
  - logret            : log-return                         (price effect, for reference)
  - vol_z             : abnormal volume z-score
  - dvol_z            : abnormal dollar-volume z-score
  - signed_vol_tick   : signed volume (tick-rule)
  - OFI_bvc           : BVC order-flow imbalance
  - vpin_z            : abnormal VPIN z-score
  - kyle_lambda_100   : local Kyle's lambda

Pre- and post-window aggregates:
  pre_sum        = sum of signed_vol_tick over [-6..-1]        (front-running indicator)
  post_sum       = sum of signed_vol_tick over [+1..+6]        (reaction indicator)
  pre_mean_volz  = mean vol_z over [-6..-1]
  post_mean_volz = mean vol_z over [+1..+6]
  (ditto OFI, VPIN, Kyle's lambda)

Output:
  /sessions/sleepy-gifted-ptolemy/work/data/orderflow_event_study_5m.json
  /sessions/sleepy-gifted-ptolemy/work/data/orderflow_event_study_1m.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

WORK = Path("/sessions/sleepy-gifted-ptolemy/work")
SIG = WORK / "data" / "signals"
POSTS_PARQUET = WORK / "data" / "posts_60d.parquet"
OUT_DIR = WORK / "data"

# 5m resolution config
CFG_5M = {
    "resolution": "5m",
    "bar_minutes": 5,
    "pre_bars": 6,
    "post_bars": 6,
    "initiated_pre_bars": 24,       # 2h pre for classifier (24 × 5m)
    "initiated_sigma_bars": 250,    # baseline for abn-return sigma
    "initiated_threshold": 1.5,
    "topics": [
        "tariff_trade", "iran_military", "energy_oil", "market_economy", "djt_media",
        "musk_tesla", "big_tech", "big_oil_companies", "big_banks",
    ],
    "assets": [
        "DJT", "VXX", "SPY", "QQQ", "XLE", "USO", "GLD", "UUP", "XLF", "XLK",
        "TSLA", "NVDA", "MSFT", "AAPL", "META", "XOM", "CVX", "JPM", "GS",
    ],
    "bootstrap_draws": 5000,
    "rng_seed": 42,
}

CFG_1M = {
    "resolution": "1m",
    "bar_minutes": 1,
    "pre_bars": 30,
    "post_bars": 30,
    "initiated_pre_bars": 120,     # 2h pre for classifier (120 × 1m)
    "initiated_sigma_bars": 250,
    "initiated_threshold": 1.5,
    "topics": [
        "tariff_trade", "iran_military", "energy_oil", "market_economy", "djt_media",
        "musk_tesla", "big_tech", "big_oil_companies", "big_banks",
    ],
    "assets": [
        "DJT", "VXX", "SPY", "QQQ", "XLE", "USO", "GLD", "UUP", "XLF", "XLK",
        "TSLA", "NVDA", "MSFT", "AAPL", "META", "XOM", "CVX", "JPM", "GS",
    ],
    "bootstrap_draws": 5000,
    "rng_seed": 42,
}

SIGNAL_COLS = [
    "logret", "vol_z", "dvol_z", "signed_vol_tick", "OFI_bvc",
    "vpin_50", "vpin_z", "kyle_lambda_100", "dollar_volume",
]


def nearest_bar_index(bar_index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    """Find the index of the bar that contains or is closest-to-after ts."""
    # searchsorted: insertion index to keep sorted; bar timestamps are left-closed,
    # so the event time ts "fires" within the bar whose stamp is <= ts < next stamp.
    pos = bar_index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return -1
    return int(pos)


def gather_window(signals: pd.DataFrame, t_idx: int, pre: int, post: int) -> pd.DataFrame | None:
    """Return a window DataFrame with offsets -pre..+post, or None if out-of-bounds."""
    lo = t_idx - pre
    hi = t_idx + post + 1
    if lo < 0 or hi > len(signals):
        return None
    w = signals.iloc[lo:hi].copy()
    w["offset"] = np.arange(-pre, post + 1)
    return w


def initiated_filter(window: pd.DataFrame, pre_bars: int, sigma: float,
                     threshold: float) -> bool:
    """|sum(pre-window logret) / sigma| < threshold => initiated."""
    pre = window[window["offset"].between(-pre_bars, -1)]
    if len(pre) == 0 or not np.isfinite(sigma) or sigma <= 0:
        return False
    car_pre = pre["logret"].sum()
    return abs(car_pre / (sigma * np.sqrt(pre_bars))) < threshold


def bootstrap_p(values: np.ndarray, draws: int, rng: np.random.Generator) -> float:
    """Two-sided null-centred bootstrap p-value on the mean."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan
    obs = v.mean()
    v0 = v - obs
    sims = rng.choice(v0, size=(draws, len(v)), replace=True).mean(axis=1)
    return float(2 * min((sims >= obs).mean(), (sims <= obs).mean()))


def run_study(cfg: dict, posts: pd.DataFrame) -> dict:
    rng = np.random.default_rng(cfg["rng_seed"])
    out = {"config": cfg, "results": {}}

    # Pre-load all asset signal frames (indexed on UTC datetime)
    signals = {}
    for a in cfg["assets"]:
        p = SIG / cfg["resolution"] / f"{a}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = df.sort_index()
        # Ensure a monotonic UTC index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        signals[a] = df

    # Compute per-asset 2h-pre abnormal-return sigma using rolling (pre_bars period)
    pre_bars = cfg["initiated_pre_bars"]
    sigma_bars = cfg["initiated_sigma_bars"]
    pre_sigmas = {}
    for a, df in signals.items():
        # pre_bars-sum logret rolling std, shifted by 1
        roll = df["logret"].rolling(pre_bars).sum()
        sig = roll.shift(1).rolling(sigma_bars, min_periods=sigma_bars // 5).std() / np.sqrt(pre_bars)
        pre_sigmas[a] = sig  # per-bar sigma of per-bar logret (conservative normaliser)

    for topic in cfg["topics"]:
        topic_col = f"topic_{topic}"
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

            windows_all = []
            windows_init = []

            for ts in sub["created_at"]:
                t_idx = nearest_bar_index(bar_idx, ts)
                if t_idx < cfg["pre_bars"] + pre_bars or t_idx >= len(df) - cfg["post_bars"]:
                    continue
                # Initiated classifier — use the sigma at t_idx (already shift-1 baseline)
                sig = float(sigma_per_bar.iloc[t_idx]) if t_idx < len(sigma_per_bar) else np.nan
                w = gather_window(df, t_idx, cfg["pre_bars"], cfg["post_bars"])
                if w is None:
                    continue
                windows_all.append(w)
                if initiated_filter(w, cfg["pre_bars"], sig, cfg["initiated_threshold"]):
                    windows_init.append(w)

            def summarise(windows: list[pd.DataFrame], tag: str) -> dict | None:
                if len(windows) < 5:
                    return None
                # Stack each signal into an (n, window) matrix
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
                    # Aggregated pre/post statistics (sum for flow cols, mean for z / lambda)
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

            asset_out = {
                "all": summarise(windows_all, "all"),
                "initiated": summarise(windows_init, "initiated"),
            }
            topic_out[asset] = asset_out
        out["results"][topic] = topic_out
    return out


def main():
    posts = pd.read_parquet(POSTS_PARQUET)
    if "created_at" not in posts.columns:
        posts = posts.reset_index()
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    print(f"Loaded {len(posts)} posts from {POSTS_PARQUET}")

    # 5m study (headline)
    res5 = run_study(CFG_5M, posts)
    with (OUT_DIR / "orderflow_event_study_5m.json").open("w") as f:
        json.dump(res5, f, default=str)
    print(f"Wrote {OUT_DIR/'orderflow_event_study_5m.json'}")

    # 1m study (DJT-focused, but we still run for all assets; posts within 7-day window only)
    last_1m_end = pd.Timestamp("2026-04-09", tz="UTC")  # truth archive end
    last_1m_start = pd.Timestamp("2026-04-13", tz="UTC")  # 1m bars start here
    # => 1m overlap with posts is empty (truth archive ends before 1m bars start).
    #    So the 1m study runs on zero posts. Skip it and note the gap.
    overlap = posts[(posts["created_at"] >= last_1m_start)]
    print(f"1m overlap posts: {len(overlap)} (expected 0 — truth archive ends Apr 9, 1m data starts Apr 13)")
    if len(overlap) > 0:
        res1 = run_study(CFG_1M, overlap)
        with (OUT_DIR / "orderflow_event_study_1m.json").open("w") as f:
            json.dump(res1, f, default=str)
        print(f"Wrote {OUT_DIR/'orderflow_event_study_1m.json'}")


if __name__ == "__main__":
    main()
