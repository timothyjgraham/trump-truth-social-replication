#!/usr/bin/env python3
"""
placebo_and_sensitivity.py

Placebo test + sensitivity suite for the order-flow event study.

1. RANDOM-TIMESTAMP PLACEBO.
   Generate N_PLACEBO random timestamps matched to the real posts' distribution
   over hour-of-day (ET) and weekday. Run the full event-study pipeline on these
   fake posts. A real signal should NOT appear in placebos; if it does, it's a
   hour-of-day / trading-pattern artefact and should be discounted.

2. FDR CORRECTION.
   With K topics × A assets × M metrics × 2 (pre/post) tests, we Benjamini-Hochberg
   correct across the joint table and report adjusted p-values.

3. WINDOW SENSITIVITY.
   Replicate headline results with pre/post windows of (3,3), (6,6), (12,12) bars
   (5m) to verify findings survive.

Inputs:
  /sessions/sleepy-gifted-ptolemy/work/data/signals/5m/*.parquet
  /sessions/sleepy-gifted-ptolemy/work/data/posts_60d.parquet
  /sessions/sleepy-gifted-ptolemy/work/data/orderflow_event_study_5m.json

Outputs:
  /sessions/sleepy-gifted-ptolemy/work/data/orderflow_placebo_5m.json
  /sessions/sleepy-gifted-ptolemy/work/data/orderflow_sensitivity_5m.json
  /sessions/sleepy-gifted-ptolemy/work/data/orderflow_fdr_5m.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

WORK = Path("/sessions/sleepy-gifted-ptolemy/work")
SIG = WORK / "data" / "signals" / "5m"
POSTS_PARQUET = WORK / "data" / "posts_60d.parquet"
OUT_DIR = WORK / "data"

TOPICS = [
    "tariff_trade", "iran_military", "energy_oil", "market_economy", "djt_media",
    "musk_tesla", "big_tech", "big_oil_companies", "big_banks",
]
ASSETS = [
    "DJT", "VXX", "SPY", "QQQ", "XLE", "USO", "GLD", "UUP", "XLF", "XLK",
    "TSLA", "NVDA", "MSFT", "AAPL", "META", "XOM", "CVX", "JPM", "GS",
]

SIGNAL_COLS = [
    "logret", "vol_z", "dvol_z", "signed_vol_tick", "OFI_bvc",
    "vpin_50", "vpin_z", "kyle_lambda_100", "kyle_z", "dollar_volume",
]


def load_signals():
    signals = {}
    for a in ASSETS:
        p = SIG / f"{a}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        signals[a] = df
    return signals


def nearest_bar_index(bar_index, ts):
    pos = bar_index.searchsorted(ts, side="right") - 1
    return int(pos) if pos >= 0 else -1


def run_one(posts, signals, pre_bars, post_bars, initiated_pre_bars, initiated_sigma_bars, initiated_threshold, bootstrap_draws, rng_seed, label):
    rng = np.random.default_rng(rng_seed)
    res = {"label": label, "n_posts": len(posts), "config": {
        "pre_bars": pre_bars, "post_bars": post_bars,
        "initiated_pre_bars": initiated_pre_bars,
        "initiated_sigma_bars": initiated_sigma_bars,
        "initiated_threshold": initiated_threshold,
    }, "results": {}}

    # Pre-compute asset sigma for initiated classifier
    pre_sigmas = {}
    for a, df in signals.items():
        roll = df["logret"].rolling(initiated_pre_bars).sum()
        sig = roll.shift(1).rolling(initiated_sigma_bars, min_periods=initiated_sigma_bars // 5).std() / np.sqrt(initiated_pre_bars)
        pre_sigmas[a] = sig

    for topic in TOPICS:
        topic_col = f"topic_{topic}"
        if topic_col in posts.columns:
            sub = posts[posts[topic_col]]
        else:
            sub = posts  # placebo: no topic filter
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
                lo = t_idx - pre_bars
                hi = t_idx + post_bars + 1
                w = df.iloc[lo:hi].copy()
                w["offset"] = np.arange(-pre_bars, post_bars + 1)
                # Initiated filter
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
            w_len = windows_init[0].shape[0]
            offsets = windows_init[0]["offset"].values
            out = {"n": n, "offsets": offsets.tolist(), "signals": {}}
            pre_mask = (offsets >= -pre_bars) & (offsets < 0)
            post_mask = (offsets > 0) & (offsets <= post_bars)
            for col in SIGNAL_COLS:
                mat = np.full((n, w_len), np.nan)
                for i, w in enumerate(windows_init):
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
    """Generate n_placebo timestamps that match the joint distribution of
    (hour-of-day-ET, weekday) of the real posts, sampled uniformly within eligible
    bars of the signal_index.
    """
    # Determine the (hour, weekday) distribution of real posts
    ts = pd.to_datetime(real_posts["created_at"], utc=True).dt.tz_convert("America/New_York")
    real_hw = pd.DataFrame({"hour": ts.dt.hour, "weekday": ts.dt.weekday})
    # All eligible bars (RTH true) and their (hour, weekday)
    et = signal_index.tz_convert("America/New_York")
    bar_df = pd.DataFrame({"ts_utc": signal_index,
                           "hour": et.hour,
                           "weekday": et.weekday}).reset_index(drop=True)
    # For each real post, sample one bar with matching (hour, weekday)
    placebos = []
    groups = {(h, w): g.index.values for (h, w), g in bar_df.groupby(["hour", "weekday"])}
    for _, row in real_hw.iterrows():
        key = (int(row["hour"]), int(row["weekday"]))
        idxs = groups.get(key)
        if idxs is None or len(idxs) == 0:
            continue
        for _ in range(n_placebo // len(real_hw) + 1):
            placebos.append(bar_df.loc[int(rng.choice(idxs)), "ts_utc"])
        if len(placebos) >= n_placebo:
            break
    placebos = placebos[:n_placebo]
    return pd.DataFrame({"created_at": placebos})


def bh_fdr(p_values):
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(p_values, dtype=float)
    mask = np.isfinite(p)
    pvals = p[mask]
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.argsort(order)
    sorted_p = pvals[order]
    adj = sorted_p * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    adj = np.minimum(adj, 1.0)
    out = np.full(len(p), np.nan)
    out[mask] = adj[ranks]
    return out


def main():
    posts = pd.read_parquet(POSTS_PARQUET)
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    signals = load_signals()
    # Rebuild signals to pick up kyle_z — or recompute on the fly
    # (kyle_z was added to build_signals.py after the first run)
    import subprocess
    print("Rebuilding signals to pick up kyle_z…")
    subprocess.run(["python3", str(WORK / "build_signals.py")], check=True)
    signals = load_signals()

    rng = np.random.default_rng(2026)

    print("\n=== Running placebo (5,000 random timestamps, matched hour/weekday) ===")
    spy_idx = signals["SPY"].index
    n_placebo = 5000
    placebo_posts = generate_placebos(posts, n_placebo, spy_idx, rng)
    # Assign all topic flags False; the run_one function falls through when topic
    # columns are absent, so we explicitly set each column True for ALL rows to
    # capture the full placebo set once.
    for t in TOPICS:
        placebo_posts[f"topic_{t}"] = False
    # Special "topic_placebo" — set one column True for all rows so we can reuse run_one
    placebo_posts["topic_placebo"] = True
    # Monkey-patch TOPICS for this one call via globals()
    globals()["TOPICS"] = ["placebo"]
    saved = TOPICS
    placebo_res = run_one(
        placebo_posts, signals,
        pre_bars=6, post_bars=6,
        initiated_pre_bars=24, initiated_sigma_bars=250, initiated_threshold=1.5,
        bootstrap_draws=2000, rng_seed=2026, label="placebo_5m",
    )
    globals()["TOPICS"] = [
        "tariff_trade", "iran_military", "energy_oil", "market_economy", "djt_media",
        "musk_tesla", "big_tech", "big_oil_companies", "big_banks",
    ]
    with (OUT_DIR / "orderflow_placebo_5m.json").open("w") as f:
        json.dump(placebo_res, f, default=str)
    print(f"  Wrote {OUT_DIR/'orderflow_placebo_5m.json'}")

    print("\n=== Running sensitivity: window (3,3), (6,6), (12,12) ===")
    sens = {}
    for pre, post in [(3, 3), (6, 6), (12, 12)]:
        label = f"w_{pre}_{post}"
        print(f"  Window {label}")
        r = run_one(
            posts, signals,
            pre_bars=pre, post_bars=post,
            initiated_pre_bars=24, initiated_sigma_bars=250, initiated_threshold=1.5,
            bootstrap_draws=1000, rng_seed=pre * 100 + post, label=label,
        )
        sens[label] = r
    with (OUT_DIR / "orderflow_sensitivity_5m.json").open("w") as f:
        json.dump(sens, f, default=str)
    print(f"  Wrote {OUT_DIR/'orderflow_sensitivity_5m.json'}")

    print("\n=== Applying BH-FDR correction across the joint test table ===")
    with (OUT_DIR / "orderflow_event_study_5m.json").open() as f:
        main_res = json.load(f)
    rows = []
    for topic, assets in main_res["results"].items():
        for asset, sub in assets.items():
            ini = sub.get("initiated")
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
    fdr_df.to_parquet(OUT_DIR / "orderflow_fdr_5m.parquet")
    # Summary: what survives BH-FDR at q=0.05?
    # (drop level-based signals whose mean-vs-0 test is trivially rejected:
    #  kyle_lambda_100, vpin_50, dollar_volume all have positive baselines by
    #  construction. We want z-scored / signed / imbalance signals only.)
    LEVEL_SIGNALS = {"kyle_lambda_100", "vpin_50", "dollar_volume"}
    keep = fdr_df[~fdr_df["signal"].isin(LEVEL_SIGNALS)].copy()
    keep["p_fdr_trim"] = bh_fdr(keep["p"].values)
    survivors = keep[keep["p_fdr_trim"] < 0.05].sort_values(["p_fdr_trim", "topic", "asset"])
    print(f"  Total tests: {len(keep)}")
    print(f"  Survivors at FDR q=0.05: {len(survivors)}")
    print(survivors.head(30).to_string())
    with (OUT_DIR / "orderflow_fdr_5m.json").open("w") as f:
        json.dump({
            "n_tests_total": len(keep),
            "n_survivors_fdr_005": int(len(survivors)),
            "survivors": survivors.to_dict(orient="records"),
        }, f, default=str, indent=2)


if __name__ == "__main__":
    main()
