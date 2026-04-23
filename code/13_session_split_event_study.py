#!/usr/bin/env python3
"""
13_session_split_event_study.py
===============================

Re-cut of the headline event study with oil-themed posts split by the
session in which they were published (US Eastern Time):

  - regular     : Mon-Fri, 09:30-16:00 ET
  - pre_market  : Mon-Fri, 04:00-09:30 ET
  - after_hours : Mon-Fri, 16:00-04:00 ET (next morning)
  - weekend     : all day Sat & Sun

For each session bucket we re-run the OFI_bvc and vpin_z pre and post
event tests on XLE and USO using the existing 5-minute signals, and
compare each subset's mean against (a) zero with a 5,000-draw bootstrap
and (b) the asset's overall matched-placebo baseline (from the existing
5,000-event placebo pool).

Hypothesis (from Section 5.2 of the report): the OFI/VPIN elevation
should concentrate in the after-hours bucket, since 75% of the 81
triggered events sit there.

Outputs:
  data/results/session_split.json
  report/figures/fig6_session_split.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import (
    POSTS_PARQUET, SIGNALS_5M, RESULTS_DIR, FIGURES_DIR
)

ET = ZoneInfo("America/New_York")

# ── palette (matches figs 1-5) ───────────────────────────────────────────────
NAVY        = "#1f3b6e"
LIGHT_BLUE  = "#7aa9d6"
GREY        = "#9aa1aa"
ORANGE      = "#d97c2a"
RED         = "#b73a3a"
GREEN       = "#3a7d5a"

CFG = {
    "pre_bars": 6, "post_bars": 6,
    "initiated_pre_bars": 24, "initiated_sigma_bars": 250,
    "initiated_threshold": 1.5,
    "bootstrap_draws": 5000,
    "rng_seed": 20260423,
}

ASSETS  = ["XLE", "USO"]
SIGNALS = ["OFI_bvc", "vpin_z"]
SESSIONS = ["regular", "after_hours", "weekend", "pre_market"]


def session_label(dt: pd.Series) -> pd.Series:
    """Return one of {regular, pre_market, after_hours, weekend} per row."""
    out = pd.Series(["regular"] * len(dt), index=dt.index)
    is_weekend = dt.dt.weekday >= 5
    out[is_weekend] = "weekend"
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    pre = (~is_weekend) & (minute_of_day >= 4*60) & (minute_of_day < 9*60 + 30)
    after = (~is_weekend) & ((minute_of_day >= 16*60) | (minute_of_day < 4*60))
    out[pre] = "pre_market"
    out[after] = "after_hours"
    return out


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


def load_signals(asset):
    p = SIGNALS_5M / f"{asset}.parquet"
    df = pd.read_parquet(p).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def precompute_initiation_sigma(signals_df):
    pre_bars = CFG["initiated_pre_bars"]
    sigma_bars = CFG["initiated_sigma_bars"]
    roll = signals_df["logret"].rolling(pre_bars).sum()
    sig = roll.shift(1).rolling(sigma_bars,
                                min_periods=sigma_bars // 5).std() / np.sqrt(pre_bars)
    return sig


def collect_windows(posts_subset, signals_df, pre_sigmas):
    windows = []
    for ts in posts_subset["created_at"]:
        t_idx = nearest_bar_index(signals_df.index, ts)
        if t_idx < CFG["pre_bars"] + CFG["initiated_pre_bars"] or \
           t_idx >= len(signals_df) - CFG["post_bars"]:
            continue
        sig = float(pre_sigmas.iloc[t_idx]) if t_idx < len(pre_sigmas) else np.nan
        w = gather_window(signals_df, t_idx, CFG["pre_bars"], CFG["post_bars"])
        if w is None:
            continue
        if initiated_filter(w, CFG["pre_bars"], sig, CFG["initiated_threshold"]):
            windows.append(w)
    return windows


def summarise_window_set(windows, rng):
    if len(windows) < 5:
        return None
    w_len = windows[0].shape[0]
    offsets = windows[0]["offset"].values
    res = {"n": len(windows), "signals": {}}
    pre_mask = (offsets >= -CFG["pre_bars"]) & (offsets < 0)
    post_mask = (offsets > 0) & (offsets <= CFG["post_bars"])
    for col in SIGNALS:
        mat = np.full((len(windows), w_len), np.nan)
        for i, w in enumerate(windows):
            mat[i, :] = w[col].values
        pre_vals = np.nanmean(mat[:, pre_mask], axis=1)
        post_vals = np.nanmean(mat[:, post_mask], axis=1)
        res["signals"][col] = {
            "pre_mean": float(np.nanmean(pre_vals)),
            "pre_std":  float(np.nanstd(pre_vals, ddof=1)),
            "pre_se":   float(np.nanstd(pre_vals, ddof=1) /
                              np.sqrt(np.isfinite(pre_vals).sum())),
            "pre_boot_p":  bootstrap_p(pre_vals, CFG["bootstrap_draws"], rng),
            "post_mean": float(np.nanmean(post_vals)),
            "post_std":  float(np.nanstd(post_vals, ddof=1)),
            "post_se":   float(np.nanstd(post_vals, ddof=1) /
                               np.sqrt(np.isfinite(post_vals).sum())),
            "post_boot_p": bootstrap_p(post_vals, CFG["bootstrap_draws"], rng),
        }
    return res


def load_placebo_baselines():
    pl = json.load(open(RESULTS_DIR / "orderflow_placebo_5m.json"))
    out = {}
    for asset in ASSETS:
        if asset not in pl["results"]["placebo"]:
            continue
        out[asset] = {}
        for sig in SIGNALS:
            s = pl["results"]["placebo"][asset]["signals"][sig]
            out[asset][sig] = {"pre_mean": s["pre_mean"],
                               "post_mean": s["post_mean"]}
    return out


# ── figure ───────────────────────────────────────────────────────────────────

def make_figure(results, placebo):
    """4-panel figure: rows = assets (XLE, USO), cols = signals (OFI_bvc, vpin_z).
    Each panel shows pre-event mean by session, with placebo line."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.0),
                              gridspec_kw=dict(hspace=0.50, wspace=0.30,
                                               left=0.09, right=0.97,
                                               top=0.93, bottom=0.11))
    sess_colors = {
        "regular":     LIGHT_BLUE,
        "after_hours": NAVY,
        "weekend":     ORANGE,
        "pre_market":  GREY,
    }
    sess_labels = {
        "regular":     "Regular\n(Mon-Fri 09:30-16:00)",
        "after_hours": "After-hours\n(Mon-Fri 16:00-04:00)",
        "weekend":     "Weekend\n(Sat-Sun)",
        "pre_market":  "Pre-market\n(Mon-Fri 04:00-09:30)",
    }

    for r, asset in enumerate(ASSETS):
        for c, sig in enumerate(SIGNALS):
            ax = axes[r, c]
            xs, ys, errs, ns, ps, colors = [], [], [], [], [], []
            for sess in SESSIONS:
                cell = results.get(asset, {}).get(sess)
                if cell is None or sig not in cell["signals"]:
                    xs.append(sess_labels[sess]); ys.append(np.nan)
                    errs.append(0); ns.append(0); ps.append(np.nan)
                    colors.append(sess_colors[sess])
                    continue
                s = cell["signals"][sig]
                xs.append(sess_labels[sess])
                ys.append(s["pre_mean"])
                errs.append(s["pre_se"])
                ns.append(cell["n"])
                ps.append(s["pre_boot_p"])
                colors.append(sess_colors[sess])
            x_pos = np.arange(len(SESSIONS))
            bars = ax.bar(x_pos, ys, yerr=errs, capsize=4, color=colors,
                          edgecolor="white", linewidth=0.6, width=0.78)
            # Placebo baseline as a horizontal line
            pb = placebo.get(asset, {}).get(sig, {}).get("pre_mean")
            if pb is not None:
                ax.axhline(pb, color=RED, linewidth=1.2, linestyle="--",
                           label=f"placebo baseline = {pb:+.4f}")
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
            ax.axhline(0, color="black", linewidth=0.4, alpha=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(xs, fontsize=8)
            ax.set_ylabel(f"pre-event mean ({sig})", fontsize=9)
            ax.set_title(f"{asset} - {sig} pre-event by session",
                         loc="left", weight="bold", fontsize=10.5)
            # Pad y limits BEFORE placing annotations so we have headroom
            ymin, ymax = ax.get_ylim()
            yspan = ymax - ymin
            ax.set_ylim(ymin - 0.18 * yspan, ymax + 0.18 * yspan)
            ymin, ymax = ax.get_ylim()
            yspan = ymax - ymin
            # Annotate n and p just inside the bar end, away from x-axis labels
            for i, (b, n, p) in enumerate(zip(bars, ns, ps)):
                h = b.get_height()
                if not np.isfinite(h):
                    continue
                err = errs[i] if np.isfinite(errs[i]) else 0
                # Place text just past the error bar tip, but never below the plot bottom
                if h >= 0:
                    y_text = h + err + 0.02 * yspan
                    va = "bottom"
                else:
                    y_text = h - err - 0.02 * yspan
                    va = "top"
                # Cap so the label stays in the plot area (top 12% / bottom 12%)
                y_text = min(y_text, ymax - 0.04 * yspan)
                y_text = max(y_text, ymin + 0.04 * yspan)
                txt = f"n={n}"
                if np.isfinite(p):
                    txt += f"\np={p:.3f}" if p >= 0.001 else f"\np<.001"
                ax.text(b.get_x() + b.get_width()/2, y_text, txt,
                        ha="center", va=va,
                        fontsize=7.5, color="black",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.65, pad=0.5))

    fig.suptitle("Pre-event order-flow on the oil complex, split by posting session "
                 "(oil-themed initiated posts)", fontsize=11, weight="bold", y=0.985)
    plt.savefig(FIGURES_DIR / "fig6_session_split.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig6_session_split.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  wrote {FIGURES_DIR / 'fig6_session_split.pdf'}")
    print(f"  wrote {FIGURES_DIR / 'fig6_session_split.png'}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("loading 60-day topic-tagged posts ...")
    posts = pd.read_parquet(POSTS_PARQUET).reset_index()
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    posts["created_at_et"] = posts["created_at"].dt.tz_convert(ET)
    posts["session"] = session_label(posts["created_at_et"])
    print(f"  {len(posts)} posts in window")

    oil = posts[posts["topic_energy_oil"] == True].copy()
    print(f"  {len(oil)} oil-themed posts")
    print("  oil posts by session:")
    print(oil["session"].value_counts().to_string())

    rng = np.random.default_rng(CFG["rng_seed"])
    results = {}

    for asset in ASSETS:
        print(f"\nprocessing {asset} ...")
        df = load_signals(asset)
        sigmas = precompute_initiation_sigma(df)
        results[asset] = {}
        for sess in SESSIONS:
            sub = oil[oil["session"] == sess]
            if len(sub) == 0:
                print(f"  [{sess}] no posts")
                continue
            windows = collect_windows(sub, df, sigmas)
            print(f"  [{sess}] n_oil_posts={len(sub)} -> n_initiated_windows={len(windows)}")
            res = summarise_window_set(windows, rng)
            results[asset][sess] = res

    placebo = load_placebo_baselines()

    # Print per-asset per-session table
    print("\n=== pre-event means by session ===")
    for asset in ASSETS:
        print(f"\n  --- {asset} ---")
        for sig in SIGNALS:
            print(f"    {sig}:")
            pb = placebo.get(asset, {}).get(sig, {}).get("pre_mean")
            print(f"      placebo (all timestamps): {pb:+.5f}" if pb is not None
                  else "      placebo: n/a")
            for sess in SESSIONS:
                cell = results[asset].get(sess)
                if cell is None or sig not in cell["signals"]:
                    print(f"      {sess:>12s}: (n<5, skipped)")
                    continue
                s = cell["signals"][sig]
                ratio = (abs(s["pre_mean"]) / abs(pb)) if pb else np.nan
                print(f"      {sess:>12s}: pre_mean={s['pre_mean']:+.5f} "
                      f"se={s['pre_se']:.5f}  n={cell['n']:3d}  "
                      f"boot_p={s['pre_boot_p']:.4f}  "
                      f"ratio_vs_placebo={ratio:.2f}")

    # Build figure
    print("\nbuilding figure ...")
    make_figure(results, placebo)

    # Write JSON
    out_path = RESULTS_DIR / "session_split.json"
    payload = {
        "method": ("Session-split event study on oil-themed initiated posts. "
                   "For each (asset, session, signal, window) cell we report the "
                   "pre/post mean across initiated oil-themed posts in that session, "
                   "the standard error, the bootstrap p-value vs zero, and the "
                   "asset's overall matched-placebo baseline for context."),
        "config": CFG,
        "sessions_tested": SESSIONS,
        "assets": ASSETS,
        "signals": SIGNALS,
        "n_oil_posts_by_session": oil["session"].value_counts().to_dict(),
        "results": results,
        "placebo_baselines": placebo,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
