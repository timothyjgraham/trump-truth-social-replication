#!/usr/bin/env python3
"""
10_build_figures.py
===================

Generate the three paper figures (PDF + PNG) from the cached results.

Figure 1 — XLE OFI_bvc:  real (n=165) vs matched placebo vs +24h time-shifted
Figure 2 — USO vpin_z:   real (n=147) vs matched placebo vs +24h time-shifted
Figure 3 — Dollar fragility:
            (a) per-event P&L scatter for the 81-event triggered slice
            (b) leave-one-out range of the $160M headline

Outputs to report/figures/
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import (
    POSTS_PARQUET, MINUTE_BARS_5M, SIGNALS_5M, LOO_JSON, FIGURES_DIR, ensure_dirs,
)

ensure_dirs()

NAVY = "#1f3b73"
LIGHT_BLUE = "#7fb1e0"
GREY = "#9aa0a6"
ORANGE = "#d96f32"
RED = "#b03030"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
})


def fig1_xle_ofi():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = ["Real event\n(n=165)", "Matched placebo\n(non-event timestamps)",
              "Time-shifted +24h\n(n=170)"]
    values = [0.0483, 0.0111, -0.0014]
    bars = ax.bar(labels, values, color=[NAVY, LIGHT_BLUE, GREY],
                  edgecolor="white", linewidth=1.2, width=0.62)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(r"Pre-event mean $\mathrm{OFI}_{\mathrm{bvc}}$  (30-min window)")
    ax.set_title("Pre-event order-flow imbalance on XLE around oil-themed posts",
                 loc="left", weight="bold")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + (0.002 if v >= 0 else -0.004),
                f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, weight="bold")
    ax.set_ylim(-0.015, 0.065)
    ax.text(0.99, 0.02, "FDR-adjusted q = 7.3e-11 (real vs placebo)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#444444", style="italic")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_xle_ofi.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig1_xle_ofi.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig2_uso_vpin():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = ["Real event\n(n=147)", "Matched placebo\n(non-event timestamps)",
              "Time-shifted +24h\n(n=159)"]
    values = [0.380, 0.056, -1.021]
    bars = ax.bar(labels, values, color=[NAVY, LIGHT_BLUE, GREY],
                  edgecolor="white", linewidth=1.2, width=0.62)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(r"Pre-event mean $\mathrm{vpin}_z$  (30-min window)")
    ax.set_title("Pre-event informed-trading intensity on USO around oil-themed posts",
                 loc="left", weight="bold")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + (0.04 if v >= 0 else -0.04),
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, weight="bold")
    ax.set_ylim(-1.25, 0.55)
    ax.annotate("Sign flip:  real-event\nelevation reverses under\nthe time-shift falsification",
                xy=(2, -1.021), xytext=(1.55, -0.55),
                fontsize=8.5, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=0.8))
    ax.text(0.99, 0.02, "FDR-adjusted q = 1.3e-05 (real vs placebo)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#444444", style="italic")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_uso_vpin.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig2_uso_vpin.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig3_dollar_fragility():
    with LOO_JSON.open() as f:
        loo = json.load(f)["conventions"]["A_production"]
    total = loo["total_sum_pnl_usd"]
    loo_min = loo["loo_sum_min_usd"]
    loo_max = loo["loo_sum_max_usd"]

    posts = pd.read_parquet(POSTS_PARQUET)
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    oil = posts[posts["topic_energy_oil"] == True].copy()

    bars = pd.read_parquet(MINUTE_BARS_5M / "USO.parquet")
    sigs = pd.read_parquet(SIGNALS_5M / "USO.parquet")
    bars.index = pd.to_datetime(bars.index, utc=True)
    sigs.index = pd.to_datetime(sigs.index, utc=True)

    PRE = POST = 12
    close = bars["Close"].astype(float)
    idx = bars.index
    pnls = []
    for _, p in oil.iterrows():
        pos = idx.searchsorted(p["created_at"])
        if pos < PRE or pos + POST - 1 >= len(idx):
            continue
        if sigs["vpin_z"].iloc[pos - PRE:pos].mean() <= 0.5:
            continue
        pre_sv = sigs["signed_vol_tick"].iloc[pos - PRE:pos].sum()
        entry = close.iloc[pos - 1]
        exitp = close.iloc[pos + POST - 1]
        pnls.append(pre_sv * (exitp - entry))
    pnls = np.array(pnls)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    sorted_pnl = np.sort(pnls)
    xs = np.arange(len(sorted_pnl))
    colors = [RED if abs(v)/abs(total) > 0.10 else
              ORANGE if abs(v)/abs(total) > 0.04 else
              NAVY for v in sorted_pnl]
    ax.scatter(xs, sorted_pnl/1e6, c=colors, s=22, alpha=0.85,
               edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Event rank (low to high P&L)")
    ax.set_ylabel("Per-event gross P&L (USD, millions)")
    ax.set_title("(a) Per-event P&L distribution, 81 triggered events",
                 loc="left", weight="bold", fontsize=10.5)
    worst_idx = int(np.argmin(sorted_pnl))
    ax.annotate(f"23 Mar 2026 post\n-${abs(sorted_pnl[worst_idx])/1e6:.1f}M\n"
                f"({100*abs(sorted_pnl[worst_idx])/abs(total):.1f}% of total)",
                xy=(worst_idx, sorted_pnl[worst_idx]/1e6),
                xytext=(worst_idx + 12, sorted_pnl[worst_idx]/1e6 + 8),
                fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=0.7))
    dup_value = 9.106
    dup_idxs = [i for i, v in enumerate(sorted_pnl) if abs(v/1e6 - dup_value) < 0.01]
    if dup_idxs:
        ax.annotate("4 near-duplicate posts\n7 Apr 2026, $9.1M each\n($36.4M total)",
                    xy=(dup_idxs[len(dup_idxs)//2], dup_value),
                    xytext=(dup_idxs[0] - 30, dup_value + 9),
                    fontsize=8.5, color=ORANGE,
                    arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=0.7))

    ax = axes[1]
    labels = ["LOO min\n(drop top +ve)", "Headline\nall 81 events", "LOO max\n(drop top -ve)"]
    values = [loo_min/1e6, total/1e6, loo_max/1e6]
    bars = ax.bar(labels, values, color=[GREY, NAVY, GREY],
                  edgecolor="white", linewidth=1.2, width=0.62)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Aggregate gross P&L (USD, millions)")
    ax.set_title("(b) Leave-one-out range", loc="left", weight="bold", fontsize=10.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 5, f"${v:.0f}M",
                ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_ylim(0, max(values) * 1.18)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_dollar_fragility.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig3_dollar_fragility.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fig1_xle_ofi()
    fig2_uso_vpin()
    fig3_dollar_fragility()
    print(f"saved figures to {FIGURES_DIR}")
    for f in sorted(FIGURES_DIR.iterdir()):
        print(f"  {f.name}")
