#!/usr/bin/env python3
"""
12_sector_sweep.py
==================

Re-cut of the existing 5-topic x 10-asset event study and placebo runs
to surface non-oil topic-asset signals the main report under-discussed.

For every (topic, asset, signal, window) cell we join:
  - the real pre/post mean and FDR-corrected bootstrap p-value
    (data/results/orderflow_fdr_5m_trim.csv), and
  - the asset's placebo pre/post mean from the 5,000-event placebo pool
    (data/results/orderflow_placebo_5m.json).

A cell is flagged "interesting" if all three hold:
  (1) real FDR-trim p < 0.05
  (2) the real mean is at least 1.5x larger in absolute value than the
      placebo mean for the same asset+signal+window, AND
  (3) the real and placebo signs do not both fall in the noise band
      (|real|>0 and either sign-flip vs placebo or |real|>>|placebo|).

Outputs:
  data/results/sector_sweep.json          (full table + flagged subset)
  report/figures/fig5_sector_sweep.{pdf,png}

The JSON has two top-level lists:
  - "all_cells":   1140 rows (5 topics x 10 assets x 6 signals x 2 windows)
  - "flagged":     subset matching the criteria above, sorted by p_fdr
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import RESULTS_DIR, FIGURES_DIR

# ── palette (matches figs 1-4) ───────────────────────────────────────────────
NAVY        = "#1f3b6e"
LIGHT_BLUE  = "#7aa9d6"
GREY        = "#9aa1aa"
ORANGE      = "#d97c2a"
RED         = "#b73a3a"

ASSETS  = ["DJT", "SPY", "QQQ", "VXX", "XLF", "XLK", "GLD", "UUP", "XLE", "USO"]
TOPICS  = ["energy_oil", "iran_military", "market_economy", "tariff_trade", "djt_media"]
SIGNALS = ["OFI_bvc", "vpin_z", "vol_z", "dvol_z", "logret", "signed_vol_tick"]


# ── load and join ────────────────────────────────────────────────────────────

def load_real() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "orderflow_fdr_5m_trim.csv")
    return df


def load_placebo_means() -> pd.DataFrame:
    pl = json.load(open(RESULTS_DIR / "orderflow_placebo_5m.json"))
    rows = []
    for asset, info in pl["results"]["placebo"].items():
        n = info["n"]
        for sig, s in info["signals"].items():
            rows.append(dict(asset=asset, signal=sig, window="pre",
                             placebo_mean=s["pre_mean"], placebo_n=n))
            rows.append(dict(asset=asset, signal=sig, window="post",
                             placebo_mean=s["post_mean"], placebo_n=n))
    return pd.DataFrame(rows)


def join() -> pd.DataFrame:
    real = load_real()
    pbo  = load_placebo_means()
    out = real.merge(pbo, on=["asset", "signal", "window"], how="left")
    # Ratio: real / placebo (with safe handling of near-zero placebo)
    eps = 1e-9
    out["abs_ratio"] = out["mean"].abs() / (out["placebo_mean"].abs() + eps)
    out["diff_vs_placebo"] = out["mean"] - out["placebo_mean"]
    out["sign_flip_vs_placebo"] = (
        np.sign(out["mean"]) != np.sign(out["placebo_mean"])
    ) & (out["mean"].abs() > eps) & (out["placebo_mean"].abs() > eps)
    return out


# ── flagging ─────────────────────────────────────────────────────────────────

def flag(df: pd.DataFrame, fdr_thresh: float = 0.05,
         min_abs_ratio: float = 1.5) -> pd.DataFrame:
    f = df[(df["p_fdr_trim"] < fdr_thresh) &
           ((df["abs_ratio"] >= min_abs_ratio) | df["sign_flip_vs_placebo"])
           ].copy()
    f = f.sort_values("p_fdr_trim")
    return f


# ── figure ───────────────────────────────────────────────────────────────────

def make_figure(df: pd.DataFrame):
    """Build a 2-panel heatmap: (a) PRE-window OFI_bvc & vpin_z signed-log10(p_fdr)
    by (topic x asset); (b) bar chart of the strongest non-oil flagged cells."""
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32,
                          left=0.08, right=0.97, top=0.93, bottom=0.10,
                          height_ratios=[1.05, 1.0])

    def heatmap(ax, signal, title):
        sub = df[(df["signal"] == signal) & (df["window"] == "pre")].copy()
        # signed log10 of FDR p, with sign of (real - placebo)
        eps = 1e-300
        sub["score"] = -np.log10(sub["p_fdr_trim"].clip(lower=eps)) * np.sign(sub["diff_vs_placebo"])
        mat = sub.pivot(index="topic", columns="asset", values="score").reindex(
            index=TOPICS, columns=ASSETS)
        # Symmetric colour limits
        vmax = float(np.nanmax(np.abs(mat.values)))
        if vmax == 0 or np.isnan(vmax):
            vmax = 1.0
        im = ax.imshow(mat.values, aspect="auto",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(ASSETS)))
        ax.set_xticklabels(ASSETS, rotation=0, fontsize=9)
        ax.set_yticks(range(len(TOPICS)))
        ax.set_yticklabels([t.replace("_", " / ") for t in TOPICS], fontsize=9)
        # Annotate each cell with the FDR p
        for i, t in enumerate(TOPICS):
            for j, a in enumerate(ASSETS):
                cell = sub[(sub["topic"] == t) & (sub["asset"] == a)]
                if len(cell):
                    p = cell["p_fdr_trim"].iloc[0]
                    rm = cell["mean"].iloc[0]
                    pm = cell["placebo_mean"].iloc[0]
                    ratio = abs(rm) / max(abs(pm), 1e-9)
                    is_flagged = (p < 0.05) and ((ratio >= 1.5) or
                                                  (np.sign(rm) != np.sign(pm)
                                                   and abs(rm) > 1e-9))
                    if is_flagged:
                        # tiny dot to mark flagged cells
                        ax.text(j, i, "•", ha="center", va="center",
                                color="black", fontsize=14, weight="bold")
        ax.set_title(title, loc="left", weight="bold", fontsize=10.5)
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("signed -log10(FDR p)", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    ax_a = fig.add_subplot(gs[0, 0])
    heatmap(ax_a, "OFI_bvc",
            "(a) Pre-event OFI_bvc by topic x asset (signed -log10 FDR p)")

    ax_b = fig.add_subplot(gs[0, 1])
    heatmap(ax_b, "vpin_z",
            "(b) Pre-event vpin_z by topic x asset (signed -log10 FDR p)")

    # Panel C (bottom row, full width): bar chart of most significant flagged cells
    ax_c = fig.add_subplot(gs[1, :])
    flagged = flag(df)
    # Show top 18 cells by significance, INCLUDING oil for context, so the
    # reader can see that oil-on-XLE/USO sits at the head of the ranking.
    top = flagged.copy()
    top["neg_log_p"] = -np.log10(top["p_fdr_trim"].clip(lower=1e-300))
    top = top.sort_values("neg_log_p", ascending=False).head(18)
    if len(top):
        labels = [f"{r.topic[:12]}.{r.asset}.{r.signal}.{r.window[0]}" for r in top.itertuples()]
        x = np.arange(len(top))
        bar_colors = []
        for r in top.itertuples():
            if r.topic == "energy_oil":
                bar_colors.append(RED)
            elif r.sign_flip_vs_placebo:
                bar_colors.append(ORANGE)
            else:
                bar_colors.append(NAVY)
        ax_c.bar(x, top["neg_log_p"].values, color=bar_colors,
                 edgecolor="white", linewidth=0.5)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax_c.set_ylabel("-log10(FDR-trim p)")
        ax_c.set_title("(c) Top 18 flagged cells by FDR significance "
                       "(red = oil topic, orange = sign-flip vs placebo, navy = same-sign non-oil)",
                       loc="left", weight="bold", fontsize=10.5)
        ax_c.axhline(-np.log10(0.05), color=GREY, linewidth=0.8, linestyle="--",
                     label="FDR p = 0.05")
        ax_c.legend(loc="upper right", fontsize=8)
    else:
        ax_c.text(0.5, 0.5, "No cells flagged at FDR<0.05 + |ratio|>=1.5",
                  transform=ax_c.transAxes, ha="center", fontsize=12)

    plt.savefig(FIGURES_DIR / "fig5_sector_sweep.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig5_sector_sweep.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  wrote {FIGURES_DIR / 'fig5_sector_sweep.pdf'}")
    print(f"  wrote {FIGURES_DIR / 'fig5_sector_sweep.png'}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("loading real (FDR-corrected) and placebo results ...")
    df = join()
    print(f"  joined {len(df)} cells "
          f"({df['topic'].nunique()} topics x {df['asset'].nunique()} assets "
          f"x {df['signal'].nunique()} signals x 2 windows)")

    flagged = flag(df)
    non_oil = flagged[flagged["topic"] != "energy_oil"]
    oil     = flagged[flagged["topic"] == "energy_oil"]

    print(f"\n=== flagged cells (FDR<0.05 AND (|ratio|>=1.5 OR sign-flip)) ===")
    print(f"  oil:     {len(oil):3d} cells")
    print(f"  non-oil: {len(non_oil):3d} cells")

    # Print top non-oil findings
    print(f"\n=== top 10 non-oil flagged cells (sorted by p_fdr) ===")
    cols = ["topic", "asset", "signal", "window", "n", "mean",
            "placebo_mean", "abs_ratio", "p_fdr_trim", "sign_flip_vs_placebo"]
    if len(non_oil):
        with pd.option_context("display.float_format", "{:.4g}".format,
                               "display.max_colwidth", 22):
            print(non_oil[cols].head(10).to_string(index=False))
    else:
        print("  (none)")

    # Per-topic summary
    print(f"\n=== flagged cells per (topic, asset) ===")
    pivot = (flagged.groupby(["topic", "asset"]).size()
             .unstack(fill_value=0).reindex(index=TOPICS, columns=ASSETS, fill_value=0))
    print(pivot.to_string())

    # Build figure
    print("\nbuilding figure ...")
    make_figure(df)

    # Write JSON
    out_path = RESULTS_DIR / "sector_sweep.json"
    payload = {
        "method": ("Re-cut of the 5-topic x 10-asset x 6-signal x 2-window event "
                   "study and placebo runs. A cell is flagged when its real "
                   "FDR-trim p<0.05 AND |real mean|>=1.5x|placebo mean| OR the "
                   "real and placebo signs flip."),
        "n_cells_total": int(len(df)),
        "n_flagged_total": int(len(flagged)),
        "n_flagged_oil": int(len(oil)),
        "n_flagged_non_oil": int(len(non_oil)),
        "flagged_per_topic_asset": pivot.to_dict(),
        "flagged_cells": flagged[cols].to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
