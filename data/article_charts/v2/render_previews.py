#!/usr/bin/env python3
"""
render_previews.py - generate sanity-check PNGs from the build outputs.

Produces:
  previews/falsification_publication_frame.png  - the recommended publication chart
  previews/chart1v2_zerovol_compare.png         - all-bars vs nonzero-only median
  previews/chart2alt_word_compare.png           - broad vs strict word frequencies
  previews/burst13_extended.png                 - burst 13 in the wider window

Run after build.py.
"""
from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR
DEFAULT_OUT  = SCRIPT_DIR / "previews"


def render_falsification_publication(data: Path, out: Path) -> None:
    ts_env = pd.read_csv(data / "chart1v2_timeshift_envelope.csv")
    pub = ts_env[(ts_env["minute_offset"] >= -60) & (ts_env["minute_offset"] <= 60)].copy()

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(pub["minute_offset"], pub["real_median"],
            color="crimson", lw=3, label="Real-event windows", zorder=3)
    ax.fill_between(pub["minute_offset"], pub["real_p25"], pub["real_p75"],
                     color="crimson", alpha=0.18, zorder=1)

    ax.plot(pub["minute_offset"], pub["shifted_median"],
            color="seagreen", lw=3, label="Same hour 24h later", zorder=3)
    ax.fill_between(pub["minute_offset"], pub["shifted_p25"], pub["shifted_p75"],
                     color="seagreen", alpha=0.18, zorder=1)

    ax.axvline(0, color="black", linestyle="--", lw=1, alpha=0.6, zorder=2)
    ax.set_xlabel("Minutes from post", fontsize=11)
    ax.set_ylabel("|% deviation from price at t=0|  (median across 15 bursts)", fontsize=11)
    ax.set_title("USO price movement: real Trump-post windows vs +24h-shifted controls\n",
                  fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, max(pub["real_p75"].max(), pub["shifted_p75"].max()) * 1.2)
    ax.text(0, ax.get_ylim()[1] * 0.95, "  Trump posts", fontsize=9, color="black",
            alpha=0.7, verticalalignment="top", zorder=4)

    footnote = (
        "n = 15 collapsed Trump Truth Social post-bursts. "
        "8 of 15 fired on weekends or outside trading hours and so contribute "
        "post-event data only — the median at negative offsets is computed over the "
        "7 RTH-anchored bursts (1, 2, 3, 4, 8, 13, 14)."
    )
    fig.text(0.05, 0.005, footnote, fontsize=8, color="#444", wrap=True, style="italic")

    callout = (
        "Statistical claim: not from this chart. The order-flow signal\n"
        "elevated around real posts (XLE OFI = +0.048) collapses around\n"
        "+24h shifted controls (−0.001, bootstrap p = 0.75).\n"
        "Source: report §3.4 / phase2b_timeshift.json"
    )
    ax.text(0.98, 0.97, callout, transform=ax.transAxes, fontsize=8.5, color="#222",
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f6f6f6",
                       edgecolor="#aaa", linewidth=0.7), zorder=5)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out / "falsification_publication_frame.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved: falsification_publication_frame.png")


def render_zerovol_compare(data: Path, out: Path) -> None:
    df = pd.read_csv(data / "chart1v2_event_window_5min.csv")
    env = pd.read_csv(data / "chart1v2_envelope_abs.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for bid, g in df.groupby("burst_id"):
        g = g.sort_values("minute_offset")
        ax.plot(g["minute_offset"], g["cum_pct_abs"], color="lightcoral", alpha=0.4, lw=0.7)
    ax.plot(env["minute_offset"], env["median"], color="crimson", lw=2.5)
    ax.set_title("All bars (incl. zero-volume / extended hours)")
    ax.set_xlabel("Minutes from post")
    ax.set_ylabel("|% deviation|")
    ax.grid(alpha=0.3)
    ax.axvline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 12.5)

    ax = axes[1]
    df_nonzero = df[~df["is_zero_vol"]]
    n_bursts_nonzero = df_nonzero["burst_id"].nunique()
    for bid, g in df_nonzero.groupby("burst_id"):
        g = g.sort_values("minute_offset")
        ax.plot(g["minute_offset"], g["cum_pct_abs"], color="lightcoral", alpha=0.5, lw=0.7)
    pivot = df_nonzero.pivot_table(index="minute_offset", columns="burst_id",
                                     values="cum_pct_abs", aggfunc="first")
    nz_med = pivot.median(axis=1)
    nz_p25 = pivot.quantile(0.25, axis=1)
    nz_p75 = pivot.quantile(0.75, axis=1)
    ax.plot(nz_med.index, nz_med, color="crimson", lw=2.5, label="Median (nonzero-vol)")
    ax.fill_between(nz_med.index, nz_p25, nz_p75, color="crimson", alpha=0.18)
    ax.set_title(f"Nonzero-volume bars only ({n_bursts_nonzero} bursts have data)")
    ax.set_xlabel("Minutes from post")
    ax.set_ylabel("|% deviation|")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axvline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 12.5)

    plt.tight_layout()
    plt.savefig(out / "chart1v2_zerovol_compare.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved: chart1v2_zerovol_compare.png")


def render_word_compare(data: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, prefix, title in [
        (axes[0], "chart2alt_broad",
         "Broad universe (173 oil-themed posts)\n— mostly endorsement spam"),
        (axes[1], "chart2alt_strict",
         "Strict universe (121 posts with explicit oil/Iran terms)\n— actual Iran/oil narrative"),
    ]:
        nodes = pd.read_csv(data / f"{prefix}_word_nodes.csv").sort_values(
            "post_count", ascending=True)
        colors = plt.cm.OrRd(np.linspace(0.4, 0.9, len(nodes)))
        ax.barh(nodes["label"], nodes["post_count"], color=colors)
        ax.set_xlabel("Post count containing word")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out / "chart2alt_word_compare.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved: chart2alt_word_compare.png")


def render_burst13(data: Path, out: Path) -> None:
    df5 = pd.read_csv(data / "chart1v2_event_window_5min.csv")
    df1 = pd.read_csv(data / "chart1v2_event_window_1min.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    b13_5 = df5[df5["burst_id"] == 13].sort_values("minute_offset")
    ax.plot(b13_5["minute_offset"], b13_5["cum_pct_abs"], color="crimson", lw=2,
            marker="o", ms=3, label="5-min real bars")
    b13_1 = df1[df1["burst_id"] == 13].sort_values("minute_offset")
    ax.plot(b13_1["minute_offset"], b13_1["cum_pct_abs"], color="lightcoral", lw=1,
            alpha=0.6, label="1-min interpolated")
    ax.axvline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Minutes from post (t=0)")
    ax.set_ylabel("|% deviation from t=0|")
    ax.set_title("Burst 13 (23 Mar Iran ceasefire RT, single post)\n"
                  "Widened window: covers Matt's extra ±2h pre-event request")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(-180, 180)
    plt.tight_layout()
    plt.savefig(out / "burst13_extended.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved: burst13_extended.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Folder containing the build outputs")
    parser.add_argument("--out",  type=Path, default=DEFAULT_OUT,
                        help="Folder for preview PNGs")
    args = parser.parse_args()

    data = args.data.resolve()
    out  = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {data}")
    print(f"Writing to:   {out}\n")

    render_falsification_publication(data, out)
    render_zerovol_compare(data, out)
    render_word_compare(data, out)
    render_burst13(data, out)


if __name__ == "__main__":
    main()
