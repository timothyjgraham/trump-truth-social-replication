#!/usr/bin/env python3
"""
15_pnl_concentration_chart.py
=============================

Matt's second chart ask (23 Apr email): "a simplified version of figure 3a"
— a single page that makes the concentration of the $160M upper-bound
P&L immediately legible.

The full Fig 3 (`10_build_figures.py`) has three panels: leave-one-out
fragility, per-event P&L distribution, and the post-window decay. The
simplified version drops everything except the per-event P&L story:

  Top panel: All 81 triggered events sorted by per-event P&L
             (largest positive on the left, largest negative on the right).
             Bars are coloured green / red / grey. A cumulative line on
             a twin y-axis shows what fraction of the $160M total comes
             from how few events.

  Bottom panel: Same data, but collapsed into 'bursts' — clusters of
             posts that share a 5-minute pre/post window and therefore
             contribute identical per-burst P&L. This makes the point
             that the headline $160M comes from a tiny number of
             *independent* moments, not 81 separate trades.

Outputs:
  report/figures/fig8_pnl_concentration.{pdf,png}
  data/results/pnl_concentration_bursts.csv      (one row per burst)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import RESULTS_DIR, FIGURES_DIR


NAVY        = "#1f3b6e"
LIGHT_BLUE  = "#7aa9d6"
GREY        = "#9aa1aa"
ORANGE      = "#d97c2a"
RED         = "#b73a3a"
GREEN       = "#3a7d5a"

TRIGGER_VPINZ = 0.5
BURST_GAP_S = 600   # posts within 10 minutes are one "burst"


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(RESULTS_DIR / "dollar_upper_bound_uso_events.csv")
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, format="ISO8601")
    trig = ev[ev["pre_vpinz"] > TRIGGER_VPINZ].copy()
    trig = trig.sort_values("ts").reset_index(drop=True)
    return trig


def collapse_bursts(trig: pd.DataFrame, gap_s: int = BURST_GAP_S) -> pd.DataFrame:
    """Group consecutive posts whose timestamps are within `gap_s` seconds.

    Posts within the same burst share the same 5-minute pre/post window
    and therefore the same per-event P&L (this is the fragility shown in
    leave-one-out). Collapsing avoids double-counting independent
    information.
    """
    t = trig.sort_values("ts").reset_index(drop=True).copy()
    gap = t["ts"].diff().dt.total_seconds().fillna(99999)
    new_burst = (gap > gap_s).astype(int)
    t["burst_id"] = new_burst.cumsum()
    # Per-burst totals: posts within a burst can still span different
    # 5-minute pre/post windows, so we sum (not first × n) to recover
    # the headline aggregate exactly.
    bursts = (
        t.groupby("burst_id")
         .agg(
             ts_first=("ts", "min"),
             ts_last=("ts", "max"),
             n_posts=("post_id", "size"),
             pnl_total=("pnl", "sum"),
             pnl_max=("pnl", "max"),
             pnl_min=("pnl", "min"),
             initiated_any=("initiated", "max"),
         )
         .reset_index(drop=True)
    )
    bursts["pnl_per_post_mean"] = bursts["pnl_total"] / bursts["n_posts"]
    return bursts


def main() -> None:
    print("loading triggered events ...")
    trig = load_events()
    print(f"  {len(trig)} triggered events (pre_vpinz > {TRIGGER_VPINZ})")
    print(f"  sum P&L = ${trig['pnl'].sum()/1e6:,.2f}M")

    print("\ncollapsing into bursts ...")
    bursts = collapse_bursts(trig)
    print(f"  {len(bursts)} bursts")
    print(f"  burst sizes: {bursts['n_posts'].value_counts().sort_index().to_dict()}")
    bursts_out = RESULTS_DIR / "pnl_concentration_bursts.csv"
    bursts.to_csv(bursts_out, index=False)
    print(f"  wrote {bursts_out}")

    # ── Per-event sorted ────────────────────────────────────────────────
    ev_sorted = trig.sort_values("pnl", ascending=False).reset_index(drop=True)
    ev_sorted["cum_pnl"] = ev_sorted["pnl"].cumsum()
    total_pnl = ev_sorted["pnl"].sum()
    n_events = len(ev_sorted)

    # ── Per-burst sorted ────────────────────────────────────────────────
    # Sort by the burst's share of the headline ($M total contribution),
    # so the bars on the bottom panel reconcile with the $159.88M figure.
    b_sorted = bursts.sort_values("pnl_total", ascending=False).reset_index(drop=True)
    b_sorted["cum_pnl"] = b_sorted["pnl_total"].cumsum()
    n_bursts = len(b_sorted)
    total_burst_pnl = b_sorted["pnl_total"].sum()   # equals total_pnl

    # Concentration headlines
    top1_burst_pct = b_sorted["pnl_total"].iloc[0] / total_burst_pnl * 100
    top3_burst_pct = b_sorted["pnl_total"].head(3).sum() / total_burst_pnl * 100

    # ── Figure ──────────────────────────────────────────────────────────
    fig, (ax_ev, ax_b) = plt.subplots(
        2, 1, figsize=(11, 8.5),
        gridspec_kw=dict(hspace=0.55, left=0.09, right=0.93,
                         top=0.91, bottom=0.10),
    )

    def color_for(p):
        if p > 0: return GREEN
        if p < 0: return RED
        return GREY

    # ── Top panel: 81 individual events ────────────────────────────────
    x_ev = np.arange(n_events)
    colors_ev = [color_for(p) for p in ev_sorted["pnl"]]
    ax_ev.bar(x_ev, ev_sorted["pnl"] / 1e6, color=colors_ev,
              edgecolor="white", linewidth=0.3, width=0.9)
    ax_ev.axhline(0, color="black", linewidth=0.5)
    ax_ev.set_ylabel("Per-event P&L ($M)", fontsize=10)
    ax_ev.set_xlabel(f"{n_events} triggered oil-themed events, sorted by per-event P&L",
                     fontsize=9.5)
    ax_ev.set_title("a) All 81 triggered events — note the duplicate-bar plateaus from clustered posts",
                    loc="left", weight="bold", fontsize=10.5)
    ax_ev.grid(axis="y", alpha=0.25)
    ax_ev.set_xlim(-0.5, n_events - 0.5)

    # cumulative line on twin axis
    ax_ev2 = ax_ev.twinx()
    ax_ev2.plot(x_ev, ev_sorted["cum_pnl"] / total_pnl * 100,
                color=NAVY, linewidth=1.6, alpha=0.85)
    ax_ev2.set_ylabel("Cumulative % of total $M",
                      fontsize=9.5, color=NAVY)
    ax_ev2.tick_params(axis="y", labelcolor=NAVY)
    ax_ev2.set_ylim(0, 110)
    ax_ev2.axhline(100, color=NAVY, linewidth=0.4, linestyle=":", alpha=0.6)

    # Concentration annotations on top panel
    top10 = ev_sorted["pnl"].head(10).sum() / total_pnl * 100
    ax_ev.text(
        0.99, 0.04,
        f"Top 10 events  = {top10:.0f}% of ${total_pnl/1e6:,.0f}M\n"
        f"Top 20 events  = {ev_sorted['pnl'].head(20).sum()/total_pnl*100:.0f}%\n"
        f"Bottom 1 event = {ev_sorted['pnl'].iloc[-1]/1e6:,.1f}M (drag)",
        transform=ax_ev.transAxes, ha="right", va="bottom",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor=GREY,
                  boxstyle="round,pad=0.4", alpha=0.95),
    )

    # ── Bottom panel: collapsed to bursts ──────────────────────────────
    x_b = np.arange(n_bursts)
    colors_b = [color_for(p) for p in b_sorted["pnl_total"]]
    bar_labels = [f"{n}" for n in b_sorted["n_posts"]]
    bars_b = ax_b.bar(x_b, b_sorted["pnl_total"] / 1e6, color=colors_b,
                      edgecolor="white", linewidth=0.4, width=0.75)
    ax_b.axhline(0, color="black", linewidth=0.5)
    ax_b.set_ylabel("Burst contribution to headline ($M)", fontsize=10)
    ax_b.set_xlabel(f"{n_bursts} independent posting bursts (consecutive posts within 10 min collapsed); "
                    f"bar label = number of posts in burst",
                    fontsize=9.5)
    ax_b.set_title(r"b) Collapsed to independent bursts — one 14-min burst produces \$213M, "
                   r"negatives drag it back to \$160M",
                   loc="left", weight="bold", fontsize=10.5)
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.set_xlim(-0.5, n_bursts - 0.5)

    # Bar labels showing burst size
    ymin_b, ymax_b = ax_b.get_ylim()
    yspan_b = ymax_b - ymin_b
    for bar, n_posts in zip(bars_b, b_sorted["n_posts"]):
        h = bar.get_height()
        if h >= 0:
            y = h + 0.015 * yspan_b
            va = "bottom"
        else:
            y = h - 0.015 * yspan_b
            va = "top"
        ax_b.text(bar.get_x() + bar.get_width()/2, y, f"n={n_posts}",
                  ha="center", va=va, fontsize=7.5, color="black")

    ax_b2 = ax_b.twinx()
    ax_b2.plot(x_b, b_sorted["cum_pnl"] / total_burst_pnl * 100,
               color=NAVY, linewidth=1.8, alpha=0.85, marker="o", markersize=4)
    ax_b2.set_ylabel("Cumulative % of total $M",
                     fontsize=9.5, color=NAVY)
    ax_b2.tick_params(axis="y", labelcolor=NAVY)
    ax_b2.set_ylim(0, 110)
    ax_b2.axhline(100, color=NAVY, linewidth=0.4, linestyle=":", alpha=0.6)

    # Annotate the top burst
    top_burst = b_sorted.iloc[0]
    burst_dur = int((top_burst["ts_last"] - top_burst["ts_first"]).total_seconds())
    burst_min = burst_dur // 60
    burst_sec = burst_dur % 60
    ax_b.annotate(
        f"Apr 7 burst\n"
        f"({top_burst['n_posts']} posts in "
        f"{burst_min}m{burst_sec:02d}s)\n"
        f"= {top1_burst_pct:.0f}% of $160M\n"
        f"headline (all of\n"
        f"the positive total)",
        xy=(0, top_burst["pnl_total"] / 1e6),
        xytext=(2.5, top_burst["pnl_total"] / 1e6 * 0.55),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8),
        bbox=dict(facecolor="white", edgecolor=GREY,
                  boxstyle="round,pad=0.3", alpha=0.95),
    )

    worst = b_sorted.iloc[-1]
    ax_b.text(
        0.99, 0.06,
        f"Top 1 burst  = {top1_burst_pct:.0f}% of ${total_burst_pnl/1e6:,.0f}M\n"
        f"Top 3 bursts = {top3_burst_pct:.0f}%\n"
        f"Worst burst  = {worst['pnl_total']/1e6:+,.1f}M ({worst['n_posts']} posts)",
        transform=ax_b.transAxes, ha="right", va="bottom",
        fontsize=8.5, family="monospace",
        bbox=dict(facecolor="white", edgecolor=GREY,
                  boxstyle="round,pad=0.4", alpha=0.95),
    )

    fig.suptitle(
        f"Where the headline ${total_pnl/1e6:,.0f}M upper-bound comes from",
        fontsize=12.5, weight="bold", y=0.985,
    )

    out_pdf = FIGURES_DIR / "fig8_pnl_concentration.pdf"
    out_png = FIGURES_DIR / "fig8_pnl_concentration.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out_pdf}")
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
