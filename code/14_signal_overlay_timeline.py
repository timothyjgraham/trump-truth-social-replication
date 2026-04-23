#!/usr/bin/env python3
"""
14_signal_overlay_timeline.py
=============================

Matt's first chart ask (23 Apr email): "overlay all the times there was a
signal of pre-event trading pressure" so a reader can see at a glance,
on a single page, where in the 73-day window the 81 triggered events sit
and how clustered they are.

Builds a 2-row figure:
  Row 1: USO close-to-close price across the window, with a vertical
         tick at every triggered oil-themed event (vpin_z > 0.5
         pre-event), coloured by session.
  Row 2: XLE close-to-close price across the window, same overlay.
A small calendar strip below shows the 81 events as a strip of dots
across the date axis, again coloured by session, so the clustering
pattern is obvious even when the price line is busy.

Outputs:
  report/figures/fig7_signal_overlay.{pdf,png}
  data/results/signal_overlay_events.csv      (one row per triggered event)
"""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from _paths import (
    POSTS_PARQUET, SIGNALS_5M, RESULTS_DIR, FIGURES_DIR
)

ET = ZoneInfo("America/New_York")

NAVY        = "#1f3b6e"
LIGHT_BLUE  = "#7aa9d6"
GREY        = "#9aa1aa"
ORANGE      = "#d97c2a"
RED         = "#b73a3a"
GREEN       = "#3a7d5a"

SESSION_COLOR = {
    "regular":     LIGHT_BLUE,
    "after_hours": NAVY,
    "weekend":     ORANGE,
    "pre_market":  GREY,
}

# Trigger threshold matches Section 6 / Table dollar_upper_bound:
#   triggered = oil-themed AND pre_event vpin_z > 0.5
TRIGGER_VPINZ = 0.5


def session_label(dt: pd.Series) -> pd.Series:
    """Match 11_posting_patterns.py / 13_session_split_event_study.py."""
    out = pd.Series(["regular"] * len(dt), index=dt.index)
    is_weekend = dt.dt.weekday >= 5
    out[is_weekend] = "weekend"
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    pre = (~is_weekend) & (minute_of_day >= 4*60) & (minute_of_day < 9*60 + 30)
    after = (~is_weekend) & ((minute_of_day >= 16*60) | (minute_of_day < 4*60))
    out[pre] = "pre_market"
    out[after] = "after_hours"
    return out


def load_signals(asset):
    df = pd.read_parquet(SIGNALS_5M / f"{asset}.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def nearest_bar_index(bar_index, ts):
    pos = bar_index.searchsorted(ts, side="right") - 1
    return int(pos) if pos >= 0 else -1


def gather_pre_window(signals, t_idx, pre_bars):
    lo, hi = t_idx - pre_bars, t_idx + 1
    if lo < 0 or hi > len(signals):
        return None
    return signals.iloc[lo:hi]


def main():
    print("loading 60-day topic-tagged posts ...")
    posts = pd.read_parquet(POSTS_PARQUET).reset_index()
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    posts["created_at_et"] = posts["created_at"].dt.tz_convert(ET)
    posts["session"] = session_label(posts["created_at_et"])
    print(f"  {len(posts)} posts in window")

    oil = posts[posts["topic_energy_oil"] == True].copy()
    print(f"  {len(oil)} oil-themed posts")

    # Use the canonical event list from §6 dollar_upper_bound (n=81 triggered)
    print("\nloading canonical event list from dollar_upper_bound_uso_events.csv ...")
    ev = pd.read_csv(RESULTS_DIR / "dollar_upper_bound_uso_events.csv")
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, format="ISO8601")
    triggered = ev[ev["pre_vpinz"] > TRIGGER_VPINZ].copy()
    triggered["ts_et"] = triggered["ts"].dt.tz_convert(ET)
    triggered["session"] = session_label(triggered["ts_et"])
    triggered = triggered.sort_values("ts_et").reset_index(drop=True)
    triggered = triggered.rename(columns={"pre_vpinz": "vpinz_pre"})
    print(f"  {len(triggered)} triggered events (pre_vpinz > {TRIGGER_VPINZ})")
    print("  by session:")
    print(triggered["session"].value_counts().to_string())

    # Save event list for the report
    out_csv = RESULTS_DIR / "signal_overlay_events.csv"
    triggered.to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv}")

    # ─── Figure ────────────────────────────────────────────────────────────
    print("\nbuilding figure ...")
    uso = load_signals("USO")
    xle = load_signals("XLE")

    # Determine the window from posts
    t_min = posts["created_at_et"].min().normalize()
    t_max = posts["created_at_et"].max().normalize() + pd.Timedelta(days=1)

    # Slice price series to window and restrict to regular trading hours
    # (Mon-Fri 09:30-16:00 ET) so we don't draw long flat lines across
    # weekends and overnight gaps.
    def slice_window(df):
        d = df.copy()
        d.index = d.index.tz_convert(ET)
        d = d.loc[t_min:t_max]
        is_weekday = d.index.weekday < 5
        minute_of_day = d.index.hour * 60 + d.index.minute
        is_rth = (minute_of_day >= 9*60+30) & (minute_of_day < 16*60)
        d = d[is_weekday & is_rth]
        return d

    uso_w = slice_window(uso)
    xle_w = slice_window(xle)

    fig = plt.figure(figsize=(12, 9.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.5, 3.5, 1.2],
                          hspace=0.32, left=0.08, right=0.97,
                          top=0.93, bottom=0.10)

    ax_uso = fig.add_subplot(gs[0])
    ax_xle = fig.add_subplot(gs[1], sharex=ax_uso)
    ax_strip = fig.add_subplot(gs[2], sharex=ax_uso)

    # ── USO panel ──
    ax_uso.plot(uso_w.index, uso_w["Close"], color=NAVY, linewidth=0.9, alpha=0.85)
    ax_uso.set_ylabel("USO close ($)", fontsize=10)
    ax_uso.set_title("USO (United States Oil Fund)  — close price with triggered oil-themed posts overlaid",
                     loc="left", weight="bold", fontsize=10.5)
    ax_uso.grid(axis="y", alpha=0.25)

    # ── XLE panel ──
    ax_xle.plot(xle_w.index, xle_w["Close"], color=NAVY, linewidth=0.9, alpha=0.85)
    ax_xle.set_ylabel("XLE close ($)", fontsize=10)
    ax_xle.set_title("XLE (Energy Select Sector SPDR)  — close price with triggered oil-themed posts overlaid",
                     loc="left", weight="bold", fontsize=10.5)
    ax_xle.grid(axis="y", alpha=0.25)

    # ── Vertical event markers on USO + XLE ──
    for sess, color in SESSION_COLOR.items():
        sub = triggered[triggered["session"] == sess]
        if len(sub) == 0:
            continue
        for ts in sub["ts_et"]:
            ax_uso.axvline(ts, color=color, alpha=0.45, linewidth=0.8, zorder=0)
            ax_xle.axvline(ts, color=color, alpha=0.45, linewidth=0.8, zorder=0)

    # ── Calendar strip (bottom) ──
    ax_strip.set_facecolor("#f7f7f8")
    for sess, color in SESSION_COLOR.items():
        sub = triggered[triggered["session"] == sess]
        if len(sub) == 0:
            continue
        ax_strip.scatter(sub["ts_et"],
                         np.full(len(sub), 0.5),
                         color=color, s=40, alpha=0.85,
                         edgecolor="white", linewidth=0.5,
                         label=f"{sess.replace('_',' ')} (n={len(sub)})")
    ax_strip.set_ylim(0, 1)
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("Triggered\nevents", fontsize=9, rotation=0,
                        labelpad=42, ha="right", va="center")
    ax_strip.grid(axis="x", alpha=0.25)
    ax_strip.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45),
                    ncol=4, fontsize=8.5, frameon=False)

    # X-axis formatting on shared axis
    ax_strip.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax_strip.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax_strip.get_xticklabels(), rotation=0, ha="center", fontsize=8.5)
    plt.setp(ax_uso.get_xticklabels(), visible=False)
    plt.setp(ax_xle.get_xticklabels(), visible=False)

    fig.suptitle(f"Where the {len(triggered)} triggered oil-themed events sit in the 73-day window",
                 fontsize=11.5, weight="bold", y=0.985)

    out_pdf = FIGURES_DIR / "fig7_signal_overlay.pdf"
    out_png = FIGURES_DIR / "fig7_signal_overlay.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
