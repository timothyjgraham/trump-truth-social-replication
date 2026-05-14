#!/usr/bin/env python3
"""
render.py - matched-format charts for the three case-study events.

Three USO ETF charts, identical layout / colour scheme / units / time-window.
Same Kobeissi-style dark theme used for the Mar 23 case study, applied to
Mar 4 and Apr 7 in matching format. Trump-post timestamps annotated on each.

Run after build.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR  = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR
DEFAULT_OUT  = SCRIPT_DIR / "previews"

# Colorblind-safe palette (Okabe-Ito)
UP_COLOR        = "#0072B2"   # blue
DOWN_COLOR      = "#D55E00"   # vermillion
TRUMP_POST_COL  = "#56B4E9"   # sky blue
SESSION_COL     = "#8b949e"   # neutral gray
DRAMA_COL       = "#D55E00"   # vermillion (drop callouts)


def render_event_chart(df: pd.DataFrame, meta: dict, out: Path) -> None:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts_et"].str.replace(" ET", ""), format="%Y-%m-%d %H:%M")

    fig = plt.figure(figsize=(14, 9), facecolor="#0d1117")
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.05], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax)
    ax.set_facecolor("#0d1117")
    ax_v.set_facecolor("#0d1117")

    # Candlesticks. For zero-volume (extended-hours) bars, yfinance's
    # high/low fields are unreliable single-tick quote anomalies; suppress
    # the wick on those bars so the chart reflects the executed-price
    # range honestly. RTH bars get full candles with wicks.
    bar_width = pd.Timedelta(minutes=4)
    for _, r in df.iterrows():
        color = UP_COLOR if r["close"] >= r["open"] else DOWN_COLOR
        body_low  = min(r["open"], r["close"])
        body_high = max(r["open"], r["close"])
        if r["is_zero_vol"]:
            # No wick - high/low aren't reliable on zero-volume bars
            wick_low, wick_high = body_low, body_high
        else:
            wick_low, wick_high = r["low"], r["high"]
        ax.plot([r["ts"], r["ts"]], [wick_low, wick_high], color=color, lw=0.7, zorder=2)
        rect_h = max(body_high - body_low, 0.02)
        ax.add_patch(plt.Rectangle((r["ts"] - bar_width/2, body_low),
                                     bar_width, rect_h,
                                     facecolor=color, edgecolor=color, zorder=3))

    # Volume bars
    for _, r in df.iterrows():
        color = UP_COLOR if r["close"] >= r["open"] else DOWN_COLOR
        ax_v.bar(r["ts"], r["volume"], width=bar_width, color=color, alpha=0.7,
                  edgecolor="none", zorder=3)

    # Ignore zero-volume highs/lows for y-axis sizing (they're quote anomalies)
    rth_or_real = df[~df["is_zero_vol"]]
    if len(rth_or_real) > 0:
        # Use real-trade range for axis sizing; fall back to body range for zero-vol bars
        body_lows  = pd.concat([rth_or_real["low"], df[["open", "close"]].min(axis=1)])
        body_highs = pd.concat([rth_or_real["high"], df[["open", "close"]].max(axis=1)])
    else:
        body_lows  = df[["open", "close"]].min(axis=1)
        body_highs = df[["open", "close"]].max(axis=1)
    y_low  = body_lows.min()
    y_high = body_highs.max()
    ymin = y_low  - max(0.5, (y_high - y_low) * 0.05)
    ymax = y_high + max(0.5, (y_high - y_low) * 0.10)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(df["ts"].min() - pd.Timedelta(minutes=10),
                 df["ts"].max() + pd.Timedelta(minutes=10))

    # Style axes
    for axis in [ax, ax_v]:
        axis.tick_params(colors="#c9d1d9")
        for spine in axis.spines.values():
            spine.set_color("#30363d")
        axis.grid(alpha=0.15, color="#30363d", linestyle="-", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.get_xticklabels(), visible=False)

    event_date_str = pd.Timestamp(meta["anchor_et"]).strftime("%B %-d, %Y")
    ax_v.set_xlabel(f"Time (ET) - {event_date_str}", color="#c9d1d9", fontsize=11)
    ax.set_ylabel("USO ETF close (USD)", color="#c9d1d9", fontsize=11)
    ax_v.set_ylabel("Vol", color="#c9d1d9", fontsize=10)

    # Title
    fig.text(0.5, 0.94,
              f"USO ETF . 5-min bars . NYSE Arca   |   {meta['event_label']}",
              ha="center", color="#c9d1d9", fontsize=12, fontweight="bold")

    # Annotate the Trump post (vertical line + label).
    # Use the actual post timestamp (from anchor_et) for the visual position,
    # NOT the 5-min anchor-bar floor — the post time is what readers will see
    # on Trump's Truth Social and is what should appear on the chart.
    post_ts = pd.Timestamp(meta["anchor_et"])  # e.g. "2026-03-04 15:09"
    ax.axvline(post_ts, color=TRUMP_POST_COL, linestyle="--", lw=1.5, alpha=0.8, zorder=4)
    post_text_short = (meta["post_text"][:90] + "...") if len(meta["post_text"]) > 90 else meta["post_text"]
    label = f"Trump post {post_ts.strftime('%H:%M')} ET\n\"{post_text_short}\""
    text_y = ymax - (ymax - ymin) * 0.05
    ax.text(post_ts, text_y, "  " + label, color=TRUMP_POST_COL, fontsize=8.5,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                       edgecolor=TRUMP_POST_COL, linewidth=0.9), zorder=10)

    # If RTH session boundaries fall inside the window, mark them
    rth_open  = pd.Timestamp(f"{post_ts.strftime('%Y-%m-%d')} 09:30")
    rth_close = pd.Timestamp(f"{post_ts.strftime('%Y-%m-%d')} 16:00")
    for boundary, label_text in [(rth_open, "9:30 ET\nRTH open"),
                                    (rth_close, "16:00 ET\nRTH close")]:
        if df["ts"].min() <= boundary <= df["ts"].max():
            ax.axvline(boundary, color=SESSION_COL, linestyle=":", lw=1, alpha=0.6, zorder=3)
            ax.text(boundary, ymin + (ymax - ymin) * 0.02, "  " + label_text,
                    color=SESSION_COL, fontsize=8, verticalalignment="bottom",
                    horizontalalignment="left", zorder=4)

    # Anchor-price reference line
    anchor_price = float(meta["anchor_price"])
    ax.axhline(anchor_price, color="#8b949e", lw=0.8, linestyle="--", alpha=0.4, zorder=1)
    ax.text(df["ts"].iloc[-1], anchor_price + (ymax - ymin) * 0.005,
            f" ${anchor_price:.2f} (close at t=0)",
            color="#8b949e", fontsize=8, ha="right", va="bottom")

    # Footer
    fig.text(0.5, 0.02,
              "USO is the United States Oil Fund (ETF). USO ETF does not trade in pre-market or after-hours; "
              "non-RTH bars carry zero volume and reflect quote movement on overnight crude futures rather than executed trades. "
              "Anchor (t=0) is the 5-min bar at or just before the Trump post.",
              ha="center", color="#8b949e", fontsize=8.5, style="italic", wrap=True)

    out_path = out / f"{meta['event_id']}.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                 facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved: {out_path.name}")


def render_three_panel_overlay(df_all: pd.DataFrame, md: pd.DataFrame, out: Path) -> None:
    """Bonus: a single panel showing the three events on one cum_pct_signed
    chart, x-axis = minute_offset, so the shapes are directly comparable."""
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="white")

    colors = {"mar4_venezuela_oil":   "#888888",
              "mar23_iran_ceasefire": "#D55E00",
              "apr7_indiana_burst":   "#0072B2"}
    labels = {"mar4_venezuela_oil":   "Mar 4 - Venezuela oil (\"normal signal\")",
              "mar23_iran_ceasefire": "Mar 23 - Iran ceasefire (\"pre-event signal\")",
              "apr7_indiana_burst":   "Apr 7 - Indiana endorsement burst (\"the massive one\")"}
    linestyles = {"mar4_venezuela_oil":   "-",
                  "mar23_iran_ceasefire": "-",
                  "apr7_indiana_burst":   "--"}

    for ev_id in ["mar4_venezuela_oil", "mar23_iran_ceasefire", "apr7_indiana_burst"]:
        sub = df_all[df_all["event_id"] == ev_id].sort_values("minute_offset")
        ax.plot(sub["minute_offset"], sub["cum_pct_signed"],
                color=colors[ev_id], lw=2.5, label=labels[ev_id],
                linestyle=linestyles[ev_id], zorder=4)

    ax.axvline(0, color="black", linestyle="--", lw=1, alpha=0.5)
    ax.axhline(0, color="black", lw=0.7, alpha=0.4)
    ax.set_xlabel("Minutes from Trump post (t=0)", fontsize=11)
    ax.set_ylabel("USO % deviation from anchor close", fontsize=11)
    ax.set_title("Three case-study events, cumulative % return aligned at t=0\n"
                  "Each line is USO's 5-min close as a percentage of the anchor (post-time) close",
                  fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=10, frameon=True)
    ax.grid(alpha=0.3)
    ax.text(0, ax.get_ylim()[1] * 0.95, "  Trump posts", fontsize=9, color="black",
             alpha=0.7, verticalalignment="top")

    fig.text(0.5, 0.005,
              "All three events anchored at 0% at the post-time bar. Direct comparison of how USO moved before "
              "and after each post in the same units. Apr 7's wider window post-event extends into after-hours "
              "where USO trades on zero volume - the price drift there reflects overnight repricing rather than "
              "intraday trades.",
              ha="center", color="#555", fontsize=8.5, style="italic", wrap=True)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out / "three_events_cum_pct_overlay.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved: three_events_cum_pct_overlay.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out",  type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = args.data.resolve()
    out  = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {data}")
    print(f"Writing to:   {out}\n")

    df_all = pd.read_csv(data / "three_events_uso_5min.csv")
    md     = pd.read_csv(data / "three_events_metadata.csv")

    for _, m in md.iterrows():
        sub = df_all[df_all["event_id"] == m["event_id"]]
        render_event_chart(sub, m.to_dict(), out)

    render_three_panel_overlay(df_all, md, out)


if __name__ == "__main__":
    main()
