#!/usr/bin/env python3
"""
render.py — Kobeissi-style annotated charts for the March 23 Iran ceasefire
morning, plus the Mar 23 vs Mar 24 falsification overlay.

Run after build.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd

SCRIPT_DIR  = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR
DEFAULT_OUT  = SCRIPT_DIR / "previews"


def render_kobeissi_style_overview(data: Path, out: Path) -> None:
    """Headline chart: USO price + volume on Mar 23 morning, with the same
    annotations Kobeissi used (plus Trump's posts marked).

    Kobeissi-style design: dark theme, bold annotations, arrows, volume bars,
    specific timestamp callouts. Honest about what's USO vs CL futures.
    """
    df = pd.read_csv(data / "mar23_uso_5min.csv")
    df["ts"] = pd.to_datetime(df["ts_et"].str.replace(" ET", ""), format="%Y-%m-%d %H:%M")

    fig = plt.figure(figsize=(14, 9), facecolor="#0d1117")
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.05], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax)
    ax.set_facecolor("#0d1117")
    ax_v.set_facecolor("#0d1117")

    # Candlesticks: colorblind-safe palette
    # Up bars: blue (#0072B2), Down bars: vermillion (#D55E00)
    UP_COLOR   = "#0072B2"
    DOWN_COLOR = "#D55E00"
    bar_width = pd.Timedelta(minutes=4)
    for _, r in df.iterrows():
        color = UP_COLOR if r["close"] >= r["open"] else DOWN_COLOR
        # Wick
        ax.plot([r["ts"], r["ts"]], [r["low"], r["high"]], color=color, lw=0.7, zorder=2)
        # Body
        body_low  = min(r["open"], r["close"])
        body_high = max(r["open"], r["close"])
        rect_h = max(body_high - body_low, 0.02)
        ax.add_patch(plt.Rectangle((r["ts"] - bar_width/2, body_low),
                                     bar_width, rect_h,
                                     facecolor=color, edgecolor=color, zorder=3))

    # Volume bars (will be flat through pre-market; jumps at 09:30 RTH open)
    for _, r in df.iterrows():
        color = UP_COLOR if r["close"] >= r["open"] else DOWN_COLOR
        ax_v.bar(r["ts"], r["volume"], width=bar_width, color=color, alpha=0.7,
                  edgecolor="none", zorder=3)

    # Y-axis
    ymin = df["low"].min() - 1
    ymax = df["high"].max() + 1
    ax.set_ylim(ymin, ymax)
    # Extend right edge to make room for the 10:29 RT annotation
    ax.set_xlim(df["ts"].min() - pd.Timedelta(minutes=10),
                 df["ts"].max() + pd.Timedelta(minutes=45))

    # Style axis
    for axis in [ax, ax_v]:
        axis.tick_params(colors="#c9d1d9")
        for spine in axis.spines.values():
            spine.set_color("#30363d")
        axis.grid(alpha=0.15, color="#30363d", linestyle="-", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.get_xticklabels(), visible=False)
    ax_v.set_xlabel("Time (ET) — March 23, 2026", color="#c9d1d9", fontsize=11)
    ax.set_ylabel("USO ETF close (USD)", color="#c9d1d9", fontsize=11)
    ax_v.set_ylabel("Vol", color="#c9d1d9", fontsize=10)
    fig.text(0.5, 0.94, "USO ETF · 5-min bars · NYSE Arca",
             ha="center", color="#c9d1d9", fontsize=12, fontweight="bold")

    # === Annotations ===
    def anno(x_str: str, label: str, y_offset_frac: float = 0.10,
             x_text_offset_min: int = 0, color: str = "#ef5350",
             fontweight: str = "bold") -> None:
        x = pd.Timestamp(f"2026-03-23 {x_str}")
        # Find nearest bar's high for vertical positioning
        if x in df["ts"].values:
            y = float(df.loc[df["ts"] == x, "high"].iloc[0])
        else:
            # Use linear interp on highs
            y = float(df["high"].mean())
        text_y = y + (ymax - ymin) * y_offset_frac
        text_x = x + pd.Timedelta(minutes=x_text_offset_min)
        ax.annotate(label, xy=(x, y), xytext=(text_x, text_y),
                     color=color, fontsize=10, fontweight=fontweight,
                     ha="center" if x_text_offset_min == 0 else ("right" if x_text_offset_min < 0 else "left"),
                     va="bottom",
                     arrowprops=dict(arrowstyle="->", color=color,
                                       lw=1.5, connectionstyle="arc3,rad=0"),
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                                edgecolor=color, linewidth=1.0),
                     zorder=10)

    # External markers (per Kobeissi) — orange for news annotations
    anno("04:50",
         "4:50 AM ET\nAxios reports a deal to end\nthe Iran war is imminent\n(per Kobeissi)",
         y_offset_frac=0.20, x_text_offset_min=10, color="#E69F00")

    # The big drop — vermillion for the down-move callout
    anno("07:05",
         "7:05 AM ET\nUSO drops from $122 to $113\nin a single 5-min bar\n(-8.0%)",
         y_offset_frac=-0.20, x_text_offset_min=-25, color="#D55E00")

    # Trump's original announcement — sky blue for Trump-post markers
    anno("07:25",
         "7:23 AM ET\nTrump posts:\n\"I AM PLEASED TO REPORT…\nIRAN… COMPLETE AND TOTAL RESOLUTION\"",
         y_offset_frac=0.50, x_text_offset_min=30, color="#56B4E9",
         fontweight="bold")

    # Trump's RT
    anno("10:30",
         "10:29 AM ET\nTrump RTs the same\nIran ceasefire post",
         y_offset_frac=0.18, x_text_offset_min=25, color="#56B4E9")

    # RTH open
    anno("09:30",
         "9:30 AM ET\nRegular hours open",
         y_offset_frac=0.05, x_text_offset_min=-25, color="#8b949e", fontweight="normal")

    # Friday close reference line
    fri_close = 121.44
    ax.axhline(fri_close, color="#8b949e", lw=1, linestyle="--", alpha=0.5, zorder=1)
    ax.text(df["ts"].iloc[-1], fri_close + 0.3, f"  Friday Mar 20 close: ${fri_close}",
            color="#8b949e", fontsize=8, ha="right", va="bottom")

    # Footer note
    fig.text(0.5, 0.02,
             "USO is the United States Oil Fund (ETF). The $920M / $125M short-position figures "
             "in the Kobeissi chart referenced light crude futures (CL on NYMEX), a different "
             "instrument that closely tracks USO. Pre-market USO bars (before 09:30 ET) trade on "
             "minimal volume — price discovery here reflects quote movement.",
             ha="center", color="#8b949e", fontsize=8.5, style="italic", wrap=True)

    plt.savefig(out / "mar23_overview.png", dpi=140, bbox_inches="tight",
                 facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved: mar23_overview.png")


def render_falsification_overlay(data: Path, out: Path) -> None:
    """Mar 23 (real news day) vs Mar 24 (next morning, no Iran ceasefire).

    Both lines plotted as % deviation from each day's 07:00 ET close. The Mar 23
    line drops sharply; the Mar 24 line should sit close to flat.
    """
    f = pd.read_csv(data / "mar23_vs_mar24_falsification.csv")

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="white")

    # Convert minute_of_day to a plottable HH:MM time
    f["et_dt"] = pd.to_datetime("2026-03-23 " + f["et_time"], format="%Y-%m-%d %H:%M")

    # Colorblind-safe palette: vermillion for event day, blue for control
    ax.plot(f["et_dt"], f["real_pct_vs_07_00"], color="#D55E00", lw=2.5,
            label="March 23, 2026 (Iran ceasefire announcement morning)", zorder=4,
            linestyle="-")
    ax.plot(f["et_dt"], f["shifted_pct_vs_07_00"], color="#0072B2", lw=2.5,
            label="March 24, 2026 (+24h shifted control)", zorder=4,
            linestyle="--")

    ax.axhline(0, color="black", lw=0.7, alpha=0.4)
    ax.axvline(pd.Timestamp("2026-03-23 07:00"), color="black",
                linestyle="--", lw=0.8, alpha=0.5, zorder=2)

    # Mark Trump posts on the real-day line
    for ts_label, content in [
        ("07:23", "Trump posts Iran\nceasefire announcement"),
        ("10:29", "Trump RTs the same\nIran ceasefire post"),
    ]:
        x = pd.Timestamp(f"2026-03-23 {ts_label}")
        # interp y on real series
        nearest = f.iloc[(f["et_dt"] - x).abs().argsort()[:1]]
        y = float(nearest["real_pct_vs_07_00"].iloc[0])
        ax.scatter(x, y, color="#2c3e50", s=80, zorder=6, edgecolor="white", linewidth=1.5)
        ax.annotate(content, xy=(x, y),
                     xytext=(x, y - 1.2),
                     color="#2c3e50", fontsize=9, ha="center", va="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
                                edgecolor="#bdc3c7", linewidth=0.8),
                     zorder=7)

    # Mark the Axios news per Kobeissi (orange marker, distinct from line colors)
    x_axios = pd.Timestamp("2026-03-23 04:50")
    nearest = f.iloc[(f["et_dt"] - x_axios).abs().argsort()[:1]]
    y_axios = float(nearest["real_pct_vs_07_00"].iloc[0])
    ax.scatter(x_axios, y_axios, color="#E69F00", s=80, zorder=6,
                edgecolor="white", linewidth=1.5, marker="^")
    ax.annotate("Axios reports deal\nimminent (per Kobeissi)",
                 xy=(x_axios, y_axios), xytext=(x_axios, y_axios + 1.5),
                 color="#B07000", fontsize=9, ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FBE9CC",
                            edgecolor="#E69F00", linewidth=0.8), zorder=7)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time (ET)", fontsize=11)
    ax.set_ylabel("USO % deviation from each day's 07:00 ET close", fontsize=11)
    ax.set_title("Iran ceasefire morning vs the next morning (24-hour shifted control)\n"
                  "USO ETF, 03:00–11:30 ET",
                  fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)

    fig.text(0.5, 0.005,
              "Both series anchored to 0% at 07:00 ET on each respective day. The Mar 23 line falls "
              "to roughly −9% before Trump's official announcement post (07:23 ET); the Mar 24 line "
              "stays within ±0.5% over the same hours — consistent with the news-driven move being "
              "specific to the Mar 23 ceasefire event, not a recurring daily pattern.",
              ha="center", color="#555", fontsize=8.5, style="italic", wrap=True)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out / "mar23_vs_mar24_falsification.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved: mar23_vs_mar24_falsification.png")


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

    render_kobeissi_style_overview(data, out)
    render_falsification_overlay(data, out)


if __name__ == "__main__":
    main()
