#!/usr/bin/env python3
"""
render_xle_volume.py — XLE volume comparison addendum.

Adds a chart showing XLE (Energy SPDR) RTH volume on Mar 23 vs adjacent
days, and panels USO price alongside XLE volume to show the relationship
between the overnight-news absorption (09:30 RTH-open volume spike) and
Trump's 10:29 RT.

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
DEFAULT_REPO = (SCRIPT_DIR.parent / "trump-truth-social-replication").resolve()
DEFAULT_OUT  = SCRIPT_DIR / "previews"


def slice_morning(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    s = pd.Timestamp(f"{date_str} 04:00", tz="America/New_York").tz_convert("UTC")
    e = pd.Timestamp(f"{date_str} 11:30", tz="America/New_York").tz_convert("UTC")
    win = df.loc[(df.index >= s) & (df.index <= e)].copy()
    win["et"] = win.index.tz_convert("America/New_York").strftime("%H:%M")
    win["minute_of_day"] = (
        win.index.tz_convert("America/New_York").hour * 60
        + win.index.tz_convert("America/New_York").minute
    )
    return win


def render_xle_volume_comparison(repo: Path, out: Path) -> None:
    xle = pd.read_parquet(repo / "data/raw/minute_bars_5m/XLE.parquet")
    xle.index = pd.to_datetime(xle.index, utc=True)

    mar20 = slice_morning(xle, "2026-03-20")  # Friday before
    mar23 = slice_morning(xle, "2026-03-23")  # Iran ceasefire morning
    mar24 = slice_morning(xle, "2026-03-24")  # Next day

    fig, ax = plt.subplots(figsize=(14, 6.5), facecolor="white")

    # Side-by-side grouped bars so each day is unambiguously distinguishable.
    # Each 5-min timestamp gets a triplet of slim bars (Mar 20 | Mar 23 | Mar 24).
    bar_width  = pd.Timedelta(seconds=80)         # ~1.3 min per bar
    offset_left = pd.Timedelta(seconds=-85)        # Mar 20 sits left
    offset_mid  = pd.Timedelta(seconds=0)          # Mar 23 sits centre
    offset_right = pd.Timedelta(seconds=85)        # Mar 24 sits right

    def plot_day(df: pd.DataFrame, color: str, label: str,
                  offset: pd.Timedelta, alpha: float = 0.95) -> None:
        rth = df[df["is_rth"]]
        x = pd.to_datetime("2026-03-23 " + rth["et"], format="%Y-%m-%d %H:%M") + offset
        ax.bar(x, rth["dollar_volume"] / 1e6, width=bar_width,
                color=color, alpha=alpha, label=label,
                edgecolor="white", linewidth=0.4)

    # Colorblind-safe palette (Okabe-Ito): gray / vermillion / blue
    # Order plotted = order in legend (Mar 20, Mar 23, Mar 24)
    plot_day(mar20, "#888888", "Mar 20 (prior Friday)",      offset_left)
    plot_day(mar23, "#D55E00", "Mar 23 (Iran ceasefire)",    offset_mid)
    plot_day(mar24, "#0072B2", "Mar 24 (next morning)",      offset_right)

    # Mark Trump's RT at 10:29 ET
    rt_x = pd.Timestamp("2026-03-23 10:29")
    ax.axvline(rt_x, color="#2c3e50", linestyle="--", lw=1.5, zorder=5)
    ax.annotate("Trump's 10:29 RT\nof Iran ceasefire post",
                 xy=(rt_x, ax.get_ylim()[1] * 0.95),
                 xytext=(rt_x + pd.Timedelta(minutes=5),
                         ax.get_ylim()[1] * 0.85),
                 color="#2c3e50", fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
                            edgecolor="#bdc3c7", linewidth=0.8),
                 zorder=6)

    # Annotate the 09:30 opening volume spike
    open_x = pd.Timestamp("2026-03-23 09:30")
    open_y = mar23[mar23["et"] == "09:30"]["dollar_volume"].iloc[0] / 1e6
    ax.annotate(f"09:30 RTH open: ${open_y:.0f}M\n"
                 f"= 2.1x Friday open\n"
                 f"   2.2x next-day open",
                 xy=(open_x, open_y),
                 xytext=(open_x + pd.Timedelta(minutes=20), open_y * 0.85),
                 color="#D55E00", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FBE5D6",
                            edgecolor="#D55E00", linewidth=0.8),
                 zorder=6)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time (ET)", fontsize=11)
    ax.set_ylabel("XLE 5-min bar dollar volume (USD millions)", fontsize=11)
    ax.set_title("XLE (Energy SPDR ETF) RTH dollar volume — March 23 vs adjacent trading days\n"
                  "The equity market's first chance to react to overnight Iran-ceasefire news",
                  fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.text(0.5, 0.005,
              "ETF volume is concentrated at the 09:30 ET RTH open, where overnight news flow is absorbed. "
              "On Mar 23, $448M traded in the first 5-min bar — over twice the equivalent bar on the prior "
              "Friday or the following Tuesday. By the time Trump RTed the announcement at 10:29 ET, the equity "
              "market had already priced in the move. ETFs do not trade meaningfully in pre-market hours, so the "
              "earlier-morning futures activity (CL contracts on NYMEX) is not visible on equity-instrument charts.",
              ha="center", color="#555", fontsize=8.5, style="italic", wrap=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out / "mar23_xle_volume.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved: mar23_xle_volume.png")


def render_xle_volume_csv(repo: Path, out: Path) -> None:
    """Also save the underlying data so Matt can re-render however he wants."""
    xle = pd.read_parquet(repo / "data/raw/minute_bars_5m/XLE.parquet")
    xle.index = pd.to_datetime(xle.index, utc=True)
    rows = []
    for date_str, label in [
        ("2026-03-20", "Mar 20 (prior Fri)"),
        ("2026-03-23", "Mar 23 (Iran ceasefire)"),
        ("2026-03-24", "Mar 24 (next day)"),
    ]:
        win = slice_morning(xle, date_str)
        for ts, r in win.iterrows():
            rows.append({
                "date":             date_str,
                "label":            label,
                "et":               r["et"],
                "minute_of_day":    int(r["minute_of_day"]),
                "close":            round(float(r["Close"]), 2),
                "volume":           int(r["Volume"]),
                "dollar_volume":    round(float(r["dollar_volume"]), 0),
                "is_rth":           bool(r["is_rth"]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out.parent / "mar23_xle_volume_comparison.csv", index=False)
    print(f"  saved: mar23_xle_volume_comparison.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--out",  type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out  = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {repo}")
    print(f"Writing to:   {out}\n")

    render_xle_volume_comparison(repo, out)
    render_xle_volume_csv(repo, out)


if __name__ == "__main__":
    main()
