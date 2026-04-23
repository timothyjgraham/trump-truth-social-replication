#!/usr/bin/env python3
"""
11_posting_patterns.py
======================

Posting-pattern analysis on the full Trump Truth Social archive (32,433
posts, Feb 2022 - Apr 2026), with three deliverables:

1. Descriptive stats: posts/day, hour-of-day, day-of-week, weekend vs
   weekday, market-hours vs off-hours.
2. Stephen's weekend hypothesis test:
     (a) Is the daily posting rate different on weekends?
     (b) Does the topic mix shift on weekends?
     (c) Are weekend posts followed by larger Monday-morning oil moves?
3. Overlay of the 81-event triggered oil subset on the time-of-day
   distribution.

Outputs:
  data/results/posting_patterns.json
  report/figures/fig4_posting_patterns.{pdf,png}
"""

from __future__ import annotations

import json
from zoneinfo import ZoneInfo
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _paths import (
    TRUTH_ARCHIVE_JSON,
    POSTS_PARQUET,
    RESULTS_DIR,
    FIGURES_DIR,
    MINUTE_BARS_5M,
    ensure_dirs,
)

ensure_dirs()

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Palette consistent with figs 1-3
NAVY = "#1f3b73"
LIGHT_BLUE = "#7fb1e0"
GREY = "#9aa0a6"
ORANGE = "#d96f32"
RED = "#b03030"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
})


# ── data loading ─────────────────────────────────────────────────────────────

def load_full_archive() -> pd.DataFrame:
    """Load every post from the raw Truth Social archive with ET and UTC times."""
    with TRUTH_ARCHIVE_JSON.open() as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["dt_et"] = df["created_at"].dt.tz_convert(ET)
    df["date_et"] = df["dt_et"].dt.date
    df["hour_et"] = df["dt_et"].dt.hour
    df["weekday"] = df["dt_et"].dt.weekday  # 0 = Monday
    df["weekday_name"] = df["dt_et"].dt.day_name()
    df["is_weekend"] = df["weekday"] >= 5
    df["is_market_hours"] = is_market_hours_et(df["dt_et"])
    return df


def is_market_hours_et(dt_et: pd.Series) -> pd.Series:
    """True if the timestamp falls inside US regular trading hours (09:30-16:00 ET, weekdays)."""
    hour = dt_et.dt.hour
    minute = dt_et.dt.minute
    after_open = (hour > 9) | ((hour == 9) & (minute >= 30))
    before_close = hour < 16
    on_weekday = dt_et.dt.weekday < 5
    return after_open & before_close & on_weekday


def session_label(dt_et: pd.Series) -> pd.Series:
    """Classify each timestamp into market session bucket."""
    out = pd.Series("weekend", index=dt_et.index)
    on_weekday = dt_et.dt.weekday < 5
    hour = dt_et.dt.hour
    minute = dt_et.dt.minute
    pre_market = on_weekday & ((hour < 9) | ((hour == 9) & (minute < 30)))
    regular = on_weekday & ((hour > 9) | ((hour == 9) & (minute >= 30))) & (hour < 16)
    after_hours = on_weekday & (hour >= 16)
    out[pre_market] = "pre_market"
    out[regular] = "regular"
    out[after_hours] = "after_hours"
    return out


# ── analysis ─────────────────────────────────────────────────────────────────

def daily_rate_test(df: pd.DataFrame) -> dict:
    """Compare the average posts/day on weekends vs weekdays.

    Uses a Welch two-sample t-test on the daily-count series.
    """
    daily = df.groupby("date_et").size().rename("posts").reset_index()
    daily["weekday"] = pd.to_datetime(daily["date_et"]).dt.weekday
    daily["is_weekend"] = daily["weekday"] >= 5
    we = daily.loc[daily["is_weekend"], "posts"].values
    wd = daily.loc[~daily["is_weekend"], "posts"].values
    t, p = welch_t_test(we, wd)
    return {
        "weekday_days": int(len(wd)),
        "weekend_days": int(len(we)),
        "weekday_mean_posts_per_day": float(np.mean(wd)),
        "weekday_median_posts_per_day": float(np.median(wd)),
        "weekend_mean_posts_per_day": float(np.mean(we)),
        "weekend_median_posts_per_day": float(np.median(we)),
        "ratio_weekend_to_weekday": float(np.mean(we) / np.mean(wd)) if np.mean(wd) > 0 else np.nan,
        "welch_t": float(t),
        "welch_p_two_sided": float(p),
    }


def welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test (no scipy dependency)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    # two-sided p from t distribution; use normal approx for large df
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, p


def topic_mix_weekend_vs_weekday(posts60: pd.DataFrame) -> dict:
    """Compare topic share for weekend vs weekday posts inside the 60-day window."""
    posts60 = posts60.copy()
    posts60["dt_et"] = posts60["created_at"].dt.tz_convert(ET)
    posts60["is_weekend"] = posts60["dt_et"].dt.weekday >= 5
    topics = [c for c in posts60.columns if c.startswith("topic_")]
    out = {}
    we = posts60[posts60["is_weekend"]]
    wd = posts60[~posts60["is_weekend"]]
    for t in topics:
        we_share = float(we[t].mean()) if len(we) else 0.0
        wd_share = float(wd[t].mean()) if len(wd) else 0.0
        out[t] = {
            "weekend_share": we_share,
            "weekday_share": wd_share,
            "weekend_n": int(we[t].sum()),
            "weekday_n": int(wd[t].sum()),
            "diff_we_minus_wd_pct": (we_share - wd_share) * 100,
        }
    return {
        "weekend_post_count": int(len(we)),
        "weekday_post_count": int(len(wd)),
        "by_topic": out,
    }


def monday_oil_response(posts60: pd.DataFrame) -> dict:
    """Sun-PM-ET / Mon-pre-market posts about oil — does USO open higher/lower on the next Monday?

    A loose proxy for 'weekend posts moving Monday markets'. Compares the
    Mon-open-to-09:35 return on Mondays preceded by an oil-themed weekend post
    vs all other Mondays.
    """
    bars = pd.read_parquet(MINUTE_BARS_5M / "USO.parquet")
    bars.index = pd.to_datetime(bars.index, utc=True)
    et_idx = bars.index.tz_convert(ET)
    bars = bars.assign(_et=et_idx,
                       _date=et_idx.date,
                       _hourmin=et_idx.hour * 100 + et_idx.minute,
                       _wd=et_idx.weekday)

    # Mon open and Mon 09:35 close per Monday
    mondays = bars[bars["_wd"] == 0].copy()
    mon_open = mondays[mondays["_hourmin"] == 930].groupby("_date")["Open"].first()
    mon_first_5m = mondays[mondays["_hourmin"] == 930].groupby("_date")["Close"].first()
    mon_returns = (mon_first_5m / mon_open - 1).dropna()

    # Weekend (Sat/Sun) oil-themed posts
    posts60 = posts60.copy()
    posts60["dt_et"] = posts60["created_at"].dt.tz_convert(ET)
    posts60["is_weekend"] = posts60["dt_et"].dt.weekday >= 5
    we_oil = posts60[posts60["is_weekend"] & posts60["topic_energy_oil"]].copy()
    # Find the next Monday for each weekend post
    we_oil["next_monday"] = we_oil["dt_et"].apply(_next_monday)
    flagged_mondays = sorted(set(we_oil["next_monday"]))

    flagged_returns = mon_returns.reindex(flagged_mondays).dropna().values * 1e4  # bps
    other_mondays = [d for d in mon_returns.index if d not in set(flagged_mondays)]
    other_returns = mon_returns.reindex(other_mondays).dropna().values * 1e4

    if len(flagged_returns) >= 2 and len(other_returns) >= 2:
        t, p = welch_t_test(flagged_returns, other_returns)
    else:
        t, p = np.nan, np.nan

    return {
        "n_mondays_with_weekend_oil_post": int(len(flagged_returns)),
        "n_other_mondays": int(len(other_returns)),
        "uso_first_5min_return_bps_flagged_mean": float(np.mean(flagged_returns)) if len(flagged_returns) else np.nan,
        "uso_first_5min_return_bps_other_mean": float(np.mean(other_returns)) if len(other_returns) else np.nan,
        "welch_t": float(t) if not np.isnan(t) else None,
        "welch_p_two_sided": float(p) if not np.isnan(p) else None,
    }


def _next_monday(dt) -> pd.Timestamp:
    days_ahead = (7 - dt.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (dt + pd.Timedelta(days=days_ahead)).date()


def overlay_triggered(posts60: pd.DataFrame) -> dict:
    """Pull the 81 triggered events (pre_vpinz > 0.5) and tabulate their hour/day distribution."""
    ev = pd.read_csv(RESULTS_DIR / "dollar_upper_bound_uso_events.csv")
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, format="ISO8601")
    triggered = ev[ev["pre_vpinz"] > 0.5].copy()
    triggered["dt_et"] = triggered["ts"].dt.tz_convert(ET)
    triggered["hour_et"] = triggered["dt_et"].dt.hour
    triggered["weekday"] = triggered["dt_et"].dt.weekday
    triggered["session"] = session_label(triggered["dt_et"])
    return {
        "n_triggered": int(len(triggered)),
        "by_hour_et": dict(Counter(triggered["hour_et"]).most_common()),
        "by_weekday": dict(Counter(triggered["weekday"]).most_common()),
        "by_session": dict(Counter(triggered["session"]).most_common()),
    }


# ── figure ───────────────────────────────────────────────────────────────────

def make_figure(df: pd.DataFrame, posts60: pd.DataFrame, triggered_overlay: dict):
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32,
                          left=0.07, right=0.95, top=0.94, bottom=0.08)

    # Panel A: posts per day, full archive
    ax = fig.add_subplot(gs[0, :])
    daily = df.groupby("date_et").size()
    daily.index = pd.to_datetime(daily.index)
    ax.plot(daily.index, daily.values, color=NAVY, linewidth=0.9, alpha=0.85)
    # Shade the 60-day study window
    win_lo = posts60["created_at"].dt.tz_convert(ET).min().date()
    win_hi = posts60["created_at"].dt.tz_convert(ET).max().date()
    ax.axvspan(pd.to_datetime(win_lo), pd.to_datetime(win_hi),
               color=ORANGE, alpha=0.18, label=f"60-day study window\n({win_lo} to {win_hi})")
    ax.set_ylabel("Posts per day")
    ax.set_title("(a) Daily posting rate, full Truth Social archive (Feb 2022 - Apr 2026)",
                 loc="left", weight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(daily.index.min(), daily.index.max())

    # Panel B: hour of day, ET, with market hours shaded and triggered overlay
    ax = fig.add_subplot(gs[1, 0])
    by_hour = df.groupby("hour_et").size()
    bars = ax.bar(by_hour.index, by_hour.values, color=LIGHT_BLUE,
                  edgecolor="white", linewidth=0.6, width=0.85,
                  label=f"All posts (n={len(df):,})")
    ax.axvspan(9.5, 16, color=GREY, alpha=0.15, zorder=0,
               label="US regular trading hours")
    # Triggered overlay (count per hour) on second axis
    ax2 = ax.twinx()
    trig_hours = triggered_overlay["by_hour_et"]
    h = sorted(trig_hours.keys())
    counts = [trig_hours[k] for k in h]
    ax2.scatter(h, counts, color=RED, s=46, zorder=5,
                edgecolor="white", linewidth=0.8,
                label=f"Triggered oil events (n={triggered_overlay['n_triggered']})")
    ax2.set_ylabel("Triggered oil events (count)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(RED)
    ax2.grid(False)
    ax.set_xlabel("Hour of day (US Eastern)")
    ax.set_ylabel("Posts (all topics)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_title("(b) Posts by hour of day, with triggered oil events overlaid",
                 loc="left", weight="bold", fontsize=10.5)
    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = ax2.get_legend_handles_labels()
    ax.legend(handles_a + handles_b, labels_a + labels_b,
              loc="upper left", framealpha=0.9, fontsize=8)

    # Panel C: day of week with weekend/weekday colouring
    ax = fig.add_subplot(gs[1, 1])
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_day = df.groupby("weekday").size().reindex(range(7), fill_value=0)
    colors = [NAVY] * 5 + [ORANGE] * 2
    bars = ax.bar(range(7), by_day.values, color=colors,
                  edgecolor="white", linewidth=0.6, width=0.78)
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    ax.set_ylabel("Posts (full archive)")
    for b, v in zip(bars, by_day.values):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:,}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, by_day.max() * 1.15)
    ax.set_title("(c) Posts by day of week (weekend in orange)",
                 loc="left", weight="bold", fontsize=10.5)

    plt.savefig(FIGURES_DIR / "fig4_posting_patterns.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig4_posting_patterns.png", dpi=200, bbox_inches="tight")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("loading full archive ...")
    df = load_full_archive()
    print(f"  {len(df):,} posts, {df['date_et'].nunique():,} unique calendar days")
    print(f"  date range (ET): {df['dt_et'].min()} -> {df['dt_et'].max()}")

    print("\nloading 60-day topic-tagged subset ...")
    posts60 = pd.read_parquet(POSTS_PARQUET)
    posts60["created_at"] = pd.to_datetime(posts60["created_at"], utc=True)
    print(f"  {len(posts60):,} posts; oil-themed: {int(posts60['topic_energy_oil'].sum())}")

    print("\n=== posting-rate test (weekend vs weekday) ===")
    rate = daily_rate_test(df)
    for k, v in rate.items():
        print(f"  {k}: {v}")

    print("\n=== topic-mix shift on weekends (60-day window) ===")
    mix = topic_mix_weekend_vs_weekday(posts60)
    print(f"  weekend posts: {mix['weekend_post_count']}, "
          f"weekday posts: {mix['weekday_post_count']}")
    for t, b in sorted(mix["by_topic"].items(),
                       key=lambda kv: abs(kv[1]["diff_we_minus_wd_pct"]),
                       reverse=True):
        print(f"  {t:24s}  weekend={b['weekend_share']*100:5.1f}%  "
              f"weekday={b['weekday_share']*100:5.1f}%  "
              f"Δ={b['diff_we_minus_wd_pct']:+5.1f} pct")

    print("\n=== Monday-morning USO response to weekend oil posts ===")
    mon = monday_oil_response(posts60)
    for k, v in mon.items():
        print(f"  {k}: {v}")

    print("\n=== triggered oil events overlay ===")
    overlay = overlay_triggered(posts60)
    print(f"  n_triggered = {overlay['n_triggered']}")
    print(f"  by session: {overlay['by_session']}")
    print(f"  by weekday: {overlay['by_weekday']}")

    print("\nbuilding figure ...")
    make_figure(df, posts60, overlay)
    print(f"  wrote {FIGURES_DIR / 'fig4_posting_patterns.pdf'}")
    print(f"  wrote {FIGURES_DIR / 'fig4_posting_patterns.png'}")

    out = {
        "method": "Posting-pattern analysis on the full 32,433-post archive plus "
                  "60-day topic-tagged subset, with overlay of the 81 triggered "
                  "oil events.",
        "n_posts_full_archive": int(len(df)),
        "n_posts_60d_window": int(len(posts60)),
        "rate_test_weekend_vs_weekday": rate,
        "topic_mix_60d": mix,
        "monday_morning_uso_response": mon,
        "triggered_overlay": overlay,
    }
    out_path = RESULTS_DIR / "posting_patterns.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
