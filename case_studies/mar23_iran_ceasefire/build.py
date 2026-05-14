#!/usr/bin/env python3
"""
build.py — single-event case study, March 23 2026 Iran ceasefire morning.

Produces three CSVs that document the morning's USO price movement, the
Trump posts that fired during the window, and a +24h-shifted control series
(Mar 24 morning) for the falsification comparison.

Path resolution:
  - The replication repo is expected at ../trump-truth-social-replication/
  - Override with --repo /path/to/repo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
DEFAULT_REPO = (SCRIPT_DIR.parent / "trump-truth-social-replication").resolve()

# Window of interest: Mar 23 03:00 - 11:30 ET
EVENT_DATE      = "2026-03-23"
WIN_START_ET    = pd.Timestamp(f"{EVENT_DATE} 03:00", tz="America/New_York")
WIN_END_ET      = pd.Timestamp(f"{EVENT_DATE} 11:30", tz="America/New_York")
SHIFT_DATE      = "2026-03-24"
SHIFT_START_ET  = pd.Timestamp(f"{SHIFT_DATE} 03:00", tz="America/New_York")
SHIFT_END_ET    = pd.Timestamp(f"{SHIFT_DATE} 11:30", tz="America/New_York")


def load_uso(repo: Path) -> pd.DataFrame:
    df = pd.read_parquet(repo / "data/raw/minute_bars_5m/USO.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_posts(repo: Path) -> pd.DataFrame:
    p = pd.read_parquet(repo / "data/raw/posts_60d.parquet")
    p["created_at"] = pd.to_datetime(p["created_at"], utc=True)
    return p


def slice_window(uso: pd.DataFrame, start_et: pd.Timestamp,
                  end_et: pd.Timestamp) -> pd.DataFrame:
    start_utc = start_et.tz_convert("UTC")
    end_utc   = end_et.tz_convert("UTC")
    win = uso.loc[(uso.index >= start_utc) & (uso.index <= end_utc)].copy()
    win["ts_utc"]  = win.index
    win["ts_et"]   = win.index.tz_convert("America/New_York")
    win["et_time"] = win["ts_et"].dt.strftime("%H:%M")
    win["et_date"] = win["ts_et"].dt.strftime("%Y-%m-%d")
    return win


def build_uso_window(uso: pd.DataFrame, out: Path) -> None:
    """Mar 23 03:00-11:30 ET, every 5-min bar with OHLCV + flags."""
    win = slice_window(uso, WIN_START_ET, WIN_END_ET)
    out_df = pd.DataFrame({
        "ts_utc":      win["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ts_et":       win["ts_et"].dt.strftime("%Y-%m-%d %H:%M ET"),
        "open":        win["Open"].round(2),
        "high":        win["High"].round(2),
        "low":         win["Low"].round(2),
        "close":       win["Close"].round(2),
        "volume":      win["Volume"].astype(int),
        "is_zero_vol": (win["Volume"] == 0).astype(bool),
        "is_rth_bar":  win["is_rth"].astype(bool),
    })
    out_df.to_csv(out / "mar23_uso_5min.csv", index=False)
    print(f"  mar23_uso_5min.csv: {len(out_df)} bars from "
          f"{out_df['ts_et'].iloc[0]} to {out_df['ts_et'].iloc[-1]}")
    print(f"    Pre-market (zero-vol) bars: {int(out_df['is_zero_vol'].sum())}")
    print(f"    RTH bars:                   {int(out_df['is_rth_bar'].sum())}")


def build_event_timeline(posts: pd.DataFrame, out: Path) -> None:
    """News + Trump posts on Mar 23 morning, in one timeline file."""
    # External news markers (per Kobeissi annotations on light crude futures)
    external = [
        {"ts_et":   f"{EVENT_DATE} 03:40",
         "kind":    "external_marker",
         "source":  "Kobeissi (light crude futures, NYMEX)",
         "label":   "$920M of crude oil short positions taken (per Kobeissi)",
         "note":    "Refers to CL futures, not USO ETF. Cited for context."},
        {"ts_et":   f"{EVENT_DATE} 04:50",
         "kind":    "external_marker",
         "source":  "Kobeissi (citing Axios)",
         "label":   "Axios reports a deal to end the Iran war is imminent",
         "note":    "Public news at this time per the Kobeissi annotation."},
        {"ts_et":   f"{EVENT_DATE} 07:00",
         "kind":    "external_marker",
         "source":  "Kobeissi (light crude futures)",
         "label":   "Crude oil shorts at +$125M profit (per Kobeissi)",
         "note":    "On CL futures."},
        {"ts_et":   f"{EVENT_DATE} 09:30",
         "kind":    "session_marker",
         "source":  "NYSE",
         "label":   "Regular trading hours open",
         "note":    "USO begins trading on actual volume."},
    ]

    # Trump posts on Mar 22-23 (focus on Iran/oil-related)
    mar22_23 = posts[(posts["created_at"] >= "2026-03-22") &
                       (posts["created_at"] < "2026-03-24")].sort_values("created_at")
    trump_rows = []
    for _, p in mar22_23.iterrows():
        text = (p["text"] or "").replace("\n", " ").strip()
        if not text and not p.get("is_reblog"):
            continue
        is_oil_iran = bool(p.get("topic_energy_oil") or p.get("topic_iran_military"))
        ts_et = p["created_at"].tz_convert("America/New_York")
        trump_rows.append({
            "ts_et":      ts_et.strftime("%Y-%m-%d %H:%M"),
            "kind":       "trump_post",
            "source":     "Truth Social",
            "post_id":    str(p["id"]),
            "is_reblog":  bool(p["is_reblog"]),
            "is_oil_or_iran_topic": is_oil_iran,
            "label":      text[:140] + ("…" if len(text) > 140 else ""),
            "note":       ("Triggered event in pipeline" if is_oil_iran else ""),
        })

    timeline = pd.DataFrame(external + trump_rows)
    timeline = timeline.sort_values("ts_et").reset_index(drop=True)
    timeline.to_csv(out / "mar23_event_timeline.csv", index=False)
    print(f"  mar23_event_timeline.csv: {len(timeline)} entries "
          f"({len(external)} external markers, {len(trump_rows)} Trump posts)")


def build_falsification_pair(uso: pd.DataFrame, out: Path) -> None:
    """USO bars for Mar 23 morning vs Mar 24 morning (the +24h-shifted control).

    The Mar 24 morning is included to demonstrate that the same hours of day
    on the next trading day did NOT show the Iran-news-driven move — i.e.,
    the Mar 23 pattern is event-specific, not a recurring time-of-day artifact.
    """
    real    = slice_window(uso, WIN_START_ET, WIN_END_ET).copy()
    shifted = slice_window(uso, SHIFT_START_ET, SHIFT_END_ET).copy()

    # Common minute-of-day index for alignment
    real["minute_of_day"] = real["ts_et"].dt.hour * 60 + real["ts_et"].dt.minute
    shifted["minute_of_day"] = shifted["ts_et"].dt.hour * 60 + shifted["ts_et"].dt.minute

    real_anchor    = real.loc[real["et_time"] == "07:00", "Close"].iloc[0]
    shifted_anchor = shifted.loc[shifted["et_time"] == "07:00", "Close"].iloc[0]

    real["pct_vs_07_00"]    = (real["Close"] / real_anchor - 1) * 100
    shifted["pct_vs_07_00"] = (shifted["Close"] / shifted_anchor - 1) * 100

    combined = pd.DataFrame({
        "minute_of_day":      real["minute_of_day"].values,
        "et_time":            real["et_time"].values,
        "real_close":         real["Close"].round(2).values,
        "real_pct_vs_07_00":  real["pct_vs_07_00"].round(3).values,
        "real_anchor_close":  round(real_anchor, 2),
    })

    # Align shifted onto real's minute_of_day index
    sh_indexed = shifted.set_index("minute_of_day")
    combined["shifted_close"] = combined["minute_of_day"].map(
        sh_indexed["Close"].round(2)).values
    combined["shifted_pct_vs_07_00"] = combined["minute_of_day"].map(
        sh_indexed["pct_vs_07_00"].round(3)).values
    combined["shifted_anchor_close"] = round(shifted_anchor, 2)

    combined.to_csv(out / "mar23_vs_mar24_falsification.csv", index=False)
    print(f"  mar23_vs_mar24_falsification.csv: {len(combined)} bars")
    print(f"    Mar 23 anchor (07:00 ET): ${real_anchor:.2f}")
    print(f"    Mar 24 anchor (07:00 ET): ${shifted_anchor:.2f}")
    real_min = real["pct_vs_07_00"].min()
    real_max = real["pct_vs_07_00"].max()
    shifted_min = shifted["pct_vs_07_00"].min()
    shifted_max = shifted["pct_vs_07_00"].max()
    print(f"    Mar 23 range vs anchor: {real_min:+.2f}% to {real_max:+.2f}%")
    print(f"    Mar 24 range vs anchor: {shifted_min:+.2f}% to {shifted_max:+.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--out",  type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()

    repo = args.repo.resolve()
    out  = args.out.resolve()
    if not repo.exists():
        sys.exit(f"ERROR: repo not found at {repo}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Repo:   {repo}")
    print(f"Output: {out}\n")

    uso   = load_uso(repo)
    posts = load_posts(repo)

    print("[1/3] Mar 23 USO 5-min bars (03:00-11:30 ET)...")
    build_uso_window(uso, out)

    print("\n[2/3] Event timeline (news + Trump posts)...")
    build_event_timeline(posts, out)

    print("\n[3/3] Mar 23 vs Mar 24 (+24h shifted) falsification pair...")
    build_falsification_pair(uso, out)

    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
