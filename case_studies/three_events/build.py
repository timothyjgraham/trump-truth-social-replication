#!/usr/bin/env python3
"""
build.py - matched-format USO data for three case-study events.

Produces a single combined CSV (and JSON) with all three events in the
same column structure, so they can be rendered with one chart template:

  - Mar 4 (15:09 ET) - "normal signal" Venezuela oil exemplar
  - Mar 23 (07:23 ET) - pre-event signal, Iran ceasefire announcement morning
  - Apr 7 (16:13 ET) - "the massive one", start of the 14-min Indiana
                       endorsement burst that drives the headline PnL

All three are 5-min USO bars, +/- 180 minutes around each post anchor,
with both raw OHLCV columns and cum_pct deviation columns - so the same
data file supports either a price-and-volume chart or a cumulative-return
chart.

Path: assumes ../trump-truth-social-replication/ alongside this folder;
override with --repo if needed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
DEFAULT_REPO = (SCRIPT_DIR.parent / "trump-truth-social-replication").resolve()

WINDOW_MIN = 180  # +/- 180 min around each anchor

EVENTS = [
    {
        "id":            "mar4_venezuela_oil",
        "label":         "Mar 4: Venezuela oil (\"normal signal\")",
        "anchor_et":     "2026-03-04 15:09",
        "post_id":       "116172714486213504",
        "session":       "RTH",
        "type":          "post-event reaction (no pre-event drift)",
    },
    {
        "id":            "mar23_iran_ceasefire",
        "label":         "Mar 23: Iran ceasefire announcement (\"pre-event signal\")",
        "anchor_et":     "2026-03-23 07:23",
        "post_id":       "116278232362967212",
        "session":       "pre-market",
        "type":          "trading and news preceded the post",
    },
    {
        "id":            "apr7_indiana_burst",
        "label":         "Apr 7: 14-min endorsement burst (\"the massive one\")",
        "anchor_et":     "2026-04-07 16:13",
        "post_id":       "116365249022129961",
        "session":       "after-hours",
        "type":          "headline PnL event (28 posts in 14 min)",
    },
]


def load_uso(repo: Path) -> pd.DataFrame:
    df = pd.read_parquet(repo / "data/raw/minute_bars_5m/USO.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_posts(repo: Path) -> pd.DataFrame:
    p = pd.read_parquet(repo / "data/raw/posts_60d.parquet")
    p["created_at"] = pd.to_datetime(p["created_at"], utc=True)
    return p


def event_window(uso: pd.DataFrame, anchor_et: pd.Timestamp,
                  window_min: int) -> pd.DataFrame:
    """Return USO 5-min bars +/- window_min around the anchor.

    Anchor is snapped to the nearest 5-min bar at or before the post timestamp.
    If no bar exists at that floor (e.g. weekend post), advance to next bar.
    """
    anchor_utc = anchor_et.tz_convert("UTC")
    floor = anchor_utc.floor("5min")
    if floor in uso.index:
        anchor_bar = floor
    else:
        future = uso.index[uso.index >= floor]
        if len(future) == 0:
            return None, None
        anchor_bar = future[0]

    lo = anchor_bar - pd.Timedelta(minutes=window_min)
    hi = anchor_bar + pd.Timedelta(minutes=window_min)
    win = uso.loc[(uso.index >= lo) & (uso.index <= hi)].copy()

    win["ts_utc"]        = win.index
    win["ts_et"]         = win.index.tz_convert("America/New_York")
    win["et_time"]       = win["ts_et"].dt.strftime("%H:%M")
    win["minute_offset"] = ((win.index - anchor_bar).total_seconds() // 60).astype(int)
    anchor_price         = float(win.loc[anchor_bar, "Close"])
    win["cum_pct_signed"] = (win["Close"] / anchor_price - 1) * 100
    win["cum_pct_abs"]    = win["cum_pct_signed"].abs()
    win["is_zero_vol"]    = (win["Volume"] == 0)
    win["is_rth_bar"]     = win["is_rth"].astype(bool)

    return win, anchor_bar


def snippet(text: str, n: int = 220) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "..."


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
    posts["id"] = posts["id"].astype(str)

    all_rows = []
    metadata_rows = []

    for ev in EVENTS:
        anchor_et = pd.Timestamp(ev["anchor_et"], tz="America/New_York")
        win, anchor_bar = event_window(uso, anchor_et, WINDOW_MIN)
        if win is None:
            print(f"  {ev['id']}: no data, skipping")
            continue
        anchor_price = float(win.loc[anchor_bar, "Close"])

        # Pull post text for this event's anchor post
        match = posts[posts["id"] == ev["post_id"]]
        post_text = snippet(match.iloc[0]["text"]) if len(match) else ""

        for ts, r in win.iterrows():
            all_rows.append({
                "event_id":         ev["id"],
                "event_label":      ev["label"],
                "ts_utc":           pd.Timestamp(r["ts_utc"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ts_et":            r["ts_et"].strftime("%Y-%m-%d %H:%M ET"),
                "minute_offset":    int(r["minute_offset"]),
                "open":             round(float(r["Open"]),  2),
                "high":             round(float(r["High"]),  2),
                "low":              round(float(r["Low"]),   2),
                "close":            round(float(r["Close"]), 2),
                "volume":           int(r["Volume"]),
                "is_zero_vol":      bool(r["is_zero_vol"]),
                "is_rth_bar":       bool(r["is_rth_bar"]),
                "anchor_price":     round(anchor_price, 2),
                "cum_pct_signed":   round(float(r["cum_pct_signed"]), 4),
                "cum_pct_abs":      round(float(r["cum_pct_abs"]),    4),
            })

        metadata_rows.append({
            "event_id":          ev["id"],
            "event_label":       ev["label"],
            "anchor_et":         ev["anchor_et"],
            "anchor_bar_ts_utc": anchor_bar.isoformat(),
            "anchor_bar_ts_et":  anchor_bar.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M ET"),
            "anchor_price":      round(anchor_price, 2),
            "post_id":           ev["post_id"],
            "session":           ev["session"],
            "type":              ev["type"],
            "post_text":         post_text,
            "window_min":        WINDOW_MIN,
            "n_bars":            int((win["minute_offset"] >= -WINDOW_MIN).sum()),
            "n_zero_vol_bars":   int(win["is_zero_vol"].sum()),
            "n_rth_bars":        int(win["is_rth_bar"].sum()),
        })

        print(f"  {ev['id']}: anchor at {anchor_bar.tz_convert('America/New_York')} "
              f"({len(win)} bars, ${anchor_price:.2f})")

    df = pd.DataFrame(all_rows)
    md = pd.DataFrame(metadata_rows)

    df.to_csv(out / "three_events_uso_5min.csv", index=False)
    md.to_csv(out / "three_events_metadata.csv", index=False)

    # JSON nested by event for D3 ergonomics
    nested = {}
    for ev in EVENTS:
        ev_meta = md[md["event_id"] == ev["id"]].iloc[0].to_dict()
        ev_rows = df[df["event_id"] == ev["id"]].to_dict(orient="records")
        nested[ev["id"]] = {**ev_meta, "series": ev_rows}
    with open(out / "three_events.json", "w") as f:
        json.dump({
            "window_minutes": WINDOW_MIN,
            "asset":          "USO",
            "events":         nested,
        }, f, indent=2, default=str)

    print(f"\nAll outputs in: {out}")
    print(f"  three_events_uso_5min.csv:  {len(df)} rows ({df['event_id'].nunique()} events)")
    print(f"  three_events_metadata.csv:  {len(md)} rows")
    print(f"  three_events.json:          D3-friendly nested form")


if __name__ == "__main__":
    main()
