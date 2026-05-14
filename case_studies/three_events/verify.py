#!/usr/bin/env python3
"""verify.py - consistency checks for the three-event matched bundle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR
DEFAULT_REPO = (SCRIPT_DIR.parent / "trump-truth-social-replication").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    args = parser.parse_args()

    data = args.data.resolve()
    repo = args.repo.resolve()
    print(f"Verifying: {data}")
    print(f"Source repo: {repo}\n")

    df = pd.read_csv(data / "three_events_uso_5min.csv")
    md = pd.read_csv(data / "three_events_metadata.csv")

    posts = pd.read_parquet(repo / "data/raw/posts_60d.parquet")
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    posts["id"] = posts["id"].astype(str)

    ok, fail = 0, 0
    def check(label: str, condition: bool, expected: str = "?", actual: str = "?") -> None:
        nonlocal ok, fail
        if condition:
            ok += 1
            print(f"  [OK]   {label}")
        else:
            fail += 1
            print(f"  [FAIL] {label}: expected {expected}, got {actual}")

    print("--- Structure ---")
    check("All three events present",
          set(df["event_id"].unique()) ==
          {"mar4_venezuela_oil", "mar23_iran_ceasefire", "apr7_indiana_burst"})
    check("Same column set across events",
          all(set(df[df["event_id"] == eid].columns) == set(df.columns)
              for eid in df["event_id"].unique()))

    # Each event should have ~73 bars (+/- 180 min in 5-min steps = 73)
    for eid in df["event_id"].unique():
        n = len(df[df["event_id"] == eid])
        check(f"Event {eid}: ~73 bars (got {n})", 60 <= n <= 80)

    print("\n--- Anchor / cum_pct math ---")
    for _, m in md.iterrows():
        eid = m["event_id"]
        sub = df[df["event_id"] == eid].sort_values("minute_offset")
        anchor_row = sub[sub["minute_offset"] == 0]
        check(f"{eid}: t=0 cum_pct_signed == 0",
              len(anchor_row) == 1 and abs(anchor_row.iloc[0]["cum_pct_signed"]) < 0.001)
        check(f"{eid}: anchor close = ${m['anchor_price']}",
              abs(anchor_row.iloc[0]["close"] - m["anchor_price"]) < 0.01 if len(anchor_row) == 1 else False)
        check(f"{eid}: cum_pct_abs == |cum_pct_signed|",
              (sub["cum_pct_abs"] - sub["cum_pct_signed"].abs()).abs().max() < 0.001)

    print("\n--- Trump posts referenced ---")
    for _, m in md.iterrows():
        post = posts[posts["id"] == str(m["post_id"])]
        check(f"{m['event_id']}: post id {m['post_id']} exists in source",
              len(post) == 1)

    print("\n--- Specific values ---")
    # Mar 4 anchor was $90.75 (verified earlier); Mar 23 was $112.89; Apr 7 was $137.76
    for eid, expected_price in [
        ("mar4_venezuela_oil", 90.75),
        ("mar23_iran_ceasefire", 112.89),
        ("apr7_indiana_burst", 137.76),
    ]:
        actual_price = float(md[md["event_id"] == eid].iloc[0]["anchor_price"])
        check(f"{eid}: anchor price = ${expected_price}",
              abs(actual_price - expected_price) < 0.01,
              f"${expected_price}", f"${actual_price}")

    print()
    print("=" * 60)
    print(f"SUMMARY: {ok} passed, {fail} failed")
    print("=" * 60)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
