#!/usr/bin/env python3
"""
verify.py — consistency checks for the Mar 23 case study.

Confirms numerical claims in the README against source data. Returns
exit code 0 if all checks pass.
"""
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

    df = pd.read_csv(data / "mar23_uso_5min.csv")
    timeline = pd.read_csv(data / "mar23_event_timeline.csv")
    falsification = pd.read_csv(data / "mar23_vs_mar24_falsification.csv")

    uso = pd.read_parquet(repo / "data/raw/minute_bars_5m/USO.parquet")
    uso.index = pd.to_datetime(uso.index, utc=True)
    posts = pd.read_parquet(repo / "data/raw/posts_60d.parquet")
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)

    ok, fail = 0, 0
    def check(label: str, condition: bool, expected: str = "?", actual: str = "?") -> None:
        nonlocal ok, fail
        if condition:
            ok += 1
            print(f"  [OK]   {label}")
        else:
            fail += 1
            print(f"  [FAIL] {label}: expected {expected}, got {actual}")

    print("--- USO 5-min bars ---")

    # Friday close ref
    fri_close = float(uso.loc[pd.Timestamp("2026-03-20 19:55:00", tz="UTC"), "Close"])
    check("Friday Mar 20 close = $121.44", abs(fri_close - 121.44) < 0.01,
          "$121.44", f"${fri_close:.2f}")

    # Mar 23 04:00 ET pre-market open
    open_0400 = float(df.loc[df["ts_et"].str.endswith("04:00 ET"), "open"].iloc[0])
    check("Mar 23 04:00 ET pre-market open = $125.80",
          abs(open_0400 - 125.80) < 0.01, "$125.80", f"${open_0400:.2f}")

    # The big drop at 07:05
    bar_07_05 = df.loc[df["ts_et"].str.endswith("07:05 ET")].iloc[0]
    check("Mar 23 07:05 ET bar opens at $122.82",
          abs(bar_07_05["open"] - 122.82) < 0.01, "$122.82", f"${bar_07_05['open']:.2f}")
    check("Mar 23 07:05 ET bar low = $109.00",
          abs(bar_07_05["low"] - 109.00) < 0.01, "$109.00", f"${bar_07_05['low']:.2f}")
    check("Mar 23 07:05 ET bar close = $113.05",
          abs(bar_07_05["close"] - 113.05) < 0.01, "$113.05", f"${bar_07_05['close']:.2f}")

    # 07:00 anchor close
    bar_07_00 = df.loc[df["ts_et"].str.endswith("07:00 ET")].iloc[0]
    check("Mar 23 07:00 ET close = $122.84 (anchor for falsification)",
          abs(bar_07_00["close"] - 122.84) < 0.01, "$122.84", f"${bar_07_00['close']:.2f}")

    # Volume profile
    pre_rth = df[~df["is_rth_bar"]]
    rth = df[df["is_rth_bar"]]
    check("All pre-RTH bars have zero volume",
          (pre_rth["volume"] == 0).all(),
          "all zero", f"{(pre_rth['volume'] == 0).sum()}/{len(pre_rth)}")
    check("RTH bars have nonzero volume",
          (rth["volume"] > 0).all(),
          "all nonzero", f"{(rth['volume'] > 0).sum()}/{len(rth)}")

    # 09:30 RTH open
    bar_09_30 = df.loc[df["ts_et"].str.endswith("09:30 ET")].iloc[0]
    check("Mar 23 09:30 ET RTH-open close = $112.97",
          abs(bar_09_30["close"] - 112.97) < 0.01, "$112.97", f"${bar_09_30['close']:.2f}")

    print("\n--- Trump posts on Mar 23 ---")

    trump_posts_1 = posts[(posts["created_at"] >= "2026-03-23 11:23:00") &
                           (posts["created_at"] < "2026-03-23 11:24:00")]
    check("Trump's original Iran post on Mar 23 at 07:23:40 ET (= UTC 11:23)",
          len(trump_posts_1) == 1)
    if len(trump_posts_1) == 1:
        p = trump_posts_1.iloc[0]
        text_starts_iran = "I AM PLEASED TO REPORT THAT THE UNITED STATES OF AMERICA, AND THE COUNTRY OF IRAN" in (p["text"] or "")
        check("Original Iran post text matches expected announcement",
              text_starts_iran)
        check("Original post is tagged oil + iran",
              bool(p["topic_energy_oil"]) and bool(p["topic_iran_military"]))

    # The 10:29 ET RT (= burst 13). Truth Social labels this with text starting
    # "RT @realDonaldTrump..." rather than setting is_reblog, so we match on text.
    trump_posts_2 = posts[(posts["created_at"] >= "2026-03-23 14:29:00") &
                           (posts["created_at"] < "2026-03-23 14:30:00")]
    iran_rt = trump_posts_2[trump_posts_2["text"].astype(str).str.contains(
        "I AM PLEASED TO REPORT", regex=False, na=False)]
    check("Mar 23 10:29 ET Iran RT exists (= burst 13 in v2 data)",
          len(iran_rt) >= 1)

    print("\n--- Mar 23 vs Mar 24 falsification ---")

    real_min  = falsification["real_pct_vs_07_00"].min()
    real_max  = falsification["real_pct_vs_07_00"].max()
    shifted_min = falsification["shifted_pct_vs_07_00"].min()
    shifted_max = falsification["shifted_pct_vs_07_00"].max()
    check(f"Mar 23 reaches at least -8% vs 07:00 anchor ({real_min:.2f}%)",
          real_min < -8)
    # Mar 24 should stay within a much tighter range than Mar 23, but isn't
    # strictly flat — small intraday moves are normal. The point is the
    # contrast with Mar 23's -12.96% drop.
    check(f"Mar 24 control range much smaller than Mar 23 "
          f"(Mar 24: {shifted_min:.2f}% to {shifted_max:.2f}%; "
          f"Mar 23: {real_min:.2f}% to {real_max:.2f}%)",
          (shifted_max - shifted_min) < (real_max - real_min) / 3)

    print()
    print("=" * 60)
    print(f"SUMMARY: {ok} passed, {fail} failed")
    print("=" * 60)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
