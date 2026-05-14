#!/usr/bin/env python3
"""
verify.py - consistency checks on the build outputs.

Re-derives every numerical claim in the README from source data and confirms
the produced CSV/JSON files match. Returns exit code 0 if all checks pass,
non-zero if any fail. Run after build.py.
"""
from __future__ import annotations

import argparse
import json
import re
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

    print(f"Verifying outputs in: {data}")
    print(f"Against source repo:  {repo}\n")

    df5    = pd.read_csv(data / "chart1v2_event_window_5min.csv")
    df1    = pd.read_csv(data / "chart1v2_event_window_1min.csv")
    meta   = pd.read_csv(data / "chart1v2_burst_metadata.csv")
    env    = pd.read_csv(data / "chart1v2_envelope_abs.csv")
    ts_env = pd.read_csv(data / "chart1v2_timeshift_envelope.csv")
    ts_meta = pd.read_csv(data / "chart1v2_timeshift_burst_metadata.csv")
    ts_combined = pd.read_csv(data / "chart1v2_timeshift_combined.csv")
    pub    = pd.read_csv(data / "chart1v2_falsification_publication.csv")
    broad_nodes  = pd.read_csv(data / "chart2alt_broad_word_nodes.csv")
    strict_nodes = pd.read_csv(data / "chart2alt_strict_word_nodes.csv")
    strict_edges = pd.read_csv(data / "chart2alt_strict_word_edges.csv")

    posts_pq = pd.read_parquet(repo / "data/raw/posts_60d.parquet")
    bursts_pq = pd.read_csv(repo / "data/results/pnl_concentration_bursts.csv")
    bursts_pq["ts_first"] = pd.to_datetime(bursts_pq["ts_first"], utc=True)
    bursts_pq = bursts_pq.sort_values("ts_first").reset_index(drop=True)
    bursts_pq["burst_id"] = bursts_pq.index + 1

    ok, fail = 0, 0

    def check(label: str, condition: bool, expected: str = "?", actual: str = "?") -> None:
        nonlocal ok, fail
        if condition:
            ok += 1
            print(f"  [OK]   {label}")
        else:
            fail += 1
            print(f"  [FAIL] {label}: expected {expected}, got {actual}")

    # === Chart 1 v2 ===
    print("--- Chart 1 v2 ---")
    check("All 15 bursts present", df5["burst_id"].nunique() == 15)
    check("All cum_pct_abs >= 0", (df5["cum_pct_abs"] >= 0).all())
    check("cum_pct_abs == |cum_pct_signed|",
          (df5["cum_pct_abs"] - df5["cum_pct_signed"].abs()).abs().max() < 0.001)
    zero = df5[df5["minute_offset"] == 0]
    check("All bursts at 0% at t=0",
          (zero["cum_pct_abs"] == 0).all() and (zero["cum_pct_signed"] == 0).all())
    check("volume column present", "volume" in df5.columns)
    check("is_zero_vol flag present", "is_zero_vol" in df5.columns)
    check("is_rth_bar flag present", "is_rth_bar" in df5.columns)

    b13 = df5[df5["burst_id"] == 13].sort_values("minute_offset")
    b13_pre_max = b13[b13["minute_offset"] < 0]["cum_pct_abs"].max()
    b13_pre_max_off = b13[b13["minute_offset"] < 0].sort_values(
        "cum_pct_abs", ascending=False).iloc[0]["minute_offset"]
    check("Burst 13 pre-event peak ≈ 3.75% at -145 min",
          abs(b13_pre_max - 3.75) < 0.01 and b13_pre_max_off == -145,
          "3.75% @ -145", f"{b13_pre_max:.2f}% @ {b13_pre_max_off}")

    b13_post_max = b13[b13["minute_offset"] > 0]["cum_pct_abs"].max()
    b13_post_max_off = b13[b13["minute_offset"] > 0].sort_values(
        "cum_pct_abs", ascending=False).iloc[0]["minute_offset"]
    check("Burst 13 post-event peak ≈ 4.07% at +45 min",
          abs(b13_post_max - 4.07) < 0.01 and b13_post_max_off == 45,
          "4.07% @ +45", f"{b13_post_max:.2f}% @ {b13_post_max_off}")

    b15 = df5[df5["burst_id"] == 15].sort_values("minute_offset")
    for off, expected in [(-60, 2.72), (0, 0.0), (60, -1.17), (180, -12.34)]:
        r = b15[b15["minute_offset"] == off]
        if len(r):
            actual = r.iloc[0]["cum_pct_signed"]
            check(f"Burst 15 signed @ {off:+d} min ≈ {expected:+.2f}%",
                  abs(actual - expected) < 0.01,
                  f"{expected:+.2f}", f"{actual:+.2f}")

    # === Time-shift falsification ===
    print("\n--- Time-shift falsification ---")
    for _, r in ts_meta.iterrows():
        real = pd.Timestamp(r["real_anchor"])
        shift = pd.Timestamp(r["shifted_anchor"])
        delta_h = (shift - real).total_seconds() / 3600
        same_tod = (real.tz_convert("America/New_York").strftime("%H:%M") ==
                    shift.tz_convert("America/New_York").strftime("%H:%M"))
        check(f"Burst {int(r['burst_id'])}: +24h shift, same ET time",
              abs(delta_h - 24) < 0.01 and same_tod,
              "24h same time-of-day", f"{delta_h:.2f}h same={same_tod}")

    zero_row = ts_env[ts_env["minute_offset"] == 0].iloc[0]
    check("Real median at t=0 == 0", abs(zero_row["real_median"]) < 0.001)
    check("Shifted median at t=0 == 0", abs(zero_row["shifted_median"]) < 0.001)

    stat = ts_env[(ts_env["minute_offset"] >= -30) & (ts_env["minute_offset"] <= 30)]
    wide = ts_env
    stat_pct = (stat["real_median"].mean() / stat["shifted_median"].mean() - 1) * 100
    wide_pct = (wide["real_median"].mean() / wide["shifted_median"].mean() - 1) * 100

    check(f"Stat window gap ≈ +30%", abs(stat_pct - 30) < 1,
          "+30%", f"{stat_pct:+.0f}%")
    check(f"Wider window gap ≈ +48%", abs(wide_pct - 48) < 1,
          "+48%", f"{wide_pct:+.0f}%")

    # === Word network ===
    print("\n--- Word network ---")
    check("173 oil-themed posts", int(posts_pq["topic_energy_oil"].sum()) == 173)

    OIL_RE = re.compile(
        r"\b(oil|crude|opec|barrel|refinery|gasoline|petroleum|hormuz|saudi|iran|"
        r"venezuela|drill|drilling|frack|pipeline|wti|brent)\b", re.I)
    n_strict = posts_pq["text"].astype(str).apply(lambda t: bool(OIL_RE.search(t))).sum()
    check("121 strict oil-term posts", n_strict == 121)

    claims = {"iran": 91, "strait": 23, "hormuz": 22, "middle": 20, "east": 19,
              "regime": 18, "nuclear": 17, "war": 15, "attack": 14, "israel": 12}
    for w, claimed in claims.items():
        actual = strict_nodes[strict_nodes["label"] == w]
        if len(actual):
            a = int(actual["post_count"].iloc[0])
            check(f"Strict word '{w}' = {claimed}", a == claimed, str(claimed), str(a))

    edges_d = {(r["source"], r["target"]): r["weight"] for _, r in strict_edges.iterrows()}

    def get_edge(a: str, b: str) -> int | None:
        return edges_d.get((a, b)) or edges_d.get((b, a))

    check("Edge hormuz-strait = 22", get_edge("hormuz", "strait") == 22)
    check("Edge iran-strait = 18", get_edge("iran", "strait") == 18)
    check("Edge iran-hormuz = 17", get_edge("iran", "hormuz") == 17)
    check("Edge iran-nuclear = 16", get_edge("iran", "nuclear") == 16)
    check("Edge iran-deal = 13", get_edge("iran", "deal") == 13)

    # === PnL split ===
    print("\n--- PnL split ---")
    iran_oil_pnl = bursts_pq[bursts_pq["burst_id"].isin([2, 11, 13])]["pnl_total"].sum() / 1e6
    endorsement_pnl = bursts_pq[bursts_pq["burst_id"].isin([1, 5, 7, 14, 15])]["pnl_total"].sum() / 1e6
    total = bursts_pq["pnl_total"].sum() / 1e6

    check("Iran/oil bursts (2,11,13) PnL ≈ -$35M", abs(iran_oil_pnl - (-35.48)) < 0.01,
          "-$35.48M", f"${iran_oil_pnl:.2f}M")
    check("Endorsement bursts PnL ≈ +$197M", abs(endorsement_pnl - 197.28) < 0.01,
          "+$197.28M", f"${endorsement_pnl:.2f}M")
    check("Total PnL ≈ +$160M", abs(total - 159.88) < 0.5,
          "+$159.88M", f"${total:.2f}M")
    check("Burst 13 PnL = -$31.93M",
          abs(bursts_pq[bursts_pq["burst_id"] == 13]["pnl_total"].iloc[0] / 1e6 - (-31.93)) < 0.01)
    check("Burst 15 PnL = +$212.92M",
          abs(bursts_pq[bursts_pq["burst_id"] == 15]["pnl_total"].iloc[0] / 1e6 - 212.92) < 0.01)

    # === Publication frame ===
    print("\n--- Publication frame ---")
    check("Publication CSV has 25 rows (±60 in 5-min steps)", len(pub) == 25)
    check("Publication frame anchored at 0% at t=0",
          pub.loc[pub["minute_offset"] == 0, "real_median"].iloc[0] == 0 and
          pub.loc[pub["minute_offset"] == 0, "shifted_median"].iloc[0] == 0)
    check("Publication frame: 15 bursts at t=0",
          int(pub.loc[pub["minute_offset"] == 0, "real_n_bursts"].iloc[0]) == 15)

    with open(data / "chart1v2_falsification_publication.json") as f:
        pj = json.load(f)
    check("Publication JSON includes §3.4 rigorous-test numbers",
          "rigorous_falsification_test" in pj and
          pj["rigorous_falsification_test"]["asset_xle_ofi_real_pre_mean"] == 0.04878)

    print()
    print("=" * 70)
    print(f"SUMMARY: {ok} passed, {fail} failed")
    print("=" * 70)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
