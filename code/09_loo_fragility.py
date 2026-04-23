#!/usr/bin/env python3
"""
09_loo_fragility.py
===================

Leave-one-out fragility audit on the headline $159.88M dollar upper bound
(triggered slice: oil-themed events with pre-window mean(vpin_z) > 0.5).

For each of the n triggered events, recompute the aggregate sum with that
event removed. Report:
  - LOO sum range (min, max)  — how much the headline moves under LOO
  - max single-event |contribution|  — fragility flag if > 15%
  - top 5 events by absolute contribution

Also runs the same analysis under a stricter pre-event convention (no
look-ahead at the post bar) for comparison.

Outputs:
  data/results/phase2c_loo.json
"""

import json

import numpy as np
import pandas as pd

from _paths import POSTS_PARQUET, MINUTE_BARS_5M, SIGNALS_5M, LOO_JSON, ensure_dirs

PRE_BARS = 12
POST_BARS = 12
TRIGGER = 0.5


def load_uso():
    bars = pd.read_parquet(MINUTE_BARS_5M / "USO.parquet")
    sigs = pd.read_parquet(SIGNALS_5M / "USO.parquet")
    bars.index = pd.to_datetime(bars.index, utc=True)
    sigs.index = pd.to_datetime(sigs.index, utc=True)
    return bars, sigs


def build_events(posts, bars, sigs, convention):
    close = bars["Close"].astype(float)
    idx = bars.index
    rows = []
    for _, p in posts.iterrows():
        ts = p["created_at"]
        if convention == "A_production":
            pos = idx.searchsorted(ts)
            entry_idx = pos - 1
            exit_idx = pos + POST_BARS - 1
            pre_slice = slice(pos - PRE_BARS, pos)
        elif convention == "B_strict_pre":
            pos = idx.searchsorted(ts, side="right") - 1
            entry_idx = pos
            exit_idx = pos + POST_BARS
            pre_slice = slice(pos - PRE_BARS, pos)
        else:
            raise ValueError(convention)
        if pos < PRE_BARS or exit_idx >= len(idx) or pos < 0:
            continue
        pre_sv = sigs["signed_vol_tick"].iloc[pre_slice].sum()
        pre_vpinz = sigs["vpin_z"].iloc[pre_slice].mean()
        entry_price = close.iloc[entry_idx]
        exit_price = close.iloc[exit_idx]
        pnl = pre_sv * (exit_price - entry_price)
        rows.append({"post_id": p["id"], "ts": ts, "pre_sv": pre_sv,
                     "pre_vpinz": pre_vpinz, "entry_price": entry_price,
                     "exit_price": exit_price, "pnl": pnl})
    return pd.DataFrame(rows)


def loo_summary(triggered, target_total):
    n = len(triggered)
    total = float(triggered["pnl"].sum())
    contribs = triggered["pnl"].values
    abs_share = np.abs(contribs) / abs(total) * 100 if abs(total) > 0 else np.zeros_like(contribs)
    triggered = triggered.assign(abs_share_pct=abs_share)
    by_size = triggered.reindex(triggered["pnl"].abs().sort_values(ascending=False).index)
    loo_sums = total - contribs
    return {
        "n": n,
        "total_sum_pnl_usd": total,
        "target_sum_pnl_usd": target_total,
        "mean_pnl_usd": float(np.mean(contribs)),
        "median_pnl_usd": float(np.median(contribs)),
        "hit_rate_pct": float(100 * (contribs > 0).mean()),
        "loo_sum_min_usd": float(loo_sums.min()),
        "loo_sum_max_usd": float(loo_sums.max()),
        "loo_sum_range_usd": float(loo_sums.max() - loo_sums.min()),
        "max_single_event_share_pct": float(abs_share.max()),
        "top5_events": by_size.head(5)[["ts", "pnl", "abs_share_pct", "pre_sv", "pre_vpinz"]]
                              .to_dict(orient="records"),
    }


def main():
    ensure_dirs()
    posts = pd.read_parquet(POSTS_PARQUET)
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    oil = posts[posts["topic_energy_oil"] == True].copy()
    print(f"oil-themed posts: {len(oil)}")
    bars, sigs = load_uso()

    results = {
        "method": "Leave-one-out audit on the $159.88M triggered (vpin_z > 0.5) USO bound",
        "trigger": "pre-window mean(vpin_z) > 0.5",
        "pre_bars": PRE_BARS,
        "post_bars": POST_BARS,
        "target_headline_sum_pnl_usd": 159881804.59,
        "target_n": 81,
        "conventions": {},
    }

    for convention in ("A_production", "B_strict_pre"):
        print(f"\n=== {convention} ===")
        ev = build_events(oil, bars, sigs, convention)
        triggered = ev[ev["pre_vpinz"] > TRIGGER].copy()
        print(f"  built {len(ev)} of {len(oil)} events; triggered={len(triggered)}")
        print(f"  sum P&L: ${triggered['pnl'].sum():,.0f}")
        s = loo_summary(triggered, 159881804.59)
        print(f"  max single-event |share|: {s['max_single_event_share_pct']:.1f}%")
        print(f"  LOO sum range: ${s['loo_sum_min_usd']:,.0f} -> ${s['loo_sum_max_usd']:,.0f}")
        print("  top 5:")
        for r in s["top5_events"]:
            print(f"    {r['ts']}  pnl=${r['pnl']:>14,.0f}  "
                  f"|share|={r['abs_share_pct']:5.1f}%  pre_sv={r['pre_sv']:>10,.0f}  "
                  f"pre_vpinz={r['pre_vpinz']:.2f}")
        results["conventions"][convention] = s

    with LOO_JSON.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {LOO_JSON}")


if __name__ == "__main__":
    main()
