#!/usr/bin/env python3
"""
08_dollar_upper_bound.py
========================

Back-of-envelope dollar P&L upper bound on USO.

Premise: if a single perfectly-informed trader had taken the entire net
pre-event signed-volume imbalance as their position at the last pre-event
bar close, and exited at the last post-window bar close, how much money
would they have made (gross)?

This is a CEILING, not a claim. It ignores:
  - transaction costs and bid-ask spread
  - market impact from their own order
  - borrow/financing costs
  - the realism of having that position at that price

Reports three slices:
  (1) ALL  — every oil-themed event with a usable window
  (2) INITIATED  — those that pass the |pre-CAR z| < 1.5 filter
  (3) INITIATED + PRE-BUYING  — initiated events where pre_sv > 0
  (4) TRIGGERED (vpin_z > 0.5)  — the slice headlined in the report

Outputs:
  data/results/dollar_upper_bound_strategies.json
  data/results/dollar_upper_bound_uso_events.csv     (per-event detail)
"""

import json

import numpy as np
import pandas as pd

from _paths import POSTS_PARQUET, MINUTE_BARS_5M, SIGNALS_5M, DOLLAR_BOUND_JSON, RESULTS_DIR, ensure_dirs

PRE_BARS = 12      # 60 min at 5m
POST_BARS = 12
TRIGGER = 0.5      # pre-window mean(vpin_z) > 0.5


def main():
    ensure_dirs()
    posts = pd.read_parquet(POSTS_PARQUET)
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    oil = posts[posts["topic_energy_oil"] == True].copy()
    print(f"oil-themed posts: {len(oil)}")

    bars = pd.read_parquet(MINUTE_BARS_5M / "USO.parquet")
    sigs = pd.read_parquet(SIGNALS_5M / "USO.parquet")
    bars.index = pd.to_datetime(bars.index, utc=True)
    sigs.index = pd.to_datetime(sigs.index, utc=True)
    if "signed_vol_tick" not in sigs.columns:
        raise RuntimeError("signed_vol_tick missing — run stage 03 first")

    close = bars["Close"].astype(float)
    log_close = np.log(close)
    ret = log_close.diff()
    idx = bars.index

    rows = []
    for _, post in oil.iterrows():
        ts = post["created_at"]
        pos = idx.searchsorted(ts)
        if pos < PRE_BARS or pos + POST_BARS > len(idx):
            continue
        pre_slice = slice(pos - PRE_BARS, pos)
        post_slice = slice(pos, pos + POST_BARS)
        pre_sv = sigs["signed_vol_tick"].iloc[pre_slice].sum()
        pre_vpinz = sigs["vpin_z"].iloc[pre_slice].mean()
        entry_price = close.iloc[pos - 1]
        exit_price = close.iloc[post_slice].iloc[-1]
        pnl = pre_sv * (exit_price - entry_price)

        pre_ret_cum = ret.iloc[pre_slice].sum()
        pre_ret_std = ret.iloc[pos - 288:pos].std() if pos >= 288 else ret.iloc[:pos].std()
        z = pre_ret_cum / (pre_ret_std * np.sqrt(PRE_BARS)) if pre_ret_std and pre_ret_std > 0 else np.nan
        is_initiated = abs(z) < 1.5 if not np.isnan(z) else False

        rows.append({
            "post_id": post["id"], "ts": ts, "pre_sv": pre_sv, "pre_vpinz": pre_vpinz,
            "entry_price": entry_price, "exit_price": exit_price,
            "pnl": pnl, "pre_car_z": z, "initiated": is_initiated,
        })

    df = pd.DataFrame(rows).dropna(subset=["pnl"])
    print(f"events with usable window & non-null P&L: {len(df)}")

    def stats_block(d, label):
        n = int(len(d))
        s = float(d["pnl"].sum())
        m = float(d["pnl"].mean()) if n else 0.0
        med = float(d["pnl"].median()) if n else 0.0
        hr = float(100 * (d["pnl"] > 0).mean()) if n else 0.0
        print(f"\n  [{label}] n={n}  sum=${s:,.0f}  mean=${m:,.0f}  median=${med:,.0f}  hit_rate={hr:.1f}%")
        return {
            "n": n, "sum_pnl_usd": s, "mean_pnl_usd": m, "median_pnl_usd": med,
            "hit_rate_pct": hr,
            "p25_pnl_usd": float(d["pnl"].quantile(0.25)) if n else 0.0,
            "p75_pnl_usd": float(d["pnl"].quantile(0.75)) if n else 0.0,
        }

    out = {
        "method": "Per-event upper bound P&L for the USO oil-finding.",
        "pre_bars": PRE_BARS,
        "post_bars": POST_BARS,
        "definition": ("Per-event P&L = pre-event signed-volume imbalance (shares) "
                       "× (post-window-exit price − last pre-event bar close). "
                       "Entry at close.iloc[pos-1]; exit at close.iloc[pos+POST_BARS-1]."),
        "interpretation": ("Ceiling on what a perfectly-informed trader holding the "
                           "entire pre-event imbalance could have netted — gross of "
                           "transaction costs, spread, market impact, and borrow."),
        "all_events": stats_block(df, "ALL"),
        "initiated_events": stats_block(df[df["initiated"]], "INITIATED"),
        "initiated_and_pre_buying": stats_block(
            df[(df["initiated"]) & (df["pre_sv"] > 0)], "INITIATED + pre-buying"),
        "triggered_vpinz_gt_0_5": stats_block(df[df["pre_vpinz"] > TRIGGER],
                                              "TRIGGERED (vpin_z > 0.5)"),
    }

    with DOLLAR_BOUND_JSON.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    df.to_csv(RESULTS_DIR / "dollar_upper_bound_uso_events.csv", index=False)
    print(f"\nwrote {DOLLAR_BOUND_JSON}")
    print(f"wrote {RESULTS_DIR/'dollar_upper_bound_uso_events.csv'}")


if __name__ == "__main__":
    main()
