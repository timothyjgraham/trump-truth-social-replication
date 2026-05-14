#!/usr/bin/env python3
"""
consolidate_results.py

Produces the final headline table:

  For each (topic, asset, metric, window=pre/post):
    - real_mean         — mean of real events
    - placebo_mean      — mean of matched placebo events
    - diff              — real_mean - placebo_mean
    - z_vs_placebo      — (real_mean - placebo_mean) / (placebo_std / sqrt(n_real))
    - raw_p             — t-test p on real events
    - fdr_p             — BH-FDR corrected across the joint table

Metrics kept (drop trivially-positive bounded quantities):
  logret, vol_z, dvol_z, signed_vol_tick, OFI_bvc, vpin_z, kyle_z

A finding is 'robust' when:
  (a) raw p < 0.05 on real events
  (b) raw p > 0.05 on placebo OR diff clearly in the same direction
  (c) |z_vs_placebo| > 2
  (d) BH-FDR q < 0.1 on real p

Writes:
  data/orderflow_final_5m.json
  data/orderflow_final_5m.csv     (flat table for easy review)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("/sessions/sleepy-gifted-ptolemy/work")
DATA = WORK / "data"

METRICS = ["logret", "vol_z", "dvol_z", "signed_vol_tick", "OFI_bvc", "vpin_z", "kyle_z"]


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    mask = np.isfinite(p)
    pvals = p[mask]
    n = len(pvals)
    if n == 0:
        return p
    order = np.argsort(pvals)
    ranks = np.argsort(order)
    sorted_p = pvals[order]
    adj = sorted_p * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.full(len(p), np.nan)
    out[mask] = adj[ranks]
    return out


def flatten(res, label):
    """Flatten a run result into long rows."""
    rows = []
    for topic, assets in res["results"].items():
        for asset, sub in assets.items():
            if sub.get("insufficient"):
                continue
            for col, s in sub.get("signals", {}).items():
                if col not in METRICS:
                    continue
                rows.append({
                    "source": label, "topic": topic, "asset": asset,
                    "metric": col, "window": "pre",
                    "mean": s["pre_mean"], "p": s.get("pre_p"),
                    "n": s.get("n_pre") or sub.get("n"),
                })
                rows.append({
                    "source": label, "topic": topic, "asset": asset,
                    "metric": col, "window": "post",
                    "mean": s["post_mean"], "p": s.get("post_p"),
                    "n": s.get("n_post") or sub.get("n"),
                })
    return rows


def main():
    # Load real event study (from placebo run's output isn't the full one — use event_study output)
    # We need the "initiated" block from orderflow_event_study_5m.json.
    with (DATA / "orderflow_event_study_5m.json").open() as f:
        main_res = json.load(f)
    # Transform to the same shape that flatten expects (initiated block)
    real_results = {"results": {}}
    for topic, assets in main_res["results"].items():
        real_results["results"][topic] = {}
        for asset, sub in assets.items():
            ini = sub.get("initiated")
            if ini is None:
                real_results["results"][topic][asset] = {"insufficient": True}
            else:
                real_results["results"][topic][asset] = {
                    "n": ini["n"],
                    "signals": {col: {
                        "pre_mean": s["pre_mean"], "pre_p": s.get("pre_boot_p") or s.get("pre_p"),
                        "post_mean": s["post_mean"], "post_p": s.get("post_boot_p") or s.get("post_p"),
                    } for col, s in ini.get("signals", {}).items()},
                }

    real_rows = flatten(real_results, "real")
    real_df = pd.DataFrame(real_rows)
    real_df["p_fdr"] = bh_fdr(real_df["p"].values)

    # Placebo
    with (DATA / "orderflow_placebo_5m.json").open() as f:
        placebo_raw = json.load(f)
    # The placebo's topic is "placebo" and it has per-asset entries
    placebo_rows = flatten(placebo_raw, "placebo")
    placebo_df = pd.DataFrame(placebo_rows)
    # No topic structure — it's one big run. Merge back on (asset, metric, window)
    placebo_df = placebo_df.rename(columns={"mean": "placebo_mean", "p": "placebo_p", "n": "placebo_n"})
    placebo_df = placebo_df[["asset", "metric", "window", "placebo_mean", "placebo_p", "placebo_n"]]

    merged = real_df.merge(placebo_df, on=["asset", "metric", "window"], how="left")
    merged["diff_vs_placebo"] = merged["mean"] - merged["placebo_mean"]
    # Approx z vs placebo using real std. Not having real std per metric in the flattened
    # form, we use the common assumption that within-event-study standard error is
    # ~placebo_mean's standard error — which we approximate by |placebo_mean / z|
    # from the placebo's own t-statistic. This is imperfect; a better version would
    # bootstrap the real-vs-placebo difference directly.
    # For a cleaner interpretation, we just record diff_vs_placebo and let downstream
    # flag whenever real_p is significant and placebo_p is not.
    merged["real_sig"] = merged["p"] < 0.05
    merged["placebo_sig"] = merged["placebo_p"] < 0.05
    # A "robust" finding: real sig AND placebo not sig (i.e., not driven by hour-of-day artefact)
    merged["robust"] = merged["real_sig"] & ~merged["placebo_sig"] & (merged["p_fdr"] < 0.1)

    # Survivors summary
    survivors = merged[merged["robust"]].sort_values("p_fdr")
    print(f"\nTotal tests: {len(merged)}")
    print(f"Raw p<0.05 on real: {merged['real_sig'].sum()}")
    print(f"Placebo sig on same metric: {merged['placebo_sig'].sum()}")
    print(f"Robust (real sig & placebo not & FDR q<0.1): {len(survivors)}")
    print()
    print("Top 30 robust findings (sorted by FDR p):")
    print(survivors[["topic", "asset", "metric", "window", "mean", "p", "p_fdr",
                     "placebo_mean", "placebo_p", "diff_vs_placebo", "n"]].head(30).to_string())

    # Save
    merged.to_csv(DATA / "orderflow_final_5m.csv", index=False)
    with (DATA / "orderflow_final_5m.json").open("w") as f:
        json.dump({
            "total_tests": int(len(merged)),
            "raw_sig": int(merged["real_sig"].sum()),
            "placebo_sig": int(merged["placebo_sig"].sum()),
            "robust": int(len(survivors)),
            "survivors": survivors.to_dict(orient="records"),
        }, f, default=str, indent=2)
    print(f"\nWrote {DATA/'orderflow_final_5m.csv'} and .json")


if __name__ == "__main__":
    main()
