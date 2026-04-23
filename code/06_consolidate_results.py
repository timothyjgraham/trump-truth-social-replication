#!/usr/bin/env python3
"""
06_consolidate_results.py
=========================

Consolidate the real event study (stage 04) and the matched placebo
(stage 05) into a single robust-findings table.

For each (topic, asset, metric, window=pre/post):
  real_mean, real_p, p_fdr, placebo_mean, placebo_p, diff_vs_placebo,
  real_sig, placebo_sig, robust

A finding is flagged "robust" when:
  (a) raw bootstrap p < 0.05 on real events
  (b) placebo p > 0.05 on the same (asset, metric, window)
  (c) BH-FDR q < 0.1 on real p

Outputs:
  data/results/orderflow_final_5m.csv
  data/results/orderflow_final_5m.json
"""

import json

import numpy as np
import pandas as pd

from _paths import EVENT_STUDY_5M_JSON, PLACEBO_5M_JSON, RESULTS_DIR, ensure_dirs

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
    adj = sorted_p * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.full(len(p), np.nan)
    out[mask] = adj[ranks]
    return out


def flatten(res, label):
    rows = []
    for topic, assets in res["results"].items():
        for asset, sub in assets.items():
            if not isinstance(sub, dict) or sub.get("insufficient"):
                continue
            for col, s in sub.get("signals", {}).items():
                if col not in METRICS:
                    continue
                rows.append({"source": label, "topic": topic, "asset": asset,
                             "metric": col, "window": "pre",
                             "mean": s["pre_mean"], "p": s.get("pre_p"),
                             "n": s.get("n_pre") or sub.get("n")})
                rows.append({"source": label, "topic": topic, "asset": asset,
                             "metric": col, "window": "post",
                             "mean": s["post_mean"], "p": s.get("post_p"),
                             "n": s.get("n_post") or sub.get("n")})
    return rows


def main():
    ensure_dirs()
    with EVENT_STUDY_5M_JSON.open() as f:
        main_res = json.load(f)
    real_results = {"results": {}}
    for topic, assets in main_res["results"].items():
        real_results["results"][topic] = {}
        for asset, sub in assets.items():
            ini = sub.get("initiated") if isinstance(sub, dict) else None
            if ini is None:
                real_results["results"][topic][asset] = {"insufficient": True}
                continue
            real_results["results"][topic][asset] = {
                "n": ini["n"],
                "signals": {col: {
                    "pre_mean": s["pre_mean"], "pre_p": s.get("pre_boot_p") or s.get("pre_p"),
                    "post_mean": s["post_mean"], "post_p": s.get("post_boot_p") or s.get("post_p"),
                } for col, s in ini.get("signals", {}).items()},
            }

    real_df = pd.DataFrame(flatten(real_results, "real"))
    real_df["p_fdr"] = bh_fdr(real_df["p"].values)

    with PLACEBO_5M_JSON.open() as f:
        placebo_raw = json.load(f)
    placebo_df = pd.DataFrame(flatten(placebo_raw, "placebo"))
    placebo_df = (placebo_df.rename(columns={"mean": "placebo_mean", "p": "placebo_p", "n": "placebo_n"})
                            [["asset", "metric", "window", "placebo_mean", "placebo_p", "placebo_n"]])

    merged = real_df.merge(placebo_df, on=["asset", "metric", "window"], how="left")
    merged["diff_vs_placebo"] = merged["mean"] - merged["placebo_mean"]
    merged["real_sig"] = merged["p"] < 0.05
    merged["placebo_sig"] = merged["placebo_p"] < 0.05
    merged["robust"] = merged["real_sig"] & ~merged["placebo_sig"] & (merged["p_fdr"] < 0.1)

    survivors = merged[merged["robust"]].sort_values("p_fdr")
    print(f"\ntotal tests:           {len(merged)}")
    print(f"raw p<0.05 on real:    {int(merged['real_sig'].sum())}")
    print(f"placebo sig (same row):{int(merged['placebo_sig'].sum())}")
    print(f"robust:                {len(survivors)}")
    print("\ntop 20 robust findings:")
    if len(survivors) > 0:
        print(survivors[["topic", "asset", "metric", "window", "mean", "p", "p_fdr",
                         "placebo_mean", "placebo_p", "diff_vs_placebo", "n"]]
              .head(20).to_string())

    csv_path = RESULTS_DIR / "orderflow_final_5m.csv"
    json_path = RESULTS_DIR / "orderflow_final_5m.json"
    merged.to_csv(csv_path, index=False)
    with json_path.open("w") as f:
        json.dump({
            "total_tests": int(len(merged)),
            "raw_sig": int(merged["real_sig"].sum()),
            "placebo_sig": int(merged["placebo_sig"].sum()),
            "robust": int(len(survivors)),
            "survivors": survivors.to_dict(orient="records"),
        }, f, default=str, indent=2)
    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
