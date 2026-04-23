#!/usr/bin/env python3
"""
17_crypto_event_study.py
========================

Crypto leg of Matt's "is the same story showing up elsewhere?" question.

The pre-registered plan was: run the 5-minute OFI/VPIN event study on
BTC-USD and ETH-USD around Trump's *crypto-themed* posts during the
Jan-26 → Apr-22 2026 window. The window contains exactly **one** post
that mentions crypto policy substantively (the 3 Mar 22:02 UTC
"Genius Act / Market Structure / Crypto Agenda" post). N = 1 has no
statistical power, so the planned design is dead on arrival.

We do two analyses instead:

  A) Cross-asset placebo. Run the same event-study summary on BTC and
     ETH around the **81 oil-themed triggered events** that drive the
     headline finding. If the oil signal reflects genuinely
     oil-specific information leakage, BTC/ETH should be flat. If they
     light up, the "Trump moves all risk assets" alternative gains
     ground.

  B) Single-event case study. Walk through what happened to BTC and
     ETH in the ±30 minute window around the one crypto post. Purely
     descriptive — labelled as such.

Outputs:
  data/results/crypto_placebo.json
  data/results/crypto_case_study.json
  report/figures/fig9_crypto_placebo.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats

from _paths import POSTS_PARQUET, SIGNALS_5M, RESULTS_DIR, FIGURES_DIR

ET = ZoneInfo("America/New_York")

# Same cosmetic palette as fig1/fig2/fig6/fig7
NAVY = "#1f3b6e"
LIGHT_BLUE = "#7aa9d6"
GREY = "#9aa1aa"
ORANGE = "#d97c2a"
RED = "#b73a3a"
GREEN = "#3a7d5a"

# Match script 04 / oil pipeline
PRE_BARS = 6
POST_BARS = 6
TRIGGER_VPINZ = 0.5
BOOTSTRAP_DRAWS = 5000
RNG_SEED = 42

CRYPTO_ASSETS = ["BTC-USD", "ETH-USD"]
OIL_ASSETS = ["XLE", "USO"]   # for side-by-side comparison panels

SIGNAL_COLS = [
    "logret", "vol_z", "dvol_z", "OFI_bvc", "vpin_z",
]


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def load_signal(asset: str) -> pd.DataFrame:
    df = pd.read_parquet(SIGNALS_5M / f"{asset}.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def nearest_bar_index(idx, ts):
    pos = idx.searchsorted(ts, side="right") - 1
    return int(pos) if pos >= 0 else -1


def gather_window(signals: pd.DataFrame, t_idx: int):
    lo, hi = t_idx - PRE_BARS, t_idx + POST_BARS + 1
    if lo < 0 or hi > len(signals):
        return None
    w = signals.iloc[lo:hi].copy()
    w["offset"] = np.arange(-PRE_BARS, POST_BARS + 1)
    return w


def bootstrap_p(values, draws, rng):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return float("nan")
    obs = v.mean()
    v0 = v - obs
    sims = rng.choice(v0, size=(draws, len(v)), replace=True).mean(axis=1)
    return float(2 * min((sims >= obs).mean(), (sims <= obs).mean()))


# ────────────────────────────────────────────────────────────────────────
# A) Placebo around oil-themed triggered events
# ────────────────────────────────────────────────────────────────────────

def run_placebo(events: pd.DataFrame) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    out = {
        "method": "Cross-asset placebo: same ±30-min event window summary "
                  "around the 81 oil-themed triggered events, computed for "
                  "BTC-USD and ETH-USD.",
        "n_events_input": int(len(events)),
        "pre_bars": PRE_BARS, "post_bars": POST_BARS,
        "by_asset": {},
    }

    for asset in CRYPTO_ASSETS + OIL_ASSETS:
        sig = load_signal(asset)
        bar_idx = sig.index
        windows = []
        for ts in events["ts"]:
            t_idx = nearest_bar_index(bar_idx, ts)
            if t_idx < PRE_BARS or t_idx >= len(sig) - POST_BARS:
                continue
            w = gather_window(sig, t_idx)
            if w is not None:
                windows.append(w)

        if len(windows) < 5:
            out["by_asset"][asset] = {"n": len(windows), "note": "too few windows"}
            continue

        n = len(windows)
        w_len = windows[0].shape[0]
        offsets = windows[0]["offset"].values
        pre_mask = (offsets >= -PRE_BARS) & (offsets < 0)
        post_mask = (offsets > 0) & (offsets <= POST_BARS)

        asset_res = {"n": n, "offsets": offsets.tolist(), "signals": {}}
        for col in SIGNAL_COLS:
            mat = np.full((n, w_len), np.nan)
            for i, w in enumerate(windows):
                mat[i, :] = w[col].values
            by_offset_mean = np.nanmean(mat, axis=0)
            by_offset_se = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n)
            pre_vals = np.nanmean(mat[:, pre_mask], axis=1)
            post_vals = np.nanmean(mat[:, post_mask], axis=1)
            t_pre, p_pre = stats.ttest_1samp(pre_vals[np.isfinite(pre_vals)], 0.0)
            t_post, p_post = stats.ttest_1samp(post_vals[np.isfinite(post_vals)], 0.0)
            boot_pre = bootstrap_p(pre_vals, BOOTSTRAP_DRAWS, rng)
            boot_post = bootstrap_p(post_vals, BOOTSTRAP_DRAWS, rng)
            asset_res["signals"][col] = {
                "by_offset_mean": by_offset_mean.tolist(),
                "by_offset_se": by_offset_se.tolist(),
                "pre_mean": float(np.nanmean(pre_vals)),
                "pre_t": float(t_pre) if np.isfinite(t_pre) else None,
                "pre_p": float(p_pre) if np.isfinite(p_pre) else None,
                "pre_boot_p": boot_pre if np.isfinite(boot_pre) else None,
                "post_mean": float(np.nanmean(post_vals)),
                "post_t": float(t_post) if np.isfinite(t_post) else None,
                "post_p": float(p_post) if np.isfinite(p_post) else None,
                "post_boot_p": boot_post if np.isfinite(boot_post) else None,
            }
        out["by_asset"][asset] = asset_res
    return out


# ────────────────────────────────────────────────────────────────────────
# B) Case study around the single crypto-themed post
# ────────────────────────────────────────────────────────────────────────

def run_case_study(post_ts: pd.Timestamp) -> dict:
    out = {
        "method": "Descriptive ±60-bar (5-hour) BTC/ETH window around the "
                  "one crypto-themed Trump post in the study window.",
        "post_ts_utc": str(post_ts),
        "by_asset": {},
    }
    for asset in CRYPTO_ASSETS:
        sig = load_signal(asset)
        t_idx = nearest_bar_index(sig.index, post_ts)
        lo, hi = max(0, t_idx - 60), min(len(sig), t_idx + 60 + 1)
        w = sig.iloc[lo:hi].copy()
        w["offset"] = np.arange(lo - t_idx, hi - t_idx)
        out["by_asset"][asset] = {
            "n_bars": int(len(w)),
            "ts_first": str(w.index.min()),
            "ts_last": str(w.index.max()),
            "close_at_event": float(w.loc[w["offset"] == 0, "Close"].iloc[0])
                              if (w["offset"] == 0).any() else None,
            "ret_pre30m_sum": float(w.loc[w["offset"].between(-6, -1), "logret"].sum()),
            "ret_post30m_sum": float(w.loc[w["offset"].between(1, 6), "logret"].sum()),
            "ret_pre1h_sum": float(w.loc[w["offset"].between(-12, -1), "logret"].sum()),
            "ret_post1h_sum": float(w.loc[w["offset"].between(1, 12), "logret"].sum()),
            "vpin_z_pre30m_mean": float(w.loc[w["offset"].between(-6, -1), "vpin_z"].mean()),
            "vpin_z_post30m_mean": float(w.loc[w["offset"].between(1, 6), "vpin_z"].mean()),
            "OFI_pre30m_mean": float(w.loc[w["offset"].between(-6, -1), "OFI_bvc"].mean()),
            "OFI_post30m_mean": float(w.loc[w["offset"].between(1, 6), "OFI_bvc"].mean()),
        }
    return out


# ────────────────────────────────────────────────────────────────────────
# Figure
# ────────────────────────────────────────────────────────────────────────

def plot_placebo(placebo: dict) -> None:
    """4×2 panel: rows = (XLE, USO, BTC-USD, ETH-USD); cols = (OFI, vpin_z).

    Pre-event lift on the oil rows, flat on the crypto rows, is the
    intended visual contrast.
    """
    rows = ["XLE", "USO", "BTC-USD", "ETH-USD"]
    cols = ["OFI_bvc", "vpin_z"]
    col_titles = {
        "OFI_bvc": "Order Flow Imbalance (BVC)",
        "vpin_z":  "VPIN-z (volume-toxicity z-score)",
    }
    asset_label = {
        "XLE": "XLE  (oil — primary headline)",
        "USO": "USO  (oil — primary headline)",
        "BTC-USD": "BTC  (cross-asset placebo)",
        "ETH-USD": "ETH  (cross-asset placebo)",
    }
    asset_color = {
        "XLE": NAVY, "USO": NAVY,
        "BTC-USD": ORANGE, "ETH-USD": ORANGE,
    }

    fig, axes = plt.subplots(
        len(rows), len(cols), figsize=(11, 11),
        gridspec_kw=dict(hspace=0.55, wspace=0.30,
                         left=0.10, right=0.97,
                         top=0.93, bottom=0.06),
    )

    for i, asset in enumerate(rows):
        a = placebo["by_asset"].get(asset, {})
        for j, col in enumerate(cols):
            ax = axes[i, j]
            sigs = a.get("signals", {}).get(col)
            if sigs is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color=GREY)
                ax.axis("off")
                continue
            offs = np.array(a["offsets"])
            mu = np.array(sigs["by_offset_mean"])
            se = np.array(sigs["by_offset_se"])
            ax.fill_between(offs, mu - 1.96 * se, mu + 1.96 * se,
                            color=asset_color[asset], alpha=0.15)
            ax.plot(offs, mu, color=asset_color[asset], linewidth=1.6)
            ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.axhline(0, color=GREY, linewidth=0.5, alpha=0.6)
            ax.grid(axis="y", alpha=0.25)

            # Annotate pre p-value
            pp = sigs.get("pre_boot_p")
            sig_pre = sigs.get("pre_mean", 0.0)
            txt = f"pre mean = {sig_pre:+.3f}\np = {pp:.3g}" if pp is not None and np.isfinite(pp) else "pre p = n/a"
            ax.text(0.02, 0.97, txt, transform=ax.transAxes,
                    ha="left", va="top", fontsize=8.5,
                    bbox=dict(facecolor="white", edgecolor=GREY,
                              boxstyle="round,pad=0.3", alpha=0.85))

            if i == 0:
                ax.set_title(col_titles[col], fontsize=10.5, weight="bold")
            if j == 0:
                ax.set_ylabel(asset_label[asset], fontsize=9.5)
            if i == len(rows) - 1:
                ax.set_xlabel("bar offset from post (5-min bars)", fontsize=9)

    fig.suptitle(
        f"Cross-asset placebo: oil-themed events ({placebo['n_events_input']} triggered) "
        f"in oil tickers vs. BTC/ETH",
        fontsize=12, weight="bold", y=0.985,
    )

    out_pdf = FIGURES_DIR / "fig9_crypto_placebo.pdf"
    out_png = FIGURES_DIR / "fig9_crypto_placebo.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_pdf}")
    print(f"  wrote {out_png}")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("loading 81 triggered oil-themed events ...")
    ev = pd.read_csv(RESULTS_DIR / "dollar_upper_bound_uso_events.csv")
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, format="ISO8601")
    triggered = ev[ev["pre_vpinz"] > TRIGGER_VPINZ].copy()
    print(f"  {len(triggered)} triggered events")

    print("\nrunning cross-asset placebo on BTC/ETH (and oil controls) ...")
    placebo = run_placebo(triggered)
    out_p = RESULTS_DIR / "crypto_placebo.json"
    with out_p.open("w") as f:
        json.dump(placebo, f, indent=2, default=str)
    print(f"  wrote {out_p}")

    # Print a quick summary table
    print("\n== Pre-event mean (bootstrap two-sided p) ==")
    for asset in ["XLE", "USO", "BTC-USD", "ETH-USD"]:
        a = placebo["by_asset"].get(asset, {})
        sigs = a.get("signals", {})
        print(f"  {asset:9s} (n={a.get('n','-')}):")
        for col in ["OFI_bvc", "vpin_z"]:
            s = sigs.get(col, {})
            print(f"      {col:9s}  pre_mean={s.get('pre_mean', float('nan')):+.4f}   "
                  f"boot_p={s.get('pre_boot_p', float('nan'))!s:>10}")

    # Locate the single crypto post
    posts = pd.read_parquet(POSTS_PARQUET).reset_index()
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
    crypto_posts = posts[posts["topic_crypto"] == True]
    print(f"\ncase study: {len(crypto_posts)} crypto-themed post(s) in window")

    case_study = {"posts": []}
    for _, row in crypto_posts.iterrows():
        cs = run_case_study(row["created_at"])
        cs["text_snippet"] = row["text"][:220]
        case_study["posts"].append(cs)
    out_cs = RESULTS_DIR / "crypto_case_study.json"
    with out_cs.open("w") as f:
        json.dump(case_study, f, indent=2, default=str)
    print(f"  wrote {out_cs}")
    for cs in case_study["posts"]:
        print(f"\n  post @ {cs['post_ts_utc']}")
        for asset, m in cs["by_asset"].items():
            print(f"    {asset}:  ret_pre30m={m['ret_pre30m_sum']:+.4f}, "
                  f"ret_post30m={m['ret_post30m_sum']:+.4f},  "
                  f"vpin_z_pre30m={m['vpin_z_pre30m_mean']:+.2f},  "
                  f"vpin_z_post30m={m['vpin_z_post30m_mean']:+.2f}")

    print("\nbuilding figure ...")
    plot_placebo(placebo)


if __name__ == "__main__":
    main()
