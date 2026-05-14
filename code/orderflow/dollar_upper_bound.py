#!/usr/bin/env python3
"""
Back-of-envelope dollar P&L upper bound for the USO finding.

Premise: if a single informed trader had the entire net pre-event buying imbalance
in USO as their position, held through the post, and exited at the post-window end,
how much did they make?

This is the "perfect-trader-with-the-signal" upper bound. It is NOT evidence that
anyone actually made this money. Published alongside the finding with that caveat.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

D = Path("/sessions/sleepy-gifted-ptolemy/work/data")
PRE_BARS = 12   # 12 × 5min = 60 minutes pre-event
POST_BARS = 12

# ── 1. Load posts + topic mask ────────────────────────────────────────────────
posts = pd.read_parquet(D / "posts_60d.parquet")
posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True)
oil = posts[posts["topic_energy_oil"] == True].copy()
print(f"Oil-themed posts (topic_energy_oil=True): {len(oil)}")

# ── 2. Load USO bars and signals ──────────────────────────────────────────────
bars = pd.read_parquet(D / "minute_bars/5m/USO.parquet")
sigs = pd.read_parquet(D / "signals/5m/USO.parquet")
bars.index = pd.to_datetime(bars.index, utc=True)
sigs.index = pd.to_datetime(sigs.index, utc=True)
print(f"USO bars: {len(bars)} rows, range {bars.index.min()} → {bars.index.max()}")
print(f"USO signal cols: {list(sigs.columns)[:15]}")

# We need: signed_vol_tick per bar (net shares), Close price per bar
if "signed_vol_tick" not in sigs.columns:
    raise RuntimeError("signed_vol_tick missing — check signals build")

# Make sure bars cover the post range
idx = bars.index

# ── 3. For each event, snap to next bar at or after post time; build windows ──
def idx_at_or_after(ts):
    pos = idx.searchsorted(ts)
    if pos >= len(idx):
        return None
    return pos

# "Initiated" filter — use pre-CAR over prior 12 bars, classify |z|<1.5 as initiated
# Match the main study's definition.
close = bars["Close"].astype(float)
log_close = np.log(close)
ret = log_close.diff()

rows = []
for _, post in oil.iterrows():
    ts = post["created_at"]
    pos = idx_at_or_after(ts)
    if pos is None or pos < PRE_BARS or pos + POST_BARS > len(idx):
        continue

    # windows
    pre_slice  = slice(pos - PRE_BARS, pos)
    post_slice = slice(pos, pos + POST_BARS)

    # Pre-event "informed buying" imbalance (net aggressive shares)
    pre_sv = sigs["signed_vol_tick"].iloc[pre_slice].sum()

    # Prices: entry = mean close over pre window; exit = close at post-window end
    pre_price_mean  = close.iloc[pre_slice].mean()
    exit_price      = close.iloc[post_slice].iloc[-1]
    entry_price     = close.iloc[pos - 1]  # price at the moment of the post (last pre bar close)
    post_price_mean = close.iloc[post_slice].mean()
    post_return     = np.log(exit_price / entry_price)  # log return across post window
    post_return_pct = exit_price / entry_price - 1.0

    # "Initiated" flag
    pre_ret_cum = ret.iloc[pre_slice].sum()
    pre_ret_std = ret.iloc[pos - 288:pos].std() if pos >= 288 else ret.iloc[:pos].std()
    pre_car_z = pre_ret_cum / (pre_ret_std * np.sqrt(PRE_BARS)) if pre_ret_std and pre_ret_std > 0 else np.nan
    is_initiated = abs(pre_car_z) < 1.5 if not np.isnan(pre_car_z) else False

    # Per-event upper-bound dollar P&L
    # If someone had pre_sv shares as position at entry_price, exits at exit_price
    pnl_per_event = pre_sv * (exit_price - entry_price)

    rows.append({
        "post_id": post["id"],
        "ts": ts,
        "pre_sv": pre_sv,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "post_return_log": post_return,
        "post_return_pct": post_return_pct,
        "pnl": pnl_per_event,
        "pre_car_z": pre_car_z,
        "initiated": is_initiated,
    })

df = pd.DataFrame(rows)
print(f"\nBuilt P&L rows: {len(df)} (of {len(oil)} oil posts)")
df = df.dropna(subset=["pnl"])
print(f"With non-null P&L: {len(df)}")

# ── 4. Aggregate ──────────────────────────────────────────────────────────────
print("\n=== Dollar upper bound (ALL oil-themed events) ===")
all_stats = {
    "n": int(len(df)),
    "sum_pnl_usd": float(df["pnl"].sum()),
    "median_pnl_usd": float(df["pnl"].median()),
    "mean_pnl_usd": float(df["pnl"].mean()),
    "p25_pnl_usd": float(df["pnl"].quantile(0.25)),
    "p75_pnl_usd": float(df["pnl"].quantile(0.75)),
    "pnl_hits_positive": int((df["pnl"] > 0).sum()),
    "total_pre_sv_shares": float(df["pre_sv"].sum()),
    "mean_pre_sv_shares": float(df["pre_sv"].mean()),
    "mean_post_return_pct": float(df["post_return_pct"].mean()),
}
for k, v in all_stats.items(): print(f"  {k:28s}  {v:>15,.2f}" if isinstance(v, float) else f"  {k:28s}  {v:>15,}")

print("\n=== Dollar upper bound (INITIATED oil events only — matches story finding) ===")
ini = df[df["initiated"]].copy()
ini_stats = {
    "n": int(len(ini)),
    "sum_pnl_usd": float(ini["pnl"].sum()),
    "median_pnl_usd": float(ini["pnl"].median()),
    "mean_pnl_usd": float(ini["pnl"].mean()),
    "p25_pnl_usd": float(ini["pnl"].quantile(0.25)),
    "p75_pnl_usd": float(ini["pnl"].quantile(0.75)),
    "pnl_hits_positive": int((ini["pnl"] > 0).sum()),
    "hit_rate_pct": float(100 * (ini["pnl"] > 0).mean()),
    "total_pre_sv_shares": float(ini["pre_sv"].sum()),
    "mean_pre_sv_shares": float(ini["pre_sv"].mean()),
    "mean_post_return_pct": float(ini["post_return_pct"].mean()),
}
for k, v in ini_stats.items(): print(f"  {k:28s}  {v:>15,.2f}" if isinstance(v, float) else f"  {k:28s}  {v:>15,}")

# Bootstrap CI on mean P&L (initiated)
rng = np.random.default_rng(42)
pnls = ini["pnl"].values
boot_means = [rng.choice(pnls, size=len(pnls), replace=True).mean() for _ in range(5000)]
ci_low = float(np.percentile(boot_means, 2.5))
ci_high = float(np.percentile(boot_means, 97.5))
print(f"\n  Bootstrap 95% CI on mean P&L per event: [${ci_low:,.0f}, ${ci_high:,.0f}]")
ini_stats["mean_pnl_ci95"] = [ci_low, ci_high]
ini_stats["total_pnl_ci95"] = [ci_low * len(ini), ci_high * len(ini)]

# Additional framing: only positive-imbalance events (where signal said "buy")
print("\n=== Conditional on PRE-EVENT BUYING (pre_sv > 0), initiated oil events ===")
buys = ini[ini["pre_sv"] > 0].copy()
buy_stats = {
    "n": int(len(buys)),
    "sum_pnl_usd": float(buys["pnl"].sum()),
    "median_pnl_usd": float(buys["pnl"].median()),
    "mean_pnl_usd": float(buys["pnl"].mean()),
    "hit_rate_pct": float(100 * (buys["pnl"] > 0).mean()),
    "total_pre_sv_shares": float(buys["pre_sv"].sum()),
    "mean_pre_sv_shares": float(buys["pre_sv"].mean()),
    "mean_post_return_pct": float(buys["post_return_pct"].mean()),
}
for k, v in buy_stats.items(): print(f"  {k:28s}  {v:>15,.2f}" if isinstance(v, float) else f"  {k:28s}  {v:>15,}")

# ── 5. Save summary + detail ──────────────────────────────────────────────────
out = {
    "method": "Back-of-envelope dollar P&L upper bound for the USO finding.",
    "definition": "Per-event P&L = pre-event signed-volume imbalance (shares) × (post-window-exit price − pre-window-close price). Entry at the last pre-event bar close; exit at the last post-window bar close. Post-window = 60 min.",
    "interpretation": "This is the maximum gross P&L available to a perfectly-informed trader who took the entire pre-event net imbalance as their position. Not a claim anyone actually made this money. Ignores transaction costs, bid-ask spread, borrowing costs, and any market impact from their own order.",
    "pre_bars": PRE_BARS,
    "post_bars": POST_BARS,
    "event_filter": "topic_energy_oil == True",
    "all_events": all_stats,
    "initiated_events": ini_stats,
    "initiated_and_pre_buying": buy_stats,
}
with open(D / "dollar_upper_bound_uso.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

df.to_csv(D / "dollar_upper_bound_uso_events.csv", index=False)
print(f"\nSaved: {D/'dollar_upper_bound_uso.json'}")
print(f"Saved: {D/'dollar_upper_bound_uso_events.csv'}")
