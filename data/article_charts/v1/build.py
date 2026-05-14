"""
Build the four chart-ready datasets for Matt.

Outputs land in: <repo>/matt_chart_data/
Run from the trump-truth-social-replication repo root.

Decisions I landed on for the composition of the charts (Tim Graham):
  - Chart 1 (event window): 15 collapsed bursts, USO only, cum % return, +-60 min
  - Chart 2 (bubble): topic co-occurrence in posts (8 topic tags)
  - Chart 3 (heatmap): rows = topics, cols = assets, cell = pre-event mean signal
  - Chart 4 (timeline): USO close + post markers + daily post counts
"""
from __future__ import annotations

import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/sessions/laughing-vigilant-gauss/mnt/Trump_Truth_Social_analysis/trump-truth-social-replication")
OUT  = Path("/sessions/laughing-vigilant-gauss/mnt/Trump_Truth_Social_analysis/matt_chart_data")
OUT.mkdir(parents=True, exist_ok=True)

BAR_MIN     = 5            # data is in 5-min bars
WINDOW_MIN  = 60           # +/- 60 minutes around t=0
STAT_WINDOW = 30           # the rigorous stat window we used

TOPIC_COLS = [
    "topic_tariff_trade",
    "topic_fed_rates",
    "topic_china",
    "topic_iran_military",
    "topic_energy_oil",
    "topic_market_economy",
    "topic_crypto",
    "topic_djt_media",
]
TOPIC_LABELS = {
    "topic_tariff_trade":   "Tariffs / Trade",
    "topic_fed_rates":      "Fed / Rates",
    "topic_china":          "China",
    "topic_iran_military":  "Iran / Military",
    "topic_energy_oil":     "Energy / Oil",
    "topic_market_economy": "Markets / Economy",
    "topic_crypto":         "Crypto",
    "topic_djt_media":      "DJT / Media",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_uso_bars() -> pd.DataFrame:
    df = pd.read_parquet(REPO / "data/raw/minute_bars_5m/USO.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_xle_bars() -> pd.DataFrame:
    df = pd.read_parquet(REPO / "data/raw/minute_bars_5m/XLE.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_posts() -> pd.DataFrame:
    p = pd.read_parquet(REPO / "data/raw/posts_60d.parquet")
    p["created_at"] = pd.to_datetime(p["created_at"], utc=True)
    return p


def load_bursts() -> pd.DataFrame:
    b = pd.read_csv(REPO / "data/results/pnl_concentration_bursts.csv")
    b["ts_first"] = pd.to_datetime(b["ts_first"], utc=True)
    b["ts_last"]  = pd.to_datetime(b["ts_last"],  utc=True)
    b = b.sort_values("ts_first").reset_index(drop=True)
    b["burst_id"] = b.index + 1
    return b


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(REPO / "data/results/signal_overlay_events.csv")
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, format="ISO8601")
    return ev


def snippet(text: str, n: int = 160) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Chart 1 - Event window (15 bursts x +-60 min, USO cum % return)
# ---------------------------------------------------------------------------
def build_chart1(uso: pd.DataFrame, bursts: pd.DataFrame, posts: pd.DataFrame) -> None:
    """Anchor each burst at t=0 = post timestamp (snapped to nearest 5-min bar).

    For weekend / overnight bursts where no bar exists at t=0, we anchor to the
    next available bar within +60 min and *flag* it as anchored_to_next_session.
    Lines for those bursts will not have data at negative offsets - that is the
    honest representation. Matt can choose to hide them in the overlay or shade
    them differently.
    """
    rows = []
    burst_meta = []

    bar_index = uso.index

    for _, b in bursts.iterrows():
        bid    = int(b["burst_id"])
        anchor = b["ts_first"]
        anchor_floor = anchor.floor(f"{BAR_MIN}min")

        # Try to find a bar at exactly the post time (5-min snapped)
        anchored_to_next_session = False
        if anchor_floor not in bar_index:
            # Weekend / overnight - use the *next* available bar within a few hours
            future_bars = bar_index[bar_index >= anchor_floor]
            if len(future_bars) == 0:
                print(f"  burst {bid}: no future bar available, skipping")
                continue
            next_bar = future_bars[0]
            gap_hours = (next_bar - anchor_floor).total_seconds() / 3600
            if gap_hours > 72:
                print(f"  burst {bid}: gap to next bar > 72h, skipping")
                continue
            anchor_floor = next_bar
            anchored_to_next_session = True

        # Pull window
        lo = anchor_floor - pd.Timedelta(minutes=WINDOW_MIN)
        hi = anchor_floor + pd.Timedelta(minutes=WINDOW_MIN)
        win = uso.loc[(uso.index >= lo) & (uso.index <= hi)].copy()

        anchor_price = float(win.loc[anchor_floor, "Close"])
        win["minute_offset"]  = ((win.index - anchor_floor).total_seconds() // 60).astype(int)
        win["cum_pct_return"] = (win["Close"] / anchor_price - 1) * 100.0
        win["price_close"]    = win["Close"]
        win["in_stat_window"] = win["minute_offset"].abs() <= STAT_WINDOW

        burst_posts = posts[
            (posts["created_at"] >= b["ts_first"] - pd.Timedelta(seconds=1)) &
            (posts["created_at"] <= b["ts_last"]  + pd.Timedelta(seconds=1))
        ]
        sample_text = snippet(burst_posts["text"].iloc[0]) if len(burst_posts) else ""

        for _, row in win.iterrows():
            rows.append({
                "burst_id":        bid,
                "burst_ts_anchor": anchor_floor.isoformat(),
                "minute_offset":   int(row["minute_offset"]),
                "cum_pct_return":  round(float(row["cum_pct_return"]), 4),
                "price_close":     round(float(row["price_close"]),    4),
                "anchor_price":    round(anchor_price,                 4),
                "in_stat_window":  bool(row["in_stat_window"]),
            })

        burst_meta.append({
            "burst_id":                  bid,
            "ts_first":                  b["ts_first"].isoformat(),
            "ts_last":                   b["ts_last"].isoformat(),
            "duration_minutes":          round((b["ts_last"] - b["ts_first"]).total_seconds() / 60, 2),
            "n_posts":                   int(b["n_posts"]),
            "anchor_bar_ts":             anchor_floor.isoformat(),
            "anchored_to_next_session":  anchored_to_next_session,
            "anchor_price":              round(anchor_price, 4),
            "pnl_total":                 round(float(b["pnl_total"]),         2),
            "pnl_per_post_mean":         round(float(b["pnl_per_post_mean"]), 2),
            "initiated_any":             bool(b["initiated_any"]),
            "sample_post_text":          sample_text,
            "is_apr07_burst":            (b["ts_first"].date().isoformat() == "2026-04-07"
                                          and (b["ts_last"] - b["ts_first"]).total_seconds() / 60 > 5),
        })

    df = pd.DataFrame(rows)
    meta = pd.DataFrame(burst_meta)

    # Long CSV (D3-friendly)
    df.to_csv(OUT / "chart1_event_window.csv", index=False)
    meta.to_csv(OUT / "chart1_burst_metadata.csv", index=False)

    # Nested JSON: array of burst objects, each with a "series" array
    nested = []
    for m in burst_meta:
        series = [
            {"minute_offset": r["minute_offset"],
             "cum_pct_return": r["cum_pct_return"],
             "in_stat_window": r["in_stat_window"]}
            for r in rows if r["burst_id"] == m["burst_id"]
        ]
        series.sort(key=lambda x: x["minute_offset"])
        nested.append({**m, "series": series})

    with open(OUT / "chart1_event_window.json", "w") as f:
        json.dump({"window_minutes": WINDOW_MIN,
                   "stat_window_minutes": STAT_WINDOW,
                   "bar_minutes": BAR_MIN,
                   "asset": "USO",
                   "y_axis_units": "cumulative percent change vs t=0 close",
                   "n_bursts": len(nested),
                   "bursts": nested}, f, indent=2)

    # Median / quantile envelope (for the "all events overlaid" view robustness)
    pivot = df.pivot_table(index="minute_offset", columns="burst_id",
                           values="cum_pct_return", aggfunc="first")
    envelope = pd.DataFrame({
        "minute_offset": pivot.index,
        "median":        pivot.median(axis=1).round(4).values,
        "p25":           pivot.quantile(0.25, axis=1).round(4).values,
        "p75":           pivot.quantile(0.75, axis=1).round(4).values,
        "p10":           pivot.quantile(0.10, axis=1).round(4).values,
        "p90":           pivot.quantile(0.90, axis=1).round(4).values,
        "n_bursts":      pivot.notna().sum(axis=1).values,
    })
    envelope.to_csv(OUT / "chart1_envelope.csv", index=False)

    print(f"chart1: {len(rows)} rows across {len(nested)} bursts")


# ---------------------------------------------------------------------------
# Chart 1b - Per-burst posts (full text for tooltips / annotations)
# ---------------------------------------------------------------------------
def build_posts_by_burst(bursts: pd.DataFrame, posts: pd.DataFrame,
                           events: pd.DataFrame) -> None:
    """One row per post that falls inside a burst's [ts_first, ts_last] window.

    The source `pnl_concentration_bursts.csv` is built from the 81 *triggered*
    events; some posts within a burst's time window were not individually
    triggered events (e.g. they failed the initiated-event filter). We keep
    those posts in the file because they are real Trump posts that happened
    inside the burst, but flag them with `is_triggered_event=False` so the
    metadata `n_posts` counts (which are triggered-only and sum to 81) can be
    reconciled.
    """
    triggered_ids = set(events["post_id"].astype(str))

    rows = []
    for _, b in bursts.iterrows():
        bid = int(b["burst_id"])
        in_burst = posts[
            (posts["created_at"] >= b["ts_first"]) &
            (posts["created_at"] <= b["ts_last"])
        ].sort_values("created_at")

        for _, p in in_burst.iterrows():
            pid = str(p["id"])
            rows.append({
                "burst_id":             bid,
                "post_id":              pid,
                "is_triggered_event":   pid in triggered_ids,
                "ts_utc":               p["created_at"].isoformat(),
                "ts_et":                p["created_at"].tz_convert("America/New_York").isoformat(),
                "minute_offset_in_burst": round(
                    (p["created_at"] - b["ts_first"]).total_seconds() / 60, 2),
                "text":                 p["text"],
                "is_reblog":            bool(p["is_reblog"]),
                "favourites_count":     int(p["favourites_count"]),
                "reblogs_count":        int(p["reblogs_count"]),
                "replies_count":        int(p["replies_count"]),
                # Topic flags
                "topic_energy_oil":     bool(p["topic_energy_oil"]),
                "topic_iran_military":  bool(p["topic_iran_military"]),
                "topic_market_economy": bool(p["topic_market_economy"]),
                "topic_tariff_trade":   bool(p["topic_tariff_trade"]),
                "topic_china":          bool(p["topic_china"]),
                "topic_fed_rates":      bool(p["topic_fed_rates"]),
                "topic_djt_media":      bool(p["topic_djt_media"]),
                "topic_crypto":         bool(p["topic_crypto"]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "chart1_posts_by_burst.csv", index=False)

    # Nested JSON keyed by burst_id (ergonomic for D3 tooltips)
    by_burst = {}
    for r in rows:
        by_burst.setdefault(int(r["burst_id"]), []).append(r)
    with open(OUT / "chart1_posts_by_burst.json", "w") as f:
        json.dump({"posts_by_burst": {str(k): v for k, v in by_burst.items()}},
                  f, indent=2)

    print(f"posts_by_burst: {len(df)} posts across {df['burst_id'].nunique()} bursts "
          f"(max {df.groupby('burst_id').size().max()} posts/burst)")


# ---------------------------------------------------------------------------
# Chart 1c - Stage 1 helpers: placebo null + clean post-event exemplar
# ---------------------------------------------------------------------------
def build_chart1_stage1(uso: pd.DataFrame, bursts: pd.DataFrame,
                         events: pd.DataFrame, posts: pd.DataFrame,
                         n_placebo: int = 500, seed: int = 42) -> None:
    """Two deliverables for Matt's stage 1 ('a normal signal: market moves
    after post'):

    1. PLACEBO ENVELOPE - sample N random USO 5-min bars (excluding any bar
       within +-30 min of a known post timestamp), compute cum % return over
       +-60 min around each, return median + 25/75 + 10/90 percentiles. This
       is the null distribution: 'what does an average non-event window look
       like'. Should be flat with noise.

    2. EXEMPLAR EVENTS - pick a small handful of individual triggered events
       where the market reacted *after* the post (clean post-event rise) but
       not *before* (no pre-event drift). These are the 'efficient market
       textbook' examples Matt can use as the single illustrative line in
       stage 1.
    """
    rng = np.random.default_rng(seed)

    # ------ 1. Placebo envelope ------
    # Build exclusion zones: +-30 min around every post timestamp
    post_times = pd.DatetimeIndex(posts["created_at"].sort_values())
    if post_times.tz is None:
        post_times = post_times.tz_localize("UTC")

    # Candidate anchor bars: bars where a +-60 min window fully fits in the data
    bar_index   = uso.index
    earliest    = bar_index[0]  + pd.Timedelta(minutes=WINDOW_MIN)
    latest      = bar_index[-1] - pd.Timedelta(minutes=WINDOW_MIN)
    candidates  = bar_index[(bar_index >= earliest) & (bar_index <= latest)]

    # Drop candidates within +-30 min of any post
    keep = np.ones(len(candidates), dtype=bool)
    for pt in post_times:
        delta = (candidates - pt).total_seconds() / 60
        keep &= (np.abs(delta) > 30)

    eligible = candidates[keep]
    # Sample without replacement
    n_sample = min(n_placebo, len(eligible))
    sampled  = rng.choice(eligible, size=n_sample, replace=False)

    # For each sampled anchor, compute cum_pct_return over +-60 min
    placebo_curves = []  # each entry: dict {minute_offset -> cum_pct}
    for anchor in sampled:
        anchor = pd.Timestamp(anchor)
        lo = anchor - pd.Timedelta(minutes=WINDOW_MIN)
        hi = anchor + pd.Timedelta(minutes=WINDOW_MIN)
        win = uso.loc[(uso.index >= lo) & (uso.index <= hi)]
        if anchor not in win.index or len(win) < 13:
            continue
        anchor_price = float(win.loc[anchor, "Close"])
        for ts, row in win.iterrows():
            offset = int((ts - anchor).total_seconds() // 60)
            cum    = (float(row["Close"]) / anchor_price - 1) * 100
            placebo_curves.append({"minute_offset": offset, "cum_pct_return": cum})

    pdf = pd.DataFrame(placebo_curves)
    env = (pdf.groupby("minute_offset")["cum_pct_return"]
              .agg(median="median",
                   p25=lambda s: s.quantile(0.25),
                   p75=lambda s: s.quantile(0.75),
                   p10=lambda s: s.quantile(0.10),
                   p90=lambda s: s.quantile(0.90),
                   n_windows="count")
              .round(4)
              .reset_index())
    env.to_csv(OUT / "chart1_placebo_envelope.csv", index=False)
    print(f"placebo: {len(sampled)} sampled non-event windows; envelope spans "
          f"{env['minute_offset'].min()} to {env['minute_offset'].max()} min")

    # ------ 2. Exemplar events ------
    # Scan ALL posts (not just the 81 triggered ones) for clean
    # 'market reacts after post' cases. Criteria:
    #   - Anchor in trading hours with full +-60 min coverage
    #   - Post NOT part of any of the 15 bursts (we want a 'normal' post)
    #   - Small pre-event movement (low abs cum return at every pre offset)
    #   - Monotonic-ish post-event reaction (clean directional move, not noise)
    #   - Final +60 min cum return materially non-zero (signal of real reaction)

    burst_post_ids = set()  # post_ids that are inside any burst window
    for _, b in bursts.iterrows():
        in_b = posts[
            (posts["created_at"] >= b["ts_first"] - pd.Timedelta(seconds=1)) &
            (posts["created_at"] <= b["ts_last"]  + pd.Timedelta(seconds=1))
        ]
        burst_post_ids.update(in_b["id"].astype(str))

    rows = []
    for _, p in posts.iterrows():
        pid = str(p["id"])
        if pid in burst_post_ids:
            continue
        ts = p["created_at"]
        anchor = ts.floor(f"{BAR_MIN}min")
        if anchor not in uso.index:
            continue

        lo = anchor - pd.Timedelta(minutes=WINDOW_MIN)
        hi = anchor + pd.Timedelta(minutes=WINDOW_MIN)
        win = uso.loc[(uso.index >= lo) & (uso.index <= hi)]
        # Need a fully-populated window (25 bars across +-60 min)
        if anchor not in win.index or len(win) < 25:
            continue

        anchor_price = float(win.loc[anchor, "Close"])
        cum = (win["Close"] / anchor_price - 1) * 100

        pre  = cum.loc[cum.index <  anchor]
        post = cum.loc[cum.index >  anchor]
        if pre.empty or post.empty:
            continue

        pre_max_abs = float(pre.abs().max())
        # Tail cum return at +60 min as a measure of the reaction
        post_final  = float(post.iloc[-1])
        post_max    = float(post.abs().max())
        # Monotonicity score: how much of the post-event movement is in one direction?
        # Sign-consistency: fraction of post bars whose cum return has the same sign as the final.
        sign_consistent = float((np.sign(post.values) == np.sign(post_final)).mean())
        # Range vs final magnitude: low = clean, high = noisy
        post_volatility = float(post.std())

        # Classify session for the +-60 min window (matters because pre-market
        # liquidity is thin and behaviour differs from RTH)
        et = ts.tz_convert("America/New_York")
        et_minutes = et.hour * 60 + et.minute
        if 9 * 60 + 30 <= et_minutes < 16 * 60:
            session = "RTH"
        elif 4 * 60 <= et_minutes < 9 * 60 + 30:
            session = "pre-market"
        elif 16 * 60 <= et_minutes < 20 * 60:
            session = "after-hours"
        else:
            session = "overnight/closed"

        rows.append({
            "post_id":           pid,
            "ts":                ts.isoformat(),
            "session":           session,
            "anchor_bar_ts":     anchor.isoformat(),
            "anchor_price":      round(anchor_price, 4),
            "pre_max_abs_pct":   round(pre_max_abs,    4),
            "post_final_pct":    round(post_final,     4),
            "post_max_abs_pct":  round(post_max,       4),
            "post_sign_consistency": round(sign_consistent, 3),
            "post_volatility":   round(post_volatility, 4),
            "topic_oil":         bool(p["topic_energy_oil"]),
            "topic_iran":        bool(p["topic_iran_military"]),
            "is_reblog":         bool(p["is_reblog"]),
        })
    rdf = pd.DataFrame(rows)

    if not rdf.empty:
        # Hard filters
        cand = rdf[
            (rdf["pre_max_abs_pct"]      < 0.30)   # truly quiet pre-event
            & (rdf["post_max_abs_pct"]   > 0.20)   # meaningful reaction
            & (rdf["post_sign_consistency"] > 0.65)  # monotonic-ish
            & (~rdf["is_reblog"])
        ].copy()

        # Score: prefer large clean post-event move with little pre-event drift
        cand["score"] = (
            cand["post_max_abs_pct"]
            - 3 * cand["pre_max_abs_pct"]
            + 0.5 * cand["post_sign_consistency"]
            - 0.3 * cand["post_volatility"]
        )
        cand = cand.sort_values("score", ascending=False).head(5)

        if cand.empty:
            print("stage 1 exemplars: no posts met the 'clean post-only reaction' criteria - "
                  "use placebo envelope instead")
        else:
            # Emit full +-60 min series for each
            exemplar_rows = []
            for _, r in cand.iterrows():
                anchor = pd.Timestamp(r["anchor_bar_ts"])
                lo = anchor - pd.Timedelta(minutes=WINDOW_MIN)
                hi = anchor + pd.Timedelta(minutes=WINDOW_MIN)
                win = uso.loc[(uso.index >= lo) & (uso.index <= hi)].copy()
                anchor_price = float(win.loc[anchor, "Close"])
                for ts, row in win.iterrows():
                    exemplar_rows.append({
                        "post_id":         r["post_id"],
                        "anchor_bar_ts":   r["anchor_bar_ts"],
                        "minute_offset":   int((ts - anchor).total_seconds() // 60),
                        "cum_pct_return":  round((float(row["Close"]) / anchor_price - 1) * 100, 4),
                        "anchor_price":    round(anchor_price, 4),
                        "pre_max_abs_pct": r["pre_max_abs_pct"],
                        "post_final_pct":  r["post_final_pct"],
                    })
            pd.DataFrame(exemplar_rows).to_csv(OUT / "chart1_stage1_exemplars.csv", index=False)

            # Metadata with text
            cand["post_id"] = cand["post_id"].astype(str)
            posts_str = posts.copy()
            posts_str["id"] = posts_str["id"].astype(str)
            cand = cand.merge(
                posts_str[["id", "text"]].rename(columns={"id": "post_id"}),
                on="post_id", how="left"
            )
            cand["text"] = cand["text"].apply(lambda t: snippet(t, 200))
            cand[[
                "post_id", "ts", "session", "anchor_price",
                "pre_max_abs_pct", "post_final_pct", "post_max_abs_pct",
                "post_sign_consistency", "topic_oil", "topic_iran",
                "text",
            ]].to_csv(OUT / "chart1_stage1_exemplar_metadata.csv", index=False)
            print(f"stage 1 exemplars: {len(cand)} clean single-event illustrations "
                  f"(top: post_id {cand.iloc[0]['post_id']}, "
                  f"final +{cand.iloc[0]['post_final_pct']:.2f}%)")


# ---------------------------------------------------------------------------
# Chart 2 - Topic co-occurrence (bubble / network)
# ---------------------------------------------------------------------------
def build_chart2(posts: pd.DataFrame) -> None:
    p = posts.copy()
    # Only count posts that have at least one topic tag
    p["n_tags"] = p[TOPIC_COLS].sum(axis=1)
    tagged = p[p["n_tags"] >= 1]

    # Node sizes = post count per topic
    counts = {c: int(tagged[c].sum()) for c in TOPIC_COLS}

    # Edge weights = co-occurrence within the same post
    edges = Counter()
    for _, row in tagged.iterrows():
        active = [c for c in TOPIC_COLS if bool(row[c])]
        for a, b in combinations(active, 2):
            key = tuple(sorted([a, b]))
            edges[key] += 1

    # Self-loops (topic post counts) for sizing
    nodes = [
        {"id": c,
         "label": TOPIC_LABELS[c],
         "post_count": counts[c]}
        for c in TOPIC_COLS
    ]
    nodes.sort(key=lambda n: -n["post_count"])

    links = [
        {"source": a, "target": b, "weight": int(w)}
        for (a, b), w in sorted(edges.items(), key=lambda x: -x[1])
    ]

    payload = {
        "n_posts_total":           int(len(p)),
        "n_posts_with_any_topic":  int(len(tagged)),
        "n_posts_multi_topic":     int((tagged["n_tags"] > 1).sum()),
        "nodes": nodes,
        "links": links,
    }
    with open(OUT / "chart2_topic_cooccurrence.json", "w") as f:
        json.dump(payload, f, indent=2)

    pd.DataFrame(nodes).to_csv(OUT / "chart2_topic_nodes.csv", index=False)
    pd.DataFrame(links).to_csv(OUT / "chart2_topic_edges.csv", index=False)
    print(f"chart2: {len(nodes)} nodes, {len(links)} edges, "
          f"{payload['n_posts_with_any_topic']} tagged posts ({payload['n_posts_multi_topic']} multi-topic)")


# ---------------------------------------------------------------------------
# Chart 3 - Heatmap: topic x asset, pre-event mean signal
# ---------------------------------------------------------------------------
def build_chart3() -> None:
    """Heatmap from orderflow_event_study_5m.json.

    JSON shape:
      results -> topic -> asset -> {"all", "initiated"} -> {n, offsets, signals}
        where signals[<sig>] has pre_mean, pre_t, pre_p, pre_boot_p, post_mean, ...
    """
    src = json.load(open(REPO / "data/results/orderflow_event_study_5m.json"))
    cfg = src["config"]
    res = src["results"]

    rows = []
    for topic, by_asset in res.items():
        for asset, subsets in by_asset.items():
            for subset_name, payload in subsets.items():     # "all" | "initiated"
                n_events = int(payload.get("n", 0))
                signals  = payload.get("signals", {})
                for sig_name, stats in signals.items():
                    rows.append({
                        "topic":           topic,
                        "topic_label":     TOPIC_LABELS.get(f"topic_{topic}", topic),
                        "asset":           asset,
                        "signal":          sig_name,
                        "subset":          subset_name,
                        "n_events":        n_events,
                        "pre_mean":        round(stats.get("pre_mean",  float("nan")), 6)
                                              if stats.get("pre_mean")  is not None else None,
                        "pre_t":           round(stats.get("pre_t",     float("nan")), 4)
                                              if stats.get("pre_t")     is not None else None,
                        "pre_p":           stats.get("pre_p"),
                        "pre_boot_p":      stats.get("pre_boot_p"),
                        "post_mean":       round(stats.get("post_mean", float("nan")), 6)
                                              if stats.get("post_mean") is not None else None,
                        "post_t":          round(stats.get("post_t",    float("nan")), 4)
                                              if stats.get("post_t")    is not None else None,
                        "post_p":          stats.get("post_p"),
                        "post_boot_p":     stats.get("post_boot_p"),
                    })

    df = pd.DataFrame(rows)

    # Long CSV (every signal kept; D3 can filter)
    df.to_csv(OUT / "chart3_heatmap_long.csv", index=False)

    # Recommended headline view: OFI_bvc on the "initiated" subset, pre-event
    # (matches the universe the report uses for the asset-specificity claim)
    headline = df[(df["signal"] == "OFI_bvc") & (df["subset"] == "initiated")].copy()
    headline.to_csv(OUT / "chart3_heatmap_OFI_pre_initiated.csv", index=False)

    # Wide pivot (topic rows, asset cols, cell = pre_mean)
    if not headline.empty:
        wide = headline.pivot(index="topic_label", columns="asset", values="pre_mean")
        wide.to_csv(OUT / "chart3_heatmap_OFI_pre_initiated_wide.csv")

    # Also ship the "all events" version for comparison
    all_view = df[(df["signal"] == "OFI_bvc") & (df["subset"] == "all")].copy()
    if not all_view.empty:
        all_view.to_csv(OUT / "chart3_heatmap_OFI_pre_all.csv", index=False)
        wide_all = all_view.pivot(index="topic_label", columns="asset", values="pre_mean")
        wide_all.to_csv(OUT / "chart3_heatmap_OFI_pre_all_wide.csv")

    # Also surface vpin_z (toxicity) and dvol_z - other signals discussed in the report
    for sig in ["vpin_z", "dvol_z"]:
        sub = df[(df["signal"] == sig) & (df["subset"] == "initiated")]
        if not sub.empty:
            sub.to_csv(OUT / f"chart3_heatmap_{sig}_pre_initiated.csv", index=False)

    # JSON
    with open(OUT / "chart3_heatmap.json", "w") as f:
        json.dump({
            "config":          cfg,
            "primary_signal": "OFI_bvc",
            "primary_subset": "initiated",
            "primary_window": "pre",
            "long":            rows,
        }, f, indent=2, default=str)

    print(f"chart3: {len(df)} cells "
          f"({df['topic'].nunique()} topics x {df['asset'].nunique()} assets x "
          f"{df['signal'].nunique()} signals x {df['subset'].nunique()} subsets); "
          f"headline view = {len(headline)} cells")

    # --- Supplementary: oil-complex extension (XOM, CVX from finding4) -------
    f4 = json.load(open(REPO / "data/results/finding4_oil_complex.json"))
    oil_rows = []
    for s in f4.get("oil_complex_survivors_fdr05", []):
        oil_rows.append({
            "topic":           "energy_oil",
            "topic_label":     TOPIC_LABELS["topic_energy_oil"],
            "asset":           s.get("asset"),
            "signal":          s.get("signal"),
            "subset":          "initiated",
            "n_events":        s.get("n"),
            "pre_mean":        s.get("mean"),
            "placebo_mean":    s.get("placebo_mean"),
            "delta_vs_placebo": s.get("delta_vs_placebo"),
            "p_fdr_trim":      s.get("p_fdr_trim"),
            "fdr_passed":      True,
        })
    if oil_rows:
        pd.DataFrame(oil_rows).to_csv(OUT / "chart3_oil_complex_extension.csv", index=False)
        print(f"chart3 oil_complex extension: {len(oil_rows)} survivor cells (XOM/CVX/etc.)")


# ---------------------------------------------------------------------------
# Chart 4 - Timeline: USO close + post markers + daily post counts
# ---------------------------------------------------------------------------
def build_chart4(uso: pd.DataFrame, xle: pd.DataFrame,
                  bursts: pd.DataFrame,
                  posts: pd.DataFrame, events: pd.DataFrame) -> None:

    def _daily_rth(df: pd.DataFrame, label: str) -> pd.DataFrame:
        d = df[df["is_rth"]].copy()
        d.index = d.index.tz_convert("America/New_York")
        out = (d["Close"].resample("1D").last().dropna()
                .reset_index()
                .rename(columns={"Datetime": "date_et", "Close": label}))
        out["date_et"] = out["date_et"].dt.strftime("%Y-%m-%d")
        return out

    def _hourly_rth(df: pd.DataFrame, label: str) -> pd.DataFrame:
        h = df[df["is_rth"]]["Close"].resample("1h").last().dropna().reset_index()
        h = h.rename(columns={"Datetime": "ts_utc", "Close": label})
        h["ts_utc"] = h["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return h

    # ----- USO -----
    daily_uso = _daily_rth(uso, "uso_close")
    daily_uso.to_csv(OUT / "chart4_uso_daily_close.csv", index=False)
    _hourly_rth(uso, "uso_close").to_csv(OUT / "chart4_uso_hourly_close.csv", index=False)

    # ----- XLE (oil sector ETF, secondary line) -----
    daily_xle = _daily_rth(xle, "xle_close")
    daily_xle.to_csv(OUT / "chart4_xle_daily_close.csv", index=False)
    _hourly_rth(xle, "xle_close").to_csv(OUT / "chart4_xle_hourly_close.csv", index=False)

    # Combined daily series for convenience: USO + XLE on the same dates
    daily_combined = daily_uso.merge(daily_xle, on="date_et", how="outer").sort_values("date_et")
    daily_combined.to_csv(OUT / "chart4_oil_complex_daily_close.csv", index=False)

    # Keep the historical names for backwards compat
    daily = daily_uso

    # Post markers - one per burst (15)
    burst_markers = []
    for _, b in bursts.iterrows():
        burst_posts = posts[
            (posts["created_at"] >= b["ts_first"] - pd.Timedelta(seconds=1)) &
            (posts["created_at"] <= b["ts_last"]  + pd.Timedelta(seconds=1))
        ]
        sample_text = snippet(burst_posts["text"].iloc[0]) if len(burst_posts) else ""
        burst_markers.append({
            "burst_id":      int(b["burst_id"]),
            "ts_first_utc":  b["ts_first"].isoformat(),
            "ts_first_et":   b["ts_first"].tz_convert("America/New_York").isoformat(),
            "date_et":       b["ts_first"].tz_convert("America/New_York").strftime("%Y-%m-%d"),
            "duration_min":  round((b["ts_last"] - b["ts_first"]).total_seconds() / 60, 2),
            "n_posts":       int(b["n_posts"]),
            "pnl_total":     round(float(b["pnl_total"]), 2),
            "sample_text":   sample_text,
        })
    pd.DataFrame(burst_markers).to_csv(OUT / "chart4_burst_markers.csv", index=False)

    # Post markers - one per individual triggered event (81), in case Matt wants finer-grained dots
    event_markers = events.copy()
    event_markers["date_et"] = (event_markers["ts"]
                                  .dt.tz_convert("America/New_York")
                                  .dt.strftime("%Y-%m-%d"))
    event_markers["ts_utc"] = event_markers["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    cols = ["post_id", "ts_utc", "date_et", "session", "pnl",
            "pre_sv", "vpinz_pre", "entry_price", "exit_price"]
    event_markers[cols].to_csv(OUT / "chart4_event_markers.csv", index=False)

    # Daily post counts (whole 60-day window)
    posts_local = posts.copy()
    posts_local["date_et"] = (posts_local["created_at"]
                                .dt.tz_convert("America/New_York")
                                .dt.strftime("%Y-%m-%d"))
    daily_posts_total = (posts_local
                          .groupby("date_et").size()
                          .rename("n_posts")
                          .reset_index())

    # Plus daily counts of oil-themed posts (the topic that actually matters)
    daily_posts_oil = (posts_local[posts_local["topic_energy_oil"]]
                          .groupby("date_et").size()
                          .rename("n_oil_posts")
                          .reset_index())

    daily_posts = daily_posts_total.merge(daily_posts_oil, on="date_et", how="left")
    daily_posts["n_oil_posts"] = daily_posts["n_oil_posts"].fillna(0).astype(int)
    daily_posts.to_csv(OUT / "chart4_daily_post_counts.csv", index=False)

    # Combined JSON for D3 convenience
    payload = {
        "uso_daily_close":   daily_uso.to_dict(orient="records"),
        "xle_daily_close":   daily_xle.to_dict(orient="records"),
        "burst_markers":     burst_markers,
        "daily_post_counts": daily_posts.to_dict(orient="records"),
    }
    with open(OUT / "chart4_timeline.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"chart4: {len(daily_uso)} daily USO closes, "
          f"{len(daily_xle)} daily XLE closes, "
          f"{len(burst_markers)} burst markers, "
          f"{len(event_markers)} event markers, "
          f"{len(daily_posts)} daily post-count rows")


# ---------------------------------------------------------------------------
def main() -> None:
    uso    = load_uso_bars()
    xle    = load_xle_bars()
    posts  = load_posts()
    bursts = load_bursts()
    events = load_events()

    print(f"USO bars: {uso.shape}, XLE bars: {xle.shape}, "
          f"posts: {posts.shape}, bursts: {bursts.shape}, events: {events.shape}")

    build_chart1(uso, bursts, posts)
    build_posts_by_burst(bursts, posts, events)
    build_chart1_stage1(uso, bursts, events, posts)
    build_chart2(posts)
    build_chart3()
    build_chart4(uso, xle, bursts, posts, events)

    print(f"\nAll outputs in: {OUT}")
    for f in sorted(OUT.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:48s} {size:>10,} bytes")


if __name__ == "__main__":
    main()
