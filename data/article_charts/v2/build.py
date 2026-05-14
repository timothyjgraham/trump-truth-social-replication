#!/usr/bin/env python3
"""
build.py - reproducible chart-data build for Matt's Trump/Truth Social/oil article.

Produces every CSV / JSON in this folder. Run this then render_previews.py then
verify.py for the full pipeline (or use ./run.sh).

Path resolution:
  - The replication repo is expected to live at ../trump-truth-social-replication/
    relative to this script's directory. This is the same parent folder as this
    release bundle.
  - Override with --repo /path/to/trump-truth-social-replication if needed.

Usage:
  python build.py                                # default paths
  python build.py --repo /custom/path/to/repo    # override repo location

Source data this script reads:
  - data/raw/posts_60d.parquet                   - 1,341 Truth Social posts
  - data/raw/minute_bars_5m/USO.parquet          - 5-min OHLCV bars
  - data/results/pnl_concentration_bursts.csv    - 15 collapsed bursts
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO = (SCRIPT_DIR.parent / "trump-truth-social-replication").resolve()

BAR_MIN     = 5            # 5-min bar resolution
WINDOW_MIN  = 180          # +/- 180 minutes around t=0
STAT_WINDOW = 30           # the analytical stat window
SHIFT_HOURS = 24           # falsification time-shift

OIL_TERMS_RE = re.compile(
    r"\b(oil|crude|opec|barrel|refinery|gasoline|petroleum|hormuz|saudi|iran|"
    r"venezuela|drill|drilling|frack|pipeline|wti|brent)\b", re.I)

STOPWORDS = {
    "the","a","an","and","or","but","if","of","on","in","to","for","with","at",
    "by","from","up","down","is","are","was","were","be","been","being","have",
    "has","had","do","does","did","will","would","should","could","may","might",
    "must","can","this","that","these","those","i","you","he","she","it","we",
    "they","me","him","her","us","them","my","your","his","its","our","their",
    "as","not","no","so","than","too","very","just","also","then","there",
    "here","what","who","whom","which","why","how","because","over","under",
    "between","through","into","out","onto","upon","over","than","such",
    "amp","like","via","off","still","always","never","ever","quot","apos",
    "very","much","more","most","many","some","any","all","both","each","few",
    "other","another","new","old","one","two","three","first","last","next",
    "https","http","com","www","html","co","amp",
    "great","good","big","huge","said","says","say","get","got","make","made",
    "going","go","come","came","see","know","want","need","got","let","take",
    "thank","thanks","tonight","today","yesterday","tomorrow","just","really",
    "people","time","year","years","day","days","way","things","thing",
    "everybody","everyone","nobody","anyone","someone","somebody","everything",
    "nothing","anything","something","please","look","looks","looking",
}
BOILERPLATE = {"matter", "attention", "before"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_uso(repo: Path) -> pd.DataFrame:
    df = pd.read_parquet(repo / "data/raw/minute_bars_5m/USO.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_posts(repo: Path) -> pd.DataFrame:
    p = pd.read_parquet(repo / "data/raw/posts_60d.parquet")
    p["created_at"] = pd.to_datetime(p["created_at"], utc=True)
    return p


def load_bursts(repo: Path) -> pd.DataFrame:
    b = pd.read_csv(repo / "data/results/pnl_concentration_bursts.csv")
    b["ts_first"] = pd.to_datetime(b["ts_first"], utc=True)
    b["ts_last"]  = pd.to_datetime(b["ts_last"],  utc=True)
    b = b.sort_values("ts_first").reset_index(drop=True)
    b["burst_id"] = b.index + 1
    return b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def snippet(text: str, n: int = 160) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def find_anchor(uso: pd.DataFrame, ts: pd.Timestamp) -> tuple[pd.Timestamp | None, bool]:
    """Floor to nearest 5-min bar; advance to next available bar within 72h
    if floored timestamp isn't in the index. Flag advanced anchors."""
    floor = ts.floor(f"{BAR_MIN}min")
    if floor in uso.index:
        return floor, False
    future = uso.index[uso.index >= floor]
    if len(future) == 0 or (future[0] - floor).total_seconds() / 3600 > 72:
        return None, False
    return future[0], True


def get_window(uso: pd.DataFrame, anchor: pd.Timestamp,
               window_min: int = WINDOW_MIN) -> pd.DataFrame:
    lo = anchor - pd.Timedelta(minutes=window_min)
    hi = anchor + pd.Timedelta(minutes=window_min)
    win = uso.loc[(uso.index >= lo) & (uso.index <= hi)].copy()
    win["minute_offset"] = ((win.index - anchor).total_seconds() // 60).astype(int)
    return win


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"&[a-z]+;",      " ", text)
    text = re.sub(r"<[^>]+>",       " ", text)
    tokens = re.findall(r"[A-Za-z']{3,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 3]


# ---------------------------------------------------------------------------
# Chart 1 v2 — pre/post overlay (positive-only, ±180 min)
# ---------------------------------------------------------------------------
def build_chart1_v2(uso: pd.DataFrame, bursts: pd.DataFrame, posts: pd.DataFrame,
                     out: Path) -> None:
    rows_5min, rows_1min, burst_meta = [], [], []

    for _, b in bursts.iterrows():
        bid = int(b["burst_id"])
        anchor, anchored_next = find_anchor(uso, b["ts_first"])
        if anchor is None:
            print(f"  burst {bid}: no anchor available, skipping")
            continue

        win = get_window(uso, anchor, WINDOW_MIN)
        anchor_price = float(win.loc[anchor, "Close"])
        win["cum_pct_signed"] = (win["Close"] / anchor_price - 1) * 100.0
        win["cum_pct_abs"]    = win["cum_pct_signed"].abs()
        win["in_stat_window"] = win["minute_offset"].abs() <= STAT_WINDOW
        win["is_zero_vol"]    = (win["Volume"] == 0)
        win["is_rth_bar"]     = win["is_rth"].astype(bool)

        burst_posts = posts[
            (posts["created_at"] >= b["ts_first"] - pd.Timedelta(seconds=1)) &
            (posts["created_at"] <= b["ts_last"]  + pd.Timedelta(seconds=1))
        ]
        sample_text = snippet(burst_posts["text"].iloc[0]) if len(burst_posts) else ""

        for _, r in win.iterrows():
            rows_5min.append({
                "burst_id":         bid,
                "burst_ts_anchor":  anchor.isoformat(),
                "minute_offset":    int(r["minute_offset"]),
                "cum_pct_abs":      round(float(r["cum_pct_abs"]),    4),
                "cum_pct_signed":   round(float(r["cum_pct_signed"]), 4),
                "price_close":      round(float(r["Close"]),          4),
                "anchor_price":     round(anchor_price,               4),
                "volume":           int(r["Volume"]),
                "is_zero_vol":      bool(r["is_zero_vol"]),
                "is_rth_bar":       bool(r["is_rth_bar"]),
                "in_stat_window":   bool(r["in_stat_window"]),
            })

        win_indexed = win[["minute_offset", "cum_pct_abs", "cum_pct_signed", "Close"]] \
            .set_index("minute_offset").sort_index()
        full_offsets = range(int(win["minute_offset"].min()),
                             int(win["minute_offset"].max()) + 1)
        win_interp = win_indexed.reindex(full_offsets).interpolate(method="linear")

        for off, r in win_interp.iterrows():
            rows_1min.append({
                "burst_id":        bid,
                "burst_ts_anchor": anchor.isoformat(),
                "minute_offset":   int(off),
                "cum_pct_abs":     round(float(r["cum_pct_abs"]),    4) if pd.notna(r["cum_pct_abs"]) else None,
                "cum_pct_signed":  round(float(r["cum_pct_signed"]), 4) if pd.notna(r["cum_pct_signed"]) else None,
                "is_real_bar":     int(off) % BAR_MIN == 0,
                "in_stat_window":  abs(int(off)) <= STAT_WINDOW,
            })

        burst_meta.append({
            "burst_id":                  bid,
            "ts_first":                  b["ts_first"].isoformat(),
            "ts_last":                   b["ts_last"].isoformat(),
            "duration_minutes":          round((b["ts_last"] - b["ts_first"]).total_seconds() / 60, 2),
            "n_posts":                   int(b["n_posts"]),
            "anchor_bar_ts":             anchor.isoformat(),
            "anchored_to_next_session":  anchored_next,
            "anchor_price":              round(anchor_price, 4),
            "pnl_total":                 round(float(b["pnl_total"]),         2),
            "pnl_per_post_mean":         round(float(b["pnl_per_post_mean"]), 2),
            "initiated_any":             bool(b["initiated_any"]),
            "sample_post_text":          sample_text,
            "min_offset_with_data":      int(win["minute_offset"].min()),
            "max_offset_with_data":      int(win["minute_offset"].max()),
            "max_abs_pct_pre":           round(float(win.loc[win["minute_offset"] < 0, "cum_pct_abs"].max() if (win["minute_offset"] < 0).any() else 0), 4),
            "max_abs_pct_post":          round(float(win.loc[win["minute_offset"] > 0, "cum_pct_abs"].max() if (win["minute_offset"] > 0).any() else 0), 4),
        })

    df_5min = pd.DataFrame(rows_5min)
    df_1min = pd.DataFrame(rows_1min)
    meta    = pd.DataFrame(burst_meta)

    df_5min.to_csv(out / "chart1v2_event_window_5min.csv", index=False)
    df_1min.to_csv(out / "chart1v2_event_window_1min.csv", index=False)
    meta.to_csv(out / "chart1v2_burst_metadata.csv", index=False)

    pivot_abs = df_5min.pivot_table(index="minute_offset", columns="burst_id",
                                     values="cum_pct_abs", aggfunc="first")
    envelope = pd.DataFrame({
        "minute_offset": pivot_abs.index,
        "median":        pivot_abs.median(axis=1).round(4).values,
        "p25":           pivot_abs.quantile(0.25, axis=1).round(4).values,
        "p75":           pivot_abs.quantile(0.75, axis=1).round(4).values,
        "p10":           pivot_abs.quantile(0.10, axis=1).round(4).values,
        "p90":           pivot_abs.quantile(0.90, axis=1).round(4).values,
        "n_bursts":      pivot_abs.notna().sum(axis=1).values,
    })
    envelope.to_csv(out / "chart1v2_envelope_abs.csv", index=False)

    nested = []
    for m in burst_meta:
        series_5 = sorted(
            [{"minute_offset": r["minute_offset"],
              "cum_pct_abs":    r["cum_pct_abs"],
              "cum_pct_signed": r["cum_pct_signed"],
              "in_stat_window": r["in_stat_window"]}
             for r in rows_5min if r["burst_id"] == m["burst_id"]],
            key=lambda x: x["minute_offset"])
        nested.append({**m, "series_5min": series_5})

    with open(out / "chart1v2_event_window.json", "w") as f:
        json.dump({
            "window_minutes":     WINDOW_MIN,
            "stat_window_minutes": STAT_WINDOW,
            "bar_minutes":         BAR_MIN,
            "asset":               "USO",
            "y_axis_primary":      "cum_pct_abs (positive-only |% deviation from t=0|)",
            "y_axis_supplementary": "cum_pct_signed (original signed cum % return)",
            "n_bursts":            len(nested),
            "bursts":              nested,
        }, f, indent=2)

    print(f"  chart1v2 5-min:   {len(df_5min)} rows across {df_5min['burst_id'].nunique()} bursts")
    print(f"  chart1v2 1-min:   {len(df_1min)} rows (linearly interpolated between 5-min bars)")
    print(f"  chart1v2 envelope: {len(envelope)} offset buckets")


# ---------------------------------------------------------------------------
# Time-shift falsification overlay (anchor + 24h, same time-of-day)
# ---------------------------------------------------------------------------
def build_timeshift_overlay(uso: pd.DataFrame, bursts: pd.DataFrame, out: Path) -> None:
    real_rows, shifted_rows, burst_rows = [], [], []
    skipped = []

    for _, b in bursts.iterrows():
        bid = int(b["burst_id"])
        anchor_real, anc_real_next = find_anchor(uso, b["ts_first"])
        if anchor_real is None:
            skipped.append((bid, "no real anchor"))
            continue

        # Shift the *anchor* by 24h, not the post timestamp. This preserves
        # same-time-of-day matching, which is the falsification logic from
        # §3.4 of the report.
        ts_shifted_anchor = anchor_real + pd.Timedelta(hours=SHIFT_HOURS)
        anchor_shifted, anc_shifted_next = find_anchor(uso, ts_shifted_anchor)
        if anchor_shifted is None:
            skipped.append((bid, "no shifted anchor available"))
            continue

        et_real    = anchor_real.tz_convert("America/New_York").strftime("%H:%M")
        et_shifted = anchor_shifted.tz_convert("America/New_York").strftime("%H:%M")
        if et_real != et_shifted:
            skipped.append((bid, f"session mismatch real={et_real} shifted={et_shifted}"))
            continue
        delta_h = (anchor_shifted - anchor_real).total_seconds() / 3600
        if not (20 < delta_h < 28):
            skipped.append((bid, f"shift delta {delta_h:.1f}h outside [20,28]"))
            continue

        win_real = get_window(uso, anchor_real, WINDOW_MIN)
        anchor_price_real = float(win_real.loc[anchor_real, "Close"])
        win_real["cum_pct_signed"] = (win_real["Close"] / anchor_price_real - 1) * 100
        win_real["cum_pct_abs"]    = win_real["cum_pct_signed"].abs()

        win_sh = get_window(uso, anchor_shifted, WINDOW_MIN)
        if anchor_shifted not in win_sh.index:
            continue
        anchor_price_sh = float(win_sh.loc[anchor_shifted, "Close"])
        win_sh["cum_pct_signed"] = (win_sh["Close"] / anchor_price_sh - 1) * 100
        win_sh["cum_pct_abs"]    = win_sh["cum_pct_signed"].abs()

        for _, r in win_real.iterrows():
            real_rows.append({
                "burst_id":      bid,
                "anchor_ts":     anchor_real.isoformat(),
                "minute_offset": int(r["minute_offset"]),
                "cum_pct_abs":   round(float(r["cum_pct_abs"]),    4),
                "cum_pct_signed":round(float(r["cum_pct_signed"]), 4),
                "series":        "real",
            })
        for _, r in win_sh.iterrows():
            shifted_rows.append({
                "burst_id":      bid,
                "anchor_ts":     anchor_shifted.isoformat(),
                "minute_offset": int(r["minute_offset"]),
                "cum_pct_abs":   round(float(r["cum_pct_abs"]),    4),
                "cum_pct_signed":round(float(r["cum_pct_signed"]), 4),
                "series":        "shifted_+24h",
            })

        burst_rows.append({
            "burst_id":           bid,
            "real_anchor":        anchor_real.isoformat(),
            "shifted_anchor":     anchor_shifted.isoformat(),
            "real_anchored_next_session":    anc_real_next,
            "shifted_anchored_next_session": anc_shifted_next,
            "real_max_abs_pct":   round(float(win_real["cum_pct_abs"].max()), 4),
            "shifted_max_abs_pct":round(float(win_sh["cum_pct_abs"].max()),   4),
        })

    real_df    = pd.DataFrame(real_rows)
    shifted_df = pd.DataFrame(shifted_rows)
    bm         = pd.DataFrame(burst_rows)

    combined = pd.concat([real_df, shifted_df], ignore_index=True)
    combined.to_csv(out / "chart1v2_timeshift_combined.csv", index=False)
    bm.to_csv(out / "chart1v2_timeshift_burst_metadata.csv", index=False)

    def envelope(df: pd.DataFrame, label: str) -> pd.DataFrame:
        pivot = df.pivot_table(index="minute_offset", columns="burst_id",
                                values="cum_pct_abs", aggfunc="first")
        return pd.DataFrame({
            "minute_offset": pivot.index,
            f"{label}_median": pivot.median(axis=1).round(4).values,
            f"{label}_p25":    pivot.quantile(0.25, axis=1).round(4).values,
            f"{label}_p75":    pivot.quantile(0.75, axis=1).round(4).values,
            f"{label}_n":      pivot.notna().sum(axis=1).values,
        })

    env_real = envelope(real_df,    "real")
    env_sh   = envelope(shifted_df, "shifted")
    env = env_real.merge(env_sh, on="minute_offset", how="outer").sort_values("minute_offset")
    env["gap_median"] = (env["real_median"] - env["shifted_median"]).round(4)
    env.to_csv(out / "chart1v2_timeshift_envelope.csv", index=False)

    with open(out / "chart1v2_timeshift.json", "w") as f:
        json.dump({
            "window_minutes": WINDOW_MIN,
            "shift_hours":    SHIFT_HOURS,
            "asset":          "USO",
            "y_axis":         "cum_pct_abs (positive-only |% from t=0|)",
            "envelope":       env.to_dict(orient="records"),
            "n_bursts":       len(bm),
            "interpretation": (
                "real_median minus shifted_median = the falsification gap. "
                "Positive values mean real-event windows show more movement "
                "than time-shifted controls at that offset. The signal of "
                "interest is the gap."
            ),
        }, f, indent=2)

    # Publication frame: ±60 min, with §3.4 rigorous test numbers built in
    pub = env[(env["minute_offset"] >= -60) & (env["minute_offset"] <= 60)].copy()
    pub_clean = pd.DataFrame({
        "minute_offset":      pub["minute_offset"],
        "real_median":        pub["real_median"],
        "real_p25":           pub["real_p25"],
        "real_p75":           pub["real_p75"],
        "real_n_bursts":      pub["real_n"].astype(int),
        "shifted_median":     pub["shifted_median"],
        "shifted_p25":        pub["shifted_p25"],
        "shifted_p75":        pub["shifted_p75"],
        "shifted_n_bursts":   pub["shifted_n"].astype(int),
        "gap_median":         pub["gap_median"],
    })
    pub_clean.to_csv(out / "chart1v2_falsification_publication.csv", index=False)

    pub_json = {
        "frame":            "publication",
        "window_minutes":   60,
        "asset":            "USO",
        "n_bursts":         15,
        "rth_anchored_burst_ids": [1, 2, 3, 4, 8, 13, 14],
        "weekend_or_overnight_anchored_burst_ids": [5, 6, 7, 9, 10, 11, 12, 15],
        "note_pre_event_coverage": (
            "8 of 15 bursts fired on weekends or outside trading hours; their next "
            "available trading bar (the anchor) lands several hours after the post, "
            "so they contribute only non-negative minute offsets to the median."
        ),
        "rigorous_falsification_test": {
            "source_file":                  "data/results/phase2b_timeshift.json",
            "reference":                    "report §3.4",
            "asset_xle_ofi_real_pre_mean":  0.04878,
            "asset_xle_ofi_real_boot_p":    0.0,
            "asset_xle_ofi_shifted_pre_mean": -0.00135,
            "asset_xle_ofi_shifted_boot_p": 0.7495,
            "asset_uso_vpinz_real_pre_mean":  0.351,
            "asset_uso_vpinz_real_boot_p":    0.0,
            "asset_uso_vpinz_shifted_pre_mean": -1.021,
            "asset_uso_vpinz_shifted_boot_p": 0.0,
        },
        "series": pub_clean.to_dict(orient="records"),
    }
    with open(out / "chart1v2_falsification_publication.json", "w") as f:
        json.dump(pub_json, f, indent=2)

    print(f"  timeshift: {len(real_df)} real rows + {len(shifted_df)} shifted rows "
          f"across {bm['burst_id'].nunique()} bursts")
    if skipped:
        print(f"  skipped {len(skipped)} bursts:")
        for bid, reason in skipped:
            print(f"    burst {bid}: {reason}")


# ---------------------------------------------------------------------------
# Chart 2 alternative — word-level networks (broad + strict)
# ---------------------------------------------------------------------------
def _build_word_network(post_texts: list[str], universe: str, n_posts: int,
                         out: Path, top_k: int = 30, min_edge_weight: int = 3,
                         out_prefix: str = "chart2alt") -> None:
    # Sort tokens before inserting into Counter to make insertion order
    # deterministic regardless of Python's set-hash randomisation. Otherwise
    # Counter.most_common ties resolve in different orders between runs.
    word_lists = [sorted(set(tokenize(t)) - BOILERPLATE) for t in post_texts]
    word_counts = Counter()
    for ws in word_lists:
        word_counts.update(ws)

    # Sort by (-count, alphabetical) so ties resolve deterministically
    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    top_words = [w for w, _ in sorted_words[:top_k]]
    top_set = set(top_words)

    edge_counts = Counter()
    for ws in word_lists:
        present = [w for w in ws if w in top_set]
        for a, b in combinations(sorted(present), 2):
            edge_counts[(a, b)] += 1

    nodes = [{"id": w, "label": w, "post_count": int(word_counts[w])} for w in top_words]
    nodes.sort(key=lambda n: (-n["post_count"], n["label"]))
    # Deterministic edge ordering: (-weight, source, target)
    links = [
        {"source": a, "target": b, "weight": int(w)}
        for (a, b), w in sorted(edge_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
        if w >= min_edge_weight
    ]

    payload = {
        "universe":         universe,
        "n_posts":          n_posts,
        "n_words_kept":     len(nodes),
        "n_edges_kept":     len(links),
        "top_k":            top_k,
        "min_edge_weight":  min_edge_weight,
        "nodes":            nodes,
        "links":            links,
    }
    with open(out / f"{out_prefix}_word_cooccurrence.json", "w") as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(nodes).to_csv(out / f"{out_prefix}_word_nodes.csv", index=False)
    pd.DataFrame(links).to_csv(out / f"{out_prefix}_word_edges.csv", index=False)
    print(f"  {out_prefix}: {len(nodes)} nodes, {len(links)} edges over {n_posts} posts")


def build_chart2_alt(posts: pd.DataFrame, out: Path) -> None:
    broad = posts[posts["topic_energy_oil"]].copy()
    _build_word_network(
        broad["text"].astype(str).tolist(),
        universe="all topic_energy_oil posts (broad — topic-classifier hits)",
        n_posts=len(broad),
        out=out, out_prefix="chart2alt_broad",
    )

    strict = posts[posts["text"].astype(str).apply(
        lambda t: bool(OIL_TERMS_RE.search(t)))].copy()
    _build_word_network(
        strict["text"].astype(str).tolist(),
        universe="posts containing explicit oil/Iran/Hormuz/crude/etc. terms",
        n_posts=len(strict),
        out=out, out_prefix="chart2alt_strict",
    )


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                        help="Path to trump-truth-social-replication repo (default: ../trump-truth-social-replication/)")
    parser.add_argument("--out",  type=Path, default=SCRIPT_DIR,
                        help="Output directory (default: this script's directory)")
    args = parser.parse_args()

    repo = args.repo.resolve()
    out  = args.out.resolve()

    if not repo.exists():
        sys.exit(f"ERROR: repo not found at {repo}\n"
                 f"Pass --repo /path/to/trump-truth-social-replication if it lives elsewhere.")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Repo:   {repo}")
    print(f"Output: {out}")
    print()

    uso    = load_uso(repo)
    posts  = load_posts(repo)
    bursts = load_bursts(repo)
    print(f"USO bars: {uso.shape}, posts: {posts.shape}, bursts: {bursts.shape}")
    print()

    print("[1/3] Chart 1 v2 (positive-only, ±180 min, 1-min interp)...")
    build_chart1_v2(uso, bursts, posts, out)

    print("\n[2/3] Time-shift falsification overlay (anchor + 24h)...")
    build_timeshift_overlay(uso, bursts, out)

    print("\n[3/3] Chart 2 alternative (word-level networks)...")
    build_chart2_alt(posts, out)

    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
