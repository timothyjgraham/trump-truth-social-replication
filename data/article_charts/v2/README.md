# Chart data v2 — Trump / Truth Social / oil

This folder contains the v2 chart-data bundle for the article — a
follow-up to the original `matt_chart_data/` bundle that addresses the
questions raised in the email thread.

## What's in this folder

```
README.md                                   # this file

# Code (for reproducibility — running it isn't required to use the data)
build.py                                    # rebuilds every CSV/JSON
render_previews.py                          # rebuilds the preview PNGs
verify.py                                   # consistency checks against source
run.sh                                      # build + render + verify
requirements.txt                            # Python dependencies

# Chart 1 — pre/post event overlay
chart1v2_event_window_5min.csv              # 745 rows; ±180 min × 15 bursts
chart1v2_event_window_1min.csv              # 3,735 rows; linearly interpolated
chart1v2_event_window.json                  # nested D3-friendly version
chart1v2_envelope_abs.csv                   # cross-burst median + IQR + 10/90
chart1v2_burst_metadata.csv                 # one row per burst

# Time-shift falsification overlay (real vs +24h-shifted)
chart1v2_timeshift_combined.csv             # both series stacked
chart1v2_timeshift_envelope.csv             # gap medians at each offset
chart1v2_timeshift_burst_metadata.csv       # per-burst real/shifted summary
chart1v2_timeshift.json                     # full ±180 D3-friendly bundle
chart1v2_falsification_publication.csv      # ±60 min publication frame
chart1v2_falsification_publication.json     # publication frame + §3.4 numbers

# Chart 2 alternative — word-level network (two universes)
chart2alt_broad_word_nodes.csv              # 173 oil-themed posts (broad)
chart2alt_broad_word_edges.csv
chart2alt_broad_word_cooccurrence.json
chart2alt_strict_word_nodes.csv             # 121 explicit-oil-term posts
chart2alt_strict_word_edges.csv
chart2alt_strict_word_cooccurrence.json

previews/                                   # PNG previews
├── falsification_publication_frame.png     # the recommended publication chart
├── chart1v2_zerovol_compare.png            # all bars vs nonzero-volume only
├── chart2alt_word_compare.png              # broad vs strict word universes
└── burst13_extended.png                    # burst 13 in the wider window
```

## What changed from v1

  - **Chart 1 window widened** from ±60 min → ±180 min (3 hours each side).
    This automatically covers the burst 13 ask for an extra 2 hours of
    pre-event data.
  - **Chart 1 y-axis transformed to positive-only**:
    `cum_pct_abs = |cum % return from t=0|`. Up-moves and down-moves both
    register as movement away from baseline. The original signed series
    (`cum_pct_signed`) is preserved alongside, so a directional toggle is
    a one-column swap in the renderer.
  - **1-min resolution offered** as a linearly-interpolated overlay on
    the underlying 5-min real bars. The `is_real_bar` column distinguishes
    real vs interpolated rows. (Why interpolated rather than fetched: see
    *Constraints* below.)
  - **Volume / RTH flags added** to every Chart 1 row (`volume`,
    `is_zero_vol`, `is_rth_bar`). Useful for filtering or visually
    distinguishing thin-quote bars.
  - **New: time-shift falsification overlay.** For every burst, a parallel
    window anchored 24 hours after the real anchor (same time of day) is
    computed. The aggregate envelopes for the two populations are the
    "red vs green" visual.
  - **New: word-level networks** (broad and strict universes) as an
    alternative to the topic-tag bubble chart in v1.

## Constraints worth knowing before captioning anything

  1. **Zero-volume bars in extended hours.** yfinance ships extended-hours
     bars (4:00–9:30 ET pre-market, 16:00–20:00 ET after-hours) where USO
     can have **zero traded volume** but quote-driven price movement.
     For bursts anchored outside RTH (5, 6, 7, 9, 10, 11, 12, 14, 15 —
     9 of 15), most or all of the pre/post-event bars in the wider window
     are zero-volume quote walks rather than executed trades. Every row
     in `chart1v2_event_window_5min.csv` carries `volume`, `is_zero_vol`
     and `is_rth_bar` so this can be filtered or shaded as preferred.
     The right-hand panel of `previews/chart1v2_zerovol_compare.png`
     shows what the median looks like with zero-volume bars excluded.

  2. **Burst 15's wider-window post-event move (-12.3% by +180 min) is
     dominated by zero-volume after-hours bars** — but it is not
     fictional. USO opened the next morning (April 8) at $118.90 (vs the
     $137.76 anchor close on April 7), confirming the overnight reprice
     did happen. The after-hours bars are gradually pricing toward that
     next-day open via quote movements rather than trades. Accurate
     framing: this is real overnight repricing, not intraday trading.

  3. **Data gap on Feb 2, 2026.** The USO parquet has no bars between
     Sunday Feb 1 and Monday Feb 2 13:45 ET. Bursts 1, 2 and 3 all have
     post timestamps that fall in this gap and are anchored at Feb 2
     13:45 ET. Their pre-event windows are empty because there's no
     data behind the gap. This is a yfinance limitation, not a pipeline
     issue. The `anchored_to_next_session` flag in
     `chart1v2_burst_metadata.csv` captures this.

  4. **Real 1-min bars aren't retrospectively available.** yfinance caps
     1-min history at the last 7 days, so the historical Jan-Apr 2026
     window can't be re-fetched at that resolution. The 1-min data here
     is linearly interpolated between 5-min real bars — fine for visual
     smoothness, not actual granular data. A paid intraday feed
     (Polygon, Alpaca, IEX Cloud) would let real 1-min bars be
     substituted; quick to wire up if needed.

  5. **The falsification visual is a storytelling proxy for the report's
     §3.4 falsification check, not the rigorous test itself.** The
     rigorous test uses microstructure signals (OFI_bvc, VPIN-z,
     signed-volume-tick) on a 30-min pre-event window with bootstrap
     p-values. The chart in this bundle uses |cum % return| as a rough
     cumulative price-movement proxy — it visualises the *concept*, not
     the test. The statistical claim should be sourced from the original
     `data/results/phase2b_timeshift.json`. Reference values:
     - **XLE OFI**: real +0.04878 (bootstrap p ≈ 0); +24h shifted
       −0.00135 (p = 0.7495).
     - **USO VPIN-z**: real +0.351 (p ≈ 0); +24h shifted −1.021
       (p ≈ 0).
     These numbers are also embedded in
     `chart1v2_falsification_publication.json` under
     `rigorous_falsification_test`.

---

## Chart 1 — pre/post event overlay

**Files**
  - `chart1v2_event_window_5min.csv` — real 5-min bars, ±180 min around
    each of the 15 burst anchors. Columns:
    - `burst_id` — 1..15
    - `burst_ts_anchor` — anchor timestamp (5-min snapped post timestamp;
      for weekend-anchored bursts this is the next available trading bar)
    - `minute_offset` — −180..+180 (5-min steps)
    - `cum_pct_abs` — **|% deviation from t=0|** (the recommended
      headline series)
    - `cum_pct_signed` — original signed cumulative % return
    - `price_close`, `anchor_price` — raw prices in USD
    - `volume`, `is_zero_vol`, `is_rth_bar` — for filtering / shading
    - `in_stat_window` — `True` where `|minute_offset| ≤ 30` (the rigorous
      analytical window the underlying study uses)
  - `chart1v2_event_window_1min.csv` — same shape with linear
    interpolation between bars. `is_real_bar = True` only at multiples
    of 5.
  - `chart1v2_event_window.json` — D3-friendly nested form; one object
    per burst with metadata + a `series_5min` array.
  - `chart1v2_envelope_abs.csv` — median + 25/75 + 10/90 percentiles
    across the 15 bursts at each minute offset, on the absolute series.
    Drop in as a bold median over faint individual lines.
  - `chart1v2_burst_metadata.csv` — one row per burst with summary
    columns including `n_posts`, `pnl_total`, `sample_post_text`,
    `anchored_to_next_session`, `max_abs_pct_pre`, `max_abs_pct_post`.

**Burst 13 specifically**
  - Anchor: 23 Mar 2026, 10:25 ET (RTH).
  - Window now covers 07:25 → 13:25 ET.
  - Pre-event peak: |% deviation| 3.75% at −145 min (~08:00 ET, pre-market;
    note pre-market USO trades on thin volume).
  - Post-event peak: 4.07% at +45 min, then mean-reverts.

---

## Time-shift falsification overlay

For every burst, a parallel window is anchored 24 hours after the real
anchor (same wall-clock time of day). The gap between the two populations
is the visual signal of interest.

### Recommended publication frame

  - **Window: ±60 min** (the analytical frame the underlying study uses).
    The ±180 min data is in the bundle for completeness, but the wider
    frame is partly inflated by extended-hours quote walks and isn't
    recommended for the lead chart.
  - **Population: full 15 bursts.** At negative offsets the median is
    computed over the 7 RTH-anchored bursts (the other 8 are weekend or
    overnight-anchored and contribute only post-event data); at non-
    negative offsets all 15 bursts contribute. A small footnote on the
    chart should explain this — see
    `previews/falsification_publication_frame.png` for an example
    rendering.
  - **Caption** the chart as a visual companion to the §3.4 falsification,
    sourcing any quoted statistical claim from the rigorous test rather
    than this chart.

**Publication-ready files** (drop straight into the renderer):
  - `chart1v2_falsification_publication.csv` — 25 rows, ±60 min frame,
    real and shifted median + IQR + n_bursts at each offset, plus a
    `gap_median` column (`real_median − shifted_median`).
  - `chart1v2_falsification_publication.json` — same data plus the
    rigorous §3.4 test numbers in a `rigorous_falsification_test` block.
  - `previews/falsification_publication_frame.png` — example rendering
    with footnote and §3.4 callout.

### Wider exploratory data (in the bundle for completeness)

`chart1v2_timeshift_combined.csv`, `_envelope.csv`,
`_burst_metadata.csv`, and `chart1v2_timeshift.json` cover the full
±180 min window. Useful for diagnostic views; not recommended for the
lead chart.

---

## Chart 2 alternative — word-level networks

Two universe options:

### Broad — all 173 topic_energy_oil posts
  - Files: `chart2alt_broad_word_*.{csv,json}`
  - Top words: military, energy, american, dominance, complete, total,
    taxes, siege, second, endorsement, amendment, cut, regulations…
  - **What this surfaces**: most posts the topic classifier flagged as
    oil-themed are endorsement-spam content where "energy" /
    "regulations" appear as campaign-rhetoric. This is the honest view
    of the classifier's limits.

### Strict — 121 posts containing explicit oil/Iran/Hormuz/crude terms
  - Files: `chart2alt_strict_word_*.{csv,json}`
  - Top words: iran (91), president (76), trump (74), donald (60),
    military (39), states (38), united (38), america (34), oil (23),
    strait (23), hormuz (22), middle (20), east (19), regime (18),
    deal (17), nuclear (17), war (15), attack (14), israel (12)…
  - Headline triangles in the network: hormuz–strait (22), iran–strait
    (18), iran–hormuz (17), iran–nuclear (16), iran–deal (13).
  - **What this surfaces**: the actual Iran / Middle East / Strait of
    Hormuz / regime / nuclear-deal narrative.

The strict universe gives the cleaner narrative chart; the broad
universe gives the honest "this is the classifier's full output" view.
Both are in the bundle so the choice is a one-file swap.

**JSON shape** (both variants identical):
```
{
  "universe": "...",
  "n_posts":  int,
  "nodes":    [{"id": word, "label": word, "post_count": int}],
  "links":    [{"source": word, "target": word, "weight": int}]
}
```

---

## Framing notes for the published article

A few things worth keeping in mind given what's in the data:

  1. **The "$160M PnL across 15 bursts" headline** breaks down as **−$35M
     across the genuine Iran/oil single-post bursts** (2, 11, 13 — these
     collectively *lost* in the hypothetical short-side framework,
     dominated by burst 13's −$31.93M) and **+$197M across the multi-post
     endorsement bursts** (1, 5, 7, 14, 15, dominated by burst 15's
     +$212.92M). The hypothetical PnL is a stylised short-side measure
     (enter at t=0 close, exit at +60 min), not real-money.

  2. **Burst content vs timing**: most of the multi-post bursts were
     classified as "oil-themed" because they fired during high-Iran-
     tension news cycles, not because their *content* was Iran/oil. The
     genuine Iran/oil content is mostly in the single-post bursts (2, 11,
     13). The aggregate signal is real (and survives both the placebo
     and time-shift checks at the population level), but the most
     defensible sentence is "USO moved in a non-random, asset-specific
     way around 81 Trump Truth Social timestamps, 36 of which the topic
     classifier identifies as carrying oil-related language" rather than
     "Trump's oil posts caused informed pre-positioning." A burst-content
     map is in `chart1v2_burst_metadata.csv` (`sample_post_text`).

  3. **The lead chart's effect-size language**: the gap on |% return|
     medians between real and +24h-shifted populations is a visual proxy.
     Effect-size language in the article should come from the rigorous
     §3.4 test (XLE OFI = +0.048, p ≈ 0; shifted −0.001, p = 0.75)
     rather than from this chart.

---

## Reproducing this bundle

The build is deterministic — same inputs always produce the same outputs.
You don't need to run anything to use the data; this is provided for
reproducibility/verification.

### Layout assumed

```
parent_directory/
├── trump-truth-social-replication/   # source repo (read-only)
│   └── data/
│       ├── raw/posts_60d.parquet
│       ├── raw/minute_bars_5m/USO.parquet
│       └── results/pnl_concentration_bursts.csv
└── this_folder/
    ├── build.py
    ├── render_previews.py
    ├── verify.py
    ├── run.sh
    └── ...
```

### Quickstart

```bash
pip install -r requirements.txt
./run.sh
```

`run.sh` runs three steps in order:

1. **`python build.py`** — reads from `../trump-truth-social-replication/`
   and writes the CSV/JSON outputs into this folder.
2. **`python render_previews.py`** — renders the four sanity-check PNGs
   into `previews/`.
3. **`python verify.py`** — runs ~50 consistency checks against the
   source data and exits non-zero if any fail.

If the source repo lives elsewhere, override:

```bash
python build.py  --repo /path/to/trump-truth-social-replication
python verify.py --repo /path/to/trump-truth-social-replication
```

### Source data this bundle reads

  - `data/raw/posts_60d.parquet` — 1,341 Truth Social posts (25 Jan – 8 Apr)
  - `data/raw/minute_bars_5m/USO.parquet` — 5-min OHLCV (26 Jan – 21 Apr)
  - `data/results/pnl_concentration_bursts.csv` — 15 collapsed bursts
