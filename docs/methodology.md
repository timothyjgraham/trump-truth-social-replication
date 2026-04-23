# Methodology notes

This is a longer companion to the in-text methodology section of
`report/report.pdf`. It documents the exact construction of every signal,
every filter, and every test used in the headline analysis.

## 1. Event universe

- All public statuses from `@realDonaldTrump` between `2026-02-22T00:00:00Z`
  and `2026-04-22T23:59:59Z` (the 60-day study window).
- 32,433 posts in the full archive, 1,847 in the 60-day window after
  deduplication.
- A post is **oil-themed** (`topic_energy_oil == True`) if its content
  matches one or more of the regular expressions in
  `code/01_scrape_truth_social.py::OIL_TOPIC_REGEXES` (oil, OPEC, crude,
  drill, gasoline, energy independence, …). 165 posts in the 60-day
  window meet this criterion.

## 2. Market data

- Tickers: DJT, SPY, QQQ, XLE, USO, GLD, UUP, VXX, XLF, XLK.
- Source: `yfinance` 5-minute OHLCV bars, regular-hours and pre/post
  combined.
- Yahoo only retains ~60 days of intraday history at 5-minute
  resolution, which is why the study window is exactly 60 days.

## 3. Microstructure signals

All signals are computed from the 5-minute bars and stored in
`data/interim/signals_5m/<TICKER>.parquet`.

### 3.1 Bulk-Volume Classification (Easley, López de Prado, O'Hara, 2012)

For bar *t* with log-return *r_t*, total volume *V_t*, and rolling-window
return standard deviation σ\_t:

```
buy_share_t  = Φ( r_t / σ_t )
sell_share_t = 1 − buy_share_t
buy_vol_t    = V_t · buy_share_t
sell_vol_t   = V_t · sell_share_t
signed_vol_t = buy_vol_t − sell_vol_t
OFI_bvc_t    = signed_vol_t / V_t        ∈ [−1, +1]
```

σ\_t is a 50-bar rolling standard deviation of log returns.

### 3.2 VPIN

VPIN is computed on volume-time buckets of size *V/N* where *V* is the
average daily volume and *N = 50* buckets per day.

```
VPIN_t = | buy_vol − sell_vol |_bucket / V_bucket
```

`vpin_z` is then VPIN z-scored against a 250-bar rolling mean and standard
deviation of itself, **shifted by one bar** so no future information can
enter the score:

```
vpin_z_t = ( VPIN_t − μ_{t−1:t−250} ) / σ_{t−1:t−250}
```

The shift is what makes pre-event mean(`vpin_z`) a clean leading
indicator rather than a contemporaneous one.

### 3.3 Kyle's λ (Kyle, 1985)

Within a rolling 50-bar window:

```
λ_t = | β_t |   where β_t = OLS slope of  Δp ~ signed_vol  on the window
```

`kyle_z` is `λ` z-scored on the same shifted 250-bar baseline as `vpin_z`.

## 4. Event study (stage 04)

For each oil-themed post at time τ:

- Bar index *p* = `bars.index.searchsorted(τ)`.
- Pre-window: bars [*p − PRE_BARS*, *p*).
- Post-window: bars [*p*, *p + POST_BARS*).
- Two window pairs are reported: (PRE = POST = 6) for the headline and
  (PRE = POST = 12) for the dollar-bound stage. Both are 5-min bars,
  so 6 = 30 min and 12 = 60 min.

For each metric *m* in `{logret, vol_z, dvol_z, signed_vol_tick,
OFI_bvc, vpin_z, kyle_z}`:

- `pre_mean_m`, `post_mean_m` are simple means of the bar-level signal
  inside the window.
- p-values are computed by stationary bootstrap (`B = 2000`, mean
  block length 12 bars), centred on the observed mean.

### Initiated-vs-reactive filter

A post is classified as **initiated** if `|pre_car_z| < 1.5`, where
`pre_car_z` is the cumulative pre-window log-return standardised by a
288-bar (one trading day) rolling return standard deviation.

The intuition is that we want to study posts that arrive *into* a quiet
tape, not posts that follow a large pre-existing move. The headline
findings are reported on the initiated subset; the report shows the
non-initiated sensitivity in the appendix.

## 5. Matched placebo (stage 05)

For each real event timestamp τ we draw 5,000 random non-event
timestamps from the same `(hour-of-day, weekday)` cell. The same
event-study pipeline runs on this distribution and the 95% interval of
its mean is the placebo band. A real-event mean outside the placebo
95% band, with placebo p > 0.05 on the same (asset, metric, window),
is what the report calls **placebo-clean**.

## 6. Multiple-testing correction (stage 06)

Across the consolidated table of (asset, metric, window) tests we apply
Benjamini–Hochberg with α = 0.10. The reported `p_fdr` column is the BH
adjusted p-value. A finding is **robust** when:

- Raw bootstrap p < 0.05 on real events.
- Placebo p > 0.05 on the same row.
- BH-adjusted p_fdr < 0.10.

The two surviving findings are (XLE, `OFI_bvc`, pre-window) and (USO,
`vpin_z`, pre-window).

## 7. +24-hour falsification (stage 07)

Each oil-themed post timestamp is shifted forward by exactly 24 hours
and the entire pipeline re-runs. If the headline finding is causally
tied to the posts themselves, the shifted version should produce a
near-zero signal that is no longer significantly different from zero.
If the finding is driven by daily seasonality, the shifted version
should look similar.

The USO `vpin_z` signal flips sign under the shift (real +0.380,
shifted −1.021), which is the strongest single piece of evidence
against a daily-seasonality interpretation.

## 8. Per-event dollar bound (stage 08)

For each oil-themed event with a usable pre/post window:

```
V_pre   = Σ signed_vol_t  over the 12 pre-event bars   (shares)
P_entry = close at bar (p − 1)                         (USD/share)
P_exit  = close at bar (p + POST_BARS − 1)             (USD/share)
PnL     = V_pre · ( P_exit − P_entry )                 (USD)
```

This is a **ceiling**, not a claim. It assumes a single perfectly-informed
trader was on the right side of the entire pre-event imbalance and could
take the position at the last pre-event close and exit at the post-window
close, with zero spread, zero market impact, and zero borrow cost.

The report headlines the **TRIGGERED** slice: events with pre-window
mean(`vpin_z`) > 0.5 (n = 81). Three other slices are reported in
`data/results/dollar_upper_bound_strategies.json` for context (ALL,
INITIATED, INITIATED + pre-buying).

## 9. Leave-one-out audit (stage 09)

For each of the 81 triggered events we recompute the aggregate sum with
that event removed. The reported LOO range is `[min, max]` of the 81
leave-one-out sums, and the `max_single_event_share_pct` is the largest
absolute single-event contribution as a percentage of the headline total.

Two conventions are reported:

- **A_production**: entry at `close[p − 1]`, exit at `close[p + POST_BARS − 1]`,
  pre-slice `[p − PRE_BARS, p)`. This is the convention used throughout the
  report and matches stage 08.
- **B_strict_pre**: a slightly stricter formulation that rolls the entry
  one bar earlier and the exit one bar later, eliminating any look-ahead
  ambiguity at the boundary. The headline number under B is $171.4M with
  max single-event share 14.9%.

The two conventions are within ~7% of each other, which is the report's
basis for treating the headline as robust to that boundary choice but
fragile to individual event removal.
