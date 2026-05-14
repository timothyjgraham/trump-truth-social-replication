# Chart data for Matt — Trump / Truth Social / oil

**Date:** 29 April 2026
**From:** Tim Graham
**To:** Matt Garrow
**Post window (ET):** 25 January → 8 April 2026 (74 calendar days)
**Price window (ET):** 26 January → 21 April 2026 (60 trading days, 86 calendar days)
**Universe:** 1,341 Truth Social posts; 173 oil-themed; 81 triggered events; 15 collapsed bursts

This folder has everything you need for the four charts you asked for, in CSV
and JSON. All times are UTC unless explicitly suffixed `_et`. Prices are in USD.

If you want one entry point per chart, use the JSON files — they're
nested in the shape D3 wants. The CSVs are the same data flattened in case it's
easier to load with `d3.csv`.

---

## Decisions baked in

A few choices we locked before generating these files — flag any you want changed:

  - **Chart 1 events = 15 collapsed bursts** (consecutive posts within 10 minutes
    treated as one episode), not the 81 raw posts. This avoids 28 near-identical
    lines for the 7 April burst dominating the overlay.
  - **Chart 1 asset = USO only.** XLE and the oil-equity signals are in
    chart 3; chart 1 stays focused on the headline ETF.
  - **Chart 1 y-axis = cumulative percent return from t=0.** Each line starts
    at 0% at the post timestamp; values are % change vs the close of the bar at
    t=0. This matches your mockup visually.
  - **Chart 1 window = ±60 minutes**, with an `in_stat_window` flag for ±30 min
    (the rigorous statistical window we report in the paper).
  - **Chart 3 cell = pre-event mean signal**, on the *initiated* subset
    (events where pre-CAR z-score < 1.5 — i.e., where price wasn't already
    moving). This matches the universe used for the asset-specificity claim
    in the report.

---

## Chart 1 — Pre/post event overlay (the three-stage scroll chart)

You wanted minute offsets on the x-axis and "market movement" on the y. I went
with cumulative % return from t=0. Each line starts at the origin.

**Files**
  - `chart1_event_window.json` — D3-friendly nested format. Top-level is an
    array of 15 burst objects; each has metadata + a `series` array of
    `{minute_offset, cum_pct_return, in_stat_window}` points.
  - `chart1_event_window.csv` — same data flat (262 rows). Columns:
    - `burst_id` — 1..15, ordered by ts_first ascending.
    - `burst_ts_anchor` — the 5-min bar timestamp used as t=0.
    - `minute_offset` — integer, −60 to +60 in 5-minute steps.
    - `cum_pct_return` — % change vs anchor close.
    - `price_close` — raw close in USD (kept in case you want a $ y-axis).
    - `anchor_price` — close at t=0 for that burst.
    - `in_stat_window` — `True` where `|minute_offset| ≤ 30` (the rigorous
      window). Use this to shade ±30 min if you want.
  - `chart1_burst_metadata.csv` — one row per burst (15 rows):
    - `n_posts`, `duration_minutes`, `pnl_total`, `sample_post_text`, etc.
    - `anchored_to_next_session` — `True` for weekend / overnight bursts that
      anchor to the next available trading bar (8 of 15). See "weekend bursts"
      below.
    - `is_apr07_burst` — flags burst 15, the 14-minute April 7 burst that
      drives the headline PnL number. Date-and-duration flag, not a content
      flag — see "burst-content map" below.
  - `chart1_envelope.csv` — median, 25/75th and 10/90th percentile across the
    15 bursts, by minute offset. Use this to overlay a bold median line on top
    of the faint individual lines (the "robust to outliers" version of the
    overlay).
  - `chart1_posts_by_burst.csv` and `chart1_posts_by_burst.json` — **one row
    per post that falls inside a burst's [ts_first, ts_last] window**
    (83 rows across 15 bursts), with full text, ET and UTC timestamps,
    engagement metrics, topic flags, and an `is_triggered_event` flag
    (True for the 81 posts that are in the source triggered-events list,
    False for the 2 in-window posts that were not individually triggered —
    see "post-count discrepancy" below). Keyed by `burst_id`. Use this for
    tooltips, side-panel annotations, or any "what did he actually post?"
    UI element.

**Post-count discrepancy.** `chart1_burst_metadata.csv` reports `n_posts =
28` for burst 15 (and `n_posts = 19` for burst 7); `chart1_posts_by_burst.csv`
contains 29 and 20 rows for those bursts respectively. The metadata count is
*triggered events* (post made the 81-event list); the per-post file is *all
posts inside the burst's time window* (which can include a post that fell in
the window but failed the analytical pipeline's filters). The
`is_triggered_event` column distinguishes them. Sum of `n_posts` in the
metadata is 81; row count of the per-post file is 83 (= 81 triggered + 2
non-triggered posts in burst windows).

### Stage 1 of the chart — "a normal signal: market moves *after* post"

This is the conceptual / control case for stage 1 of your scroll. Two
options, complementary:

  - `chart1_placebo_envelope.csv` — **the rigorous null.** 500 randomly
    sampled USO 5-minute bars (excluding any bar within ±30 min of a known
    post timestamp), each treated as a synthetic t=0, with cum % return
    computed across ±60 min. Aggregated to median + 25/75 + 10/90
    percentiles. The median line is essentially flat (max ≈0.08% drift),
    which is exactly what we want it to be. Use this as the "this is what
    no-event windows look like" reference series.
  - `chart1_stage1_exemplars.csv` and `chart1_stage1_exemplar_metadata.csv`
    — **5 single-post illustrations.** Real Trump posts that are *not* part
    of the 15 bursts and where USO reacted cleanly *after* the post with
    minimal pre-event drift. Each row in the metadata file has a `session`
    column (RTH / pre-market / after-hours / overnight). Of the 5 candidates:
    - **18 March** (post 116249915914904632, Iran "finished off the Iranian
      Terror State…") — **pre-market** (07:22 ET), USO +1.81% over +60 min,
      perfectly monotonic. Strongest signal, but the reaction is in
      pre-market thin liquidity, not RTH.
    - **4 March** (Venezuela oil, Bloomberg link) — **RTH** (15:09 ET),
      USO +1.20% over +60 min, monotonic. Cleaner journalistic choice
      because the reaction is in regular trading hours and the post
      content is genuinely oil-related.
    - 29 January (Powell / rates) — pre-market, +1.19%.
    - 27 February (Newsmax media post) — pre-market, +1.07%.
    - 26 March (Iran RT) — pre-market, −0.39% final / +0.90% peak.

  **Recommendation:** for the published article, use the 4 March Venezuela
  oil exemplar (RTH + oil content) as the single-line illustration. The
  18 March Iran post is the more dramatic line but the pre-market caveat
  matters — pre-market USO can move on very thin volume and isn't directly
  comparable to RTH activity. If you want the punchier visual, pair the
  18 March line with a small annotation noting the time of day.

How to use them in stage 1:
  - The exemplar gives you a concrete "here's what an efficient market
    reaction *should* look like" line — a real Trump post, real market move,
    no pre-positioning.
  - The placebo envelope gives you the statistical baseline — "and here's
    what nothing-happening looks like across hundreds of windows."
  - Stage 2 then introduces the inverted pattern (pre-event drift before t=0).
  - Stage 3 overlays all 15 bursts to show the pattern is systematic.

**Weekend / overnight bursts.** Eight of the 15 bursts fired on weekends or
outside trading hours, so there's no bar at the actual post timestamp. For
those bursts, t=0 is anchored to the *next* available trading bar — typically
several hours later. They'll have data points only at non-negative minute
offsets, which is honest but visually short. Two ways to handle in D3:
  - Hide weekend-anchored bursts in the overlay, show them on hover only.
  - Render them in a lighter shade and let the eye separate them from the
    fully-covered bursts.

**Outlier handling.** Burst 15 (7 April) is the visual standout in the
overlay, but the *direction* of the standout is worth understanding before
you wire it up. Across the ±60 min window, USO actually *fell*: cum return
was +2.72% at −60 min, drifted down to 0% at t=0 (the snap-to-bar anchor),
and continued falling to −1.17% by +60 min. The headline +$213M PnL for this
burst is hypothetical *short-side* P&L on that fall (entry ~$137.76, exit
~$136.15) multiplied by 28 separate "trades" — one per triggered post — not a
long position riding a +10% spike. Three options for the overlay:

  - Use `chart1_envelope.csv` (median + IQR) as the bold series and render
    individual bursts as faint lines underneath. This is what the third stage
    of your mockup is doing visually, and it's the cleanest treatment.
  - Show all 15 burst lines on a y-axis clipped to roughly ±3%, since most
    bursts move within that band even at ±60 min. Burst 15's pre-event
    +2.72% is at the upper edge of that range.
  - Keep an unclipped y-axis. The April 7 line is still the visual standout
    but in a less dramatic way than I implied earlier — it's a ~3% pre-event
    drop into t=0, not a +10% post-event spike.

**The single "pre-post" example** (your stage 2). The cleanest visual of
"price was already moving before the post" is **burst 15 itself** — the
−60 min to t=0 segment shows USO drifting down ~2.7%, exactly the kind of
pre-event drift the article is about. Use the segment from −60 to ~+10 min
and you get the pre-post pattern without the noisy post-event tail. Other
candidates: burst 14 (24 March, 12.8 min, $+9.27M) and burst 5 (10 February,
4.2 min, $0 PnL, after-hours) are both shorter intra-session bursts, but
their textual content is also endorsement-style (see burst-content map
below) so they're not Iran/oil illustrations.

If you'd rather have a cleaner stand-alone *content* match for stage 2 — a
single Trump post about Iran or oil where USO drifts before t=0 — the
candidates are burst 11 (10 March, "If Iran does anything that stops the
flow of Oil within the Strait of Hormuz, they will be hit by the United
States of America TWENTY TIMES HARDER…"), or burst 13 (23 March, Iran
ceasefire RT, $-31.93M short-side PnL). Both are single-post bursts.
Burst 11 is weekend-anchored so its line in the overlay is short.

### Burst-content map (important — read this before captioning chart 1)

The 15 bursts contain a mix of content types. The aggregate "$160M PnL across
15 oil-themed bursts" framing is a bit looser than it sounds: most of the
multi-post bursts are *endorsement-spam* posts (Indiana / Texas / Florida
state-rep endorsements that happened to be tagged `topic_energy_oil` and
`topic_iran_military` by the topic classifier — almost certainly because
they fired during high-Iran-tension news cycles when oil prices were moving).
Genuine oil/Iran content lives mostly in single-post bursts.

| Burst | Date  | Posts | Content type                        | PnL ($M) |
|-------|-------|-------|-------------------------------------|----------|
| 1     | 1 Feb |  7    | Indiana endorsements (RT)           | −24.91   |
| 2     | 2 Feb |  1    | **Venezuela oil** (Bloomberg link)  | −3.56    |
| 3     | 2 Feb |  1    | India / Modi call                   | −3.56    |
| 4     | 4 Feb |  1    | Xi / China call                     | +0.04    |
| 5     | 10 Feb|  7    | Florida endorsements                |  0       |
| 6     | 17 Feb|  4    | Texas endorsements                  |  0       |
| 7     | 17 Feb| 19    | Texas endorsements                  |  0       |
| 8     | 17 Feb|  1    | Texas endorsement (RT)              | +1.60    |
| 9     | 19 Feb|  1    | Judicial nomination                 |  0       |
| 10    | 3 Mar |  1    | Montana endorsement                 |  0       |
| 11    | 10 Mar|  1    | **Iran / Strait of Hormuz**         |  0       |
| 12    | 21 Mar|  1    | Louisiana endorsement               |  0       |
| 13    | 23 Mar|  1    | **Iran ceasefire** (RT)             | −31.93   |
| 14    | 24 Mar|  7    | Indiana endorsements                | +9.27    |
| 15    | 7 Apr | 28    | Indiana endorsements                | +212.92  |

Bolded rows are the genuine Iran/oil content. The other 12 bursts are
classifier-tagged as oil-themed but the *textual* content is something else.
The market signal is real (the events survived placebo and time-shift
falsification at the population level), but the most defensible framing is
"USO moved in a non-random way around these 81 timestamps," not "USO moved
*because of* these 81 oil-themed posts." Worth thinking about what to
caption the chart with — most of the dramatic movement is around endorsement
spam, not Iran/oil content per se.

---

## Chart 2 — Topic co-occurrence bubble / network

Nodes are the 8 topic tags; node size is post count (over the 60-day window);
edges are co-occurrences within the same post.

**Files**
  - `chart2_topic_cooccurrence.json` — D3 force-layout shape:
    `{nodes:[{id,label,post_count}], links:[{source,target,weight}]}`.
  - `chart2_topic_nodes.csv` — 8 rows; `id`, `label`, `post_count`.
  - `chart2_topic_edges.csv` — 22 rows; `source`, `target`, `weight`.

**The story this surfaces.** The dominant edges are *Iran/Military ↔
Energy/Oil* (162), *Iran/Military ↔ Markets/Economy* (139), and *Energy/Oil ↔
Markets/Economy* (136). That triangle is what's driving the article — Iran
posts triggering oil moves through a markets-economy frame. *Tariffs/Trade* is
a much smaller node and connects more weakly. *Crypto* is essentially
non-existent in this window (1 post).

**Sample sizes.** 463 of 1,341 posts (34.5%) carry at least one of the 8 topic
tags; 233 of those carry more than one. The remaining 65% are off-topic
(sports, media commentary, personal posts) and aren't part of this chart by
design.

If you want word-level node labels rather than topic labels, that's a separate
extraction — not done here. Easy to do as a follow-up if you want it.

---

## Chart 3 — Topic × asset heatmap of pre-event signal

Rows = topics (5 — those we ran the event study on), columns = assets (10).
Cell = mean of the chosen signal in the 30-minute window before each post,
averaged across events.

**Files (recommended starting point first)**
  - `chart3_heatmap_OFI_pre_initiated.csv` — long format, 50 cells.
    Headline view: pre-event OFI_bvc on the initiated subset. Use this.
  - `chart3_heatmap_OFI_pre_initiated_wide.csv` — same data pivoted to a
    5×10 matrix (topic_label rows, asset cols).
  - `chart3_heatmap.json` — long format, all signals × subsets × topics ×
    assets (900 cells) for D3 if you want a signal/subset toggle.
  - `chart3_heatmap_long.csv` — same as above in CSV.
  - `chart3_heatmap_OFI_pre_all.csv` / `_all_wide.csv` — same view but on
    *all* events (not just the initiated subset). Less clean but a useful
    sanity check.
  - `chart3_heatmap_vpin_z_pre_initiated.csv` — pre-event VPIN-z (toxicity)
    on initiated events. This is where USO's signal lives (see below).
  - `chart3_heatmap_dvol_z_pre_initiated.csv` — pre-event dollar-volume z.
  - `chart3_oil_complex_extension.csv` — *important supplement.* The
    extended oil-equity finding (XOM, CVX, plus XLE / USO survivor cells from
    the same finding-4 analysis) — these aren't in the main 10-asset
    universe but are part of the report's headline. 10 rows. Columns:
    `pre_mean`, `placebo_mean`, `delta_vs_placebo`, `p_fdr_trim` (the
    FDR-corrected p-value, ≤0.05 for every survivor), `fdr_passed`.

**Columns in the long files**
  - `topic`, `topic_label`, `asset`, `signal`, `subset` (`initiated` | `all`),
    `n_events`, `pre_mean`, `pre_t`, `pre_p`, `pre_boot_p` (bootstrap p-value),
    `post_mean`, `post_t`, `post_p`, `post_boot_p`.

**A nuance worth understanding before you wire the heatmap.** The cleanest
buy-side informed-flow signal in OFI_bvc lives on **XLE (+0.048, p ≈ 2×10⁻¹²)**,
not USO. USO's pre-event signature lives in **vpin_z (+0.38)** and
**signed_vol_tick (negative)** rather than directional OFI. So if you want
the visual story to read "the oil row lights up only on oil-related assets,"
you want one of two things:
  - Show the OFI_bvc heatmap and accept that USO's cell is muted; the story
    is then *XLE / XOM / CVX*. This is the cleanest visual.
  - Build a multi-panel heatmap (one per signal) so OFI tells the buy-side
    story while VPIN tells the toxicity story. More information, more
    cluttered.

The `chart3_oil_complex_extension.csv` adds **XOM (+0.033 OFI)** and
**CVX (+0.038 OFI)** so the "oil complex" version of the heatmap shows the
oil-row lighting up cleanly across XLE, XOM, CVX.

**P-values and FDR.** The `pre_boot_p` column is the bootstrap p-value (most
trustworthy). The full FDR-corrected survivors list lives in the source
report at `data/results/orderflow_fdr_5m.json` if you need the multiple-testing
adjusted version.

---

## Chart 4 — Overall market timeline + post overlay + posting-pressure

Three series for the same x-axis.

**Files**
  - `chart4_timeline.json` — combined: `uso_daily_close`, `xle_daily_close`,
    `burst_markers`, `daily_post_counts`. Single load.
  - `chart4_uso_daily_close.csv` — 60 rows, `date_et`, `uso_close` (USD).
  - `chart4_xle_daily_close.csv` — 60 rows, `date_et`, `xle_close` (USD).
    XLE = energy-sector ETF, our second oil-complex line. USO swung
    $73.48 → $138.91 (+89%) over the window; XLE swung $49.20 → $62.56
    (+27%) on the same news cycle. Same shape, different magnitudes.
  - `chart4_oil_complex_daily_close.csv` — both daily series joined on
    `date_et` (`uso_close`, `xle_close`). Use this if you want a single
    file for a dual-line chart.
  - `chart4_uso_hourly_close.csv` / `chart4_xle_hourly_close.csv` — 1-hour
    resolution for smoother lines. RTH only.
  - `chart4_burst_markers.csv` — 15 rows, one per collapsed burst, with
    `ts_first_utc`, `ts_first_et`, `date_et`, `n_posts`, `pnl_total`,
    `sample_text`. Use these as scatter dots on the price line.
  - `chart4_event_markers.csv` — 81 rows, one per individual triggered post,
    in case you want finer-grained dots instead of the 15 collapsed markers.
  - `chart4_daily_post_counts.csv` — 74 rows, `date_et`, `n_posts` (all topics),
    `n_oil_posts` (oil-themed only). Use as a secondary-axis bar chart for
    posting-pressure.

**Headline numbers in the timeline.**
  - USO range over the window: **$73.48 → $138.91** (+89%, low to high).
  - XLE range over the window: **$49.20 → $62.56** (+27%).
  - Daily post counts (all topics, all 1,341 posts): median 15, max 75,
    mean 18. (This is the total Truth Social post count per day, not the
    topic-tagged subset.)
  - Daily oil-themed posts: max 28 on 7 April (all 28 inside burst 15).

**Suggested rendering.** USO as the primary y-axis. XLE on a secondary y-axis
(the magnitudes don't share a scale, but both have meaningful highs/lows over
the window). Burst markers as red dots on the USO line at each `ts_first_et`,
sized by `n_posts`. Daily post counts as a thin grey bar chart on a third
axis below the price lines, with oil posts in a darker shade. The oil-complex
file makes the dual-line plot a one-import job.

---

## Common gotchas

  - **Time zones.** The price data is UTC-indexed (NYSE bars are stored UTC
    in our pipeline). For human-readable labels in the article, the `_et`
    columns are converted to America/New_York which respects daylight saving.
    Check before you put a `Tuesday 4:13 pm` label on anything.
  - **Weekend bursts.** As above for chart 1, eight of the 15 bursts fired
    outside trading hours. For chart 4 markers this isn't a problem (the dot
    just sits at the post date). For chart 1 it is — see weekend handling.
  - **The 7 April PnL is short-side, not long-side.** The +$212.92M PnL is
    hypothetical short P&L: the strategy goes short at the entry price
    (~$137.76, the close at t=0) and exits at the lower exit price
    (~$136.15, ~+60 min later) for a ~1.2% per-trade gain, multiplied by 28
    "trades" — one per triggered post in the burst. Critically, USO *fell*
    during this burst (the price line goes *down*); the +$213M is the gain
    a short position would have realised on that fall, not a sign that USO
    spiked. Casual readers will assume USO went up — they shouldn't.
  - **Anchor bars vs RTH.** The `chart1_event_window.csv` anchors to the
    nearest 5-min bar at or after the post timestamp, including extended-
    hours bars. The `anchored_to_next_session` flag tells you whether the
    next-bar lookup crossed a session boundary; it does not tell you whether
    the anchor bar itself is RTH vs extended. If you need to filter strictly
    to RTH-anchored bursts, cross-reference `anchor_bar_ts` against trading
    hours (09:30–16:00 ET).
  - **Stage-1 exemplars are mostly pre-market.** Four of the five candidates
    have post timestamps before 09:30 ET, where USO trading volume is thin.
    The 4 March Venezuela oil exemplar is the only RTH option in the top
    five. The `session` column in the metadata file flags each.
  - **What "pre-event" means.** Pre-event statistics in chart 3 are the
    30 minutes (6 × 5-min bars) immediately before the post timestamp,
    averaged across events. The "initiated" subset in chart 3 filters to
    events where the absolute pre-CAR z-score (computed on a 24-bar / 2-hour
    lookback against a 250-bar volatility estimate) is below 1.5 — i.e.,
    events where USO wasn't already moving for unrelated reasons before the
    post fired.
  - **Post timestamps near 5-min bar boundaries.** Anchors are floored to the
    nearest 5-min bar timestamp at or before the post. For posts a few
    seconds into a 5-min bar, the t=0 anchor close already reflects the
    full preceding bar — i.e., a small amount of post-event price action
    leaks into the anchor. This is a well-known artifact of bar-aligned
    event studies and matches the underlying analytical pipeline.

---

## Source files

Everything was generated from this repo's existing outputs:
  - `data/raw/posts_60d.parquet` — post archive with topic tags
  - `data/raw/minute_bars_5m/USO.parquet` — 5-min OHLCV
  - `data/results/pnl_concentration_bursts.csv` — 15 collapsed bursts
  - `data/results/signal_overlay_events.csv` — 81 triggered events
  - `data/results/orderflow_event_study_5m.json` — heatmap source
  - `data/results/finding4_oil_complex.json` — XOM/CVX extension
  - `data/results/posting_patterns.json` — sanity-checked post counts

Build script: `outputs/build/build_matt_charts.py`. Re-runnable; deterministic.

Sing out if you want any of these reshaped.

Cheers,
Tim
