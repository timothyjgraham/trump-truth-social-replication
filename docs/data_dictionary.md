# Data dictionary

Every parquet file in this repository, with every column.

## `data/raw/truth_archive.json`

Newline-delimited JSON of every public status from `@realDonaldTrump`
returned by the Truth Social Mastodon-compatible API at the time of
collection (2026-04-22). 32,433 posts.

Each record is the full Mastodon `Status` object. Top-level fields used
downstream: `id`, `created_at`, `content`, `account`, `url`,
`replies_count`, `reblogs_count`, `favourites_count`.

`data/raw/truth_archive.csv` is the same posts flattened to seven
columns: `id, created_at, content_text, url, replies_count,
reblogs_count, favourites_count`.

## `data/raw/posts_60d.parquet`

The 60-day study window subset, with topic tags joined on. 1,847 rows.

| column | type | description |
| --- | --- | --- |
| `id` | int64 | Truth Social post ID |
| `created_at` | datetime64[ns, UTC] | Post timestamp |
| `content_text` | string | Plain-text body, HTML stripped, ftfy-cleaned |
| `url` | string | Canonical post URL |
| `replies_count` | int64 | Reply count at collection time |
| `reblogs_count` | int64 | Re-truth count |
| `favourites_count` | int64 | Favourite count |
| `topic_energy_oil` | bool | Matches the oil regex set (165 True) |
| `topic_china_trade` | bool | Matches the China-trade regex set |
| `topic_fed_rates` | bool | Matches the Fed/rates regex set |
| `topic_russia_ukraine` | bool | Matches the Russia/Ukraine regex set |
| `topic_election` | bool | Matches the election regex set |
| `topic_immigration` | bool | Matches the immigration regex set |

The full regex set is in `code/01_scrape_truth_social.py`.

## `data/raw/minute_bars_5m/<TICKER>.parquet`

5-minute OHLCV bars from `yfinance`. Index is UTC datetime.

| column | type | description |
| --- | --- | --- |
| `Open` | float64 | Bar open |
| `High` | float64 | Bar high |
| `Low` | float64 | Bar low |
| `Close` | float64 | Bar close |
| `Volume` | float64 | Total bar volume (shares) |
| `Adj Close` | float64 | Dividend/split-adjusted close |

Tickers shipped: DJT, SPY, QQQ, XLE, USO, GLD, UUP, VXX, XLF, XLK.

## `data/interim/signals_5m/<TICKER>.parquet`

Microstructure signals derived from the corresponding minute bars file.
Index is UTC datetime, aligned bar-for-bar with `minute_bars_5m/`.

| column | type | description |
| --- | --- | --- |
| `logret` | float64 | log(Close_t / Close_{t-1}) |
| `vol` | float64 | Bar volume |
| `vol_z` | float64 | Volume z-scored on a 250-bar shifted baseline |
| `dvol` | float64 | First difference of volume |
| `dvol_z` | float64 | dvol z-scored on 250-bar shifted baseline |
| `buy_share` | float64 | BVC buy share, Φ(r/σ) |
| `buy_vol` | float64 | buy_share × vol |
| `sell_vol` | float64 | (1 − buy_share) × vol |
| `signed_vol_tick` | float64 | buy_vol − sell_vol (shares) |
| `OFI_bvc` | float64 | signed_vol_tick / vol, ∈ [−1, +1] |
| `vpin` | float64 | Raw VPIN per bar |
| `vpin_z` | float64 | VPIN z-scored on 250-bar shifted baseline |
| `kyle_lambda` | float64 | Absolute Kyle's λ on 50-bar window |
| `kyle_z` | float64 | kyle_lambda z-scored on 250-bar shifted baseline |

All `_z` columns are constructed with a one-bar shift on the rolling
baseline so they contain no contemporaneous-or-future information.

## `data/results/orderflow_event_study_5m.json`

Output of stage 04. Nested dict keyed by `topic → asset → {n, signals,
initiated, non_initiated}`. Each `signals` block has per-metric
`pre_mean, pre_p, pre_boot_p, post_mean, post_p, post_boot_p`.

## `data/results/orderflow_placebo_5m.json`

Output of stage 05. Same shape as the event-study JSON but built from
5,000 hour-and-weekday-matched placebo timestamps.

## `data/results/orderflow_sensitivity_5m.json`

Window-sensitivity sweep from stage 05. Same metrics computed with
PRE = POST ∈ {3, 6, 12} bars.

## `data/results/orderflow_final_5m.{csv,json}`

Output of stage 06. Consolidated table with one row per
(topic, asset, metric, window) combination:

| column | description |
| --- | --- |
| `source` | always `"real"` in this table |
| `topic` | one of `topic_*` from `posts_60d.parquet` |
| `asset` | ticker |
| `metric` | one of the seven signal columns |
| `window` | `pre` or `post` |
| `mean` | real-event mean |
| `p` | bootstrap p on real events |
| `n` | sample size |
| `p_fdr` | BH-adjusted p across the full table |
| `placebo_mean` | placebo mean for the same (asset, metric, window) |
| `placebo_p` | bootstrap p on the placebo distribution |
| `placebo_n` | placebo sample size |
| `diff_vs_placebo` | mean − placebo_mean |
| `real_sig` | bool, p < 0.05 |
| `placebo_sig` | bool, placebo_p < 0.05 |
| `robust` | real_sig & ~placebo_sig & p_fdr < 0.10 |

## `data/results/dollar_upper_bound_strategies.json`

Output of stage 08. Four named slices on USO, each reporting `n,
sum_pnl_usd, mean_pnl_usd, median_pnl_usd, hit_rate_pct, p25_pnl_usd,
p75_pnl_usd`. The `triggered_vpinz_gt_0_5` slice is the report's
headline.

## `data/results/dollar_upper_bound_uso_events.csv`

Per-event detail underlying the dollar-bound JSON. One row per oil-themed
post with a usable window:

| column | description |
| --- | --- |
| `post_id` | Truth Social ID |
| `ts` | Post timestamp |
| `pre_sv` | Sum of `signed_vol_tick` in the pre-window (shares) |
| `pre_vpinz` | Mean of `vpin_z` in the pre-window |
| `entry_price` | Close at bar (p − 1) |
| `exit_price` | Close at bar (p + POST_BARS − 1) |
| `pnl` | pre_sv × (exit_price − entry_price) |
| `pre_car_z` | Pre-window cumulative-return z-score |
| `initiated` | bool, |pre_car_z| < 1.5 |

## `data/results/phase2b_timeshift.json`

Output of stage 07. For each of USO and XLE, two blocks
(`real_reproduced` and `shifted`), each with per-metric
`{n, pre_mean, boot_p_two_sided, boot_ci95}`.

## `data/results/phase2c_loo.json`

Output of stage 09. For each convention (`A_production`, `B_strict_pre`),
a block with `n, total_sum_pnl_usd, mean_pnl_usd, median_pnl_usd,
hit_rate_pct, loo_sum_min_usd, loo_sum_max_usd, loo_sum_range_usd,
max_single_event_share_pct, top5_events`.

## `data/results/finding4_oil_complex.json`

Pre-existing intermediate from the original analysis. Kept for
provenance; not used by any pipeline stage in this repository.
