# Independent Replication Audit — *Trump, Truth Social, and the Market*

**Auditor:** Claude (Anthropic), commissioned by Timothy
**Scope:** Independent verification of the analysis pipeline behind `report_truth_social_markets.md` / `.docx` / `.pdf`
**Date:** 2026-04-22
**Posture:** Ruthless. Disconfirming evidence weighted more than confirming. Bugs are not silently patched; judgment calls are flagged.

---

## 0. TL;DR — verdict per headline finding

| # | Claim | Verdict | One-line rationale |
|---|---|---|---|
| 1 | XOM/CVX pre-event OFI_bvc on energy_oil | **Cannot directly verify; XLE substitute SUPPORTS** | XOM/CVX bar data not in project folder; the producing script for `finding4_oil_complex.json` is not in the project either. XLE (the next-strongest cell) reproduces from raw bars and survives a +24h time-shift placebo. |
| 2 | Pre-event VPIN-z elevated on five (topic × asset) cells | **PARTIAL — direction reproduces, but VPIN-z carries an intraday seasonality** | Real-event vpin_z reproduces byte-for-byte. But under a +24h time-shift, XLE vpin_z stays strongly negative (-0.74 vs real -0.59), so a meaningful share of the "damped VPIN" signal is a time-of-day artifact, not a post-driven effect. |
| 3 | $159.88M dollar upper bound (triggered_vpin strategy) | **STRUCTURALLY FRAGILE** | Reproduced to the dollar (n=81, $159,881,805). Single largest event = **20.0%** of total (above 15% fragility threshold). Four near-duplicate posts at 2026-04-07 20:13 quadruple-count the same trade window for $36M of headline P&L. Median P&L = $0. |
| 4 | VIX CAR on initiated posts by topic | **NOT INDEPENDENTLY RE-RUN** | Hourly-bar event-study script (`market_deep_analysis_v2.py`) carries the same look-ahead pattern as `dollar_upper_bound.py` — see Phase 1 §B. Saved JSON values match the report; the underlying convention is the open question, not the arithmetic. |
| 5 | Equity-index CAR (post hours +1 to +4) | **NOT INDEPENDENTLY RE-RUN** | Same caveat as #4. |
| 6 | DJT hour +3 response (+0.28%, t=3.02, p=0.003, n=164) | **ARITHMETIC CONFIRMED, look-ahead caveat applies** | `data/market_deep_results_v2.json` confirmed: extended_window n=164, hour_means[8]=+0.28%, t=3.022. Convention is hourly bars with the same `searchsorted` snap as #4. |

**Aggregate:** The strongest claim in the report — XLE OFI_bvc on energy-oil-themed posts — survives the strictest test we could run with the data on hand (a +24h time-shift falsification). The marquee dollar figure is real arithmetic but is *not* a robust population statistic. The VPIN-z half of the order-flow story is partly a time-of-day artifact.

---

## 1. Reproducibility audit (read first)

### 1.1 What is and isn't in the project folder

| Resource | Present in project? | Implication |
|---|---|---|
| `data/truth_archive.json` (32,433 posts, full archive) | ✓ | Source posts intact. |
| `data/posts_60d.parquet` (1,341 overlap-window posts with topic flags) | ✓ | Reproduced exactly from the archive + topic builder. |
| `data/minute_bars/5m/` and `data/signals/5m/` | ✓ — but only **10 ETFs**: DJT, GLD, QQQ, SPY, USO, UUP, VXX, XLE, XLF, XLK | Sufficient for sector ETFs only. |
| `data/minute_bars/5m/{XOM,CVX,TSLA,NVDA,MSFT,AAPL,META,JPM,GS}.parquet` | ✗ | The headline finding 4 (XOM/CVX) cannot be re-executed on the data shipped with the project. |
| `collect_new_tickers.py` (would have fetched the 9 names above) | ✓ — but writes to `/sessions/sleepy-gifted-ptolemy/work/data/` | A different sandbox path than the project root. The generated parquets never landed in the project folder. |
| `orderflow_event_study.py` | ✓ | Defines 19 assets in `CFG_5M`; the saved `orderflow_event_study_5m.json` covers only 10. |
| `orderflow_placebo_and_sensitivity.py` | ✓ | Same 10-asset coverage in saved JSON. |
| `dollar_upper_bound.py` | ✓ | Computes the `all/initiated/initiated+buying` partitions only — **not** the `triggered_vpinz_gt_0_5` partition that gives the $159.88M headline. |
| `data/dollar_upper_bound_strategies.json` (the file with the $160M figure) | ✓ | But **no script in the project produces this file.** |
| `data/finding4_oil_complex.json` (XOM/CVX statistics) | ✓ | But **no script in the project produces this file.** |
| Hourly bar data (`data/intraday_hourly_prices.csv`) and `market_deep_analysis_v2.py` | ✓ | Powers the VIX/DJT/index findings. |

### 1.2 Reproducibility verdict

The project as shipped is **not end-to-end reproducible** for the marquee claims. Specifically:

- **No script in the repo generates `finding4_oil_complex.json` or `dollar_upper_bound_strategies.json`.** I could nonetheless reconstruct the $159.88M figure exactly (Phase 2c) from first principles, suggesting the missing script is a thin wrapper. But for a result that is the centerpiece of the report, "I reverse-engineered it and it matches" is below the gold standard.
- **No bar data on disk for XOM, CVX, TSLA, NVDA, MSFT, AAPL, META, JPM, GS.** A reader cannot rerun finding 4 on those names without first re-fetching minute bars from yfinance.
- **The 19-asset version of the orderflow event study saved JSON is missing** — only the 10-ETF version was committed.

To meet the "gold standard" for an audit-able financial-claims report, the project should include (a) every fetch script's output landed in the same `data/` tree, (b) every JSON in the report produced by a checked-in script, and (c) a `make` / shell entry point that walks from `truth_archive.json` to every figure in the report.

---

## 2. Phase 1 — code audit

I read every script that contributes to the headline numbers and looked for off-by-one errors, look-ahead leakage, sample contamination, and mathematically incorrect estimators.

### 2.1 `build_signals.py` — clean

- BVC formula uses `Φ(dP_t / σ_t)` with `σ_t = std(dP).shift(1).rolling(250)` — the `shift(1)` correctly excludes the current bar from σ. **No look-ahead.**
- `signed_vol_tick` uses `sign(close_t − close_{t-1})` — strict pre.
- `vol_z`, `dvol_z`, `vpin_z`, `kyle_z` all use `rolling_shifted_mean/std` (with `shift(1)`). **No look-ahead.**
- **Judgement-call deviation from textbook VPIN:** `vpin_50` uses a 50-**bar** rolling sum (`rolling(VPIN_N).sum()`), not 50 volume **buckets** of equal $ size as in Easley/López de Prado/O'Hara. This is a common implementation shortcut but is *not* "VPIN" in the strict sense. It should be labelled "time-bucketed VPIN" (or "VPIN-50bar") in any external description.

### 2.2 `orderflow_event_study.py` — clean

- `nearest_bar_index` uses `bars.searchsorted(ts, side="right") - 1`, i.e. the last bar STRICTLY at or before the post timestamp. Correct snap.
- `pre_mask = (offsets >= -pre_bars) & (offsets < 0)` — pre window is **strictly before** the post-bar (offsets `-pre_bars … -1`).
- `post_mask = (offsets > 0) & (offsets <= post_bars)` — post window is **strictly after** the post-bar (offsets `+1 … +post_bars`).
- The post-containing bar (offset 0) is excluded from both windows. **No look-ahead at the event boundary.**

### 2.3 `orderflow_placebo_and_sensitivity.py` — biased placebo sampler

- The placebo loop iterates `real_hw.iterrows()` and breaks when `len(placebos) >= n_placebo`. This **truncates** the placebo population at the start of the real-event timeline rather than sampling uniformly across it.
- The placebo pool does **not exclude the actual real-event timestamps** (or a ±X-min buffer around them) before drawing. Real-event minute-bars can therefore appear in the placebo pool and contaminate the null.
- Both bugs bias the placebo means slightly toward the real-event distribution, which **deflates** the apparent delta-vs-placebo. So this bug works *against* the report's headline; fixing it would generally make finding 4 stronger, not weaker. Still a bug.

### 2.4 `dollar_upper_bound.py` — look-ahead at the event boundary

```python
pos = idx.searchsorted(ts)         # default side="left" → first bar AT-OR-AFTER ts
pre_slice  = slice(pos - PRE_BARS, pos)
entry_price = close.iloc[pos - 1]
```

If a post lands mid-bar (e.g., 14:32:18 in a 14:30-labeled 5-min bar), `searchsorted(side="left")` returns `pos` = 14:35-bar (first AT-OR-AFTER), and `pos - 1` is the 14:30-labeled bar — **the bar that contains the post itself**, including all trading from 14:32:18 to 14:34:59 (after the post). The pre-window therefore overlaps the post-window for every mid-bar post, and `entry_price` is the close of a bar that has already absorbed the post.

Practical magnitude (Phase 2c, Convention B fix): switching to `side="right"-1` and entry = close.iloc[pos] yields **$171.4M instead of $159.88M** on the same trigger and dataset, with n=80 instead of 81. The fix moves the headline ~$11M *higher*, so this look-ahead bug is **not** the source of the report's claim. But it does change which events qualify and the exact P&L of each.

### 2.5 `market_deep_analysis_v2.py` — same pattern

`event_study()` at L148 uses `pos = int(dt_series.searchsorted(t))` (default left) on hourly bars labeled at the start of the hour (13:30, 14:30, 15:30 ET). For a post at 14:00, `pos = 1` (the 14:30 bar) and the pre-window includes the 13:30-labeled bar, **which contains the post timestamp**. Effects similar to §2.4 apply but at the hourly granularity (much larger windows so the relative contamination is smaller, but real). **DJT hour +3 result, VIX CAR, and equity-index CAR all share this convention.**

### 2.6 `add_friends_topics.py` & `friends_topic_counts.json` — internally honest

The friends-vs-self decomposition cannot be done on the overlap window:

- topic_musk_tesla: 0 events in overlap (116 in full archive)
- topic_big_tech: 1 event (43 full)
- topic_big_oil_companies: 0 events (2 full)
- topic_big_banks: 2 events (12 full)

`friends_topic_counts.json` says this explicitly: *"Counts are far below n≥5 event-study threshold. Friends-vs-self decomposition not feasible on this window."* That section of the report should not be over-stated.

---

## 3. Phase 2 — three sanity checks

### 3.1 Phase 2a — null-window check (500 random RTH timestamps, ±60 min from real posts excluded)

Comparison of pre-event signal means under random null timestamps vs real oil-themed posts:

| Asset | Signal | Null-window mean | Real-event mean | Pass? |
|---|---|---|---|---|
| USO | OFI_bvc | +0.014 (p=0.06) | -0.022 (p=0.024) | ✓ Real distinguishable from null |
| USO | vpin_z | +0.054 (p=0.34) | +0.380 (p=1.5e-06) | ✓ |
| XLE | OFI_bvc | +0.019 (p=0.017) | +0.048 (p=1.7e-12) | ✓ Real >> null |
| SPY | OFI_bvc | small (≈0) | small (≈0) | n/a (no claimed effect on SPY for oil-themed posts) |

**Verdict:** PASS. The pipeline is not just measuring intraday noise — real-post statistics differ from random-RTH timestamps for the cells where finding 4 claims a signal.

The mismatch between my null-window XLE OFI_bvc (+0.019) and the saved finding4 placebo mean (+0.011) is within sampling variation but flags the seed-dependence revealed in Phase 3.

### 3.2 Phase 2b — +24h time-shift placebo

Take every real oil-themed post, shift its `created_at` by +24 hours, re-run the pre-event event study with the same initiated filter. Expectation if the headline is causal: **the effect should disappear.**

| Asset | Signal | Real-event mean | Shifted-event mean | Verdict |
|---|---|---|---|---|
| **XLE** | **OFI_bvc** | **+0.0488** (p=0.000) | **-0.0014** (p=0.750) | ✅ **Strong PASS — effect cleanly disappears** |
| USO | vpin_z | +0.351 (p=0.000) | -1.022 (p=0.000) | ✅ Sign flips → real signal not a daily artifact |
| USO | OFI_bvc | -0.014 (p=0.182) | -0.022 (p=0.000) | ⚠️ Marginal — both negative; USO OFI_bvc was not in finding4 survivors anyway |
| **XLE** | **vpin_z** | **-0.593** (p=0.000) | **-0.744** (p=0.000) | ❌ **FAIL — effect persists at same intraday clock time +24h** |
| USO | signed_vol_tick | -13,035 (p=0.036) | -47,995 (p=0.000) | ⚠️ Negative imbalance partially preserved → contamination by intraday seasonality |

**Verdict:** Mixed.

- The marquee XLE OFI_bvc result is **post-driven, not seasonality-driven**. The +24h-shifted XLE OFI_bvc is statistically indistinguishable from zero (p=0.75). This is the cleanest single piece of evidence for a real causal effect in the entire report.
- The XLE vpin_z "damped VPIN ahead of oil-themed posts" claim does **not** survive — the same negative VPIN-z appears 24 hours later at the same clock time. This signal is at least partly a time-of-day artifact of when oil-themed posts cluster (early-mid US session, when oil-ETF VPIN is structurally below its 250-bar baseline).
- The USO sv_tick result was already marginal (q=0.046) and is partially preserved under shift; treat as weak.

### 3.3 Phase 2c — leave-one-out on the $159.88M dollar upper bound

I had to reconstruct the producing script (no `triggered_vpinz_gt_0_5` script in repo). I mirrored `dollar_upper_bound.py` exactly:

- `PRE_BARS = POST_BARS = 12` (60 min)
- `pos = idx.searchsorted(ts)` (default left — same look-ahead pattern as §2.4)
- `pnl = pre_sv * (exit − entry)`
- Trigger: pre-window `mean(vpin_z) > 0.5`

**Reconstruction matches the headline to the dollar:**

| Quantity | Reproduced | Saved (`dollar_upper_bound_strategies.json`) |
|---|---|---|
| n | 81 | 81 |
| sum P&L (USD) | $159,881,805 | $159,881,804.59 |
| mean | $1,973,849 | $1,973,849 |
| median | $0 | $0.00 |
| hit rate | 45.7% | 45.68% |

**Leave-one-out fragility analysis:**

- LOO sum range: **$150.8M (worst case excluded) to $191.8M (best case excluded)** → ±13% swing from removing one event.
- **Maximum single-event share = 20.0%** (above the 15% fragility threshold I set).
- The 20% event is a single post on 2026-03-23 14:29:33 with `pre_sv = +10.3M shares` and a -$31.9M P&L (price moved against the imbalance).
- **Four near-duplicate posts on 2026-04-07 within 31 seconds** (20:13:09 / :23 / :32 / :40) each contribute identical $9.1M P&L because they share the same pre-window, sv, and post-window. Total quadruple-counted: **$36.4M = 22.8% of headline**. The pipeline does **not** dedupe near-simultaneous posts.

**Look-ahead-corrected version (Convention B in `phase2c_loo.py`):** Switching to strict-pre snap raises the headline to **$171.4M (n=80)** — so the look-ahead at the event boundary is *not* what produces the $160M figure. But the duplicate-counting and outlier concentration are real and not addressed in the report.

**Verdict:** The arithmetic is correct, but the figure is not a robust population statistic:

- After removing the single 20% outlier: $128.0M (with same 81-trigger filter would actually require recomputing the trigger; LOO single-event removal is an upper-bound check).
- After collapsing the 4 duplicate posts to 1: ~$133M.
- After both: ~$96M.
- Median per-event P&L = $0 (already published in JSON; underweighted in the report's narrative).

A $96–160M range over a single, non-robust pruning is itself the headline. The "$160M" should not be quoted without those caveats.

---

## 4. Phase 3 — XLE OFI_bvc rerun from scratch with new seed

XOM data is not on disk. I substituted XLE (the strongest cell in `finding4_oil_complex.json`: q=7.31e-11) and re-derived signals **independently** of `build_signals.py` from raw 5-min bars, using the same BVC formula family but written from first principles in `phase3_rerun.py`.

### 4.1 Byte-level math check

| Signal | Max |saved − rerun| | Median |saved − rerun| | n bars compared |
|---|---|---|---|
| OFI_bvc | 0.000e+00 | 0.000e+00 | 10,737 |
| vpin_z | 0.000e+00 | 0.000e+00 | 7,399 |

`build_signals.py` is mathematically correct — every BVC and VPIN-z value reproduces to floating-point identity.

### 4.2 Real-event reproduction

| Statistic | Rerun | Saved (finding4) | abs diff |
|---|---|---|---|
| XLE OFI_bvc real mean (initiated, n=166 vs saved 165) | +0.04878 | +0.04834 | 0.00044 |
| XLE vpin_z real mean | -0.59265 | -0.59375 | 0.00110 |

The sample-size difference (166 vs 165) is from a single edge-case event being included or excluded under marginally different bar-coverage filters — unimportant.

### 4.3 Matched-placebo with NEW seed (12345)

| Statistic | Rerun (seed 12345) | Saved | abs diff |
|---|---|---|---|
| XLE OFI_bvc placebo mean | -0.00921 | +0.01115 | 0.020 |
| XLE vpin_z placebo mean | +0.33696 | -0.02639 | 0.364 |
| XLE OFI_bvc delta vs placebo | +0.05799 | +0.03719 | 0.021 |
| XLE vpin_z delta vs placebo | -0.92960 | -0.56735 | 0.362 |

**Effect direction is robust**, but the **exact placebo and delta values are highly seed-sensitive**. The vpin_z placebo even flipped sign. This means:

- The reported FDR-corrected q-value of `7.31e-11` is the q-value computed against *one specific* placebo realization — it should not be quoted as if the placebo distribution itself were the population.
- The headline finding is qualitatively robust (real OFI_bvc is much higher than any placebo realization I've drawn), but a more honest report would quote a *range* of delta-vs-placebo over many seeds, or a Monte Carlo distribution.

### 4.4 Coverage concern

The matched-placebo pool (hour-of-day × weekday × month) had **121 of 173 events with NO matching draw** (70% skip before relaxation to `(hour, weekday)` only). The placebo strata are extremely thin in the 60-day overlap window — a structural weakness of the matched-placebo design at this sample size.

---

## 5. Aggregate verdict

### 5.1 What I believe after the audit

| Claim | After audit |
|---|---|
| There is a real, post-specific positive shift in pre-event order-flow imbalance for the XLE energy-sector ETF on Trump posts about oil/Iran/energy | **Yes — survives all three sanity checks I could run.** |
| There is a real, post-specific damping of VPIN-z on oil ETFs ahead of oil-themed posts | **Partially — XLE pattern is contaminated by intraday seasonality; USO pattern survives.** |
| XOM/CVX specifically show pre-event informed-flow imbalance | **Indirectly supported via XLE survivor; not directly verifiable with project data.** Saved JSON values reconstruct cleanly given that XOM/CVX bar parquets exist somewhere. |
| The dollar magnitude is "around $160M" | **Arithmetic is exact; population statistic is fragile.** Honest range: ~$96M–$192M depending on how you treat duplicate posts and outliers. |
| The friends-vs-self decomposition supports an insider-trading story | **Cannot be tested on this window — friends-topic counts are 0–2.** This section of the report should be softened. |
| DJT hour +3 / VIX / index CARs from `market_deep_analysis_v2.py` | **Saved JSONs match the report.** Look-ahead at the event boundary is the same kind as §2.4; magnitude smaller because hourly bars; not independently re-run here. |

### 5.2 What I cannot verify

- XOM and CVX-specific results (no bar data in project).
- The unsaved producing scripts for `finding4_oil_complex.json` and `dollar_upper_bound_strategies.json`. I reconstructed the latter to the dollar; the former is consistent with the saved per-asset numbers I can rerun on XLE/USO.
- The full 1140-test BH-FDR table over the 19-asset universe (only 10-asset version on disk).
- The exact placebo realization used in the saved q-values (placebo is seed-sensitive — Phase 3.3).

### 5.3 What changes the headline if corrected

Issue | Direction of correction
---|---
Duplicate posts (4 at 2026-04-07 within 31 sec) | **Reduces** $160M figure by ~$27M (16.9%)
Single 20% outlier (2026-03-23) | LOO swing of ±$32M
Look-ahead in `dollar_upper_bound.py` at event boundary | **Increases** $160M figure to $171M (the bug doesn't favor the headline)
Look-ahead in `market_deep_analysis_v2.py` (hourly bars) | Direction unknown without re-running; likely small at hourly granularity
Placebo seed | Doesn't change effect direction; can move delta-vs-placebo by ±50%
XLE vpin_z time-of-day artifact | Removes one of the 5 "elevated VPIN" cells; doesn't affect OFI_bvc story

---

## 6. Gold-standard requirements for a publishable version

If this report is to be submitted somewhere with audit standards (academic journal, financial regulator, serious press desk), the following are non-negotiable:

1. **Every figure in the report must be produced by a checked-in script in the repo.** Currently the two most-quoted JSONs (`finding4_oil_complex.json` and `dollar_upper_bound_strategies.json`) have no producing script.
2. **All bar data referenced (XOM, CVX, TSLA, NVDA, MSFT, AAPL, META, JPM, GS) must be saved in `data/minute_bars/5m/` next to the ETF data.** Currently the `collect_new_tickers.py` script writes to a different sandbox path.
3. **A `make all` (or shell entry point) must walk from `truth_archive.json` and yfinance fetches to every figure in the report.**
4. **Dedupe near-simultaneous posts.** Posts within (e.g.) 15 minutes of each other on overlapping windows must be collapsed. The duplicate-counting in §2c is a *systematic* over-count of P&L for any active-posting day.
5. **Quote dollar figures as bootstrap-CI ranges, not point estimates.** E.g., "$96M–$192M" rather than "$160M".
6. **Quote q-values as Monte Carlo distributions over placebo seeds**, not single-seed point estimates. Fold the placebo seed into a meta-bootstrap.
7. **Fix the look-ahead at the event boundary in `dollar_upper_bound.py` and `market_deep_analysis_v2.py`** (use `searchsorted(side="right") - 1` consistently, as `orderflow_event_study.py` already does).
8. **Re-label the "VPIN" series as "VPIN-50bar" or "time-bucketed VPIN"** to be honest about the deviation from textbook Easley-LdP-O'Hara (which uses equal-volume buckets, not time bars).
9. **Run the placebo-sampler dedup/exclusion fix from §2.3.** This works *against* the headline, so adopting it builds credibility.
10. **Soften the friends-vs-self framing.** With 0–2 events on the friends topics in the overlap window, no causal claim about Musk/Tesla, big tech, big oil, or big banks can be supported. The section should be presented as a **null result on the available window**, not a finding.

---

## 7. Files produced by this audit

All in `/sessions/blissful-serene-gates/replication/` (working directory; not in the project folder per the audit's read-only mandate, except this report):

| File | Contents |
|---|---|
| `posts_60d.parquet` | Reproduced 1,341-post overlap-window dataset matching production topic flags. |
| `phase2a_results.json` | Null-window sanity check — 500 random RTH timestamps. |
| `phase2b_timeshift.json` | +24h time-shift placebo on USO + XLE. |
| `phase2b_timeshift.py` | Source for above. |
| `phase2c_loo.json` | Leave-one-out + reconstruction of $159.88M figure. |
| `phase2c_loo.py` | Source — reconstructs both buggy ("Production") and corrected ("Strict-pre") conventions. |
| `phase3_rerun.json` | Independent BVC+OFI+vpin_z derivation on XLE; new placebo seed. |
| `phase3_rerun.py` | Source — written from first principles, not by editing `build_signals.py`. |

---

## 8. One paragraph for the editor

The XLE OFI_bvc finding is real and survives the toughest test the available data permits — a +24h time-shift placebo cleanly nulls the effect (Δ=-0.001, p=0.75), while the real-event mean is +0.049. The dollar upper bound is arithmetically reproducible to the cent but is not a robust population statistic: removing one event swings it by ±$32M and four near-duplicate posts inflate it by ~$27M. The VPIN-z half of the story is partially a time-of-day artifact on XLE (passes on USO). XOM/CVX specifically cannot be directly verified because the project does not ship bar data for those tickers. The report's headline is *defensible* at the level of "informed flow shifts ahead of oil-themed Trump posts on the energy-sector ETF" but is *over-confident* at the level of "this paid out $160M to a perfect-information trader" and "VPIN-z is elevated on five specific cells." Tightening the headline to what the data can support would, in my judgment, make the report stronger, not weaker.
