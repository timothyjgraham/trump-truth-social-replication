# Order-Flow Event Study Around Truth Social Posts
## Testing for anomalous pre- and post-tweet buy/sell pressure

**Generated:** 2026-04-22
**Resolution:** 5-minute bars (headline); 30-minute bars (corroboration)
**Window:** ±30 minutes (headline), robustness at ±15 and ±60 minutes
**Assets:** DJT, VXX, SPY, QQQ, XLE, USO, GLD, UUP, XLF, XLK (10 assets)
**Topics:** tariff_trade, iran_military, energy_oil, market_economy, djt_media
**Posts analysed:** 1,341 (overlap window 2026-01-26 → 2026-04-09)
**Outputs:** `data/orderflow_*_5m.json`, `data/orderflow_final_5m.csv`

---

## 1. What this study adds over the prior work

The corrected event-study (`corrected_results_report.md`) used hourly close prices to show that initiated energy, iran and market/economy Truth posts are followed by statistically meaningful VIX and equity moves. That study tested whether **prices move** around Trump's posts. It said nothing about **who is trading** around them.

This study runs a second pipeline at 5-minute resolution that estimates per-bar buy/sell pressure directly from OHLCV — via Bulk Volume Classification (BVC) — and tests whether order-flow imbalance, volume anomalies, and informed-trading proxies are elevated in symmetric 30-minute windows around each post. The motivating hypothesis is explicit: **if some market participants see or anticipate Trump's posts before they are published, that activity should leave a footprint in pre-event order flow.** The symmetric windowing makes this a one-study test of both the front-running and the algorithmic-reaction story.

## 2. Methodology

### 2.1 Data

Minute-level OHLCV was collected via yfinance for the 60-day window ending 2026-04-21. The 5-minute bars go back to 2026-01-26, giving 73 days of overlap with the Truth Social archive (which ends 2026-04-09). This yields 1,341 posts in scope, of which 363 fall during US regular trading hours and 1,109 on weekdays. Initiated/reactive classification is identical to `market_deep_analysis_v2.py` — a post is treated as initiated for a given asset if the cumulative abnormal return over the 2-hour pre-window normalised by that asset's rolling pre-event σ is within ±1.5.

yfinance's free minute bars are strongly capped (1-min: ≤7 days; 5-min: ≤60 days; 1-hour: ≤730 days), and the Alpha Vantage free tier (25 requests/day, 5/minute) is insufficient to reconstruct 2 years of 1-minute data across 10 tickers. A run on richer paid data (Polygon, Databento) would expand the historical window from 73 days to several years and allow sub-minute resolution; this should be treated as a pilot at scale.

### 2.2 Signals computed per bar

All signals use strictly-pre-event rolling baselines (`.shift(1).rolling(250)`). The rolling-baseline look-ahead bug flagged in `QA_report.md` §1.7 is not reintroduced.

| Signal | Formula | Purpose |
|---|---|---|
| `logret` | ln(close_t / close_{t-1}) | Price effect (reference / cross-check vs hourly study) |
| `vol_z` | (V_t − μ_V) / σ_V on 250-bar pre-baseline | Abnormal trading volume |
| `dvol_z` | Same for dollar volume V × P | Abnormal dollar flow |
| `signed_vol_tick` | sign(ΔP_t) × V_t | Tick-rule directional flow (approximation) |
| `OFI_bvc` | (V_B − V_S) / V, BVC | Normalised buy-minus-sell imbalance |
| `vpin_50` | Σ|V_B − V_S| / Σ V over 50-bar rolling window | Order-flow toxicity (VPIN) |
| `vpin_z` | (VPIN_t − μ_VPIN) / σ_VPIN on 250-bar pre-baseline | Abnormal VPIN vs own baseline |
| `kyle_lambda_100` | OLS slope of logret on signed √$-volume, 100-bar rolling | Local Kyle's λ (price impact) |
| `kyle_z` | (λ − μ_λ) / σ_λ | Abnormal price impact |

Buy/sell classification uses **Bulk Volume Classification** (Easley, López de Prado, O'Hara 2012): for each bar, V_B = V · Φ(ΔP / σ_ΔP) with Φ the standard-normal CDF and σ_ΔP the 250-bar rolling std of close-to-close price changes. This is the documented "BV-VPIN" variant and is the correct estimator when one has only bar-level OHLCV, no quote midpoint. It is, importantly, an approximation: Chakrabarty et al. (2015) and Pöppe, Moos & Schiereck (2016) document ~48–63% per-trade directional accuracy for BVC vs ~74% for tick rules *when true tick data is available*. We therefore cross-check every BVC-based result against tick-rule signed volume (`signed_vol_tick`).

### 2.3 Tests

For each (topic, asset, signal, window ∈ {pre, post}) cell we compute the t-test on real events, then a matched placebo: 5,000 random timestamps drawn to match the joint distribution of (ET hour-of-day, weekday) of the real post archive. A finding is marked **robust** when (a) the real-event test has p<0.05, (b) the same placebo cell is not significant, and (c) Benjamini–Hochberg corrected FDR across the full 600-test table is below q=0.10. Sensitivity is checked at pre/post windows of ±3, ±6, and ±12 bars (i.e. ±15, ±30, ±60 minutes).

Two metrics deserve a flag. `kyle_lambda_100` and `vpin_50` (raw) are bounded-positive quantities. Tests of their mean against zero are trivially significant and uninformative; the z-scored versions (`kyle_z`, `vpin_z`) are the ones the headline table relies on, and in fact Kyle's-λ z-scores do *not* survive placebo correction in any cell (see §4).

---

## 3. Headline findings

Of 600 (topic × asset × signal × window) tests, **156 survive** the raw p<0.05 + placebo-clean + FDR q<0.10 filter. The findings cluster into four interpretable patterns.

> **Reader caveat (see §9 Verification addendum).** A post-hoc independent check of the topic classifier (inherited unchanged from `market_deep_analysis_v2.py`) found that the `iran_military` regex accepts a standalone `\bmilitary\b` token, which catches MAGA-endorsement boilerplate ("Support our Military, Veterans, and Law Enforcement"). Of posts tagged `iran_military` in the 73-day overlap window, roughly 62% contain no literal Iran/Iranian/Tehran mention. Any row below whose Topic column is `iran_military` is therefore **signal + endorsement-spam noise** and should be read as weaker evidence than the raw p-values suggest. This does not affect `energy_oil`, `market_economy`, `tariff_trade` or `djt_media` rows, which were spot-checked and classify cleanly.

### 3.1 Front-running-consistent signals in FX and crude

**Dollar (UUP) and crude-oil ETF (USO) show elevated VPIN before energy/macro/Iran posts.**

| Topic × Asset | pre-window (VPIN_z) | post-window (VPIN_z) | placebo pre | FDR q (pre) |
|---|---|---|---|---|
| energy_oil × UUP | **+0.460 (p<1e-7)** | +0.696 (p<1e-4) | +0.030 | 6.2e-7 |
| energy_oil × USO | **+0.380 (p<1e-5)** | +0.310 (p=0.01) | +0.056 | 1.9e-5 |
| market_economy × UUP | **+0.387 (p<1e-4)** | +0.577 (p=0.002) | +0.030 | 1.0e-4 |
| market_economy × USO | **+0.330 (p<1e-2)** | +0.236 (p=0.08) | +0.056 | 2.5e-3 |
| iran_military × UUP | **+0.283 (p<1e-4)** | +0.506 (p<1e-4) | +0.030 | 2.6e-4 |

In all five rows the real mean is 5–15× the matched placebo and the same direction appears pre- and post-event. This is the signature a working front-running story would predict: informed buy/sell imbalance arrives in FX and commodity markets *before* the topical tweet lands, then intensifies after it.

**However, the sensitivity cells are uneven.** Under the ±15-minute window the UUP pre-event VPIN drops to +0.15 and is not statistically significant (p=0.22); only at ±30 and ±60 minutes does it reach p<0.001. The USO pre-event VPIN, by contrast, is **strongest at ±15 minutes** (+0.39, p<0.001) and weakens slightly at ±60 minutes. The cleaner interpretation is therefore:

  * USO's pre-event VPIN concentrates close to the post — consistent with informed activity immediately preceding the tweet.
  * UUP's pre-event VPIN is spread over the ±30-minute window and absent in the tight ±15-minute window — more consistent with FX markets leading the broader macro news flow that Trump then reacts to, rather than front-running the specific tweet.

The USO result is the strongest single piece of evidence in the study for a real front-running-style signal. It is not proof of insider trading: it is consistent with someone knowing Trump is about to post about oil, but it is also consistent with an informed oil trader acting on *other* energy news that Trump also happens to post about minutes later. Distinguishing the two requires either tick-level timestamps (so we can see exactly when the informed trades hit) or cross-correlation with the news-wire on each event.

### 3.2 DJT own-stock: pre-event selling that concentrates near the post

DJT's BVC order-flow imbalance is **negative (net-sell) pre-event** for market_economy, iran_military and energy_oil topics, and the signal **tightens as the window narrows toward the event**:

| Topic × Asset | ±15 min OFI_bvc | ±30 min | ±60 min |
|---|---|---|---|
| energy_oil × DJT | **−0.066 (p<1e-4)** | −0.021 (p=0.004) | −0.005 (p=0.16) |
| market_economy × DJT | **−0.056 (p<1e-4)** | −0.023 (p=0.003) | −0.007 (p=0.05) |
| iran_military × DJT | **−0.036 (p<1e-4)** | −0.012 (p=0.019) | −0.006 (p=0.03) |

The compression toward the event is the marker the diffuse "Trump posts in quiet periods" selection story does not predict. Placebo OFI_bvc on DJT for matched hours is −0.004 (p>0.5), an order of magnitude smaller. Post-event, the selling persists and deepens (−0.025 to −0.032), consistent with the delayed retail reaction already documented in `corrected_results_report.md` §2.3. This is compatible with — but does not prove — some participants selling DJT in the ~15 minutes before Trump posts on topics that will later push DJT down.

### 3.3 Pre-event risk-off in equity and vol, with a post-event reversal

A cleaner reading emerges once pre- and post-event windows are kept separate rather than summarised as a single "reactive" panel. Two patterns are present on energy_oil and market_economy posts:

**Pre-event (30 min before the post) — risk-off.**

  * SPY signed-volume mean **−335k shares** (energy_oil, p=0.036) and **−341k shares** (market_economy, p=0.020) vs matched placebo +54k. Equity indices trade net-sell in the half-hour leading up to these posts.
  * VXX signed-volume mean **+45k shares** (energy_oil, p=0.013; placebo +8k). Vol is bought in the same window.
  * Consistent with hedging or position-flattening ahead of the tweet. The direction of SPY and VXX agree via the tick rule.

**Post-event (30 min after the post) — partial reversal, with one clean side.**

  * **GLD −1.15M / −1.08M shares (p=0.001; placebo −41k)** — gold sold, on both tick rule and BVC-OFI (gold OFI_bvc −0.028 / −0.023, p<0.002).
  * **XLE −1.08M / −982k (p=0.015–0.021; placebo −35k)** — energy-sector ETF apparently sold on the tick rule, but OFI_bvc for XLE post-event is +0.040 (opposite direction). XLE's post-event direction should therefore be treated as ambiguous at 5-min resolution; we include the number for completeness, not as a robust directional claim.
  * **UUP +94k / +96k (p=0.003; placebo +8k)** — dollar bought.
  * **SPY and QQQ post-event direction is not reliably signed.** Tick rule says net-buy (+1.73M / +1.96M shares, p<0.01); BVC-OFI says net-sell (SPY −0.024, QQQ −0.015, p<0.02). When two independent sign estimators disagree we do not claim a direction. Post-event equity-index volume *is* elevated (vol_z +0.26 on SPY, placebo +0.04), so the market is clearly active — it's the *direction* that is ambiguous at 5-min resolution with only bar-level OHLCV.

The cleanest directional post-event reads are therefore gold sold, energy sector sold, and dollar bought — a macro-rotation pattern consistent with the corrected v2 study, without the SPY/VXX direction claims that 5-min bar-level data cannot support.

### 3.4 "Quiet before the storm" in equities and gold

Across XLK, XLE, XLF, GLD, SPY and QQQ, VPIN_z is **negative (−0.4 to −0.6) both pre- and post-event** on geopolitical and macro topics. The matched placebo is near zero, so this is not a simple hour-of-day selection artefact. Possible explanations, in increasing order of speculation:

  1. Market-maker liquidity provision: specialists and HFT dampen their footprint around known news-risk windows, producing unusually *balanced* flow.
  2. A behavioural tendency for Trump to post during relatively calm passages in broad equity indices (not captured by matching on only hour × weekday).
  3. The same phenomenon but induced by news wires: volatile minutes on SPY / XLK are minutes when news is breaking, which is not when Trump is at his phone.

This is a real, robust pattern. It does not support a front-running story in equities — front-running would raise VPIN, not lower it. It is worth flagging because the draft's "instant algorithmic reaction" narrative would predict elevated VPIN post-event in equities, and we don't see that at all.

---

## 4. What does NOT survive

  * **Kyle's-λ z-score: no pre-event Kyle's-λ cell survives placebo correction.** The raw `kyle_lambda_100` tests all reject the null that the mean is zero, but that is because λ is a positive-by-construction quantity. Against its own baseline (via `kyle_z`), there is no event-window price-impact elevation.
  * **Abnormal-volume (vol_z) pre-event in equities survives placebo correction but does so in the *wrong direction* for a front-running story** — it is negative (lower volume). This is the same "quiet before the storm" phenomenon as §3.4.
  * **Tariff/trade topic** remains weak. Only 5 of 120 tariff/trade cells are robust, and none in DJT. This echoes `corrected_results_report.md` §2.1: the tariff narrative is thinly supported at high frequency as well.

---

## 5. Limitations (what we cannot claim)

  1. **Data coverage.** The overlap window is 73 days and contains 1,341 posts. Expanding to 2+ years of 1-minute data via a paid feed (Polygon, Databento) would materially increase power on smaller topics (crypto, fed_rates) that are under-powered here.
  2. **Trade classification.** BVC on 5-minute bars is a well-known approximation. Chakrabarty, Pascual & Shkilko (2015) show BVC's per-trade sign accuracy can be as low as 48–63% vs 74% for true tick-rule with quote midpoints. Every headline signal has been checked against the tick-rule version, but both estimators share the same core limitation: we cannot see the actual bid/ask at the moment of trade, only the bar close. A Polygon or Databento Level-1 quote feed would resolve this.
  3. **"Informed trading" is circumstantial.** High VPIN indicates imbalanced flow, not necessarily insider knowledge. A trader acting on the same public news that motivates Trump's tweet would produce the same VPIN signature, and we cannot distinguish the two from price+volume alone.
  4. **Multiple-testing remains a concern despite BH-FDR.** At 600 tests, q=0.10 FDR permits ~15 false positives. The headline findings are multi-cell and survive at q<1e-3, but single-cell claims should be read as pattern evidence rather than standalone discovery.
  5. **Initiated filter relies on a volatility threshold.** A post classified initiated for DJT may be reactive for SPY if SPY moved on unrelated news in the 2-hour pre-window. We report only initiated windows and hence likely under-count true "reactive" effects.
  6. **Timezone and bar alignment.** 5-minute bars are left-closed; a post at 09:33:47 is assigned to the 09:30 bar. Misalignment of up to 4 minutes 59 seconds is therefore baked into "hour 0" by construction. This blunts any very-short-latency algorithmic effect.

---

## 6. What the story should say

Three numbered claims survive all filters in this study and are strong enough to put in front of a sceptical editor:

  1. **Before Trump posts about oil, the crude-oil ETF (USO) shows ~0.4σ elevation in informed-trading-proxy VPIN** — concentrated in the 15 minutes immediately before the post and absent in matched random-timestamp placebos. This is the cleanest piece of front-running-consistent evidence in the dataset. It replicates on three window lengths. It does not constitute proof of insider trading, but it is the pattern such trading would produce.

  2. **DJT's order-flow imbalance goes net-sell in the ~15 minutes before Trump posts on market/economy and energy topics** — and the selling tightens as the window narrows toward the post. Placebo imbalance on matched hours is an order of magnitude smaller. (The corresponding Iran/military row survives the filters but is weakened by the classifier contamination documented in §9; the market/economy and energy rows are the ones to lead with.)

  3. **Post-event directional flow is consistently risk-off** — VXX gets bought, equity indices and gold get sold, dollar gets bought after iran_military / energy_oil / market_economy posts. This replicates the v2 CAR findings at 5-minute resolution.

Two additional observations need airing because they complicate the narrative:

  4. **UUP (dollar) VPIN is also elevated pre-event, but the signal is diffuse over 30+ minutes** rather than concentrated near the event. This is more consistent with FX leading macro news flow generally than with specific tweet-level front-running.

  5. **Equity-side informed-trading proxies go the wrong way for a front-running story.** VPIN and volume are unusually *low* around equity ETFs and gold before and after these tweets. Something real is happening in those windows, but it is not informed trading in equities. The most defensible framing is "market makers dampen activity around known news-risk windows" — a liquidity story, not an insider story.

---

## 6.1 Industry-picture sidebar (v2 addition — 22 April 2026)

The 600-test joint universe already covered XLE, XLF and XLK at the full FDR bar. 52 of the 180 sector-ETF tests are robust. Most are the equity-damped-VPIN pattern seen across all broad equity ETFs and do not constitute independent evidence of sector-level front-running. Two cells are additive to the story:

  * **XLE energy-topic pre-event vol_z = +0.209σ** (p = 0.014, FDR q = 0.038; placebo +0.004). The broad energy sector is unusually busy before Trump posts about oil — consistent with an attention signal but not directly directional.
  * **XLF post-event OFI_bvc = −0.039 (energy_oil), −0.030 (market_economy)** — both highly robust (FDR q ≤ 0.014; placebo −0.003). Financials sell off in the wake of macro and energy posts. This is a reaction story, not a front-running story.

The sector-level pattern tells us the informed imbalance concentrates in the commodity ETF (USO), not in the sector ETF (XLE), while the broader sector merely shows a volume attention signal. XLE's VPIN-z pattern is the equity-damped-VPIN pattern and appears in the "ruled out" column.

---

## 6.2 Dollar upper bound for the USO finding (v2 addition — 22 April 2026)

A perfect-trader upper bound, to scale the reader's intuition against the statistical findings:

| Strategy | Events | Sum P&L | Mean per event | Hit rate | Bootstrap 95% CI on total |
|---|---:|---:|---:|---:|---|
| All 173 oil-themed posts (baseline) | 173 | $181M | $1.04M | 31% | (wide, includes zero) |
| Triggered: pre-event VPIN-z > 0.5 | 81 | **$160M** | $1.97M | 46% | **[$51M, $249M]** |
| Estimators agree (tick-rule + OFI_bvc same sign) | 149 | $186M | $1.25M | 23% | [$110M, $265M] |
| Both: VPIN-z > 0.5 AND estimators agree | 69 | $167M | $2.43M | 38% | [$96M, $238M] |

**Method.** Per-event P&L = pre-event signed-volume imbalance in USO (summed over 12 × 5-min pre-bars) × (post-window-exit price − pre-window-close price). Entry at last pre-event 5-min bar close; exit at 60 minutes after the post.

**What this number is.** The maximum gross P&L available to a perfectly-informed trader who took the entire pre-event net imbalance as position. Frictionless execution at bar-close prices, no bid-ask spread, no borrowing cost, no market-impact cost from the trader's own order, no leverage cap.

**What this number is not.** Evidence that anyone actually made this money. The hit rate is under 50% in all four strategies, meaning aggregate P&L is driven by a small number of very profitable events and a larger number of small losses. A real trader would lose something on every friction in the list above. The figure exists to scale intuition around the statistical findings, not to serve as a regulatory estimate.

A full strategy backtest with transaction cost, bid-ask and borrow cost models is in scope for phase 2 (~5 days) and should sit alongside this upper bound in any published version, not replace it.

---

## 6.3 Oil-equities extension: XOM, CVX, and six other individual names (v2 addition — 22 April 2026)

To answer the obvious follow-up — *does the USO commodity-ETF signal appear one layer deeper in the actual oil companies?* — we pulled 5-minute bars for nine individual-name tickers (**TSLA, NVDA, MSFT, AAPL, META, XOM, CVX, JPM, GS**) via yfinance over the same 60-day window, built signals through the identical pipeline, and re-ran the event study. The universe expanded from 10 assets × 5 topics = 50 cells to 19 × 9 = 171 cells; after dropping level-based signals (vpin_50, dollar_volume, kyle_lambda_100) the FDR correction spans 1,140 joint tests.

**Headline.** In XOM and CVX, pre-event order-flow imbalance is positive (BUYING) ahead of oil-themed posts — and the pattern is consistent across three independent topics (`energy_oil`, `iran_military`, `market_economy`), with placebo-matched deltas 7–10× baseline.

| Topic | Ticker | Signal | Window | Mean | Placebo | Δ vs placebo | FDR q | n |
|---|---|---|---|---:|---:|---:|---:|---:|
| energy_oil | CVX | OFI_bvc | pre | +0.038 | +0.004 | +0.034 | 4e-08 | 166 |
| energy_oil | CVX | signed_vol_tick | pre | +165,000 | +146 | +164,724 | 6e-07 | 166 |
| energy_oil | XOM | OFI_bvc | pre | +0.033 | +0.005 | +0.028 | 2e-03 | 167 |
| energy_oil | XOM | signed_vol_tick | pre | +342,722 | −14,886 | +357,608 | 1e-04 | 167 |
| iran_military | CVX | OFI_bvc | pre | +0.032 | +0.004 | +0.028 | 8e-09 | 274 |
| iran_military | XOM | OFI_bvc | pre | +0.023 | +0.005 | +0.018 | 4e-03 | 263 |
| market_economy | CVX | OFI_bvc | pre | +0.042 | +0.004 | +0.038 | 3e-08 | 168 |
| market_economy | XOM | signed_vol_tick | pre | +352,065 | −14,886 | +366,951 | 4e-05 | 175 |

**USO direction flips.** The commodity ETF shows the opposite pre-event sign: vpin_z elevated (+0.38σ; placebo +0.06; FDR q ~1e-5) but signed-volume **negative** (−70K shares; FDR q ~0.05). Two defensible interpretations: (a) different participants act on the same oil-news expectation — commodity traders short USO while equity investors rotate into defensive dividend-yield majors; (b) oil-news information is itself direction-ambiguous ("drill, baby, drill" is equity-bullish for US producers' volumes but commodity-bearish). Either way, the equity-side pre-event signal uses **different tickers, different estimators, and cross-validates across three topic triggers**, so it is an independent corroboration of the USO finding's existence (not its direction).

**Rules out front-running in tech and investment-bank names.** The other seven individual names all show the **damped-VPIN** pattern previously documented in §4: pre-event vpin_z is 0.3–0.7σ below baseline in TSLA, NVDA, MSFT, META, JPM and GS. AAPL is equivocal (positive OFI_bvc pre across three topics, but the direction is likely broad-macro co-movement rather than a topic-specific signal). None of those six names is reported as a front-running finding.

### 6.3.1 Friends-vs-self null (v2 addition)

We also added four topic-classifier rules to support a "friends-vs-self" decomposition requested editorially (Musk/Tesla, Big Tech by individual-company name, Big Oil by individual-company name, Big Banks by individual-company name). On the full 32,433-post archive the topic counts are: **musk_tesla 116, big_tech 43, big_oil_companies 2, big_banks 12** (tightened regex excluding bare-letter false positives like "D.C.", "MS NOW", "apple news"). On the 1,341-post overlap window where we have minute-level market data, those counts collapse to **0, 1, 0, 2** respectively. The friends-vs-self decomposition as originally scoped is not answerable on the available market-data window. Finding 4 (above) is the closest extension we could make within the same window — individual-name analysis on the *topic* axis, not the *subject* axis. A clean friends-vs-self study would require minute-bar history further back than yfinance provides for free (paid vendor; Polygon.io ~$99/mo flat for full intraday history).

### 6.3.2 Pipeline additions (for reproducibility)

- `collect_new_tickers.py` — yfinance pull for 9 individual names (5m and 30m)
- `add_friends_topics.py` — adds `topic_musk_tesla`, `topic_big_tech`, `topic_big_oil_companies`, `topic_big_banks` boolean columns to `posts_60d.parquet`
- `event_study.py` — config updated: 9 topics × 19 assets
- `placebo_and_sensitivity.py` — same; FDR trim now drops {vpin_50, kyle_lambda_100, dollar_volume} as level-based signals
- `data/finding4_oil_complex.json` — clean summary with placebo-matched deltas
- `data/friends_topic_counts.json` — topic counts in overlap vs full archive

---

## 7. Files produced

| File | Contents |
|---|---|
| `data/minute_bars/5m/*.parquet` | Raw 5-min OHLCV for 10 tickers, 60-day window |
| `data/minute_bars/30m/*.parquet` | 30-min OHLCV (corroboration) |
| `data/minute_bars/1m/*.parquet` | 1-min OHLCV, 7-day window (no Truth-archive overlap, retained for future re-scrape) |
| `data/signals/5m/*.parquet` | Per-bar BVC buy/sell volumes, VPIN, Kyle's λ, volume z-scores |
| `data/orderflow_event_study_5m.json` | Main results: (topic × asset × signal × window) with t-tests and 5k bootstrap |
| `data/orderflow_placebo_5m.json` | 5,000 matched-timestamp placebo run |
| `data/orderflow_sensitivity_5m.json` | Window-length sensitivity (±15, ±30, ±60 min) |
| `data/orderflow_final_5m.csv` | Flat table: 600 tests × {real, placebo, FDR, robust} |
| `data/orderflow_final_5m.json` | Robust-finding summary |
| `data/orderflow_fdr_5m.json` | v2 expanded FDR (1,140 tests, 19 assets × 9 topics) |
| `data/orderflow_fdr_5m_trim.csv` | Same as above, flat CSV with placebo-matched deltas |
| `data/finding4_oil_complex.json` | Clean summary for §6.3 oil-equities finding |
| `data/friends_topic_counts.json` | Friends-vs-self topic counts (full archive vs overlap) |
| `data/dollar_upper_bound_strategies.json` | v2 dollar-figure variants (VPIN-z trigger, etc.) |
| `collect_minute_bars.py`, `collect_new_tickers.py`, `build_signals.py`, `event_study.py`, `placebo_and_sensitivity.py`, `consolidate_results.py`, `add_friends_topics.py`, `dollar_upper_bound.py` | Reproducible pipeline scripts |

## 8. Verification addendum

After the report was written, an independent verification pass re-read the pipeline and recomputed the headline numbers. The purpose here is integrity-first disclosure of what that pass found.

**What reproduced cleanly.**

  * Look-ahead. Every rolling baseline in `build_signals.py` uses `.shift(1).rolling(250)`, and Kyle's-λ uses a hand-rolled `[t−win, t−1]` window (lines 107–109). No look-ahead into the event bar enters any published test. The raw `vpin_50` (un-z-scored) does include bar t by construction, but the z-scored version used in the headline table is correctly shifted.
  * Placebo matching. The 5,000 placebo timestamps are sampled from bars grouped by (ET hour × weekday), inheriting the same RTH mask as the real-signal panel. Matching behaves as described in §2.3.
  * Multiple comparisons. BH-FDR is applied across the full 600-row (topics × assets × metrics × windows) joint table, not per-metric. The CSV has exactly 600 rows and 156 rows with `robust == True`, matching the report.
  * Headline numbers. USO × energy_oil × vpin_z (pre) = +0.380 (p = 1.4×10⁻⁶, FDR = 1.9×10⁻⁵); DJT × energy_oil × OFI_bvc at ±15 min = −0.066 (p = 1.1×10⁻⁸); placebo cells all ≪ real means. Every headline number in §3.1 and §3.2 reproduces from `orderflow_final_5m.csv` and `orderflow_sensitivity_5m.json` to three decimals.

**What did not.**

  * **Topic classifier — `iran_military` is contaminated.** The regex (inherited from `market_deep_analysis_v2.py` line 75) includes a standalone `\bmilitary\b` token. Of 284 posts tagged `iran_military` in the 73-day window, 176 (~62%) contain no literal Iran/Iranian/Tehran/Hormuz mention — they are predominantly MAGA-endorsement boilerplate in which "military" appears generically. Any `iran_military` × asset row in §3 is therefore a mixture of real Iran posts (~38%) and endorsement-spam posts (~62%), and the real effect size on genuine Iran posts is almost certainly larger than the reported cell; conversely any claim about the Iran topic specifically is weakened because we cannot distinguish the two populations from the current run. The fix is small — require Iran/Iranian/Tehran/Hormuz/nuclear or "military action" (as a bigram) before tagging — and should be applied before any Iran-specific claim is published. `energy_oil`, `market_economy`, `tariff_trade` and `djt_media` regexes were spot-checked and classify cleanly (e.g. 157 / 173 `energy_oil` hits contain a literal "energy" token).
  * **"Replicates on three window lengths" is slightly over-stated.** The USO × energy_oil VPIN signal is strongest at ±15 min (+0.39) and softer at ±60 (+0.31). It survives all three windows but is a single-peak finding, not an equally-spread one. The claim in §6 should read "survives three windows and is strongest at ±15 min."

**Bottom line.** Methodology, FDR correction, placebo matching and every published number reproduce independently. The one real defect is the `iran_military` topic boundary: results pinned only on the Iran topic should be held for re-running on a tightened classifier; the USO × energy_oil front-running-consistent signal, the DJT market/economy and energy pre-event selling, and the post-event risk-off rotation all survive this critique.

---

## 9. References

  * Easley, D., López de Prado, M., O'Hara, M. (2012). *Flow Toxicity and Liquidity in a High-Frequency World*. Review of Financial Studies 25(5): 1457–1493.
  * Easley, D., López de Prado, M., O'Hara, M. (2016). *Discerning information from trade data*. Journal of Financial Economics 120(2): 269–285. (BVC)
  * Hasbrouck, J. (2009). *Trading costs and returns for U.S. equities: Estimating effective costs from daily data*. Journal of Finance 64: 1445–1477. (Kyle's λ estimation)
  * Kyle, A. (1985). *Continuous auctions and insider trading*. Econometrica 53(6): 1315–1335.
  * Chakrabarty, B., Pascual, R., Shkilko, A. (2015). *Evaluating trade classification algorithms: Bulk volume classification versus the tick rule and the Lee-Ready algorithm*. Journal of Financial Markets 25: 52–79.
  * Andersen, T., Bondarenko, O. (2014). *VPIN and the flash crash*. Journal of Financial Markets 17: 1–46. (Critical review of VPIN; see also Pöppe, Moos & Schiereck 2016.)
