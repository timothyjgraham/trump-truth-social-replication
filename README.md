# Trump, Truth Social, and Moving the Market

A fully reproducible replication package for the report **_Trump, Truth Social,
and Moving the Market: Evidence of pre-event order-flow imbalances on the U.S.
oil complex (Feb–Apr 2026)._**

Everything you need to rebuild the report end-to-end is in this repository:
the raw data, the intermediate signals, the analysis scripts, the result
files, the LaTeX source, and the compiled PDF.

---

## TL;DR of the finding

- We collected every public post by `@realDonaldTrump` on Truth Social over
  a 60-day window (22 February → 22 April 2026) and matched each post to
  5-minute order-flow signals on ten U.S. equities and ETFs (DJT, SPY, QQQ,
  XLE, USO, GLD, UUP, VXX, XLF, XLK).
- For oil-themed posts (n = 165), pre-event order-flow imbalance on **XLE**
  (energy ETF) is roughly **4.4× the matched-placebo baseline** and survives
  a Benjamini–Hochberg false-discovery correction at q ≈ 7×10⁻¹¹.
- For oil-themed posts on **USO** (oil-price-tracking ETF), pre-event
  informed-trading intensity (`vpin_z`) is significantly elevated relative
  to placebo (q ≈ 1.3×10⁻⁵). A +24-hour time-shift falsification flips the
  sign — the pattern is locked to the actual posting times, not to
  daily-seasonality artifacts.
- A **gross-of-frictions ceiling** on what a perfectly-informed trader
  holding the entire pre-event imbalance could have netted from the 81
  triggered USO events (those with pre-window mean(`vpin_z`) > 0.5) is
  **≈ $159.88M**. This is an upper bound, not a claim about realised P&L.
  The headline is fragile to a single 23 March 2026 event (20% of the total)
  and to four near-duplicate posts on 7 April 2026 (a further $36.4M).

The full numbers, methodology, and caveats are in
[`report/report.pdf`](report/report.pdf).

---

## Repository layout

```
trump-truth-social-replication/
├── README.md                ← you are here
├── LICENSE                  ← MIT
├── CITATION.cff             ← academic citation block
├── Makefile                 ← one-command reproduction
├── requirements.txt         ← Python dependencies
├── paths.py                 ← canonical filesystem paths (do not move)
│
├── code/                    ← seventeen numbered pipeline stages
│   ├── _paths.py
│   ├── 01_scrape_truth_social.py
│   ├── 02_collect_minute_bars.py
│   ├── 03_build_signals.py
│   ├── 04_event_study.py
│   ├── 05_placebo_and_sensitivity.py
│   ├── 06_consolidate_results.py
│   ├── 07_time_shift_test.py
│   ├── 08_dollar_upper_bound.py
│   ├── 09_loo_fragility.py
│   ├── 10_build_figures.py
│   ├── 11_posting_patterns.py            (weekend / weekday split)
│   ├── 12_sector_sweep.py                (non-oil topic × sector grid)
│   ├── 13_session_split_event_study.py   (RTH / extended / weekend)
│   ├── 14_signal_overlay_timeline.py     (Fig 7)
│   ├── 15_pnl_concentration_chart.py     (Fig 8 — burst structure of $160M)
│   ├── 16_collect_crypto_bars.py         (Coinbase 5m for BTC, ETH)
│   └── 17_crypto_event_study.py          (Fig 9 — cross-asset placebo)
│
├── data/
│   ├── raw/                 ← inputs you would re-collect from scratch
│   │   ├── truth_archive.json   (32,433 posts, full Truth Social archive)
│   │   ├── truth_archive.csv    (same posts, flattened)
│   │   ├── posts_60d.parquet    (the 60-day window, topic-tagged)
│   │   └── minute_bars_5m/      (one .parquet per ticker, from yfinance)
│   ├── interim/
│   │   └── signals_5m/      ← BVC, VPIN, Kyle's λ derived from raw bars
│   └── results/             ← cached JSON / CSV outputs of every stage
│
├── report/
│   ├── report.tex           ← LaTeX source
│   ├── report.pdf           ← compiled, 23 pages A4
│   └── figures/             ← nine paper figures (PDF + PNG)
│
└── docs/
    ├── methodology.md       ← extended notes on signal construction
    └── data_dictionary.md   ← every column in every parquet
```

---

## Setup

Tested on Python 3.10–3.12. About 10 minutes for a full local install.

```bash
git clone https://github.com/<your-username>/trump-truth-social-replication.git
cd trump-truth-social-replication
python3 -m venv .venv
source .venv/bin/activate         # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

To compile the LaTeX report yourself you also need a working TeX
distribution (TeX Live, MacTeX, or MikTeX) with `pdflatex` and `bibtex`
on `$PATH`. Skip this if you only want to read the cached
`report/report.pdf`.

---

## Reproducing the analysis

### One-shot reproduction from cached raw data (recommended)

The repository ships with the raw inputs already in place. The full
downstream pipeline runs in roughly five minutes on a laptop:

```bash
make reproduce
```

This re-runs stages 03–10 — signals → event study → placebos → time-shift
falsification → dollar bound → leave-one-out fragility → figures — and
overwrites the contents of `data/interim/`, `data/results/`, and
`report/figures/`. The headline numbers should reproduce exactly, because
all stages with random sampling (placebos, bootstraps) seed
`numpy.random.default_rng(20260422)`.

### Re-collecting the raw data from scratch

Stages 01 and 02 talk to the public internet and are not run by
`make reproduce`. To rebuild the entire pipeline from scratch, including
re-scraping Truth Social and re-fetching minute bars:

```bash
make collect          # stage 01: scrape Truth Social
make signals          # stage 02 + 03: bars + microstructure signals
make reproduce        # stages 04–10
```

Two notes on re-collection:

1. **Truth Social.** Stage 01 hits the public Mastodon-compatible endpoint
   `https://truthsocial.com/api/v1/accounts/107780257626128497/statuses`.
   It is unauthenticated and rate-limited; please be polite. The script
   backfills against the CNN/Stiles Truth Social archive for any window
   the live API has trimmed.
2. **Minute bars.** Stage 02 uses `yfinance` to fetch 5-minute OHLCV bars.
   Yahoo only serves a rolling ~60 days of intraday history at this
   resolution, so re-running stage 02 today will not return the same
   window of bars used in the report. The shipped
   `data/raw/minute_bars_5m/*.parquet` files are the canonical inputs.

### Targeting an individual stage

```bash
make event_study      # stage 04
make placebo          # stage 05 (5,000 hour-and-weekday-matched placebos)
make timeshift        # stage 07 (+24h falsification)
make dollar_bound     # stage 08
make loo              # stage 09 (leave-one-out fragility)
make figures          # stage 10
make extras           # stages 11, 12, 13, 14, 15, 17 — post-feedback analyses
make collect_crypto   # stage 16 (network: Coinbase 5-min bars for BTC, ETH)
make report           # pdflatex + bibtex + pdflatex × 2
```

Each `make <target>` is just shorthand for `python code/NN_<name>.py`.

The `extras` group covers everything added in response to peer-reviewer
feedback: the posting-patterns / weekend-vs-weekday hypothesis test, the
cross-topic sector ETF sweep, the RTH-vs-after-hours-vs-weekend session
split, the signal-overlay timeline (Figure 7), the per-event P&L
concentration chart (Figure 8), and the cross-asset BTC/ETH placebo
around the 81 triggered oil events (Figure 9). They are independent of
the headline pipeline (stages 03–10) and can be re-run individually.

---

## Key findings, with exact numbers

The headline numbers in the report all map back to a single JSON file. To
reproduce them yourself:

```bash
python -c "import json; print(json.dumps(json.load(open('data/results/phase2c_loo.json'))['conventions']['A_production'], indent=2, default=str))"
```

| Result | File | Key |
| --- | --- | --- |
| XLE pre-event OFI (real vs placebo) | `data/results/orderflow_final_5m.json` | `survivors[?(@.asset==XLE && @.metric==OFI_bvc)]` |
| USO pre-event vpin_z (real vs placebo) | `data/results/orderflow_final_5m.json` | `survivors[?(@.asset==USO && @.metric==vpin_z)]` |
| +24h time-shift sign-flip | `data/results/phase2b_timeshift.json` | `assets.USO.shifted.vpin_z` |
| $159.88M aggregate gross P&L | `data/results/dollar_upper_bound_strategies.json` | `triggered_vpinz_gt_0_5.sum_pnl_usd` |
| 20% single-event share (23 Mar 2026) | `data/results/phase2c_loo.json` | `conventions.A_production.max_single_event_share_pct` |
| LOO range $150.8M → $191.8M | `data/results/phase2c_loo.json` | `conventions.A_production.loo_sum_min_usd / loo_sum_max_usd` |

---

## Methodology in two paragraphs

For each oil-themed Truth Social post we anchor a 30-minute pre-event
window and a 30-minute post-event window on the corresponding 5-minute
bar grid. Inside the pre-window we compute three microstructure signals:
order-flow imbalance using Easley–López de Prado–O'Hara bulk-volume
classification (`OFI_bvc`); Kyle's λ as the absolute slope of price-on-
signed-volume; and VPIN, z-scored against a 250-bar rolling baseline
*shifted by one bar* so no future information enters the score
(`vpin_z`). We test pre-window means against zero with a stationary
bootstrap and then against an event-time-matched placebo distribution
of 5,000 random non-event timestamps on the same hour-and-weekday grid.
P-values are corrected with Benjamini–Hochberg.

Two falsifications guard the headline. (1) Shifting every post timestamp
forward by exactly 24 hours and re-running the same pipeline kills the
USO `vpin_z` signal and flips its sign — the elevation is locked to the
actual posting times, not to daily seasonality. (2) A leave-one-out
audit on the 81-event triggered slice shows the $159.88M headline
moving to $150.8M when the largest negative-P&L event (23 March 2026,
–$31.9M) is removed, and to $191.8M when the largest positive event is
removed — a fragility we surface explicitly in Figure 3 of the report.

For more, see `docs/methodology.md` and the report itself.

---

## Citation

If you use this work please cite it as follows. A
[`CITATION.cff`](CITATION.cff) is provided so GitHub can render a citation
widget on the repo page automatically.

> Graham, T., Harrington, S., & Chorazy, E. (2026). *Trump, Truth Social, and
> the Market: Evidence of pre-event order-flow imbalances on the U.S. oil
> complex (Feb–Apr 2026).* Replication package, version 1.0.

---

## Licence

Code and analysis released under the MIT licence (see [`LICENSE`](LICENSE)).

The Truth Social posts in `data/raw/` are © Donald J. Trump and are
redistributed here for non-commercial research use under fair-use
provisions. The 5-minute equity bars in `data/raw/minute_bars_5m/` were
retrieved via `yfinance` from public Yahoo Finance endpoints and are
provided strictly for replication purposes.

---

## Contact

Questions, issues, or pull requests welcome via the GitHub issue tracker.
