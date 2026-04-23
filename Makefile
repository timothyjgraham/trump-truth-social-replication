# Reproducibility Makefile for the Trump / Truth Social / oil markets study.
#
# Usage:
#   make help          show this help
#   make install       create venv and install Python deps
#   make reproduce     run the full analysis pipeline from cached raw data
#   make collect       (slow, network) re-collect posts and minute bars
#   make clean         remove derived signals and results
#   make report        compile the LaTeX report to PDF
#
# Stage targets (run individually if you want to step through):
#   make signals       build OFI_bvc / vpin_z / Kyle's lambda from minute bars
#   make event_study   pre/post event study across topics × assets
#   make placebo       matched-placebo + window-sensitivity + FDR
#   make timeshift     +24h time-shift falsification
#   make dollar_bound  back-of-envelope $ upper bound on USO
#   make loo           leave-one-out fragility audit
#   make figures       regenerate the three headline paper figures
#
# Post-feedback extras (run individually or together via `make extras`):
#   make posting       posting-patterns / weekend hypothesis test (stage 11)
#   make sector_sweep  non-oil topic × sector ETF grid (stage 12)
#   make session_split RTH / extended / weekend event-study (stage 13)
#   make fig7          signal-overlay timeline (stage 14)
#   make fig8          per-event P&L concentration / burst structure (stage 15)
#   make collect_crypto  (slow, network) Coinbase 5m for BTC, ETH (stage 16)
#   make crypto_study  cross-asset BTC/ETH placebo + Fig 9 (stage 17)
#   make extras        run stages 11, 12, 13, 14, 15, 17 in sequence

PY            ?= python3
VENV          ?= .venv
PYBIN         := $(VENV)/bin/python
PIP           := $(VENV)/bin/pip
CODE          := code

.PHONY: help install reproduce collect signals event_study placebo timeshift \
        dollar_bound loo figures report clean clean_all \
        posting sector_sweep session_split fig7 fig8 collect_crypto \
        crypto_study extras

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) 2>/dev/null \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' \
	  || sed -n '1,40p' $(MAKEFILE_LIST)

install: ## Create venv and install Python dependencies
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ── End-to-end: from cached raw → results → figures → PDF ───────────────────
reproduce: signals event_study placebo timeshift dollar_bound loo figures report

# ── Stage targets ───────────────────────────────────────────────────────────
collect: ## (slow, network) re-collect posts and minute bars
	$(PYBIN) $(CODE)/01_scrape_truth_social.py
	$(PYBIN) $(CODE)/02_collect_minute_bars.py

signals: ## Build per-bar order-flow signals
	$(PYBIN) $(CODE)/03_build_signals.py

event_study: ## Pre/post event study, all topics × assets
	$(PYBIN) $(CODE)/04_event_study.py

placebo: ## Matched-placebo + window-sensitivity + FDR
	$(PYBIN) $(CODE)/05_placebo_and_sensitivity.py
	$(PYBIN) $(CODE)/06_consolidate_results.py

timeshift: ## +24h time-shift falsification
	$(PYBIN) $(CODE)/07_time_shift_test.py

dollar_bound: ## Back-of-envelope $ upper bound on USO
	$(PYBIN) $(CODE)/08_dollar_upper_bound.py

loo: ## Leave-one-out fragility audit
	$(PYBIN) $(CODE)/09_loo_fragility.py

figures: ## Regenerate the three headline paper figures
	$(PYBIN) $(CODE)/10_build_figures.py

# ── Post-feedback extras ────────────────────────────────────────────────────
posting: ## Posting-patterns / weekend hypothesis test
	$(PYBIN) $(CODE)/11_posting_patterns.py

sector_sweep: ## Non-oil topic × sector ETF grid
	$(PYBIN) $(CODE)/12_sector_sweep.py

session_split: ## RTH / extended / weekend session split event study
	$(PYBIN) $(CODE)/13_session_split_event_study.py

fig7: ## Signal-overlay timeline
	$(PYBIN) $(CODE)/14_signal_overlay_timeline.py

fig8: ## Per-event P&L concentration / burst structure
	$(PYBIN) $(CODE)/15_pnl_concentration_chart.py

collect_crypto: ## (slow, network) Coinbase 5m bars for BTC-USD, ETH-USD
	$(PYBIN) $(CODE)/16_collect_crypto_bars.py

crypto_study: ## Cross-asset BTC/ETH placebo + Fig 9
	$(PYBIN) $(CODE)/17_crypto_event_study.py

extras: posting sector_sweep session_split fig7 fig8 crypto_study  ## Run all post-feedback extras (assumes crypto bars already collected)

report: ## Compile the LaTeX report to PDF (requires pdflatex)
	cd report && pdflatex -interaction=nonstopmode report.tex && \
	             pdflatex -interaction=nonstopmode report.tex

clean: ## Remove derived signals and JSON results (keep raw data)
	rm -rf data/interim/signals_5m/*.parquet
	rm -f data/results/*.json data/results/*.csv
	rm -f report/*.aux report/*.log report/*.out report/*.toc

clean_all: clean ## Also remove raw cached data (will require make collect)
	rm -rf data/raw/minute_bars_5m/*.parquet
	rm -f data/raw/posts_60d.parquet data/raw/truth_archive.json data/raw/truth_archive.csv
