# Order-flow extended pipeline

These scripts are the extended order-flow event-study pipeline that
produces the results documented in `docs/orderflow_methodology.md`. They
share methodology with the numbered pipeline in the parent folder
(01_-17_) but operate on a wider asset universe and with a more
detailed BVC / VPIN / Kyle's-lambda treatment.

The numbered pipeline in the parent `code/` folder is the canonical
end-to-end build for `report/report.pdf`. The scripts here are the
extension used for the v2 addendum (XOM / CVX individual-name extension,
the 1,140-test FDR universe, the dollar upper bound triggered-VPIN
variant, and the friends-vs-self topic addition).

## Files

- `build_signals.py` -- BVC, VPIN, Kyle's lambda construction on 5-min
  bars (extended over `code/03_build_signals.py`).
- `collect_minute_bars.py`, `collect_new_tickers.py` -- yfinance pulls
  for the extended 19-ticker universe.
- `orderflow_event_study.py` -- per-(topic, asset, signal, window) cell
  testing across the 19-asset universe.
- `orderflow_placebo_and_sensitivity.py` -- 5,000 matched-timestamp
  placebo run plus pre/post window sensitivity.
- `orderflow_consolidate_results.py` -- consolidates the JSON outputs
  into the flat tables consumed by the article visuals.
- `add_friends_topics.py` -- adds Musk/Tesla, Big Tech, Big Oil, Big
  Banks topic flags to the posts table.
- `dollar_upper_bound.py` -- the gross-of-frictions dollar ceiling
  calculation for the 81 triggered USO events.

## Relationship to the numbered pipeline

The numbered scripts (01-17) are the canonical build for the headline
report. The scripts in this folder are the extension layer used during
review and refinement. The two pipelines are designed to produce
compatible results -- numbers cited in the methodology and audit docs
in `docs/` reproduce in both pipelines to three decimal places.

