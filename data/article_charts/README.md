# Article chart data

The CSV / JSON / preview-PNG bundles that feed the visualisations in
the public-facing Conversation article.

## Layout

- `v1/` -- first iteration of the four article charts. Used during the
  initial editorial round. Includes the 15-event population overlay
  (chart 1), the topic-cooccurrence network (chart 2), the topic x
  asset heatmap (chart 3), and the 73-day timeline with event markers
  (chart 4).
- `v2/` -- revised handover bundle, with positive-only-y-axis recasts,
  wider event windows for the bursts that span market close, the
  Mar 23-vs-shifted-control falsification chart, and a tightened
  word-network (broad and strict variants). This is the bundle the
  published article actually uses.

Each subfolder has its own `build.py` (or `build_matt_charts.py`),
`run.sh`, `requirements.txt`, and `verify.py`, plus the data files
themselves.

## Relationship to the report figures

The article chart data is a publication-tuned subset of the analysis
that produces the report figures in `report/figures/` and
`report/figures_revised/`. The numbers are the same. The presentation
differs because the article and the report serve different audiences.

