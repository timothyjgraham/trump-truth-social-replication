# Case studies

Single-event drill-downs that supplement the main report. Each subfolder
is a self-contained bundle with its own README, build/render/verify
pipeline, and source CSVs.

## Available case studies

- **`mar23_iran_ceasefire/`** -- the morning of 23 March 2026. USO ETF
  drops about 8% in a single 5-minute bar at 07:05 ET, eighteen minutes
  before Trump posts the official US-Iran ceasefire announcement at
  07:23 ET. Includes a +24h-shifted control series (March 24) showing
  the same hours stay within roughly +/- 2.5% on the next trading day.
  This is the marquee single event referenced in the Conversation
  article and the press coverage.

## Reproducing

Each bundle has its own `run.sh` that runs `build.py`, `render.py`, and
`verify.py` in order. The pipelines are deterministic -- same inputs
always produce the same outputs.

