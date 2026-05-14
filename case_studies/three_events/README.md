# Three case-study events - matched format

USO ETF data and charts for the three events lined up in the article -
all in identical column structure, time window, and units so they can be
rendered with one chart template.

| Event | Date / time (ET) | Role |
|---|---|---|
| Venezuela oil | Mar 4, 15:09 | "normal signal" - clean post-event reaction |
| Iran ceasefire | Mar 23, 07:23 | "pre-event signal" - trading and news preceded the post |
| Indiana endorsement burst | Apr 7, 16:13 | "the massive one" - headline PnL event |

## Files

```
README.md
build.py / render.py / verify.py / run.sh / requirements.txt

three_events_uso_5min.csv          # all three events stacked, same columns
three_events_metadata.csv          # one row per event (anchor, post text, session)
three_events.json                  # nested D3-friendly form

previews/
├── mar4_venezuela_oil.png         # Kobeissi-style chart, Mar 4
├── mar23_iran_ceasefire.png       # Kobeissi-style chart, Mar 23
├── apr7_indiana_burst.png         # Kobeissi-style chart, Apr 7
└── three_events_cum_pct_overlay.png  # all three on one cum % return chart
```

## Data format (same across all three events)

`three_events_uso_5min.csv` - 5-min USO bars, +/- 180 min around each
event's anchor.

| column | meaning |
|---|---|
| `event_id` | one of the three IDs above |
| `event_label` | human-readable label |
| `ts_utc` / `ts_et` | bar timestamp (UTC and US Eastern) |
| `minute_offset` | minutes from the event's anchor (0 = the post-time bar) |
| `open` / `high` / `low` / `close` | OHLC in USD |
| `volume` | shares traded in this bar (zero on pre-market and after-hours) |
| `is_zero_vol` | True where `volume == 0` (extended-hours bars) |
| `is_rth_bar` | True where the bar is in regular trading hours |
| `anchor_price` | the close at minute_offset = 0 (constant per event) |
| `cum_pct_signed` | `(close / anchor_price - 1) * 100` |
| `cum_pct_abs` | `\|cum_pct_signed\|` |

`three_events_metadata.csv` - per-event metadata including the anchor
timestamp, post id, post text, session classification, and bar counts.

## Window and anchor convention

Each event's anchor is the 5-min bar at or just before the Trump post
timestamp. The window extends +/- 180 minutes around that anchor (~73
bars total), giving 3 hours each side of t=0. This is wider than the
30-min analytical window the underlying study uses but matches the Mar
23 case-study window so the three events sit in a directly comparable
frame.

To use a tighter window in the renderer, just filter on
`abs(minute_offset) <= N` for whatever N you want.

## Charts

All three event charts use the same Kobeissi-style dark-theme layout:
candlestick price panel + volume panel beneath, anchor price as a
horizontal reference line, RTH session boundaries as dotted vertical
lines where they fall in the window, and the Trump post timestamp as a
dashed vertical line with the post text in a callout box.

**Colour conventions** (colorblind-safe Okabe-Ito):
  - Up bars: blue (#0072B2)
  - Down bars: vermillion (#D55E00)
  - Trump post markers: sky blue (#56B4E9)
  - Session boundaries: neutral gray

**A note on candlestick wicks**: USO does not trade in pre-market or
after-hours, so non-RTH bars carry zero volume. yfinance's high/low
fields on those bars are unreliable single-tick quote anomalies, so
the wicks on zero-volume bars have been suppressed (only the body is
drawn). RTH bars get full candles with wicks.

## Reproducing

```bash
pip install -r requirements.txt
./run.sh
```

`run.sh` runs:
1. `build.py` - reads from `../trump-truth-social-replication/`, writes
   the CSV and JSON outputs.
2. `render.py` - produces the four preview PNGs.
3. `verify.py` - runs consistency checks against the source data.

If the source repo lives elsewhere, override:
```bash
python build.py --repo /path/to/trump-truth-social-replication
```

## Source data

  - `data/raw/posts_60d.parquet` - Trump Truth Social post archive
  - `data/raw/minute_bars_5m/USO.parquet` - 5-min OHLCV
