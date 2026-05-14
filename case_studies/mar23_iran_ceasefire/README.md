# March 23, 2026 — Iran ceasefire morning case study

A focused, single-event look at the Iran ceasefire announcement morning,
matching the timeline shown in the Kobeissi Letter chart on light crude
futures. This bundle replicates the same morning's price movement on
**USO ETF** (the listed equivalent that closely tracks light crude
futures), pins the news and Trump-post timestamps to the price
sequence, and provides a +24h-shifted control series for falsification.

## Headline

USO ETF dropped from $122.84 to $113.05 in a **single 5-minute bar at
07:05 ET on Mar 23**, lining up with the Kobeissi-marked "$125M short
profit at 7:00 AM ET" timestamp. **Trump's official Iran ceasefire
announcement** ("I AM PLEASED TO REPORT THAT THE UNITED STATES OF
AMERICA, AND THE COUNTRY OF IRAN, HAVE HAD…") posted at **07:23:40 ET**
— eighteen minutes after the big move. The Axios deal-imminent report
that Kobeissi cites preceded the move by a couple of hours. Trump's
later 10:29 ET RT of the same content fired during regular trading
hours, by which point the move was already in the price.

The next morning (March 24) shows no such pattern at the same hours —
USO stayed within roughly ±2.5% of its 07:00 ET anchor across the
entire morning, vs Mar 23's −13% trough. Consistent with the Mar 23
move being event-specific, not a recurring time-of-day artifact.

## What's in the bundle

```
README.md                                   # this file
build.py / render.py / verify.py / run.sh   # the pipeline (deterministic)
requirements.txt                            # Python dependencies

mar23_uso_5min.csv                          # 88 USO 5-min bars, 03:00-11:30 ET
mar23_event_timeline.csv                    # external markers + Trump posts
mar23_vs_mar24_falsification.csv            # Mar 23 vs +24h shifted control

previews/
├── mar23_overview.png                      # the headline chart
└── mar23_vs_mar24_falsification.png        # control comparison
```

## The headline chart (`previews/mar23_overview.png`)

USO 5-min bars, 04:00–11:30 ET on Mar 23 2026. Annotations in time order:

  - **04:50 ET** — Axios reports a deal to end the Iran war is imminent
    (per the Kobeissi annotation; we cite this rather than independently
    sourcing it).
  - **07:05 ET** — USO bar opens at $122.82, prints a low of $109.00,
    closes $113.05. A −8% move in five minutes. Lines up with the
    Kobeissi-marked "shorts profit $125M at 7:00 AM" event.
  - **07:23 ET** — Trump posts the original Iran ceasefire announcement
    on Truth Social: *"I AM PLEASED TO REPORT THAT THE UNITED STATES OF
    AMERICA, AND THE COUNTRY OF IRAN, HAVE HAD, OVER THE LAST TWO DAYS,
    VERY GOOD AND PRODUCTIVE CONVERSATIONS REGARDING A COMPLETE AND
    TOTAL RESOLUTION…"* — eighteen minutes after the move on USO.
  - **09:30 ET** — Regular trading hours open. USO's first RTH bar
    closes at $112.97 on 12.8M shares (the first non-zero-volume bar).
  - **10:29 ET** — Trump RTs the same Iran post (this is the post
    flagged in the wider event study).

Friday Mar 20 close ($121.44) is shown as a reference dashed line. USO
gapped up overnight to ~$125.80 at 04:00 ET pre-market, then drifted
down through 07:00 ET before the −8% bar.

## The falsification chart (`previews/mar23_vs_mar24_falsification.png`)

Same window (03:00–11:30 ET) on the next trading day, anchored to each
day's 07:00 ET close as 0%. Mar 23 (red) drops to about −13% by 09:00
ET and recovers partially. Mar 24 (green) stays within −1.0% to +2.5%
across the full window. The contrast is the +24h-shifted falsification
applied to a single event.

## Important context — USO vs CL futures

The Kobeissi Letter chart references **light crude oil futures (CL on
NYMEX)**, where the $920M short position and $125M profit numbers were
calculated. This bundle uses **USO ETF** (United States Oil Fund),
which is a listed instrument that closely tracks the same crude price
but trades on NYSE Arca with its own volume and liquidity profile.
Treat the Kobeissi $920M / $125M figures as cited context on the
underlying futures market, not as numbers we've independently derived.
USO's price move on this morning lines up tightly with what the
Kobeissi chart shows on CL.

## Important context — pre-market USO bars are zero-volume

USO ETF doesn't trade on meaningful volume before 09:30 ET (the
regular session open). All pre-market bars in `mar23_uso_5min.csv`
have `is_zero_vol = True` and `is_rth_bar = False`. The price discovery
visible in those bars reflects quote movement (specialists / market
makers updating quotes in response to overnight futures activity)
rather than executed ETF trades. This is normal for ETFs in extended
hours and doesn't make the price moves fictional — USO opened the
regular session at $112.97, locking in roughly the −8% repriced level
that the pre-market bars had been tracking. But the article should
caption pre-market bars as "quote-driven" rather than implying
intraday ETF trading at those prices.

## File-level documentation

### `mar23_uso_5min.csv` (88 rows)
Every USO 5-min bar from 03:00 ET to 11:30 ET on Mar 23 2026.

| column        | meaning                                                |
|---------------|--------------------------------------------------------|
| `ts_utc`      | bar timestamp (UTC)                                    |
| `ts_et`       | bar timestamp (US Eastern)                             |
| `open` / `high` / `low` / `close` | OHLC in USD                        |
| `volume`      | shares traded in the bar (zero pre-market)             |
| `is_zero_vol` | True where `volume == 0`                               |
| `is_rth_bar`  | True where the bar is in regular trading hours         |

### `mar23_event_timeline.csv`
External markers (Kobeissi / news / session boundaries) and all Trump
posts on Mar 22–23, 2026 in one timeline. `kind` distinguishes
`external_marker` / `session_marker` / `trump_post`. The
`is_oil_or_iran_topic` flag marks the two posts (07:23 ET original,
10:29 ET RT) that are tagged as oil + Iran by the topic classifier.

### `mar23_vs_mar24_falsification.csv` (88 rows)
Each row is one minute-of-day from 03:00 to 11:30 ET. Columns
`real_pct_vs_07_00` (Mar 23) and `shifted_pct_vs_07_00` (Mar 24)
express each bar's close as a percentage deviation from the
respective day's 07:00 ET close — anchoring both lines at 0% so the
shapes are directly comparable.

## Reproducing this bundle

```bash
pip install -r requirements.txt
./run.sh
```

`run.sh` runs three steps:

1. `python build.py` — reads from `../trump-truth-social-replication/`
   and writes the three CSVs to this folder.
2. `python render.py` — produces the two PNGs in `previews/`.
3. `python verify.py` — runs 15 consistency checks against the source
   data and exits non-zero if any fail.

If the source repo lives elsewhere, override:

```bash
python build.py --repo /path/to/trump-truth-social-replication
python verify.py --repo /path/to/trump-truth-social-replication
```

The pipeline is deterministic — same inputs always produce the same
outputs.

### Source data

  - `data/raw/posts_60d.parquet` — Truth Social posts (1,341 posts)
  - `data/raw/minute_bars_5m/USO.parquet` — 5-min OHLCV from yfinance
