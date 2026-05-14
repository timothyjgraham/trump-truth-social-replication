# Word co-occurrence matrix - oil / Iran-themed Trump posts

Cleaned co-occurrence dataset and triangular heatmap for the substantive
content vocabulary in Trump's oil and Iran-related Truth Social posts. An
alternative to the bubble / network rendering of the same data.

## Universe

121 Truth Social posts that contain at least one explicit oil / Iran /
Hormuz / crude / Venezuela / OPEC term in the body. This is the same
"strict" universe used elsewhere in the bundle - filters out posts the
topic classifier flagged as oil-themed but where the actual content was
something else (endorsement spam etc.).

## What's filtered out

To surface the substantive content vocabulary, three categories of words
are dropped from the candidate set:

  - **Generic English stopwords** (the, a, of, to, etc.)
  - **Self-references**: trump, donald, president, djt - these otherwise
    dominate the matrix as a top-left block because Trump's posts often
    sign off with "President Donald J. Trump"
  - **Content-light fillers**: now, again, far, about, only, else,
    well, back, completely, highly, against, help, immediately, plus
    fragments of the "Thank you for your attention to this matter"
    boilerplate

## What's left

The top 30 substantive words:

| Rank | Word | Posts containing |
|---|---|---|
| 1 | iran | 91 |
| 2 | military | 39 |
| 3 | states | 38 |
| 4 | united | 38 |
| 5 | america | 34 |
| 6 | world | 26 |
| 7 | country | 25 |
| 8 | oil | 23 |
| 9 | strait | 23 |
| 10 | hormuz | 22 |
| 11 | middle | 20 |
| 12 | east | 19 |
| 13 | regime | 18 |
| 14 | deal | 17 |
| 15 | nuclear | 17 |
| 16 | war | 15 |
| 17 | american | 14 |
| 18 | attack | 14 |

(remaining 12 are in `word_nodes.csv`).

## What the matrix surfaces

The dark cells in the heatmap are the substantive story:

  - **Iran <-> Military** (top of matrix; very dark)
  - **United <-> States** (geographic anchor)
  - **Iran <-> America / United / States / Country** (geopolitical framing)
  - **Hormuz <-> Strait** (~22; the Strait of Hormuz)
  - **Middle <-> East** (the Middle East)
  - **Iran <-> Strait / Hormuz / Nuclear / Deal / Regime** (Iran-deal /
    regime-change narrative)
  - **War / Attack / Israel** (military escalation vocabulary)

## Files

```
README.md
build.py / render.py / verify.py / run.sh / requirements.txt

word_nodes.csv          # 30 rows: word, post_count, rank
word_edges.csv          # pair co-occurrences with weight >= 3
word_matrix.csv         # 30x30 symmetric co-occurrence matrix
                        # (diagonal = post count for that word)
word_network.json       # D3-friendly nested form (nodes + edges + matrix)

previews/
└── word_matrix.png     # triangular heatmap rendering
```

## How to read the chart

  - Each row / column is one of the top 30 substantive words.
  - Each cell shows how many posts contain *both* of the words in that
    row/column pair.
  - Darker cells = higher co-occurrence (i.e. those two words appear
    together in more of the 121 posts).
  - The matrix is symmetric, so only the lower-left triangle is drawn.
  - The diagonal would show each word's own post count - that's already
    in the row labels (`word (N posts)`) so the diagonal is left blank.

## Reproducing

```bash
pip install -r requirements.txt
./run.sh
```

`run.sh` runs:
1. `python build.py` - reads from `../trump-truth-social-replication/`,
   writes the four data files.
2. `python render.py` - produces `previews/word_matrix.png`.
3. `python verify.py` - 15 consistency checks against the source data.

If the source repo lives elsewhere, override:
```bash
python build.py --repo /path/to/trump-truth-social-replication
```

The pipeline is deterministic - same inputs always produce the same
outputs.

## Source data

  - `data/raw/posts_60d.parquet` - Truth Social post archive (1,341 posts)
