#!/usr/bin/env python3
"""
Add topic_* boolean columns for the "friends-vs-self" analysis.

Topics added:
  - topic_musk_tesla        : Musk / Tesla / xAI / SpaceX / Neuralink
  - topic_big_tech          : Apple / Microsoft / Google / Meta / Nvidia / Amazon (individual names)
  - topic_big_oil_companies : ExxonMobil / Chevron / major integrated oil companies
  - topic_big_banks         : JPMorgan / Goldman / BofA / Citi / Wells Fargo / Morgan Stanley

Rationale: these topics let us separate self-reference posts (about DJT/Trump Media,
Trump family, Trump properties) from "friends" posts (about allied/adjacent entities).

Match rules are KEYWORD-BASED on the HTML-stripped post content, case-insensitive,
word-boundaried. Tickers in brackets ($TSLA, etc.) and brand names are both matched.
"""
import re
import json
from pathlib import Path
import pandas as pd

WORK = Path("/sessions/sleepy-gifted-ptolemy/work")
POSTS_P = WORK / "data" / "posts_60d.parquet"
ARCHIVE_P = WORK / "data" / "truth_archive.json"

# Strip HTML for safety, even though content is usually clean
def strip_html(s):
    if not isinstance(s, str): return ""
    return re.sub(r"<[^>]+>", " ", s)

# Regex patterns (compile once, case-insensitive)
# Keep these relatively strict to minimise false positives
PATTERNS = {
    "musk_tesla": re.compile(
        r"\b(?:elon(?:\s+musk)?|musk|tesla|\$tsla|xai|spacex|neuralink|starlink)\b",
        re.IGNORECASE,
    ),
    # Big Tech: only unambiguous company names or tickers; Apple News is excluded
    # (Apple News posts are about media bias, not about Apple the company)
    "big_tech": re.compile(
        r"\b(?:"
        r"apple\s+inc|iphone|ipad|tim\s+cook|"
        r"microsoft|satya\s+nadella|"
        r"alphabet\s+inc|sundar\s+pichai|"
        r"meta\s+platforms|zuckerberg|"
        r"nvidia|jensen\s+huang|"
        r"jeff\s+bezos|andy\s+jassy|"
        r"\$aapl|\$msft|\$googl?|\$meta|\$nvda|\$amzn"
        r")\b",
        re.IGNORECASE,
    ),
    # Big Oil: only unambiguous names / tickered symbols
    "big_oil_companies": re.compile(
        r"\b(?:"
        r"exxon(?:mobil)?|chevron|conocophillips|british\s+petroleum|big\s+oil|"
        r"\$xom|\$cvx|\$cop|\$bp"
        r")\b",
        re.IGNORECASE,
    ),
    # Big Banks: only unambiguous full names or tickered symbols (not bare letters)
    "big_banks": re.compile(
        r"\b(?:"
        r"jpmorgan|jp\s+morgan|jamie\s+dimon|"
        r"goldman\s+sachs|david\s+solomon|"
        r"bank\s+of\s+america|brian\s+moynihan|"
        r"citigroup|citibank|jane\s+fraser|"
        r"wells\s+fargo|morgan\s+stanley|"
        r"\$jpm|\$gs\b|\$bac\b|\$wfc|\$ms\b"
        r")\b",
        re.IGNORECASE,
    ),
}

# Load posts
posts = pd.read_parquet(POSTS_P)
print(f"Loaded {len(posts)} posts from {POSTS_P}")
print(f"Existing topic cols: {[c for c in posts.columns if c.startswith('topic_')]}")

# Apply each pattern to the content column
for name, pat in PATTERNS.items():
    col = f"topic_{name}"
    text = posts["content"].astype(str).apply(strip_html)
    posts[col] = text.str.contains(pat, regex=True, na=False)
    n = int(posts[col].sum())
    print(f"  {col:28s}  matches: {n}")

# Sanity check: show a handful of matches for each new topic
print("\n=== Sample matches ===")
for name in PATTERNS:
    col = f"topic_{name}"
    hits = posts[posts[col]].head(3)
    print(f"\n{col}:")
    for _, r in hits.iterrows():
        snippet = strip_html(r["content"])[:160].replace("\n", " ")
        print(f"  [{r['created_at']}] {snippet}...")

# Save back
posts.to_parquet(POSTS_P)
print(f"\nSaved {POSTS_P} with new topic columns.")

# Emit summary JSON for memo / report
summary = {
    "source": str(POSTS_P),
    "overlap_posts_total": int(len(posts)),
    "topic_counts_in_overlap": {
        f"topic_{k}": int(posts[f"topic_{k}"].sum()) for k in PATTERNS
    },
    "topic_counts_legacy": {
        c: int(posts[c].sum()) for c in [
            "topic_tariff_trade", "topic_fed_rates", "topic_china",
            "topic_iran_military", "topic_energy_oil", "topic_market_economy",
            "topic_crypto", "topic_djt_media",
        ] if c in posts.columns
    },
}
with (WORK / "data" / "friends_topic_counts.json").open("w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary: {WORK/'data/friends_topic_counts.json'}")
