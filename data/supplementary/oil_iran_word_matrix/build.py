#!/usr/bin/env python3
"""
build.py - cleaned word co-occurrence matrix for the oil/Iran post universe.

Loads posts that contain explicit oil / Iran / Hormuz / etc. terms (~121 posts),
tokenizes them, filters out self-references and content-light fillers, and
emits a co-occurrence matrix suitable for rendering as a triangular heatmap.

Outputs:
  word_nodes.csv          - top 30 substantive words with post counts
  word_edges.csv          - co-occurrence pairs with weights
  word_matrix.csv         - 30x30 symmetric co-occurrence matrix
  word_network.json       - D3-friendly nested form

Path: assumes ../trump-truth-social-replication/ alongside this folder.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
DEFAULT_REPO = (SCRIPT_DIR.parent / "trump-truth-social-replication").resolve()

# Strict universe: posts containing at least one explicit oil / Iran term
OIL_TERMS_RE = re.compile(
    r"\b(oil|crude|opec|barrel|refinery|gasoline|petroleum|hormuz|saudi|iran|"
    r"venezuela|drill|drilling|frack|pipeline|wti|brent)\b", re.I)

# Generic English stopwords + Truth-Social-specific noise
STOPWORDS = {
    "the","a","an","and","or","but","if","of","on","in","to","for","with","at",
    "by","from","up","down","is","are","was","were","be","been","being","have",
    "has","had","do","does","did","will","would","should","could","may","might",
    "must","can","this","that","these","those","i","you","he","she","it","we",
    "they","me","him","her","us","them","my","your","his","its","our","their",
    "as","not","no","so","than","too","very","just","also","then","there",
    "here","what","who","whom","which","why","how","because","over","under",
    "between","through","into","out","onto","upon","such",
    "amp","like","via","off","still","always","never","ever","quot","apos",
    "much","more","most","many","some","any","all","both","each",
    "other","another","new","old","one","two","three","first","last","next",
    "https","http","com","www","html","co",
    "great","good","big","huge","said","says","say","get","got","make","made",
    "going","go","come","came","see","know","want","need","let","take",
    "thank","thanks","tonight","today","yesterday","tomorrow","really",
    "people","time","year","years","day","days","way","things","thing",
    "everybody","everyone","nobody","anyone","someone","somebody","everything",
    "nothing","anything","something","please","look","looks","looking",
}

# Self-reference vocabulary - Trump's signature sign-offs and naming patterns
# ("President Donald J. Trump", "DJT" etc.). These dominate the matrix as a
# block in the top-left when included; dropping them surfaces the substantive
# Iran/oil narrative beneath.
SELF_REFERENCES = {
    "trump", "donald", "president", "djt",
}

# Filler / boilerplate that crowds the matrix without carrying meaning
# ("thank you for your attention to this matter", various preposition-y words)
BOILERPLATE = {
    "matter", "attention", "before",
    "now", "again", "far", "about", "only", "else", "iran's",
    "left",         # ambiguous: usually Trump's "Radical Left" framing
    "well", "back", "completely", "highly", "against", "help", "immediately",
}

DROP_WORDS = STOPWORDS | SELF_REFERENCES | BOILERPLATE

TOP_K           = 30
MIN_EDGE_WEIGHT = 3   # drop very-weak edges from the network output


def load_posts(repo: Path) -> pd.DataFrame:
    p = pd.read_parquet(repo / "data/raw/posts_60d.parquet")
    p["created_at"] = pd.to_datetime(p["created_at"], utc=True)
    return p


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"&[a-z]+;",      " ", text)
    text = re.sub(r"<[^>]+>",       " ", text)
    return [t for t in re.findall(r"[A-Za-z']{3,}", text.lower())
            if t not in DROP_WORDS and len(t) >= 3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--out",  type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out  = args.out.resolve()
    if not repo.exists():
        sys.exit(f"ERROR: repo not found at {repo}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Repo:   {repo}")
    print(f"Output: {out}\n")

    posts = load_posts(repo)
    universe = posts[posts["text"].astype(str).apply(
        lambda t: bool(OIL_TERMS_RE.search(t)))].copy()
    print(f"Universe: {len(universe)} posts containing explicit oil/Iran/Hormuz terms")

    # Per-post token sets, sorted for determinism
    word_lists = [sorted(set(tokenize(t))) for t in universe["text"].astype(str)]

    # Post-level word counts
    word_counts = Counter()
    for ws in word_lists:
        word_counts.update(ws)

    # Top-K substantive words (deterministic tie-break: highest count, then alpha)
    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    top_words = [w for w, _ in sorted_words[:TOP_K]]
    top_set   = set(top_words)
    print(f"Top {TOP_K} substantive words selected (self-references + fillers dropped)")

    # Co-occurrence: pairs within the same post, restricted to top-K
    edge_counts = Counter()
    for ws in word_lists:
        present = [w for w in ws if w in top_set]
        for a, b in combinations(sorted(present), 2):
            edge_counts[(a, b)] += 1

    # Nodes (sorted by frequency, then alphabetical for tie stability)
    nodes_df = pd.DataFrame([
        {"word": w, "post_count": int(word_counts[w])} for w in top_words
    ]).sort_values(["post_count", "word"], ascending=[False, True]).reset_index(drop=True)
    nodes_df["rank"] = nodes_df.index + 1
    nodes_df.to_csv(out / "word_nodes.csv", index=False)

    # Edges (sorted deterministic)
    edge_rows = sorted(
        [(a, b, w) for (a, b), w in edge_counts.items() if w >= MIN_EDGE_WEIGHT],
        key=lambda r: (-r[2], r[0], r[1]))
    edges_df = pd.DataFrame(edge_rows, columns=["source", "target", "weight"])
    edges_df.to_csv(out / "word_edges.csv", index=False)

    # Symmetric co-occurrence matrix (rows = top words, cols = top words)
    word_order = nodes_df["word"].tolist()
    matrix = pd.DataFrame(0, index=word_order, columns=word_order, dtype=int)
    for a, b, w in edge_rows:
        matrix.at[a, b] = w
        matrix.at[b, a] = w
    # Diagonal = self-count (post_count of that word)
    for w in word_order:
        matrix.at[w, w] = int(word_counts[w])
    matrix.to_csv(out / "word_matrix.csv")

    # JSON for D3 etc.
    payload = {
        "universe":           "posts containing explicit oil/Iran/Hormuz/crude/etc. terms",
        "n_posts":            int(len(universe)),
        "n_words":            len(word_order),
        "min_edge_weight":    MIN_EDGE_WEIGHT,
        "filtered_out": {
            "self_references": sorted(SELF_REFERENCES),
            "boilerplate":     sorted(BOILERPLATE - {"matter", "attention", "before"}),
            "stopwords_total": len(STOPWORDS),
        },
        "nodes": nodes_df.to_dict(orient="records"),
        "edges": edges_df.to_dict(orient="records"),
        "matrix": {
            "row_order": word_order,
            "values":    matrix.values.tolist(),
        },
    }
    with open(out / "word_network.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  word_nodes.csv:    {len(nodes_df)} words")
    print(f"  word_edges.csv:    {len(edges_df)} edges (min weight {MIN_EDGE_WEIGHT})")
    print(f"  word_matrix.csv:   {len(word_order)}x{len(word_order)} matrix")
    print(f"  word_network.json: D3-friendly bundle")
    print(f"\nTop 10 words:")
    for _, r in nodes_df.head(10).iterrows():
        print(f"    {int(r['rank']):>2}. {r['word']:<14}  ({int(r['post_count']):>3} posts)")


if __name__ == "__main__":
    main()
