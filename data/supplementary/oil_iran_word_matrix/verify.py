#!/usr/bin/env python3
"""verify.py - quick consistency checks for the cleaned word matrix bundle."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR
DEFAULT_REPO = (SCRIPT_DIR.parent / "trump-truth-social-replication").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    args = parser.parse_args()
    data = args.data.resolve()
    repo = args.repo.resolve()
    print(f"Verifying: {data}")
    print(f"Source: {repo}\n")

    nodes  = pd.read_csv(data / "word_nodes.csv")
    edges  = pd.read_csv(data / "word_edges.csv")
    matrix = pd.read_csv(data / "word_matrix.csv", index_col=0)

    # Reproduce universe size from source
    posts = pd.read_parquet(repo / "data/raw/posts_60d.parquet")
    OIL_RE = re.compile(
        r"\b(oil|crude|opec|barrel|refinery|gasoline|petroleum|hormuz|saudi|iran|"
        r"venezuela|drill|drilling|frack|pipeline|wti|brent)\b", re.I)
    n_universe = posts["text"].astype(str).apply(lambda t: bool(OIL_RE.search(t))).sum()

    ok, fail = 0, 0
    def check(label, condition, expected="?", actual="?"):
        nonlocal ok, fail
        if condition:
            ok += 1
            print(f"  [OK]   {label}")
        else:
            fail += 1
            print(f"  [FAIL] {label}: expected {expected}, got {actual}")

    print("--- Universe + structure ---")
    check(f"121 posts in oil-term universe", n_universe == 121, "121", str(n_universe))
    check("30 nodes in word list", len(nodes) == 30, "30", str(len(nodes)))
    check("Matrix is 30x30", matrix.shape == (30, 30),
          "(30, 30)", str(matrix.shape))
    check("Nodes & matrix have same word order",
          list(nodes["word"]) == list(matrix.index))

    print("\n--- Self-references excluded ---")
    drops = {"trump", "donald", "president", "djt"}
    found = drops.intersection(set(nodes["word"]))
    check(f"No self-references (trump/donald/president/djt) in matrix",
          len(found) == 0, "{}", str(found))

    print("\n--- Matrix symmetry & weight floor ---")
    M = matrix.values
    import numpy as np
    upper = M[np.triu_indices_from(M, k=1)]
    lower = M[np.tril_indices_from(M, k=-1)]
    upper_sorted = sorted(upper.tolist())
    lower_sorted = sorted(lower.tolist())
    check("Matrix is symmetric (upper == lower)", upper_sorted == lower_sorted)
    check("Diagonal = post counts",
          all(int(matrix.iat[i, i]) == int(nodes.iloc[i]["post_count"]) for i in range(30)))

    print("\n--- Edge consistency ---")
    matrix_pairs = set()
    for i, w_i in enumerate(matrix.index):
        for j, w_j in enumerate(matrix.columns):
            if i < j and matrix.iat[i, j] >= 3:
                # canonicalize pair alphabetically so it matches edges file
                a, b = sorted([w_i, w_j])
                matrix_pairs.add((a, b, int(matrix.iat[i, j])))
    edge_pairs = set()
    for _, e in edges.iterrows():
        a, b = sorted([e["source"], e["target"]])
        edge_pairs.add((a, b, int(e["weight"])))
    check(f"Edges (>=3 weight) match matrix off-diagonal cells",
          matrix_pairs == edge_pairs,
          f"{len(matrix_pairs)} pairs", f"{len(edge_pairs)} pairs")

    print("\n--- Spot checks (key Iran narrative co-occurrences) ---")
    def cell(a, b):
        return int(matrix.at[a, b]) if a in matrix.index and b in matrix.columns else None
    check("iran in node list", "iran" in nodes["word"].values)
    check("hormuz in node list", "hormuz" in nodes["word"].values)
    check("strait in node list", "strait" in nodes["word"].values)
    check("nuclear in node list", "nuclear" in nodes["word"].values)
    if all(w in matrix.index for w in ["iran", "hormuz", "strait", "nuclear"]):
        check("hormuz-strait edge present (~22)",
              cell("hormuz", "strait") >= 15)
        check("iran-hormuz edge present", cell("iran", "hormuz") >= 5)
        check("iran-nuclear edge present", cell("iran", "nuclear") >= 5)

    print()
    print("=" * 60)
    print(f"SUMMARY: {ok} passed, {fail} failed")
    print("=" * 60)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
