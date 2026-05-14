#!/usr/bin/env python3
"""
render.py - triangular co-occurrence heatmap matrix.

Lower-triangle matrix; words listed top-down on the left and angled at top.
"Word (N posts)" labels show post-level frequency for each word. Pinkish-red
colour scale shows pair co-occurrence intensity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR  = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR
DEFAULT_OUT  = SCRIPT_DIR / "previews"


def render_matrix(data: Path, out: Path) -> None:
    nodes  = pd.read_csv(data / "word_nodes.csv")
    matrix = pd.read_csv(data / "word_matrix.csv", index_col=0)

    word_order = nodes["word"].tolist()
    matrix = matrix.loc[word_order, word_order]
    n = len(word_order)

    # Mask the upper triangle (including diagonal) so only the lower triangle
    # shows pair co-occurrences. Diagonal would be the post count which is
    # already on the row label.
    M = matrix.values.astype(float)
    mask = np.triu(np.ones_like(M, dtype=bool), k=0)
    M_masked = np.ma.array(M, mask=mask)

    # Colour scale: light pink to dark red. Use the off-diagonal max as the
    # ceiling so the diagonal (which we masked anyway) doesn't blow out the
    # scale.
    off_diag = M.copy()
    np.fill_diagonal(off_diag, 0)
    vmax = off_diag.max() if off_diag.max() > 0 else 1.0

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pink_red",
        ["#fcebec", "#f4b8bb", "#e7777b", "#d44349", "#a01e23"],
        N=256)
    cmap.set_bad(color="white")

    # Figure sized to comfortably fit n words plus angled top labels
    fig, ax = plt.subplots(figsize=(11, 11), facecolor="white")
    ax.set_facecolor("white")

    # Draw cells. Use pcolormesh with cell-edge coordinates so each cell is
    # a clean square; aspect = equal so cells stay square.
    x = np.arange(n + 1)
    y = np.arange(n + 1)
    pc = ax.pcolormesh(x, y, M_masked, cmap=cmap, vmin=0, vmax=vmax,
                        edgecolors="white", linewidth=0.6)

    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at top

    # Strip default ticks/spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Row labels on the left: "word"
    for i, w in enumerate(word_order):
        ax.text(-0.3, i + 0.5, w, ha="right", va="center",
                fontsize=10, color="#222")

    # Top angled labels: "word (N posts)"
    # Place each at the column position; angle them up-right
    for j, w in enumerate(word_order):
        post_count = int(nodes.iloc[j]["post_count"])
        is_top = (j == 0)
        ax.text(j + 0.5, -0.4,
                f"{w} ({post_count} posts)",
                ha="left", va="bottom",
                rotation=40, rotation_mode="anchor",
                fontsize=9 if not is_top else 10,
                fontweight="bold" if is_top else "normal",
                color="#222")

    # Title
    fig.text(0.5, 0.93,
              "Word co-occurrence in oil / Iran-themed Trump posts",
              ha="center", color="#222", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.905,
              "Cells show how often each pair of words appears together in the same post. "
              "121 posts in the universe (containing explicit oil / Iran / Hormuz / crude / "
              "Venezuela / etc. terms).",
              ha="center", color="#555", fontsize=9, style="italic")

    # Footnote
    fig.text(0.5, 0.04,
              "Self-referential terms (Trump, Donald, President, DJT) and content-light "
              "fillers excluded so the matrix surfaces substantive content vocabulary. "
              "Darker cells = higher co-occurrence in the same post.",
              ha="center", color="#666", fontsize=8.5, style="italic", wrap=True)

    # Squeeze axis area to leave room for top labels and side labels
    ax.set_xlim(-0.5, n + 0.2)
    ax.set_ylim(n + 0.2, -3.5)  # negative top makes room for angled labels

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(out / "word_matrix.png", dpi=160, bbox_inches="tight",
                 facecolor="white")
    plt.close()
    print(f"  saved: word_matrix.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out",  type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = args.data.resolve()
    out  = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {data}")
    print(f"Writing to:   {out}\n")

    render_matrix(data, out)


if __name__ == "__main__":
    main()
