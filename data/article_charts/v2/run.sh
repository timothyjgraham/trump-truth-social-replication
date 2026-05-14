#!/usr/bin/env bash
# Reproduce the full v2 chart-data bundle end-to-end.
#
# Assumes:
#   - Python 3.10+
#   - The trump-truth-social-replication repo is at ../trump-truth-social-replication/
#     (use --repo on each script to override)
#   - pip install -r requirements.txt has been run
set -euo pipefail

cd "$(dirname "$0")"

echo "=== [1/3] build.py ==="
python3 build.py "$@"

echo ""
echo "=== [2/3] render_previews.py ==="
python3 render_previews.py

echo ""
echo "=== [3/3] verify.py ==="
python3 verify.py "$@"

echo ""
echo "Done. Bundle in: $(pwd)"
