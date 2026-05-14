#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== [1/3] build.py ==="
python3 build.py "$@"

echo ""
echo "=== [2/3] render.py ==="
python3 render.py

echo ""
echo "=== [3/3] verify.py ==="
python3 verify.py "$@"

echo ""
echo "Done. Bundle in: $(pwd)"
