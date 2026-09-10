#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi
echo "==> Generating synthetic payment graph..."
"$PYTHON" scripts/generate_data.py
echo "==> Training + evaluating..."
"$PYTHON" scripts/train_and_evaluate.py
echo "==> Done. Metrics in reports/metrics.json"
echo "Launch app: streamlit run app/streamlit_app.py"
