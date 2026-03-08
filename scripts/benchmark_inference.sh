#!/usr/bin/env bash
set -euo pipefail

# Wrapper around tools/benchmark_inference.py
# Pass all arguments through.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

python tools/benchmark_inference.py "$@"
