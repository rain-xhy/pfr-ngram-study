#!/usr/bin/env bash
# 比较换不同语料后五元模型的效果。见 scripts/compare_data_gain.py。
set -euo pipefail
cd "$(dirname "$0")/.."
"${PFR_PYTHON:-$HOME/pfr-venv/bin/python}" scripts/compare_data_gain.py "$@"
