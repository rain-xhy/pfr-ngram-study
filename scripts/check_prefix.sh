#!/usr/bin/env bash
# 核对 E10 各个开头的每个词在训练集里是否出现过。见 scripts/check_prefix.py。
set -euo pipefail
cd "$(dirname "$0")/.."
"${PFR_PYTHON:-$HOME/pfr-venv/bin/python}" scripts/check_prefix.py "$@"
