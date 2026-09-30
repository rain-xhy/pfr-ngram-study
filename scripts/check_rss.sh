#!/usr/bin/env bash
# 核对峰值内存统计的口径，以及剪枝训练的折扣行能否解析。见 scripts/check_rss.py。
set -euo pipefail
cd "$(dirname "$0")/.."
"${PFR_PYTHON:-$HOME/pfr-venv/bin/python}" scripts/check_rss.py "$@"
