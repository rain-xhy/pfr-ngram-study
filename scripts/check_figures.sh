#!/usr/bin/env bash
# 核对图表：中文字体是否加载、每张图是否有内容。见 scripts/check_figures.py。
set -euo pipefail
cd "$(dirname "$0")/.."
"${PFR_PYTHON:-$HOME/pfr-venv/bin/python}" scripts/check_figures.py "$@"
