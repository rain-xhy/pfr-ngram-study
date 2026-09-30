#!/usr/bin/env bash
# 核对从北大开放研究数据平台下载的语料，并与现有两个版本比较。见 scripts/inspect_pku.py。
set -euo pipefail
cd "$(dirname "$0")/.."
"${PFR_PYTHON:-$HOME/pfr-venv/bin/python}" scripts/inspect_pku.py "$@"
