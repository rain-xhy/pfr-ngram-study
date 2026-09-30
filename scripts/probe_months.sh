#!/usr/bin/env bash
# 探测 LynxPeng 仓库里 1998 年 1–6 月语料的格式。见 scripts/probe_months.py。
set -euo pipefail
cd "$(dirname "$0")/.."
"${PFR_PYTHON:-$HOME/pfr-venv/bin/python}" scripts/probe_months.py "$@"
