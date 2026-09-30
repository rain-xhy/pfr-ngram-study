#!/usr/bin/env bash
# 粗测换成 1998 年 1–6 月语料后训练与生成要多久。见 scripts/probe_scale6.py。
set -euo pipefail
cd "$(dirname "$0")/.."
"${PFR_PYTHON:-$HOME/pfr-venv/bin/python}" scripts/probe_scale6.py "$@"
