#!/usr/bin/env bash
# 运行 E20 SRILM 对比实验
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
PY="${PFR_PYTHON:-$HOME/pfr-venv/bin/python}"
CORPUS="${PFR_CORPUS:-pfr6}"
echo "SRILM 路径：$(which ngram-count)"
echo "语料：$CORPUS"
PFR_CORPUS="$CORPUS" "$PY" -m src.srilm_runner
