#!/usr/bin/env bash
# 小样本核对 smoothing_lab 的 Modified KN 与 KenLM 三元模型：条目数、折扣、逐位置概率、困惑度。
# 用法：bash scripts/check_kn.sh [训练句数，默认 8000]
set -euo pipefail
cd "$(dirname "$0")/.."
"${PFR_PYTHON:-$HOME/pfr-venv/bin/python}" scripts/check_kn.py "$@"
