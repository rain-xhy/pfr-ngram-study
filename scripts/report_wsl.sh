#!/usr/bin/env bash
# 由 results/<语料>/runs/*.json 生成该语料的 tables.md、metrics.csv 与 figures/。
# 语料由环境变量 PFR_CORPUS 选择：pfr6（默认）或 figshare_jan。
set -euo pipefail
cd "$(dirname "$0")/.."
PFR_CORPUS="${PFR_CORPUS:-pfr6}" "${PFR_PYTHON:-$HOME/pfr-venv/bin/python}" -m src.reporting "$@"
