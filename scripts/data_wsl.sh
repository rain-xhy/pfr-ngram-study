#!/usr/bin/env bash
# 语料准备：探测、清洗、按文章划分、建词表、规模抽样，结果写到 corpus/processed/<语料>/ 与 results/<语料>/。
# 语料由环境变量 PFR_CORPUS 选择：pfr6（默认，1998 年 1-6 月）或 figshare_jan（1998 年 1 月）。
set -euo pipefail
cd "$(dirname "$0")/.."
PFR_CORPUS="${PFR_CORPUS:-pfr6}" PYTHONUNBUFFERED=1 "${PFR_PYTHON:-$HOME/pfr-venv/bin/python}" -m src.data "$@"
