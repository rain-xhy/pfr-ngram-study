#!/usr/bin/env bash
# 在 WSL 里跑补充实验 E14–E19（src/ext_runner.py），终端输出同时写入 results/logs/。
# 用法：bash scripts/run_ext_wsl.sh [E14 E15 ... | 完整实验名] [--force]；不带参数时全部执行，已有结果的跳过。
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PFR_PYTHON:-$HOME/pfr-venv/bin/python}"
CORPUS="${PFR_CORPUS:-pfr6}"
mkdir -p results/logs
LOG="results/logs/ext_${CORPUS}_$(date +%Y%m%d_%H%M%S).log"
echo "语料：$CORPUS；日志：$LOG"
PFR_CORPUS="$CORPUS" PYTHONUNBUFFERED=1 "$PY" -m src.ext_runner "$@" 2>&1 | tee "$LOG"
