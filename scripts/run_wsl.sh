#!/usr/bin/env bash
# 在 WSL 里跑实验：用 ~/pfr-venv 的 Python（装有 KenLM 绑定），模型文件写到 ~/pfr-work/<语料>/。
# 用法：bash scripts/run_wsl.sh [phase1|phase2|phase3|phase4|实验名 ...] [--force]
# 语料由环境变量 PFR_CORPUS 选择：pfr6（默认，1998 年 1-6 月）或 figshare_jan（1998 年 1 月）。
# 不带参数时按 phase1 → phase4 全部执行；终端输出同时写入 results/logs/。
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PFR_PYTHON:-$HOME/pfr-venv/bin/python}"
CORPUS="${PFR_CORPUS:-pfr6}"
mkdir -p results/logs
LOG="results/logs/run_${CORPUS}_$(date +%Y%m%d_%H%M%S).log"
echo "语料：$CORPUS；日志：$LOG"
PFR_CORPUS="$CORPUS" PYTHONUNBUFFERED=1 "$PY" -m src.experiment_runner "$@" 2>&1 | tee "$LOG"
