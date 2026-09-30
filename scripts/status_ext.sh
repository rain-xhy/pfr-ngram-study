#!/usr/bin/env bash
# 查看补充实验 E14–E20 的进度：进程是否在跑、已有的结果文件、最新日志的末尾。只读，不改动任何东西。
set -uo pipefail
cd "$(dirname "$0")/.."
CORPUS="${PFR_CORPUS:-pfr6}"
echo "== 进程"
pgrep -af 'src.ext_runner|src.srilm_runner|src.backoff_lab|lmplz|build_binary|ngram-count' || echo "  没有实验进程"
echo "== 结果"
ls -la results/$CORPUS/runs/E1[4-9]_*.json results/$CORPUS/runs/E2[0-9]_*.json 2>/dev/null || echo "  没有补充实验的结果"
echo "== 最新日志"
LOG=$(ls -t results/logs/ext_* 2>/dev/null | head -1)
echo "$LOG"
[ -n "$LOG" ] && grep -v -E '^(Loading|Reading|----|\*\*\*)' "$LOG" | tail -n 30
echo "== 内存"
free -m | head -2
uptime
