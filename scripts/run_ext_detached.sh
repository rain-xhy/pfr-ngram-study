#!/usr/bin/env bash
# 在 WSL 里后台跑补充实验（与启动它的终端脱离），适合比终端超时更长的运行。
# 用法：bash scripts/run_ext_detached.sh [实验名 ...] [--force]；进度看 results/logs/ext_pfr6_*.log，
# 停止用 bash scripts/stop_wsl.sh。已有结果的实验会跳过，重复启动前先确认没有正在跑的进程。
set -euo pipefail
cd "$(dirname "$0")/.."
if pgrep -f 'src.ext_runner' >/dev/null; then
  echo "已有 ext_runner 在跑："
  pgrep -af 'src.ext_runner'
  exit 1
fi
nohup setsid bash scripts/run_ext_wsl.sh "$@" >/dev/null 2>&1 < /dev/null &
sleep 2
pgrep -af 'src.ext_runner' || echo "没有启动成功"
ls -t results/logs/ext_* | head -1
