#!/usr/bin/env bash
# 停掉 WSL 里还在跑的实验进程（experiment_runner 及其启动的 lmplz / build_binary），并列出停之前的状态。
set -uo pipefail
echo "停止前："
pgrep -af 'src.experiment_runner|src.ext_runner|lmplz|build_binary' || echo "  没有实验进程"
pkill -f 'src.experiment_runner' 2>/dev/null
pkill -f 'src.ext_runner' 2>/dev/null
pkill -f 'src.backoff_lab' 2>/dev/null
pkill -f 'lmplz' 2>/dev/null
pkill -f 'build_binary' 2>/dev/null
sleep 1
echo "停止后："
pgrep -af 'src.experiment_runner|src.ext_runner|lmplz|build_binary' || echo "  没有实验进程"
