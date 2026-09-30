#!/usr/bin/env bash
# 在 WSL 的 ~/pfr-venv 里跑全部单元测试（含需要 numpy 的采样测试）。
set -euo pipefail
cd "$(dirname "$0")/.."
"${PFR_PYTHON:-$HOME/pfr-venv/bin/python}" -m pytest tests -q -rs "$@"
