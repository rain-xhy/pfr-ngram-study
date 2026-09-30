#!/usr/bin/env bash
# 补充实验（E14–E19）开工前的环境探测：内存、磁盘、已有模型、Python 依赖、SRILM 编译条件与网络。
set -uo pipefail
cd "$(dirname "$0")/.."
echo "== 内存 / 磁盘"
free -m
df -h "$HOME" | tail -1
echo "== 已有模型（pfr6）"
ls -la "$HOME/pfr-work/pfr6/models" 2>/dev/null | head -60
du -sh "$HOME/pfr-work/pfr6" 2>/dev/null
echo "== Python"
PY="${PFR_PYTHON:-$HOME/pfr-venv/bin/python}"
"$PY" -c "import sys, numpy, kenlm; print(sys.version); print('numpy', numpy.__version__)"
"$PY" -c "import matplotlib; print('matplotlib', matplotlib.__version__)" 2>&1 | tail -1
"$PY" -c "import scipy; print('scipy', scipy.__version__)" 2>&1 | tail -1
echo "== 编译工具"
for t in gcc g++ make gawk csh tcsh git curl; do printf '%-6s ' "$t"; command -v "$t" || echo "缺"; done
dpkg -l | grep -E 'tcl-dev|libtcl|liblbfgs' | awk '{print $2, $3}'
echo "== SRILM"
ls -d "$HOME/tools/srilm" 2>/dev/null && ls "$HOME/tools/srilm/bin" 2>/dev/null | head
echo "== 网络"
timeout 20 git ls-remote https://github.com/BitSpeech/SRILM.git HEAD 2>&1 | head -3 || echo "github 不通"
echo "== CPU"
nproc
