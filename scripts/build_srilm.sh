#!/usr/bin/env bash
# 编译 SRILM 到 ~/tools/srilm。官网（speech.sri.com）要填表下载，这里用 BitSpeech 在 GitHub 上的镜像。
# 只编 misc、dstruct、lm 三个模块（ngram-count 与 ngram 都在 lm 里），不链接 Tcl。
# 输出同时写入 results/logs/build_srilm_*.log，编译记录是报告「环境」一节的素材。
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="$PWD/results/logs/build_srilm_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results/logs
ROOT="$HOME/tools/srilm"
MT=i686-m64
{
  echo "== 依赖"
  # SRILM 的部分辅助脚本是 csh 脚本，gawk 已有
  command -v tcsh >/dev/null || sudo apt-get install -y tcsh
  gcc --version | head -1
  echo "== 源码"
  if [ ! -d "$ROOT/.git" ]; then
    git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/BitSpeech/SRILM.git "$ROOT"
  fi
  cd "$ROOT"
  git log -1 --format='commit %H (%cd)'
  cat RELEASE 2>/dev/null || true
  echo "== 编译"
  export SRILM="$ROOT"
  time make World SRILM="$ROOT" MACHINE_TYPE=$MT MODULES="misc dstruct lm" \
    NO_TCL=X TCL_INCLUDE= TCL_LIBRARY= -j8
  echo "== 产物"
  ls -la "bin/$MT" | head -40
  "bin/$MT/ngram-count" -version 2>&1 | head -5 || true
} 2>&1 | tee "$LOG"
