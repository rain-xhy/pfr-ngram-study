#!/usr/bin/env bash
# 解压 1998 年 1-6 月 PFR 语料，写出来源与每个文件的 SHA256。
#
# 压缩包来自 https://github.com/chenhui-bupt/PeopleDaily1998 （master 分支，最后提交 2018-11-04 "upload"），
# 在 Windows 上下载到 corpus/raw/pfr_1998H1/199801.zip：
#   Invoke-WebRequest https://raw.githubusercontent.com/chenhui-bupt/PeopleDaily1998/master/199801.zip -OutFile ...
# （WSL 里 git clone GitHub 时常超时，所以下载放在 Windows 这一侧。）
# 包里除 6 个月的语料外还有 shengming.doc：人民日报社新闻信息中心、北京大学计算语言学研究所、
# 富士通研究开发中心有限公司 2001 年 4 月的声明。
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PFR_PYTHON:-$HOME/pfr-venv/bin/python}"
DEST="corpus/raw/pfr_1998H1"
ZIP="$DEST/199801.zip"
EXPECTED="17474bbf2b360921f2eae0089fd83e22601e5a0649943c3f59403650997b4ed2"
[ -f "$ZIP" ] || { echo "缺少 $ZIP，先按上面的命令下载"; exit 1; }
actual=$(sha256sum "$ZIP" | cut -d' ' -f1)
[ "$actual" = "$EXPECTED" ] || { echo "压缩包 SHA256 不符：$actual"; exit 1; }
"$PY" - "$ZIP" "$DEST" <<'PY'
import sys, zipfile
from pathlib import Path
z, dest = zipfile.ZipFile(sys.argv[1]), Path(sys.argv[2])
for info in z.infolist():
    name = Path(info.filename).name
    if info.is_dir() or info.filename.startswith("__MACOSX") or not name:
        continue
    (dest / name).write_bytes(z.read(info))
    print(f"  {name}  {info.file_size / 2**20:.1f} MiB")
PY
{
  echo "来源：https://github.com/chenhui-bupt/PeopleDaily1998 （master 分支，最后提交 2018-11-04）"
  echo "压缩包：199801.zip，SHA256 $EXPECTED"
  echo "取得日期：$(date +%F)"
  echo "说明：包内 shengming.doc 是 PFR 语料库三方 2001 年 4 月的声明，当时免费公开的是 1 月份语料；"
  echo "      2-6 月的公开与授权情况声明中没有写。"
  echo
  (cd "$DEST" && sha256sum 1998*.txt shengming.doc)
} > "$DEST/SOURCE.txt"
cat "$DEST/SOURCE.txt"
