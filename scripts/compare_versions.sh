#!/usr/bin/env bash
# 取 chenhui-bupt/PeopleDaily1998 的压缩包，解到临时目录，与本实验的 Figshare 版、LynxPeng 的版本比较，看完即删。
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PFR_PYTHON:-$HOME/pfr-venv/bin/python}"
W="$HOME/pfr-work/compare_versions"
rm -rf "$W" && mkdir -p "$W"
for try in 1 2 3; do
  timeout 300 git clone -q --depth 1 --filter=blob:none --no-checkout \
    https://github.com/chenhui-bupt/PeopleDaily1998.git "$W/repo" 2>/dev/null && break
  rm -rf "$W/repo"; sleep 5
done
[ -d "$W/repo" ] || { echo "克隆失败"; rm -rf "$W"; exit 0; }
for try in 1 2 3; do git -C "$W/repo" checkout -q HEAD -- 199801.zip 2>/dev/null && break; sleep 3; done
"$PY" - "$W/repo/199801.zip" "$W/chenhui" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = [i for i in z.infolist() if not i.filename.startswith("__MACOSX") and not i.is_dir()]
print("压缩包内容：" + "，".join(f"{i.filename} {i.file_size / 2**20:.1f} MiB" for i in names))
for i in names:
    z.extract(i, sys.argv[2])
PY
"$PY" scripts/compare_versions.py "$W/chenhui" "$HOME/peer/b/data/raw"
rm -rf "$W"
