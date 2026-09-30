"""比较几份人民日报 1998 标注语料是不是同一个版本。

对给定目录里的每个 1998MM.txt：段落数、日期范围、词条数、特殊标注；
再拿它的 1 月与本实验用的 Figshare 版、LynxPeng 仓库里的 1 月逐段比较（按段落编号对齐）。
用法（WSL）：~/pfr-venv/bin/python scripts/compare_versions.py <目录> [<目录> ...]
"""
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import RAW_CORPUS  # noqa: E402

LYNX_JAN = Path.home() / "peer" / "b" / "data" / "raw" / "199801.txt"
DOCID = re.compile(r"^\d{8}-\d{2}-\d{3}-\d{3}/m$")


def read_text(path):
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8（有无法解码的字节）"


def paragraphs(path):
    text, enc = read_text(path)
    paras = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and DOCID.match(parts[0]):
            paras.setdefault(parts[0], []).append(parts[1:])
    return paras, enc


def summary(path):
    paras, enc = paragraphs(path)
    tokens = sum(len(p) for ps in paras.values() for p in ps)
    special = collections.Counter()
    for ps in paras.values():
        for p in ps:
            for t in p:
                if t.endswith("/%"):
                    special["/%"] += 1
                if t.startswith("/"):
                    special["空词形"] += 1
                if "{" in t:
                    special["读音注释"] += 1
    days = sorted({k[:8] for k in paras})
    return {"encoding": enc, "paragraphs": sum(len(v) for v in paras.values()), "tokens": tokens,
            "days": len(days), "range": (days[0], days[-1]) if days else None, "special": dict(special)}


def diff(a_path, b_path):
    a, _ = paragraphs(a_path)
    b, _ = paragraphs(b_path)
    keys = set(a) | set(b)
    same = sum(1 for k in keys if a.get(k) == b.get(k))
    only_a = sum(1 for k in keys if k not in b)
    only_b = sum(1 for k in keys if k not in a)
    return {"ids": len(keys), "identical": same, "different": len(keys) - same - only_a - only_b,
            "only_first": only_a, "only_second": only_b}


def main():
    for d in map(Path, sys.argv[1:]):
        files = sorted(d.rglob("1998[01][0-9].txt"))
        files = [f for f in files if "__MACOSX" not in f.parts]
        print(f"== {d}：{len(files)} 个月份文件")
        total = 0
        for f in files:
            s = summary(f)
            total += s["tokens"]
            print(f"  {f.name}：{s['paragraphs']:,} 段，{s['days']} 天 {s['range']}，{s['tokens']:,} 词条，"
                  f"{s['encoding']}，特殊标注 {s['special'] or '无'}")
        print(f"  合计 {total:,} 词条")
        jan = next((f for f in files if f.name == "199801.txt"), None)
        if jan:
            for label, other in (("Figshare 版（本实验）", RAW_CORPUS), ("LynxPeng 的 1 月", LYNX_JAN)):
                if other.exists():
                    r = diff(jan, other)
                    print(f"  1 月 对比 {label}：{r['ids']:,} 个段落编号，完全相同 {r['identical']:,}，"
                          f"内容不同 {r['different']:,}，只在前者 {r['only_first']:,}，只在后者 {r['only_second']:,}")


if __name__ == "__main__":
    main()
