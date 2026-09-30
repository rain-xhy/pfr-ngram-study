"""探测 LynxPeng 仓库里 1998 年 1–6 月语料的格式，判断能否与本实验的清洗流程衔接。

只读不改。对每个月份：编码、非空行数、日期范围、词数，以及本实验 clean_corpus 不认识的标注
（读音注释 {..}、词后的 /% 附加标记、没有词形的空标注项、实体括号内嵌套）各有多少。
另外核对他的 1 月文件与本实验用的 Figshare 版是否同一版本。
用法（WSL）：~/pfr-venv/bin/python scripts/probe_months.py
"""
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import RAW_CORPUS  # noqa: E402

PEER = Path.home() / "peer" / "b" / "data" / "raw"
DOCID = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-(\d{3})/m$")


def probe(path, encoding):
    lines = [l for l in path.read_text(encoding=encoding).splitlines() if l.strip()]
    dates, tokens = set(), 0
    special = collections.Counter()
    bad_docid = 0
    for line in lines:
        parts = line.split()
        m = DOCID.match(parts[0])
        if not m:
            bad_docid += 1
            continue
        dates.add(m.group(1))
        for tok in parts[1:]:
            tokens += 1
            if "{" in tok:
                special["读音注释 {..}"] += 1
            if tok.endswith("/%"):
                special["/% 附加标记"] += 1
            if tok.startswith("/"):
                special["空词形"] += 1
            if tok.count("[") > 1:
                special["一个词前多个 ["] += 1
            if re.search(r"\][a-z]+\]", tok):
                special["连续闭括号（嵌套实体）"] += 1
    return {"lines": len(lines), "bad_docid": bad_docid, "dates": (min(dates), max(dates)) if dates else None,
            "days": len(dates), "tokens": tokens, "special": dict(special)}


def main():
    if not PEER.exists():
        print("找不到", PEER)
        return
    total = 0
    for f in sorted(PEER.glob("1998*.txt")):
        r = probe(f, "utf-8-sig")
        total += r["tokens"]
        print(f"{f.name}：{r['lines']:,} 段，{r['days']} 天 {r['dates']}，{r['tokens']:,} 个词条，"
              f"编号格式不符 {r['bad_docid']}，特殊标注 {r['special'] or '无'}")
    print(f"六个月合计 {total:,} 个词条")
    ours = probe(RAW_CORPUS, "gb18030")
    print(f"本实验 Figshare 版：{ours['lines']:,} 段，{ours['tokens']:,} 个词条，特殊标注 {ours['special'] or '无'}")


if __name__ == "__main__":
    main()
