"""核对从北京大学开放研究数据平台下载的人民日报标注语料：编码、格式、日期范围、规模、特殊标注，
并与本实验用的 Figshare 版、chenhui-bupt/LynxPeng 版的 1 月逐段比较。只读不改。

用法（WSL）：~/pfr-venv/bin/python scripts/inspect_pku.py
"""
import collections
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import RAW_CORPUS  # noqa: E402

PKU_DIR = Path(__file__).resolve().parents[2] / "数据"
LYNX_JAN = Path.home() / "peer" / "b" / "data" / "raw" / "199801.txt"
DOCID = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-(\d{3})/m$")


def decode(raw):
    for enc in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("gb18030", errors="replace"), "gb18030（有无法解码的字节）"


def load(path):
    raw = path.read_bytes()
    text, enc = decode(raw)
    return raw, text, enc


def paragraphs(text):
    paras = collections.OrderedDict()
    other = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if DOCID.match(parts[0]):
            paras.setdefault(parts[0], []).append(parts[1:])
        else:
            other.append(line)
    return paras, other


def describe(path):
    raw, text, enc = load(path)
    paras, other = paragraphs(text)
    toks = [t for ps in paras.values() for p in ps for t in p]
    days = sorted({k[:8] for k in paras})
    tags = collections.Counter()
    special = collections.Counter()
    samples = collections.defaultdict(list)
    for t in toks:
        body = t[1:] if t.startswith("[") else t
        word, sep, tag = body.rpartition("/")
        if not sep:
            special["没有斜杠"] += 1
            samples["没有斜杠"].append(t)
            continue
        tags[re.sub(r"\][a-z]+$", "", tag)] += 1
        for name, pat in (("读音注释 {..}", r"\{"), ("/% 附加标记", r"/%$"), ("拼音或注音", r"[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]")):
            if re.search(pat, t):
                special[name] += 1
                if len(samples[name]) < 4:
                    samples[name].append(t)
        if not word:
            special["空词形"] += 1
    print(f"== {path.name}")
    print(f"  大小 {len(raw):,} 字节，SHA256 {hashlib.sha256(raw).hexdigest()[:16]}…，编码 {enc}")
    lines = [l for l in text.splitlines() if l.strip()]
    for l in lines[:3]:
        print("  |", l[:150])
    print(f"  非空行 {len(lines):,}，其中带标准段落编号的 {sum(len(v) for v in paras.values()):,}，"
          f"不带编号的 {len(other):,}")
    if other:
        print("  不带编号的行示例：", other[:2])
    print(f"  日期 {days[0] if days else '无'} ~ {days[-1] if days else '无'}，共 {len(days)} 天；"
          f"词条 {len(toks):,}；词性标记 {len(tags)} 种")
    print(f"  特殊标注：{dict(special) or '无'}")
    for k, v in samples.items():
        print(f"    {k} 示例：{v[:4]}")
    return paras


def compare(name_a, a, name_b, b):
    keys = set(a) | set(b)
    same = sum(1 for k in keys if a.get(k) == b.get(k))
    only_a = sum(1 for k in keys if k not in b)
    only_b = sum(1 for k in keys if k not in a)
    print(f"  {name_a} 对 {name_b}：{len(keys):,} 个段落编号，完全相同 {same:,}，内容不同 "
          f"{len(keys) - same - only_a - only_b:,}，只在前者 {only_a:,}，只在后者 {only_b:,}")


READING = re.compile(r"\{[^}]*\}")


def word_paragraphs(text):
    """只保留词形：去掉读音注释 {..}（先在整行上去，读音里偶尔有空格）、词性、方括号与 ]实体类型。"""
    out = {}
    brackets = 0
    for line in text.splitlines():
        line = READING.sub("", line)
        parts = line.split()
        if not parts or not DOCID.match(parts[0]):
            continue
        words = []
        for t in parts[1:]:
            if t.startswith("["):
                brackets += 1
                t = t[1:]
            w = t.rpartition("/")[0] if "/" in t else t
            if w:
                words.append(w)
        out.setdefault(parts[0], []).append(words)
    return out, brackets


def compare_words(name_a, a, name_b, b):
    keys = sorted(set(a) & set(b))
    same = [k for k in keys if a[k] == b[k]]
    diff = [k for k in keys if a[k] != b[k]]
    ta = sum(len(w) for k in keys for p in a[k] for w in [p])
    tb = sum(len(w) for k in keys for p in b[k] for w in [p])
    print(f"  {name_a} 对 {name_b}（只比词形）：共有 {len(keys):,} 段，词序列完全相同 {len(same):,}，"
          f"不同 {len(diff):,}；共有段落上的词数 {ta:,} 对 {tb:,}")
    for k in diff[:2]:
        wa = " ".join(a[k][0])[:70]
        wb = " ".join(b[k][0])[:70]
        print(f"    {k}\n      前者：{wa}\n      后者：{wb}")


def main():
    files = sorted(p for p in PKU_DIR.iterdir() if p.is_file())
    parsed = {f.name: describe(f) for f in files}
    print("== 只比词形（去掉词性、读音、方括号）")
    words = {}
    for f in files:
        words[f.name], nb = word_paragraphs(load(f)[1])
        print(f"  {f.name}：方括号实体 {nb:,} 个")
    fig_words, nb = word_paragraphs(load(RAW_CORPUS)[1])
    print(f"  Figshare 版：方括号实体 {nb:,} 个")
    for name, w in words.items():
        compare_words(name, w, "Figshare 版", fig_words)
    if len(words) == 2:
        (na, wa), (nb_, wb) = words.items()
        compare_words(na, wa, nb_, wb)
    refs = {"Figshare 版（本实验）": paragraphs(load(RAW_CORPUS)[1])[0]}
    if LYNX_JAN.exists():
        refs["chenhui-bupt / LynxPeng 版"] = paragraphs(load(LYNX_JAN)[1])[0]
    print("== 逐段比较（按段落编号对齐）")
    names = list(parsed)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            compare(a, parsed[a], b, parsed[b])
        for rname, rp in refs.items():
            compare(a, parsed[a], rname, rp)


if __name__ == "__main__":
    main()
