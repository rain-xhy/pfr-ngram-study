"""换语料能不能让模型变好：同一批训练、测试文章（主划分的文章编号），比较四种训练数据下的五元模型。

A 现在的 Figshare 1 月；B 北大平台的 1998-01-2003版-带音；C chenhui-bupt 版的 1 月；
D chenhui-bupt 版的 1 月训练文章 + 2–6 月全部。C 与 D 用同一份测试集，差别只在训练数据多少。
不同版本的分词不同，每词困惑度不能直接比，另给每字比特数（总比特 / 测试集字数 + 句数）。
另统计测试集里 10 词以上的句子有多少在训练集里原样出现过（泄漏检查）。
用法（WSL）：~/pfr-venv/bin/python scripts/compare_data_gain.py
"""
import math
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as pfr_data  # noqa: E402
from src import kenlm_wrapper as kw  # noqa: E402
from src.common import PROCESSED_DIR, WORK_DIR, read_sentences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PKU = ROOT.parent / "数据" / "1998-01-2003版-带音.txt"
CHENHUI = Path.home() / "peer" / "b" / "data" / "raw"
SPLIT = PROCESSED_DIR / "splits_main"
READING = re.compile(r"\{[^}]*\}")
DOCID = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-\d{3}/m$")


def norm_token(t):
    if t.startswith("["):
        t = t[1:]
    if t.endswith("/%"):
        t = t[:-2]
    word, sep, tag = t.rpartition("/")
    if not sep:
        return t, None
    tag = re.sub(r"\][a-zA-Z]+$", "", tag).split("!")[0]
    return word, ("w" if tag.startswith("w") else tag)  # 北大 2003 版的标点细分为 wj、wd 等


def sentences(path, encoding):
    """逐句产出 (文章编号, 词列表)：先在整行上去掉读音 {..}，再去词性、方括号，按句末标点切句。"""
    for line in path.read_text(encoding=encoding).splitlines():
        parts = READING.sub("", line).split()
        if not parts:
            continue
        m = DOCID.match(parts[0])
        if not m:
            continue
        toks = [(w, tg) for w, tg in (norm_token(t) for t in parts[1:]) if w]
        for sent in pfr_data._segment(toks):
            yield "-".join(m.groups()), [w for w, _ in sent]


def write_train(path, sents):
    words, long_hashes = 0, set()
    with open(path, "w", encoding="utf-8") as f:
        for s in sents:
            f.write(" ".join(s) + "\n")
            words += len(s)
            if len(s) >= 10:
                long_hashes.add(hash(tuple(s)))
    return words, long_hashes


def run(label, train_sents, test, work):
    path = work / "train.txt"
    words, hashes = write_train(path, train_sents)
    arpa = work / "m5.arpa"
    kw.train(path, arpa, 5, memory="512M", temp_dir=work)
    ev = kw.evaluate(kw.load(arpa), test)
    chars = sum(len(w) for s in test for w in s) + len(test)
    bpc = -ev["log10_prob"] * math.log2(10) / chars
    long_test = [s for s in test if len(s) >= 10]
    leak = sum(hash(tuple(s)) in hashes for s in long_test)
    print(f"{label}：训练 {words:,} 词；测试 {sum(len(s) for s in test):,} 词；未登录词 {ev['oov_rate']:.2%}；"
          f"五元困惑度 {ev['ppl_including_oov']:.2f}；每字 {bpc:.3f} 比特；"
          f"测试集长句在训练集原样出现 {leak}/{len(long_test)}", flush=True)
    path.unlink()
    arpa.unlink()


def main():
    work = WORK_DIR / "compare_data_gain"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    sets = {n: set((SPLIT / f"{n}_articles.txt").read_text(encoding="utf-8").split()) for n in ("train", "test")}

    run("A Figshare 1 月（现在）", read_sentences(SPLIT / "train.txt"), read_sentences(SPLIT / "test.txt"), work)

    pku = list(sentences(PKU, "gb18030"))
    run("B 北大 2003 版 1 月", (s for a, s in pku if a in sets["train"]),
        [s for a, s in pku if a in sets["test"]], work)
    del pku

    jan = list(sentences(CHENHUI / "199801.txt", "utf-8-sig"))
    test_c = [s for a, s in jan if a in sets["test"]]
    run("C chenhui 版 1 月", (s for a, s in jan if a in sets["train"]), test_c, work)

    def jan_to_jun():
        yield from (s for a, s in jan if a in sets["train"])
        for f in sorted(CHENHUI.glob("1998*.txt")):
            if f.name != "199801.txt":
                yield from (s for _, s in sentences(f, "utf-8-sig"))

    run("D chenhui 版 1 月训练文章 + 2–6 月", jan_to_jun(), test_c, work)
    shutil.rmtree(work)


if __name__ == "__main__":
    main()
