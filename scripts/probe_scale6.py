"""粗测换成 1998 年 1–6 月语料后各环节要多久，用来估计重跑全部实验的时间。

用 chenhui-bupt/PeopleDaily1998 版本（与 LynxPeng 仓库里的六个文件逐段相同，本机现成的副本在
~/peer/b/data/raw/）。这里只做粗略清洗（去编号、词性、方括号、/% 标记，一段一行，不切句），
训练三元、五元、六元各一次，再测五元模型的加载时间和对整个词表打分一步的时间。
结束后删掉临时语料和模型。
用法（WSL）：~/pfr-venv/bin/python scripts/probe_scale6.py
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import kenlm_wrapper as kw  # noqa: E402
from src import sampling  # noqa: E402
from src.common import MAIN_PROMPT, WORK_DIR  # noqa: E402

RAW = Path.home() / "peer" / "b" / "data" / "raw"


def clean_token(tok):
    if tok.startswith("["):
        tok = tok[1:]
    if tok.endswith("/%"):
        tok = tok[:-2]
    word, sep, _ = tok.rpartition("/")
    return word if sep else tok


def main():
    work = WORK_DIR / "probe6"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    free = shutil.disk_usage(work).free / 2**30
    print(f"WSL 磁盘剩余 {free:.1f} GiB")
    if free < 6:
        print("磁盘剩余不足 6 GiB，停止")
        return

    t0 = time.perf_counter()
    corpus = work / "corpus.txt"
    tokens, types = 0, set()
    with open(corpus, "w", encoding="utf-8") as out:
        for f in sorted(RAW.glob("1998*.txt")):
            for line in f.read_text(encoding="utf-8-sig").splitlines():
                parts = line.split()
                if not parts:
                    continue
                words = [w for w in (clean_token(t) for t in parts[1:]) if w]
                tokens += len(words)
                types.update(words)
                out.write(" ".join(words) + "\n")
    print(f"粗清洗：{tokens:,} 词，{len(types):,} 个词形，用时 {time.perf_counter() - t0:.1f}s")

    for n in (3, 5, 6):
        r = kw.train(corpus, work / f"m{n}.arpa", n, memory="512M", temp_dir=work)
        cpu = r.get("user_seconds", 0) + r.get("sys_seconds", 0)
        print(f"{n} 元：墙钟 {r['wall_seconds']:.1f}s，CPU {cpu:.1f}s，峰值 {r.get('peak_rss_mb', 0):.0f} MiB，"
              f"ARPA {r['arpa_bytes'] / 2**20:.0f} MiB，条目 {sum(r['ngram_counts']):,}")
        if n != 5:
            (work / f"m{n}.arpa").unlink()

    t0 = time.perf_counter()
    model = kw.load(work / "m5.arpa")
    print(f"五元 ARPA 加载 {time.perf_counter() - t0:.1f}s")
    gen = sampling.Generator(model, sampling.vocab_from_arpa(work / "m5.arpa"))
    state = gen.start(MAIN_PROMPT)
    t0 = time.perf_counter()
    for _ in range(3):
        gen.scores(state)
    step = (time.perf_counter() - t0) / 3
    print(f"生成时对全部 {len(gen.vocab):,} 个候选打分一步 {step:.3f}s（现在的 1 月语料约 0.025s）")
    del model, gen
    shutil.rmtree(work)
    print("临时文件已删除")


if __name__ == "__main__":
    main()
