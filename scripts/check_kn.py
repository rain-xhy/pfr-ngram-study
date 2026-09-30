"""核对 smoothing_lab 的 Modified KN 与 KenLM 的三元模型：条目数、折扣、逐位置概率、困惑度。

用训练集前 N 句训练（默认 8000），在测试集前 300 句上比较。E1 在完整训练集上做同样的核对，
这里先用小样本快速确认实现没有写错，再去跑耗时的全部实验。
用法（WSL）：~/pfr-venv/bin/python scripts/check_kn.py [N]
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import kenlm_wrapper as kw  # noqa: E402
from src import sampling  # noqa: E402
from src import smoothing_lab as sl  # noqa: E402
from src.common import MAIN_PROMPT, PROCESSED_DIR, WORK_DIR, read_sentences  # noqa: E402


def main(n_train=8000):
    train = read_sentences(PROCESSED_DIR / "splits_main" / "train.txt")[:n_train]
    test = read_sentences(PROCESSED_DIR / "splits_main" / "test.txt")[:300]
    work = WORK_DIR / "check_kn"
    work.mkdir(parents=True, exist_ok=True)
    corpus = work / "train.txt"
    corpus.write_text("\n".join(" ".join(s) for s in train) + "\n", encoding="utf-8")
    res = kw.train(corpus, work / "m3.arpa", 3, memory="200M", temp_dir=work)
    counts = sl.TrigramCounts(train)
    ours = [len(counts.c1) + 2, len(counts.a2), len(counts.c3)]
    print(f"训练 {len(train)} 句；KenLM 条目数 {res['ngram_counts']}，本模块 {ours}")
    for d, name, D in zip(res["discounts"], ("一元", "二元", "三元"), (counts.D1, counts.D2, counts.D3)):
        print(f"  {name}折扣 KenLM {d['D1']:.6f} {d['D2']:.6f} {d['D3+']:.6f} | "
              f"本模块 {D[0]:.6f} {D[1]:.6f} {D[2]:.6f}")

    model = kw.load(res["arpa_path"])
    diffs, worst = [], None
    for s in test:
        mine = [(counts.p_kn(w, u, v), (u, v, w)) for u, v, w in sl.events([s], counts.vocab)]
        theirs = [10 ** lp for lp, _, _ in model.full_scores(" ".join(s), bos=True, eos=True)]
        assert len(mine) == len(theirs), "预测位置数不一致"
        for (a, ev), b in zip(mine, theirs):
            d = abs(math.log10(a) - math.log10(b))
            diffs.append(d)
            if worst is None or d > worst[0]:
                worst = (d, ev, a, b)
    diffs.sort()
    print(f"逐位置 |Δlog10 p|：中位数 {diffs[len(diffs) // 2]:.2e}，99% 分位 {diffs[int(len(diffs) * .99)]:.2e}，"
          f"最大 {diffs[-1]:.2e}（{len(diffs)} 个位置）")
    print(f"  差得最多的位置：{worst[1]}，本模块 {worst[2]:.6g}，KenLM {worst[3]:.6g}")
    mine_ppl = sl.evaluate(counts.p_kn, test, counts.vocab)
    theirs_ev = kw.evaluate(model, test)
    print(f"测试集 PPL：本模块 {mine_ppl['ppl']:.3f}（<unk> 事件 {mine_ppl['unk_events']}），"
          f"KenLM {theirs_ev['ppl_including_oov']:.3f}（OOV {theirs_ev['oov_events']}）")

    gen = sampling.Generator(model, sampling.vocab_from_arpa(res["arpa_path"]))
    g = gen.generate(MAIN_PROMPT, 1.0, 50, max_tokens=30, seed=11)
    print("生成：", "".join(g["generated"]), "| 自然结束", g["hit_eos"],
          "| 匹配阶数", [s["matched_order"] for s in g["steps"]])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
