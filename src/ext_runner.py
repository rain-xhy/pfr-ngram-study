"""补充实验 E14–E19：Stupid Backoff、字级建模、词表口径、缓存插值、beam search、自适应解码。

在 WSL 里运行：`bash scripts/run_ext_wsl.sh [实验名 ...] [--force]`。设计与修正说明见
`../文档/执行指南/E14-E19补充实验计划.md` 开头的「修正说明」一节；E17 已取消（E6 做过查询速度）。
结果写进 results/<语料>/runs/<实验名>.json，与 E0–E13 同一目录、同一格式。
"""
import argparse
import itertools
import math
import random
import statistics
import subprocess
import sys
import time

import numpy as np

from . import backoff_lab as bl
from . import ext_common as xc
from . import kenlm_wrapper as kw
from . import sampling
from .common import GEN_SEEDS, MAIN_PROMPT, MAX_GEN_TOKENS, RUNS_DIR, STAT_SEEDS, WORK_DIR, read_json, \
    read_sentences, write_json
from .experiment_runner import (MODELS, SPLIT_DIR, add_overlap, environment, load_model, local_copy, log,
                                median_range, query_index, repeated_train, save, train_summary, vocab_of)

TIME_BIN = "/usr/bin/time"


def kenlm_scores_by_word(model, vocab, prefix_words):
    """KenLM 模型在给定上下文（句首 <s> 加 prefix_words）下对 vocab 里每个词的 log10 概率。"""
    import kenlm
    state = kenlm.State()
    model.BeginSentenceWrite(state)
    for w in prefix_words:
        nxt = kenlm.State()
        model.BaseScore(state, w, nxt)
        state = nxt
    tmp = kenlm.State()
    return np.fromiter((model.BaseScore(state, w, tmp) for w in vocab), dtype=np.float64, count=len(vocab))


def timed_python(args, timeout=3600):
    """经 /usr/bin/time 启动一个独立的 Python 进程，返回它打印的最后一行 JSON 与峰值内存、CPU 时间。"""
    import json
    cmd = [TIME_BIN, "-f", kw.TIME_FORMAT, sys.executable] + args
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    wall = time.perf_counter() - t0
    err = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{args} 退出码 {proc.returncode}：{err[-800:]}")
    out = json.loads(proc.stdout.decode("utf-8").strip().splitlines()[-1])
    return {"wall_seconds": wall, **kw._parse_time(err), **out}


# ---------------------------------------------------------------- E14 Stupid Backoff
E14_ORDERS = (3, 5)
E14_ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
E14_TUNE_POSITIONS = 20_000   # 验证集上选 α 用的位置数
E14_EVAL_POSITIONS = 50_000   # 测试集上算归一化困惑度的位置数
E14_RANK_POSITIONS = 3_000    # 测试集上对整个词表排序的位置数
E14_PAIRS = 2_000             # 原句与交换相邻两词后的句子
E14_BUILD_REPEATS = 3


def corrupt_pairs(sentences, vocab, n, seed=xc.SEED):
    """原句与「交换相邻两个不同的词」后的句子。两句词数相同、词的集合相同，
    未归一化的分数也能直接比较；一元模型给两句同样的分数，判别率恰为 50%。"""
    rng = random.Random(seed)
    pool = [s for s in sentences if 8 <= len(s) <= 40 and all(w in vocab for w in s)]
    pairs = []
    for s in rng.sample(pool, min(n * 2, len(pool))):
        cand = [i for i in range(len(s) - 1) if s[i] != s[i + 1]]
        if not cand:
            continue
        i = rng.choice(cand)
        bad = s[:i] + [s[i + 1], s[i]] + s[i + 2:]
        pairs.append({"original": s, "corrupted": bad, "swap_at": i})
        if len(pairs) == n:
            break
    return pairs


def judge(a, b):
    """原句分数 a 高于改坏句 b 记 1，相等记 0.5。"""
    return 1.0 if a > b else 0.5 if a == b else 0.0


def sb_collect(sb, sentences, order, pos_list=None):
    """逐位置收集 profile（pos_list 为 None 时取全部位置），之后每个 α 的归一化概率都由它直接算出。"""
    rows, cur, ids = [], -1, None
    for i, j in (pos_list if pos_list is not None else xc.positions(sentences)):
        if i != cur:
            ids, cur = [sb.bos] + sb.ids(sentences[i]) + [sb.eos], i
        rows.append(sb.profile(ids[max(0, j + 2 - order):j + 1], ids[j + 1], order))
    return rows


def sb_sentence_norm(sb, sent, alphas, order):
    """整句（含 </s>）归一化后的 log10 概率。"""
    return sum(r[0] for r in sb_eval(sb_collect(sb, [sent], order), alphas))


def sb_eval(profiles, alphas):
    """由 profile 求每个位置的 (归一化 log10 概率, 未归一化 log10 分数, 词表上的分数总和 Z)；
    未登录词位置返回 None。"""
    out = []
    for n, level, value, S in profiles:
        if value <= 0:
            out.append(None)
            continue
        f = {n: 1.0}
        for k in range(n - 1, 0, -1):
            f[k] = f[k + 1] * alphas[k + 1]
        z = sum(f[k] * S[k] for k in range(1, n + 1))
        raw = f[level] * value
        out.append((math.log10(raw / z), math.log10(raw), z))
    return out


def ppl_over(values):
    """values 里每项是一个位置的 log10 概率，None 表示未登录词位置（不计入）。"""
    vals = [v for v in values if v is not None]
    return {"ppl_excluding_oov": 10 ** (-sum(vals) / len(vals)), "events": len(vals),
            "oov_events": len(values) - len(vals)}


def sb_dev_ppl(profiles, alphas):
    return ppl_over([None if r is None else r[0] for r in sb_eval(profiles, alphas)])["ppl_excluding_oov"]


def tune_alphas(profiles, order, grid=E14_ALPHAS, rounds=2):
    """先在网格上选统一的 α，再从它出发逐阶做坐标下降，得到分阶 α。都用验证集上归一化后的困惑度。"""
    grid = list(grid)
    uni = {a: sb_dev_ppl(profiles, bl.uniform_alphas(a, order)) for a in grid}
    while min(uni, key=uni.get) == max(uni) and max(uni) < 10:  # 最优值落在网格上端时往上扩
        a = max(uni) * 1.5
        grid.append(a)
        uni[a] = sb_dev_ppl(profiles, bl.uniform_alphas(a, order))
    best_u = min(uni, key=uni.get)
    alphas, best = bl.uniform_alphas(best_u, order), uni[best_u]
    for _ in range(rounds):
        for k in range(order, 1, -1):
            for a in grid:
                trial = {**alphas, k: a}
                p = sb_dev_ppl(profiles, trial)
                if p < best - 1e-9:
                    best, alphas = p, trial
    return {"uniform_dev_ppl": {f"{a:g}": p for a, p in sorted(uni.items())}, "best_uniform": best_u,
            "per_order_alphas": {str(k): v for k, v in alphas.items()}, "per_order_dev_ppl": best,
            "_alphas": alphas}


def sb_build_timing(train_path, order):
    """Stupid Backoff 建表在独立进程里计时：预热一次，再计时 E14_BUILD_REPEATS 次。"""
    args = ["-m", "src.backoff_lab", str(train_path), str(order)]
    timed_python(args)
    runs = [timed_python(args) for _ in range(E14_BUILD_REPEATS)]
    cpu = [r["user_seconds"] + r["sys_seconds"] for r in runs]
    return {"wall_seconds": median_range([r["wall_seconds"] for r in runs]),
            "build_seconds": median_range([r["build_seconds"] for r in runs]),
            "cpu_seconds": median_range(cpu), "peak_rss_mb": median_range([r["peak_rss_mb"] for r in runs]),
            "structure_mib": runs[-1]["nbytes"] / 2 ** 20, "entry_counts": runs[-1]["entry_counts"],
            "vocab": runs[-1]["vocab"], "runs": runs}


def kn_word_logprobs(model, sentences):
    """KenLM 逐位置 log10 概率与是否未登录词，顺序同 xc.positions。"""
    out = []
    for s in sentences:
        out.extend((lp, oov) for lp, _, oov in model.full_scores(" ".join(s), bos=True, eos=True))
    return out


def flat_offsets(sentences):
    """xc.positions 顺序下，每句第一个预测位置在逐位置列表里的下标。"""
    return list(itertools.accumulate((len(s) + 1 for s in sentences), initial=0))


def e14_one_order(sb, n, dev, test, dev_pos, test_pos, rank_pos, pairs, e2n):
    tune = tune_alphas(sb_collect(sb, dev, n, dev_pos), n)
    tuned = tune.pop("_alphas")
    configs = {"α=0.4": bl.uniform_alphas(0.4, n), "α=1.0": bl.uniform_alphas(1.0, n),
               f"统一 α={tune['best_uniform']:g}": bl.uniform_alphas(tune["best_uniform"], n), "分阶 α": tuned}
    model = load_model(n)
    kn_all, off = kn_word_logprobs(model, test), flat_offsets(test)
    kn_sample = [None if kn_all[off[i] + j][1] else kn_all[off[i] + j][0] for i, j in test_pos]
    prof = sb_collect(sb, test, n, test_pos)
    configs_out, tuned_eval = {}, None
    for name, a in configs.items():
        ev = sb_eval(prof, a)
        norm = ppl_over([None if r is None else r[0] for r in ev])
        raw = ppl_over([None if r is None else r[1] for r in ev])
        zs = [r[2] for r in ev if r is not None]
        configs_out[name] = {"alphas": {str(k): v for k, v in a.items()}, "normalized_ppl": norm["ppl_excluding_oov"],
                             "unnormalized_pseudo_ppl": raw["ppl_excluding_oov"], "score_sum_mean": statistics.mean(zs),
                             "score_sum_median": statistics.median(zs), "events": norm["events"],
                             "oov_events": norm["oov_events"]}
        if name == "分阶 α":
            tuned_eval = ev
        log(f"  SB {n} 元 {name}：归一化困惑度 {norm['ppl_excluding_oov']:.2f}，"
            f"未归一化的「困惑度」{raw['ppl_excluding_oov']:.2f}，分数总和中位数 {statistics.median(zs):.2f}")
    kn_ppl = ppl_over(kn_sample)
    # 按句子分组重抽样：分阶 α 的 SB 相对 KN 的困惑度变化
    groups = {}
    for (i, _), k, s in zip(test_pos, kn_sample, tuned_eval):
        if k is None or s is None:
            continue
        g = groups.setdefault(i, [0.0, 0.0, 0])
        g[0] -= k
        g[1] -= s[0]
        g[2] += 1
    ci = xc.bootstrap_ppl_change(*zip(*groups.values()))
    ranks, disc = e14_ranks_and_pairs(sb, n, model, test, rank_pos, pairs, tuned, configs["α=0.4"])
    tr = e2n["training"]
    return {"tuning": tune, "configs": configs_out,
            "kn": {"sampled_ppl_excluding_oov": kn_ppl["ppl_excluding_oov"], "events": kn_ppl["events"],
                   "oov_events": kn_ppl["oov_events"], "full_test_ppl_excluding_oov": e2n["test"]["ppl_excluding_oov"],
                   "full_test_ppl_including_oov": e2n["test"]["ppl_including_oov"],
                   "wall_seconds": tr["wall_seconds"], "cpu_seconds": tr["cpu_seconds"],
                   "peak_rss_mb": tr["peak_rss_mb"], "arpa_bytes": tr["arpa_bytes"],
                   "probing_bytes": (MODELS / f"kn{n}.probing").stat().st_size, "ngram_counts": tr["ngram_counts"]},
            "sb_vs_kn_ppl_change": ci, "ranks": ranks, "discrimination": disc}


def e14_ranks_and_pairs(sb, n, model, test, rank_pos, pairs, tuned, fixed):
    """排名指标与句子判别。KN 的词表是 ARPA 一元表（去掉 <s>、<unk>），SB 的是训练集词形加 </s>，
    两者相同；未登录词位置在两边都算未命中。"""
    vocab = vocab_of(n)
    kn_index = {w: i for i, w in enumerate(vocab)}
    rows = {"KN": [], "SB 分阶 α": [], "SB α=0.4": []}
    for i, j in rank_pos:
        s = test[i]
        w = xc.target(s, j)
        # KenLM 的状态只保留最近 n-1 个词，从句首状态喂最近 n-1 个词与喂整段前缀得到的状态相同
        kn = kenlm_scores_by_word(model, vocab, s[max(0, j - n + 1):j])
        rows["KN"].append(xc.rank_metrics(kn, kn_index.get(w)))
        ctx = ([sb.bos] + sb.ids(s[:j]))[-(n - 1):]
        wid = sb.index.get(w)
        true = None if wid is None or wid in (sb.unk, sb.bos) else wid
        for name, a in (("SB 分阶 α", tuned), ("SB α=0.4", fixed)):
            sc = sb.scores(ctx, a, n)
            sc[[sb.unk, sb.bos]] = -1.0  # 不参与排序
            rows[name].append(xc.rank_metrics(sc, true))
    ranks = {name: xc.summarize_ranks(r) for name, r in rows.items()}
    disc = {"KN": [], "SB 分阶 α（未归一化）": [], "SB 分阶 α（归一化）": []}
    for p in pairs:
        o, c = p["original"], p["corrupted"]
        disc["KN"].append(judge(model.score(" ".join(o), bos=True, eos=True),
                                model.score(" ".join(c), bos=True, eos=True)))
        disc["SB 分阶 α（未归一化）"].append(judge(sb.sentence_log10(o, tuned, n), sb.sentence_log10(c, tuned, n)))
        disc["SB 分阶 α（归一化）"].append(judge(sb_sentence_norm(sb, o, tuned, n), sb_sentence_norm(sb, c, tuned, n)))
    out = {name: {"accuracy": statistics.mean(v), "n": len(v)} for name, v in disc.items()}
    out["SB 未归一化 − KN"] = xc.bootstrap_mean_diff(disc["KN"], disc["SB 分阶 α（未归一化）"])
    out["examples"] = [{"original": "".join(p["original"]), "corrupted": "".join(p["corrupted"])} for p in pairs[:5]]
    for name, r in ranks.items():
        log(f"  {n} 元 {name}：Top-1 {r['hit@1']:.1%}，Top-5 {r['hit@5']:.1%}，MRR {r['mrr']:.3f}")
    log(f"  {n} 元 判别：" + "，".join(f"{k} {v['accuracy']:.1%}" for k, v in out.items() if "accuracy" in v))
    return ranks, out


def e14_stupid_backoff():
    """Stupid Backoff 与插值 Modified KN（KenLM）在三元、五元上的对比，训练集是全部六个月。

    SB 的分数不归一化，直接求出来的「困惑度」不是困惑度。这里用三类不依赖归一化的指标：
    1. 在整个词表上除以分数总和得到概率，再算困惑度（去掉未登录词位置，KN 同口径对照）；
    2. 对整个词表排序后真实词的 Top-1/5/10 命中率与平均倒数排名（排名与是否归一化无关）；
    3. 原句与交换相邻两词后的句子，谁的分数高。
    α 在验证集上选：统一 α 扫网格，再逐阶坐标下降得到分阶 α（初稿「按 E8 匹配分布拍定 α」改为这种做法）。
    """
    train_path = local_copy(SPLIT_DIR / "train.txt")
    train = read_sentences(SPLIT_DIR / "train.txt")
    dev = read_sentences(SPLIT_DIR / "dev.txt")
    test = read_sentences(SPLIT_DIR / "test.txt")
    e2 = read_json(RUNS_DIR / "E2_orders.json")["orders"]
    timing = {}
    for n in E14_ORDERS:
        timing[str(n)] = sb_build_timing(train_path, n)
        log(f"  SB {n} 元建表：{timing[str(n)]['build_seconds']['median']:.1f}s，"
            f"峰值 {timing[str(n)]['peak_rss_mb']['median']:.0f} MiB")
    t0 = time.perf_counter()
    sb = bl.StupidBackoff(train, max(E14_ORDERS))
    log(f"  SB 五元建表（本进程）{time.perf_counter() - t0:.1f}s，词表 {sb.V:,}")
    del train
    dev_pos = xc.sample(xc.positions(dev), E14_TUNE_POSITIONS)
    test_pos = xc.sample(xc.positions(test), E14_EVAL_POSITIONS, seed=xc.SEED + 1)
    rank_pos = xc.sample(test_pos, E14_RANK_POSITIONS, seed=xc.SEED + 2)
    vocab_set = set(sb.words)
    pairs = corrupt_pairs(test, vocab_set, E14_PAIRS)
    out = {"orders": {}, "train_events": sb.N, "sb_vocab": sb.V,
           "positions": {"tune_dev": len(dev_pos), "eval_test": len(test_pos), "rank_test": len(rank_pos)},
           "pairs": len(pairs)}
    for n in E14_ORDERS:
        out["orders"][str(n)] = e14_one_order(sb, n, dev, test, dev_pos, test_pos, rank_pos, pairs, e2[str(n)])
        out["orders"][str(n)]["sb_training"] = {k: v for k, v in timing[str(n)].items() if k != "runs"}
    out["sb_timing_runs"] = {n: t["runs"] for n, t in timing.items()}
    return out


# ---------------------------------------------------------------- E15 字级建模
def to_chars(sentences):
    """词级句子拆成字级：每个词拆成单字，句子边界不变。全角数字、字母同样一字一个 token。"""
    return [[c for w in s for c in w] for s in sentences]


def char_count(sentences):
    """每字比特数的分母：测试集字数加句数（每句的 </s> 也算一个要预测的单位），词级与字级共用。"""
    return sum(len(w) for s in sentences for w in s) + len(sentences)


def bits_per_char(ev, chars):
    return -ev["log10_prob"] * math.log2(10) / chars


# 本机 KenLM 按默认 KENLM_MAX_ORDER=6 编译（见 results/s4_install_log.md），字级最高也只到六元
E15_CHAR_ORDERS = (3, 5, 6)


def oov_cost_check(word_model, char_model, test_w):
    """词级模型在未登录词处付的比特数，与字级模型拼出同一批词付的比特数。

    每字比特数的比较里，词级模型遇到未登录词只付 <unk> 的代价；这里把那部分单独拿出来，
    看它比字级模型逐字拼出这些词便宜还是贵，也就是每字比特数的比较对词级偏了多少。
    """
    wb = cb = 0.0
    n_oov = n_chars = 0
    for s in test_w:
        ws = list(word_model.full_scores(" ".join(s), bos=True, eos=True))
        cs = list(char_model.full_scores(" ".join(c for w in s for c in w), bos=True, eos=True))
        k = 0
        for i, w in enumerate(s):
            if ws[i][2]:
                wb -= ws[i][0] * math.log2(10)
                cb -= sum(x[0] for x in cs[k:k + len(w)]) * math.log2(10)
                n_oov += 1
                n_chars += len(w)
            k += len(w)
    return {"oov_words": n_oov, "oov_chars": n_chars, "word_model_bits": wb, "char_model_bits": cb,
            "word_model_bits_per_oov": wb / n_oov if n_oov else None,
            "char_model_bits_per_oov": cb / n_oov if n_oov else None}


def e15_char_level():
    """同一批训练/测试文章，词级与字级 KN 模型（字级三元、五元、六元），比较每字比特数。

    每词困惑度与每字困惑度不可直接比：预测单位不同。可比的是整份测试集的总信息量除以同一个字数。
    词级模型的未登录词只付了 <unk> 的代价、没付拼出这个词的代价，这个比较对词级偏有利。
    """
    test_w = read_sentences(SPLIT_DIR / "test.txt")
    dev_w = read_sentences(SPLIT_DIR / "dev.txt")
    chars = char_count(test_w)
    e2 = read_json(RUNS_DIR / "E2_orders.json")["orders"]
    word_rows = {}
    for n in (3, 5):
        t = e2[str(n)]["test"]
        word_rows[str(n)] = {"ppl_including_oov": t["ppl_including_oov"], "oov_rate": t["oov_rate"],
                             "bits_per_token": t["cross_entropy_bits"], "bits_per_char": bits_per_char(t, chars),
                             "training": {k: e2[str(n)]["training"][k] for k in
                                          ("wall_seconds", "cpu_seconds", "peak_rss_mb", "arpa_bytes", "ngram_counts")}}
    char_dir = WORK_DIR / "corpus" / "char"
    char_dir.mkdir(parents=True, exist_ok=True)
    train_c = to_chars(read_sentences(SPLIT_DIR / "train.txt"))
    train_path = char_dir / "train.txt"
    train_path.write_text("\n".join(" ".join(s) for s in train_c) + "\n", encoding="utf-8")
    char_vocab = {c for s in train_c for c in s}
    del train_c
    test_c, dev_c = to_chars(test_w), to_chars(dev_w)
    char_rows = {}
    for n in E15_CHAR_ORDERS:
        arpa, runs = repeated_train(f"char{n}", train_path, n, repeats=3)
        m = kw.load(arpa)
        ev, evd = kw.evaluate(m, test_c), kw.evaluate(m, dev_c)
        char_rows[str(n)] = {"ppl_including_oov": ev["ppl_including_oov"], "oov_rate": ev["oov_rate"],
                             "oov_events": ev["oov_events"], "dev_ppl": evd["ppl_including_oov"],
                             "bits_per_token": ev["cross_entropy_bits"], "bits_per_char": bits_per_char(ev, chars),
                             "training": train_summary(runs)}
        log(f"  字级 {n} 元：每字困惑度 {ev['ppl_including_oov']:.2f}，每字 {char_rows[str(n)]['bits_per_char']:.3f} 比特")
        if n == 5:
            gen = sampling.Generator(m, sampling.vocab_from_arpa(arpa))
            prompt = [c for w in MAIN_PROMPT for c in w]
            samples = [gen.generate(prompt, 1.0, 20, seed=s, max_tokens=80) for s in GEN_SEEDS]
            char_samples = [{"seed": g["params"]["seed"], "text": "".join(g["generated"]), "hit_eos": g["hit_eos"]}
                            for g in samples]
            oov_check = oov_cost_check(load_model(5), m, test_w)
        del m
        if n != 5:
            arpa.unlink()
    for n in (3, 5):
        word_rows[str(n)]["ppl_excluding_oov"] = e2[str(n)]["test"]["ppl_excluding_oov"]
    return {"test_chars_plus_sentences": chars, "test_words": sum(len(s) for s in test_w),
            "chars_per_word": (chars - len(test_w)) / sum(len(s) for s in test_w),
            "char_vocab": len(char_vocab), "word": word_rows, "char": char_rows, "char5_samples": char_samples,
            "oov_cost_check": oov_check}


# ---------------------------------------------------------------- E16 词表口径
E16_TOP_K = (10_000, 20_000, 40_000, 65_533)   # 65,533 是 Highsun 实现的词表上限（16 位编号减 3 个特殊符号）
E16_COVERAGE = (0.95, 0.98, 0.99)             # 训练集词次覆盖率


def vocab_by_top_k(freq, k):
    return {w for w, _ in freq.most_common(k)}


def vocab_by_coverage(freq, target):
    total, run, keep = sum(freq.values()), 0, set()
    for w, c in freq.most_common():
        keep.add(w)
        run += c
        if run >= target * total:
            break
    return keep


def e16_vocab_protocol():
    """按词频截取前 K 个词、按训练集词次覆盖率截取两种方式建封闭词表，其余词在训练集与测试集里
    都换成「低频词」，五元 KN 训练、评估。与 E13 的频次阈值合在一起，画出困惑度与被换掉的词次比例的关系。

    这些数字不能当作模型变好：被换掉的词越多，要预测的东西越少。同一批模型另报两个不随词表变化的量：
    每字比特数（换掉的词按字数计入分母，但它的拼写信息没有被预测，对小词表偏有利，所以只作说明），
    以及只在「两种词表都保留的词」位置上的困惑度，用来看换词表是否改变了模型对常见词的预测。
    """
    from .common import RARE_TOKEN
    from .experiment_runner import _map_rare, _write_sentences
    train = read_sentences(SPLIT_DIR / "train.txt")
    test = read_sentences(SPLIT_DIR / "test.txt")
    freq = xc.train_freq()
    chars = char_count(test)
    specs = [("前 K 个词", f"K={k:,}", vocab_by_top_k(freq, k)) for k in E16_TOP_K] + \
            [("覆盖率", f"覆盖 {c:.0%}", vocab_by_coverage(freq, c)) for c in E16_COVERAGE]
    core = set.intersection(*(v for _, _, v in specs))  # 所有词表都保留的词
    rows = []
    for kind, label, vocab in specs:
        tr, te = _map_rare(train, vocab), _map_rare(test, vocab)
        path = WORK_DIR / "corpus" / "closed" / "e16.txt"
        _write_sentences(path, tr)
        arpa, _ = repeated_train("e16_kn5", path, 5, repeats=1)
        m = kw.load(arpa)
        scores = kw.position_scores(m, te)
        ev = kw.summarize_scores(scores, len(te))
        core_lp, core_n, k = 0.0, 0, 0
        for s in te:
            for j in range(len(s) + 1):
                if j < len(s) and s[j] in core:
                    core_lp += scores[k][0]
                    core_n += 1
                k += 1
        rows.append({"kind": kind, "label": label, "vocab_size": len(vocab) + 1,
                     "train_token_coverage": sum(freq[w] for w in vocab) / sum(freq.values()),
                     "test_rare_rate": sum(w == RARE_TOKEN for s in te for w in s) / sum(len(s) for s in te),
                     "ppl": ev["ppl_including_oov"], "bits_per_char": bits_per_char(ev, chars),
                     "core_word_ppl": 10 ** (-core_lp / core_n), "core_positions": core_n,
                     "arpa_bytes": arpa.stat().st_size})
        log(f"  {label}：词表 {len(vocab) + 1:,}，测试集替换为低频词 {rows[-1]['test_rare_rate']:.1%}，"
            f"五元困惑度 {ev['ppl_including_oov']:.2f}，常用词位置 {rows[-1]['core_word_ppl']:.2f}")
        del m
        arpa.unlink()
        path.unlink()
    # 开放词表（E2 的五元）在同一批常用词位置上的困惑度，作为对照
    m = load_model(5)
    scores = kw.position_scores(m, test)
    core_lp, core_n, k = 0.0, 0, 0
    for s in test:
        for j in range(len(s) + 1):
            if j < len(s) and s[j] in core:
                core_lp += scores[k][0]
                core_n += 1
            k += 1
    e2 = read_json(RUNS_DIR / "E2_orders.json")["orders"]["5"]["test"]
    open_row = {"label": "开放词表", "vocab_size": len(freq), "ppl": e2["ppl_including_oov"],
                "test_rare_rate": sum(1 for s in test for w in s if w not in freq) / sum(len(s) for s in test),
                "bits_per_char": bits_per_char(e2, chars),
                "core_word_ppl": 10 ** (-core_lp / core_n), "core_positions": core_n}
    e13 = read_json(RUNS_DIR / "E13_protocol.json")["closed_vocab_full"]
    thresholds = [{"label": f"频次 ≥{c['min_freq']}", "vocab_size": c["vocab_size"],
                   "test_rare_rate": c["test_rare_rate"], "ppl": c["orders"]["5"]["ppl_including_oov"]} for c in e13]
    return {"open": open_row, "rows": rows, "e13_thresholds": thresholds, "core_vocab_size": len(core)}


# ---------------------------------------------------------------- E17 文档缓存插值
E17_LAMBDAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)
E17_DECAYS = (None, 0.999, 0.995, 0.99, 0.98)   # None 为不衰减的缓存
E17_BASE_ORDERS = (3, 5, 6)


def cache_arrays(sentences, articles, oov, decay):
    """逐位置的缓存量，顺序同 xc.positions。返回 (pc, has, seen, art)：

    pc 是真实词在缓存里的概率；has 表示缓存非空；seen 表示真实词在本篇此前出现过；art 是文章序号。
    缓存是同一篇文章里此前出现过的词表内词（含 </s>）的频率分布，每篇文章开头清空。未登录词不进缓存，
    在缓存里的概率为 0：插值只可能让未登录词位置变差，困惑度的下降全部来自词表内的词。
    decay 不为 None 时，距离当前 d 个位置的历史权重为 decay^d（指数衰减缓存）；实现上不逐项衰减，
    而是让新加入的权重按 1/decay 递增，比值不变。
    """
    n = len(oov)
    pc, has, seen = np.zeros(n), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    art_idx = np.zeros(n, dtype=np.int32)
    k, prev, a = 0, None, -1
    counts, total, g = {}, 0.0, 1.0
    for s, art in zip(sentences, articles):
        if art != prev:
            counts, total, g, prev, a = {}, 0.0, 1.0, art, a + 1
        for j in range(len(s) + 1):
            w = xc.target(s, j)
            if total > 0:
                has[k] = True
                if not oov[k]:
                    c = counts.get(w, 0.0)
                    pc[k], seen[k] = c / total, c > 0
            art_idx[k] = a
            if not oov[k]:
                counts[w] = counts.get(w, 0.0) + g
                total += g
            if decay is not None:
                g /= decay
                if g > 1e100:
                    counts = {x: v / g for x, v in counts.items()}
                    total, g = total / g, 1.0
            k += 1
    return pc, has, seen, art_idx


def mix_log10(lp, pc, has, lam):
    """插值 (1−λ)·P_KN + λ·P_缓存；缓存为空的位置（每篇第一个词）只用 P_KN，保证处处是合法分布。"""
    if lam == 0:
        return lp
    return np.where(has, np.log10((1 - lam) * np.power(10.0, lp) + lam * pc), lp)


def split_arrays(model, split):
    sents = read_sentences(SPLIT_DIR / f"{split}.txt")
    arts = (SPLIT_DIR / f"{split}_articles.txt").read_text(encoding="utf-8").split()
    scores = kw.position_scores(model, sents)
    lp = np.array([s[0] for s in scores])
    oov = np.array([s[2] for s in scores], dtype=bool)
    return sents, arts, lp, oov


def e17_cache():
    """文档缓存插值（Kuhn & De Mori 1990）：同一篇文章里出现过的词，在后文更可能再出现。

    n-gram 只看前 n-1 个词，看不到几句之前提到的人名、地名；缓存把本篇已出现的词的频率与 KN 概率
    插值。λ 与衰减系数在验证集上选，测试集只用一次；与 E2 的困惑度同一个测试集、同一个词表、
    同一个口径（含未登录词），可以直接比。按文章重抽样给出困惑度变化的 95% 区间。
    """
    out = {}
    for n in E17_BASE_ORDERS:
        model = load_model(n)
        d_sents, d_arts, d_lp, d_oov = split_arrays(model, "dev")
        t_sents, t_arts, t_lp, t_oov = split_arrays(model, "test")
        grid = {}
        for decay in E17_DECAYS:
            pc, has, _, _ = cache_arrays(d_sents, d_arts, d_oov, decay)
            for lam in E17_LAMBDAS:
                grid[(decay, lam)] = xc.ppl(-mix_log10(d_lp, pc, has, lam).sum(), len(d_lp))
        decay, lam = min(grid, key=grid.get)
        pc, has, seen, art = cache_arrays(t_sents, t_arts, t_oov, decay)
        mixed = mix_log10(t_lp, pc, has, lam)
        base_ppl, mix_ppl = xc.ppl(-t_lp.sum(), len(t_lp)), xc.ppl(-mixed.sum(), len(t_lp))
        g = art.max() + 1
        la, lb = np.bincount(art, weights=-t_lp, minlength=g), np.bincount(art, weights=-mixed, minlength=g)
        ev = np.bincount(art, minlength=g)
        ci = xc.bootstrap_ppl_change(la, lb, ev)
        # 困惑度的变化来自哪里：真实词在本篇此前出现过 / 没出现过 / 未登录词
        # 四组互不重叠、合起来是全部位置
        groups = {"本篇此前出现过的词": seen, "本篇首次出现的词表内词": (~seen) & (~t_oov) & has,
                  "每篇第一个位置": (~has) & (~t_oov), "未登录词": t_oov}
        parts = {}
        gain = float((t_lp - mixed).sum()) or 1.0  # λ=0 时没有变化，占比记 0
        for name, mask in groups.items():
            parts[name] = {"positions": int(mask.sum()), "share": float(mask.mean()),
                           "base_ppl": xc.ppl(-t_lp[mask].sum(), int(mask.sum())) if mask.any() else None,
                           "mixed_ppl": xc.ppl(-mixed[mask].sum(), int(mask.sum())) if mask.any() else None,
                           "loss_change_share": float((t_lp[mask] - mixed[mask]).sum()) / gain}
        out[str(n)] = {"best_decay": decay, "best_lambda": lam, "dev_base_ppl": grid[(decay, 0.0)],
                       "dev_best_ppl": grid[(decay, lam)],
                       "dev_grid": [{"decay": d, "lambda": l, "ppl": p} for (d, l), p in grid.items()],
                       "test_base_ppl": base_ppl, "test_mixed_ppl": mix_ppl, "bootstrap": ci, "parts": parts,
                       "e2_test_ppl": read_json(RUNS_DIR / "E2_orders.json")["orders"][str(n)]["test"]["ppl_including_oov"]}
        log(f"  {n} 元 + 缓存（衰减 {decay}，λ={lam}）：测试集 {base_ppl:.2f} → {mix_ppl:.2f}"
            f"（{ci['change']:+.1%}，95% 区间 {ci['ci95'][0]:+.1%} ~ {ci['ci95'][1]:+.1%}）")
        del model
    return {"lambdas": list(E17_LAMBDAS), "decays": list(E17_DECAYS), "by_order": out}


# ---------------------------------------------------------------- E18 beam search
E18_BEAMS = (1, 5, 10)
E18_LENGTH_ALPHAS = (0.0, 0.6, 1.0)


def beam_search(gen, prefix, beam, max_tokens=MAX_GEN_TOKENS):
    """在整个词表上做 beam search，返回全部候选 (log10 概率, 词序列, 是否自然结束)。

    每步每条候选取概率最高的 beam 个后继，保留总分最高的 beam 条；碰到 </s> 的候选移入完成集合。
    到 max_tokens 仍未结束的候选也留下，记为未自然结束。长度惩罚只影响最后从候选里挑哪一条，
    不影响搜索过程，所以同一个 beam 宽度只搜一次，各个 α 从同一批候选里挑（见 pick_beam）。
    """
    import kenlm
    live = [(0.0, [], gen.start(prefix))]
    done = []
    for _ in range(max_tokens):
        cand = []
        for lp, toks, state in live:
            raw = gen.scores(state)
            # 稳定排序：同分时取编号小的词，与贪心用的 np.argmax 一致
            top = np.argsort(-raw, kind="stable")[:beam]
            cand.extend((lp + float(raw[i]), toks, state, int(i)) for i in top)
        cand.sort(key=lambda c: -c[0])
        live = []
        for lp, toks, state, i in cand:
            if len(live) >= beam:
                break
            if i == gen.eos_index:
                done.append((lp, toks, True))
            else:
                nxt = kenlm.State()
                gen.model.BaseScore(state, gen.vocab[i], nxt)
                live.append((lp, toks + [gen.vocab[i]], nxt))
        if not live:
            break
    done.extend((lp, toks, False) for lp, toks, _ in live)
    return done


def pick_beam(done, prefix, length_alpha):
    """按 log P / 长度^α 从候选里挑一条（α=0 即原始对数概率，偏爱短句；α=1 是每词平均对数概率）。
    长度把句末 </s> 也算一次预测。"""
    def norm(c):
        return c[0] / max(len(c[1]) + c[2], 1) ** length_alpha
    lp, toks, eos = max(done, key=norm)
    return {"prefix": list(prefix), "generated": toks, "hit_eos": eos, "log10_prob": lp,
            "finished_candidates": sum(1 for c in done if c[2]), "candidates": len(done)}


def e18_beam():
    """六元模型上 beam search 与贪心、采样的对比。beam=1 就是贪心。

    E7 里贪心在各阶数都陷入循环（三元自重复 62%–93%）；这里看 beam 宽度与长度惩罚能否摆脱循环，
    以及得到的是不是训练集原句（E9 的照搬检测）。
    """
    best = read_json(RUNS_DIR / "E2_orders.json")["best_order_by_dev"]
    model = load_model(best)
    gen = sampling.Generator(model, vocab_of(best))
    pending = []
    for b in E18_BEAMS:
        t0 = time.perf_counter()
        done = beam_search(gen, MAIN_PROMPT, b)
        seconds = time.perf_counter() - t0
        for a in (E18_LENGTH_ALPHAS if b > 1 else (0.0,)):
            g = pick_beam(done, MAIN_PROMPT, a)
            g["seconds"] = seconds
            g["params"] = {"beam": b, "length_alpha": a, "temperature": None, "top_k": None, "top_p": None,
                           "seed": None}
            pending.append(g)
            log(f"  beam={b}, α={a}：{len(g['generated'])} 词，{'自然结束' if g['hit_eos'] else '到长度上限'}，"
                f"{seconds:.1f}s：{''.join(g['generated'])[:40]}")
    index = query_index([g["prefix"] + g["generated"] for g in pending])
    rows = []
    for g in pending:
        rec = {"params": g["params"], "text": "".join(g["generated"]), "tokens": g["generated"],
               "hit_eos": g["hit_eos"], "log10_prob": g["log10_prob"], "seconds": g["seconds"],
               "finished_candidates": g["finished_candidates"], **sampling.repetition_stats(g["generated"])}
        rows.append(add_overlap(rec, g, index))
    e7 = read_json(RUNS_DIR / "E7_generation.json")
    ref = {c["name"]: c["summary"] for c in e7["grid"] if c["name"] in ("T=0.7, k=10", "T=1.0, k=50")}
    # beam=1 就是贪心，应当与 E7 的贪心输出逐词相同（同一个模型、同一个词表、同一个开头）
    greedy_matches_e7 = rows[0]["tokens"] == e7["greedy"]["tokens"]
    log(f"  beam=1 与 E7 贪心输出{'相同' if greedy_matches_e7 else '不同'}")
    return {"model_order": best, "prompt": MAIN_PROMPT, "rows": rows, "sampling_reference": ref,
            "beam1_equals_e7_greedy": greedy_matches_e7}


# ---------------------------------------------------------------- E19 自适应解码
E19_SEEDS = STAT_SEEDS + list(range(2001, 2021))   # 40 个种子
E19_TARGET_BITS = (5.0, 6.0, 7.0)                   # 按熵定温度时的目标熵


def entropy_bits_of(raw):
    p = np.power(10.0, raw)
    return sampling.entropy_bits(p / p.sum())


def entropy_at(ln_scores, t):
    """温度 t 下 q ∝ p^(1/t) 的熵（比特）。不排序，二分时每次只要一遍向量运算。"""
    x = ln_scores / t
    x = x - x.max()
    q = np.exp(x)
    q /= q.sum()
    q = q[q > 0]
    return float(-(q * np.log2(q)).sum())


def solve_temperature(raw, target_bits, lo=0.05, hi=3.0, iters=25):
    """二分求温度，使缩放后分布的熵等于 target_bits。熵随温度单调增。raw 是 log10 概率。"""
    ln = np.asarray(raw, dtype=np.float64) * math.log(10)
    if target_bits <= entropy_at(ln, lo):
        return lo
    if target_bits >= entropy_at(ln, hi):
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if entropy_at(ln, mid) < target_bits:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class SortedGramIndex:
    """训练集 2..max_n 元 n-gram 的哈希，存成有序 uint64 数组（六个月约三千万个、二百多 MB）。

    生成过程中要随时判断「末尾这几个词在训练集出现过没有」，不能像 E9 那样先生成完再建索引。
    哈希与 backoff_lab 相同的多项式（词先转成编号），碰撞误判的概率在 1e-9 量级。句内不跨句。
    """

    def __init__(self, sentences, max_n=8):
        self.max_n = max_n
        words = sorted({w for s in sentences for w in s})
        self.index = {w: i + 1 for i, w in enumerate(words)}
        seq = []
        for s in sentences:
            seq.append(0)  # 句间分隔
            seq.extend(self.index[w] for w in s)
        t = np.asarray(seq, dtype=np.int64)
        del seq
        pos = np.arange(len(t))
        start = np.maximum.accumulate(np.where(t == 0, pos, 0)) + 1  # 每个位置所在句子第一个词的位置
        h = np.zeros(len(t), dtype=np.uint64)
        parts = []
        for L in range(1, max_n + 1):  # 以每个位置结尾、长度为 L 的 n-gram，从结尾往前混入
            prev = np.zeros(len(t), dtype=np.int64)
            prev[L - 1:] = t[:len(t) - L + 1]
            h = h * np.uint64(bl.MULT) + prev.astype(np.uint64)
            if L >= 2:
                parts.append(h[(t != 0) & (pos - (L - 1) >= start)])
        self.hashes = np.unique(np.concatenate(parts))

    def has_hash(self, h):
        key = np.uint64(h)
        j = int(np.searchsorted(self.hashes, key))
        return j < len(self.hashes) and self.hashes[j] == key

    def copy_run(self, tokens):
        """末尾连续多少个词构成训练集里出现过的 n-gram（2..max_n，不足 2 记 0）。"""
        run, h = 0, 0
        for L in range(1, min(len(tokens), self.max_n) + 1):
            i = self.index.get(tokens[-L])
            if i is None:
                break
            h = (h * bl.MULT + i) & bl.MASK
            if L >= 2:
                if not self.has_hash(h):
                    break
                run = L
        return run


ANTICOPY_BASE_T, ANTICOPY_HOT_T = 0.7, 1.3


def adaptive_generate(gen, index, prefix, mode, param, seed, max_tokens=MAX_GEN_TOKENS):
    """逐步改温度的采样。mode：
    entropy —— 每步按当前分布的熵求温度，使采样分布的熵固定为 param 比特（高确定处不压、低确定处收紧）；
    anticopy —— 基础温度 0.7（E9 里照搬最多的温度），末尾已连续照搬训练集 ≥ param 个词时，
    本步温度升到 1.3，把 E9 的照搬检测做成生成时的控制。
    都在整个词表上采样，屏蔽 <s> 与 <unk>，与 E7 相同。index 是 SortedGramIndex。"""
    rng = np.random.default_rng(seed)
    state = gen.start(prefix)
    out, temps, hit_eos = [], [], False
    for _ in range(max_tokens):
        raw = gen.scores(state)
        if mode == "entropy":
            t = solve_temperature(raw, param)
        else:
            # 只看生成部分加提示词最后一个词：提示词本身在训练集里的重合不触发
            run = index.copy_run((prefix + out)[-(len(out) + 1):])
            t = ANTICOPY_HOT_T if run >= param else ANTICOPY_BASE_T
        order, probs = sampling.transform(raw, t)
        idx = int(order[int(rng.choice(len(order), p=probs))])
        temps.append(t)
        if idx == gen.eos_index:
            hit_eos = True
            break
        out.append(gen.vocab[idx])
        state = gen._advance(state, gen.vocab[idx])
    return {"prefix": list(prefix), "generated": out, "hit_eos": hit_eos, "temperatures": temps,
            "params": {"mode": mode, "param": param, "seed": seed, "temperature": None, "top_k": 0, "top_p": 1.0}}


def fixed_generate(gen, prefix, temperature, seed):
    g = gen.generate(prefix, temperature, 0, 1.0, max_tokens=MAX_GEN_TOKENS, seed=seed)
    g["temperatures"] = [temperature] * len(g["steps"])
    g["params"] = {"mode": "fixed", "param": temperature, "seed": seed, "temperature": temperature,
                   "top_k": 0, "top_p": 1.0}
    return g


def gen_quality(g, model):
    """生成部分的质量指标：同一个模型打分的困惑度（流畅度的代理）、自重复、多样性。"""
    toks = g["generated"]
    scores = list(model.full_scores(" ".join(g["prefix"] + toks), bos=True, eos=g["hit_eos"]))
    part = scores[len(g["prefix"]):]
    self_ppl = 10 ** (-sum(s for s, _, _ in part) / len(part)) if part else float("nan")
    return {"params": g["params"], "text": "".join(toks), "tokens": toks, "hit_eos": g["hit_eos"],
            "self_ppl": self_ppl, "mean_temperature": statistics.mean(g["temperatures"]) if g["temperatures"] else None,
            **sampling.repetition_stats(toks)}


def e19_adaptive_decoding():
    """固定温度与两种逐步调温度的采样，六元模型、作业开头、每种 40 个种子、全词表。

    比较的是同一张「流畅（生成部分的自困惑度低）」与「不照搬（最长照搬词数短）」的取舍图：
    固定温度 0.7–1.3 画出一条曲线，自适应方法的点若落在曲线左下方，说明同样照搬程度下更流畅。
    """
    best = read_json(RUNS_DIR / "E2_orders.json")["best_order_by_dev"]
    model = load_model(best)
    gen = sampling.Generator(model, vocab_of(best))
    t0 = time.perf_counter()
    gram_index = SortedGramIndex(xc.train_sentences())
    log(f"  训练集 2-8 元哈希 {len(gram_index.hashes):,} 个，{time.perf_counter() - t0:.0f}s")
    methods = [("fixed", t) for t in (0.5, 0.7, 0.85, 1.0, 1.15, 1.3)] + \
              [("entropy", b) for b in E19_TARGET_BITS] + [("anticopy", r) for r in (4, 6)]
    raw = {}
    for mode, p in methods:
        gs = [fixed_generate(gen, MAIN_PROMPT, p, s) if mode == "fixed"
              else adaptive_generate(gen, gram_index, MAIN_PROMPT, mode, p, s) for s in E19_SEEDS]
        raw[(mode, p)] = gs
        log(f"  {mode} {p}：已生成 {len(gs)} 次")
    del gram_index
    index = query_index([g["prefix"] + g["generated"] for gs in raw.values() for g in gs])
    rows = []
    for (mode, p), gs in raw.items():
        recs = [add_overlap(gen_quality(g, model), g, index) for g in gs]
        keys = ("length", "self_ppl", "longest_copied_span", "overlap_5gram", "repeated_3gram_rate",
                "type_token_ratio", "mean_temperature")
        summary = {k: _mean(r[k] for r in recs) for k in keys}
        summary["eos_rate"] = statistics.mean(r["hit_eos"] for r in recs)
        summary["copied_10plus_rate"] = statistics.mean(r["copied_10plus"] for r in recs)
        # 各次生成的自困惑度差几个数量级（E7 里 T=1.3 全词表是 14,086），算术平均被个别样本主导，另给几何平均
        logs = [math.log(r["self_ppl"]) for r in recs if r["self_ppl"] == r["self_ppl"]]
        summary["self_ppl_geomean"] = math.exp(statistics.mean(logs)) if logs else None
        rows.append({"mode": mode, "param": p, "summary": summary,
                     "samples": [{k: r[k] for k in ("text", "hit_eos", "self_ppl", "longest_copied_span")}
                                 for r in recs if r["params"]["seed"] in GEN_SEEDS]})
        log(f"  {mode} {p}：自困惑度几何平均 {summary['self_ppl_geomean']:.1f}，"
            f"最长照搬 {summary['longest_copied_span']:.1f}，平均温度 {summary['mean_temperature']:.2f}")
    return {"model_order": best, "seeds": E19_SEEDS, "rows": rows}


def _mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.mean(vals) if vals else None


EXPERIMENTS = [("E14_stupid_backoff", e14_stupid_backoff), ("E15_char_level", e15_char_level),
               ("E16_vocab_protocol", e16_vocab_protocol), ("E17_cache", e17_cache),
               ("E18_beam", e18_beam), ("E19_adaptive_decoding", e19_adaptive_decoding)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="*", default=[name for name, _ in EXPERIMENTS], help="实验名，如 E14_stupid_backoff")
    ap.add_argument("--force", action="store_true", help="已有结果也重跑")
    args = ap.parse_args(argv)
    todo = [(n, f) for n, f in EXPERIMENTS if n in args.targets or n.split("_")[0] in args.targets]
    (WORK_DIR / "tmp").mkdir(parents=True, exist_ok=True)
    write_json(RUNS_DIR / "environment.json", environment())
    for name, fn in todo:
        if (RUNS_DIR / f"{name}.json").exists() and not args.force:
            log(f"跳过 {name}（结果已存在）")
            continue
        log(f"开始 {name}")
        started = time.perf_counter()
        result = fn()
        save(name, {"elapsed_seconds": time.perf_counter() - started, **result})
        log(f"完成 {name}，用时 {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
