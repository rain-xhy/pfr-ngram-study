"""按实验总清单的顺序执行全部实验：Phase 1（E0 E1 E2 E3 E4 E7）→ Phase 2（E5 E8 E9 E6）→ Phase 3（E10 E11 E12）。

在 WSL 里运行：`bash scripts/run_wsl.sh`，或 `~/pfr-venv/bin/python -m src.experiment_runner [阶段名 ...]`。
每个实验的原始结果写进 results/runs/<实验名>.json，模型文件放在 WSL 本地的 ~/pfr-work/。
中途失败后重跑时，已有结果的实验默认跳过；加 --force 全部重跑。
改动了某个实验的代码，就删掉 results/runs/ 下对应的 JSON 再运行，依赖它的实验一并删掉：
E3 E7 E8 E10 E11 E12 读 E2 的模型文件，E9 读 E7 的结果，E12 读 E3 的结果。

E0 是对作业里「这次必须清理掉所有多余的 tag 和标签，为什么？」的实证：同一批训练/测试文章，
分别用清洗版和带词性版训练，看词表、稀疏程度与生成结果差多少。
"""
import argparse
import math
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

from . import data as pfr_data
from . import kenlm_wrapper as kw
from . import sampling
from . import smoothing_lab as sl
from .common import (DEFAULT_MEMORY, GEN_SEEDS, KENLM_COMMIT, MAIN_PROMPT, MAX_GEN_TOKENS, PROCESSED_DIR,
                     RARE_TOKEN, REPEATS, RESULTS_DIR, RUNS_DIR, STAT_SEEDS, WORK_DIR, now_iso, read_json,
                     read_sentences, sha256_file, write_json)

SPLIT_DIR = PROCESSED_DIR / "splits_main"
TAGGED_DIR = PROCESSED_DIR / "splits_tagged"
SCALE_DIR = PROCESSED_DIR / "scale_samples"
MODELS = WORK_DIR / "models"
ORDERS = [2, 3, 4, 5, 6]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def local_copy(src: Path) -> Path:
    """训练语料复制到 WSL 本地盘，计时不受跨系统文件访问的开销影响。"""
    dst = WORK_DIR / "corpus" / src.parent.name / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return dst


def median_range(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def repeated_train(name, corpus, order, memory=DEFAULT_MEMORY, prune=None, repeats=REPEATS):
    """同一配置训练 repeats 次，返回每次的计时与最后一份模型。

    每次都写到同一个路径（后一次覆盖前一次），同时记录每次 ARPA 的 SHA256，
    用来确认重复训练的产物逐字节一致，时间差异只来自系统调度与缓存。

    计时前先不计时地训练一次（预热）。完整运行里出现过 lmplz 启动后迟迟不进入第一阶段的情况，
    最长等了 4.4 秒，而同样的训练单独在 bash 或新的 Python 进程里启动时，这段等待不超过 0.1 秒；
    等待期间 CPU 时间不增加。所以结果里另记「启动等待」与 CPU 时间，墙钟时间以预热后的中位数为准。
    """
    arpa = MODELS / f"{name}.arpa"
    if repeats > 1:
        kw.train(corpus, arpa, order, memory=memory, prune=prune, temp_dir=WORK_DIR / "tmp")
    runs, stalled = [], []
    max_retries = RETRY_FACTOR * repeats
    while len(runs) < repeats:
        res = kw.train(corpus, arpa, order, memory=memory, prune=prune, temp_dir=WORK_DIR / "tmp")
        res["arpa_sha256"] = sha256_file(arpa)
        # 墙钟时间比 CPU 时间对应的正常耗时长出很多，说明这次在等内存换页或被其他程序抢占，
        # 不代表 lmplz 本身的速度；/usr/bin/time 的统计行缺失时也无法判断，同样重测。
        # 这类干扰是成片出现的，所以每次重测前等一会儿，等待时间逐次加倍，最多 8 秒
        bad = res["time_parse_failed"] or is_stalled(res)
        if bad and len(stalled) < max_retries:
            stalled.append(res)
            why = "没有拿到 CPU 时间" if res["time_parse_failed"] else \
                f"{res['wall_seconds']:.2f}s 中 CPU 只用了 {cpu_of(res):.2f}s"
            wait = min(2 ** (len(stalled) - 1), 8)
            log(f"  {name}：{why}，判为受干扰，{wait}s 后重测（第 {len(stalled)}/{max_retries} 次重测）")
            time.sleep(wait)
            continue
        # 重测次数用完仍受干扰的，照样收下，但打上标记，报告里单独注明
        res["accepted_despite_interference"] = bool(bad)
        res["repeat"] = len(runs) + 1
        runs.append(res)
        log(f"  {name} 第 {len(runs)}/{repeats} 次：{res['wall_seconds']:.2f}s，CPU {cpu_of(res):.2f}s，"
            f"峰值 {res.get('peak_rss_mb', 0):.0f} MiB" + ("（重测次数用完，仍受干扰）" if bad else ""))
    for r in runs:
        r["stalled_runs_discarded"] = len(stalled)
    if stalled:
        runs[-1]["stalled_wall_seconds"] = [s["wall_seconds"] for s in stalled]
    return arpa, runs


RETRY_FACTOR = 3  # 每个配置最多重测 3 × 计时次数


def cpu_of(res) -> float:
    return res.get("user_seconds", 0.0) + res.get("sys_seconds", 0.0)


STALL_RATIO = 2.0       # 墙钟时间超过 CPU 时间的这个倍数视为被拖慢
STALL_MIN_EXCESS = 1.0  # 且多出来的部分至少 1 秒，避免对 0.2 秒的小训练过度敏感


def is_stalled(res) -> bool:
    """判断一次训练是不是在等内存换页或被其他程序抢占，墙钟时间不能代表 lmplz 本身的速度。

    lmplz 的排序与计算阶段是多线程的，未受干扰时墙钟时间介于 CPU 时间的 0.6 到 1.3 倍之间
    （1 月语料五元 1.47–1.78 秒对 CPU 2.1 秒；六个月语料二元 3.0–3.3 秒对 CPU 2.9–3.1 秒）。
    受干扰时墙钟是 CPU 的 3 到 6 倍（如 2.72 秒对 0.48 秒）。所以墙钟超过 CPU 的 STALL_RATIO 倍、
    且多出 STALL_MIN_EXCESS 秒以上，才判为被拖慢。最初按「CPU 的一半」为基准判断，在六个月语料上
    把 3.90 秒对 CPU 3.03 秒这种正常波动也判成了受干扰，改为直接和 CPU 时间比。
    """
    cpu = cpu_of(res)
    if cpu <= 0:
        return False
    return res["wall_seconds"] > cpu * STALL_RATIO and res["wall_seconds"] - cpu > STALL_MIN_EXCESS


def train_summary(runs):
    last = runs[-1]
    stage_names = [s["name"] for s in last["stages"]]
    cpu = [r["user_seconds"] + r["sys_seconds"] for r in runs if "user_seconds" in r]
    return {
        "wall_seconds": median_range([r["wall_seconds"] for r in runs]),
        # CPU 时间（user + sys）不受等待 I/O、被别的进程抢占的影响；墙钟时间抖动时用它判断是不是程序本身变慢
        "cpu_seconds": median_range(cpu) if cpu else None,
        # 从启动 lmplz 到它打印第一阶段标题；正常不到 0.1 秒
        "startup_seconds": median_range([r["startup_seconds"] or 0.0 for r in runs]),
        "kenlm_real_seconds": median_range([r["kenlm_real_seconds"] for r in runs if "kenlm_real_seconds" in r]),
        "peak_rss_mb": median_range([r["peak_rss_mb"] for r in runs if "peak_rss_mb" in r]),
        "kenlm_reported_rssmax_mb": median_range([r["kenlm_reported_rssmax_mb"] for r in runs
                                                  if "kenlm_reported_rssmax_mb" in r]),
        "stage_seconds_median": {name: statistics.median(r["stages"][i]["seconds"] for r in runs)
                                 for i, name in enumerate(stage_names)},
        "stalled_runs_discarded": last.get("stalled_runs_discarded", 0),
        "stalled_wall_seconds": last.get("stalled_wall_seconds", []),
        "accepted_despite_interference": sum(bool(r.get("accepted_despite_interference")) for r in runs),
        "arpa_bytes": last["arpa_bytes"],
        "ngram_counts": last["ngram_counts"],
        "discounts": last["discounts"],
        "fallback_discount_used": any(r["fallback_discount_used"] for r in runs),
        "deterministic_arpa": len({r["arpa_sha256"] for r in runs}) == 1,
    }


def eval_on(model, split_name):
    sents = read_sentences(SPLIT_DIR / f"{split_name}.txt")
    return kw.evaluate(model, sents)


def save(name, payload):
    payload = {"experiment": name, "finished_at": now_iso(), "kenlm_commit": KENLM_COMMIT, **payload}
    write_json(RUNS_DIR / f"{name}.json", payload)
    return payload


def type_stats(sentences) -> dict:
    counts = {}
    for s in sentences:
        for w in s:
            counts[w] = counts.get(w, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1)
    return {"tokens": sum(counts.values()), "types": len(counts), "hapax_types": hapax,
            "hapax_type_share": hapax / len(counts) if counts else 0.0}


def e0_tag_cleaning():
    """同一批训练/测试文章：清洗版 vs 带词性版，各训练一个三元 KN 模型。"""
    out = {}
    for label, d in (("clean", SPLIT_DIR), ("tagged", TAGGED_DIR)):
        train_path = local_copy(d / "train.txt")
        arpa, runs = repeated_train(f"e0_{label}_3", train_path, 3, repeats=1)
        model = kw.load(arpa)
        test = read_sentences(d / "test.txt")
        ev = kw.evaluate(model, test)
        gen = sampling.Generator(model, sampling.vocab_from_arpa(arpa))
        prompt = MAIN_PROMPT if label == "clean" else _tag_prompt(read_sentences(d / "train.txt"), MAIN_PROMPT)
        samples = [gen.generate(prompt, temperature=1.0, top_k=20, seed=s, max_tokens=40)
                   for s in GEN_SEEDS]
        out[label] = {
            "train": type_stats(read_sentences(d / "train.txt")),
            "training": train_summary(runs),
            "test": {k: ev[k] for k in ("events", "oov_events", "oov_rate", "ppl_including_oov",
                                        "ppl_excluding_oov", "cross_entropy_bits")},
            "prompt": prompt,
            "samples": [{"seed": g["params"]["seed"], "text": " ".join(g["generated"]),
                         "hit_eos": g["hit_eos"]} for g in samples],
        }
        del model
    return out


def _tag_prompt(tagged_train, words):
    """带词性版的提示词：每个词取它在带词性训练集里出现次数最多的那个「词/词性」。"""
    freq = {}
    for s in tagged_train:
        for tok in s:
            freq[tok] = freq.get(tok, 0) + 1
    best = {}
    for tok, c in freq.items():
        word = tok.rpartition("/")[0]
        if word and (word not in best or c > freq[best[word]]):
            best[word] = tok
    return [best.get(w, w) for w in words]


E1_MAX_TRAIN_MONTH = "199801"  # E1 只用 1 月的训练文章，见 e1_smoothing 的说明


def _train_up_to_month(last_month):
    """主划分训练集里、日期不晚于 last_month 的那部分文章，写成一个训练文件，返回 (路径, 句子)。"""
    sents = read_sentences(SPLIT_DIR / "train.txt")
    arts = (SPLIT_DIR / "train_articles.txt").read_text(encoding="utf-8").split()
    keep = [s for s, a in zip(sents, arts) if a[:6] <= last_month]
    path = WORK_DIR / "corpus" / "e1" / f"train_upto_{last_month}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(" ".join(s) for s in keep) + "\n", encoding="utf-8")
    return path, keep


def e1_smoothing():
    """MLE / Add-k / JM / Witten-Bell / Kneser-Ney（本模块实现）与 KenLM 的 Modified KN 对照。

    本模块的计数用 Python 字典存，六个月的训练集（580 万词）要占约 4 GB 内存，这台机器吃不消；
    E1 比较的是平滑方法，结论不随语料量变，所以只用主划分训练集里 1 月的文章（约 90 万词）训练，
    验证集、测试集仍是主划分的全部六个月。KenLM 对照用同一份训练文件。
    """
    if pfr_data.CORPUS_TAG == "figshare_jan":
        train_path, train = local_copy(SPLIT_DIR / "train.txt"), read_sentences(SPLIT_DIR / "train.txt")
    else:
        train_path, train = _train_up_to_month(E1_MAX_TRAIN_MONTH)
    log(f"  E1 训练句 {len(train):,}、词 {sum(len(s) for s in train):,}")
    dev = read_sentences(SPLIT_DIR / "dev.txt")
    test = read_sentences(SPLIT_DIR / "test.txt")
    t0 = time.perf_counter()
    counts = sl.TrigramCounts(train)
    count_seconds = time.perf_counter() - t0
    log(f"  三元计数：{len(counts.c3):,} 个三元，{len(counts.c2):,} 个二元，词表 {counts.V:,}，{count_seconds:.1f}s")

    # Add-k 的 k 与 JM 的 λ 都在验证集上选，测试集只用来报告。k 按对数间隔从 1 往下扫；
    # 最优值落在网格边界时继续往下扩，直到验证集困惑度开始回升，保证选到的是网格内部的极小值
    def addk_ppl(k):
        return sl.evaluate(lambda w, u, v: counts.p_addk(w, u, v, k), dev, counts.vocab)["ppl"]

    addk_grid = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 5e-4, 1e-4, 5e-5, 1e-5]
    addk_dev = {k: addk_ppl(k) for k in addk_grid}
    while min(addk_dev, key=addk_dev.get) == min(addk_dev) and min(addk_dev) > 1e-9:
        for k in (min(addk_dev) / 2, min(addk_dev) / 10):
            addk_dev[k] = addk_ppl(k)
    best_k = min(addk_dev, key=addk_dev.get)
    log(f"  Add-k 验证集：{ {f'{k:g}': round(p, 1) for k, p in sorted(addk_dev.items(), reverse=True)} } -> k={best_k:g}")

    # λ0 固定 0.01 给均匀分布兜底，其余三项在网格上搜，四项之和为 1
    jm_grid = []
    for l3 in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        for l2 in (0.2, 0.3, 0.4, 0.5):
            l1 = round(1 - l3 - l2 - 0.01, 4)
            if l1 >= 0.04:
                jm_grid.append((l3, l2, l1, 0.01))
    jm_dev = {lam: sl.evaluate(lambda w, u, v, lam=lam: counts.p_jm(w, u, v, lam), dev, counts.vocab)["ppl"]
              for lam in jm_grid}
    best_lam = min(jm_dev, key=jm_dev.get)
    log(f"  JM 验证集最优 λ={best_lam}，PPL {jm_dev[best_lam]:.1f}（共试 {len(jm_grid)} 组）")

    methods = {
        "MLE": counts.p_mle,
        "Add-1": lambda w, u, v: counts.p_addk(w, u, v, 1.0),
        f"Add-k (k={best_k:g})": lambda w, u, v: counts.p_addk(w, u, v, best_k),
        "Jelinek-Mercer": lambda w, u, v: counts.p_jm(w, u, v, best_lam),
        "Witten-Bell": counts.p_wb,
        "Kneser-Ney（本模块）": counts.p_kn,
    }
    results = {}
    for name, fn in methods.items():
        results[name] = {"dev": sl.evaluate(fn, dev, counts.vocab), "test": sl.evaluate(fn, test, counts.vocab)}
        log(f"  {name}: 测试集 PPL {results[name]['test']['ppl']:.2f}，"
            f"零概率事件 {results[name]['test']['zero_prob_events']:,}")

    # 对照：KenLM 的三元 Modified KN，训练句、测试句与上面完全相同
    arpa, runs = repeated_train("e1_kenlm_3", train_path, 3, repeats=1)
    model = kw.load(arpa)
    kenlm_test = kw.evaluate(model, test)
    kenlm_dev = kw.evaluate(model, dev)
    # KenLM 的一元表另含 <s> 与 <unk>，本模块的 c1 只含训练词形与 </s>，所以加 2 再比
    ours_counts = [len(counts.c1) + 2, len(counts.a2), len(counts.c3)]
    check = {
        "kenlm_ngram_counts": runs[-1]["ngram_counts"],
        "our_ngram_counts_1_2_3": ours_counts,
        "kenlm_discounts": runs[-1]["discounts"],
        "our_discounts": {"1": counts.D1, "2": counts.D2, "3": counts.D3},
        "kenlm_test_ppl": kenlm_test["ppl_including_oov"],
        "ours_test_ppl": results["Kneser-Ney（本模块）"]["test"]["ppl"],
    }
    log(f"  核对 KenLM：条目数 {check['kenlm_ngram_counts']} vs 本模块 {ours_counts}；"
        f"测试 PPL {check['kenlm_test_ppl']:.2f} vs {check['ours_test_ppl']:.2f}")

    # 逐位置对照：取测试集前 200 句，比较两边给每个位置的概率
    diffs = []
    for s in test[:200]:
        ours = [counts.p_kn(w, u, v) for u, v, w in sl.events([s], counts.vocab)]
        theirs = [10 ** lp for lp, _, _ in model.full_scores(" ".join(s), bos=True, eos=True)]
        diffs.extend(abs(math.log10(a) - math.log10(b)) for a, b in zip(ours, theirs))
    check["position_log10_absdiff"] = {"max": max(diffs), "mean": sum(diffs) / len(diffs), "positions": len(diffs)}

    # 每种平滑在真实历史上对整个词表求和，应为 1（MLE 在没见过的历史上没有定义，和为 0）
    first_events = list(sl.events([MAIN_PROMPT], counts.vocab))
    histories = [(u, v) for u, v, _ in first_events[:4]] + [("从未", "出现")]
    normalization = {name: sl.normalization_check(counts, fn, histories) for name, fn in methods.items()}

    # 作业给的开头逐个三元事件：训练集里的计数，以及各平滑给出的概率
    prompt_table = []
    for u, v, w in first_events[:-1]:
        hist, cont = counts.history_counts(u, v, w)
        prompt_table.append({"context": [u, v], "word": w, "history_count": hist, "ngram_count": cont,
                             **{name: fn(w, u, v) for name, fn in methods.items()}})

    return {
        "train_sentences": len(train), "train_tokens": sum(len(s) for s in train),
        "train_scope": "全部训练集" if pfr_data.CORPUS_TAG == "figshare_jan" else f"训练集中 {E1_MAX_TRAIN_MONTH} 及以前的文章",
        "count_seconds": count_seconds, "vocab_size": counts.V,
        "addk_dev_ppl": {f"{k:g}": v for k, v in sorted(addk_dev.items(), reverse=True)}, "best_k": best_k,
        "jm_dev_ppl": {",".join(map(str, lam)): v for lam, v in jm_dev.items()}, "best_lambda": best_lam,
        "methods": results,
        "kenlm": {"dev": kenlm_dev, "test": kenlm_test, "training": train_summary(runs)},
        "kenlm_check": check,
        "normalization": normalization,
        "prompt_trigrams": prompt_table,
        "zero_prob_examples": zero_prob_examples(counts, test, best_k, best_lam),
        "continuation_diversity": sl.continuation_table(counts, top=15, min_count=20),
    }


def zero_prob_examples(counts, test, best_k, best_lam, n=6):
    """从测试集里挑几个 MLE 给 0 的三元事件，看各平滑分别给了多少概率。"""
    picked, seen = [], set()
    for s in test:
        for u, v, w in sl.events([s], counts.vocab):
            if u is None or w in (sl.UNK, sl.EOS) or (u, v, w) in seen:
                continue
            hist = counts.h3.get((u, v), 0)
            if hist >= 20 and counts.c3.get((u, v, w), 0) == 0 and counts.c2.get((v, w), 0) > 0:
                seen.add((u, v, w))
                picked.append({
                    "context": [u, v], "word": w, "history_count": hist,
                    "bigram_count": counts.c2[(v, w)],
                    "MLE": counts.p_mle(w, u, v),
                    "Add-1": counts.p_addk(w, u, v, 1.0),
                    f"Add-k (k={best_k:g})": counts.p_addk(w, u, v, best_k),
                    "Jelinek-Mercer": counts.p_jm(w, u, v, best_lam),
                    "Witten-Bell": counts.p_wb(w, u, v),
                    "Kneser-Ney": counts.p_kn(w, u, v),
                })
            if len(picked) >= n:
                return picked
    return picked


def ensure_probing(name):
    """E2 训练出的 ARPA 转成 probing 二进制，并把词表存成 JSON，后面各实验加载更快。

    六个月语料的五元 ARPA 加载一次要 12 秒，E7 E8 E10 E11 E12 要反复加载；二进制与 ARPA 概率完全相同
    （E6 逐位置核对过），生成用的词表仍从 ARPA 的一元表取。转换耗时不计入 E2 的训练时间。
    """
    arpa, binary, vocab = MODELS / f"{name}.arpa", MODELS / f"{name}.probing", MODELS / f"{name}.vocab.json"
    if not binary.exists() or binary.stat().st_mtime < arpa.stat().st_mtime:
        kw.build_binary(arpa, binary, "probing")
    if not vocab.exists() or vocab.stat().st_mtime < arpa.stat().st_mtime:
        write_json(vocab, sampling.vocab_from_arpa(arpa))


def load_model(n_or_name):
    """加载模型：参数是阶数（E2 的 kn{n}）或模型名；有二进制用二进制，没有就读 ARPA。"""
    name = f"kn{n_or_name}" if isinstance(n_or_name, int) else n_or_name
    binary = MODELS / f"{name}.probing"
    return kw.load(binary if binary.exists() else MODELS / f"{name}.arpa")


def vocab_of(n_or_name):
    name = f"kn{n_or_name}" if isinstance(n_or_name, int) else n_or_name
    vocab = MODELS / f"{name}.vocab.json"
    return read_json(vocab) if vocab.exists() else sampling.vocab_from_arpa(MODELS / f"{name}.arpa")


def e2_orders():
    """2-6 元，Modified KN，-S 512M，不剪枝，每个阶数预热一次后计时训练 REPEATS 次。"""
    corpus = local_copy(SPLIT_DIR / "train.txt")
    dev = read_sentences(SPLIT_DIR / "dev.txt")
    test = read_sentences(SPLIT_DIR / "test.txt")
    out = {}
    for n in ORDERS:
        arpa, runs = repeated_train(f"kn{n}", corpus, n)
        ensure_probing(f"kn{n}")
        model = load_model(n)
        out[str(n)] = {"training": train_summary(runs), "runs": runs,
                       "dev": kw.evaluate(model, dev), "test": kw.evaluate(model, test)}
        log(f"  {n}-gram：验证 PPL {out[str(n)]['dev']['ppl_including_oov']:.2f}，"
            f"测试 PPL {out[str(n)]['test']['ppl_including_oov']:.2f}")
        del model
    best = min(ORDERS, key=lambda n: out[str(n)]["dev"]["ppl_including_oov"])
    return {"orders": out, "best_order_by_dev": best}


def e3_pruning():
    """三元与五元各训练一个剪枝版：三阶及以上计数不超过 1 的条目删掉，一元二元保留。

    lmplz 的剪枝按原始计数判断；低阶里被更高阶「引用」着的条目不会删（ARPA 要求前缀完整）。
    """
    corpus = local_copy(SPLIT_DIR / "train.txt")
    dev = read_sentences(SPLIT_DIR / "dev.txt")
    test = read_sentences(SPLIT_DIR / "test.txt")
    e2 = read_json(RUNS_DIR / "E2_orders.json")["orders"]
    out = {}
    for n, prune in ((3, [0, 0, 1]), (5, [0, 0, 1, 1, 1])):
        arpa, runs = repeated_train(f"kn{n}_prune", corpus, n, prune=prune)
        model = kw.load(arpa)
        full = e2[str(n)]
        # 两种格式的体积都记下：ARPA 是文本，probing 二进制才是实际加载用的大小
        full_bin = kw.build_binary(MODELS / f"kn{n}.arpa", MODELS / f"kn{n}_probing.bin")
        pruned_bin = kw.build_binary(arpa, MODELS / f"kn{n}_prune_probing.bin")
        out[str(n)] = {
            "prune": prune, "training": train_summary(runs),
            "dev": kw.evaluate(model, dev), "test": kw.evaluate(model, test),
            "probing_bytes": pruned_bin["bytes"],
            "unpruned": {"arpa_bytes": full["training"]["arpa_bytes"],
                         "probing_bytes": full_bin["bytes"],
                         "ngram_counts": full["training"]["ngram_counts"],
                         "wall_seconds": full["training"]["wall_seconds"],
                         "test_ppl": full["test"]["ppl_including_oov"],
                         "dev_ppl": full["dev"]["ppl_including_oov"]},
        }
        for p in (MODELS / f"kn{n}_probing.bin", MODELS / f"kn{n}_prune_probing.bin"):
            p.unlink()
        p = out[str(n)]
        log(f"  {n}-gram 剪枝：ARPA {full['training']['arpa_bytes'] / 2**20:.1f} -> "
            f"{p['training']['arpa_bytes'] / 2**20:.1f} MiB，测试 PPL {full['test']['ppl_including_oov']:.2f} -> "
            f"{p['test']['ppl_including_oov']:.2f}")
        del model
    return out


def e4_memory():
    """三元不剪枝，排序内存 -S 128M / 512M / 1G，每档 3 次，看耗时与峰值内存，产物应逐字节一致。"""
    corpus = local_copy(SPLIT_DIR / "train.txt")
    out = {}
    for mem in ("128M", "512M", "1G"):
        _, runs = repeated_train(f"kn3_mem{mem}", corpus, 3, memory=mem)
        out[mem] = {"training": train_summary(runs), "arpa_sha256": runs[-1]["arpa_sha256"]}
    out["all_arpa_identical"] = len({v["arpa_sha256"] for v in out.values() if isinstance(v, dict)}) == 1
    return out


def gen_record(g, index, model) -> dict:
    """一条生成结果的完整记录：原文、逐步轨迹、E7 的定量指标、E9 的重合度指标。"""
    return add_overlap(model_record(g, model), g, index)


def add_overlap(rec, g, index) -> dict:
    """给 model_record 的结果补上与训练集的重合度指标（E9），只用训练集索引，不需要模型。"""
    tokens = g["generated"]
    full = g["prefix"] + tokens
    copied = index.longest_copied_span(full, gen_start=len(g["prefix"]))
    nan = float("nan")
    # 窗口从提示词末尾四个词开始，保证每个 5-gram 至少含一个生成词
    rec["overlap_5gram"] = index.overlap_rate(g["prefix"][-4:] + tokens, 5) if tokens else nan
    rec["longest_copied_span"] = copied
    rec["copied_10plus"] = copied >= 10
    return rec


def model_record(g, model) -> dict:
    """一条生成结果里只依赖模型的那部分记录：原文、逐步轨迹、E7 的定量指标。"""
    tokens = g["generated"]
    steps = g["steps"]
    rep = sampling.repetition_stats(tokens)
    # 反向 PPL：把「提示词 + 生成内容」交给同一个模型打分，只统计生成部分（含结束的 </s>）
    scores = list(model.full_scores(" ".join(g["prefix"] + tokens), bos=True, eos=g["hit_eos"]))
    gen_scores = scores[len(g["prefix"]):]
    self_ppl = 10 ** (-sum(s for s, _, _ in gen_scores) / len(gen_scores)) if gen_scores else float("nan")
    nan = float("nan")
    return {
        "params": g["params"], "text": "".join(tokens), "tokens": tokens, "hit_eos": g["hit_eos"],
        **rep,
        "self_ppl": self_ppl,
        "top1_prob_mean": statistics.mean(s["top1_prob"] for s in steps) if steps else nan,
        "chosen_prob_mean": statistics.mean(s["model_prob"] for s in steps) if steps else nan,
        "chose_top1_rate": statistics.mean(s["chose_top1"] for s in steps) if steps else nan,
        "entropy_bits_mean": statistics.mean(s["entropy_bits"] for s in steps) if steps else nan,
        "matched_order_mean": statistics.mean(s["matched_order"] for s in steps) if steps else nan,
        # 逐步轨迹只留每步选中的词、概率与匹配阶数；前五候选只留第一步（首词分布）
        "steps": [{k: s[k] for k in ("chosen", "model_prob", "top1_prob", "entropy_bits", "matched_order")}
                  for s in steps],
        "first_step_top": steps[0]["top"] if steps else [],
    }


def _mean_finite(values) -> float:
    """忽略 NaN 后取平均（生成长度为 0 时部分指标没有定义）；全是 NaN 时返回 NaN。"""
    vals = [v for v in values if not math.isnan(v)]
    return statistics.mean(vals) if vals else float("nan")


def aggregate(records) -> dict:
    keys = ["length", "type_token_ratio", "repeated_2gram_rate", "repeated_3gram_rate", "self_ppl",
            "top1_prob_mean", "chose_top1_rate", "entropy_bits_mean", "matched_order_mean",
            "overlap_5gram", "longest_copied_span"]
    out = {k: _mean_finite(r[k] for r in records) for k in keys}
    out["eos_rate"] = statistics.mean(r["hit_eos"] for r in records)
    out["copied_10plus_rate"] = statistics.mean(r["copied_10plus"] for r in records)
    out["n"] = len(records)
    return out


def query_index(token_lists):
    """只为要查询的这些词序列建训练集 n-gram 索引（1-12 元）。

    先收集它们里面出现的 n-gram，再扫一遍训练集，只记命中的那部分。六个月的训练集全量建索引
    要好几 GB 内存；这样做查询结果与全量索引完全相同，内存只和要查的文本长度有关。
    """
    wanted = sampling.TrainIndex.collect_queries(token_lists, max_n=12)
    log(f"  为 {len(token_lists)} 条文本建训练集 n-gram 索引（1-12 元，供重合度检测）")
    return sampling.TrainIndex(train_sentences(), max_n=12, wanted=wanted)


def _generate_set(gen, model, temperature, top_k, top_p):
    """同一组采样参数下，按 STAT_SEEDS 各生成一次；先算只依赖模型的指标，重合度留到建好索引后再补。"""
    raw = [gen.generate(MAIN_PROMPT, temperature, top_k, top_p, max_tokens=MAX_GEN_TOKENS, seed=s)
           for s in STAT_SEEDS]
    return [(g, model_record(g, model)) for g in raw]


def _finish_set(pairs, index):
    """补上重合度指标并汇总。

    定量指标用全部 20 次求平均（records 里去掉逐步轨迹，只留指标与原文）；
    报告里展示原文的是前三个种子（GEN_SEEDS），这三条保留完整轨迹放在 samples。
    """
    recs = [add_overlap(rec, g, index) for g, rec in pairs]
    shown = [r for r in recs if r["params"]["seed"] in GEN_SEEDS]
    slim = [{k: v for k, v in r.items() if k not in ("steps", "first_step_top")} for r in recs]
    return {"summary": aggregate(recs), "records": slim, "samples": shown}


def e7_generation():
    """最优阶数模型上：temperature × top-k 网格、两档 top-p、贪心；每组参数 20 个种子。

    另用 2-6 元各自的模型、同一组参数（T=1.0，k=50）各生成 20 次，给 E9 做「阶数与照搬」的对照。
    """
    best = read_json(RUNS_DIR / "E2_orders.json")["best_order_by_dev"]
    arpa = MODELS / f"kn{best}.arpa"
    model = kw.load(arpa)
    gen = sampling.Generator(model, sampling.vocab_from_arpa(arpa))
    cells = []
    for t in (0.7, 1.0, 1.3):
        for k in (10, 50, 0):
            cells.append({"name": f"T={t}, k={k if k else '全词表'}", "temperature": t, "top_k": k, "top_p": 1.0})
    for p in (0.5, 0.9):
        cells.append({"name": f"T=1.0, top-p={p}", "temperature": 1.0, "top_k": 0, "top_p": p})
    # 第一遍：生成，并算只依赖模型的指标（重合度要等索引建好再补）
    pending = []
    for cell in cells:
        pending.append((cell, _generate_set(gen, model, cell["temperature"], cell["top_k"], cell["top_p"])))
        log(f"  {cell['name']}：已生成 {len(STAT_SEEDS)} 次")
    g_greedy = gen.generate(MAIN_PROMPT, greedy=True, max_tokens=MAX_GEN_TOKENS)
    greedy_pair = (g_greedy, model_record(g_greedy, model))
    del model

    order_pending = {}
    for n in ORDERS:
        m = load_model(n)
        g = sampling.Generator(m, vocab_of(n))
        g_greedy_n = g.generate(MAIN_PROMPT, greedy=True, max_tokens=MAX_GEN_TOKENS)
        order_pending[str(n)] = (_generate_set(g, m, 1.0, 50, 1.0), (g_greedy_n, model_record(g_greedy_n, m)))
        log(f"  {n}-gram（T=1.0, k=50）：已生成 {len(STAT_SEEDS)} 次")
        del m, g

    # 第二遍：只为这些生成结果建训练集索引，补上重合度指标
    all_pairs = [p for _, pairs in pending for p in pairs] + [greedy_pair]
    for pairs, gp in order_pending.values():
        all_pairs += pairs + [gp]
    index = query_index([g["prefix"] + g["generated"] for g, _ in all_pairs])
    grid = []
    for cell, pairs in pending:
        grid.append({**cell, **_finish_set(pairs, index)})
        s = grid[-1]["summary"]
        log(f"  {cell['name']}：平均长度 {s['length']:.1f}，自然结束 {s['eos_rate']:.0%}，"
            f"最长照搬 {s['longest_copied_span']:.1f} 词")
    greedy = add_overlap(greedy_pair[1], greedy_pair[0], index)
    by_order = {}
    for n, (pairs, (gg, grec)) in order_pending.items():
        by_order[n] = {**_finish_set(pairs, index), "greedy": add_overlap(grec, gg, index)}
        s = by_order[n]["summary"]
        log(f"  {n}-gram（T=1.0, k=50）：平均长度 {s['length']:.1f}，最长照搬 {s['longest_copied_span']:.1f} 词")
    return {"model_order": best, "prompt": MAIN_PROMPT, "stat_seeds": STAT_SEEDS, "shown_seeds": GEN_SEEDS,
            "grid": grid, "greedy": greedy, "by_order": by_order}


SCALE_ORDERS = [2, 3, 5]


def e5_scale():
    """训练集规模 10/20/35/50/全部（约 90 万）词，二元、三元、五元各训练 3 次，统一在主划分的验证/测试集上评估。

    规模样本都从训练集按整篇文章抽取，同一种子下小样本是大样本的子集（data.sample_by_size）。
    同时记录词表随语料增长的情况（Heaps 定律 V = K·N^β）。
    """
    samples = read_json(RESULTS_DIR / "s1_scale_samples.json")
    dev = read_sentences(SPLIT_DIR / "dev.txt")
    test = read_sentences(SPLIT_DIR / "test.txt")
    points = []
    for info in samples:
        src = SCALE_DIR / Path(info["path"].replace("\\", "/")).name
        corpus = local_copy(src)
        sents = read_sentences(src)
        point = {"target_words": info["target_words"], "actual_words": info["actual_words"],
                 "article_count": info["article_count"], "is_full_train": info["is_full_train"],
                 "vocab": type_stats(sents), "orders": {}}
        for n in SCALE_ORDERS:
            arpa, runs = repeated_train(f"scale_{info['target_words'] // 1000}k_kn{n}", corpus, n)
            model = kw.load(arpa)
            point["orders"][str(n)] = {"training": train_summary(runs),
                                       "dev": kw.evaluate(model, dev), "test": kw.evaluate(model, test)}
            del model
            arpa.unlink()
        points.append(point)
        p3 = point["orders"]["3"]["test"]
        log(f"  {info['actual_words']:>9,} 词：三元测试 PPL {p3['ppl_including_oov']:.2f}，OOV {p3['oov_rate']:.2%}")
    # Heaps 定律：log V = log K + β log N，最小二乘拟合
    xs = [math.log(p["vocab"]["tokens"]) for p in points]
    ys = [math.log(p["vocab"]["types"]) for p in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    heaps = {"beta": beta, "K": math.exp(my - beta * mx)}
    # 相邻两点之间，每增加 10 万词测试 PPL 下降多少
    marginal = {}
    for n in SCALE_ORDERS:
        rows = []
        for a, b in zip(points, points[1:]):
            pa = a["orders"][str(n)]["test"]["ppl_including_oov"]
            pb = b["orders"][str(n)]["test"]["ppl_including_oov"]
            added = (b["actual_words"] - a["actual_words"]) / 1e5
            rows.append({"from_words": a["actual_words"], "to_words": b["actual_words"],
                         "ppl_from": pa, "ppl_to": pb, "ppl_drop_per_100k": (pa - pb) / added,
                         "relative_drop": (pa - pb) / pa})
        marginal[str(n)] = rows
    return {"points": points, "heaps": heaps, "marginal": marginal}


def e8_matched_orders():
    """E2 的各阶模型在测试集上逐位置统计实际匹配到的 n-gram 阶数（KenLM full_scores 给出）。

    再把测试位置按「这个位置能否在训练集里找到完整的高阶上下文」分组，看高阶模型在哪些位置真正起作用。
    """
    e2 = read_json(RUNS_DIR / "E2_orders.json")["orders"]
    out = {}
    for n in ORDERS:
        hist = e2[str(n)]["test"]["matched_orders"]
        total = sum(hist.values())
        out[str(n)] = {"counts": hist, "share": {k: v / total for k, v in hist.items()},
                       "full_order_share": hist.get(str(n), 0) / total,
                       "mean_matched_order": sum(int(k) * v for k, v in hist.items()) / total,
                       "oov_events": e2[str(n)]["test"]["oov_events"],
                       "test_ppl": e2[str(n)]["test"]["ppl_including_oov"]}
        log(f"  {n}-gram：满阶匹配 {out[str(n)]['full_order_share']:.1%}，平均匹配阶数 "
            f"{out[str(n)]['mean_matched_order']:.2f}")
    # 同一个测试位置在 5 元模型下匹配到 k 阶时，它在 2..4 元模型下的对数概率分别是多少：
    # 解释「阶数加上去，困惑度几乎不动」来自哪些位置
    test = read_sentences(SPLIT_DIR / "test.txt")
    per_model = {}
    for n in (2, 3, 4, 5):
        m = kw.load(MODELS / f"kn{n}.arpa")
        per_model[n] = kw.position_scores(m, test)
        del m
    buckets = {}
    for i, (_, len5, oov5) in enumerate(per_model[5]):
        key = "OOV" if oov5 else str(len5)
        b = buckets.setdefault(key, {"positions": 0, **{f"log10_sum_{n}": 0.0 for n in (2, 3, 4, 5)}})
        b["positions"] += 1
        for n in (2, 3, 4, 5):
            b[f"log10_sum_{n}"] += per_model[n][i][0]
    for b in buckets.values():
        for n in (2, 3, 4, 5):
            b[f"mean_log10_{n}"] = b.pop(f"log10_sum_{n}") / b["positions"]
    return {"by_model": out, "by_5gram_match_length": buckets}


def e9_leakage():
    """汇总 E7 的全部生成：照搬训练集的最长连续词串、5-gram 重合率，按阶数与温度分组。"""
    e7 = read_json(RUNS_DIR / "E7_generation.json")
    by_order = {n: v["summary"] for n, v in e7["by_order"].items()}
    by_order_greedy = {n: {"longest_copied_span": v["greedy"]["longest_copied_span"],
                           "overlap_5gram": v["greedy"]["overlap_5gram"], "text": v["greedy"]["text"]}
                       for n, v in e7["by_order"].items()}
    by_temp = {}
    for cell in e7["grid"]:
        if cell["top_p"] != 1.0:  # top-p 那两格另算，不混进同温度的 T×k 网格
            continue
        t = str(cell["temperature"])
        by_temp.setdefault(t, []).extend(cell["records"])
    by_temp = {t: {"longest_copied_span": statistics.mean(r["longest_copied_span"] for r in recs),
                   "overlap_5gram": _mean_finite(r["overlap_5gram"] for r in recs),
                   "copied_10plus_rate": statistics.mean(r["copied_10plus"] for r in recs), "n": len(recs)}
               for t, recs in by_temp.items()}
    # 各阶数下「照搬了多少个词」的分布：每条生成的最长照搬长度落在哪个区间
    span_hist = {}
    for n, v in e7["by_order"].items():
        spans = [r["longest_copied_span"] for r in v["records"]]
        span_hist[n] = {b: sum(lo <= s <= hi for s in spans) for b, (lo, hi) in
                        {"0-3": (0, 3), "4-6": (4, 6), "7-9": (7, 9), "10+": (10, 10 ** 6)}.items()}
    # 照搬最长的几条，附上训练集里的原句，报告里直接展示
    all_recs = [r for c in e7["grid"] for r in c["records"]]
    for v in e7["by_order"].values():
        all_recs.extend(v["records"])
    all_recs.sort(key=lambda r: -r["longest_copied_span"])
    top = all_recs[:6]
    index = query_index([e7["prompt"] + r["tokens"] for r in top])
    examples = [{"params": r["params"], "text": r["text"], "longest_copied_span": r["longest_copied_span"],
                 "source": find_source_sentence(e7["prompt"] + r["tokens"], len(e7["prompt"]), index)}
                for r in top]
    return {"by_order": by_order, "by_order_greedy": by_order_greedy, "by_temperature": by_temp,
            "copied_span_histogram": span_hist, "top_copies": examples}


_TRAIN_SENTS = None


def train_sentences():
    global _TRAIN_SENTS
    if _TRAIN_SENTS is None:
        _TRAIN_SENTS = read_sentences(SPLIT_DIR / "train.txt")
    return _TRAIN_SENTS


def find_source_sentence(tokens, gen_start, index):
    """找出照搬最长的那段词串在训练集里的原句，并在原句里向两端延伸，得到不受 max_n 上限限制的真实长度。

    索引只存哈希，这里用原句逐字比对再确认一次；找不到原句时返回 None。
    """
    length = index.longest_copied_span(tokens, gen_start)
    if length == 0:
        return None
    for i in range(len(tokens) - length + 1):
        if i + length <= gen_start or not index.has(tokens[i:i + length]):
            continue
        needle = tokens[i:i + length]
        for s in train_sentences():
            for j in range(len(s) - length + 1):
                if s[j:j + length] != needle:
                    continue
                a, b, sa, sb = i, i + length, j, j + length
                while a > 0 and sa > 0 and tokens[a - 1] == s[sa - 1]:
                    a, sa = a - 1, sa - 1
                while b < len(tokens) and sb < len(s) and tokens[b] == s[sb]:
                    b, sb = b + 1, sb + 1
                return {"span": "".join(tokens[a:b]), "span_words": b - a,
                        "span_generated_words": b - max(a, gen_start),
                        "train_sentence": "".join(s)}
    return None


def _load_timed(model_path):
    t0 = time.perf_counter()
    model = kw.load(model_path)
    return model, time.perf_counter() - t0


def _score_pass(model, text) -> float:
    t0 = time.perf_counter()
    for t in text:
        model.score(t, bos=True, eos=True)
    return time.perf_counter() - t0


def e6_query_speed(paths, sentences, rounds=7) -> dict:
    """四种格式轮流打分，测查询速度。

    第一次完整运行里 ARPA 与 probing 谁快谁慢在两次运行间颠倒过，每种格式只测 3 轮、各测各的，
    结果受缓存与系统调度影响太大。这里先各打分一遍预热，再按格式轮换测 rounds 轮（一轮里四种各测一次），
    报告中位数与最快一轮；轮换让同一时段的系统抖动均摊到各格式上。
    """
    text = [" ".join(s) for s in sentences]
    events = sum(len(s) + 1 for s in sentences)
    models, load = {}, {}
    for name, path in paths.items():
        models[name], load[name] = _load_timed(path)
        _score_pass(models[name], text)  # 预热
    per = {name: [] for name in paths}
    for _ in range(rounds):
        for name in paths:
            per[name].append(_score_pass(models[name], text))
    out = {}
    for name in paths:
        med = statistics.median(per[name])
        out[name] = {"load_seconds": load[name], "score_seconds_median": med, "score_seconds_min": min(per[name]),
                     "score_seconds_all": per[name], "queries_per_second": events / med,
                     "queries_per_second_best": events / min(per[name])}
    return out


def e6_storage():
    """六元模型的四种存储：ARPA 文本、probing 哈希、trie 前缀树、8 位量化 trie。

    比较文件大小、转换耗时、加载耗时、查询速度、测试集困惑度，以及同一组采样参数下生成是否一致。
    """
    arpa = MODELS / "kn6.arpa"
    test = read_sentences(SPLIT_DIR / "test.txt")
    formats = {"ARPA": (arpa, None)}
    for name, structure, q in (("probing", "probing", None), ("trie", "trie", None), ("trie_q8", "trie", 8)):
        out_path = MODELS / f"kn6_{name}.bin"
        formats[name] = (out_path, kw.build_binary(arpa, out_path, structure, q))
    vocab = sampling.vocab_from_arpa(arpa)
    speed = e6_query_speed({name: path for name, (path, _) in formats.items()}, test)
    ref_text, ref_scores = None, None
    out = {}
    for name, (path, build) in formats.items():
        model = kw.load(path)
        scores = kw.position_scores(model, test)
        ev = kw.summarize_scores(scores, len(test))
        gen = sampling.Generator(model, vocab)
        texts = ["".join(gen.generate(MAIN_PROMPT, 1.0, 50, max_tokens=MAX_GEN_TOKENS, seed=s)["generated"])
                 for s in STAT_SEEDS]
        if ref_text is None:
            ref_text, ref_scores = texts, scores
        diffs = [abs(a[0] - b[0]) for a, b in zip(scores, ref_scores)]
        out[name] = {"bytes": Path(path).stat().st_size, "build": build, **speed[name],
                     "test_ppl": ev["ppl_including_oov"],
                     "position_log10_absdiff_vs_arpa": {"max": max(diffs), "mean": statistics.mean(diffs),
                                                        "changed": sum(d > 1e-6 for d in diffs)},
                     "generations_shown": texts[:len(GEN_SEEDS)],
                     "same_generation_count": sum(a == b for a, b in zip(texts, ref_text)),
                     "generation_count": len(texts)}
        log(f"  {name}：{out[name]['bytes'] / 2**20:.1f} MiB，加载 {speed[name]['load_seconds']:.2f}s，"
            f"{speed[name]['queries_per_second'] / 1e6:.2f} M 次查询/秒，PPL {ev['ppl_including_oov']:.3f}，"
            f"生成与 ARPA 相同 {out[name]['same_generation_count']}/{len(texts)}")
        del model, gen
    return out


PREFIXES = {
    "作业开头": "在 阳光 明媚 的 五月 ， 我们 学校 胜利 召开 了",
    "新闻套语": "新华社 北京",
    "高频句首": "在 新 的 一 年 里 ，",
    "单个词": "北京",
    "含未登录词": "在 人工智能 的 帮助 下 ，",
    # 语料把「人民大会堂」切成「人民 大会堂」（训练集 52 次，从未作为一个词出现），按语料的切分写
    "长前缀": "本报 北京 １月 ５日 讯 记者 报道 ： 国务院 总理 李 鹏 今天 在 人民 大会堂",
}


def e10_prefixes():
    """六类开头，最优阶数模型，T=1.0、k=50，每个开头生成 10 次。

    统计首词分布的熵、10 次生成的长度方差、10 次之间的平均两两重合（越低越多样）、
    开头的最后几个词在训练集里出现过多少次。
    """
    best = read_json(RUNS_DIR / "E2_orders.json")["best_order_by_dev"]
    model = load_model(best)
    gen = sampling.Generator(model, vocab_of(best))
    seeds = list(range(101, 111))  # 实验清单规定每个开头生成 10 次
    # 先把六个开头的生成全部做完，再只为这些文本建训练集索引、只数用得到的 n-gram
    generated = {name: [gen.generate(text.split(), 1.0, 50, max_tokens=MAX_GEN_TOKENS, seed=s) for s in seeds]
                 for name, text in PREFIXES.items()}
    index = query_index([g["prefix"] + g["generated"] for recs in generated.values() for g in recs])
    wanted = set()
    for text in PREFIXES.values():
        p = text.split()
        wanted.update((w,) for w in p)
        wanted.update(tuple(p[-n:]) for n in (2, 3) if len(p) >= n)
    train_counts = dict.fromkeys(wanted, 0)
    for s in train_sentences():
        for n in (1, 2, 3):
            for i in range(len(s) - n + 1):
                g = tuple(s[i:i + n])
                if g in train_counts:
                    train_counts[g] += 1
    train_counts = {g: c for g, c in train_counts.items() if c}
    out = {}
    for name, text in PREFIXES.items():
        prefix = text.split()
        state = gen.start(prefix)
        raw = gen.scores(state)
        probs = 10.0 ** raw
        probs = probs / probs.sum()
        first_top = sorted(zip(probs, gen.vocab), reverse=True)[:5]
        recs = generated[name]
        lengths = [r["token_count"] for r in recs]
        bags = [set(r["generated"]) for r in recs]
        pairs = [(a, b) for i, a in enumerate(bags) for b in bags[i + 1:]]
        jaccard = statistics.mean(len(a & b) / len(a | b) if a | b else 1.0 for a, b in pairs)
        first_words = [r["generated"][0] if r["generated"] else "</s>" for r in recs]
        oov = [w for w in prefix if (w,) not in train_counts]
        out[name] = {
            "prefix": text, "prefix_words": len(prefix), "oov_words": oov,
            "last2_train_count": train_counts.get(tuple(prefix[-2:]), 0) if len(prefix) >= 2 else None,
            "last3_train_count": train_counts.get(tuple(prefix[-3:]), 0) if len(prefix) >= 3 else None,
            "first_step_entropy_bits": sampling.entropy_bits(probs),
            "first_step_top": [[w, float(p)] for p, w in first_top],
            "distinct_first_words": len(set(first_words)),
            "length_mean": statistics.mean(lengths), "length_stdev": statistics.pstdev(lengths),
            "eos_rate": statistics.mean(r["hit_eos"] for r in recs),
            "pairwise_jaccard_mean": jaccard,
            "longest_copied_span_mean": statistics.mean(
                index.longest_copied_span(prefix + r["generated"], len(prefix)) for r in recs),
            "matched_order_mean": statistics.mean(s["matched_order"] for r in recs for s in r["steps"]),
            "samples": ["".join(r["generated"]) for r in recs],
        }
        log(f"  {name}：首词熵 {out[name]['first_step_entropy_bits']:.2f} bit，"
            f"平均长度 {out[name]['length_mean']:.1f}±{out[name]['length_stdev']:.1f}")
    return {"model_order": best, "seeds": seeds, "prefixes": out}


E11_TEMPERATURES = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]


def e11_temperature():
    """固定一个状态（作业开头之后），看温度如何改变下一个词的分布。

    对每个温度算采样分布的熵、最可能词的概率、有效候选数 2^H，以及覆盖 90% 概率质量要多少个词。
    另取两个状态作对照：句首（模型最不确定的位置之一）与「新华社 北京」之后（高度确定的位置）。
    """
    best = read_json(RUNS_DIR / "E2_orders.json")["best_order_by_dev"]
    model = load_model(best)
    gen = sampling.Generator(model, vocab_of(best))
    import numpy as np
    contexts = {"作业开头之后": MAIN_PROMPT, "句首": [], "「新华社 北京」之后": ["新华社", "北京"]}
    out = {}
    for label, prefix in contexts.items():
        raw = gen.scores(gen.start(prefix))
        rows = []
        for t in E11_TEMPERATURES:
            order, probs = sampling.transform(raw, temperature=t)
            cum = np.cumsum(probs)
            h = sampling.entropy_bits(probs)
            rows.append({"temperature": t, "entropy_bits": h, "effective_candidates": 2 ** h,
                         "top1_word": gen.vocab[int(order[0])], "top1_prob": float(probs[0]),
                         "words_for_90pct": int(np.searchsorted(cum, 0.9) + 1),
                         "top10": [[gen.vocab[int(i)], float(p)] for i, p in zip(order[:10], probs[:10])]})
        out[label] = {"prefix": prefix, "rows": rows}
        r1 = next(r for r in rows if r["temperature"] == 1.0)
        log(f"  {label}：T=1 时熵 {r1['entropy_bits']:.2f} bit，有效候选 {r1['effective_candidates']:.0f}")
    return {"model_order": best, "contexts": out}


def e12_pruning_detail():
    """E3 的五元剪枝前后，测试集每个位置的对数概率变化，按「该位置的五元在训练集里出现过几次」分组。

    剪枝只删了三阶及以上计数为 1 的条目，预期受影响的主要是训练集里只见过一次的高阶组合。
    """
    test = read_sentences(SPLIT_DIR / "test.txt")
    full = kw.position_scores(load_model(5), test)
    pruned = kw.position_scores(kw.load(MODELS / "kn5_prune.arpa"), test)
    # 只数测试集里出现的五元：六个月的训练集有几千万个不同的五元，全数要好几 GB 内存
    counts5 = {}
    for s in test:
        seq = ["<s>"] + s + ["</s>"]
        for i in range(len(seq) - 4):
            counts5[tuple(seq[i:i + 5])] = 0
    for s in train_sentences():
        seq = ["<s>"] + s + ["</s>"]
        for i in range(len(seq) - 4):
            g = tuple(seq[i:i + 5])
            if g in counts5:
                counts5[g] += 1
    def bucket_of(c):
        return "0（训练集未见）" if c == 0 else "1" if c == 1 else "2-4" if c < 5 else "≥5"
    groups = {}
    pos = 0
    for s in test:
        seq = ["<s>"] + s + ["</s>"]
        for i in range(1, len(seq)):
            (lf, nf, oov), (lp, np_, _) = full[pos], pruned[pos]
            pos += 1
            ctx = tuple(seq[max(0, i - 4):i + 1])
            key = "OOV" if oov else bucket_of(counts5.get(ctx, 0)) if len(ctx) == 5 else "句首不足五词"
            g = groups.setdefault(key, {"positions": 0, "full_log10": 0.0, "pruned_log10": 0.0,
                                         "changed": 0, "order_full": 0, "order_pruned": 0})
            g["positions"] += 1
            g["full_log10"] += lf
            g["pruned_log10"] += lp
            g["changed"] += abs(lf - lp) > 1e-6
            g["order_full"] += nf
            g["order_pruned"] += np_
    total = sum(g["positions"] for g in groups.values())
    for g in groups.values():
        n = g["positions"]
        g["share_of_positions"] = n / total
        g["ppl_full"] = 10 ** (-g["full_log10"] / n)
        g["ppl_pruned"] = 10 ** (-g["pruned_log10"] / n)
        g["changed_rate"] = g["changed"] / n
        g["mean_order_full"] = g.pop("order_full") / n
        g["mean_order_pruned"] = g.pop("order_pruned") / n
        # 这一组对整体 log 概率损失的贡献
        g["log10_loss"] = g["full_log10"] - g["pruned_log10"]
    e3 = read_json(RUNS_DIR / "E3_pruning.json")["5"]
    removed = [a - b for a, b in zip(e3["unpruned"]["ngram_counts"], e3["training"]["ngram_counts"])]
    return {"groups": groups, "positions": total, "removed_ngrams_by_order": removed,
            "ngram_counts_full": e3["unpruned"]["ngram_counts"],
            "ngram_counts_pruned": e3["training"]["ngram_counts"]}


E13_MIN_FREQS = [2, 3, 5, 10]
CLEMENTINE_TRAIN_WORDS = 280_200   # 70clementine 报告里训练集的词数
LYNXPENG_TRAIN_WORDS = 6_689_269   # LynxPeng 报告里训练集的词数（1998 年 1–6 月）


def _map_rare(sentences, vocab):
    return [[w if w in vocab else RARE_TOKEN for w in s] for s in sentences]


def _write_sentences(path, sentences):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(" ".join(s) for s in sentences) + "\n", encoding="utf-8")


def _closed_vocab_eval(tag, train, test, min_freq, orders):
    """训练集里出现不到 min_freq 次的词，在训练集与测试集里都换成 RARE_TOKEN，再训练、评估。

    RARE_TOKEN 在训练集里有真实计数，测试集里不再有 KenLM 意义上的未登录词。
    """
    freq = {}
    for s in train:
        for w in s:
            freq[w] = freq.get(w, 0) + 1
    vocab = {w for w, c in freq.items() if c >= min_freq}
    tr, te = _map_rare(train, vocab), _map_rare(test, vocab)
    path = WORK_DIR / "corpus" / "closed" / f"{tag}_min{min_freq}.txt"
    _write_sentences(path, tr)
    row = {"min_freq": min_freq, "vocab_size": len(vocab) + 1,  # +1 是 RARE_TOKEN 本身
           "train_rare_rate": sum(w == RARE_TOKEN for s in tr for w in s) / sum(len(s) for s in tr),
           "test_rare_rate": sum(w == RARE_TOKEN for s in te for w in s) / sum(len(s) for s in te),
           "orders": {}}
    for n in orders:
        arpa, _ = repeated_train(f"e13_{tag}_min{min_freq}_kn{n}", path, n, repeats=1)
        ev = kw.evaluate(kw.load(arpa), te)
        row["orders"][str(n)] = {k: ev[k] for k in ("ppl_including_oov", "oov_events", "events")}
        arpa.unlink()
    path.unlink()
    return row


def _powerlaw_extrapolate(points, n, target_words):
    """用 E5 的五个规模点拟合 log 困惑度 = a + b·log 词数（最小二乘），外推到 target_words。"""
    xs = [math.log(p["actual_words"]) for p in points]
    ys = [math.log(p["orders"][n]["test"]["ppl_including_oov"]) for p in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    return {"slope": b, "predicted_ppl": math.exp(a + b * math.log(target_words))}


def e13_protocol():
    """同一份数据，换一种困惑度的算法，数字差多少。

    本实验的主结果是开放词表口径：测试集里训练时没见过的词由 KenLM 的 <unk> 给概率，每个约 1e-6。
    70clementine 用封闭词表口径：训练集出现不到 2 次的词换成一个「低频词」符号，这个符号在训练集里
    很常见，容易预测。LynxPeng 是开放词表、1998 年 1–6 月语料（1 月语料上他的训练集是本实验的 7.4 倍）。
    这里在本实验的数据上把这些做法都算一遍：
    1. 全部训练集：开放词表（取自 E2）、去掉未登录词位置，以及阈值 2 / 3 / 5 / 10 的封闭词表；
    2. 从训练集抽 70clementine 规模的约 28 万词，开放词表与阈值 2 的封闭词表各算一次；
    3. 用 E5 的学习曲线外推到 LynxPeng 的训练规模。外推超出了实测范围，
       他的标注版本和测试集也与本实验不同，只能当作量级核对。
    测试集一律是本实验的测试集（116,528 词），与两位同学的测试集不同。
    """
    train = read_sentences(SPLIT_DIR / "train.txt")
    test = read_sentences(SPLIT_DIR / "test.txt")
    e2 = read_json(RUNS_DIR / "E2_orders.json")["orders"]
    open_full = {n: {k: e2[n]["test"][k] for k in ("ppl_including_oov", "ppl_excluding_oov", "oov_rate")}
                 for n in ("3", "5")}
    closed_full = []
    for mf in E13_MIN_FREQS:
        row = _closed_vocab_eval("full", train, test, mf, (3, 5))
        closed_full.append(row)
        log(f"  全部训练集，阈值 {mf}：测试集换成低频词 {row['test_rare_rate']:.1%}，"
            f"三元 {row['orders']['3']['ppl_including_oov']:.2f}，五元 {row['orders']['5']['ppl_including_oov']:.2f}")

    sample_path = WORK_DIR / "corpus" / "closed" / "train_280k.txt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    info = pfr_data.sample_by_size(SPLIT_DIR / "train.txt", SPLIT_DIR / "train_articles.txt", sample_path,
                                   CLEMENTINE_TRAIN_WORDS, seed=pfr_data.SPLIT_SEED)
    small = read_sentences(sample_path)
    arpa, _ = repeated_train("e13_280k_open_kn5", sample_path, 5, repeats=1)
    ev = kw.evaluate(kw.load(arpa), test)
    arpa.unlink()
    small_open = {k: ev[k] for k in ("ppl_including_oov", "ppl_excluding_oov", "oov_rate")}
    small_closed = _closed_vocab_eval("280k", small, test, 2, (5,))
    sample_path.unlink()
    log(f"  约 28 万词：开放词表五元 {small_open['ppl_including_oov']:.2f}，"
        f"阈值 2 五元 {small_closed['orders']['5']['ppl_including_oov']:.2f}")

    points = read_json(RUNS_DIR / "E5_scale.json")["points"]
    extrap = {n: _powerlaw_extrapolate(points, n, LYNXPENG_TRAIN_WORDS) for n in ("3", "5")}
    return {
        "test_tokens": sum(len(s) for s in test),
        "open_vocab_full": open_full,
        "closed_vocab_full": closed_full,
        "clementine_scale": {"sample_words": info["actual_words"], "sample_articles": info["article_count"],
                             "open_vocab_5gram": small_open, "closed_min2": small_closed},
        "lynxpeng_scale_extrapolation": {"target_words": LYNXPENG_TRAIN_WORDS, "by_order": extrap},
        # 两位同学报告里的数字（见 文档/分析/同学作业分析.md），只作对照
        "classmates": {
            "70clementine": {"train_words": CLEMENTINE_TRAIN_WORDS, "min_freq": 2, "test_rare_rate": 0.1060,
                             "test_ppl": {"5": 250.24}},
            "LynxPeng": {"train_words": LYNXPENG_TRAIN_WORDS, "test_oov_rate": 0.0135,
                         "test_ppl": {"3": 272.155, "5": 260.77, "7": 254.688}},
        },
    }


PHASES = {
    "phase1": [("E0_tag_cleaning", e0_tag_cleaning), ("E1_smoothing", e1_smoothing),
               ("E2_orders", e2_orders), ("E3_pruning", e3_pruning), ("E4_memory", e4_memory),
               ("E7_generation", e7_generation)],
    "phase2": [("E5_scale", e5_scale), ("E8_matched_orders", e8_matched_orders),
               ("E9_leakage", e9_leakage), ("E6_storage", e6_storage)],
    "phase3": [("E10_prefixes", e10_prefixes), ("E11_temperature", e11_temperature),
               ("E12_pruning_detail", e12_pruning_detail)],
    # 韩雨看到本实验困惑度比两位同学高后加的：同一份数据上换算法对照（读 E2、E5 的结果）
    "phase4": [("E13_protocol", e13_protocol)],
}
# E3 E7 E8 E10 E11 E12 用到 E2 训练出的模型文件；E6 用六元；E12 用 E3 的五元剪枝版
NEEDS_MODELS = {"E3_pruning": ["kn3", "kn5"], "E7_generation": [f"kn{n}" for n in ORDERS],
                "E8_matched_orders": ["kn2", "kn3", "kn4", "kn5"], "E6_storage": ["kn6"],
                "E10_prefixes": [f"kn{n}" for n in ORDERS], "E11_temperature": [f"kn{n}" for n in ORDERS],
                "E12_pruning_detail": ["kn5", "kn5_prune"]}


def environment() -> dict:
    import platform
    import kenlm  # noqa: F401
    import numpy
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": numpy.__version__, "kenlm_commit": KENLM_COMMIT, "cpu_count": os.cpu_count()}
    try:
        with open("/proc/meminfo") as f:
            info["mem_total_kb"] = int(f.readline().split()[1])
        with open("/proc/cpuinfo") as f:
            info["cpu_model"] = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="*", default=["phase1", "phase2", "phase3", "phase4"],
                    help="阶段名（phase1–phase4）或实验名（如 E2_orders）")
    ap.add_argument("--force", action="store_true", help="已有结果也重跑")
    args = ap.parse_args(argv)

    todo = []
    for t in args.targets:
        if t in PHASES:
            todo.extend(PHASES[t])
        else:
            todo.extend(e for p in PHASES.values() for e in p if e[0] == t)
    MODELS.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "tmp").mkdir(parents=True, exist_ok=True)
    write_json(RUNS_DIR / "environment.json", environment())

    for name, fn in todo:
        out_file = RUNS_DIR / f"{name}.json"
        missing = [m for m in NEEDS_MODELS.get(name, []) if not (MODELS / f"{m}.arpa").exists()]
        if out_file.exists() and not args.force:
            log(f"跳过 {name}（结果已存在）")
            continue
        if missing:
            log(f"{name} 需要的模型文件不存在：{missing}，先重跑 E2_orders / E3_pruning")
            sys.exit(1)
        log(f"开始 {name}")
        started = time.perf_counter()
        result = fn()
        save(name, {"elapsed_seconds": time.perf_counter() - started, **result})
        log(f"完成 {name}，用时 {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
