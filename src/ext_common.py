"""E14–E19 共用的小工具：预测位置的枚举与抽样、训练集词频、按组重抽样的置信区间、排名指标。"""
import collections
import math
import random

import numpy as np

from .experiment_runner import SPLIT_DIR, TAGGED_DIR, train_sentences  # noqa: F401

EOS = "</s>"
SEED = 20260930  # E14 以后的抽样与重抽样共用的种子


def positions(sentences):
    """全部预测位置 (句子下标, 词下标)；词下标等于句长时预测的是句末 </s>。"""
    return [(i, j) for i, s in enumerate(sentences) for j in range(len(s) + 1)]


def target(sent, j):
    return sent[j] if j < len(sent) else EOS


def sample(seq, n, seed=SEED):
    """不放回抽 n 个，按原顺序返回。"""
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(seq)), min(n, len(seq))))
    return [seq[i] for i in idx]


_FREQ = None


def train_freq():
    global _FREQ
    if _FREQ is None:
        _FREQ = collections.Counter()
        for s in train_sentences():
            _FREQ.update(s)
    return _FREQ


def ppl(neg_log10_sum, events):
    return 10 ** (neg_log10_sum / events)


def bootstrap_ppl_change(loss_a, loss_b, events, reps=1000, seed=SEED):
    """按组重抽样，求模型 B 相对模型 A 的困惑度变化（B/A − 1）的点估计与 95% 区间。

    loss_a、loss_b 是每组（文章或句子）的 −log10 概率之和，events 是每组的预测次数，两个模型共用。
    """
    la, lb, ev = (np.asarray(x, dtype=np.float64) for x in (loss_a, loss_b, events))
    point = 10 ** ((lb.sum() - la.sum()) / ev.sum()) - 1
    rng = np.random.default_rng(seed)
    g = len(ev)
    draws = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, g, size=g)
        draws[r] = 10 ** ((lb[idx].sum() - la[idx].sum()) / ev[idx].sum()) - 1
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"change": float(point), "ci95": [float(lo), float(hi)], "groups": g, "reps": reps}


def bootstrap_mean_diff(a, b, reps=1000, seed=SEED):
    """成对样本（同一批句子上的 0/0.5/1 判别结果）的均值差 b − a 与 95% 区间。"""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.array([d[rng.integers(0, len(d), size=len(d))].mean() for _ in range(reps)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "n": len(d)}


def harmonic(n):
    if n <= 0:
        return 0.0
    if n < 64:
        return sum(1.0 / i for i in range(1, n + 1))
    return math.log(n) + 0.5772156649015329 + 1 / (2 * n) - 1 / (12 * n * n)


def rank_metrics(scores, true_idx, ks=(1, 5, 10)):
    """真实词在整个词表里的排名指标。同分的词按随机顺序排，给出 hit@k 与倒数排名的期望值。

    true_idx 为 None（未登录词）时一律算未命中。
    """
    if true_idx is None:
        return {**{f"hit@{k}": 0.0 for k in ks}, "rr": 0.0, "rank": None}
    t = scores[true_idx]
    greater = int(np.count_nonzero(scores > t))
    equal = int(np.count_nonzero(scores == t)) - 1
    hits = {f"hit@{k}": min(max((k - greater) / (equal + 1), 0.0), 1.0) for k in ks}
    rr = (harmonic(greater + equal + 1) - harmonic(greater)) / (equal + 1)
    return {**hits, "rr": rr, "rank": greater + 1 + equal / 2}


def summarize_ranks(rows, ks=(1, 5, 10)):
    out = {f"hit@{k}": float(np.mean([r[f"hit@{k}"] for r in rows])) for k in ks}
    out["mrr"] = float(np.mean([r["rr"] for r in rows]))
    ranks = [r["rank"] for r in rows if r["rank"] is not None]
    out["median_rank"] = float(np.median(ranks)) if ranks else None
    out["n"] = len(rows)
    return out
