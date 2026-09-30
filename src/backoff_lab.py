"""E14 用的 Stupid Backoff（Brants et al. 2007）：numpy 存 1..n 元计数，按历史取后继，给整个词表打分。

S(w|h) = c(hw)/c(h)，hw 在训练集出现过时；否则乘 α 回退到去掉最远一个词的历史；一元 S(w) = c(w)/N。
分数不归一化，对整个词表求和一般不等于 1，不能直接当概率算困惑度；需要概率时在整个词表上除以总和。
回退系数可以按阶给：alphas[k] 是从 k 元回退到 k-1 元时乘的系数，各阶相同就是原始的 Stupid Backoff。
句子边界与 KenLM 相同：句首补一个 <s> 只作上下文，句末 </s> 是一次预测。

历史用 64 位多项式哈希表示（同 sampling.TrainIndex，只存哈希），一千多万个历史的碰撞期望次数在 1e-5 量级。
每一阶按 (历史哈希, 词) 排序，同一历史的后继连续存放、按词编号有序。
"""
import math

import numpy as np

from .smoothing_lab import BOS, EOS, UNK

MULT = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1


def uniform_alphas(alpha, order):
    return {k: float(alpha) for k in range(2, order + 1)}


class StupidBackoff:
    def __init__(self, sentences, order):
        self.words = sorted({w for s in sentences for w in s}) + [EOS, UNK, BOS]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.eos, self.unk, self.bos = (self.index[x] for x in (EOS, UNK, BOS))
        self.order, self.V = order, len(self.words)
        seq = []
        for s in sentences:
            seq.append(self.bos)
            seq.extend(self.index[w] for w in s)
            seq.append(self.eos)
        t = np.asarray(seq, dtype=np.int64)
        del seq
        pos = np.arange(len(t))
        is_bos = t == self.bos
        start = np.maximum.accumulate(np.where(is_bos, pos, 0))  # 每个位置所在句子的 <s> 位置
        pred = ~is_bos
        uni = np.bincount(t[pred], minlength=self.V).astype(np.float64)
        self.N = int(uni.sum())
        self.unigram = uni / self.N
        self.tables = {}
        h = np.zeros(len(t), dtype=np.uint64)
        for m in range(1, order):  # 历史长度 m，对应 m+1 元；最近的词先混入
            prev = np.zeros(len(t), dtype=np.int64)
            prev[m:] = t[:-m]
            h = h * np.uint64(MULT) + (prev + 1).astype(np.uint64)
            valid = pred & (pos - m >= start)  # 历史不跨句
            self.tables[m + 1] = self._build(h[valid], t[valid])
        # 每个二元历史的后继在一元上的频率之和，profile 算一元剩余质量时直接查，不必逐词求和
        tb = self.tables[2]
        self.uni_succ2 = np.add.reduceat(self.unigram[tb["word"]], tb["offsets"][:-1])

    @staticmethod
    def _build(hist, word):
        idx = np.lexsort((word, hist))
        hist, word = hist[idx], word[idx]
        new = np.ones(len(hist), dtype=bool)
        new[1:] = (hist[1:] != hist[:-1]) | (word[1:] != word[:-1])
        first = np.flatnonzero(new)
        count = np.diff(np.append(first, len(hist)))
        uh = hist[first]
        hnew = np.ones(len(uh), dtype=bool)
        hnew[1:] = uh[1:] != uh[:-1]
        hfirst = np.flatnonzero(hnew)
        return {"hist": uh[hfirst], "offsets": np.append(hfirst, len(uh)), "word": word[first].astype(np.int32),
                "count": count.astype(np.int32), "total": np.add.reduceat(count, hfirst)}

    def entry_counts(self):
        return [int(np.count_nonzero(self.unigram))] + [len(self.tables[k]["word"]) for k in range(2, self.order + 1)]

    def nbytes(self):
        return self.unigram.nbytes + sum(a.nbytes for tb in self.tables.values() for a in tb.values())

    def ids(self, tokens):
        return [self.index.get(w, self.unk) for w in tokens]

    def _hashes(self, ctx, n):
        hs, h = {}, 0
        for m in range(1, n):
            h = (h * MULT + int(ctx[-m]) + 1) & MASK
            hs[m + 1] = h
        return hs

    def _find(self, k, h):
        tb = self.tables[k]
        key = np.uint64(h)
        j = int(np.searchsorted(tb["hist"], key))
        return j if j < len(tb["hist"]) and tb["hist"][j] == key else None

    def _successors(self, k, h):
        j = self._find(k, h)
        if j is None:
            return None
        tb = self.tables[k]
        a, b = tb["offsets"][j], tb["offsets"][j + 1]
        return tb["word"][a:b], tb["count"][a:b], int(tb["total"][j])

    def factors(self, n, alphas):
        """最高可用阶为 n 时，用第 k 阶的计数要乘的系数（回退了 n-k 次）。"""
        f = {n: 1.0}
        for k in range(n - 1, 0, -1):
            f[k] = f[k + 1] * alphas[k + 1]
        return f

    def scores(self, ctx, alphas, order=None):
        """整个词表的 SB 分数；ctx 是按时间顺序的上下文编号（含句首 <s>）。高阶覆盖低阶。"""
        n = self._top(order, ctx)
        hs, f = self._hashes(ctx, n), self.factors(n, alphas)
        out = self.unigram * f[1]
        for k in range(2, n + 1):
            found = self._successors(k, hs[k])
            if found is not None:
                w, c, tot = found
                out[w] = f[k] * c / tot
        return out

    def profile(self, ctx, w, order=None):
        """一个位置上与 α 无关的量，用来在任意 α 下求归一化概率，不必对整个词表重新打分。

        返回 (n, level, value, S)：n 是该位置可用的最高阶；level 是真实词 w 匹配到的最高阶，value 是它在
        该阶的相对频率；S[k] 是「最高匹配阶恰为 k」的那些词在第 k 阶的相对频率之和。任意 α 下分数总和
        Z = Σ f[k]·S[k]，w 的归一化概率是 f[level]·value / Z。后继集合按阶嵌套（hw 出现过，h 的后缀后面
        也接过 w），所以第 k 阶只需扣掉第 k+1 阶的后继，一元的剩余质量是 1 减去第 2 阶后继的一元频率之和。
        """
        n = self._top(order, ctx)
        hs = self._hashes(ctx, n)
        S = np.zeros(n + 1)
        level, value = 1, float(self.unigram[w])
        above = None
        for k in range(n, 1, -1):
            found = self._successors(k, hs[k])
            if found is None:
                continue
            words, c, tot = found
            if above is None:
                S[k] = 1.0
            else:  # 上一阶的后继都在本阶后继里（按词编号有序），扣掉它们的相对频率
                j = np.minimum(np.searchsorted(words, above), len(words) - 1)
                hit = words[j] == above
                S[k] = 1.0 - c[j[hit]].sum() / tot
            if level == 1:
                j = int(np.searchsorted(words, w))
                if j < len(words) and words[j] == w:
                    level, value = k, float(c[j]) / tot
            above = words
        # 有任何一阶找到时二元历史必然找到（后继集合按阶嵌套），此时 above 就是二元历史的后继
        j2 = self._find(2, hs[2]) if above is not None else None
        S[1] = 1.0 - (float(self.uni_succ2[j2]) if j2 is not None else 0.0)
        return n, level, value, S

    def _top(self, order, ctx):
        return min(order or self.order, self.order, len(ctx) + 1)

    def word_score(self, ctx, w, alphas, order=None):
        n = self._top(order, ctx)
        hs, penalty = self._hashes(ctx, n), 1.0
        for k in range(n, 1, -1):
            found = self._successors(k, hs[k])
            if found is not None:
                words, c, tot = found
                j = int(np.searchsorted(words, w))
                if j < len(words) and words[j] == w:
                    return penalty * float(c[j]) / tot
            penalty *= alphas[k]
        return penalty * float(self.unigram[w])

    def sentence_log10(self, tokens, alphas, order=None):
        """整句（含 </s>）的 log10 分数之和，未归一化；含未登录词时为 -inf。"""
        ctx, total = [self.bos], 0.0
        for w in self.ids(tokens) + [self.eos]:
            s = self.word_score(ctx, w, alphas, order)
            total += math.log10(s) if s > 0 else float("-inf")
            ctx.append(w)
        return total


def _build_cli(train_path, order):
    """供 /usr/bin/time 计时的独立进程：读训练集、建表，打印一行 JSON（建表耗时、结构体积、各阶条目数）。"""
    import json
    import time
    from pathlib import Path
    sents = [line.split() for line in Path(train_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    t0 = time.perf_counter()
    sb = StupidBackoff(sents, order)
    print(json.dumps({"build_seconds": time.perf_counter() - t0, "nbytes": sb.nbytes(),
                      "entry_counts": sb.entry_counts(), "vocab": sb.V}), flush=True)


if __name__ == "__main__":
    import sys
    _build_cli(sys.argv[1], int(sys.argv[2]))
