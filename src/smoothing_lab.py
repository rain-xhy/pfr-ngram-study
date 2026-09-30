"""E1 平滑方法对比：在同一份三元计数上计算 MLE、Add-k、Jelinek-Mercer、Witten-Bell、Kneser-Ney。

lmplz 只有插值 Modified Kneser-Ney，没有其他平滑的开关，所以其他平滑在这里按公式实现。
本模块的插值 Modified Kneser-Ney 与 KenLM 用相同的约定：句首只补一个 <s>，句子第一个词
由二元 (<s>, w) 预测；以 <s> 开头的 n-gram 保留原始计数，其余低阶 n-gram 用续接计数。
这样本模块的各阶条目数、三组折扣、逐位置概率都能和 lmplz 的输出逐一核对。

记号：历史 h = (u, v)，句首位置 u 为 None；c(h,w) 是训练集里 h 后面接 w 的次数，
N1+(•w) 是 w 前面出现过的不同词数（续接多样性）。所有平滑都在同一个封闭词表上归一化：
V = 训练集词形 ∪ {</s>, <unk>}，测试集里没见过的词一律当作 <unk>。<s> 只作上下文。
"""
import collections
import math

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def _coc_discounts(counts, fallback=None):
    """Modified KN 的三个折扣 D1、D2、D3+，由计数的计数 n1..n4 估计（Chen & Goodman 1998）。

    n1..n4 有一个为零时公式没法算。KenLM 此时报错，除非给了 --discount_fallback（默认 0.5 1 1.5）；
    这里同样处理：fallback 为 None 就报错，否则直接用给定的三个折扣。
    """
    # n_k 是「恰好出现 k 次」的 n-gram 种数，k=1..4；出现 5 次及以上的不参与折扣估计
    coc = collections.Counter(c for c in counts if c <= 4)
    n1, n2, n3, n4 = (coc.get(i, 0) for i in (1, 2, 3, 4))
    problem = None
    if min(n1, n2, n3, n4) == 0:
        problem = f"计数的计数含零 {n1, n2, n3, n4}"
    else:
        y = n1 / (n1 + 2 * n2)
        D = (1 - 2 * y * n2 / n1, 2 - 3 * y * n3 / n2, 3 - 4 * y * n4 / n3)
        # 与 KenLM 相同的合法范围：第 k 档折扣须在 [0, k] 内，否则折扣后的计数会变负或超过原计数
        if all(0 <= d <= k for k, d in enumerate(D, 1)):
            return D
        problem = f"折扣超出合法范围 {tuple(round(d, 4) for d in D)}"
    if fallback is None:
        raise ValueError(f"{problem}，无法估计 modified KN 折扣")
    return tuple(fallback)


def _disc(D, c):
    return 0.0 if c <= 0 else D[min(c, 3) - 1]


def _gamma(D, buckets, denom):
    return (D[0] * buckets[0] + D[1] * buckets[1] + D[2] * buckets[2]) / denom


def events(sentences, vocab):
    """把句子展开成预测事件 (u, v, w)：第一个词的 u 为 None，未登录词替换为 <unk>。"""
    for s in sentences:
        seq = [BOS] + [w if w in vocab else UNK for w in s] + [EOS]
        for i in range(1, len(seq)):
            yield (seq[i - 2] if i >= 2 else None), seq[i - 1], seq[i]


class TrigramCounts:
    """训练集上的 1-3 元计数，以及 Kneser-Ney 需要的续接计数与折扣。"""

    def __init__(self, sentences, fallback_discounts=None):
        self.vocab = {w for s in sentences for w in s} | {EOS, UNK}
        self.V = len(self.vocab)
        c3, c2, c1 = collections.Counter(), collections.Counter(), collections.Counter()
        for s in sentences:
            seq = [BOS] + list(s) + [EOS]
            for i in range(1, len(seq)):
                w = seq[i]
                c1[w] += 1
                c2[(seq[i - 1], w)] += 1
                if i >= 2:
                    c3[(seq[i - 2], seq[i - 1], w)] += 1
        self.c3, self.c2, self.c1 = c3, c2, c1
        self.tokens = sum(c1.values())
        self.h3, self.h3_types = collections.Counter(), collections.Counter()
        self.h3_buckets = collections.defaultdict(lambda: [0, 0, 0])
        for (u, v, _), c in c3.items():
            self.h3[(u, v)] += c
            self.h3_types[(u, v)] += 1
            self.h3_buckets[(u, v)][min(c, 3) - 1] += 1
        self.h2, self.h2_types = collections.Counter(), collections.Counter()
        for (v, _), c in c2.items():
            self.h2[v] += c
            self.h2_types[v] += 1
        self._build_kn(fallback_discounts)

    def _build_kn(self, fallback):
        # 续接计数（KenLM 称 adjusted counts）：低阶 n-gram 前面出现过多少种不同的词。
        # 以 <s> 开头的 n-gram 前面不可能再有词，保留原始计数。
        a2 = collections.Counter()
        for (_, v, w) in self.c3:
            a2[(v, w)] += 1
        for (v, w), c in self.c2.items():
            if v == BOS:
                a2[(v, w)] = c
        self.a2 = a2
        self.a1 = collections.Counter(w for (_, w) in self.c2)
        self.D3 = _coc_discounts(self.c3.values(), fallback)
        self.D2 = _coc_discounts(a2.values(), fallback)
        self.D1 = _coc_discounts(self.a1.values(), fallback)
        self.a2_hist = collections.Counter()
        self.a2_buckets = collections.defaultdict(lambda: [0, 0, 0])
        for (v, _), c in a2.items():
            self.a2_hist[v] += c
            self.a2_buckets[v][min(c, 3) - 1] += 1
        self.a1_total = sum(self.a1.values())
        buckets = [0, 0, 0]
        for c in self.a1.values():
            buckets[min(c, 3) - 1] += 1
        self.gamma1 = _gamma(self.D1, buckets, self.a1_total)

    def history_counts(self, u, v, w):
        """(历史出现次数, 该 n-gram 出现次数)：句首位置 u 为 None，用二元统计。"""
        if u is None:
            return self.h2.get(v, 0), self.c2.get((v, w), 0)
        return self.h3.get((u, v), 0), self.c3.get((u, v, w), 0)

    def p_mle(self, w, u, v):
        hist, cont = self.history_counts(u, v, w)
        return cont / hist if hist else 0.0

    def p_addk(self, w, u, v, k):
        hist, cont = self.history_counts(u, v, w)
        return (cont + k) / (hist + k * self.V)

    def p_jm(self, w, u, v, lambdas):
        """线性插值 λ3·P_ML(w|u,v) + λ2·P_ML(w|v) + λ1·P_ML(w) + λ0/|V|。

        某一阶的历史在训练集里没出现过时，该阶的最大似然估计没有定义，
        它的权重按比例分给其余各项，对任何历史都是合法分布。
        """
        l3, l2, l1, l0 = lambdas
        parts = [(l1, self.c1.get(w, 0) / self.tokens), (l0, 1.0 / self.V)]
        d2 = self.h2.get(v, 0)
        if d2:
            parts.append((l2, self.c2.get((v, w), 0) / d2))
        d3 = self.h3.get((u, v), 0) if u is not None else 0
        if d3:
            parts.append((l3, self.c3.get((u, v, w), 0) / d3))
        return sum(l * p for l, p in parts) / sum(l for l, _ in parts)

    def _p1_wb(self, w):
        t1 = len(self.c1)
        return (self.c1.get(w, 0) + t1 / self.V) / (self.tokens + t1)

    def _p2_wb(self, w, v):
        p1 = self._p1_wb(w)
        d2, t2 = self.h2.get(v, 0), self.h2_types.get(v, 0)
        return (self.c2.get((v, w), 0) + t2 * p1) / (d2 + t2) if d2 else p1

    def p_wb(self, w, u, v):
        """递归 Witten-Bell：历史 h 留给低阶的质量是 N1+(h•) / (c(h) + N1+(h•))。"""
        p2 = self._p2_wb(w, v)
        d3 = self.h3.get((u, v), 0) if u is not None else 0
        if not d3:
            return p2
        t3 = self.h3_types[(u, v)]
        return (self.c3.get((u, v, w), 0) + t3 * p2) / (d3 + t3)

    def p1_kn(self, w):
        a = self.a1.get(w, 0)
        return max(a - _disc(self.D1, a), 0.0) / self.a1_total + self.gamma1 / self.V

    def p2_kn(self, w, v):
        p1 = self.p1_kn(w)
        denom = self.a2_hist.get(v, 0)
        if not denom:
            return p1
        a = self.a2.get((v, w), 0)
        return (max(a - _disc(self.D2, a), 0.0) / denom
                + _gamma(self.D2, self.a2_buckets[v], denom) * p1)

    def p_kn(self, w, u, v):
        """插值 Modified Kneser-Ney：最高阶用原始计数，低阶用续接计数，一元再与均匀分布插值。"""
        p2 = self.p2_kn(w, v)
        denom = self.h3.get((u, v), 0) if u is not None else 0
        if not denom:
            return p2
        c = self.c3.get((u, v, w), 0)
        return (max(c - _disc(self.D3, c), 0.0) / denom
                + _gamma(self.D3, self.h3_buckets[(u, v)], denom) * p2)


def evaluate(prob_fn, sentences, vocab) -> dict:
    """在测试句上算困惑度（以 2 为底，分母为预测次数 = 词数 + 句数）。

    零概率事件单独计数：MLE 下只要有一个事件概率为 0，整体困惑度就是无穷大，
    ppl_over_nonzero 只在概率非零的事件上平均，用来看 MLE 在「见过的部分」上有多好。
    """
    total, n, zero, unk = 0.0, 0, 0, 0
    for u, v, w in events(sentences, vocab):
        p = prob_fn(w, u, v)
        n += 1
        unk += w == UNK
        if p <= 0:
            zero += 1
        else:
            total += math.log2(p)
    nonzero = n - zero
    return {
        "events": n, "unk_events": unk,
        "zero_prob_events": zero, "zero_prob_rate": zero / n if n else 0.0,
        "ppl": float("inf") if zero else 2 ** (-total / n),
        "ppl_over_nonzero": 2 ** (-total / nonzero) if nonzero else float("nan"),
    }


def normalization_check(counts, prob_fn, histories) -> list:
    """在给定历史上把 P(w | h) 对整个词表求和，应当等于 1。"""
    return [{"history": [u, v], "sum": sum(prob_fn(w, u, v) for w in counts.vocab)}
            for u, v in histories]


def continuation_table(counts, top=15, min_count=20) -> dict:
    """slide 120 的 Francisco / glasses 现象：词频高但前面几乎只跟一种词 vs 前面跟很多种词。

    对出现至少 min_count 次的词，比较词频 c(w) 与续接数 N1+(•w)（不同左邻词的个数）。
    返回两端各 top 个：left_bound 是比值 N1+(•w)/c(w) 最小的（左邻高度固定），
    left_free 是比值最大的（左邻最多样）。<s> 算作一种左邻。
    """
    rows = [(w, c, counts.a1.get(w, 0)) for w, c in counts.c1.items()
            if w != EOS and c >= min_count]
    # 左邻最常见的那个词，用来说明「几乎只跟某个词」
    top_left = collections.defaultdict(collections.Counter)
    for (v, w), c in counts.c2.items():
        top_left[w][v] += c
    def fmt(w, c, n1):
        v, cv = top_left[w].most_common(1)[0]
        # 回退到一元时，MLE 按词频分配概率，Kneser-Ney 按续接数分配；两者之比就是 KN 对这个词的调整
        p_ml, p_cont = c / counts.tokens, n1 / counts.a1_total
        return {"word": w, "count": c, "distinct_left": n1, "ratio": n1 / c,
                "top_left": v, "top_left_share": cv / c,
                "p_unigram_mle": p_ml, "p_continuation": p_cont, "kn_over_mle": p_cont / p_ml}
    rows.sort(key=lambda r: r[2] / r[1])
    return {"min_count": min_count,
            "left_bound": [fmt(*r) for r in rows[:top]],
            "left_free": [fmt(*r) for r in rows[-top:][::-1]]}
