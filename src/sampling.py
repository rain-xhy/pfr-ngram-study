"""参数化文本生成与生成文本分析。

KenLM 与 SRILM 都没有带 temperature / top-k / top-p 的生成接口，这里用 KenLM 的 State 接口
对整个词表逐词打分，再在得到的分布上自己采样。每一步都对全部候选词计算
P(w | 历史)，不是只在训练中见过的后继词里抽，所以回退到低阶时的概率也完整参与采样。

采样前屏蔽 <s> 与 <unk>，保留 </s>，模型可以自然结束一句话；训练语料里本来就有的标点照常参与。
"""
import math

import numpy as np

from .smoothing_lab import BOS, EOS, UNK


def vocab_from_arpa(arpa_path) -> list:
    """ARPA 一元表里的全部词形，去掉 <s> 与 <unk>，保留 </s>。"""
    words = []
    with open(arpa_path, encoding="utf-8") as f:
        inside = False
        for line in f:
            if line.startswith("\\1-grams:"):
                inside = True
                continue
            if inside:
                if not line.strip() or line.startswith("\\2-grams:") or line.startswith("\\end\\"):
                    break
                word = line.rstrip("\n").split("\t")[1]
                if word not in (BOS, UNK):
                    words.append(word)
    return words


def transform(log10_scores, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
    """把一组 log10 分数变成采样分布。

    temperature 作用在自然对数概率上：q_i ∝ exp(ln p_i / T) = p_i^(1/T)。
    top_k=0 表示不截断；top_p 保留按概率从高到低累积到 p 为止的最小候选集。
    截断后重新归一化。返回 (保留的下标, 对应概率)，下标按概率降序。
    """
    scores = np.asarray(log10_scores, dtype=np.float64) * math.log(10)
    order = np.argsort(-scores, kind="stable")
    if top_k and top_k < len(order):
        order = order[:top_k]
    logits = scores[order] / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    if top_p < 1.0:
        keep = int(np.searchsorted(np.cumsum(probs), top_p) + 1)
        order, probs = order[:keep], probs[:keep] / probs[:keep].sum()
    return order, probs


def entropy_bits(probs) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


class Generator:
    """在一个 KenLM 模型上逐词生成。词表与模型绑定，建一次反复用。"""

    def __init__(self, model, vocab):
        import kenlm
        self._kenlm = kenlm
        self.model = model
        self.vocab = list(vocab)
        self.eos_index = self.vocab.index(EOS)

    def _advance(self, state, word):
        nxt = self._kenlm.State()
        self.model.BaseScore(state, word, nxt)
        return nxt

    def start(self, prefix):
        state = self._kenlm.State()
        self.model.BeginSentenceWrite(state)
        for w in prefix:
            state = self._advance(state, w)
        return state

    def scores(self, state) -> np.ndarray:
        """当前状态下整个词表的 log10 条件概率。"""
        tmp = self._kenlm.State()
        base = self.model.BaseScore
        return np.fromiter((base(state, w, tmp) for w in self.vocab), dtype=np.float64,
                           count=len(self.vocab))

    def matched_order(self, state, word) -> int:
        """给定状态下预测 word 时实际匹配到的 n-gram 阶数。

        BaseFullScore 的 ngram_length 与 full_scores 给出的匹配长度相同
        （在训练集前 300 句的 8,435 个位置上逐一核对过）。
        """
        tmp = self._kenlm.State()
        return self.model.BaseFullScore(state, word, tmp).ngram_length

    def generate(self, prefix, temperature=1.0, top_k=0, top_p=1.0, greedy=False,
                 max_tokens=60, seed=11, record_top=5) -> dict:
        rng = np.random.default_rng(seed)
        state = self.start(prefix)
        out, steps = [], []
        hit_eos = False
        for _ in range(max_tokens):
            raw = self.scores(state)
            full_probs = np.power(10.0, raw)
            if greedy:
                idx = int(np.argmax(raw))
                kept_n, chosen_p = 1, 1.0
            else:
                order, probs = transform(raw, temperature, top_k, top_p)
                j = int(rng.choice(len(order), p=probs))
                idx, kept_n, chosen_p = int(order[j]), len(order), float(probs[j])
            word = self.vocab[idx]
            top = np.argsort(-raw)[:record_top]
            steps.append({
                "chosen": word,
                "model_prob": float(full_probs[idx]),
                "sampling_prob": chosen_p,
                "top1_prob": float(full_probs[top[0]]),
                "chose_top1": bool(idx == int(top[0])),
                "entropy_bits": entropy_bits(full_probs / full_probs.sum()),
                "candidates_kept": kept_n,
                "matched_order": self.matched_order(state, word),
                "top": [[self.vocab[t], round(float(full_probs[t]), 6)] for t in top],
            })
            if idx == self.eos_index:
                hit_eos = True
                break
            out.append(word)
            state = self._advance(state, word)
        return {"prefix": list(prefix), "generated": out, "hit_eos": hit_eos,
                "token_count": len(out), "steps": steps,
                "params": {"temperature": temperature, "top_k": top_k, "top_p": top_p,
                           "greedy": greedy, "max_tokens": max_tokens, "seed": seed}}


def repetition_stats(tokens) -> dict:
    """自重复：生成片段里重复出现的 2/3-gram 占比，以及 type-token ratio。"""
    out = {"length": len(tokens),
           "type_token_ratio": len(set(tokens)) / len(tokens) if tokens else 0.0}
    for n in (2, 3):
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        out[f"repeated_{n}gram_rate"] = (1 - len(set(grams)) / len(grams)) if grams else 0.0
    return out


class TrainIndex:
    """训练集 1..max_n 元 n-gram 的查找表，用来检测生成文本与训练集的重合（E9）。

    只存元组的哈希值（64 位，同一进程内稳定），误判为「出现过」的概率可以忽略。
    n-gram 不跨句：训练句之间没有连接，照搬的判定只针对句内连续词串。

    wanted 为 None 时存下训练集的全部 1..max_n 元 n-gram（九十万词约九百万个，几百 MB）。
    六个月的训练集有五百多万词，全存要好几 GB，这时先收集要查询的文本里出现的 n-gram 哈希
    （collect_queries），再只把训练集里命中这些哈希的记下来，查询结果与全存完全相同。
    """

    def __init__(self, sentences, max_n=12, wanted=None):
        self.max_n = max_n
        self.grams = {n: set() for n in range(1, max_n + 1)}
        for s in sentences:
            length = len(s)
            for n in range(1, min(max_n, length) + 1):
                add = self.grams[n].add
                want = wanted[n] if wanted is not None else None
                for i in range(length - n + 1):
                    h = hash(tuple(s[i:i + n]))
                    if want is None or h in want:
                        add(h)

    @staticmethod
    def collect_queries(token_lists, max_n=12) -> dict:
        """给定要查询的若干个词序列，收集其中全部 1..max_n 元 n-gram 的哈希，按长度分组。"""
        wanted = {n: set() for n in range(1, max_n + 1)}
        for tokens in token_lists:
            for n in range(1, min(max_n, len(tokens)) + 1):
                for i in range(len(tokens) - n + 1):
                    wanted[n].add(hash(tuple(tokens[i:i + n])))
        return wanted

    def has(self, gram) -> bool:
        n = len(gram)
        return 0 < n <= self.max_n and hash(tuple(gram)) in self.grams[n]

    def overlap_rate(self, tokens, n) -> float:
        """tokens 里的 n-gram 有多大比例在训练集中出现过。"""
        total = len(tokens) - n + 1
        if total <= 0:
            return float("nan")
        return sum(self.has(tokens[i:i + n]) for i in range(total)) / total

    def longest_copied_span(self, tokens, gen_start: int = 0) -> int:
        """在训练集句子里原样出现过的最长连续词串长度，只统计结束位置落在 gen_start 之后的词串。

        传入「提示词 + 生成内容」、gen_start 取提示词长度，就只计至少含一个生成词的词串，
        提示词本身的重合不算。结果最多到 max_n，等于 max_n 表示至少照搬了 max_n 个词。
        """
        best = 0
        for i in range(len(tokens)):
            n = 1
            while i + n <= len(tokens) and n <= self.max_n and self.has(tokens[i:i + n]):
                if i + n > gen_start:
                    best = max(best, n)
                n += 1
        return best
