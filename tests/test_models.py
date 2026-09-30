"""smoothing_lab 与 sampling 的单元测试。

smoothing_lab 只依赖标准库；sampling 需要 numpy，只在 WSL 的 venv 里跑得到，
Windows 上用 importorskip 跳过。KenLM 绑定不在单元测试里用，采样逻辑用手造的分数检验。
"""
import math

import pytest

from src import smoothing_lab as sl

TRAIN = [
    "我们 学校 召开 了 大会 。".split(),
    "我们 学校 召开 了 运动会 。".split(),
    "他们 学校 召开 了 大会 。".split(),
    "五月 的 阳光 十分 明媚 。".split(),
    "我们 的 学校 十分 美丽 。".split(),
    "他们 召开 了 会议 。".split(),
]


FALLBACK = (0.5, 1.0, 1.5)  # 与 lmplz --discount_fallback 的默认值相同；玩具语料的计数的计数有零值


@pytest.fixture(scope="module")
def counts():
    return sl.TrigramCounts(TRAIN, fallback_discounts=FALLBACK)


def test_counts_basic(counts):
    # 6 句，每句词数 + 一个 </s>
    assert counts.tokens == sum(len(s) + 1 for s in TRAIN)
    assert counts.c3[("学校", "召开", "了")] == 3
    assert counts.h3[("学校", "召开")] == 3
    assert counts.h3_types[("学校", "召开")] == 1
    # 边缘化得到的二元计数与直接数二元一致
    assert counts.c2[("召开", "了")] == 4
    assert sl.EOS in counts.vocab and sl.UNK in counts.vocab
    assert sl.BOS not in counts.vocab


def test_continuation_counts(counts):
    # 「了」前面出现过「召开」一种词
    assert counts.a1["了"] == 1
    # 「学校」前面出现过「我们」「他们」「的」三种
    assert counts.a1["学校"] == 3
    # 以 <s> 开头的二元保留原始计数：<s> 我们 出现 3 次
    assert counts.a2[(sl.BOS, "我们")] == 3
    # 其余二元用续接计数：「学校 召开」前面出现过「我们」「他们」两种
    assert counts.a2[("学校", "召开")] == 2


# 句首位置 u 为 None；("未见", "历史") 在训练集里没出现过，检验回退
HISTORIES = [("学校", "召开"), (None, sl.BOS), (sl.BOS, "我们"), ("未见", "历史"), ("的", "学校")]


@pytest.mark.parametrize("name", ["addk", "jm", "wb", "kn"])
def test_distributions_sum_to_one(counts, name):
    fn = {
        "addk": lambda w, u, v: counts.p_addk(w, u, v, 0.1),
        "jm": lambda w, u, v: counts.p_jm(w, u, v, (0.5, 0.3, 0.15, 0.05)),
        "wb": counts.p_wb,
        "kn": counts.p_kn,
    }[name]
    for u, v in HISTORIES:
        total = sum(fn(w, u, v) for w in counts.vocab)
        assert total == pytest.approx(1.0, abs=1e-9), (name, u, v, total)
        assert all(fn(w, u, v) > 0 for w in counts.vocab), (name, u, v)


def test_mle_sums_to_one_on_seen_history_and_gives_zero_elsewhere(counts):
    total = sum(counts.p_mle(w, "学校", "召开") for w in counts.vocab)
    assert total == pytest.approx(1.0)
    assert counts.p_mle("运动会", "学校", "召开") == 0.0
    assert counts.p_mle("了", "未见", "历史") == 0.0


def test_evaluate_mle_is_infinite_when_any_event_is_unseen(counts):
    # 历史「我们 学校」出现过两次，后面都接「召开」，没接过「十分」
    test = ["我们 学校 十分 美丽 。".split()]
    assert counts.p_mle("十分", "我们", "学校") == 0.0
    res = sl.evaluate(counts.p_mle, test, counts.vocab)
    assert res["zero_prob_events"] >= 1
    assert math.isinf(res["ppl"])
    kn = sl.evaluate(counts.p_kn, test, counts.vocab)
    assert kn["zero_prob_events"] == 0 and math.isfinite(kn["ppl"])


def test_events_follow_kenlm_sentence_convention(counts):
    evs = list(sl.events(["从未 出现".split()], counts.vocab))
    # 句首只补一个 <s>：第一个词由二元预测，未登录词替换为 <unk>
    assert evs == [(None, sl.BOS, sl.UNK), (sl.BOS, sl.UNK, sl.UNK), (sl.UNK, sl.UNK, sl.EOS)]


def test_continuation_table(counts):
    # 频次 >=2 的词共 9 个，两端各取 5 个即可覆盖全部
    table = sl.continuation_table(counts, top=5, min_count=2)
    words = {r["word"]: r for r in table["left_bound"] + table["left_free"]}
    assert words["了"]["distinct_left"] == 1 and words["了"]["top_left"] == "召开"
    assert words["学校"]["distinct_left"] == 3


def test_discounts_without_fallback_raise_on_sparse_counts():
    with pytest.raises(ValueError):
        sl.TrigramCounts(TRAIN)


def test_discount_formula():
    # n1=4, n2=2, n3=1, n4=1 -> y = 4/8 = 0.5；D1 = 1-2*0.5*2/4 = 0.5，D2 = 2-3*0.5*1/2 = 1.25，D3+ = 3-4*0.5*1/1 = 1
    d = sl._coc_discounts([1, 1, 1, 1, 2, 2, 3, 4])
    assert d == pytest.approx((0.5, 1.25, 1.0))
    # 出现 5 次及以上的 n-gram 不进入 n1..n4，加进去结果不变
    assert sl._coc_discounts([1, 1, 1, 1, 2, 2, 3, 4, 5, 9, 30]) == pytest.approx((0.5, 1.25, 1.0))


def test_discount_out_of_range_uses_fallback_or_raises():
    # n1=20, n2=3, n3=1, n4=1 -> D3+ = 3 - 4*(20/26)*1 < 0，KenLM 会拒绝这组折扣
    values = [1] * 20 + [2] * 3 + [3, 4]
    with pytest.raises(ValueError):
        sl._coc_discounts(values)
    assert sl._coc_discounts(values, fallback=FALLBACK) == FALLBACK


def _sampling():
    pytest.importorskip("numpy")
    from src import sampling
    return sampling


def test_transform_temperature_and_truncation():
    sampling = _sampling()
    log10 = [math.log10(p) for p in (0.5, 0.3, 0.15, 0.05)]
    order, probs = sampling.transform(log10, temperature=1.0)
    assert list(order) == [0, 1, 2, 3]
    assert probs == pytest.approx([0.5, 0.3, 0.15, 0.05])
    # T=0.5 等于把概率平方后重新归一化
    _, p_half = sampling.transform(log10, temperature=0.5)
    sq = [p ** 2 for p in (0.5, 0.3, 0.15, 0.05)]
    assert p_half == pytest.approx([x / sum(sq) for x in sq])
    order_k, p_k = sampling.transform(log10, top_k=2)
    assert list(order_k) == [0, 1] and p_k == pytest.approx([0.625, 0.375])
    # 累积概率为 0.5、0.8、0.95、1.0；阈值取在两个累积值之间，避开浮点边界
    order_p, _ = sampling.transform(log10, top_p=0.75)
    assert list(order_p) == [0, 1]
    order_p2, _ = sampling.transform(log10, top_p=0.9)
    assert list(order_p2) == [0, 1, 2]


def test_repetition_stats():
    sampling = _sampling()
    r = sampling.repetition_stats("的 的 的 的".split())
    assert r["type_token_ratio"] == 0.25
    assert r["repeated_2gram_rate"] == pytest.approx(2 / 3)
    r2 = sampling.repetition_stats("我们 学校 召开 了".split())
    assert r2["repeated_2gram_rate"] == 0.0 and r2["type_token_ratio"] == 1.0


def test_train_index_copied_span():
    sampling = _sampling()
    idx = sampling.TrainIndex(TRAIN, max_n=6)
    assert idx.longest_copied_span("我们 学校 召开 了 大会 。".split()) == 6
    assert idx.longest_copied_span("我们 学校 十分 明媚".split()) == 2
    assert idx.overlap_rate("我们 学校 召开 了".split(), 2) == 1.0
    # 提示词「我们 学校」之后生成「召开 了」：照搬的四个词含生成部分，计入
    assert idx.longest_copied_span("我们 学校 召开 了".split(), gen_start=2) == 4
    # 生成部分与训练集不连贯时，提示词自身的重合不计入
    assert idx.longest_copied_span("我们 学校 召开 从未".split(), gen_start=3) == 0
