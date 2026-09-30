"""backoff_lab（Stupid Backoff）与 cache_lab（文档缓存插值）的单元测试，用手算得出答案的玩具语料。"""
import math

import pytest

np = pytest.importorskip("numpy")

from src import backoff_lab as bl  # noqa: E402

TRAIN = [
    "我们 学校 召开 了 大会 。".split(),
    "我们 学校 召开 了 运动会 。".split(),
    "他们 学校 召开 了 大会 。".split(),
    "五月 的 阳光 十分 明媚 。".split(),
    "他们 召开 了 会议 。".split(),
]


@pytest.fixture(scope="module")
def sb():
    return bl.StupidBackoff(TRAIN, 3)


def ctx_of(sb, words):
    return [sb.bos] + sb.ids(words)


def test_relative_frequencies(sb):
    a = bl.uniform_alphas(0.4, 3)
    # 「学校 召开」后面 3 次都是「了」
    assert sb.word_score(ctx_of(sb, ["学校", "召开"]), sb.index["了"], a) == pytest.approx(1.0)
    # 三元只看最近两个词：「召开 了」后接 大会 2、运动会 1、会议 1，与更早的词无关
    assert sb.word_score(ctx_of(sb, ["学校", "召开", "了"]), sb.index["大会"], a) == pytest.approx(2 / 4)
    assert sb.word_score(ctx_of(sb, ["他们", "召开", "了"]), sb.index["会议"], a) == pytest.approx(1 / 4)


def test_backoff_multiplies_alpha(sb):
    a = bl.uniform_alphas(0.4, 3)
    # 「学校 召开 会议」与「召开 会议」都没出现过，回退两次到一元
    w = sb.index["会议"]
    s = sb.word_score(ctx_of(sb, ["学校", "召开"]), w, a)
    assert s == pytest.approx(0.4 * 0.4 * sb.unigram[w])
    # 「运动会 了」没有，二元「了 大会」有：一次回退
    s = sb.word_score(ctx_of(sb, ["运动会", "了"]), sb.index["大会"], a)
    assert s == pytest.approx(0.4 * 2 / 4)


def test_sentence_start_uses_bos(sb):
    a = bl.uniform_alphas(0.4, 3)
    # 句首只补一个 <s>：第一个词由二元 (<s>, w) 预测；我们 2 次、他们 2 次、五月 1 次
    assert sb.word_score([sb.bos], sb.index["我们"], a) == pytest.approx(2 / 5)


def test_scores_match_word_score(sb):
    a = {2: 0.7, 3: 0.3}
    for words in (["学校", "召开"], ["我们"], [], ["阳光", "十分"], ["未见", "历史"]):
        ctx = ctx_of(sb, words)
        full = sb.scores(ctx, a)
        for w in range(sb.V):
            assert full[w] == pytest.approx(sb.word_score(ctx, w, a))


@pytest.mark.parametrize("alphas", [{2: 0.4, 3: 0.4}, {2: 1.0, 3: 1.0}, {2: 0.2, 3: 0.9}])
def test_profile_gives_normalized_probability(sb, alphas):
    """由 profile 在任意 α 下算出的归一化概率，与对整个词表打分再归一化的结果相同。"""
    for words in (["学校", "召开"], ["我们", "学校"], [], ["召开", "了"], ["阳光", "未见"]):
        ctx = ctx_of(sb, words)
        full = sb.scores(ctx, alphas)
        z = full.sum()
        for w in (sb.index["了"], sb.index["大会"], sb.eos, sb.index["阳光"], sb.unk):
            n, level, value, S = sb.profile(ctx, w)
            f = sb.factors(n, alphas)
            z_prof = sum(f[k] * S[k] for k in range(1, n + 1))
            assert z_prof == pytest.approx(z)
            assert f[level] * value / z_prof == pytest.approx(full[w] / z)


def test_scores_not_normalized(sb):
    s = sb.scores(ctx_of(sb, ["学校", "召开"]), bl.uniform_alphas(0.4, 3))
    assert s.sum() > 1.0  # 「了」已占 1.0，回退项再加上去


def test_unk_gets_zero_unigram(sb):
    assert sb.unigram[sb.unk] == 0.0
    assert sb.unigram[sb.bos] == 0.0
    assert math.isinf(sb.sentence_log10(["未见"], bl.uniform_alphas(0.4, 3)))


def test_collect_matches_profile(sb):
    """ext_runner.sb_collect 按整句切上下文，与逐位置调用 profile 的结果相同。"""
    from src import ext_runner as er
    sents = [TRAIN[0], "他们 学校 召开 了 会议 。".split(), "未见 的 阳光".split()]
    rows = er.sb_collect(sb, sents, 3)
    k = 0
    for s in sents:
        for j in range(len(s) + 1):
            ctx = [sb.bos] + sb.ids(s[:j])
            w = sb.index.get(s[j] if j < len(s) else "</s>", sb.unk)
            n, level, value, S = sb.profile(ctx, w, 3)
            assert rows[k][:3] == (n, level, value)
            assert np.allclose(rows[k][3], S)
            k += 1
    assert k == len(rows)


def test_sentence_norm_is_log_probability(sb):
    """归一化后整句概率只取决于逐位置的归一化概率；α 任意时句子概率都不超过 1。"""
    from src import ext_runner as er
    for a in ({2: 0.4, 3: 0.4}, {2: 1.0, 3: 1.0}):
        for s in TRAIN:
            assert er.sb_sentence_norm(sb, s, a, 3) <= 1e-12


def test_sorted_gram_index():
    from src import ext_runner as er
    idx = er.SortedGramIndex(TRAIN, max_n=4)
    assert idx.copy_run("我们 学校 召开".split()) == 3
    assert idx.copy_run("大会 学校 召开 了".split()) == 3        # 「学校 召开 了」出现过，「大会 学校」没有
    assert idx.copy_run("。 我们".split()) == 0                  # 跨句的二元不算
    assert idx.copy_run("未见 了".split()) == 0


def test_cache_arrays():
    from src import ext_runner as er
    sents = ["甲 乙 甲".split(), "乙 丙".split(), "甲".split()]
    arts = ["a1", "a1", "a2"]
    oov = np.array([False, False, False, False, False, True, False, False, False], dtype=bool)
    pc, has, seen, art = er.cache_arrays(sents, arts, oov, None)
    # a1 的位置序列：甲 乙 甲 </s> 乙 丙(未登录) </s>；a2：甲 </s>
    assert list(has) == [False, True, True, True, True, True, True, False, True]
    assert pc[2] == pytest.approx(1 / 2)      # 此前「甲 乙」，甲占一半
    assert pc[4] == pytest.approx(1 / 4)      # 此前「甲 乙 甲 </s>」，乙 1 次
    assert pc[5] == 0.0 and not seen[5]       # 未登录词不在缓存里
    assert pc[6] == pytest.approx(1 / 5)      # 未登录词不进缓存：甲 乙 甲 </s> 乙，</s> 1 次
    assert list(art) == [0] * 7 + [1] * 2     # 换文章清空
    lp = np.log10(np.full(9, 0.1))
    mixed = er.mix_log10(lp, pc, has, 0.2)
    assert mixed[0] == pytest.approx(lp[0])   # 缓存为空时不插值
    assert mixed[2] == pytest.approx(math.log10(0.8 * 0.1 + 0.2 * 0.5))


def test_decay_cache_matches_direct_weights():
    from src import ext_runner as er
    sents = ["甲 乙 甲 丙 甲".split()]
    oov = np.zeros(6, dtype=bool)
    pc, _, _, _ = er.cache_arrays(sents, ["a"], oov, 0.5)
    # 预测第 5 个位置（甲）时，历史 甲 乙 甲 丙 的权重是 0.5^3, 0.5^2, 0.5^1, 0.5^0
    w = [0.125, 0.25, 0.5, 1.0]
    assert pc[4] == pytest.approx((w[0] + w[2]) / sum(w))


def test_rank_metrics_ties():
    from src import ext_common as xc
    s = np.array([0.5, 0.2, 0.2, 0.2, 0.1])
    r = xc.rank_metrics(s, 2)                # 与另外两个词并列第 2–4 名
    assert r["hit@1"] == 0.0
    assert r["hit@5"] == 1.0
    assert r["rank"] == pytest.approx(3.0)
    assert r["rr"] == pytest.approx((1 / 2 + 1 / 3 + 1 / 4) / 3)
    assert xc.rank_metrics(s, None)["hit@5"] == 0.0


def test_solve_temperature():
    from src import ext_runner as er
    raw = np.log10(np.array([0.5, 0.25, 0.125, 0.125]))
    t = er.solve_temperature(raw, 1.5)
    assert er.entropy_at(raw * math.log(10), t) == pytest.approx(1.5, abs=1e-4)
    assert t == pytest.approx(0.8, abs=0.3)
