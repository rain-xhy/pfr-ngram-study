"""data.py 的单元测试。

用手写的几行假语料覆盖真实语料里出现过的结构：多词方括号实体、单词实体、
句号后跟右引号、连续句末标点、段落末尾的右引号、没有句末标点的标题、全角数字。
不依赖 corpus/raw/ 里的真实文件。
"""
import collections
import json

import pytest

from src import data

RAW_LINES = [
    "19980101-01-001-001/m  迈向/v  １９９８年/t  新/a  世纪/n  ",
    "19980101-01-001-002/m  通过/p  [中央/n  人民/n  广播/vn  电台/n]nt  ，/w  "
    "向/p  [中国/ns]nt  人民/n  问好/v  。/w  ",
    "19980101-01-001-003/m  他/r  说/v  ：/w  “/w  好/a  。/w  ”/w  大家/r  笑/v  了/y  。/w  ",
    "",
    "19980101-01-002-001/m  真/d  的/u  吗/y  ？/w  ！/w  是/v  的/u  。/w  ”/w  ",
]

EXPECTED_CLEAN = [
    "迈向 １９９８年 新 世纪",
    "通过 中央 人民 广播 电台 ， 向 中国 人民 问好 。",
    "他 说 ： “ 好 。 ”",
    "大家 笑 了 。",
    "真 的 吗 ？ ！",
    "是 的 。 ”",
]
EXPECTED_ARTICLES = ["19980101-01-001"] * 4 + ["19980101-01-002"] * 2


@pytest.fixture
def raw_file(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_text("\n".join(RAW_LINES) + "\n", encoding="gb18030")
    return path


@pytest.fixture
def cleaned(raw_file, tmp_path):
    out = {name: tmp_path / f"{name}.txt" for name in ("clean", "tagged", "articles")}
    stats = data.clean_corpus(raw_file, out["clean"], out["tagged"], out["articles"], encoding="gb18030")
    lines = {name: path.read_text(encoding="utf-8").splitlines() for name, path in out.items()}
    return stats, lines


def test_sentences_match_expected(cleaned):
    _, lines = cleaned
    assert lines["clean"] == EXPECTED_CLEAN


def test_entity_brackets_removed_inner_tags_kept(cleaned):
    _, lines = cleaned
    assert lines["tagged"][1] == ("通过/p 中央/n 人民/n 广播/vn 电台/n ，/w "
                                  "向/p 中国/ns 人民/n 问好/v 。/w")
    assert not any("[" in line or "]" in line for line in lines["tagged"])


def test_no_sentence_starts_with_closer_or_is_only_punctuation(cleaned):
    _, lines = cleaned
    for line in lines["clean"]:
        tokens = line.split()
        assert tokens[0] not in data.CLOSERS
        assert not all(t in data.SENT_END_CHARS or t in data.CLOSERS for t in tokens)


def test_boundary_markers_not_written(cleaned):
    _, lines = cleaned
    assert not any("<s>" in line or "</s>" in line for line in lines["clean"] + lines["tagged"])


def test_articles_aligned_with_sentences(cleaned):
    _, lines = cleaned
    assert lines["articles"] == EXPECTED_ARTICLES
    assert len(lines["articles"]) == len(lines["clean"]) == len(lines["tagged"])


def test_clean_stats(cleaned):
    stats, _ = cleaned
    assert stats["sentence_count"] == 6
    assert stats["article_count"] == 2
    assert stats["token_count"] == 35
    assert stats["sentences_without_terminal_punct"] == 1  # 只有标题那一句


def test_line_without_docid_raises(tmp_path):
    raw = tmp_path / "bad.txt"
    raw.write_text("迈向/v  充满/v  希望/n\n", encoding="gb18030")
    with pytest.raises(ValueError):
        data.clean_corpus(raw, tmp_path / "c.txt", tmp_path / "t.txt", tmp_path / "a.txt", encoding="gb18030")


def test_six_month_format_extra_mark_empty_word_and_multiple_files(tmp_path):
    # 1-6 月版本（UTF-8）的两种情况：习用语后的 `/%` 附加标记，以及只剩词性、没有词形的 `/m`
    jan = tmp_path / "199801.txt"
    feb = tmp_path / "199802.txt"
    # 另有 5 处重复标注，如 `次/q/m`，词形应是「次」
    jan.write_text("19980101-01-001-001/m  这/r  是/v  明智之举/l/%  。/w  \n", encoding="utf-8-sig")
    feb.write_text("19980201-01-001-001/m  同一/v  /m  篇/q  文章/n  。/w  两/m  次/q/m  。/w  \n",
                   encoding="utf-8-sig")
    out = {name: tmp_path / f"{name}.txt" for name in ("clean", "tagged", "articles")}
    stats = data.clean_corpus([jan, feb], out["clean"], out["tagged"], out["articles"], encoding="utf-8-sig")
    assert out["clean"].read_text(encoding="utf-8").splitlines() == ["这 是 明智之举 。", "同一 篇 文章 。", "两 次 。"]
    tagged = out["tagged"].read_text(encoding="utf-8").splitlines()
    assert tagged[0] == "这/r 是/v 明智之举/l 。/w"
    assert tagged[2] == "两/m 次/q 。/w"
    assert out["articles"].read_text(encoding="utf-8").splitlines() == ["19980101-01-001"] + ["19980201-01-001"] * 2
    assert stats["dropped_empty_word_tokens"] == 1 and stats["article_count"] == 2
    fmt = data.explore_format([jan, feb], encoding="utf-8-sig")
    assert fmt["extra_mark_tokens"] == 1 and fmt["empty_word_tokens"] == 1
    assert fmt["slash_in_word_count"] == 1
    assert fmt["date_range"] == ["19980101", "19980201"]
    assert all("%" not in tag for tag, _ in fmt["pos_tag_top25"])


def _write_split_fixture(tmp_path, sentences, articles):
    clean = tmp_path / "clean.txt"
    arts = tmp_path / "articles.txt"
    clean.write_text("\n".join(" ".join(s) for s in sentences) + "\n", encoding="utf-8")
    arts.write_text("\n".join(articles) + "\n", encoding="utf-8")
    return clean, arts


def test_split_by_article_keeps_articles_intact(tmp_path):
    # 每篇文章的句子都以文章标记开头（A1/A2/B），用来核查一篇文章是否只落在一个集合。
    sentences = [["A1", "s1"], ["A1", "s2"], ["A1", "s3"],
                 ["A2", "s1"], ["A2", "s2"], ["A2", "s3"],
                 ["B", "only"]]
    articles = ["a1"] * 3 + ["a2"] * 3 + ["b"]
    clean, arts = _write_split_fixture(tmp_path, sentences, articles)
    result = data.split_by_article(clean, arts, tmp_path / "out", seed=1, ratios=(0.7, 0.15, 0.15))

    marker_to_splits = collections.defaultdict(set)
    for name in ("train", "dev", "test"):
        for line in (tmp_path / "out" / f"{name}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                marker_to_splits[line.split()[0]].add(name)
    assert set(marker_to_splits) == {"A1", "A2", "B"}
    assert all(len(splits) == 1 for splits in marker_to_splits.values())
    total_sent = sum(result["splits"][n]["sentence_count"] for n in ("train", "dev", "test"))
    assert total_sent == 7
    # 每个集合都附带逐行对齐的文章编号文件
    for name in ("train", "dev", "test"):
        sents = (tmp_path / "out" / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        arts_out = (tmp_path / "out" / f"{name}_articles.txt").read_text(encoding="utf-8").splitlines()
        assert len([s for s in sents if s]) == len([a for a in arts_out if a])


def test_project_split_follows_main_assignment(tmp_path):
    sentences = [["甲", "一"], ["甲", "二"], ["乙", "一"], ["丙", "一"]]
    articles = ["a", "a", "b", "c"]
    clean, arts = _write_split_fixture(tmp_path, sentences, articles)
    data.split_by_article(clean, arts, tmp_path / "main", seed=5, ratios=(0.5, 0.25, 0.25))
    tagged = tmp_path / "tagged.txt"
    tagged.write_text("甲/r 一/m\n甲/r 二/m\n乙/r 一/m\n丙/r 一/m\n", encoding="utf-8")
    data.project_split(tagged, arts, tmp_path / "main", tmp_path / "tag")
    for name in ("train", "dev", "test"):
        main_lines = (tmp_path / "main" / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        tag_lines = (tmp_path / "tag" / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        stripped = [" ".join(t.rpartition("/")[0] for t in line.split()) for line in tag_lines if line]
        assert stripped == [line for line in main_lines if line]


def test_sample_by_size_smaller_is_prefix_of_larger(tmp_path):
    sentences = [[f"w{i}"] * 3 for i in range(10)]
    articles = [f"a{i}" for i in range(10)]
    clean, arts = _write_split_fixture(tmp_path, sentences, articles)
    small = data.sample_by_size(clean, arts, tmp_path / "s.txt", target_words=7, seed=3)
    large = data.sample_by_size(clean, arts, tmp_path / "l.txt", target_words=16, seed=3)
    small_lines = set((tmp_path / "s.txt").read_text(encoding="utf-8").splitlines())
    large_lines = set((tmp_path / "l.txt").read_text(encoding="utf-8").splitlines())
    assert small_lines <= large_lines
    assert small["article_count"] < large["article_count"]


def test_split_by_article_links_only_long_repeated_sentences(tmp_path):
    long_sent = list("word") * 3  # 12 词，超过 MIN_LINK_LENGTH=10
    short_sent = ["图片", "："]  # 2 词，不该触发关联
    sentences = [long_sent, short_sent, long_sent, short_sent, ["独立", "句子"]]
    articles = ["a1", "a1", "a2", "a3", "a4"]
    clean, arts = _write_split_fixture(tmp_path, sentences, articles)
    result = data.split_by_article(clean, arts, tmp_path / "out", seed=1)
    # a1、a2 共享长句必须关联；a3 只贡献短句，不该被拉进同一组
    groups = data._group_articles_by_shared_sentences(
        [s for s in sentences], articles)
    group_sets = [set(g) for g in groups]
    assert {"a1", "a2"} in group_sets
    assert not any({"a1", "a3"}.issubset(g) for g in group_sets)
    assert result["largest_group_size"] == 2


def test_split_by_article_rejects_bad_ratios(tmp_path):
    clean, arts = _write_split_fixture(tmp_path, [["x"]], ["a1"])
    with pytest.raises(ValueError):
        data.split_by_article(clean, arts, tmp_path / "out", seed=1, ratios=(0.5, 0.5, 0.5))


def test_split_by_article_misaligned_files_raise(tmp_path):
    clean = tmp_path / "clean.txt"
    arts = tmp_path / "articles.txt"
    clean.write_text("a b\nc d\n", encoding="utf-8")
    arts.write_text("only_one\n", encoding="utf-8")
    with pytest.raises(ValueError):
        data.split_by_article(clean, arts, tmp_path / "out", seed=1)


def test_build_vocab_filters_by_min_freq(tmp_path):
    train = tmp_path / "train.txt"
    train.write_text("常见 常见 常见 词 词 罕见\n常见 词\n", encoding="utf-8")
    result = data.build_vocab(train, tmp_path / "vocab.json", min_freq=2)
    vocab = json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8"))
    assert vocab == {"常见": 4, "词": 3}
    assert result["vocab_size"] == 2
    assert result["rare_type_count"] == 1  # 只有「罕见」被排除
    assert result["rare_token_count"] == 1
    assert result["total_tokens"] == 8


def test_apply_vocab_maps_rare_words(tmp_path):
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"常见": 4, "词": 3}, ensure_ascii=False), encoding="utf-8")
    sentences = tmp_path / "s.txt"
    sentences.write_text("常见 罕见 词\n", encoding="utf-8")
    result = data.apply_vocab(sentences, vocab_path, tmp_path / "out.txt")
    out = (tmp_path / "out.txt").read_text(encoding="utf-8").strip()
    assert out == f"常见 {data.RARE_TOKEN} 词"
    assert result["rare_token_count"] == 1
    assert result["token_count"] == 3


def test_sample_by_size_stops_at_or_after_target(tmp_path):
    # 三篇文章各 4 词；目标 5 词应恰好覆盖两篇（4 词不够，加第二篇到 8）。
    sentences = [["a", "b", "c", "d"], ["e", "f", "g", "h"], ["i", "j", "k", "l"]]
    articles = ["a1", "a2", "a3"]
    clean, arts = _write_split_fixture(tmp_path, sentences, articles)
    result = data.sample_by_size(clean, arts, tmp_path / "sample.txt", target_words=5, seed=1)
    assert result["actual_words"] >= 5
    assert result["article_count"] == 2
    lines = (tmp_path / "sample.txt").read_text(encoding="utf-8").splitlines()
    assert sum(len(l.split()) for l in lines) == result["actual_words"]


def test_sample_by_size_all_articles_when_target_exceeds_total(tmp_path):
    sentences = [["a", "b"], ["c", "d"]]
    articles = ["a1", "a2"]
    clean, arts = _write_split_fixture(tmp_path, sentences, articles)
    result = data.sample_by_size(clean, arts, tmp_path / "sample.txt", target_words=100, seed=1)
    assert result["article_count"] == 2
    assert result["actual_words"] == 4


def test_explore_format_on_fixture(raw_file):
    fmt = data.explore_format(raw_file, encoding="gb18030")
    assert fmt["blank_lines"] == 1
    assert fmt["lines_without_docid"] == 0
    assert fmt["article_count"] == 2
    assert fmt["token_count"] == 35
    assert fmt["untagged_token_count"] == 0
    assert fmt["entity_start_count"] == fmt["entity_end_count"] == 2
    assert fmt["entity_type_distribution"] == [("nt", 2)]
    assert fmt["nested_entity_count"] == 0
    assert fmt["entities_crossing_line"] == 0
    assert fmt["fullwidth_alnum_tokens"] == 1
    assert fmt["halfwidth_alnum_tokens"] == 0
    assert fmt["multi_sentence_end_lines"] == 2
    # 粘连形式只出现在原始 tag 字符串里，剥掉之后的词性里不再有 `]`
    assert fmt["raw_tag_string_count"] > fmt["pos_tag_count"]
    assert all("]" not in tag for tag, _ in fmt["pos_tag_top25"])
