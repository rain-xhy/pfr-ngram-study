"""语料获取、清洗、划分、词表构建。

对应执行指南 S0-S1：探测语料真实格式 -> 写清理脚本 -> 按文章划分 -> 建词表。
清理规则不能在语料到位前写死——不同版本人民日报语料的标注体系不同
（1998 版 vs 2014 版，是否含嵌套命名实体，编码 GBK/GB18030/UTF-8），
必须先跑 explore_format 的探测再决定正则。

在项目根目录执行 `python -m src.data`：对 corpus/raw/ 下的语料跑一遍探测与清洗，
清洗结果写到 corpus/processed/，统计写到 results/s1_format.json 与 results/s1_clean.json。
"""
import collections
import json
import os
import random
import re
from pathlib import Path

SPLIT_SEED = 20260929  # 划分与抽样共用的种子，写死以保证可复现

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 语料登记：环境变量 PFR_CORPUS 选哪一份，默认用 1998 年 1-6 月。
# 每份语料的清洗结果、实验结果、模型文件各放在以语料名命名的子目录里，互不覆盖。
CORPORA = {
    "pfr6": {
        "label": "人民日报 1998 年 1-6 月（PFR 标注语料，chenhui-bupt/PeopleDaily1998 版）",
        "files": [PROJECT_ROOT / "corpus" / "raw" / "pfr_1998H1" / f"1998{m:02d}.txt" for m in range(1, 7)],
        "encoding": "utf-8-sig",
        "scale_targets": [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 6_000_000],
    },
    "figshare_jan": {
        "label": "人民日报 1998 年 1 月（Figshare DOI 10.6084/m9.figshare.5777397.v1）",
        "files": [PROJECT_ROOT / "corpus" / "raw" / "人民日报语料.txt"],
        "encoding": "gb18030",
        "scale_targets": [100_000, 200_000, 350_000, 500_000, 1_000_000],
    },
}
CORPUS_TAG = os.environ.get("PFR_CORPUS", "pfr6")
if CORPUS_TAG not in CORPORA:
    raise SystemExit(f"PFR_CORPUS={CORPUS_TAG!r} 不在 {sorted(CORPORA)} 里")
CORPUS = CORPORA[CORPUS_TAG]
RAW_FILES = CORPUS["files"]
RAW_ENCODING = CORPUS["encoding"]
SCALE_TARGETS_WORDS = CORPUS["scale_targets"]  # E5 语料规模学习曲线的取点
RAW_CORPUS = RAW_FILES[0]  # 只有一个文件的语料沿用旧名字
PROCESSED_DIR = PROJECT_ROOT / "corpus" / "processed" / CORPUS_TAG
RESULTS_DIR = PROJECT_ROOT / "results" / CORPUS_TAG

# 行首文档编号：日期(8)-版次(2)-篇号(3)-段号(3)/m。一行是一个段落，
# 编号前三段相同的行属于同一篇文章。
DOCID_RE = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-(\d{3})/m\s")
# 实体末词词性后面粘连的「]实体类型」，如 `电台/n]nt` 里的 `]nt`
CLOSE_BRACKET_RE = re.compile(r"\](\w+)$")
SENT_END_TAG_RE = re.compile(r"[。！？]/w")
HALFWIDTH_ALNUM_RE = re.compile(r"[0-9A-Za-z]")
FULLWIDTH_ALNUM_RE = re.compile(r"[０-９Ａ-Ｚａ-ｚ]")

EXTRA_MARK = "/%"

SENT_END_CHARS = frozenset("。！？")
# 紧跟在句末标点后面、仍属于本句的右引号与右括号
CLOSERS = frozenset("”’」』）】》")


def _as_list(raw_path):
    """raw_path 可以是一个文件，也可以是按月份排好的一组文件。"""
    return [Path(p) for p in raw_path] if isinstance(raw_path, (list, tuple)) else [Path(raw_path)]


def _raw_lines(raw_path, encoding=None):
    """逐个文件读出全部行（去掉行尾换行），多个文件按给定顺序接起来。"""
    lines = []
    for path in _as_list(raw_path):
        with open(path, encoding=encoding or RAW_ENCODING) as f:
            lines.extend(line.rstrip("\n") for line in f)
    return lines


def _split_token(tok: str) -> tuple:
    """把一个原始 token 拆成 (词形, 词性, 是否实体首词, 实体类型)。

    `[中央/n` -> ('中央', 'n', True, None)，`电台/n]nt` -> ('电台', 'n', False, 'nt')，
    `迈向/v` -> ('迈向', 'v', False, None)。没有 `/` 的 token 词性返回 None。
    用最后一个 `/` 切分；语料里的词形不含 `/`，见 explore_format。
    """
    starts_entity = tok.startswith("[")
    if starts_entity:
        tok = tok[1:]
    # 1-6 月版本里约 1,300 个习用语后面多了一个 `/%` 附加标记（如 `明智之举/l/%`），先剥掉
    if tok.endswith(EXTRA_MARK):
        tok = tok[:-len(EXTRA_MARK)]
    word, sep, tag = tok.rpartition("/")
    if not sep:
        return tok, None, starts_entity, None
    entity_type = None
    m = CLOSE_BRACKET_RE.search(tag)
    if m:
        entity_type = m.group(1)
        tag = tag[:m.start()]
    # 1-6 月版本里有 5 处重复标注 `次/q/m`：按最后一个 `/` 切会得到词形「次/q」。语料的词形里不含 `/`
    # （Figshare 1 月为 0 处，1-6 月除这 5 处外也为 0），所以词形里还有 `/` 时取第一段作词形、第一个标记作词性
    if "/" in word:
        word, _, tag = word.partition("/")
    return word, tag, starts_entity, entity_type


def explore_format(raw_path, encoding=None) -> dict:
    """只读探测，不修改语料。回答执行指南 S1 的七个问题，并核对清洗规则依赖的假设。

    raw_path 是一个文件或按月份排好的一组文件。对 Figshare 1998 年 1 月语料的结论（数字见
    results/figshare_jan/s1_format.json）：GB18030 编码；每个非空行都以文档编号开头，一行是一个段落；
    方括号实体单层、不跨行，左括号粘在首词词形前（`[中央/n`），`]实体类型` 粘在末词词性后（`电台/n]nt`）；
    词形里不含 `/`；数字和字母全部是全角。1-6 月版本（UTF-8）另有约 1,300 个 `/%` 附加标记、
    1 个没有词形的 `/m`，其余相同。clean_corpus 的规则建立在这些结论上，换语料时先重跑本函数，
    看哪一条不再成立。

    词性有两种统计口径。raw_tag_string_count 按原始 tag 字符串计，`n` 和 `n]nt`
    算两种；pos_tag_count 先剥掉粘连的 `]实体类型` 再计，这才是词性体系本身的大小。
    """
    lines = _raw_lines(raw_path, encoding)
    nonblank = [line for line in lines if line.strip()]

    dates, banci, articles = set(), set(), set()
    raw_tags = collections.Counter()
    pos_tags = collections.Counter()
    entity_types = collections.Counter()
    lines_without_docid = multi_end_lines = 0
    token_count = untagged = slash_in_word = fullwidth = halfwidth = 0
    extra_marks = empty_words = 0
    entity_starts = entity_ends = nested = crossing = 0

    for line in nonblank:
        m = DOCID_RE.match(line)
        if m:
            dates.add(m.group(1))
            banci.add(m.group(2))
            articles.add("-".join(m.group(1, 2, 3)))
            body = line[m.end():]
        else:
            lines_without_docid += 1
            body = line
        if len(SENT_END_TAG_RE.findall(body)) > 1:
            multi_end_lines += 1
        inside_entity = False
        for tok in body.split():
            token_count += 1
            extra_marks += tok.endswith(EXTRA_MARK)
            word, tag, starts_entity, entity_type = _split_token(tok)
            if tag is None:
                untagged += 1
                continue
            if not word:
                empty_words += 1
            raw_tags[tok.rpartition("/")[2]] += 1
            pos_tags[tag] += 1
            # 去掉实体括号与 /% 之后仍有两个及以上 `/`：词形含 `/` 或重复标注，_split_token 按重复标注处理
            if tok.lstrip("[").removesuffix(EXTRA_MARK).count("/") > 1:
                slash_in_word += 1
            if FULLWIDTH_ALNUM_RE.search(word):
                fullwidth += 1
            if HALFWIDTH_ALNUM_RE.search(word):
                halfwidth += 1
            if starts_entity:
                entity_starts += 1
                if inside_entity:
                    nested += 1
                inside_entity = True
            if entity_type is not None:
                entity_ends += 1
                entity_types[entity_type] += 1
                inside_entity = False
        if inside_entity:
            crossing += 1

    return {
        "files": [p.name for p in _as_list(raw_path)],
        "encoding": encoding or RAW_ENCODING,
        "extra_mark_tokens": extra_marks,
        "empty_word_tokens": empty_words,
        "total_lines": len(lines),
        "blank_lines": len(lines) - len(nonblank),
        "lines_without_docid": lines_without_docid,
        "date_range": [min(dates), max(dates)] if dates else None,
        "date_count": len(dates),
        "banci_values": sorted(banci),
        "article_count": len(articles),
        "token_count": token_count,
        "untagged_token_count": untagged,
        "raw_tag_string_count": len(raw_tags),
        "pos_tag_count": len(pos_tags),
        "pos_tag_top25": pos_tags.most_common(25),
        "entity_start_count": entity_starts,
        "entity_end_count": entity_ends,
        "entity_type_distribution": entity_types.most_common(),
        "nested_entity_count": nested,
        "entities_crossing_line": crossing,
        "slash_in_word_count": slash_in_word,
        "fullwidth_alnum_tokens": fullwidth,
        "halfwidth_alnum_tokens": halfwidth,
        "multi_sentence_end_lines": multi_end_lines,
    }


def _segment(tokens: list) -> list:
    """把一个段落的 (词形, 词性) 序列切成句子。

    切分点在「。！？」之后，连续的句末标点（如「？！」）和紧随其后的右引号、
    右括号都留在本句，所以 `好/a 。/w ”/w 大家/r` 切成「好 。 ”」和「大家 …」。
    段落末尾没有句末标点的部分（标题、署名、括注）单独成句。
    """
    sentences, current = [], []
    i, n = 0, len(tokens)
    while i < n:
        word, tag = tokens[i]
        current.append((word, tag))
        i += 1
        if tag == "w" and word in SENT_END_CHARS:
            while i < n and tokens[i][1] == "w" and tokens[i][0] in SENT_END_CHARS:
                current.append(tokens[i])
                i += 1
            while i < n and tokens[i][1] == "w" and tokens[i][0] in CLOSERS:
                current.append(tokens[i])
                i += 1
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def clean_corpus(raw_path, clean_out, tagged_out, articles_out, encoding=None) -> dict:
    """去文档编号与方括号、按句切分，一句一行写出三份对齐的文件。

    - clean_out：纯词形，空格分隔，主线训练用
    - tagged_out：词形/词性，方括号同样拆开，给「清理 vs 未清理」对照实验用
    - articles_out：每行是对应句子所属的文章编号（日期-版次-篇号），
      split_by_article 靠它按整篇文章划分训练/验证/测试集

    方括号实体拆开、保留内部各词（`[中国/ns 政府/n]nt` -> `中国 政府`），不合并成
    一个 token，词粒度与语料其余部分一致。标点保留为 token。
    句首句尾不写 <s> </s>：lmplz 读到语料里的字面 `<s>` 会报
    「Special word <s> is not allowed in the corpus」并退出，边界符由 KenLM 按行自动添加。
    """
    clean_lines, tagged_lines, article_lines = [], [], []
    token_count = no_terminal = dropped_empty = 0
    articles = set()
    for path in _as_list(raw_path):
        with open(path, encoding=encoding or RAW_ENCODING) as f:
            numbered = list(enumerate(f, 1))
        for line_no, line in numbered:
            if not line.strip():
                continue
            m = DOCID_RE.match(line)
            if not m:
                raise ValueError(f"{path.name} 第 {line_no} 行没有文档编号：{line[:40]!r}")
            article = "-".join(m.group(1, 2, 3))
            articles.add(article)
            tokens = []
            for tok in line[m.end():].split():
                word, tag = _split_token(tok)[:2]
                if not word:  # 没有词形、只剩词性的标注（如 5 月语料里的一个 `/m`），不是正文
                    dropped_empty += 1
                    continue
                tokens.append((word, tag))
            for sent in _segment(tokens):
                clean_lines.append(" ".join(w for w, _ in sent))
                tagged_lines.append(" ".join(f"{w}/{t}" if t else w for w, t in sent))
                article_lines.append(article)
                token_count += len(sent)
                if not any(t == "w" and w in SENT_END_CHARS for w, t in sent):
                    no_terminal += 1

    for path, rows in ((clean_out, clean_lines), (tagged_out, tagged_lines),
                       (articles_out, article_lines)):
        Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")

    return {
        "sentence_count": len(clean_lines),
        "article_count": len(articles),
        "dropped_empty_word_tokens": dropped_empty,
        "token_count": token_count,
        "clean_vocab_size": len({w for s in clean_lines for w in s.split()}),
        "tagged_vocab_size": len({w for s in tagged_lines for w in s.split()}),
        "sentences_without_terminal_punct": no_terminal,
    }


MIN_LINK_LENGTH = 10  # 2026-09-29 与用户确认：只用 >=10 词的重复句关联文章


class _UnionFind:
    """按文章编号做并查集，用于把含相同长句的文章绑到一起。"""

    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _rel(path) -> str:
    """结果文件里记相对项目根目录的路径，换机器、换系统（Windows / WSL）都能对上。"""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _read_aligned(clean_path, articles_path):
    """读回 clean_corpus 产出的两份对齐文件，返回 (句子列表, 文章编号列表)。"""
    sentences = [line.split() for line in
                 Path(clean_path).read_text(encoding="utf-8").splitlines()]
    articles = Path(articles_path).read_text(encoding="utf-8").splitlines()
    if len(sentences) != len(articles):
        raise ValueError(f"句子数 {len(sentences)} 与文章编号行数 {len(articles)} 不一致，"
                         "clean_corpus 的三份输出必须逐行对齐")
    return sentences, articles


def _group_articles_by_shared_sentences(sentences, articles, min_length=MIN_LINK_LENGTH):
    """把含有相同长句（>=min_length 词）的文章合并成组，短句不参与关联。

    2026-09-29 与用户确认的做法：任意长度的重复句关联会在本语料上产生
    最大 318 篇的巨型组（源于「图片：」「本报评论员」这类版面套话，
    见 `回复同学/回复素材.md` 给 LynxPeng 的第 1 条）；改成只用 >=10 词的
    句子关联后，最大组降到 3 篇。组内文章必须整体分到同一个数据集，
    否则测试集会出现训练集里逐词见过的长句。
    """
    uf = _UnionFind(set(articles))
    owner = {}
    for sent, article in zip(sentences, articles):
        if len(sent) < min_length:
            continue
        key = tuple(sent)
        if key in owner:
            uf.union(article, owner[key])
        else:
            owner[key] = article
    groups = collections.defaultdict(list)
    for a in sorted(set(articles)):
        groups[uf.find(a)].append(a)
    return list(groups.values())


def split_by_article(clean_path, articles_path, out_dir, seed: int,
                      ratios=(0.8, 0.1, 0.1)) -> dict:
    """按文章级切分训练/验证/测试集，避免同一篇文章的句子跨集合泄漏。

    含相同长句（>=10 词）的文章先合并成组再整体分配，短的版面套话
    （「图片：」之类）不参与关联，理由见 `_group_articles_by_shared_sentences`。
    组按大小降序处理，每次把整组分给「目标占比 x 总文章数 - 已分文章数」
    最大的那个集合，这样大组不会把某个小集合的比例撑爆
    （做法与 LynxPeng 的 `split_articles` 一致，只是关联规则加了长度门槛）。

    输出 out_dir 下的 train.txt / dev.txt / test.txt，每份是清洗后的句子、
    一句一行、空格分词，可直接喂给 lmplz。
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios 之和必须为 1，收到 {ratios}")
    sentences, articles_col = _read_aligned(clean_path, articles_path)
    by_article = collections.defaultdict(list)
    for sent, article in zip(sentences, articles_col):
        by_article[article].append(sent)

    groups = _group_articles_by_shared_sentences(sentences, articles_col)
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    names = ["train", "dev", "test"]
    targets = dict(zip(names, ratios))
    total_articles = len(by_article)
    assigned = {name: [] for name in names}
    for group in groups:
        dest = max(names, key=lambda n: targets[n] * total_articles - len(assigned[n]))
        assigned[dest].extend(group)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stats = {}
    for name in names:
        article_ids = assigned[name]
        pairs = [(s, a) for a in sorted(article_ids) for s in by_article[a]]
        out_path = Path(out_dir) / f"{name}.txt"
        out_path.write_text("\n".join(" ".join(s) for s, _ in pairs) + "\n", encoding="utf-8")
        # 与 {name}.txt 逐行对齐的文章编号，E5 从训练集按整篇文章抽样时要用
        (Path(out_dir) / f"{name}_articles.txt").write_text(
            "\n".join(a for _, a in pairs) + "\n", encoding="utf-8")
        stats[name] = {
            "article_count": len(article_ids),
            "sentence_count": len(pairs),
            "token_count": sum(len(s) for s, _ in pairs),
            "path": _rel(out_path),
        }
    return {
        "seed": seed,
        "min_link_length": MIN_LINK_LENGTH,
        "group_count": len(groups),
        "largest_group_size": max((len(g) for g in groups), default=0),
        "splits": stats,
    }


def project_split(aligned_path, articles_path, split_dir, out_dir,
                  names=("train", "dev", "test")) -> dict:
    """把主划分的文章归属套到另一份与 clean 逐行对齐的文件上（如带词性版）。

    句子顺序与 split_by_article 的输出一致：先按文章编号排序，文章内保持原顺序。
    E0 标注清理对照要求几份语料的训练/测试文章完全相同，差别只来自标注本身。
    """
    lines = Path(aligned_path).read_text(encoding="utf-8").splitlines()
    articles = Path(articles_path).read_text(encoding="utf-8").splitlines()
    if len(lines) != len(articles):
        raise ValueError(f"{aligned_path} 有 {len(lines)} 行，文章编号有 {len(articles)} 行，无法对齐")
    by_article = collections.defaultdict(list)
    for line, article in zip(lines, articles):
        by_article[article].append(line)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stats = {}
    for name in names:
        ids = set(Path(split_dir, f"{name}_articles.txt").read_text(encoding="utf-8").split())
        rows = [line for a in sorted(ids) for line in by_article[a]]
        Path(out_dir, f"{name}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        stats[name] = {"article_count": len(ids), "sentence_count": len(rows)}
    return stats


RARE_TOKEN = "〈低频词〉"  # 参考 70clementine：用有真实训练计数的占位符，不用 KenLM 保留符号 <unk>


def build_vocab(train_path, out_vocab_path, min_freq: int = 2) -> dict:
    """从训练集构建词表，频次低于 min_freq 的词记为待映射的低频词。

    只统计训练集，不看验证/测试集，避免词表本身就泄漏了它们的信息。
    返回词表大小、词表本身存到 out_vocab_path（JSON，词到训练集频次的映射，
    只含保留下来的词，不含被映射掉的低频词——低频词的还原就是 RARE_TOKEN 本身）。
    """
    counts = collections.Counter(
        w for line in Path(train_path).read_text(encoding="utf-8").splitlines()
        for w in line.split())
    vocab = {w: n for w, n in counts.items() if n >= min_freq}
    rare_types = len(counts) - len(vocab)
    rare_tokens = sum(n for w, n in counts.items() if n < min_freq)
    Path(out_vocab_path).write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "min_freq": min_freq,
        "vocab_size": len(vocab),
        "rare_type_count": rare_types,
        "rare_token_count": rare_tokens,
        "total_tokens": sum(counts.values()),
        "vocab_path": _rel(out_vocab_path),
    }


def apply_vocab(sentences_path, vocab_path, out_path) -> dict:
    """把不在词表里的词替换成 RARE_TOKEN，供需要封闭词表的评估场景使用。

    E2-E6 训练用的是未替换的原始句子（KenLM 自己处理 OOV，见 kenlm_wrapper.
    evaluate_ppl），本函数只在需要「训练集内外一致词表」的场景（如报告里
    对照 70clementine 的低频映射策略）时按需调用。
    """
    vocab = set(json.loads(Path(vocab_path).read_text(encoding="utf-8")))
    lines = Path(sentences_path).read_text(encoding="utf-8").splitlines()
    mapped = [[w if w in vocab else RARE_TOKEN for w in line.split()] for line in lines]
    Path(out_path).write_text(
        "\n".join(" ".join(s) for s in mapped) + "\n", encoding="utf-8")
    total = sum(len(s) for s in mapped)
    rare = sum(w == RARE_TOKEN for s in mapped for w in s)
    return {"token_count": total, "rare_token_count": rare,
            "rare_rate": rare / total if total else 0.0}


def sample_by_size(clean_path, articles_path, out_path, target_words: int, seed: int) -> dict:
    """按目标词数抽样语料，用于 E5 语料规模学习曲线（10/20/35/50/100 万词）。

    按文章整体抽样（不按句子），理由同 split_by_article：保持一篇文章内部
    的上下文完整，抽样结果仍是「一批完整文章」而不是打散的句子集合。
    文章顺序先按种子打乱，再依次累加词数，直到达到或刚超过 target_words。
    """
    sentences, articles_col = _read_aligned(clean_path, articles_path)
    by_article = collections.defaultdict(list)
    for sent, article in zip(sentences, articles_col):
        by_article[article].append(sent)

    article_ids = sorted(by_article)
    rng = random.Random(seed)
    rng.shuffle(article_ids)

    selected, total = [], 0
    for a in article_ids:
        size = sum(len(s) for s in by_article[a])
        selected.append(a)
        total += size
        if total >= target_words:
            break

    rows = [s for a in sorted(selected) for s in by_article[a]]
    Path(out_path).write_text("\n".join(" ".join(s) for s in rows) + "\n", encoding="utf-8")
    return {
        "target_words": target_words,
        "actual_words": sum(len(s) for s in rows),
        "article_count": len(selected),
        "sentence_count": len(rows),
        "seed": seed,
        "path": _rel(out_path),
    }


def main() -> None:
    """S1 完整流水线：探测 -> 清洗 -> 划分 -> 建词表 -> 规模抽样。

    只在原始语料已存在时执行到底；探测和清洗结果已存在则跳过重跑，
    重跑请先删除 corpus/processed/ 下对应文件。
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    clean_path = PROCESSED_DIR / "corpus_clean.txt"
    tagged_path = PROCESSED_DIR / "corpus_tagged.txt"
    articles_path = PROCESSED_DIR / "articles.txt"

    print(f"语料：{CORPUS_TAG}，{CORPUS['label']}，{len(RAW_FILES)} 个文件")
    fmt = explore_format(RAW_FILES)
    stats = clean_corpus(RAW_FILES, clean_path, tagged_path, articles_path)
    for name, result in (("s1_format.json", fmt), ("s1_clean.json", stats)):
        (RESULTS_DIR / name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"段落 {fmt['total_lines'] - fmt['blank_lines']}，文章 {fmt['article_count']}，"
          f"token {fmt['token_count']}，实体 {fmt['entity_end_count']}")
    print(f"切出 {stats['sentence_count']} 句，其中 "
          f"{stats['sentences_without_terminal_punct']} 句没有句末标点")
    print(f"词表（词性对照用）：清洗后 {stats['clean_vocab_size']}，带词性 {stats['tagged_vocab_size']}")

    split_dir = PROCESSED_DIR / "splits_main"
    split_stats = split_by_article(clean_path, articles_path, split_dir, seed=SPLIT_SEED)
    (RESULTS_DIR / "s1_split.json").write_text(
        json.dumps(split_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"主划分：最大关联组 {split_stats['largest_group_size']} 篇，"
          f"train/dev/test 文章数 = "
          + "/".join(str(split_stats["splits"][n]["article_count"]) for n in ("train", "dev", "test")))
    tagged_split = project_split(tagged_path, articles_path, split_dir, PROCESSED_DIR / "splits_tagged")
    print("带词性版按同一划分写出：" + "/".join(
        str(tagged_split[n]["sentence_count"]) for n in ("train", "dev", "test")) + " 句")

    vocab_stats = build_vocab(split_dir / "train.txt", PROCESSED_DIR / "vocab.json", min_freq=2)
    (RESULTS_DIR / "s1_vocab.json").write_text(
        json.dumps(vocab_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"词表（训练集，频次>=2）：{vocab_stats['vocab_size']} 词，"
          f"低频词种类 {vocab_stats['rare_type_count']}")

    # E5 的规模样本只从训练集里抽，评估时统一用主划分的 dev/test，
    # 否则样本里会混进测试集文章。每个规模用同一个种子，小样本是大样本的前缀，
    # 曲线上相邻两点的差别只来自新增的文章。
    scale_dir = PROCESSED_DIR / "scale_samples"
    scale_dir.mkdir(exist_ok=True)
    scale_results = []
    train_words = split_stats["splits"]["train"]["token_count"]
    for target in SCALE_TARGETS_WORDS:
        out_path = scale_dir / f"train_{target // 1000}k.txt"
        result = sample_by_size(split_dir / "train.txt", split_dir / "train_articles.txt",
                                out_path, target, seed=SPLIT_SEED)
        result["is_full_train"] = result["actual_words"] >= train_words
        scale_results.append(result)
        note = "（已取满整个训练集）" if result["is_full_train"] else ""
        print(f"规模抽样 {target:>9,} 词目标 -> 实得 {result['actual_words']:>9,} 词，"
              f"{result['article_count']} 篇文章{note}")
    (RESULTS_DIR / "s1_scale_samples.json").write_text(
        json.dumps(scale_results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
