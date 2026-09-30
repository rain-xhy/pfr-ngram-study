"""看按文章划分时被长重复句绑在一起的文章组：最大的几组有多少篇、跨几天、靠哪些句子连起来。

用法（WSL）：PFR_CORPUS=pfr6 ~/pfr-venv/bin/python scripts/inspect_groups.py
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data  # noqa: E402


def main():
    sentences, articles = data._read_aligned(data.PROCESSED_DIR / "corpus_clean.txt",
                                            data.PROCESSED_DIR / "articles.txt")
    groups = data._group_articles_by_shared_sentences(sentences, articles)
    sizes = collections.Counter(len(g) for g in groups)
    print(f"语料 {data.CORPUS_TAG}：{len(set(articles)):,} 篇文章，分成 {len(groups):,} 组；"
          f"单篇成组 {sizes[1]:,}，2-3 篇 {sizes[2] + sizes[3]:,}，4 篇以上 {sum(v for k, v in sizes.items() if k >= 4):,}")
    by_article = collections.defaultdict(list)
    for s, a in zip(sentences, articles):
        by_article[a].append(s)
    for g in sorted(groups, key=len, reverse=True)[:3]:
        days = sorted({a[:8] for a in g})
        shared = collections.Counter()
        for a in g:
            for s in {tuple(x) for x in by_article[a] if len(x) >= data.MIN_LINK_LENGTH}:
                shared[s] += 1
        links = sorted(((s, c) for s, c in shared.items() if c > 1), key=lambda x: -x[1])
        print(f"\n组大小 {len(g)}，日期 {days[0]} ~ {days[-1]}（{len(days)} 天），"
              f"起连接作用的长句 {len(links)} 种，出现最多的几句：")
        for s, c in links[:4]:
            print(f"  {c} 篇：{''.join(s)[:60]}")


if __name__ == "__main__":
    main()
