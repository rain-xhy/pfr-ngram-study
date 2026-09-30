"""核对 E10 各个开头在训练集里的切分：每个词是否见过，以及训练集里怎么写「人民大会堂」「人工智能」这类词。

用法（WSL）：~/pfr-venv/bin/python scripts/check_prefix.py
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROCESSED_DIR, read_sentences  # noqa: E402
from src.experiment_runner import PREFIXES  # noqa: E402


def main():
    train = read_sentences(PROCESSED_DIR / "splits_main" / "train.txt")
    uni = collections.Counter(w for s in train for w in s)
    for name, text in PREFIXES.items():
        words = text.split()
        print(f"{name}：" + " ".join(f"{w}({uni.get(w, 0)})" for w in words))
    for target in ("人民大会堂", "人工智能", "阳光明媚"):
        hits = collections.Counter()
        for s in train:
            joined = "".join(s)
            if target not in joined:
                continue
            # 找出覆盖 target 的那段词序列
            pos = 0
            starts = []
            for w in s:
                starts.append(pos)
                pos += len(w)
            begin = joined.index(target)
            end = begin + len(target)
            span = [w for w, st in zip(s, starts) if st < end and st + len(w) > begin]
            hits[" ".join(span)] += 1
        print(f"「{target}」在训练集里的写法：{hits.most_common(5) or '没有出现'}")


if __name__ == "__main__":
    main()
