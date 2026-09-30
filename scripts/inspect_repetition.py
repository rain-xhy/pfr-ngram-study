"""核对 E7 里的三元自重复率：全部采样结果里有多少条出现过重复的三元组，以及贪心输出的自重复率。

用法：python scripts/inspect_repetition.py（只读 results/runs/E7_generation.json）
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rep3(tokens):
    grams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    return (1 - len(set(grams)) / len(grams)) if grams else 0.0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    e7 = json.loads((ROOT / "results" / "runs" / "E7_generation.json").read_text(encoding="utf-8"))
    recs = [r for c in e7["grid"] for r in c["records"]]
    recs += [r for v in e7["by_order"].values() for r in v["records"]]
    with_rep = [r for r in recs if rep3(r["tokens"]) > 0]
    print(f"采样生成共 {len(recs)} 条，三元组有重复的 {len(with_rep)} 条；"
          f"记录里的 repeated_3gram_rate 最大值 {max(r['repeated_3gram_rate'] for r in recs):.3f}")
    lengths = sorted(len(r["tokens"]) for r in recs)
    print(f"长度中位数 {lengths[len(lengths) // 2]}，≥30 词的 {sum(l >= 30 for l in lengths)} 条")
    for r in with_rep[:3]:
        print(f"  {rep3(r['tokens']):.3f}  {r['text'][:60]}")
    for n, v in sorted(e7["by_order"].items()):
        g = v["greedy"]
        print(f"{n} 元贪心：长度 {len(g['tokens'])}，三元自重复 {rep3(g['tokens']):.3f}（记录值 {g['repeated_3gram_rate']:.3f}）")


if __name__ == "__main__":
    main()
