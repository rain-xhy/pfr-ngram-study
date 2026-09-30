#!/usr/bin/env python3
"""E21 案例分析：用统一种子"在阳光明媚的五月，我们学校组织了"对比所有已训练模型。"""
import time
from pathlib import Path

from .experiment_runner import WORK_DIR, load_model
from . import sampling
from .ext_runner import RUNS_DIR, log, save

SEED_TEXT = "在阳光明媚的五月，我们学校组织了"
SEED_TOKENS = SEED_TEXT.split()
MAX_LENGTH = 60


def simple_generate(model, prefix, temperature, seed, max_tokens=60):
    """简单采样生成"""
    arpa_path = WORK_DIR / f"models/kn{model.order}.arpa"
    vocab = sampling.vocab_from_arpa(arpa_path)
    gen = sampling.Generator(model, vocab)
    output = gen.generate(prefix, temperature=temperature, top_k=50, seed=seed, max_tokens=max_tokens)
    return {"text": "".join(output["generated"]), "tokens": output["generated"], "hit_eos": output["hit_eos"]}


def case_by_order():
    """维度1：n-gram 阶数对比（2/3/4/5/6元）"""
    log("维度1：阶数对比")
    results = {}
    for n in (2, 3, 4, 5, 6):
        log(f"  {n}元模型")
        model = load_model(n)
        output = simple_generate(model, SEED_TOKENS, temperature=1.0, seed=42, max_tokens=MAX_LENGTH)
        results[str(n)] = {"text": output["text"], "tokens": output["tokens"], "hit_eos": output["hit_eos"],
                           "params": {"order": n, "temperature": 1.0, "seed": 42}}
        log(f"    {output['text'][:60]}...")
        del model
    return results


def case_by_temperature():
    """维度2：温度对比（5元，T=0.5/0.7/1.0/1.3）"""
    log("维度2：温度对比（5元模型）")
    model = load_model(5)
    results = {}
    for temp in (0.5, 0.7, 1.0, 1.3):
        log(f"  T={temp}")
        output = simple_generate(model, SEED_TOKENS, temperature=temp, seed=42, max_tokens=MAX_LENGTH)
        results[f"T{temp}"] = {"text": output["text"], "tokens": output["tokens"], "hit_eos": output["hit_eos"],
                               "params": {"order": 5, "temperature": temp, "seed": 42}}
        log(f"    {output['text'][:60]}...")
    del model
    return results


def case_greedy_vs_sampling():
    """维度3：贪心 vs 采样（5元，T→0 vs T=1.0）"""
    log("维度3：贪心 vs 采样（5元模型）")
    model = load_model(5)

    log("  贪心（T=0.01）")
    greedy_output = simple_generate(model, SEED_TOKENS, temperature=0.01, seed=42, max_tokens=MAX_LENGTH)

    log("  采样（T=1.0）")
    sampling_output = simple_generate(model, SEED_TOKENS, temperature=1.0, seed=42, max_tokens=MAX_LENGTH)

    results = {
        "greedy": {"text": greedy_output["text"], "tokens": greedy_output["tokens"], "hit_eos": greedy_output["hit_eos"]},
        "sampling": {"text": sampling_output["text"], "tokens": sampling_output["tokens"], "hit_eos": sampling_output["hit_eos"]}
    }
    del model
    return results


def case_char_vs_word():
    """维度4：字级 vs 词级（复用E15的字级模型）"""
    log("维度4：字级 vs 词级")
    results = {}

    # 词级5元
    log("  词级 5 元")
    model = load_model(5)
    output = simple_generate(model, SEED_TOKENS, temperature=1.0, seed=42, max_tokens=MAX_LENGTH)
    results["word_5gram"] = {"text": output["text"], "tokens": output["tokens"], "hit_eos": output["hit_eos"]}
    log(f"    {output['text'][:60]}...")
    del model

    # 字级6元（如果存在）
    char_model_path = WORK_DIR / "models/char5.arpa"
    if char_model_path.exists():
        log("  字级 6 元")
        import kenlm
        char_model = kenlm.Model(str(char_model_path))
        char_seed = [c for w in SEED_TOKENS for c in w]
        vocab = sampling.vocab_from_arpa(char_model_path)
        gen = sampling.Generator(char_model, vocab)
        char_output = gen.generate(char_seed, temperature=1.0, top_k=50, seed=42, max_tokens=MAX_LENGTH * 3)
        results["char_6gram"] = {"text": "".join(char_output["generated"])[:100], "hit_eos": char_output["hit_eos"]}
        log(f"    {results['char_6gram']['text'][:60]}...")
        del char_model, gen
    else:
        results["char_6gram"] = {"skipped": True, "reason": "char5.arpa 不存在"}
        log("  字级模型不存在，跳过")

    return results


def case_by_corpus_size():
    """维度5：训练集规模（复用E5的不同月份模型）"""
    log("维度5：训练集规模（5元模型）")
    results = {}

    # 检查并推理所有月份模型
    corpus_configs = [
        ("1month", "kn5_1m.arpa", "1个月语料"),
        ("2months", "kn5_2m.arpa", "2个月语料"),
        ("3months", "kn5_3m.arpa", "3个月语料"),
        ("6months", "kn5.arpa", "6个月语料（默认）")
    ]

    for key, model_file, label in corpus_configs:
        model_path = WORK_DIR / f"models/{model_file}"
        if not model_path.exists():
            log(f"  {label}: 模型不存在，跳过")
            results[key] = {"skipped": True, "reason": f"{model_file} 不存在"}
            continue

        log(f"  {label}")
        import kenlm
        model = kenlm.Model(str(model_path))
        vocab = sampling.vocab_from_arpa(model_path)
        gen = sampling.Generator(model, vocab)
        output = gen.generate(SEED_TOKENS, temperature=1.0, top_k=50, seed=42, max_tokens=MAX_LENGTH)
        results[key] = {"text": "".join(output["generated"]), "tokens": output["generated"],
                       "hit_eos": output["hit_eos"], "corpus": label}
        log(f"    {results[key]['text'][:60]}...")
        del model, gen

    return results


def e21_case_study():
    """E21 主函数"""
    log(f"统一种子：{SEED_TEXT}")
    return {
        "seed": SEED_TEXT,
        "seed_tokens": SEED_TOKENS,
        "max_length": MAX_LENGTH,
        "by_order": case_by_order(),
        "by_temperature": case_by_temperature(),
        "greedy_vs_sampling": case_greedy_vs_sampling(),
        "char_vs_word": case_char_vs_word(),
        "by_corpus_size": case_by_corpus_size()
    }


def main():
    log("开始 E21_case_study（完整版）")
    started = time.perf_counter()
    result = e21_case_study()
    save("E21_case_study", {"elapsed_seconds": time.perf_counter() - started, **result})
    log(f"完成 E21_case_study，用时 {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
