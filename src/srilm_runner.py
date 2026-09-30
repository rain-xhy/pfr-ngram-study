#!/usr/bin/env python3
"""SRILM 与 KenLM 的平滑方法、训练开销、困惑度对比（需要先安装 SRILM）。

用法：python -m src.srilm_runner

SRILM 安装：bash scripts/build_srilm.sh，装到 ~/.local/bin 或添加到 PATH。
"""
import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path

from . import kenlm_wrapper as kw
from .experiment_runner import SPLIT_DIR, WORK_DIR, repeated_train, train_summary, read_sentences
from .ext_runner import save, log, read_json, RUNS_DIR

SRILM_ORDERS = (3, 5)
SRILM_REPEATS = 3


def srilm_installed():
    return shutil.which("ngram-count") is not None


def srilm_train(corpus_path, order, method, output, gt_mins=None):
    """用 ngram-count 训练，method 是 -kndiscount, -ukndiscount, -wbdiscount 之一。

    返回 (墙钟 s, 峰值内存 MiB)。gt_mins 是 {3: min, 4: min, ...} 字典，对应 -gt3min 等参数。
    """
    cmd = ["ngram-count", "-text", str(corpus_path), "-order", str(order), "-lm", str(output), "-unk"]
    if method == "-kndiscount":
        cmd.extend(["-kndiscount", "-interpolate"])
    elif method == "-ukndiscount":
        cmd.extend(["-ukndiscount", "-interpolate"])
    elif method == "-wbdiscount":
        cmd.extend(["-wbdiscount", "-interpolate"])
    if gt_mins:
        for n, m in gt_mins.items():
            cmd.extend([f"-gt{n}min", str(m)])
    t0 = time.perf_counter()
    r = subprocess.run(["/usr/bin/time", "-v"] + cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    peak = 0
    for line in r.stderr.splitlines():
        if "Maximum resident set size" in line:
            peak = int(line.split()[-1]) / 1024
    if r.returncode != 0:
        log(f"  ngram-count 退出码 {r.returncode}，stderr：{r.stderr[:300]}")
    return wall, peak


def srilm_ppl(arpa_path, test_path, order):
    """用 ngram -ppl 算困惑度，返回 (ppl, OOV 次数, 总词数, 总句数)。"""
    cmd = ["ngram", "-lm", str(arpa_path), "-order", str(order), "-ppl", str(test_path), "-unk"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ppl, oov, words, sents = None, 0, 0, 0
    for line in r.stdout.splitlines():
        if "ppl=" in line:
            ppl = float(line.split("ppl=")[1].split()[0])
        if "OOVs" in line:
            parts = line.split()
            oov = int(parts[4])
        if line.startswith("file") and "sentences" in line:
            # 格式: file /path: 1865 sentences, 716572 words, 4970 OOVs
            # 找 "sentences" 前的数字
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "sentences,":
                    sents = int(parts[i - 1])
                if p == "words,":
                    words = int(parts[i - 1])
    return ppl, oov, words, sents


def e20_srilm():
    """SRILM 的插值修正 KN（-kndiscount）、原始 KN（-ukndiscount）、Witten-Bell、Good-Turing/Katz
    在三元、五元上与 KenLM 的修正 KN 比训练开销、困惑度。

    训练集是全部六个月（与 E14 相同），测试集同一个。SRILM 的 ppl 默认不含未登录词，
    KenLM 的 ppl_excluding_oov 同样不含，分母一致（词数 + 句数 − 未登录词数）。
    """
    if not srilm_installed():
        return {"skipped": True, "reason": "SRILM 未安装，运行 bash scripts/build_srilm.sh"}
    train_path = SPLIT_DIR / "train.txt"
    test_path = SPLIT_DIR / "test.txt"
    test_sents = read_sentences(test_path)
    srilm_dir = WORK_DIR / "srilm"
    srilm_dir.mkdir(parents=True, exist_ok=True)
    methods = {"kndiscount": "插值修正 KN", "ukndiscount": "原始 KN", "wbdiscount": "Witten-Bell"}
    out = {"orders": {}}
    for n in SRILM_ORDERS:
        log(f"  SRILM {n} 元")
        rows = {}
        for key, label in methods.items():
            arpa = srilm_dir / f"{key}{n}.arpa"
            times, peaks = [], []
            gt_mins = {3: 1, 4: 1, 5: 1} if n >= 3 else None
            for rep in range(SRILM_REPEATS):
                w, p = srilm_train(train_path, n, f"-{key}", arpa, gt_mins)
                times.append(w)
                peaks.append(p)
                if rep == 0:
                    log(f"    {label}：预热 {w:.1f}s，峰值 {p:.0f} MiB")
            ppl, oov, words, sents = srilm_ppl(arpa, test_path, n)
            rows[key] = {"label": label, "wall_seconds": {"median": sorted(times)[1], "range": [min(times), max(times)]},
                         "peak_rss_mb": {"median": sorted(peaks)[1], "range": [min(peaks), max(peaks)]},
                         "arpa_bytes": arpa.stat().st_size, "ppl_excluding_oov": ppl, "test_oov": oov,
                         "test_words": words, "test_sentences": sents}
            log(f"    {label}：中位 {rows[key]['wall_seconds']['median']:.1f}s，"
                f"困惑度 {ppl:.2f}（不含 {oov:,} 个未登录词）")
            arpa.unlink()
        kenlm_arpa, kenlm_runs = repeated_train(f"kn{n}_for_e20", train_path, n, repeats=SRILM_REPEATS)
        kenlm_model = kw.load(kenlm_arpa)
        kenlm_ev = kw.evaluate(kenlm_model, test_sents)
        rows["kenlm"] = {"label": "KenLM 修正 KN", **train_summary(kenlm_runs),
                         "arpa_bytes": kenlm_arpa.stat().st_size, "ppl_excluding_oov": kenlm_ev["ppl_excluding_oov"],
                         "test_oov": kenlm_ev["oov_events"]}
        log(f"    KenLM：中位 {rows['kenlm']['wall_seconds']['median']:.1f}s，"
            f"困惑度 {kenlm_ev['ppl_excluding_oov']:.2f}")
        del kenlm_model
        out["orders"][str(n)] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args = ap.parse_args(argv)
    if not srilm_installed():
        log("SRILM 未安装，跳过实验。安装方法：bash scripts/build_srilm.sh")
        return
    (WORK_DIR / "tmp").mkdir(parents=True, exist_ok=True)
    log("开始 E20_srilm")
    started = time.perf_counter()
    result = e20_srilm()
    save("E20_srilm", {"elapsed_seconds": time.perf_counter() - started, **result})
    log(f"完成 E20_srilm，用时 {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
