#!/usr/bin/env python3
"""重新训练 E5 的 1月/2月/3月/6月模型，保存文件供 E21 使用。"""
import time
from pathlib import Path
from .common import PROCESSED_DIR, WORK_DIR
from . import kenlm_wrapper as kw
from .ext_runner import log

# 1月/2月/3月对应 E5 的语料文件
CORPUS_MAP = {
    "1m": "train_100k.txt",   # 1个月约 10万词
    "2m": "train_200k.txt",   # 2个月约 20万词
    "3m": "train_350k.txt",   # 3个月约 35万词
    "6m": "train.txt",        # 6个月全量
}


def train_and_save_models():
    """训练并保存 1月/2月/3月/6月的 5 元模型"""
    scale_dir = PROCESSED_DIR / "scale_samples"
    split_dir = PROCESSED_DIR / "splits_main"
    models_dir = WORK_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for key, corpus_file in CORPUS_MAP.items():
        output_arpa = models_dir / f"kn5_{key}.arpa"

        if output_arpa.exists():
            log(f"  kn5_{key}.arpa 已存在，跳过")
            continue

        # 尝试两个可能的路径
        corpus_path = scale_dir / corpus_file
        if not corpus_path.exists():
            corpus_path = split_dir / corpus_file

        if not corpus_path.exists():
            log(f"  {corpus_file} 不存在，跳过 {key}")
            continue

        log(f"  训练 {key}（{corpus_file}）-> kn5_{key}.arpa")
        start = time.perf_counter()

        # 用 kenlm_wrapper 训练 5 元模型
        try:
            kw.train(corpus_path, output_arpa, order=5, memory="512M")
            elapsed = time.perf_counter() - start
            size_mb = output_arpa.stat().st_size / (1024 * 1024)
            log(f"    完成：{size_mb:.1f} MB，用时 {elapsed:.1f}s")
        except Exception as e:
            log(f"    失败：{str(e)[:200]}")
            continue


if __name__ == "__main__":
    log("开始重新训练 E5 月份模型")
    train_and_save_models()
    log("完成")
