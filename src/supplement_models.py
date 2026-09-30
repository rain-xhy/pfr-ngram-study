#!/usr/bin/env python3
"""补充训练 E5 的 1月/2月/3月模型，保存为独立文件供 E21 使用。"""
import subprocess
from pathlib import Path
from .common import PROCESSED_DIR, WORK_DIR
from .ext_runner import log

MONTHS_MAP = {
    "1m": ("199801", "kn5_1m.arpa"),
    "2m": ("199802", "kn5_2m.arpa"),
    "3m": ("199803", "kn5_3m.arpa"),
}


def train_monthly_models():
    """训练 1月/2月/3月的 5 元模型"""
    corpus_dir = PROCESSED_DIR / "scale_samples"
    models_dir = WORK_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for key, (month, model_name) in MONTHS_MAP.items():
        corpus_file = corpus_dir / f"train_until_{month}.txt"
        output_arpa = models_dir / model_name

        if output_arpa.exists():
            log(f"  {model_name} 已存在，跳过")
            continue

        if not corpus_file.exists():
            log(f"  {corpus_file} 不存在，跳过 {key}")
            continue

        log(f"  训练 {key}（{month}）-> {model_name}")

        # 用 lmplz 训练
        cmd = [
            "lmplz",
            "-o", "5",
            "--text", str(corpus_file),
            "--arpa", str(output_arpa),
            "--discount_fallback"
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        log(f"    完成：{output_arpa}")


if __name__ == "__main__":
    log("开始补充训练 E5 月份模型")
    train_monthly_models()
    log("完成")
