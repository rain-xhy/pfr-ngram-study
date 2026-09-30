"""核对 results/figures/ 下的图：每张 PNG 的尺寸与是否几乎全白，SVG 里有没有字体回退产生的豆腐块。

matplotlib 找不到中文字体时会打印 "Glyph ... missing from font(s)" 警告并画出方框；
这里重新用同样的字体设置渲染一个中文标题，检查字体是否真的被加载，再逐张统计 PNG 的非白像素比例。
用法（WSL）：~/pfr-venv/bin/python scripts/check_figures.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src import reporting  # noqa: E402
from src.common import FIG_DIR  # noqa: E402


def main():
    reporting.setup_fonts()
    print("使用的字体：", plt.rcParams["font.family"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, ax = plt.subplots()
        ax.set_title("困惑度随阶数 平滑方法 最长照搬词数")
        fig.canvas.draw()
        plt.close(fig)
    missing = [str(w.message) for w in caught if "missing from" in str(w.message)]
    print("中文渲染缺字警告：", missing or "无")
    for png in sorted(FIG_DIR.glob("*.png")):
        img = mpimg.imread(png)
        rgb = img[..., :3]
        ink = float((rgb.mean(axis=2) < 0.95).mean())
        svg = png.with_suffix(".svg")
        print(f"{png.name:28s} {img.shape[1]}×{img.shape[0]}  非白像素 {ink:.1%}  "
              f"SVG {'有' if svg.exists() else '缺'} {svg.stat().st_size / 1024:.0f} KiB" if svg.exists()
              else f"{png.name:28s} {img.shape[1]}×{img.shape[0]}  非白像素 {ink:.1%}  SVG 缺")


if __name__ == "__main__":
    main()
