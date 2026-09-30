"""把 E2 每次训练的耗时拆开看：启动到第一阶段、五个阶段、CPU 时间、/usr/bin/time 统计的耗时。

墙钟时间忽长忽短（同样的 CPU 时间，墙钟差出一半）时，看是哪个阶段变慢：
写 ARPA 阶段变慢多半是磁盘写入受干扰，计数排序阶段变慢多半是内存紧张。
用法：PFR_CORPUS=pfr6 python scripts/inspect_timing.py（Windows、WSL 都能跑，只读 results/<语料>/runs/E2_orders.json）
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    corpus = os.environ.get("PFR_CORPUS", "pfr6")
    e2 = json.loads((ROOT / "results" / corpus / "runs" / "E2_orders.json").read_text(encoding="utf-8"))
    print(f"语料 {corpus}")
    print("阶数 次序  墙钟    启动   CPU    各阶段（计数排序 调整计数 初始概率 插值概率 写ARPA）")
    for n in sorted(e2["orders"], key=int):
        for r in e2["orders"][n]["runs"]:
            stages = " ".join(f"{s['seconds']:6.2f}" for s in r["stages"])
            cpu = r.get("user_seconds", 0) + r.get("sys_seconds", 0)
            print(f"{n:>3}  {r['repeat']:>3}  {r['wall_seconds']:6.2f}  {r['startup_seconds'] or 0:5.2f}  "
                  f"{cpu:5.2f}   {stages}")


if __name__ == "__main__":
    main()
