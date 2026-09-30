"""核对训练时的峰值内存是不是 lmplz 自己的。

依次让 Python 进程涨到约 100、600 MiB 后调用 kenlm_wrapper.train，对比 /usr/bin/time 统计的峰值、
KenLM 自己打印的 RSSMax，以及不经 time、直接从 Python 启动 lmplz 时 KenLM 打印的 RSSMax。
如果直接启动时的数值随父进程内存上涨，而经 time 启动时不变，说明直接启动测到的是父进程的高水位。
用法（WSL）：bash scripts/check_rss.sh
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import kenlm_wrapper as kw  # noqa: E402
from src.common import KENLM_BIN, PROCESSED_DIR, WORK_DIR  # noqa: E402

RSS = re.compile(r"RSSMax:(\d+) kB")


def self_rss_mb():
    with open("/proc/self/status") as f:
        return next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1024


def direct_run(corpus, arpa, work):
    """不经 /usr/bin/time，直接从当前 Python 进程启动 lmplz，返回 KenLM 自报的 RSSMax。"""
    cmd = [str(KENLM_BIN / "lmplz"), "-o", "3", "-S", "512M", "-T", str(work), "--arpa", str(arpa)]
    with open(corpus, "rb") as stdin:
        proc = subprocess.run(cmd, stdin=stdin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    found = RSS.findall(proc.stderr.decode("utf-8", "replace"))
    return int(found[-1]) / 1024 if found else float("nan")


def main():
    work = WORK_DIR / "check_rss"
    work.mkdir(parents=True, exist_ok=True)
    local = work / "train.txt"
    local.write_bytes((PROCESSED_DIR / "splits_main" / "train.txt").read_bytes())
    held = []
    for target in (100, 600):
        while self_rss_mb() < target:
            block = bytearray(50 * 2 ** 20)
            block[::4096] = b"\x01" * len(block[::4096])
            held.append(block)
        r = kw.train(local, work / "m3.arpa", 3, memory="512M", temp_dir=work)
        d = direct_run(local, work / "m3_direct.arpa", work)
        print(f"父进程常驻 {self_rss_mb():.0f} MiB：经 time 启动 -> time {r['peak_rss_mb']:.1f} / "
              f"KenLM 自报 {r['kenlm_reported_rssmax_mb']:.1f} MiB；直接启动 -> KenLM 自报 {d:.1f} MiB")
    r = kw.train(local, work / "m3p.arpa", 3, memory="512M", prune=[0, 0, 1], temp_dir=work)
    print("剪枝训练的折扣行：", [(x["order"], x["count_field"], x["D1"]) for x in r["discounts"]])


if __name__ == "__main__":
    main()
