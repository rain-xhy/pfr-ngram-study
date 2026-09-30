"""实验流水线共用的路径、常量与小工具。

训练、评估、生成都在 WSL 的 ~/pfr-venv 里运行（KenLM 的 Python 绑定与二进制都装在那边），
用 scripts/run_wsl.sh 启动。data.py 只依赖标准库，Windows 与 WSL 都能跑。
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .data import CORPUS, CORPUS_TAG, PROCESSED_DIR, PROJECT_ROOT, RARE_TOKEN, RESULTS_DIR  # noqa: F401

RUNS_DIR = RESULTS_DIR / "runs"
FIG_DIR = RESULTS_DIR / "figures"
# 模型文件体积大（六个月语料的六元 ARPA 超过 1 GiB），放在 WSL 本地盘；训练计时也不受跨系统文件访问影响。
# 按语料分子目录，换语料不会覆盖另一份语料的模型
WORK_DIR = Path(os.environ.get("PFR_WORK_DIR", Path.home() / "pfr-work")) / CORPUS_TAG
KENLM_BIN = Path(os.environ.get("KENLM_BIN", Path.home() / "tools" / "kenlm" / "build" / "bin"))
KENLM_COMMIT = "4cb443e60b7bf2c0ddf3c745378f76cb59e254e5"

# 作业给的开头按语料的分词规范切分：「阳光明媚」在语料里从不作为一个词出现，
# 只以「阳光 明媚」出现（全部 1 月语料 2 次，其中训练集 1 次）
MAIN_PROMPT = "在 阳光 明媚 的 五月 ， 我们 学校 胜利 召开 了".split()
PROMPT_DISPLAY = "在阳光明媚的五月，我们学校胜利召开了"
DEFAULT_MEMORY = "512M"
# 每个训练配置重复 5 次取中位数：3 次时曾有两次被系统抖动拖慢到 5 秒（正常 0.9 秒），中位数跟着失真
REPEATS = 5
GEN_SEEDS = [11, 22, 33]  # 报告里展示原文的种子（实验清单规定的三个）
# 定量指标用 20 个种子求平均，前三个就是展示用的那三个
STAT_SEEDS = GEN_SEEDS + list(range(1001, 1018))
MAX_GEN_TOKENS = 60


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_sentences(path) -> list:
    return [line.split() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
