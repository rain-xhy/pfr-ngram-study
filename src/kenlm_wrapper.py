"""KenLM 训练、转换、评估的封装。

lmplz 只实现插值 Modified Kneser-Ney，没有 MLE、Witten-Bell 的开关（`lmplz --help` 的第一行是
「Builds unpruned language models with modified Kneser-Ney smoothing」）。阶数、剪枝、内存预算、
规模、存储格式这些对比都用 lmplz 训练；平滑方法对比（E1）里的其他平滑由 smoothing_lab.py 计算。

训练时逐行读 lmplz 的 stderr，记下每个阶段标题（`=== 1/5 Counting and sorting n-grams ===` 等）
出现的时刻，得到五个阶段各自的耗时。

峰值内存经 /usr/bin/time 取子进程自己的 ru_maxrss。直接从 Python 启动 lmplz 时，fork 出的子进程
继承父进程的内存高水位：scripts/check_rss.sh 让 Python 进程常驻 616 MiB 后直接启动三元训练，
KenLM 报出 616.0 MiB；同一训练经 time 启动时是 155.8 MiB（父进程 116 MiB 时两种方式都约 157 MiB）。
第一次完整运行没有经 time，E1 之后的峰值都停在 527.32 MiB，就是这个原因，那一版结果已作废重跑。
KenLM 自己打印的 RSSMax 也一并记下（kenlm_reported_rssmax_mb），经 time 启动时两者一致。
"""
import math
import re
import subprocess
import time
from pathlib import Path

from .common import KENLM_BIN

RESOURCE_RE = re.compile(r"Name:(\w+).*?RSSMax:(\d+) kB.*?user:([\d.]+).*?sys:([\d.]+).*?real:([\d.]+)")
# 第二列是该阶条目数；剪枝后这一列不是单纯的整数，所以按非空白串匹配
DISCOUNT_RE = re.compile(r"^(\d+) (\S+) D1=([\d.]+) D2=([\d.]+) D3\+=([\d.]+)\s*$", re.M)
TIME_BIN = "/usr/bin/time"
TIME_FORMAT = "PFR_TIME maxrss_kb=%M elapsed=%e user=%U sys=%S"
# 不要求行首：lmplz 最后一行进度条没有换行时，time 的统计会接在它后面
TIME_RE = re.compile(r"PFR_TIME maxrss_kb=(\d+) elapsed=([\d.]+) user=([\d.]+) sys=([\d.]+)")
STAGE_RE = re.compile(r"^=== (\d)/(\d) (.+?) ===")
NGRAM_COUNT_RE = re.compile(r"^ngram (\d+)=(\d+)$", re.M)


def _parse_resource(text: str) -> dict:
    """KenLM 程序结束时自己打印的资源行。其中 RSSMax 受父进程影响，只作核对用。"""
    found = list(RESOURCE_RE.finditer(text))
    if not found:
        return {}
    m = found[-1]
    return {"kenlm_reported_rssmax_mb": int(m.group(2)) / 1024, "kenlm_real_seconds": float(m.group(5))}


def _parse_time(text: str) -> dict:
    """/usr/bin/time 按 TIME_FORMAT 打印的一行：子进程自己的峰值内存与 CPU 时间。"""
    found = TIME_RE.findall(text)
    if not found:
        return {}
    kb, elapsed, user, sys_ = found[-1]
    return {"peak_rss_mb": int(kb) / 1024, "time_elapsed_seconds": float(elapsed),
            "user_seconds": float(user), "sys_seconds": float(sys_)}


def parse_discounts(text: str) -> list:
    return [{"order": int(o), "count_field": n, "D1": float(a), "D2": float(b), "D3+": float(c)}
            for o, n, a, b, c in DISCOUNT_RE.findall(text)]


def arpa_counts(arpa_path) -> list:
    """读 ARPA 文件头的各阶条目数。"""
    head = []
    with open(arpa_path, encoding="utf-8") as f:
        for line in f:
            head.append(line)
            if line.startswith("\\1-grams:"):
                break
    return [int(n) for _, n in NGRAM_COUNT_RE.findall("".join(head))]


def train(corpus_path, arpa_path, order: int, memory: str = "512M", prune=None,
          temp_dir=None, timeout: int = 1800) -> dict:
    """调用 lmplz 训练一个插值 Modified Kneser-Ney 模型。

    prune 按阶给阈值，长度与 order 相同，如五元写 [0, 0, 1, 1, 1]：计数不超过阈值的条目被删。
    lmplz 无法估计折扣时会报错退出，这里不加 --discount_fallback，出错就让实验停下，
    正式语料上五个阶数的折扣都能正常估计（见返回值里的 discounts）。
    """
    arpa_path = Path(arpa_path)
    arpa_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(KENLM_BIN / "lmplz"), "-o", str(order), "-S", memory,
           "-T", str(Path(temp_dir or arpa_path.parent)), "--arpa", str(arpa_path)]
    if prune:
        if len(prune) != order:
            raise ValueError(f"prune 长度 {len(prune)} 与阶数 {order} 不一致")
        cmd += ["--prune"] + [str(x) for x in prune]
    marks, lines = [], []
    started = time.perf_counter()
    with open(corpus_path, "rb") as stdin:
        proc = subprocess.Popen([TIME_BIN, "-f", TIME_FORMAT] + cmd, stdin=stdin,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for raw in proc.stderr:
            line = raw.decode("utf-8", errors="replace")
            lines.append(line)
            m = STAGE_RE.match(line)
            if m:
                marks.append((int(m.group(1)), m.group(3), time.perf_counter() - started))
        proc.wait(timeout=timeout)
    wall = time.perf_counter() - started
    stderr = "".join(lines)
    if proc.returncode != 0:
        raise RuntimeError(f"lmplz 退出码 {proc.returncode}：{stderr[-800:]}")
    stages = []
    for i, (idx, name, t) in enumerate(marks):
        end = marks[i + 1][2] if i + 1 < len(marks) else wall
        stages.append({"stage": idx, "name": name, "seconds": end - t})
    timing = _parse_time(stderr)
    return {
        "command": cmd, "wall_seconds": wall, **_parse_resource(stderr), **timing,
        # /usr/bin/time 的统计行没解析到时，CPU 时间与峰值内存都缺失，这次测量不能用
        "time_parse_failed": not timing,
        "stderr_tail": stderr[-600:] if not timing else None,
        "startup_seconds": marks[0][2] if marks else None, "stages": stages,
        "arpa_path": str(arpa_path), "arpa_bytes": arpa_path.stat().st_size,
        "ngram_counts": arpa_counts(arpa_path), "discounts": parse_discounts(stderr),
        "fallback_discount_used": "Substituting fallback" in stderr,
    }


def build_binary(arpa_path, out_path, structure: str = "probing", quantize_bits=None,
                 timeout: int = 1800) -> dict:
    """ARPA 转成 KenLM 二进制：probing（哈希）或 trie（前缀树），trie 可加 -q/-b 量化。

    build_binary 的排序内存默认是物理内存的 80%，这里固定为 1G（与 LynxPeng 的做法相同）。
    """
    cmd = [str(KENLM_BIN / "build_binary"), "-S", "1G"]
    if quantize_bits:
        cmd += ["-q", str(quantize_bits), "-b", str(quantize_bits)]
    cmd += [structure, str(arpa_path), str(out_path)]
    started = time.perf_counter()
    proc = subprocess.run([TIME_BIN, "-f", TIME_FORMAT] + cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout)
    wall = time.perf_counter() - started
    text = proc.stderr.decode("utf-8", errors="replace") + proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0 or "SUCCESS" not in text:
        raise RuntimeError(f"build_binary 失败：{text[-800:]}")
    return {"command": cmd, "wall_seconds": wall, **_parse_resource(text), **_parse_time(text),
            "path": str(out_path), "bytes": Path(out_path).stat().st_size}


def load(model_path):
    import kenlm  # 只在 WSL 的 venv 里有，用到时再导入
    return kenlm.Model(str(model_path))


def position_scores(model, sentences) -> list:
    """逐位置返回 (log10 概率, 匹配到的 n-gram 阶数, 是否 OOV)，每句末尾含一次 </s> 预测。"""
    out = []
    for sent in sentences:
        out.extend(model.full_scores(" ".join(sent), bos=True, eos=True))
    return out


def summarize_scores(scores, sentence_count: int) -> dict:
    """由逐位置得分算困惑度与匹配阶数分布。

    分母是预测次数，等于词数 + 句数（每句末尾的 </s> 也是一次预测，句首 <s> 只作上下文）。
    ppl_including_oov 把未登录词的 <unk> 概率算进去；ppl_excluding_oov 去掉这些位置，
    与 KenLM query 的同名输出口径一致。KenLM 对未登录词返回匹配长度 1，这里把它们单独计数。
    """
    total = oov_total = 0.0
    oov = 0
    hist = {}
    for log10_p, ngram_len, is_oov in scores:
        total += log10_p
        if is_oov:
            oov += 1
            oov_total += log10_p
        else:
            hist[ngram_len] = hist.get(ngram_len, 0) + 1
    events = len(scores)
    in_vocab = events - oov
    return {
        "events": events, "sentences": sentence_count, "oov_events": oov,
        "oov_rate": oov / events if events else 0.0, "log10_prob": total,
        "cross_entropy_bits": -total / events / math.log10(2),
        "ppl_including_oov": 10 ** (-total / events),
        "ppl_excluding_oov": 10 ** (-(total - oov_total) / in_vocab) if in_vocab else float("nan"),
        "matched_orders": {str(k): v for k, v in sorted(hist.items())},
    }


def evaluate(model, sentences) -> dict:
    return summarize_scores(position_scores(model, sentences), len(sentences))
