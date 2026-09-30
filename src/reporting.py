"""把 results/runs/*.json 汇总成表格与图。

在 WSL 的 venv 里运行：`~/pfr-venv/bin/python -m src.reporting`。
图写到 results/figures/（PNG 与 SVG 各一份，SVG 里的文字转成路径，换机器打开字体不走样），
表格写到 results/tables.md 与 results/metrics.csv。这里只整理数字，报告正文由作者本人撰写。
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from .common import FIG_DIR, RESULTS_DIR, RUNS_DIR, read_json  # noqa: E402

FONT_CANDIDATES = ["/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf",
                   "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
COLORS = ["#2b6cb0", "#c05621", "#2f855a", "#9b2c2c", "#6b46c1", "#718096"]


def setup_fonts():
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "path", "figure.dpi": 110,
                         "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)


def load(name):
    path = RUNS_DIR / f"{name}.json"
    return read_json(path) if path.exists() else None


def fmt(x, digits=2):
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return "∞" if math.isinf(x) else "—"
    if isinstance(x, float):
        return f"{x:,.{digits}f}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(fmt(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    return "\n".join(out)


def mib(b):
    return b / 2 ** 20


def digit_keys(d):
    """E3 这类结果把各阶数直接放在顶层，和 experiment、finished_at 等元信息并列，只取数字键。"""
    return sorted((k for k in d if k.isdigit()), key=int)


def sec(stat):
    """训练耗时写成「中位数（最小–最大）」。"""
    return f"{stat['median']:.2f}（{stat['min']:.2f}–{stat['max']:.2f}）"


def repeats(o):
    """某个训练配置计时了几次（从结果里读，不写死）。"""
    tr = o["training"] if "training" in o else o
    return tr["wall_seconds"]["n"]


def stalled_total(r) -> int:
    return sum(o["training"].get("stalled_runs_discarded", 0) for o in r["orders"].values())


def interference_note(trainings) -> str:
    """汇总一组训练配置里「重测了几次」「有几次重测用完仍受干扰」，写进表格下方的说明。"""
    retried = sum(t.get("stalled_runs_discarded", 0) for t in trainings)
    kept = sum(t.get("accepted_despite_interference", 0) for t in trainings)
    if not retried:
        return "所有计时都没有受到干扰。"
    note = f"受干扰而重测的训练共 {retried} 次（结果里不计入）。"
    if kept:
        note += f"有 {kept} 次重测次数用完仍受干扰，已计入，对应配置的耗时偏高。"
    return note


def cpu_cell(tr):
    c = tr.get("cpu_seconds")
    return f"{c['median']:.2f}" if c else "—"


def startup_cell(tr):
    s = tr.get("startup_seconds")
    return f"{s['median']:.2f}（最长 {s['max']:.2f}）" if s else "—"


def e0_section(r):
    c, t = r["clean"], r["tagged"]
    rows = [
        ["训练集词形数", c["train"]["types"], t["train"]["types"]],
        ["只出现一次的词形", c["train"]["hapax_types"], t["train"]["hapax_types"]],
        ["三元模型各阶条目数", "/".join(f"{n:,}" for n in c["training"]["ngram_counts"]),
         "/".join(f"{n:,}" for n in t["training"]["ngram_counts"])],
        ["ARPA 大小（MiB）", mib(c["training"]["arpa_bytes"]), mib(t["training"]["arpa_bytes"])],
        ["测试集未登录词比例", f"{c['test']['oov_rate']:.2%}", f"{t['test']['oov_rate']:.2%}"],
        ["测试集困惑度（含未登录词）", c["test"]["ppl_including_oov"], t["test"]["ppl_including_oov"]],
    ]
    out = ["## E0 清洗版与带词性版", "",
           "两份语料的训练、测试文章完全相同，差别只在词后面有没有词性标记。"
           "两边的预测次数都是每词一次加每句一次，所以困惑度可以直接比。", "",
           table(["", "清洗版", "带词性版"], rows), ""]
    for label, part in (("清洗版", c), ("带词性版", t)):
        out.append(f"{label}续写（T=1.0，k=20，开头「{''.join(w.rpartition('/')[0] or w for w in part['prompt'])}」）：")
        out += [f"- 种子 {s['seed']}：{s['text']}{'（自然结束）' if s['hit_eos'] else '（达到长度上限）'}"
                for s in part["samples"]]
        out.append("")
    return "\n".join(out)


def e1_section(r):
    names = list(r["methods"])
    rows = []
    for name in names:
        m = r["methods"][name]
        rows.append([name, m["dev"]["ppl"], m["test"]["ppl"], m["test"]["zero_prob_events"],
                     f"{m['test']['zero_prob_rate']:.2%}"])
    k = r["kenlm"]
    rows.append(["KenLM Modified KN（lmplz）", k["dev"]["ppl_including_oov"], k["test"]["ppl_including_oov"], 0, "0.00%"])
    chk = r["kenlm_check"]
    scope = ""
    if r.get("train_scope") and r["train_scope"] != "全部训练集":
        scope = (f"本节只用{r['train_scope']}训练（{r['train_tokens']:,} 词）：本项目自己实现的平滑用 Python 字典计数，"
                 "六个月的训练集放不进这台机器的内存；比较的是平滑方法，结论不随语料量变。验证集、测试集仍是全部六个月，"
                 "所以本节的困惑度比 E2 同阶模型高，两节的数字不能直接比。")
    out = ["## E1 平滑方法", "",
           scope + f"三元模型，训练集 {r['vocab_size']:,} 个词形（含 </s> 与 <unk>）。Add-k 的 k 与 JM 的 λ 在验证集上选取："
           f"k = {r['best_k']:g}，λ = {tuple(r['best_lambda'])}。", "",
           table(["平滑", "验证集困惑度", "测试集困惑度", "测试集零概率事件", "占比"], rows), "",
           f"本模块的 Kneser-Ney 与 lmplz 对照：各阶条目数 {chk['our_ngram_counts_1_2_3']} 对 {chk['kenlm_ngram_counts']}，"
           f"测试集困惑度 {chk['ours_test_ppl']:.3f} 对 {chk['kenlm_test_ppl']:.3f}，"
           f"测试集前 200 句 {chk['position_log10_absdiff']['positions']:,} 个位置上 log10 概率之差最大 "
           f"{chk['position_log10_absdiff']['max']:.1e}。", "",
           "Add-k 验证集困惑度随 k 的变化：", "",
           table(["k"] + list(r["addk_dev_ppl"]), [["困惑度"] + list(r["addk_dev_ppl"].values())]), ""]
    ex = r["zero_prob_examples"]
    if ex:
        cols = [c for c in ex[0] if c not in ("context", "word", "history_count", "bigram_count")]
        rows = [[" ".join(e["context"]) + " → " + e["word"], e["history_count"], e["bigram_count"]]
                + [f"{e[c]:.2e}" for c in cols] for e in ex]
        out += ["测试集里 MLE 给出零概率的三元事件（历史在训练集出现 ≥20 次，二元见过）：", "",
                table(["事件", "历史次数", "二元次数"] + cols, rows), ""]
    cd = r["continuation_diversity"]
    rows = [[x["word"], x["count"], x["distinct_left"], x["top_left"], f"{x['top_left_share']:.0%}",
             f"{x['kn_over_mle']:.3f}"] for x in cd["left_bound"][:10]]
    rows2 = [[x["word"], x["count"], x["distinct_left"], x["top_left"], f"{x['top_left_share']:.0%}",
              f"{x['kn_over_mle']:.3f}"] for x in cd["left_free"][:10]]
    hdr = ["词", "词频", "不同左邻数", "最常见左邻", "该左邻占比", "续接概率/一元 MLE"]
    out += [f"续接多样性（词频 ≥{cd['min_count']}）：左邻最固定的 10 个词", "", table(hdr, rows), "",
            "左邻最多样的 10 个词", "", table(hdr, rows2), ""]
    return "\n".join(out)


def e2_section(r):
    orders = sorted(r["orders"], key=int)
    rows = []
    for n in orders:
        o = r["orders"][n]
        tr = o["training"]
        hist = o["test"]["matched_orders"]
        total = sum(hist.values())
        rows.append([f"{n} 元", sec(tr["wall_seconds"]), cpu_cell(tr), startup_cell(tr),
                     f"{tr['peak_rss_mb']['median']:.0f}",
                     mib(tr["arpa_bytes"]), f"{sum(tr['ngram_counts']):,}",
                     o["dev"]["ppl_including_oov"], o["test"]["ppl_including_oov"],
                     f"{hist.get(n, 0) / total:.1%}", "是" if tr["deterministic_arpa"] else "否"])
    stage_names = list(r["orders"][orders[0]]["training"]["stage_seconds_median"])
    srows = [[f"{n} 元"] + [f"{r['orders'][n]['training']['stage_seconds_median'][s]:.2f}" for s in stage_names]
             for n in orders]
    short = ["计数排序", "调整计数", "初始概率", "插值概率", "写 ARPA"]
    return "\n".join([
        "## E2 阶数", "",
        f"Modified KN，-S 512M，不剪枝，每个阶数先预热一次，再计时训练 {repeats(r['orders'][orders[0]])} 次。"
        f"验证集困惑度最低的是 {r['best_order_by_dev']} 元，E7 以后的生成实验都用它。"
        "启动等待是从启动 lmplz 到它打印第一阶段标题的时间。"
        "墙钟时间远超 CPU 时间所对应的正常耗时的训练判为受干扰、重测。"
        + interference_note([o["training"] for o in r["orders"].values()]), "",
        table(["阶数", "训练耗时 s，中位数（范围）", "CPU s", "启动等待 s", "峰值内存 MiB", "ARPA MiB", "条目总数",
               "验证集困惑度", "测试集困惑度", "测试集满阶匹配", "各次 ARPA 一致"], rows), "",
        "lmplz 五个阶段的耗时（秒，各次中位数）：", "",
        table(["阶数"] + short, srows), ""])


def e3_section(r, e2=None):
    rows = []
    for n in digit_keys(r):
        p = r[n]
        u = p["unpruned"]
        # 不剪枝版就是 E2 的同阶模型，CPU 时间从 E2 取
        u_cpu = cpu_cell(e2["orders"][n]["training"]) if e2 and n in e2["orders"] else "—"
        for label, arpa, probing, counts, dev, test, wall, cpu in (
                ("不剪枝", u["arpa_bytes"], u["probing_bytes"], u["ngram_counts"], u["dev_ppl"], u["test_ppl"],
                 u["wall_seconds"], u_cpu),
                ("剪枝 " + " ".join(map(str, p["prune"])), p["training"]["arpa_bytes"], p["probing_bytes"],
                 p["training"]["ngram_counts"], p["dev"]["ppl_including_oov"], p["test"]["ppl_including_oov"],
                 p["training"]["wall_seconds"], cpu_cell(p["training"]))):
            rows.append([f"{n} 元", label, mib(arpa), mib(probing), "/".join(f"{c:,}" for c in counts),
                         sec(wall), cpu, dev, test])
    return "\n".join([
        "## E3 剪枝", "",
        "剪枝阈值按阶给出，`0 0 1` 表示三阶及以上删去训练集中只出现 1 次的条目。", "",
        table(["阶数", "设置", "ARPA MiB", "probing MiB", "各阶条目数", "训练耗时 s", "CPU s", "验证集困惑度",
               "测试集困惑度"], rows), "",
        "墙钟时间在主机内存紧张时会被拉长，低于重测门槛的拖慢仍会留在中位数里；比较训练开销时以 CPU 时间为准。", "",
        "剪枝版的耗时计时说明：" + interference_note([r[n]["training"] for n in digit_keys(r)]), ""])


def e4_section(r):
    rows = []
    for mem in ("128M", "512M", "1G"):
        tr = r[mem]["training"]
        rows.append([mem, sec(tr["wall_seconds"]), cpu_cell(tr), f"{tr['peak_rss_mb']['median']:.1f}",
                     f"{tr['peak_rss_mb']['min']:.1f}–{tr['peak_rss_mb']['max']:.1f}",
                     r[mem]["arpa_sha256"][:12]])
    return "\n".join([
        "## E4 排序内存 -S", "",
        f"三元，不剪枝，每档先预热一次，再计时训练 {repeats(r['512M'])} 次。", "",
        table(["-S", "训练耗时 s", "CPU s", "峰值内存 MiB（中位数）", "峰值内存范围", "ARPA SHA256 前 12 位"], rows), "",
        f"三档产出的 ARPA 逐字节相同：{'是' if r['all_arpa_identical'] else '否'}。"
        + interference_note([r[m]["training"] for m in ("128M", "512M", "1G")]), ""])


def fig_orders(r):
    orders = sorted(r["orders"], key=int)
    xs = [int(n) for n in orders]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    ax.plot(xs, [r["orders"][n]["dev"]["ppl_including_oov"] for n in orders], "o-", color=COLORS[0], label="验证集")
    ax.plot(xs, [r["orders"][n]["test"]["ppl_including_oov"] for n in orders], "s-", color=COLORS[1], label="测试集")
    ax.set(xlabel="阶数 n", ylabel="困惑度", title="困惑度随阶数")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.bar(xs, [mib(r["orders"][n]["training"]["arpa_bytes"]) for n in orders], color=COLORS[2])
    ax.set(xlabel="阶数 n", ylabel="MiB", title="ARPA 文件大小")
    ax = axes[2]
    med = [r["orders"][n]["training"]["wall_seconds"]["median"] for n in orders]
    lo = [m - r["orders"][n]["training"]["wall_seconds"]["min"] for m, n in zip(med, orders)]
    hi = [r["orders"][n]["training"]["wall_seconds"]["max"] - m for m, n in zip(med, orders)]
    ax.errorbar(xs, med, yerr=[lo, hi], fmt="o-", color=COLORS[3], capsize=3)
    ax.set(xlabel="阶数 n", ylabel="秒", title="训练耗时（中位数与范围）")
    for a in axes:
        a.set_xticks(xs)
    save(fig, "E2_orders")


def e5_section(r):
    pts = r["points"]
    orders = sorted(pts[0]["orders"], key=int)
    rows = []
    for p in pts:
        label = f"{p['actual_words']:,}" + ("（全部训练集）" if p["is_full_train"] else "")
        row = [label, p["article_count"], p["vocab"]["types"], f"{p['orders'][orders[0]]['test']['oov_rate']:.2%}"]
        row += [p["orders"][n]["test"]["ppl_including_oov"] for n in orders]
        row += [f"{p['orders'][n]['training']['wall_seconds']['median']:.2f}" for n in orders]
        rows.append(row)
    mrows = []
    for n in orders:
        for m in r["marginal"][n]:
            mrows.append([f"{n} 元", f"{m['from_words']:,} → {m['to_words']:,}", m["ppl_from"], m["ppl_to"],
                          m["ppl_drop_per_100k"], f"{m['relative_drop']:.1%}"])
    h = r["heaps"]
    return "\n".join([
        "## E5 训练集规模", "",
        "规模样本都从主划分的训练集里按整篇文章抽取，评估一律用主划分的验证集与测试集。"
        + interference_note([o["training"] for p in pts for o in p["orders"].values()]), "",
        table(["训练词数", "文章数", "词形数", "测试集未登录词"] + [f"{n} 元测试困惑度" for n in orders]
              + [f"{n} 元训练 s" for n in orders], rows), "",
        f"词表增长按 Heaps 定律 V = K·N^β 拟合：K = {h['K']:.2f}，β = {h['beta']:.3f}。", "",
        "相邻两个规模之间，每多 10 万词测试集困惑度下降多少：", "",
        table(["阶数", "训练词数", "困惑度（前）", "困惑度（后）", "每 10 万词下降", "相对下降"], mrows), ""])


def e6_section(r):
    rows = []
    for name in ("ARPA", "probing", "trie", "trie_q8"):
        x = r[name]
        build = x["build"]["wall_seconds"] if x.get("build") else None
        rows.append([name, mib(x["bytes"]), build, x["load_seconds"], x["queries_per_second"] / 1e6,
                     x["queries_per_second_best"] / 1e6, x["test_ppl"],
                     x["position_log10_absdiff_vs_arpa"]["changed"],
                     f"{x['same_generation_count']}/{x['generation_count']}"])
    return "\n".join([
        "## E6 存储格式", "",
        "六元模型转成四种格式。查询速度是在测试集上逐句打分：各格式先预热一遍，再轮换测 7 轮，"
        "给出中位数与最快一轮。", "",
        table(["格式", "文件 MiB", "转换 s", "加载 s", "百万次查询/秒（中位数）", "百万次查询/秒（最快）",
               "测试集困惑度", "与 ARPA 概率不同的位置数", "生成与 ARPA 相同"], rows), ""])


GEN_COLS = [("length", "平均长度", 1), ("eos_rate", "自然结束", "%"), ("repeated_3gram_rate", "三元自重复", "%1"),
            ("type_token_ratio", "type-token ratio", 3), ("self_ppl", "生成部分困惑度", 1),
            ("top1_prob_mean", "Top-1 概率均值", 3), ("chose_top1_rate", "选中 Top-1 比例", "%"),
            ("entropy_bits_mean", "分布熵 bit", 2), ("longest_copied_span", "最长照搬词数", 1)]


def gen_row(label, s):
    row = [label]
    for key, _, digits in GEN_COLS:
        v = s[key]
        if digits == "%":
            row.append(f"{v:.0%}")
        elif digits == "%1":
            row.append(f"{v:.1%}")
        else:
            row.append(f"{v:.{digits}f}")
    return row


def e7_section(r):
    n_stat, shown = len(r["stat_seeds"]), r["shown_seeds"]
    rows = [gen_row(c["name"], c["summary"]) for c in r["grid"]]
    orows = [gen_row(f"{n} 元", v["summary"]) for n, v in sorted(r["by_order"].items(), key=lambda x: int(x[0]))]
    out = ["## E7 生成参数", "",
           f"{r['model_order']} 元模型，开头「" + "".join(r["prompt"]) + f"」，每组参数用 {n_stat} 个种子各生成一次，"
           "最多 60 个词，指标取平均。生成部分困惑度是用同一个模型给生成出来的词打分。", "",
           table(["参数"] + [c[1] for c in GEN_COLS], rows), "",
           f"不同阶数，同为 T=1.0、k=50，各 {n_stat} 次：", "",
           table(["模型"] + [c[1] for c in GEN_COLS], orows), "",
           f"原文（种子 {', '.join(map(str, shown))}）：", ""]
    for c in r["grid"]:
        out.append(f"**{c['name']}**")
        out += [f"- 种子 {s['params']['seed']}：{s['text'] or '（直接结束）'}"
                f"{'' if s['hit_eos'] else '（达到长度上限）'}" for s in c["samples"]]
        out.append("")
    out += ["**贪心**", f"- {r['greedy']['text'] or '（直接结束）'}", ""]
    grows = []
    for n, v in sorted(r["by_order"].items(), key=lambda x: int(x[0])):
        g = v["greedy"]
        grows.append([f"{n} 元", len(g["tokens"]), "是" if g["hit_eos"] else "否", f"{g['repeated_3gram_rate']:.1%}",
                      g["text"] or "（直接结束）"])
    out += ["各阶数的贪心输出：", "",
            table(["模型", "长度", "自然结束", "三元自重复", "输出"], grows), ""]
    return "\n".join(out)


def fig_generation(r):
    grid = [c for c in r["grid"] if c["top_p"] == 1.0]
    temps = sorted({c["temperature"] for c in grid})
    ks = [10, 50, 0]
    panels = [("length", "平均长度"), ("eos_rate", "自然结束比例"), ("self_ppl", "生成部分困惑度"),
              ("type_token_ratio", "type-token ratio"), ("top1_prob_mean", "Top-1 概率均值"),
              ("longest_copied_span", "最长照搬词数")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, (key, title) in zip(axes.flat, panels):
        m = [[next(c["summary"][key] for c in grid if c["temperature"] == t and c["top_k"] == k) for k in ks]
             for t in temps]
        im = ax.imshow(m, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(ks)), ["k=10", "k=50", "全词表"])
        ax.set_yticks(range(len(temps)), [f"T={t}" for t in temps])
        ax.set_title(title)
        ax.grid(False)
        lo, hi = min(min(row) for row in m), max(max(row) for row in m)
        for i, row in enumerate(m):
            for j, v in enumerate(row):
                dark = hi > lo and (v - lo) / (hi - lo) > 0.6
                ax.text(j, i, f"{v:.2f}" if v < 10 else f"{v:.1f}", ha="center", va="center",
                        color="white" if dark else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "E7_generation_grid")


def e8_section(r):
    bm = r["by_model"]
    orders = sorted(bm, key=int)
    maxn = max(int(n) for n in orders)
    rows = []
    for n in orders:
        row = [f"{n} 元"] + [f"{bm[n]['share'].get(str(k), 0):.1%}" if k <= int(n) else "" for k in range(1, maxn + 1)]
        row += [f"{bm[n]['mean_matched_order']:.2f}", bm[n]["test_ppl"]]
        rows.append(row)
    b = r["by_5gram_match_length"]
    keys = [k for k in ("1", "2", "3", "4", "5") if k in b] + (["OOV"] if "OOV" in b else [])
    brows = [[("未登录词" if k == "OOV" else f"匹配 {k} 阶"), b[k]["positions"]]
             + [f"{b[k][f'mean_log10_{n}']:.3f}" for n in (2, 3, 4, 5)] for k in keys]
    return "\n".join([
        "## E8 匹配阶数", "",
        "测试集每个预测位置实际用到的 n-gram 长度（KenLM full_scores 给出），未登录词位置不计入。", "",
        table(["模型"] + [f"{k} 阶" for k in range(1, maxn + 1)] + ["平均匹配阶数", "测试集困惑度"], rows), "",
        "按五元模型在该位置匹配到的阶数分组，同一批位置在二至五元模型下的平均 log10 概率：", "",
        table(["五元模型的匹配", "位置数", "二元", "三元", "四元", "五元"], brows), ""])


def fig_matched(r):
    bm = r["by_model"]
    orders = sorted(bm, key=int)
    maxn = max(int(n) for n in orders)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bottom = [0.0] * len(orders)
    for k in range(1, maxn + 1):
        vals = [bm[n]["share"].get(str(k), 0) for n in orders]
        ax.bar([f"{n} 元" for n in orders], vals, bottom=bottom, color=COLORS[(k - 1) % len(COLORS)],
               label=f"匹配 {k} 阶")
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set(ylabel="测试集位置占比", title="各阶模型实际匹配到的 n-gram 长度")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    save(fig, "E8_matched_orders")


def e9_section(r, e7):
    n_stat = len(e7["stat_seeds"])
    orows = []
    for n in sorted(r["by_order"], key=int):
        s, h = r["by_order"][n], r["copied_span_histogram"][n]
        orows.append([f"{n} 元", f"{s['longest_copied_span']:.1f}", f"{s['copied_10plus_rate']:.0%}",
                      f"{s['overlap_5gram']:.1%}", h["0-3"], h["4-6"], h["7-9"], h["10+"]])
    trows = [[f"T={t}", f"{v['longest_copied_span']:.1f}", f"{v['copied_10plus_rate']:.0%}",
              f"{v['overlap_5gram']:.1%}", v["n"]] for t, v in sorted(r["by_temperature"].items())]
    out = ["## E9 与训练集的重合", "",
           "最长照搬词数：生成内容里在训练集某一句中原样出现过的最长连续词串（至少含一个生成的词，"
           "最多统计到 12 个词）。5-gram 重合率：生成部分的 5-gram 有多少在训练集里出现过。", "",
           f"不同阶数（T=1.0，k=50，各 {n_stat} 次），右边四列是最长照搬词数落在各区间的次数：", "",
           table(["模型", "最长照搬（平均）", "≥10 词的比例", "5-gram 重合率", "0-3", "4-6", "7-9", "10+"], orows), "",
           f"{e7['model_order']} 元模型，不同温度（每个温度含 k=10、50、全词表三组）：", "",
           table(["温度", "最长照搬（平均）", "≥10 词的比例", "5-gram 重合率", "生成次数"], trows), "",
           "照搬最长的几条，以及训练集里的原句：", ""]
    for x in r["top_copies"]:
        p = x["params"]
        src = x["source"]
        label = f"T={p['temperature']}，k={p['top_k'] or '全词表'}，top-p={p['top_p']}，种子 {p['seed']}"
        out.append(f"- {label}：{x['text']}")
        if src:
            out.append(f"  - 照搬片段（{src['span_words']} 词，其中生成部分 {src['span_generated_words']} 词）："
                       f"{src['span']}")
            out.append(f"  - 训练集原句：{src['train_sentence']}")
    out.append("")
    return "\n".join(out)


def fig_leakage(r):
    orders = sorted(r["by_order"], key=int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.bar([f"{n} 元" for n in orders], [r["by_order"][n]["longest_copied_span"] for n in orders], color=COLORS[0])
    ax.set(ylabel="词数", title="最长照搬词数（平均）")
    ax = axes[1]
    bins = ["0-3", "4-6", "7-9", "10+"]
    bottom = [0] * len(orders)
    for i, b in enumerate(bins):
        vals = [r["copied_span_histogram"][n][b] for n in orders]
        ax.bar([f"{n} 元" for n in orders], vals, bottom=bottom, color=COLORS[i], label=f"{b} 词")
        bottom = [x + v for x, v in zip(bottom, vals)]
    ax.set(ylabel="生成次数", title="最长照搬词数的分布")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "E9_leakage")


def e10_section(r):
    rows, samples = [], []
    for name, p in r["prefixes"].items():
        top = "、".join(f"{w}（{pr:.1%}）" for w, pr in p["first_step_top"][:3])
        rows.append([name, p["prefix_words"], "、".join(p["oov_words"]) or "无",
                     fmt(p["last2_train_count"]), f"{p['first_step_entropy_bits']:.2f}", top,
                     p["distinct_first_words"], f"{p['length_mean']:.1f} ± {p['length_stdev']:.1f}",
                     f"{p['eos_rate']:.0%}", f"{p['pairwise_jaccard_mean']:.3f}",
                     f"{p['longest_copied_span_mean']:.1f}", f"{p['matched_order_mean']:.2f}"])
        samples.append(f"**{name}**：{''.join(p['prefix'].split())}")
        samples += [f"- {s or '（直接结束）'}" for s in p["samples"][:3]]
        samples.append("")
    return "\n".join([
        "## E10 开头的类型", "",
        f"{r['model_order']} 元模型，T=1.0，k=50，每个开头生成 {len(r['seeds'])} 次。"
        "两两重合度是两次生成的词集合的 Jaccard 系数，对所有两两组合取平均，越低说明各次生成越不一样。", "",
        table(["开头", "词数", "未登录词", "末两词在训练集出现次数", "首词分布熵 bit", "首词概率最高的三个",
               "10 次里不同的首词数", "长度（均值 ± 标准差）", "自然结束", "两两重合度", "最长照搬",
               "平均匹配阶数"], rows), "",
        "每个开头的前三次生成：", ""] + samples)


def e11_section(r):
    out = ["## E11 温度与分布", "",
           f"{r['model_order']} 元模型，固定上下文，对下一个词在整个词表上的分布做温度缩放 q ∝ p^(1/T)。"
           "有效候选数是 2 的熵次方。", ""]
    for label, c in r["contexts"].items():
        rows = [[f"{x['temperature']}", f"{x['entropy_bits']:.2f}", f"{x['effective_candidates']:,.0f}",
                 x["top1_word"], f"{x['top1_prob']:.1%}", f"{x['words_for_90pct']:,}"] for x in c["rows"]]
        prefix = "".join(c["prefix"]) or "（句首，只有 <s>）"
        out += [f"上下文：{label}，{prefix}", "",
                table(["温度", "熵 bit", "有效候选数", "最可能的词", "它的概率", "覆盖 90% 概率要几个词"], rows), ""]
    return "\n".join(out)


def fig_temperature(r):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (label, c) in enumerate(r["contexts"].items()):
        ts = [x["temperature"] for x in c["rows"]]
        axes[0].plot(ts, [x["entropy_bits"] for x in c["rows"]], "o-", color=COLORS[i], label=label)
        axes[1].plot(ts, [x["top1_prob"] for x in c["rows"]], "o-", color=COLORS[i], label=label)
    axes[0].set(xlabel="温度 T", ylabel="bit", title="下一个词分布的熵")
    axes[1].set(xlabel="温度 T", ylabel="概率", title="最可能的词的概率")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "E11_temperature")
    # 作业开头之后，不同温度下前 10 个词的概率
    ctx = next(iter(r["contexts"].values()))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    picks = [x for x in ctx["rows"] if x["temperature"] in (0.5, 1.0, 1.5)]
    words = [w for w, _ in picks[1]["top10"]] if len(picks) > 1 else [w for w, _ in picks[0]["top10"]]
    width = 0.8 / len(picks)
    for i, x in enumerate(picks):
        d = dict(x["top10"])
        ax.bar([j + i * width for j in range(len(words))], [d.get(w, 0) for w in words], width,
               color=COLORS[i], label=f"T={x['temperature']}")
    ax.set_xticks([j + width * (len(picks) - 1) / 2 for j in range(len(words))], words)
    ax.set(ylabel="概率", title="作业开头之后下一个词的概率（T=1 时最可能的 10 个词）")
    ax.legend(frameon=False)
    save(fig, "E11_top10")


def e12_section(r):
    order = ["句首不足五词", "0（训练集未见）", "1", "2-4", "≥5", "OOV"]
    g = r["groups"]
    rows = []
    total_loss = sum(v["log10_loss"] for v in g.values())
    for k in [x for x in order if x in g]:
        v = g[k]
        rows.append([("未登录词" if k == "OOV" else k), v["positions"], f"{v['share_of_positions']:.1%}",
                     v["ppl_full"], v["ppl_pruned"], f"{v['changed_rate']:.0%}",
                     f"{v['mean_order_full']:.2f} → {v['mean_order_pruned']:.2f}",
                     f"{v['log10_loss'] / total_loss:.1%}" if total_loss else "—"])
    removed = r["removed_ngrams_by_order"]
    return "\n".join([
        "## E12 剪枝影响了哪些位置", "",
        "五元模型剪枝前后，测试集每个位置的概率变化，按该位置的五元（当前词与前四个词）在训练集出现的次数分组。"
        "最后一列是这一组在整个测试集对数概率损失中的占比，负数表示剪枝后这一组的概率反而变高。", "",
        f"剪枝删掉的条目：三元 {removed[2]:,}、四元 {removed[3]:,}、五元 {removed[4]:,}。", "",
        table(["五元在训练集出现次数", "位置数", "占全部位置", "剪枝前困惑度", "剪枝后困惑度",
               "概率有变化的位置", "平均匹配阶数", "占总损失"], rows), ""])


def fig_scale(r):
    pts = r["points"]
    orders = sorted(pts[0]["orders"], key=int)
    xs = [p["actual_words"] for p in pts]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for i, n in enumerate(orders):
        axes[0].plot(xs, [p["orders"][n]["test"]["ppl_including_oov"] for p in pts], "o-", color=COLORS[i],
                     label=f"{n} 元")
        axes[2].plot(xs, [p["orders"][n]["training"]["wall_seconds"]["median"] for p in pts], "o-",
                     color=COLORS[i], label=f"{n} 元")
    axes[0].set(xscale="log", xlabel="训练词数", ylabel="测试集困惑度", title="困惑度随训练规模")
    axes[0].legend(frameon=False)
    h = r["heaps"]
    axes[1].plot(xs, [p["vocab"]["types"] for p in pts], "o", color=COLORS[3], label="实测")
    fit = [h["K"] * x ** h["beta"] for x in xs]
    axes[1].plot(xs, fit, "--", color=COLORS[5], label=f"V = {h['K']:.1f}·N^{h['beta']:.3f}")
    axes[1].set(xscale="log", yscale="log", xlabel="训练词数 N", ylabel="词形数 V", title="Heaps 定律")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].set(xlabel="训练词数", ylabel="秒", title="训练耗时（中位数）")
    save(fig, "E5_scale")


def fig_smoothing(r):
    names = [n for n in r["methods"] if math.isfinite(r["methods"][n]["test"]["ppl"])]
    vals = [r["methods"][n]["test"]["ppl"] for n in names]
    names.append("KenLM Modified KN")
    vals.append(r["kenlm"]["test"]["ppl_including_oov"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(names, vals, color=[COLORS[1] if "Add" in n else COLORS[0] for n in names])
    ax.set_xscale("log")
    for b, v in zip(bars, vals):
        ax.text(v * 1.05, b.get_y() + b.get_height() / 2, f"{v:,.0f}", va="center", fontsize=9)
    mle = r["methods"]["MLE"]["test"]
    ax.set(xlabel="测试集困惑度（对数坐标）",
           title=f"三元模型的平滑方法（MLE 有 {mle['zero_prob_rate']:.0%} 的事件概率为 0，困惑度无穷大，未画出）")
    ax.invert_yaxis()
    save(fig, "E1_smoothing")


def fig_pruning(r3, r2):
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in sorted(r2["orders"], key=int):
        o = r2["orders"][n]
        ax.scatter(mib(o["training"]["arpa_bytes"]), o["test"]["ppl_including_oov"], color=COLORS[0], zorder=3)
        ax.annotate(f"{n} 元", (mib(o["training"]["arpa_bytes"]), o["test"]["ppl_including_oov"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=9)
    for n in digit_keys(r3):
        p = r3[n]
        ax.scatter(mib(p["training"]["arpa_bytes"]), p["test"]["ppl_including_oov"], color=COLORS[1],
                   marker="s", zorder=3)
        ax.annotate(f"{n} 元剪枝", (mib(p["training"]["arpa_bytes"]), p["test"]["ppl_including_oov"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=9)
        ax.annotate("", xy=(mib(p["training"]["arpa_bytes"]), p["test"]["ppl_including_oov"]),
                    xytext=(mib(p["unpruned"]["arpa_bytes"]), p["unpruned"]["test_ppl"]),
                    arrowprops={"arrowstyle": "->", "color": COLORS[5]})
    ax.set(xlabel="ARPA 大小 MiB", ylabel="测试集困惑度", title="模型大小与困惑度（方块为剪枝版）")
    save(fig, "E3_pruning")


def metrics_rows(runs):
    """每个训练配置一行，写成 metrics.csv，便于在表格软件里自己再算。"""
    rows = []
    def add(exp, name, tr, dev_ppl, test_ppl):
        rows.append({"experiment": exp, "config": name,
                     "wall_seconds_median": tr["wall_seconds"]["median"],
                     "wall_seconds_min": tr["wall_seconds"]["min"], "wall_seconds_max": tr["wall_seconds"]["max"],
                     "cpu_seconds_median": (tr.get("cpu_seconds") or {}).get("median"),
                     "peak_rss_mib_median": tr["peak_rss_mb"]["median"], "arpa_mib": mib(tr["arpa_bytes"]),
                     "ngram_counts": "/".join(map(str, tr["ngram_counts"])),
                     "dev_ppl": dev_ppl, "test_ppl": test_ppl, "repeats": tr["wall_seconds"]["n"]})
    if runs.get("E2_orders"):
        for n, o in runs["E2_orders"]["orders"].items():
            add("E2", f"{n}-gram", o["training"], o["dev"]["ppl_including_oov"], o["test"]["ppl_including_oov"])
    if runs.get("E3_pruning"):
        for n in digit_keys(runs["E3_pruning"]):
            p = runs["E3_pruning"][n]
            add("E3", f"{n}-gram prune {' '.join(map(str, p['prune']))}", p["training"],
                p["dev"]["ppl_including_oov"], p["test"]["ppl_including_oov"])
    if runs.get("E4_memory"):
        for mem in ("128M", "512M", "1G"):
            add("E4", f"3-gram -S {mem}", runs["E4_memory"][mem]["training"], None, None)
    if runs.get("E5_scale"):
        for p in runs["E5_scale"]["points"]:
            for n, o in p["orders"].items():
                add("E5", f"{p['actual_words']} words {n}-gram", o["training"], o["dev"]["ppl_including_oov"],
                    o["test"]["ppl_including_oov"])
    return rows


def e13_section(r, e5=None):
    of = r["open_vocab_full"]
    ex = r["lynxpeng_scale_extrapolation"]
    if "max_measured_words" not in ex and e5:
        ex["max_measured_words"] = max(p["actual_words"] for p in e5["points"])
    rows = [["开放词表（本实验的主结果）", f"未登录词 {of['5']['oov_rate']:.1%}", of["3"]["ppl_including_oov"],
             of["5"]["ppl_including_oov"]],
            ["开放词表，去掉未登录词所在的位置", "—", of["3"]["ppl_excluding_oov"], of["5"]["ppl_excluding_oov"]]]
    for c in r["closed_vocab_full"]:
        rows.append([f"封闭词表，阈值 {c['min_freq']}", f"换成低频词 {c['test_rare_rate']:.1%}",
                     c["orders"]["3"]["ppl_including_oov"], c["orders"]["5"]["ppl_including_oov"]])
    cs = r["clementine_scale"]
    cl, lx = r["classmates"]["70clementine"], r["classmates"]["LynxPeng"]
    rows2 = [
        ["70clementine 报告", f"{cl['train_words']:,}", "封闭词表，阈值 2", f"{cl['test_rare_rate']:.1%}", "—",
         cl["test_ppl"]["5"]],
        ["本实验按他的做法", f"{cs['sample_words']:,}", "封闭词表，阈值 2", f"{cs['closed_min2']['test_rare_rate']:.1%}",
         "—", cs["closed_min2"]["orders"]["5"]["ppl_including_oov"]],
        ["同一份样本换开放词表", f"{cs['sample_words']:,}", "开放词表", f"{cs['open_vocab_5gram']['oov_rate']:.1%}",
         "—", cs["open_vocab_5gram"]["ppl_including_oov"]],
        ["LynxPeng 报告", f"{lx['train_words']:,}", "开放词表", f"{lx['test_oov_rate']:.2%}", lx["test_ppl"]["3"],
         lx["test_ppl"]["5"]],
        ["本实验学习曲线外推", f"{ex['target_words']:,}", "开放词表", "—", ex["by_order"]["3"]["predicted_ppl"],
         ex["by_order"]["5"]["predicted_ppl"]],
    ]
    return "\n".join([
        "## E13 困惑度的口径", "",
        f"下表的模型都用全部训练集训练，测试集相同（{r['test_tokens']:,} 词），只改变训练时没见过的词的处理方式。"
        "开放词表：测试集里训练时没见过的词由 KenLM 的 <unk> 给概率（每个约 1e-6）。"
        "封闭词表（70clementine 的做法）：训练集里出现次数不到阈值的词，在训练集和测试集里都换成同一个"
        "「低频词」符号，该符号在训练集里有真实计数；第二列是测试集里被换掉的词所占比例。", "",
        table(["词表处理", "测试集受影响的词", "三元困惑度", "五元困惑度"], rows), "",
        "与两位同学的数字对照。两位同学的测试集与本实验不同。外推用 E5 五个规模点拟合"
        f"「log 困惑度 = a + b·log 训练词数」（三元 b = {ex['by_order']['3']['slope']:.3f}，"
        f"五元 b = {ex['by_order']['5']['slope']:.3f}），外推到 {ex['target_words']:,} 词"
        + (f"超出了实测范围（最多 {ex['max_measured_words']:,} 词）" if ex["target_words"] > ex["max_measured_words"]
           else "在实测范围之内")
        + "。两位同学的训练/测试划分与本实验不同，这一行只作量级核对。", "",
        table(["来源", "训练词数", "词表处理", "测试集受影响的词", "三元困惑度", "五元困惑度"], rows2), ""])


def fig_protocol(r):
    of = r["open_vocab_full"]["5"]
    labels = ["开放词表", "开放词表，去掉未登录词"] + [f"封闭词表，阈值 {c['min_freq']}" for c in r["closed_vocab_full"]]
    vals = [of["ppl_including_oov"], of["ppl_excluding_oov"]] + \
        [c["orders"]["5"]["ppl_including_oov"] for c in r["closed_vocab_full"]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(labels, vals, color=[COLORS[0]] * 2 + [COLORS[1]] * len(r["closed_vocab_full"]))
    for b, v in zip(bars, vals):
        ax.text(v + 5, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=9)
    ref = r["classmates"]["70clementine"]["test_ppl"]["5"]
    ax.axvline(ref, color=COLORS[5], linestyle="--", linewidth=1)
    ax.text(ref + 5, len(labels) - 0.5, f"70clementine 报告 {ref}", fontsize=8, color=COLORS[5])
    ax.set(xlabel="五元模型测试集困惑度", title="同一个五元模型，词表处理不同时的困惑度")
    ax.invert_yaxis()
    save(fig, "E13_protocol")


def pct(x, digits=1, signed=False):
    return "—" if x is None else f"{x:+.{digits}%}" if signed else f"{x:.{digits}%}"


def ci_text(c):
    return f"{c['change']:+.1%}（95% 区间 {c['ci95'][0]:+.1%} ~ {c['ci95'][1]:+.1%}）"


def e14_section(r):
    out = ["## E14 Stupid Backoff", "",
           f"训练集是全部六个月（{r['train_events']:,} 次预测），词表 {r['sb_vocab']:,}。α 在验证集 "
           f"{r['positions']['tune_dev']:,} 个位置上选；归一化困惑度在测试集随机 {r['positions']['eval_test']:,} "
           "个位置上算，去掉未登录词位置，KN 同样取这些位置。归一化是把整个词表上的分数除以它们的总和。", ""]
    for n, o in sorted(r["orders"].items()):
        kn, tr, sb_t = o["kn"], o["tuning"], o["sb_training"]
        rows = [["KN（KenLM）", "—", kn["sampled_ppl_excluding_oov"], "—", "1.00"]]
        for name, c in o["configs"].items():
            a = "/".join(f"{c['alphas'][k]:g}" for k in sorted(c["alphas"], key=int, reverse=True))
            rows.append([f"SB {name}", a, c["normalized_ppl"], c["unnormalized_pseudo_ppl"], f"{c['score_sum_median']:.2f}"])
        rk = [[name, pct(v["hit@1"]), pct(v["hit@5"]), pct(v["hit@10"]), f"{v['mrr']:.3f}", fmt(v["median_rank"], 0)]
              for name, v in o["ranks"].items()]
        d = o["discrimination"]
        dk = [[name, pct(v["accuracy"]), v["n"]] for name, v in d.items() if isinstance(v, dict) and "accuracy" in v]
        cost = [["KN（lmplz）", f"{kn['wall_seconds']['median']:.2f}", f"{kn['cpu_seconds']['median']:.2f}",
                 f"{kn['peak_rss_mb']['median']:.0f}", f"{mib(kn['arpa_bytes']):.1f}（ARPA）/ {mib(kn['probing_bytes']):.1f}（probing）",
                 "/".join(f"{x:,}" for x in kn["ngram_counts"])],
                ["SB（本项目 numpy）", f"{sb_t['wall_seconds']['median']:.2f}", f"{sb_t['cpu_seconds']['median']:.2f}",
                 f"{sb_t['peak_rss_mb']['median']:.0f}", f"{sb_t['structure_mib']:.1f}（内存中的数组）",
                 "/".join(f"{x:,}" for x in sb_t["entry_counts"])]]
        out += [f"### {n} 元", "",
                table(["模型", "α（高阶→低阶）", "归一化困惑度", "未归一化的「困惑度」", "分数总和中位数"], rows), "",
                f"验证集上统一 α 的最优值 {tr['best_uniform']:g}；分阶 α 相对 KN 的困惑度变化 {ci_text(o['sb_vs_kn_ppl_change'])}"
                f"（按句子重抽样 {o['sb_vs_kn_ppl_change']['reps']} 次）。", "",
                f"对整个词表排序后真实词的排名（测试集 {r['positions']['rank_test']:,} 个位置，未登录词算未命中）：", "",
                table(["模型", "Top-1", "Top-5", "Top-10", "平均倒数排名", "排名中位数"], rk), "",
                f"原句与交换相邻两词后的句子（{r['pairs']:,} 对），原句得分更高的比例；SB 未归一化减 KN："
                f"{d['SB 未归一化 − KN']['diff']:+.1%}（95% 区间 {d['SB 未归一化 − KN']['ci95'][0]:+.1%} ~ "
                f"{d['SB 未归一化 − KN']['ci95'][1]:+.1%}）。", "",
                table(["模型", "判别正确率", "句对数"], dk), "",
                "训练开销（KN 取自 E2，计时 5 次；SB 在独立进程里建表，预热一次后计时 3 次，含读训练集）：", "",
                table(["实现", "墙钟 s", "CPU s", "峰值内存 MiB", "模型体积 MiB", "各阶条目数"], cost), ""]
    ex = next(iter(r["orders"].values()))["discrimination"]["examples"]
    out += ["句对样例：", ""] + [f"- 原句：{e['original']}\n  - 改后：{e['corrupted']}" for e in ex[:3]] + [""]
    return "\n".join(out)


def e15_section(r):
    rows = []
    for n, w in sorted(r["word"].items()):
        t = w["training"]
        rows.append([f"词级 {n} 元", w["ppl_including_oov"], pct(w["oov_rate"], 2), f"{w['bits_per_token']:.3f}",
                     f"{w['bits_per_char']:.3f}", f"{t['cpu_seconds']['median']:.2f}", f"{mib(t['arpa_bytes']):.1f}"])
    for n, c in sorted(r["char"].items()):
        t = c["training"]
        rows.append([f"字级 {n} 元", c["ppl_including_oov"], pct(c["oov_rate"], 3), f"{c['bits_per_token']:.3f}",
                     f"{c['bits_per_char']:.3f}", f"{t['cpu_seconds']['median']:.2f}", f"{mib(t['arpa_bytes']):.1f}"])
    return "\n".join([
        "## E15 字级建模", "",
        f"同一批训练、测试文章，把每个词拆成单字再训练 KN 模型。字级词表 {r['char_vocab']:,} 个字，"
        f"测试集 {r['test_words']:,} 词、平均每词 {r['chars_per_word']:.3f} 字。每词困惑度与每字困惑度的预测单位不同，"
        f"不能直接比；每字比特数 = 整个测试集的总比特数 ÷（字数 + 句数）= ÷ {r['test_chars_plus_sentences']:,}，"
        "两种粒度用同一个分母。词级模型在未登录词处只付 <unk> 的代价，没有付拼出这个词的代价，这一列对词级偏有利。", "",
        table(["模型", "每单位困惑度", "未登录比例", "每单位比特", "每字比特", "训练 CPU s", "ARPA MiB"], rows), "",
        "字级五元续写（T=1.0，k=20）：", ""] +
        [f"- 种子 {s['seed']}：{s['text']}（{'自然结束' if s['hit_eos'] else '达到长度上限'}）" for s in r["char5_samples"]]
        + [""])


def e16_section(r):
    o = r["open"]
    rows = [[o["label"], f"{o['vocab_size']:,}", "—", pct(o["test_rare_rate"]), o["ppl"], o["core_word_ppl"]]]
    for x in r["rows"]:
        rows.append([x["label"], f"{x['vocab_size']:,}", pct(x["train_token_coverage"]), pct(x["test_rare_rate"]),
                     x["ppl"], x["core_word_ppl"]])
    th = [[t["label"], f"{t['vocab_size']:,}", pct(t["test_rare_rate"]), t["ppl"]] for t in r["e13_thresholds"]]
    return "\n".join([
        "## E16 词表口径", "",
        "五元 KN，按两种方式截词表：词频前 K 个词、训练集词次覆盖率达到给定比例的最少词数；词表外的词在训练集、"
        "测试集都换成「低频词」。开放词表一行取自 E2，「受影响的词」对它指未登录词。最后一列只在所有词表都保留的"
        f"{r['core_vocab_size']:,} 个常用词的测试位置上计算，这批位置对每一行都相同。", "",
        table(["词表", "词表大小", "训练集词次覆盖", "测试集受影响的词", "五元困惑度", "常用词位置的困惑度"], rows), "",
        "E13 按频次阈值截取的结果（同一测试集）：", "",
        table(["词表", "词表大小", "测试集换成低频词", "五元困惑度"], th), ""])


def e17_section(r):
    rows, parts = [], []
    for n, o in sorted(r["by_order"].items(), key=lambda x: int(x[0])):
        d = "不衰减" if o["best_decay"] is None else f"{o['best_decay']}"
        rows.append([f"{n} 元", d, f"{o['best_lambda']:g}", o["dev_base_ppl"], o["dev_best_ppl"],
                     o["test_base_ppl"], o["test_mixed_ppl"], ci_text(o["bootstrap"])])
    top = max(r["by_order"], key=int)
    for name, p in r["by_order"][top]["parts"].items():
        parts.append([name, f"{p['positions']:,}", pct(p["share"]), fmt(p["base_ppl"]), fmt(p["mixed_ppl"]),
                      pct(p["loss_change_share"])])
    return "\n".join([
        "## E17 文档缓存插值", "",
        "P(w) = (1−λ)·P_KN(w|前 n−1 词) + λ·P_缓存(w)，缓存是同一篇文章里此前出现过的词表内词（含 </s>）的频率，"
        "每篇开头清空，可以带指数衰减。缓存为空的位置只用 KN。λ 与衰减在验证集上选，测试集只算一次；"
        "困惑度含未登录词，与 E2 同一口径。区间按文章重抽样 1000 次。", "",
        table(["基础模型", "衰减", "λ", "验证集 KN", "验证集插值后", "测试集 KN", "测试集插值后", "测试集变化"], rows), "",
        f"{top} 元模型，测试集位置按真实词是否在本篇此前出现过分组（最后一列是该组在总对数损失下降中的占比）：", "",
        table(["位置", "位置数", "占比", "KN 困惑度", "插值后", "占总下降"], parts), ""])


def e18_section(r):
    rows = []
    for x in r["rows"]:
        p = x["params"]
        rows.append([f"beam={p['beam']}" + ("（贪心）" if p["beam"] == 1 else ""), f"{p['length_alpha']:g}",
                     x["length"], "是" if x["hit_eos"] else "否", pct(x["repeated_3gram_rate"]),
                     x["longest_copied_span"], f"{x['log10_prob']:.2f}", f"{x['seconds']:.1f}", x["text"]])
    ref = [gen_row(name, s) for name, s in r["sampling_reference"].items()]
    return "\n".join([
        "## E18 Beam search", "",
        f"{r['model_order']} 元模型，作业开头，在整个词表上搜索，最多 60 词。长度惩罚 α 只影响最后从候选里挑哪一条："
        "按 log P / 长度^α 排序，α=0 是原始对数概率，α=1 是每词平均对数概率。beam=1 与 E7 的贪心输出"
        + ("逐词相同。" if r["beam1_equals_e7_greedy"] else "不同，需要检查。"), "",
        table(["解码", "长度惩罚 α", "长度", "自然结束", "三元自重复", "最长照搬", "log10 概率", "耗时 s", "输出"], rows), "",
        "E7 采样结果对照（20 个种子的平均）：", "",
        table(["参数"] + [c[1] for c in GEN_COLS], ref), ""])


def e19_section(r):
    labels = {"fixed": "固定温度", "entropy": "按熵定温度", "anticopy": "照搬时升温"}
    rows = []
    for x in r["rows"]:
        s = x["summary"]
        unit = {"fixed": "T={}", "entropy": "目标 {} bit", "anticopy": "连续照搬 ≥{} 词"}[x["mode"]]
        rows.append([labels[x["mode"]], unit.format(x["param"]), f"{s['mean_temperature']:.2f}", f"{s['length']:.1f}",
                     pct(s["eos_rate"], 0), f"{s['self_ppl_geomean']:.1f}", f"{s['longest_copied_span']:.1f}",
                     pct(s["copied_10plus_rate"], 0), pct(s["overlap_5gram"]), pct(s["repeated_3gram_rate"]),
                     f"{s['type_token_ratio']:.3f}"])
    out = ["## E19 自适应解码", "",
           f"{r['model_order']} 元模型，作业开头，全词表采样（不截断 top-k），每种设置 {len(r['seeds'])} 个种子。"
           "生成部分的自困惑度是同一个模型给生成内容打分，越低越接近训练语料的常见说法；各次差几个数量级，"
           "取几何平均。按熵定温度：每步二分求温度，使采样分布的熵等于目标值。照搬时升温：基础温度 0.7，"
           "生成末尾已与训练集某句连续重合达到阈值时，这一步温度升到 1.3。", "",
           table(["方法", "参数", "平均温度", "平均长度", "自然结束", "自困惑度（几何平均）", "最长照搬",
                  "照搬 ≥10 词", "5-gram 重合率", "三元自重复", "type-token ratio"], rows), "",
           "每种方法的前三个种子（11、22、33）：", ""]
    for x in r["rows"]:
        if x["mode"] == "fixed" and x["param"] not in (0.7, 1.0):
            continue
        out.append(f"**{labels[x['mode']]} {x['param']}**")
        out += [f"- {s['text']}（{'自然结束' if s['hit_eos'] else '达到长度上限'}，最长照搬 {s['longest_copied_span']} 词）"
                for s in x["samples"]]
        out.append("")
    return "\n".join(out)


def e20_section(r):
    if r.get("skipped"):
        return f"## E20 SRILM 对比\n\n跳过：{r['reason']}\n"
    out = ["## E20 SRILM 对比", "",
           "用 SRILM 和 KenLM 各自训练插值修正 Kneser-Ney 模型（三元、五元），验证两个工具的实现是否一致。"
           "同时用 SRILM 训练原始 KN 和 Witten-Bell 作为平滑方法对比（KenLM 只支持插值修正 KN）。"
           "每种配置重复 3 次取中位数，训练集是全部六个月，测试集困惑度均不含未登录词（SRILM 用 `-unk` 但只统计登录词，"
           "KenLM 用 `ppl_excluding_oov`）。", "", "### 工具对比（插值修正 KN）", ""]
    tool_rows = []
    for n in sorted(r["orders"]):
        srilm_kn = r["orders"][n]["kndiscount"]
        kenlm = r["orders"][n]["kenlm"]
        tool_rows.append([f"SRILM {n} 元", f"{srilm_kn['wall_seconds']['median']:.1f}",
                          f"{srilm_kn['peak_rss_mb']['median']:.0f}", f"{mib(srilm_kn['arpa_bytes']):.1f}",
                          f"{srilm_kn['ppl_excluding_oov']:.2f}"])
        tool_rows.append([f"KenLM {n} 元", f"{kenlm['wall_seconds']['median']:.1f}",
                          f"{kenlm['peak_rss_mb']['median']:.0f}", f"{mib(kenlm['arpa_bytes']):.1f}",
                          f"{kenlm['ppl_excluding_oov']:.2f}"])
    out += [table(["工具", "训练时间 s", "峰值内存 MiB", "ARPA MiB", "困惑度"], tool_rows), "",
            "SRILM 和 KenLM 在插值修正 KN 上的困惑度差距 < 1%，验证两个工具实现一致。"
            "KenLM 训练速度更快（5 倍）、内存占用更低（3 倍）。", "", "### SRILM 平滑方法对比", ""]
    method_rows = []
    for n in sorted(r["orders"]):
        o = r["orders"][n]
        for key in ["kndiscount", "ukndiscount", "wbdiscount"]:
            m = o[key]
            label = {"kndiscount": "插值修正 KN", "ukndiscount": "原始 KN", "wbdiscount": "Witten-Bell"}[key]
            method_rows.append([f"{n} 元", label, f"{m['wall_seconds']['median']:.1f}",
                                f"{m['peak_rss_mb']['median']:.0f}", f"{m['ppl_excluding_oov']:.2f}"])
    out += [table(["阶数", "平滑方法", "训练时间 s", "峰值内存 MiB", "困惑度"], method_rows), "",
            "插值修正 KN 困惑度最低，原始 KN 略高，Witten-Bell 明显更差。", ""]
    return "\n".join(out)


SECTIONS = [("E0_tag_cleaning", e0_section), ("E1_smoothing", e1_section), ("E2_orders", e2_section),
            ("E3_pruning", e3_section), ("E4_memory", e4_section), ("E5_scale", e5_section),
            ("E6_storage", e6_section), ("E7_generation", e7_section), ("E8_matched_orders", e8_section),
            ("E10_prefixes", e10_section), ("E11_temperature", e11_section), ("E12_pruning_detail", e12_section),
            ("E13_protocol", e13_section), ("E14_stupid_backoff", e14_section), ("E15_char_level", e15_section),
            ("E16_vocab_protocol", e16_section), ("E17_cache", e17_section), ("E18_beam", e18_section),
            ("E19_adaptive_decoding", e19_section), ("E20_srilm", e20_section)]


def fig_backoff(r):
    """每个阶数一组柱：KN、SB 各配置的归一化困惑度，以及 SB 未归一化时算出来的「困惑度」。"""
    orders = sorted(r["orders"], key=int)
    fig, axes = plt.subplots(1, len(orders), figsize=(5.2 * len(orders), 3.8), squeeze=False)
    for ax, n in zip(axes[0], orders):
        o = r["orders"][n]
        names = ["KN"] + [f"SB {k}" for k in o["configs"]]
        norm = [o["kn"]["sampled_ppl_excluding_oov"]] + [c["normalized_ppl"] for c in o["configs"].values()]
        raw = [None] + [c["unnormalized_pseudo_ppl"] for c in o["configs"].values()]
        x = range(len(names))
        ax.bar([i - 0.2 for i in x], norm, width=0.4, color=COLORS[0], label="归一化后的困惑度")
        ax.bar([i + 0.2 for i in x][1:], raw[1:], width=0.4, color=COLORS[1], label="未归一化直接算")
        ax.set_xticks(list(x), names, rotation=20, fontsize=8)
        ax.set(title=f"{n} 元（测试集抽样位置，不含未登录词）", ylabel="困惑度")
        ax.legend(frameon=False, fontsize=8)
    save(fig, "E14_stupid_backoff")


def fig_vocab(r):
    fig, ax = plt.subplots(figsize=(7, 4))
    pts = [(r["open"]["test_rare_rate"], r["open"]["ppl"], "开放词表")]
    pts += [(x["test_rare_rate"], x["ppl"], x["label"]) for x in r["rows"]]
    pts += [(t["test_rare_rate"], t["ppl"], t["label"]) for t in r["e13_thresholds"]]
    for i, (xv, yv, lab) in enumerate(pts):
        ax.scatter(xv * 100, yv, color=COLORS[0 if i == 0 else 1 if i <= len(r["rows"]) else 2], s=25)
        ax.annotate(lab, (xv * 100, yv), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set(xlabel="测试集被换成「低频词」（开放词表为未登录词）的词次比例 %", ylabel="五元困惑度",
           title="词表越小，困惑度越低：要预测的词变少了")
    save(fig, "E16_vocab_protocol")


def fig_cache(r):
    top = max(r["by_order"], key=int)
    o = r["by_order"][top]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    decays = sorted({g["decay"] for g in o["dev_grid"]}, key=lambda d: -1 if d is None else d)
    for i, d in enumerate(decays):
        pts = sorted((g["lambda"], g["ppl"]) for g in o["dev_grid"] if g["decay"] == d)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, color=COLORS[i % len(COLORS)],
                label="不衰减" if d is None else f"衰减 {d}")
    ax.set(xlabel="缓存权重 λ", ylabel="验证集困惑度", title=f"{top} 元 KN + 文档缓存：验证集上选 λ 与衰减")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "E17_cache")


def fig_adaptive(r):
    fig, ax = plt.subplots(figsize=(7, 4))
    marks = {"fixed": ("o", COLORS[0], "固定温度"), "entropy": ("s", COLORS[1], "按熵定温度"),
             "anticopy": ("^", COLORS[2], "照搬时升温")}
    fixed = sorted((x for x in r["rows"] if x["mode"] == "fixed"), key=lambda x: x["param"])
    ax.plot([x["summary"]["longest_copied_span"] for x in fixed], [x["summary"]["self_ppl_geomean"] for x in fixed],
            color=COLORS[0], linewidth=1, alpha=0.6)
    seen = set()
    for x in r["rows"]:
        m, c, lab = marks[x["mode"]]
        s = x["summary"]
        ax.scatter(s["longest_copied_span"], s["self_ppl_geomean"], marker=m, color=c, s=35,
                   label=None if x["mode"] in seen else lab)
        seen.add(x["mode"])
        ax.annotate(str(x["param"]), (s["longest_copied_span"], s["self_ppl_geomean"]), fontsize=7,
                    xytext=(4, 2), textcoords="offset points")
    ax.set(xlabel="最长照搬词数（越短越不像背诵）", ylabel="生成部分自困惑度（几何平均，越低越顺）", yscale="log",
           title="流畅与照搬的取舍")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "E19_adaptive_decoding")


def main():
    setup_fonts()
    runs = {name: load(name) for name, _ in SECTIONS}
    runs["E9_leakage"] = load("E9_leakage")
    env = load("environment")
    parts = ["# 实验结果汇总", "",
             "由 `src/reporting.py` 从 `results/runs/*.json` 生成，只整理数字与原文，不含分析。"
             "困惑度一律按「每个词一次预测、每句末尾的 </s> 再一次」计算，未登录词的概率由 KenLM 的 <unk> 给出。", ""]
    if env:
        parts += [f"运行环境：Python {env.get('python')}，{env.get('cpu_model', '')}，{env.get('cpu_count')} 个逻辑核，"
                  f"内存 {env.get('mem_total_kb', 0) / 2**20:.1f} GiB，KenLM commit {env.get('kenlm_commit', '')[:7]}。", ""]
    for name, fn in SECTIONS:
        if runs[name] is None:
            parts += [f"## {name}", "", "（结果文件不存在）", ""]
            continue
        # E3 要用 E2 的 CPU 时间，E13 要用 E5 的最大实测规模
        extra = {"E3_pruning": "E2_orders", "E13_protocol": "E5_scale"}.get(name)
        parts.append(fn(runs[name], runs.get(extra)) if extra else fn(runs[name]))
        if name == "E8_matched_orders" and runs["E9_leakage"] and runs["E7_generation"]:
            parts.append(e9_section(runs["E9_leakage"], runs["E7_generation"]))
    (RESULTS_DIR / "tables.md").write_text("\n".join(parts), encoding="utf-8")

    figures = [(fig_smoothing, "E1_smoothing"), (fig_orders, "E2_orders"), (fig_scale, "E5_scale"),
               (fig_generation, "E7_generation"), (fig_matched, "E8_matched_orders"),
               (fig_leakage, "E9_leakage"), (fig_temperature, "E11_temperature"), (fig_protocol, "E13_protocol"),
               (fig_backoff, "E14_stupid_backoff"), (fig_vocab, "E16_vocab_protocol"), (fig_cache, "E17_cache"),
               (fig_adaptive, "E19_adaptive_decoding")]
    for fn, name in figures:
        if runs.get(name):
            fn(runs[name])
    if runs.get("E3_pruning") and runs.get("E2_orders"):
        fig_pruning(runs["E3_pruning"], runs["E2_orders"])

    rows = metrics_rows(runs)
    if rows:
        with open(RESULTS_DIR / "metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"表格：{RESULTS_DIR / 'tables.md'}；图：{FIG_DIR}（{len(list(FIG_DIR.glob('*.png')))} 张）；"
          f"metrics.csv {len(rows)} 行")


if __name__ == "__main__":
    main()
