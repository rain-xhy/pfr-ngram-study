# 步 4 安装日志：KenLM

执行日期：2026-09-28。原稿放在已删除的 `作业/1/结果/` 目录下，2026-09-29 按当时记录的内容重建于此。

## 环境

| 项 | 值 |
|---|---|
| 宿主 | Windows 11 Home China 10.0.26200 |
| 编译环境 | WSL 2 / Ubuntu 24.04.1 LTS（已存在，未新建） |
| CPU / 内存 | 16 核 / 7.6 GiB 可用 |
| KenLM commit | `4cb443e60b7bf2c0ddf3c745378f76cb59e254e5`（2025-03-30） |
| 安装位置 | WSL 内 `~/tools/kenlm/build/bin`（即 `/root/tools/kenlm/build/bin`） |

KenLM 依赖 Boost 与 POSIX 接口，在 Linux 下编译是标准流程，因此放在 WSL 里编译。

## 依赖

Ubuntu 里原有 gcc、g++、make、git、zlib.h，补装了以下几项：

| 缺失 | apt 包 | 装后版本 |
|---|---|---|
| Boost | `libboost-all-dev` | 1.83 |
| Eigen | `libeigen3-dev` | /usr/include/eigen3 |
| cmake | `cmake` | 3.28.3 |
| bzlib.h / lzma.h | `libbz2-dev` `liblzma-dev` | — |

```bash
sudo apt-get install -y libboost-all-dev libeigen3-dev cmake libbz2-dev liblzma-dev
```

第一次检查环境时，在 PowerShell 双引号里写了 `\$t`，PowerShell 不把反斜杠当转义符，bash 收到的是空变量，
`command -v` 不带参数返回 0，于是五个工具全部显示存在。改为把检查命令写进脚本文件再交给 bash 执行，才查出缺 cmake、bzlib.h 与 lzma.h。

## 编译

```bash
mkdir -p ~/tools && cd ~/tools
git clone --depth 1 https://github.com/kpu/kenlm.git
cd kenlm && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

cmake 配置用时 51.4 s，make 一次通过。OpenMP 4.5 被检出并启用。产出 11 个可执行文件，本作业用到三个：

| 程序 | 用途 |
|---|---|
| `lmplz` | 训练，产出 ARPA |
| `build_binary` | ARPA 转二进制，加快加载 |
| `query` | 打分、算困惑度 |

## 冒烟测试

用 240 行玩具中文语料（4 句循环 60 次，空格分词）训练 3-gram：

```bash
lmplz -o 3 --discount_fallback -S 20% < toy.txt > toy.arpa
```

- 词表 21 词，1-gram 21 条、2-gram 28 条、3-gram 30 条
- 训练耗时 `real 0.17 s`，`RSSMax 367 MB`
- ARPA 2472 字节，转二进制后 2030 字节
- `query` 对 `在 阳光 明媚 的 五月 我们 学校 胜利 召开 了` 给出 PPL 4.55，OOV 0

KenLM 按空白切 token，中文不需要额外处理，分词由上游完成。

## 语料格式要求（2026-09-29 补充）

用同样的玩具句子做了两组对照：

- 语料里写了字面 `<s> … </s>`：lmplz 退出码 134，报
  `Special word <s> is not allowed in the corpus. … Pass --skip_symbols to convert these symbols to whitespace.`
- 语料只有词、一句一行：正常训练，ARPA 的 1-gram 表里自动出现 `<s>` 与 `</s>` 两项

所以清洗后的语料一句一行、不写边界符，`src/data.py` 的 `clean_corpus` 按此输出。

第一组对照最初没有加 `-S` 参数时，命令在 120 秒内没有结束，被转到后台。原因没有查实，
正式训练时一律显式指定 `-S`。

## 正式实验的两条注意事项

`--discount_fallback` 在玩具语料上是必需的。语料太小时 Kneser-Ney 的折扣参数估不出来，lmplz 会报错退出。
正式跑人民日报语料时先不加，报错再加。加了等于用兜底折扣值替代真实估计，这一点要记进实验记录。

`-S` 指定排序内存上限。它会影响训练耗时，正式训练时写进计时表，否则耗时数字没有可比性。

## 尚未安装：SRILM

SRILM 需到官网填表下载，不在 GitHub 上，属于可选项。若需要第三方工具校验 KN 实现，备选是 NLTK 的 `nltk.lm.KneserNeyInterpolated`。
