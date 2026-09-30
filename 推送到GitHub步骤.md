# 推送项目到GitHub完整步骤

## 当前状态
- ✅ Git已初始化
- ✅ 在main分支上
- ⚠️  README.md有修改未提交
- 📦 准备推送到: https://github.com/rain-xhy/pfr-ngram-study.git

---

## 步骤1: 添加并提交所有更改

```bash
# 进入项目目录
cd "E:\desktop\北大软微\课程\大语言模型应用与实践\作业\1\pfr-ngram-study"

# 查看当前状态
git status

# 添加所有更改（.gitignore会自动排除不需要的文件）
git add .

# 查看将要提交的文件
git status

# 提交更改
git commit -m "完成E0-E13实验，包含1月和6月数据的n-gram建模"
```

---

## 步骤2: 添加远程仓库

```bash
# 检查是否已经添加了远程仓库
git remote -v

# 如果没有，添加远程仓库
git remote add origin https://github.com/rain-xhy/pfr-ngram-study.git

# 验证
git remote -v
```

---

## 步骤3: 推送到GitHub

```bash
# 推送到main分支（首次推送）
git push -u origin main
```

**如果遇到错误 "failed to push some refs"（远程仓库有README等文件）**：

```bash
# 先拉取远程内容并合并
git pull origin main --allow-unrelated-histories

# 解决冲突（如果有）
# 然后提交合并
git commit -m "合并远程仓库"

# 再次推送
git push -u origin main
```

**或者强制推送（会覆盖远程内容，慎用）**：

```bash
git push -u origin main --force
```

---

## 步骤4: 验证推送成功

访问: https://github.com/rain-xhy/pfr-ngram-study

应该能看到：
- ✅ README.md
- ✅ src/ 目录
- ✅ scripts/ 目录
- ✅ tests/ 目录
- ✅ results/ 目录（不包含大文件）

---

## 完整命令序列（一次性执行）

```bash
cd "E:\desktop\北大软微\课程\大语言模型应用与实践\作业\1\pfr-ngram-study"

# 提交所有更改
git add .
git commit -m "完成E0-E13实验，包含1月和6月数据的n-gram建模"

# 添加远程仓库（如果还没有）
git remote add origin https://github.com/rain-xhy/pfr-ngram-study.git

# 推送（首次）
git push -u origin main
```

---

## 如果GitHub要求身份验证

### 方法1: 使用Personal Access Token（推荐）

1. 访问 GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 点击 "Generate new token"
3. 勾选 `repo` 权限
4. 生成并复制token

**使用token推送**:
```bash
# 方法A: 在URL中包含token
git remote set-url origin https://YOUR_TOKEN@github.com/rain-xhy/pfr-ngram-study.git
git push -u origin main

# 方法B: 推送时输入
git push -u origin main
# Username: rain-xhy
# Password: 粘贴你的token
```

### 方法2: 使用SSH Key

```bash
# 生成SSH密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 复制公钥
cat ~/.ssh/id_ed25519.pub

# 在GitHub添加SSH密钥: Settings → SSH and GPG keys → New SSH key

# 修改远程URL为SSH格式
git remote set-url origin git@github.com:rain-xhy/pfr-ngram-study.git

# 推送
git push -u origin main
```

---

## 检查.gitignore是否正确

```bash
# 查看哪些文件会被提交（确保大文件被排除）
git status

# 查看忽略的文件
git status --ignored
```

**确保以下大文件被忽略**:
- ✅ corpus/raw/（原始语料）
- ✅ corpus/processed/（处理后语料）
- ✅ *.arpa（模型文件）
- ✅ *.bin（二进制模型）

---

## 后续推送

以后修改代码后，推送更简单：

```bash
# 提交更改
git add .
git commit -m "描述你的更改"

# 推送（不需要-u参数了）
git push
```

---

## 常见问题

### Q1: 推送太慢或失败？
**A**: 可能是文件太大，检查：
```bash
# 查看仓库大小
du -sh .git

# 查看大文件
find . -type f -size +10M
```

### Q2: 推送显示"Everything up-to-date"？
**A**: 说明没有新的提交，先commit再push

### Q3: 推送后GitHub显示不全？
**A**: 检查.gitignore，确保想要的文件没被排除

---

## 推荐的提交信息格式

```bash
git commit -m "feat: 添加E14 SRILM实验"
git commit -m "fix: 修复困惑度计算bug"
git commit -m "docs: 更新README添加使用说明"
git commit -m "refactor: 重构数据处理模块"
```

---

准备好了就按顺序执行命令吧！🚀
