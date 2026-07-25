# Git

## 常用配置

```bash
git config --global user.name "your-name"
git config --global user.email "your-email@example.com"
git config --global init.defaultBranch main
```

SSH 连接能减少重复输密码，适合长期维护远端仓库。

## 清理 Git 缓存

适用场景：

- `.gitignore` 改了，但某些文件早就被 Git 跟踪了。
- 想让权重、日志、临时文件停止被提交。

核心认知：

- `.gitignore` 只影响“未被跟踪”的文件。
- 已经进过 Git 的文件，需要先从索引里移除。

常用做法：

```bash
git rm -r --cached .
git add .
git status
```

说明：

- `--cached` 只清 Git 索引，不删工作区文件。
- 重新 `git add .` 后，Git 会按新的 `.gitignore` 规则决定哪些文件重新进入版本控制。

## 把当前版本变成新的 Git 起点

适用场景：

- 仓库历史太乱，想把当前整理后的内容作为全新基线。
- 不再需要保留旧 commit 历史。
- 准备重新开始管理这个仓库。

最直接的做法：

```bash
rm -rf .git
git init -b main
git add .
git commit -m "Initial cleaned repository baseline"
```

在 Windows PowerShell 里可以这样做：

```powershell
Remove-Item -LiteralPath .git -Recurse -Force
git init -b main
git add .
git commit -m "Initial cleaned repository baseline"
```

说明：

- 这会彻底删除本地历史、分支、tag、reflog。
- 文件内容保留，只有版本历史被清空。
- 如果之后推送到远端，通常需要强推或推到一个全新的远端仓库。

## 这次仓库整理里用到的思路

1. 先整理目录结构和 `.gitignore`。
2. 把大文件、权重、wheel、日志、本地资料归档到 `archive/local_ignored/`。
3. 确认主仓库只保留值得进入版本控制的内容。
4. 最后再删除 `.git`，重新初始化，形成新的起点。

## 不要混淆的两件事

- 清理 Git 缓存：
  解决“哪些文件应该被跟踪”的问题。
- 清空 Git 历史：
  解决“旧提交记录是否保留”的问题。
