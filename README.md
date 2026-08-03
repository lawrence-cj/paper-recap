# Paper Recap

一个为“快速回忆”设计的个人论文阅读库。每篇记录是 `content/papers/` 中的独立 Markdown 文件；推送到 `main` 后，GitHub Actions 会自动校验并发布到 GitHub Pages。

## 以后最简流程

你不需要自己套模板。每次和 Agent 聊完后，把以下四类原始信息保留下来即可：

```text
论文：标题或链接
我的理解：3～10 条，不必整理
重要问答：只保留真正改变理解的 Q&A
我的判断：是否可信、是否值得复现、与当前工作的关系
```

然后把它们贴给 Agent，只说：

```text
更新到 Paper Recap。
```

项目内的 `$update-paper-recap` skill 会让 Agent 自动完成：核对论文身份、整理成固定结构、创建记录、校验内容并构建网站。完整记录样式见 `content/TEMPLATE.md`，但日常无需手填。

> 注意：GitHub Pages 默认是公开网站。不要提交公司机密、未公开研究、访问凭据或私人敏感信息。

## 第一次发布

1. 在 GitHub 新建一个公开仓库，例如 `paper-recap`。
2. 把本目录推送到仓库的 `main` 分支。
3. 打开仓库 `Settings → Pages`，把 Source 设为 **GitHub Actions**。
4. 首次工作流完成后，网站地址通常是 `https://<你的用户名>.github.io/paper-recap/`。

如果仓库直接命名为 `<你的用户名>.github.io`，网站地址就是 `https://<你的用户名>.github.io/`。

## 本地检查

```bash
python3 scripts/build_site.py
python3 -m http.server 8000 --directory dist
```

打开 `http://localhost:8000`。构建器仅使用 Python 标准库，不需要安装依赖。
