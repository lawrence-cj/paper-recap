# Paper Recap

[![Deploy Paper Recap](https://github.com/lawrence-cj/paper-recap/actions/workflows/pages.yml/badge.svg)](https://github.com/lawrence-cj/paper-recap/actions/workflows/pages.yml)
[![Live Site](https://img.shields.io/badge/demo-junsongc.top-b14935)](https://junsongc.top/paper-recap/)
[![License: MIT](https://img.shields.io/badge/code_license-MIT-25231f.svg)](LICENSE)

把每天读过的 paper 和与 Agent 讨论后的关键结论，沉淀成一个可搜索、可回顾、可长期维护的个人知识网站。

Paper Recap 是一个零前端构建依赖的静态站点：每篇论文是一份 Markdown，Python 脚本负责校验和构建，GitHub Actions 自动发布到 GitHub Pages。它特别支持 LaTeX 公式、论文插图、同日精确排序，以及可跨机器安装的 Codex skill。

**在线示例：** <https://junsongc.top/paper-recap/>

## 功能

- 每篇论文使用独立 Markdown 文件，内容可读、可迁移、可通过 Git 管理。
- 支持标题、作者、标签、正文全文搜索和主题筛选。
- 支持“最近阅读”和“标题 A–Z”排序；同一天按 `read_at` 精确时间排序。
- 使用 KaTeX 渲染行内公式 `$...$` 和独立公式 `$$...$$`。
- 支持响应式论文图片、图注、懒加载和点击放大。
- 明暗主题、移动端布局和论文详情弹窗。
- 构建时自动校验元数据、章节、公式分隔符和图片路径。
- 推送 `main` 后由 GitHub Actions 自动部署到 GitHub Pages。
- 内置 `update-paper-recap` Codex skill，可把零散笔记或 Agent 对话整理并发布。
- 页脚可选隐私友好的总访问次数统计。

## 快速开始

环境要求：Git 和 Python 3。构建器只使用 Python 标准库，不需要 Node.js、npm 或额外 Python 包。

```bash
git clone https://github.com/lawrence-cj/paper-recap.git
cd paper-recap
python3 scripts/build_site.py
python3 -m http.server 8000 --directory dist
```

浏览器打开 <http://localhost:8000>。

> 公式组件从 jsDelivr 加载 KaTeX，因此本地预览公式时需要网络连接。论文内容和站点构建本身不依赖外部包。

## 创建自己的 Paper Recap

1. Fork 本仓库，或 clone 后推送到自己的 GitHub 仓库。
2. 删除示例论文，保留或复制 `content/TEMPLATE.md` 创建自己的记录。
3. 修改 `index.html` 中的站点标题、作者和版权信息。
4. 把 `.codex/skills/update-paper-recap/SKILL.md` 中的默认仓库地址改成自己的仓库。
5. 运行 `python3 scripts/build_site.py`，确认构建通过。
6. 在 GitHub 仓库中打开 `Settings → Pages`，将 Source 设置为 **GitHub Actions**。
7. 推送到 `main`；工作流完成后即可访问网站。

项目仓库的默认地址通常为：

```text
https://<username>.github.io/<repository>/
```

如果仓库名是 `<username>.github.io`，地址通常是：

```text
https://<username>.github.io/
```

### 修改访问计数器

示例站点在 `index.html` 中使用 `hits.sh` 统计页面加载次数。Fork 后请把两处：

```text
junsongc.top/paper-recap
```

替换为自己的域名或 GitHub Pages 路径，否则访问会被计入示例站点。若不需要外部访问统计，直接删除 `.visit-counter` 链接即可。

该计数器统计页面加载次数，不是严格去重访客；点击页脚 badge 可查看公开的周、月和累计数据。

## 添加一篇论文

复制模板并使用日期加稳定英文 slug 命名：

```bash
cp content/TEMPLATE.md content/papers/2026-08-03-example-paper.md
```

Frontmatter 示例：

```yaml
---
title: "Paper Title"
paper_url: "https://arxiv.org/abs/xxxx.xxxxx"
authors: "First Author et al."
venue: "Conference / Journal / arXiv"
published: "2026"
read_date: "2026-08-03"
read_at: "2026-08-03T22:35:13+08:00"
status: "已精读"
tags: ["Video Generation", "Diffusion Models"]
one_liner: "半年后只看这一句，也能恢复论文最重要的机制。"
---
```

正文必须保留以下顶级章节，并按此顺序组织：

1. `研究问题`
2. `核心方法`
3. `关键发现`
4. `我的提问`
5. `局限与疑问`
6. `我的判断`
7. `下次只看这些`

完整示例见 [`content/TEMPLATE.md`](content/TEMPLATE.md)。写完后运行：

```bash
python3 scripts/build_site.py
```

构建成功后会在 `dist/` 生成可直接部署的静态网站。不要手动修改 `dist/` 或 `assets/papers.js`，它们由构建脚本管理。

## 公式

行内公式：

```markdown
噪声预测器记为 $\epsilon_\theta(x_t,t)$。
```

独立公式：

```markdown
$$
\mathcal{L}_{\mathrm{simple}}
= \mathbb{E}_{t,x_0,\epsilon}
\left[\lVert \epsilon-\epsilon_\theta(x_t,t) \rVert_2^2\right].
$$
```

公式应保留为可编辑 LaTeX，不要使用截图替代。建议使用 KaTeX 支持的标准命令，并在公式附近解释符号和作用。

## 图片

每篇论文的图片放在自己的媒体目录中：

```text
content/
├── papers/
│   └── 2026-08-03-example-paper.md
└── media/
    └── example-paper/
        ├── method-overview.webp
        └── qualitative-results.webp
```

Markdown 写法：

```markdown
![方法总览](media/example-paper/method-overview.webp "论文 Figure 2：方法总览。来源：作者论文，CC BY 4.0。")
```

图片要求：

- 只使用相对于 `content/` 的 `media/<slug>/...` 路径。
- 必须提供有意义的 alt text、图号、来源和许可说明。
- 支持 PNG、JPEG 和 WebP；推荐 WebP。
- 每张图片不得超过 2 MiB。
- 通常只保留一张方法总览和一张决定性结果图。
- 网站是公开的；许可不明确时不要复制图片，改为链接原始来源。

## 使用 Codex skill：最简流程

项目自带 `.codex/skills/update-paper-recap/`。它会指导 Agent：

- 先按标题、URL、DOI、arXiv ID、作者和 slug 查重。
- 已存在的论文合并到原记录，禁止创建重复页面。
- 保留关键公式、用户判断和真正改变理解的问答。
- 新论文写入精确 `read_at`，保证同日排序正确。
- 在许可允许时加入一至两张关键论文图片。
- 校验、提交并推送到 GitHub Pages。

### 注册 skill

在仓库根目录运行：

```bash
python3 scripts/install_skill.py
```

如果目标位置已有旧注册：

```bash
python3 scripts/install_skill.py --replace
```

脚本会把仓库内的 skill 符号链接到：

```text
$CODEX_HOME/skills/update-paper-recap
```

未设置 `CODEX_HOME` 时使用：

```text
~/.codex/skills/update-paper-recap
```

之后在 Codex 会话中直接说：

```text
使用 $update-paper-recap，把下面的总结更新到网站：

论文：标题或链接
我的理解：随手记录的要点
重要问答：真正改变理解的 Agent Q&A
我的判断：是否可信、是否值得复现、与当前工作的关系
```

输入不需要提前整理成模板。Agent 会在信息不足时使用 `待补充`，而不是编造事实。

### 多台机器或集群

每台机器 clone 同一个仓库并执行一次安装：

```bash
git clone git@github.com:<username>/<repository>.git
cd <repository>
python3 scripts/install_skill.py
```

skill 使用符号链接，因此以后只需：

```bash
git pull
```

新会话就会读取仓库中的最新版 skill。多台机器并发更新时，工作流会先同步 `origin/main`；若同时修改同一篇论文，应停止自动解决冲突并人工合并双方判断。

## 项目结构

```text
paper-recap/
├── .codex/skills/update-paper-recap/   # 可移植的 Agent 工作流
├── .github/workflows/pages.yml          # GitHub Pages 自动部署
├── assets/
│   ├── app.js                           # 搜索、筛选、Markdown 与交互
│   └── styles.css                       # 视觉与响应式样式
├── content/
│   ├── papers/                          # 一篇论文一份 Markdown
│   ├── media/                           # 每篇论文自己的图片
│   └── TEMPLATE.md                      # 新记录模板
├── scripts/
│   ├── build_site.py                    # 校验并构建 dist/
│   └── install_skill.py                 # 注册 Codex skill
├── index.html                           # 网站页面
├── 404.html
├── COPYRIGHT                            # 内容版权说明
└── LICENSE                              # 程序代码 MIT License
```

## 自动部署

`.github/workflows/pages.yml` 在每次推送 `main` 时执行：

```text
checkout → Python 校验与构建 → 上传 dist → GitHub Pages 部署
```

任何论文格式、公式分隔符或图片校验错误都会阻止部署，避免损坏线上网站。

## 隐私与公开内容

GitHub Pages 默认是公开网站。不要提交：

- 公司机密或内部项目细节；
- 未公开论文、评审材料或实验结果；
- API key、访问令牌和账号信息；
- 个人敏感信息；
- 没有公开复用许可的论文图片。

## 贡献

欢迎通过 Issue 或 Pull Request 提交通用功能改进，例如 Markdown 渲染、可访问性、移动端体验、校验器和部署流程。

请不要在贡献中提交私人论文笔记、受限材料或未经许可的论文图片。提交前运行：

```bash
python3 scripts/build_site.py
git diff --check
```

## License 与内容版权

程序源代码使用 [MIT License](LICENSE)，版权归 Junsong Chen。

`content/papers/` 中的个人阅读笔记、网站原创文字和原创媒体不随 MIT License 授权，除非文件中另有说明；其版权归 Junsong Chen。论文标题、书目信息、公式、引用和论文图片仍属于各自作者或出版方。详见 [`COPYRIGHT`](COPYRIGHT)。
