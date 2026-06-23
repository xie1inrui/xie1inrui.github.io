# 个人知识网站长期维护指南

本文档说明如何在 macOS 中找到项目、本地预览、新增内容、调整页面以及维护到 GitHub/GitHub Pages。

## 1. 找到项目

项目路径：

```bash
/Users/xielinrui/knowledge-site
```

用 Finder 打开：

```bash
open /Users/xielinrui/knowledge-site
```

用 VS Code 打开：

```bash
open -a "Visual Studio Code" /Users/xielinrui/knowledge-site
```

进入终端目录：

```bash
cd /Users/xielinrui/knowledge-site
```

## 2. 本地预览

```bash
cd /Users/xielinrui/knowledge-site
npm run dev
```

浏览器打开终端提示的地址，通常是：

```text
http://localhost:4321/
```

部署前检查：

```bash
npm run build
```

预览正式构建结果：

```bash
npm run preview
```

## 3. 新增论文阅读

复制模板：

```bash
cp templates/paper-template.md src/content/papers/my-new-paper.md
```

编辑文件：

```bash
open -a "Visual Studio Code" src/content/papers/my-new-paper.md
```

统一 Frontmatter：

```yaml
---
title: "论文标题"
date: 2026-06-22
category: "LLM Inference"
tags: ["LLM Inference", "AI System"]
summary: "一句话总结这篇论文"
status: "reading"
authors: []
venue: ""
year: 2026
paperUrl: ""
codeUrl: ""
---
```

正文建议保持以下结构：

- 论文信息
- 研究问题
- 核心思想
- 系统架构
- 方法细节
- 实验设置
- 结果分析
- 优缺点
- 我的理解
- 可借鉴点
- 未解决问题

## 4. 新增技术笔记

```bash
cp templates/technical-note-template.md src/content/notes/my-new-note.md
```

编辑后运行：

```bash
npm run build
```

技术笔记建议包含：背景、核心概念、关键代码、命令记录、踩坑记录、总结。

## 5. 新增项目归档

在 `src/content/projects/` 下新建 Markdown 文件，例如：

```bash
src/content/projects/my-project.md
```

Frontmatter 示例：

```yaml
---
title: "项目名称"
date: 2026-06-22
category: "AI infra"
tags: ["AI infra", "LLM Inference"]
summary: "项目摘要"
status: "active"
repoUrl: ""
demoUrl: ""
---
```

## 6. 修改网站信息

网站标题、作者、导航和研究方向在：

```text
src/data/site.ts
```

页面样式主要在：

```text
src/styles/global.css
src/layouts/BaseLayout.astro
src/components/
```

页面路由在：

```text
src/pages/
```

## 7. 维护到 GitHub

如果还没有初始化 Git：

```bash
cd /Users/xielinrui/knowledge-site
git init
git add .
git commit -m "feat: initialize personal knowledge site"
```

在 GitHub 创建一个空仓库，例如 `knowledge-site`，然后：

```bash
git remote add origin https://github.com/xielinrui/knowledge-site.git
git branch -M main
git push -u origin main
```

如果使用 GitHub CLI：

```bash
gh repo create knowledge-site --public --source=. --remote=origin --push
```

## 8. GitHub Pages 部署

项目已经包含：

```text
.github/workflows/deploy.yml
```

进入 GitHub 仓库：

```text
Settings → Pages → Build and deployment → Source: GitHub Actions
```

以后每次更新：

```bash
git add .
git commit -m "docs: add new note"
git push
```

GitHub Actions 会自动构建并部署。

## 9. 推荐长期维护习惯

- 文件名使用日期或主题：`2026-06-vllm-pagedattention.md`
- 每篇内容必须写 summary，方便首页、搜索和图谱聚合。
- tags 使用稳定词表，例如：`LLM Inference`、`Distributed Training`、`AI System`、`AI infra`。
- 每次写完运行 `npm run build`，避免 Frontmatter 或链接错误。
- 每周或每月整理一次 tags，避免同义词过多。
