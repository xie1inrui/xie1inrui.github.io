---
title: "个人知识网站长期维护指南"
date: 2026-06-22
category: "Knowledge Management"
tags: ["Astro", "Markdown", "GitHub Pages", "Knowledge Management"]
summary: "记录如何在 macOS 中找到项目、本地预览、新增内容、修改页面并维护到 GitHub Pages。"
status: "stable"
---

## 找到项目

项目路径：

```bash
/Users/xielinrui/knowledge-site
```

Finder 打开：

```bash
open /Users/xielinrui/knowledge-site
```

VS Code 打开：

```bash
open -a "Visual Studio Code" /Users/xielinrui/knowledge-site
```

进入项目：

```bash
cd /Users/xielinrui/knowledge-site
```

## 本地预览

```bash
npm run dev
```

浏览器打开：

```text
http://localhost:4321/
```

部署前检查：

```bash
npm run build
```

## 新增论文阅读

```bash
cp templates/paper-template.md src/content/papers/my-new-paper.md
```

论文 Frontmatter：

```yaml
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
```

## 新增技术笔记

```bash
cp templates/technical-note-template.md src/content/notes/my-new-note.md
```

建议包含：背景、核心概念、关键代码、命令记录、踩坑记录、总结。

## 新增项目归档

在 `src/content/projects/` 下新建 Markdown 文件，并填写统一 Frontmatter。

## 修改网站信息

- 网站标题、作者、导航、研究方向：`src/data/site.ts`
- 页面样式：`src/styles/global.css`
- 组件：`src/components/`
- 页面：`src/pages/`

## 维护到 GitHub

首次初始化：

```bash
git init
git add .
git commit -m "feat: initialize personal knowledge site"
```

连接远程仓库：

```bash
git remote add origin https://github.com/xielinrui/knowledge-site.git
git branch -M main
git push -u origin main
```

## GitHub Pages 部署

GitHub 仓库中设置：

```text
Settings → Pages → Build and deployment → Source: GitHub Actions
```

后续更新：

```bash
git add .
git commit -m "docs: add new note"
git push
```

## 长期维护建议

- 文件名稳定：`2026-06-topic.md`
- tags 使用稳定词表。
- 每次提交前运行 `npm run build`。
- 定期清理重复标签和过期项目状态。
