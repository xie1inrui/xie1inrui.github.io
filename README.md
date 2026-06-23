# Personal Knowledge Site

一个面向计算机科研人员的长期维护个人知识网站，使用 Astro + TypeScript + TailwindCSS + Markdown/MDX 构建。

## 功能
- 论文阅读、技术笔记、项目归档三类内容集合
- 统一 Frontmatter：`title`、`date`、`category`、`tags`、`summary`、`status`
- 标签页、分类页、全文搜索索引、知识图谱
- 暗色模式、移动端适配、GitHub Pages 自动部署

## 本地开发
```bash
npm install
npm run dev
```

## 构建验证
```bash
npm run build
npm run preview
```

## GitHub Pages 部署
1. 将仓库推送到 GitHub。
2. Settings → Pages → Build and deployment 选择 **GitHub Actions**。
3. 修改 `.github/workflows/deploy.yml` 中的 `SITE_URL` 为你的地址。
4. 推送到 `main` 后自动部署。
