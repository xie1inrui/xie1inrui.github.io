# 网站整体架构设计

## 定位
面向计算机科研人员的个人知识网站，服务论文阅读、技术学习笔记、项目归档三类长期内容。

## 技术栈
- Astro：静态生成，适合 GitHub Pages。
- TypeScript：内容和组件类型约束。
- TailwindCSS：极简响应式样式。
- Markdown / MDX：内容源文件。
- Astro Content Collections：统一 Frontmatter schema。

## 目录结构
```text
knowledge-site/
├── .github/workflows/deploy.yml
├── docs/architecture.md
├── templates/
├── src/
│   ├── components/
│   ├── content/{config.ts,papers,notes,projects}
│   ├── data/site.ts
│   ├── layouts/
│   ├── pages/
│   ├── styles/global.css
│   └── utils/content.ts
└── astro.config.mjs
```

## 页面规划
- 首页：研究方向、最近更新、快速搜索、标签入口。
- 论文阅读：论文列表和论文详情模板。
- 技术笔记：技术笔记列表和详情模板。
- 项目归档：项目列表和详情模板。
- 关于我：研究方向和维护原则。
- 搜索：静态 JSON 索引驱动的客户端搜索。
- 标签 / 分类：自动生成聚合页。
- 知识图谱：根据标签连接内容页面，可点击跳转。

## GitHub Pages 部署方案
`.github/workflows/deploy.yml` 使用官方 Pages Actions：checkout → setup-node → npm ci → npm run build → upload-pages-artifact → deploy-pages。仓库 Settings → Pages 选择 GitHub Actions。

## 内容维护流程
1. 从 `templates/` 复制模板到 `src/content/papers|notes|projects/`。
2. 填写统一 Frontmatter。
3. 本地运行 `npm run build` 检查 schema 与链接。
4. 推送 main 自动部署。
