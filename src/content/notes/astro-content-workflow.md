---
title: "Astro Markdown 知识库工作流"
date: 2026-06-22
category: "AI infra"
tags: ["Astro", "Markdown", "Knowledge Management"]
summary: "用 Astro Content Collections 约束 Frontmatter，使长期维护的 Markdown 内容具备类型安全和可索引性。"
status: "stable"
---

## 背景

个人知识网站需要长期积累，最重要的是内容结构稳定、迁移成本低、构建链路简单。

## 核心概念

- Content Collections：为 Markdown/MDX 定义 schema。
- Static Generation：适合 GitHub Pages。
- Frontmatter：承担分类、标签、状态和摘要的索引职责。

## 关键代码

```ts
const commonSchema = z.object({
  title: z.string(),
  date: z.coerce.date(),
  category: z.string(),
  tags: z.array(z.string()),
  summary: z.string(),
  status: z.enum(['idea', 'reading', 'draft', 'active', 'stable', 'archived'])
});
```

## 命令记录

```bash
npm install
npm run dev
npm run build
```

## 踩坑记录

- GitHub Pages 项目站点需要正确设置 `base`。
- 搜索索引与知识图谱链接使用 `import.meta.env.BASE_URL` 适配项目站点。

## 总结

Markdown 是源数据，Astro 负责生成索引页、详情页、标签页和图谱页。
