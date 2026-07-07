import { defineCollection, z } from 'astro:content';

const graphRelationSchema = z.object({
  relation: z.enum(['related_to', 'depends_on', 'references', 'applies_to', 'explains', 'mentions']).default('related_to'),
  target: z.string()
});

const commonSchema = z.object({
  title: z.string(),
  date: z.coerce.date(),
  type: z.enum(['paper', 'note', 'project', 'blog']).optional(),
  section: z.enum(['Research', 'Notes', 'Projects', 'Blog', 'About']).optional(),
  category: z.string(),
  tags: z.array(z.string()).default([]),
  concepts: z.array(z.string()).default([]),
  related: z.array(z.string()).default([]),
  parents: z.array(z.string()).default([]),
  projects: z.array(z.string()).default([]),
  papers: z.array(z.string()).default([]),
  graph: z.array(graphRelationSchema).default([]),
  summary: z.string(),
  status: z.enum(['idea','reading','draft','active','stable','archived']).default('draft')
});

const papers = defineCollection({
  type: 'content',
  schema: commonSchema.extend({
    type: z.literal('paper').default('paper'),
    section: z.literal('Research').default('Research'),
    authors: z.array(z.string()).optional(),
    venue: z.string().optional(),
    year: z.number().optional(),
    paperUrl: z.string().url().optional(),
    codeUrl: z.string().url().optional()
  })
});

const notes = defineCollection({
  type: 'content',
  schema: commonSchema.extend({
    type: z.literal('note').default('note'),
    section: z.literal('Notes').default('Notes')
  })
});

const blog = defineCollection({
  type: 'content',
  schema: commonSchema.extend({
    type: z.literal('blog').default('blog'),
    section: z.literal('Blog').default('Blog')
  })
});

const projects = defineCollection({
  type: 'content',
  schema: commonSchema.extend({
    type: z.literal('project').default('project'),
    section: z.literal('Projects').default('Projects'),
    repoUrl: z.string().url().optional(),
    demoUrl: z.string().url().optional()
  })
});

export const collections = { papers, notes, blog, projects };
