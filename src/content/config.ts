import { defineCollection, z } from 'astro:content';
const commonSchema = z.object({ title: z.string(), date: z.coerce.date(), category: z.string(), tags: z.array(z.string()).default([]), summary: z.string(), status: z.enum(['idea','reading','draft','active','stable','archived']).default('draft') });
const papers = defineCollection({ type: 'content', schema: commonSchema.extend({ authors: z.array(z.string()).optional(), venue: z.string().optional(), year: z.number().optional(), paperUrl: z.string().url().optional(), codeUrl: z.string().url().optional() }) });
const notes = defineCollection({ type: 'content', schema: commonSchema });
const projects = defineCollection({ type: 'content', schema: commonSchema.extend({ repoUrl: z.string().url().optional(), demoUrl: z.string().url().optional() }) });
export const collections = { papers, notes, projects };
