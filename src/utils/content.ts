import { getCollection, type CollectionEntry } from 'astro:content';
export type KnowledgeEntry = CollectionEntry<'papers'> | CollectionEntry<'notes'> | CollectionEntry<'projects'>;
export const collectionMeta = { papers: { label: '论文阅读', href: '/papers/' }, notes: { label: '技术笔记', href: '/notes/' }, projects: { label: '项目归档', href: '/projects/' } } as const;
export function entryUrl(entry: KnowledgeEntry) { return `/${entry.collection}/${entry.slug}/`; }
export function formatDate(date: Date) { return new Intl.DateTimeFormat('zh-CN', { year:'numeric', month:'2-digit', day:'2-digit' }).format(date); }
export function normalizeSlug(value: string) { return encodeURIComponent(value.toLowerCase().replaceAll(' ', '-')); }
export async function getAllEntries() { const [papers, notes, projects] = await Promise.all([getCollection('papers'), getCollection('notes'), getCollection('projects')]); return [...papers, ...notes, ...projects].sort((a,b)=>b.data.date.valueOf()-a.data.date.valueOf()); }
export function getAllTags(entries: KnowledgeEntry[]) { const counts = new Map<string, number>(); for (const entry of entries) for (const tag of entry.data.tags) counts.set(tag, (counts.get(tag) ?? 0)+1); return [...counts.entries()].sort((a,b)=>a[0].localeCompare(b[0])); }
export function getAllCategories(entries: KnowledgeEntry[]) { const counts = new Map<string, number>(); for (const entry of entries) counts.set(entry.data.category, (counts.get(entry.data.category) ?? 0)+1); return [...counts.entries()].sort((a,b)=>a[0].localeCompare(b[0])); }
