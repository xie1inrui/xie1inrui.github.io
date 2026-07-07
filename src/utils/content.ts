import { getCollection, type CollectionEntry } from 'astro:content';

export type KnowledgeEntry =
  | CollectionEntry<'papers'>
  | CollectionEntry<'blog'>
  | CollectionEntry<'notes'>
  | CollectionEntry<'projects'>;

export type GraphNodeType = 'section' | 'category' | 'content' | 'concept' | 'tag' | 'project';
export type GraphLinkType = 'belongs_to' | 'categorized_as' | 'tagged_as' | 'mentions' | 'related_to' | 'depends_on' | 'references' | 'applies_to' | 'explains';

export type KnowledgeGraphNode = {
  id: string;
  label: string;
  type: GraphNodeType;
  url: string;
  summary: string;
  tags: string[];
};

export type KnowledgeGraphLink = {
  source: string;
  target: string;
  type: GraphLinkType;
};

export const collectionMeta = {
  papers: { label: 'Research', href: '/research/', section: 'Research' },
  blog: { label: 'Blog', href: '/blog/', section: 'Blog' },
  notes: { label: 'Notes', href: '/notes/', section: 'Notes' },
  projects: { label: 'Projects', href: '/projects/', section: 'Projects' }
} as const;

const sectionMeta = {
  Research: { label: 'Research', href: '/research/', summary: 'Paper reading, research interests, and AI systems learning.' },
  Notes: { label: 'Notes', href: '/notes/', summary: 'Technical notes, server tools, and reusable knowledge.' },
  Projects: { label: 'Projects', href: '/projects/', summary: 'Project branches, experiments, logs, and retrospectives.' },
  Blog: { label: 'Blog', href: '/blog/', summary: 'Learning life, internship reflections, and personal writing.' },
  About: { label: 'About', href: '/about/', summary: 'Personal profile, education, experience, and links.' }
} as const;

function uniqueById<T extends { id: string }>(items: T[]) {
  const seen = new Map<string, T>();
  for (const item of items) if (!seen.has(item.id)) seen.set(item.id, item);
  return [...seen.values()];
}

function uniqueLinks(links: KnowledgeGraphLink[]) {
  const seen = new Map<string, KnowledgeGraphLink>();
  for (const link of links) seen.set(`${link.source}->${link.target}:${link.type}`, link);
  return [...seen.values()];
}

function normalizeContentTarget(value: string) {
  if (value.startsWith('content:')) return value;
  const colonMatch = value.match(/^(papers|notes|projects|blog):(.+)$/);
  if (colonMatch) return `content:${colonMatch[1]}:${colonMatch[2]}`;
  const slashMatch = value.match(/^(papers|notes|projects|blog)\/(.+)$/);
  if (slashMatch) return `content:${slashMatch[1]}:${slashMatch[2]}`;
  return `content:${value}`;
}

export function entryUrl(entry: KnowledgeEntry) {
  if (entry.collection === 'papers') return `/research/${entry.slug}/`;
  return `/${entry.collection}/${entry.slug}/`;
}

export function entryGraphId(entry: KnowledgeEntry) {
  return `content:${entry.collection}:${entry.slug}`;
}

export function formatDate(date: Date) {
  return new Intl.DateTimeFormat('en-US', { year:'numeric', month:'short', day:'2-digit' }).format(date);
}

export function normalizeSlug(value: string) {
  return encodeURIComponent(value.toLowerCase().replaceAll(' ', '-'));
}

export async function getAllEntries() {
  const [papers, blog, notes, projects] = await Promise.all([
    getCollection('papers'),
    getCollection('blog'),
    getCollection('notes'),
    getCollection('projects')
  ]);
  return [...papers, ...blog, ...notes, ...projects].sort((a,b)=>b.data.date.valueOf()-a.data.date.valueOf());
}

export function getAllTags(entries: KnowledgeEntry[]) {
  const counts = new Map<string, number>();
  for (const entry of entries) {
    for (const tag of entry.data.tags) counts.set(tag, (counts.get(tag) ?? 0)+1);
    for (const concept of entry.data.concepts ?? []) counts.set(concept, (counts.get(concept) ?? 0)+1);
  }
  return [...counts.entries()].sort((a,b)=>a[0].localeCompare(b[0]));
}

export function getAllCategories(entries: KnowledgeEntry[]) {
  const counts = new Map<string, number>();
  for (const entry of entries) counts.set(entry.data.category, (counts.get(entry.data.category) ?? 0)+1);
  return [...counts.entries()].sort((a,b)=>a[0].localeCompare(b[0]));
}

export function buildKnowledgeGraph(entries: KnowledgeEntry[], limit?: number) {
  const scopedEntries = typeof limit === 'number' ? entries.slice(0, limit) : entries;
  const nodes: KnowledgeGraphNode[] = [];
  const links: KnowledgeGraphLink[] = [];

  for (const [section, meta] of Object.entries(sectionMeta)) {
    nodes.push({
      id: `section:${section}`,
      label: meta.label,
      type: 'section',
      url: meta.href,
      summary: meta.summary,
      tags: []
    });
  }

  for (const entry of scopedEntries) {
    const contentId = entryGraphId(entry);
    const section = entry.data.section ?? collectionMeta[entry.collection].section;
    const categoryId = `category:${entry.data.category}`;

    nodes.push({
      id: contentId,
      label: entry.data.title,
      type: 'content',
      url: entryUrl(entry),
      summary: entry.data.summary,
      tags: entry.data.tags
    });

    nodes.push({
      id: categoryId,
      label: entry.data.category,
      type: 'category',
      url: `/categories/${normalizeSlug(entry.data.category)}/`,
      summary: `Category: ${entry.data.category}`,
      tags: []
    });

    links.push({ source: `section:${section}`, target: contentId, type: 'belongs_to' });
    links.push({ source: categoryId, target: contentId, type: 'categorized_as' });
    links.push({ source: `section:${section}`, target: categoryId, type: 'belongs_to' });

    for (const tag of entry.data.tags) {
      const tagId = `tag:${tag}`;
      nodes.push({ id: tagId, label: tag, type: 'tag', url: `/tags/${normalizeSlug(tag)}/`, summary: `Tag: ${tag}`, tags: [] });
      links.push({ source: contentId, target: tagId, type: 'tagged_as' });
    }

    for (const concept of entry.data.concepts ?? []) {
      const conceptId = `concept:${concept}`;
      nodes.push({ id: conceptId, label: concept, type: 'concept', url: `/tags/${normalizeSlug(concept)}/`, summary: `Concept: ${concept}`, tags: [] });
      links.push({ source: contentId, target: conceptId, type: 'mentions' });
    }

    for (const parent of entry.data.parents ?? []) {
      const parentId = parent.includes('/') ? `category:${parent}` : `category:${parent}`;
      nodes.push({ id: parentId, label: parent, type: 'category', url: `/categories/${normalizeSlug(parent)}/`, summary: `Parent topic: ${parent}`, tags: [] });
      links.push({ source: parentId, target: contentId, type: 'belongs_to' });
    }

    for (const project of entry.data.projects ?? []) {
      const projectId = `project:${project}`;
      nodes.push({ id: projectId, label: project, type: 'project', url: `/projects/${normalizeSlug(project)}/`, summary: `Project: ${project}`, tags: [] });
      links.push({ source: contentId, target: projectId, type: 'applies_to' });
    }

    for (const paper of entry.data.papers ?? []) {
      const paperId = paper.startsWith('content:') ? paper : `content:papers:${paper}`;
      links.push({ source: contentId, target: paperId, type: 'references' });
    }

    for (const related of entry.data.related ?? []) {
      links.push({ source: contentId, target: normalizeContentTarget(related), type: 'related_to' });
    }

    for (const relation of entry.data.graph ?? []) {
      links.push({ source: contentId, target: relation.target, type: relation.relation });
    }
  }

  return { nodes: uniqueById(nodes), links: uniqueLinks(links) };
}
