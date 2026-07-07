#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const ROOT = process.cwd();
const CONTENT_DIR = path.join(ROOT, 'src', 'content');
const GENERATED_DIR = path.join(ROOT, '.generated', 'suggestions');

const args = process.argv.slice(2);
const writeMode = args.includes('--write');
const jsonMode = args.includes('--json');
const noLlm = args.includes('--no-llm');
const targetArgs = args.filter((arg) => !arg.startsWith('--'));

const SECTION_BY_COLLECTION = {
  papers: 'Research',
  notes: 'Notes',
  projects: 'Projects',
  blog: 'Blog',
};

const TYPE_BY_COLLECTION = {
  papers: 'paper',
  notes: 'note',
  projects: 'project',
  blog: 'blog',
};

const ONTOLOGY = [
  {
    tag: 'LLM Inference',
    parents: ['AI Systems', 'Paper Reading'],
    patterns: ['llm inference', 'serving', 'decode', 'prefill', 'kv cache', 'pagedattention', 'continuous batching', 'speculative decoding', 'prefix caching', 'throughput', 'latency'],
    concepts: ['KV Cache', 'PagedAttention', 'Continuous Batching', 'Prefix Caching', 'Speculative Decoding', 'Prefill', 'Decode', 'Serving Runtime', 'Throughput', 'Latency'],
  },
  {
    tag: 'Distributed Training',
    parents: ['AI Systems', 'Paper Reading'],
    patterns: ['distributed training', 'megatron', 'deepspeed', 'zero', 'pipeline parallelism', 'tensor parallelism', 'data parallelism', 'fsdp', 'allreduce', 'nccl'],
    concepts: ['Tensor Parallelism', 'Pipeline Parallelism', 'Data Parallelism', 'ZeRO', 'FSDP', 'NCCL', 'AllReduce', 'Gradient Checkpointing'],
  },
  {
    tag: 'AI Infra',
    parents: ['AI Infrastructure', 'Tools'],
    patterns: ['ai infra', 'infrastructure', 'deployment', 'docker', 'kubernetes', 'slurm', 'monitoring', 'benchmark', 'profiling', 'observability'],
    concepts: ['Docker', 'Kubernetes', 'Slurm', 'Monitoring', 'Benchmarking', 'Profiling', 'Observability', 'Deployment'],
  },
  {
    tag: 'Linux / Server',
    parents: ['Tools', 'Server Tools'],
    patterns: ['linux', 'server', 'ssh', 'tmux', 'conda', 'cuda driver', 'proxy', 'shell', 'bash', 'port', 'process'],
    concepts: ['Linux', 'SSH', 'tmux', 'Conda', 'CUDA Driver', 'Proxy', 'Shell', 'Process Management'],
  },
  {
    tag: 'Knowledge Management',
    parents: ['Tools', 'Knowledge Management'],
    patterns: ['knowledge', 'markdown', 'frontmatter', 'astro', 'content collections', 'github pages', 'zettelkasten', 'note', 'graph'],
    concepts: ['Markdown Workflow', 'Frontmatter', 'Content Collections', 'Static Generation', 'GitHub Pages', 'Knowledge Graph', 'Knowledge Archive'],
  },
  {
    tag: 'Personal Website',
    parents: ['Website Updates', 'Reflection'],
    patterns: ['personal website', 'homepage', 'cv', 'portfolio', 'blog', 'github pages', 'astro'],
    concepts: ['Personal Homepage', 'Portfolio', 'CV', 'Blog', 'GitHub Pages'],
  },
];

function walkMarkdown(dir) {
  if (!fs.existsSync(dir)) return [];
  const out = [];
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) out.push(...walkMarkdown(full));
    else if (name.endsWith('.md') || name.endsWith('.mdx')) out.push(full);
  }
  return out;
}

function parseFrontmatter(raw) {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!match) return { frontmatterRaw: '', body: raw, data: {}, hasFrontmatter: false };
  const frontmatterRaw = match[1];
  const data = {};
  for (const line of frontmatterRaw.split('\n')) {
    if (!line.trim() || /^\s/.test(line)) continue;
    const m = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!m) continue;
    const key = m[1];
    let value = m[2].trim();
    if (value === '') { data[key] = ''; continue; }
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) value = value.slice(1, -1);
    else if (value.startsWith('[') && value.endsWith(']')) {
      try { value = JSON.parse(value.replace(/'/g, '"')); }
      catch { value = value.slice(1, -1).split(',').map((x) => x.trim().replace(/^['"]|['"]$/g, '')).filter(Boolean); }
    }
    data[key] = value;
  }
  return { frontmatterRaw, body: raw.slice(match[0].length), data, hasFrontmatter: true };
}

function stringifyYamlValue(value) {
  if (Array.isArray(value)) return `[${value.map((x) => JSON.stringify(x)).join(', ')}]`;
  if (typeof value === 'number') return String(value);
  return JSON.stringify(String(value));
}

function replaceOrInsertField(frontmatterRaw, key, value) {
  const line = `${key}: ${stringifyYamlValue(value)}`;
  const re = new RegExp(`^${key}:.*$`, 'm');
  if (re.test(frontmatterRaw)) return frontmatterRaw.replace(re, line);
  const preferredOrder = ['title', 'date', 'type', 'section', 'category', 'tags', 'concepts', 'parents', 'projects', 'papers', 'related', 'summary', 'status'];
  const currentLines = frontmatterRaw.split('\n');
  const keyIndex = preferredOrder.indexOf(key);
  for (let i = preferredOrder.length - 1; i >= 0; i--) {
    const beforeKey = preferredOrder[i];
    if (i < keyIndex) {
      const idx = currentLines.findIndex((l) => l.startsWith(`${beforeKey}:`));
      if (idx !== -1) {
        currentLines.splice(idx + 1, 0, line);
        return currentLines.join('\n');
      }
    }
  }
  currentLines.push(line);
  return currentLines.join('\n');
}

function mergeUnique(...arrays) {
  const out = [];
  const seen = new Set();
  for (const arr of arrays) for (const item of arr || []) {
    if (!item || seen.has(item)) continue;
    seen.add(item);
    out.push(item);
  }
  return out;
}

function collectionForFile(file) {
  const rel = path.relative(CONTENT_DIR, file).split(path.sep);
  return rel[0];
}

function slugForFile(file) {
  const rel = path.relative(CONTENT_DIR, file).replace(/\\/g, '/').replace(/\.(md|mdx)$/, '');
  const [collection, ...rest] = rel.split('/');
  return `${collection}:${rest.join('/')}`;
}

function contentUrlForFile(file) {
  const collection = collectionForFile(file);
  const slug = path.relative(path.join(CONTENT_DIR, collection), file).replace(/\\/g, '/').replace(/\.(md|mdx)$/, '');
  if (collection === 'papers') return `/research/${slug}/`;
  return `/${collection}/${slug}/`;
}

function scoreText(text, terms) {
  const lower = text.toLowerCase();
  let score = 0;
  for (const term of terms) {
    const t = term.toLowerCase();
    const re = new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
    const matches = lower.match(re);
    if (matches) score += matches.length;
  }
  return score;
}

function inferByRules(file, parsed, allDocs) {
  const collection = collectionForFile(file);
  const title = String(parsed.data.title || path.basename(file));
  const text = `${title}\n${parsed.body}`;
  const existingTags = Array.isArray(parsed.data.tags) ? parsed.data.tags : [];
  const existingConcepts = Array.isArray(parsed.data.concepts) ? parsed.data.concepts : [];
  const tagScores = [];
  const concepts = [];
  const parents = [];

  for (const item of ONTOLOGY) {
    const score = scoreText(text, [...item.patterns, item.tag, ...item.concepts]);
    if (score > 0) {
      tagScores.push([item.tag, score]);
      parents.push(...item.parents);
      for (const concept of item.concepts) {
        if (scoreText(text, [concept]) > 0) concepts.push(concept);
      }
    }
  }

  const category = String(parsed.data.category || '');
  if (category) {
    const matched = ONTOLOGY.find((item) => item.tag.toLowerCase() === category.toLowerCase() || item.patterns.includes(category.toLowerCase()));
    if (matched) tagScores.push([matched.tag, 2]);
  }

  const suggestedTags = mergeUnique(existingTags, tagScores.sort((a, b) => b[1] - a[1]).map(([tag]) => tag)).slice(0, 8);
  const suggestedConcepts = mergeUnique(existingConcepts, concepts).slice(0, 12);
  const suggestedParents = mergeUnique(Array.isArray(parsed.data.parents) ? parsed.data.parents : [], parents).slice(0, 6);

  const thisSlug = slugForFile(file);
  const related = allDocs
    .filter((doc) => doc.file !== file)
    .map((doc) => {
      const sharedTags = intersection(suggestedTags, doc.tags);
      const sharedConcepts = intersection(suggestedConcepts, doc.concepts);
      const sameCategory = category && category === doc.category ? 1 : 0;
      const score = sharedTags.length * 2 + sharedConcepts.length * 3 + sameCategory;
      return { ...doc, score, sharedTags, sharedConcepts };
    })
    .filter((doc) => doc.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map((doc) => doc.graphId);

  const projects = allDocs
    .filter((doc) => doc.collection === 'projects')
    .map((doc) => ({ ...doc, score: intersection(suggestedTags, doc.tags).length * 2 + intersection(suggestedConcepts, doc.concepts).length * 3 }))
    .filter((doc) => doc.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map((doc) => doc.title);

  return {
    type: parsed.data.type || TYPE_BY_COLLECTION[collection] || 'note',
    section: parsed.data.section || SECTION_BY_COLLECTION[collection] || 'Notes',
    category: parsed.data.category || suggestedTags[0] || 'General',
    tags: suggestedTags,
    concepts: suggestedConcepts,
    parents: suggestedParents,
    projects: mergeUnique(Array.isArray(parsed.data.projects) ? parsed.data.projects : [], projects),
    related: mergeUnique(Array.isArray(parsed.data.related) ? parsed.data.related : [], related.filter((x) => x !== thisSlug)),
    confidence: tagScores.length ? 'medium' : 'low',
    source: 'rules',
  };
}

function intersection(a, b) {
  const bs = new Set(b || []);
  return (a || []).filter((x) => bs.has(x));
}

function buildDocIndex(files) {
  return files.map((file) => {
    const parsed = parseFrontmatter(fs.readFileSync(file, 'utf8'));
    return {
      file,
      collection: collectionForFile(file),
      graphId: slugForFile(file),
      url: contentUrlForFile(file),
      title: String(parsed.data.title || path.basename(file)),
      category: String(parsed.data.category || ''),
      tags: Array.isArray(parsed.data.tags) ? parsed.data.tags : [],
      concepts: Array.isArray(parsed.data.concepts) ? parsed.data.concepts : [],
      summary: String(parsed.data.summary || ''),
    };
  });
}

async function inferByLlm(parsed, allDocs, fallback) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey || noLlm) return fallback;
  const candidates = {
    knownTags: [...new Set(ONTOLOGY.map((x) => x.tag).concat(allDocs.flatMap((d) => d.tags)))].sort(),
    knownConcepts: [...new Set(ONTOLOGY.flatMap((x) => x.concepts).concat(allDocs.flatMap((d) => d.concepts)))].sort(),
    existingPages: allDocs.map((d) => ({ id: d.graphId, title: d.title, category: d.category, tags: d.tags, concepts: d.concepts })).slice(0, 80),
  };
  const prompt = `You maintain a personal AI systems knowledge website. Analyze the Markdown document and suggest metadata for a semantic knowledge graph. Return strict JSON only with keys: tags, concepts, parents, projects, related, category, confidence. Use concise technical labels. Prefer existing candidate labels when appropriate. related must use existing page ids like notes/foo or papers/bar without content: prefix.\n\nCandidate data:\n${JSON.stringify(candidates, null, 2)}\n\nCurrent frontmatter:\n${JSON.stringify(parsed.data, null, 2)}\n\nMarkdown body:\n${parsed.body.slice(0, 12000)}`;

  try {
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
        temperature: 0.1,
        messages: [
          { role: 'system', content: 'You are a precise metadata extraction engine. Return valid JSON only.' },
          { role: 'user', content: prompt },
        ],
      }),
    });
    if (!res.ok) throw new Error(`OpenAI API ${res.status}: ${await res.text()}`);
    const data = await res.json();
    const content = data.choices?.[0]?.message?.content || '{}';
    const json = JSON.parse(content.replace(/^```json\s*/i, '').replace(/```$/,'').trim());
    return {
      ...fallback,
      category: json.category || fallback.category,
      tags: mergeUnique(Array.isArray(parsed.data.tags) ? parsed.data.tags : [], json.tags || [], fallback.tags).slice(0, 10),
      concepts: mergeUnique(Array.isArray(parsed.data.concepts) ? parsed.data.concepts : [], json.concepts || [], fallback.concepts).slice(0, 15),
      parents: mergeUnique(Array.isArray(parsed.data.parents) ? parsed.data.parents : [], json.parents || [], fallback.parents).slice(0, 8),
      projects: mergeUnique(Array.isArray(parsed.data.projects) ? parsed.data.projects : [], json.projects || [], fallback.projects).slice(0, 5),
      related: mergeUnique(Array.isArray(parsed.data.related) ? parsed.data.related : [], json.related || [], fallback.related).slice(0, 8),
      confidence: json.confidence || 'medium',
      source: 'llm',
    };
  } catch (error) {
    return { ...fallback, llmError: String(error.message || error) };
  }
}

function applySuggestions(parsed, suggestion) {
  let fm = parsed.frontmatterRaw;
  for (const key of ['type', 'section', 'category', 'tags', 'concepts', 'parents', 'projects', 'related']) {
    if (suggestion[key] === undefined) continue;
    if (Array.isArray(suggestion[key]) && suggestion[key].length === 0) continue;
    fm = replaceOrInsertField(fm, key, suggestion[key]);
  }
  return `---\n${fm}\n---\n${parsed.body}`;
}

function resolveTargets() {
  if (targetArgs.length === 0) return walkMarkdown(CONTENT_DIR);
  return targetArgs.flatMap((arg) => {
    const full = path.resolve(ROOT, arg);
    if (!fs.existsSync(full)) throw new Error(`Target not found: ${arg}`);
    if (fs.statSync(full).isDirectory()) return walkMarkdown(full);
    return [full];
  });
}

async function main() {
  const files = resolveTargets();
  const allDocs = buildDocIndex(walkMarkdown(CONTENT_DIR));
  fs.mkdirSync(GENERATED_DIR, { recursive: true });
  const results = [];

  for (const file of files) {
    const raw = fs.readFileSync(file, 'utf8');
    const parsed = parseFrontmatter(raw);
    if (!parsed.hasFrontmatter) {
      results.push({ file: path.relative(ROOT, file), error: 'Missing frontmatter' });
      continue;
    }
    const fallback = inferByRules(file, parsed, allDocs);
    const suggestion = await inferByLlm(parsed, allDocs, fallback);
    const rel = path.relative(ROOT, file);
    const result = { file: rel, url: contentUrlForFile(file), suggestion };
    results.push(result);

    const outFile = path.join(GENERATED_DIR, rel.replace(/[\\/]/g, '__').replace(/\.(md|mdx)$/, '.json'));
    fs.writeFileSync(outFile, JSON.stringify(result, null, 2));

    if (writeMode) fs.writeFileSync(file, applySuggestions(parsed, suggestion));
  }

  if (jsonMode) console.log(JSON.stringify(results, null, 2));
  else {
    for (const result of results) {
      console.log(`\n${result.file}`);
      if (result.error) { console.log(`  ERROR: ${result.error}`); continue; }
      console.log(`  source: ${result.suggestion.source}${result.suggestion.llmError ? ` (LLM fallback: ${result.suggestion.llmError})` : ''}`);
      console.log(`  category: ${result.suggestion.category}`);
      console.log(`  tags: ${result.suggestion.tags.join(', ') || '(none)'}`);
      console.log(`  concepts: ${result.suggestion.concepts.join(', ') || '(none)'}`);
      console.log(`  parents: ${result.suggestion.parents.join(', ') || '(none)'}`);
      console.log(`  projects: ${result.suggestion.projects.join(', ') || '(none)'}`);
      console.log(`  related: ${result.suggestion.related.join(', ') || '(none)'}`);
    }
    console.log(`\nSuggestions written to ${path.relative(ROOT, GENERATED_DIR)}/`);
    if (!writeMode) console.log('Run with --write to merge suggestions into Markdown frontmatter.');
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
