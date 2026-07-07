import { getAllEntries, entryUrl } from '@/utils/content';

export async function GET() {
  const entries = await getAllEntries();
  return new Response(JSON.stringify(entries.map((entry)=>({
    title: entry.data.title,
    summary: entry.data.summary,
    category: entry.data.category,
    tags: entry.data.tags,
    concepts: entry.data.concepts,
    status: entry.data.status,
    collection: entry.collection,
    url: entryUrl(entry)
  }))), { headers: { 'Content-Type': 'application/json; charset=utf-8' } });
}
