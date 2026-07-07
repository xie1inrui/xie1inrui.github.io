# Auto Metadata Suggestions

This project includes a local metadata suggestion tool for new Markdown documents.

It helps infer graph-aware frontmatter fields:

```yaml
tags:
concepts:
parents:
projects:
related:
```

## How it works

The tool uses two modes:

1. **LLM mode** if `OPENAI_API_KEY` is available.
2. **Local rule mode** if no API key is configured.

Local rule mode uses a maintainable AI systems ontology for topics such as:

- LLM Inference
- Distributed Training
- AI Infra
- Linux / Server
- Knowledge Management
- Personal Website

It also compares the new document with existing Markdown files to suggest `related` pages and project links.

## Preview suggestions without changing files

Run on all content:

```bash
npm run suggest-tags
```

Run on one file:

```bash
npm run suggest-tags -- src/content/papers/vllm-pagedattention.md
```

Force local rule mode:

```bash
npm run suggest-tags -- --no-llm src/content/notes/my-note.md
```

JSON output:

```bash
npm run suggest-tags -- --json src/content/notes/my-note.md
```

Suggestions are written to:

```text
.generated/suggestions/
```

## Write suggestions back into Markdown

```bash
npm run suggest-tags:write -- src/content/notes/my-note.md
```

Always review the git diff afterwards:

```bash
git diff src/content/notes/my-note.md
```

## Optional LLM setup

To enable LLM mode locally:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_MODEL="gpt-4o-mini"
npm run suggest-tags -- src/content/notes/my-note.md
```

If the API call fails, the tool falls back to local rule mode and reports the LLM error in the output.

## Recommended workflow

1. Add a new Markdown file from a template.
2. Write `title`, `date`, `category`, `summary`, and body content.
3. Run:

```bash
npm run suggest-tags -- path/to/file.md
```

4. Review `.generated/suggestions/...json`.
5. If acceptable, run:

```bash
npm run suggest-tags:write -- path/to/file.md
```

6. Review diff and build:

```bash
npm run build
```
