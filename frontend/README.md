# set.log — AI Engineering Notebook

A personal technical blog / knowledge base built with **React + TypeScript + Vite + Tailwind CSS**.
Content lives as plain Markdown files with frontmatter in `/posts`, and the app renders them into
a linked knowledge base with a table of contents, related notes, backlinks, and an interactive graph view.

## Getting started

```bash
npm install
npm run dev
```

Open http://localhost:5173

Build for production:

```bash
npm run build
npm run preview
```

## Adding a new note

Create a new file in `/posts/your-slug.md`:

```md
---
title: "Your Title"
description: "One-line summary."
date: "2026-09-27"
updated: "2026-09-27"
category: "AI"
tags: ["RAG", "LLM"]
status: "published"
featured: false
readingTime: 6
---

## A heading

Your content here. Use `[[Another Post Title]]` to link to another note by title —
it resolves automatically to `/article/<slug>` and powers backlinks and related notes.
```

Posts are loaded automatically via `import.meta.glob` — no registry file to update.

## Project structure

```
posts/              Markdown content (frontmatter + body)
src/
  lib/
    frontmatter.ts  Minimal YAML frontmatter parser
    posts.ts        Loads posts, computes related/backlinks, topic & project data
  components/       Nav, ThemeToggle, SearchModal, PostCard, Terminal, Mermaid
  pages/            Home, Blog, Article, Topics, GraphPage, Projects, About
  index.css         Theme tokens (light / dark / warm) + article typography
```

## Features

- 3 persisted themes: light / dark / warm (`localStorage`)
- Markdown rendering with GFM tables, LaTeX (`rehype-katex`), syntax highlighting
  (`rehype-highlight`), and Mermaid diagrams (fenced ```mermaid blocks)
- `[[Wikilink]]` internal linking with automatic backlink detection
- Sticky, scroll-spy table of contents generated from `##` headings
- Global search (`Ctrl/Cmd + K`)
- Interactive knowledge graph (`/graph`) with zoom, pan, search, category filter and node preview

## Next steps you may want

- Swap the in-repo `/posts` folder for a headless CMS or a Git-based content API
- Add pagination on `/blog` if the post count grows large
- Wire `rehype-raw` if you need raw HTML inside Markdown
