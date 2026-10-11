import { parseFrontmatter, Frontmatter } from "./frontmatter";

export interface Post extends Frontmatter {
  slug: string;
  content: string;
}

// Vite loads every markdown file under /posts as raw text at build time.
const files = import.meta.glob("/posts/*.md", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

function slugify(path: string) {
  return path.split("/").pop()!.replace(/\.md$/, "");
}

export const POSTS: Post[] = Object.entries(files)
  .map(([path, raw]) => {
    const { data, content } = parseFrontmatter(raw);
    return { ...data, slug: slugify(path), content };
  })
  .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

export const TOPICS = [
  { name: "AI", desc: "General artificial intelligence systems, agents, and applied research." },
  { name: "LLM", desc: "Large language models: inference, prompting, evaluation and fine-tuning." },
  { name: "RAG", desc: "Retrieval-augmented generation architectures and design patterns." },
  { name: "RAGAS", desc: "Reference-free evaluation of RAG pipelines using LLM judges." },
  { name: "MLOps", desc: "Deploying, monitoring and maintaining models in production." },
  { name: "Backend", desc: "APIs, services and the systems that sit behind a model." },
  { name: "DevOps", desc: "Infrastructure, CI/CD and operating the stack." },
  { name: "Python", desc: "Language internals and patterns worth remembering." },
  { name: "Developer Tools", desc: "Editors, CLIs and tooling that shapes daily work." },
  { name: "Learning", desc: "Study notes and methods for learning technical material." },
];

export const PROJECTS = [
  { name: "RAGAS SLM Evaluator", desc: "A lightweight evaluator that swaps GPT-4-class judges for a fine-tuned small model, cutting eval cost by ~8x.", stack: "Python · RAGAS · Qwen · vLLM · Hugging Face", status: "Active", posts: ["ragas-evaluators", "rag-eval-pipeline"] },
  { name: "vLLM Deployment Kit", desc: "A reusable Docker + Kubernetes setup for serving open-weight LLMs with autoscaling and warm-pool health checks.", stack: "vLLM · Docker · Kubernetes · FastAPI", status: "Active", posts: ["vllm-production"] },
  { name: "Notebook Graph Engine", desc: "The link-and-backlink engine behind this site — parses frontmatter and wikilinks into a graph.", stack: "Python · Markdown · SQLite", status: "Maintained", posts: ["personal-knowledge-graph"] },
  { name: "Qwen Fine-tune Runs", desc: "A collection of SFT/LoRA experiments across Qwen model sizes, with eval deltas tracked per run.", stack: "PyTorch · PEFT · DeepSpeed", status: "Experimental", posts: ["qwen3-32b-sft"] },
];

export function getPost(slug: string) {
  return POSTS.find((p) => p.slug === slug);
}

export function relatedFor(p: Post, limit = 4) {
  return POSTS.filter((o) => o.slug !== p.slug && (o.category === p.category || o.tags.some((t) => p.tags.includes(t))))
    .sort((a, b) => b.tags.filter((t) => p.tags.includes(t)).length - a.tags.filter((t) => p.tags.includes(t)).length)
    .slice(0, limit);
}

// Backlinks: posts whose content contains a [[Wikilink]] or markdown link to this post's title/slug.
export function backlinksFor(p: Post) {
  return POSTS.filter((o) => o.slug !== p.slug && (
    o.content.includes(`[[${p.title}]]`) || o.content.includes(`(/article/${p.slug})`)
  ));
}

// Replace [[Wikilink Title]] with a markdown link to the matching post, resolved by title.
export function resolveWikilinks(content: string): string {
  return content.replace(/\[\[([^\]]+)\]\]/g, (_m, title) => {
    const target = POSTS.find((p) => p.title.toLowerCase() === title.trim().toLowerCase());
    return target ? `[${title}](/article/${target.slug})` : `**${title}**`;
  });
}
