import { Link } from "react-router-dom";
import { POSTS, TOPICS } from "../lib/posts";
import PostCard from "../components/PostCard";
import Terminal from "../components/Terminal";

const CHAIN = ["RAG", "RAGAS", "Faithfulness", "Evaluators", "SLM", "Qwen", "Fine-tuning"];

function SectionHead({ title, to, label }: { title: string; to: string; label: string }) {
  return (
    <div className="flex items-baseline justify-between mb-4 flex-wrap gap-2">
      <h2 className="text-[20px] font-bold m-0">{title}</h2>
      <Link to={to} className="font-mono text-[12.5px] text-accent hover:underline">{label}</Link>
    </div>
  );
}

export default function Home() {
  const featured = POSTS.filter((p) => p.featured);
  const latest = POSTS.slice(0, 6);
  return (
    <div>
      <div className="grid md:grid-cols-[1.3fr_1fr] gap-12 items-start py-9 pb-14">
        <div>
          <div className="font-mono text-[11.5px] text-fgdim flex items-center gap-2 mb-2.5 before:content-[''] before:w-3.5 before:h-px before:bg-fgdim">
            SYSTEM ONLINE · LAST UPDATED 2026.09.24
          </div>
          <h1 className="text-[44px] leading-[1.12] font-bold tracking-tight m-0 mb-4">Notes on AI, Engineering &amp; Learning.</h1>
          <p className="text-fgdim text-[16.5px] max-w-[52ch] mb-5">
            A personal knowledge base where I write about AI engineering, tools, experiments, systems, and the things I learn along the way.
          </p>
          <div className="flex gap-2.5 flex-wrap">
            <Link to="/blog" className="border border-accent bg-accent text-[color:var(--accent-fg)] rounded px-4 py-2.5 text-sm shadow-sm">Read the latest notes</Link>
            <Link to="/graph" className="border border-line bg-bg2 rounded px-4 py-2.5 text-sm shadow-sm hover:border-fgdim">Explore the graph</Link>
          </div>
        </div>
        <Terminal />
      </div>

      <SectionHead title="Featured" to="/blog" label="all posts →" />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {featured.map((p) => <PostCard key={p.slug} p={p} />)}
      </div>

      <div className="mt-11">
        <SectionHead title="Latest notes" to="/blog" label="view all →" />
        <div className="border-t border-line">
          {latest.map((p) => (
            <Link key={p.slug} to={`/article/${p.slug}`} className="flex gap-4 items-baseline py-3.5 border-b border-line text-sm flex-wrap hover:[&>span.title]:text-accent">
              <span className="font-mono text-fgdim text-[12.5px] w-[110px]">{p.date?.replace(/-/g, ".")}</span>
              <span className="text-fgdim">→</span>
              <span className="title flex-1 min-w-[180px]">{p.title}</span>
              <span className="font-mono text-fgdim text-[12.5px]">{p.category}</span>
              <span className="font-mono text-fgdim text-[11.5px]">{p.readingTime} min</span>
            </Link>
          ))}
        </div>
      </div>

      <div className="mt-11">
        <SectionHead title="Topics" to="/topics" label="explore →" />
        <div className="flex flex-wrap gap-2">
          {TOPICS.slice(0, 10).map((t) => (
            <Link key={t.name} to="/topics" className="font-mono text-[12.5px] border border-line bg-bg2 rounded px-3 py-2 text-fgdim hover:text-fg hover:border-accent">{t.name}</Link>
          ))}
        </div>
      </div>

      <div className="mt-11">
        <SectionHead title="Knowledge graph" to="/graph" label="expand →" />
        <div className="border border-line bg-bg2 rounded p-3 shadow-sm">
          <svg width="100%" height="200" viewBox="0 0 900 200">
            {CHAIN.map((n, i) => {
              const gap = 900 / CHAIN.length, x = gap * i + gap / 2, y = 100;
              return (
                <g key={n}>
                  {i < CHAIN.length - 1 && <line x1={x} y1={y} x2={gap * (i + 1) + gap / 2} y2={y} stroke="var(--line)" strokeWidth={1.5} />}
                  <circle cx={x} cy={y} r={6} fill="var(--accent)" />
                  <text x={x} y={y - 16} fontFamily="JetBrains Mono, monospace" fontSize={11} fill="var(--fg-dim)" textAnchor="middle">{n}</text>
                </g>
              );
            })}
          </svg>
        </div>
        <Link to="/graph" className="inline-block mt-3 border border-line bg-bg2 rounded px-4 py-2.5 text-sm shadow-sm hover:border-fgdim">Explore knowledge graph</Link>
      </div>

      <div className="mt-11">
        <SectionHead title="About" to="/about" label="full page →" />
        <div className="grid grid-cols-[120px_1fr] gap-6 items-start">
          <div className="w-[120px] h-[120px] rounded bg-gradient-to-br from-accent to-amber flex items-center justify-center font-mono text-[28px] font-bold text-[color:var(--accent-fg)]">AI</div>
          <div>
            <p className="mb-2.5">AI Engineer working on retrieval-augmented generation, small language models, and the infrastructure that keeps them running in production. This site is where I keep the notes I'd otherwise lose.</p>
            <div className="flex gap-1.5 flex-wrap">
              {["RAG", "RAGAS", "vLLM", "MLOps", "Python"].map((t) => <span key={t} className="font-mono text-[10.5px] border border-line rounded-full px-2 py-0.5 text-fgdim">{t}</span>)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
