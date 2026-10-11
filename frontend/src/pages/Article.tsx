import { useEffect, useMemo, useState } from "react";
import { useParams, Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import { getPost, relatedFor, backlinksFor, resolveWikilinks, POSTS } from "../lib/posts";
import Mermaid from "../components/Mermaid";

function slugify(s: string) {
  return s.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

export default function Article() {
  const { slug = "" } = useParams();
  const post = getPost(slug);
  const [activeId, setActiveId] = useState("");

  const headings = useMemo(() => {
    if (!post) return [];
    return [...post.content.matchAll(/^##\s+(.+)$/gm)].map((m) => ({ text: m[1].trim(), id: slugify(m[1]) }));
  }, [post]);

  const content = useMemo(() => (post ? resolveWikilinks(post.content) : ""), [post]);

  useEffect(() => {
    window.scrollTo({ top: 0 });
    const onScroll = () => {
      let cur = "";
      headings.forEach((h) => {
        const el = document.getElementById(h.id);
        if (el && el.getBoundingClientRect().top < 120) cur = h.id;
      });
      setActiveId(cur);
    };
    window.addEventListener("scroll", onScroll);
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, [slug, headings]);

  if (!post) return <p className="text-fgdim">Note not found.</p>;

  const rel = relatedFor(post);
  const back = backlinksFor(post);
  const idx = POSTS.findIndex((p) => p.slug === post.slug);
  const prev = POSTS[idx - 1];
  const next = POSTS[idx + 1];

  return (
    <div>
      <div className="grid lg:grid-cols-[190px_1fr_220px] gap-9 items-start">
        <aside className="hidden lg:block sticky top-[70px] font-mono text-xs">
          <div className="font-mono text-[11.5px] text-fgdim mb-3">CONTENTS</div>
          {headings.map((h, i) => (
            <a key={h.id} href={`#${h.id}`} className={`block py-1.5 pl-2.5 border-l-2 ${activeId === h.id ? "text-accent border-accent" : "text-fgdim border-line"}`}>
              {String(i + 1).padStart(2, "0")} {h.text}
            </a>
          ))}
        </aside>

        <article className="prose-note">
          <h1>{post.title}</h1>
          <div className="sub text-fgdim mb-4">{post.description}</div>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex, rehypeHighlight]}
            components={{
              h2: ({ children }) => { const t = String(children); return <h2 id={slugify(t)}>{children}</h2>; },
              code: ({ className, children, ...props }: any) => {
                const lang = /language-(\w+)/.exec(className || "")?.[1];
                if (lang === "mermaid") return <Mermaid code={String(children)} />;
                return <code className={className} {...props}>{children}</code>;
              },
            }}
          >
            {content}
          </ReactMarkdown>
        </article>

        <aside className="sticky top-[70px] font-mono text-xs border border-line bg-bg2 rounded p-3.5 shadow-sm">
          <h4 className="text-fgdim mt-0 mb-1.5">Category</h4>
          <span className="inline-block font-mono text-[10.5px] border border-line rounded-full px-2 py-0.5 text-fgdim">{post.category}</span>
          <h4 className="text-fgdim mt-3.5 mb-1.5">Tags</h4>
          <div className="flex gap-1.5 flex-wrap">{post.tags.map((t) => <span key={t} className="font-mono text-[10.5px] border border-line rounded-full px-2 py-0.5 text-fgdim">{t}</span>)}</div>
          <h4 className="text-fgdim mt-3.5 mb-1.5">Reading time</h4><div>{post.readingTime} min</div>
          <h4 className="text-fgdim mt-3.5 mb-1.5">Published</h4><div>{post.date?.replace(/-/g, ".")}</div>
          <h4 className="text-fgdim mt-3.5 mb-1.5">Updated</h4><div>{post.updated?.replace(/-/g, ".")}</div>
          <h4 className="text-fgdim mt-3.5 mb-1.5">Related articles</h4>
          <div>{rel.length ? rel.map((r) => <Link key={r.slug} to={`/article/${r.slug}`} className="block py-1 text-accent">{r.title}</Link>) : <span className="text-fgdim">none yet</span>}</div>
          <h4 className="text-fgdim mt-3.5 mb-1.5">Linked notes</h4>
          <div>{rel.slice(0, 2).length ? rel.slice(0, 2).map((r) => <Link key={r.slug} to={`/article/${r.slug}`} className="block py-1 text-accent">[[{r.title}]]</Link>) : <span className="text-fgdim">none yet</span>}</div>
        </aside>
      </div>

      <div className="mt-9 pt-5 border-t border-line font-mono text-[13px]">
        <div className="text-fgdim text-[11px] my-3.5">RELATED NOTES</div>
        {rel.length ? rel.map((r) => <Link key={r.slug} to={`/article/${r.slug}`} className="block py-1 text-accent">→ {r.title}</Link>) : <span className="text-fgdim">none</span>}
        <div className="text-fgdim text-[11px] my-3.5">REFERENCED BY (BACKLINKS)</div>
        {back.length ? back.map((r) => <Link key={r.slug} to={`/article/${r.slug}`} className="block py-1 text-accent">→ {r.title}</Link>) : <span className="text-fgdim">no backlinks yet</span>}
      </div>

      <div className="grid grid-cols-2 gap-3 mt-6">
        {prev ? <Link to={`/article/${prev.slug}`} className="border border-line rounded p-3 bg-bg2"><div className="font-mono text-[10.5px] text-fgdim">← PREV</div>{prev.title}</Link> : <span />}
        {next ? <Link to={`/article/${next.slug}`} className="border border-line rounded p-3 bg-bg2 text-right"><div className="font-mono text-[10.5px] text-fgdim">NEXT →</div>{next.title}</Link> : <span />}
      </div>
    </div>
  );
}
