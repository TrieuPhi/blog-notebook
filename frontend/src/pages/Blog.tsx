import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { POSTS } from "../lib/posts";

export default function Blog() {
  const [q, setQ] = useState("");
  const [cats, setCats] = useState<Set<string>>(new Set());
  const [tags, setTags] = useState<Set<string>>(new Set());
  const [sort, setSort] = useState<"new" | "old">("new");

  const allCats = useMemo(() => [...new Set(POSTS.map((p) => p.category))], []);
  const tagCounts = useMemo(() => {
    const c: Record<string, number> = {};
    POSTS.forEach((p) => p.tags.forEach((t) => (c[t] = (c[t] || 0) + 1)));
    return c;
  }, []);

  const toggle = (set: Set<string>, setFn: (s: Set<string>) => void, v: string) => {
    const next = new Set(set);
    next.has(v) ? next.delete(v) : next.add(v);
    setFn(next);
  };

  const list = POSTS.filter((p) => {
    if (cats.size && !cats.has(p.category)) return false;
    if (tags.size && ![...tags].every((t) => p.tags.includes(t))) return false;
    if (q && !(p.title + p.description + p.tags.join(" ")).toLowerCase().includes(q.toLowerCase())) return false;
    return true;
  }).sort((a, b) => sort === "old" ? +new Date(a.date) - +new Date(b.date) : +new Date(b.date) - +new Date(a.date));

  return (
    <div>
      <div className="font-mono text-[11.5px] text-fgdim mb-2">BLOG / {POSTS.length} NOTES INDEXED</div>
      <h2 className="text-[26px] font-bold mb-5">All notes</h2>
      <div className="grid md:grid-cols-[220px_1fr] gap-8">
        <aside className="border border-line bg-bg2 rounded p-3.5 shadow-sm h-fit md:sticky md:top-[70px]">
          <h4 className="font-mono text-[11px] text-fgdim mt-0 mb-2">Search</h4>
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="filter notes..." className="w-full px-2.5 py-2 border border-line rounded bg-bg text-fg font-mono text-[12.5px]" />
          <h4 className="font-mono text-[11px] text-fgdim mt-3.5 mb-2">Category</h4>
          <div>{allCats.map((c) => (
            <button key={c} onClick={() => toggle(cats, setCats, c)} className={`inline-flex font-mono text-[11.5px] border rounded-full px-2.5 py-1 mr-1.5 mb-1.5 ${cats.has(c) ? "bg-accent text-[color:var(--accent-fg)] border-accent" : "border-line text-fgdim"}`}>{c}</button>
          ))}</div>
          <h4 className="font-mono text-[11px] text-fgdim mt-3.5 mb-2">Tags</h4>
          <div>{Object.keys(tagCounts).slice(0, 12).map((t) => (
            <button key={t} onClick={() => toggle(tags, setTags, t)} className={`inline-flex font-mono text-[11.5px] border rounded-full px-2.5 py-1 mr-1.5 mb-1.5 ${tags.has(t) ? "bg-accent text-[color:var(--accent-fg)] border-accent" : "border-line text-fgdim"}`}>{t} ({tagCounts[t]})</button>
          ))}</div>
          <h4 className="font-mono text-[11px] text-fgdim mt-3.5 mb-2">Sort</h4>
          <select value={sort} onChange={(e) => setSort(e.target.value as any)} className="w-full p-2 border border-line rounded bg-bg text-fg font-mono text-[12.5px]">
            <option value="new">Newest first</option>
            <option value="old">Oldest first</option>
          </select>
        </aside>
        <div className="flex flex-col gap-3.5">
          {list.map((p) => (
            <Link key={p.slug} to={`/article/${p.slug}`} className="flex flex-col gap-2 bg-bg2 border border-line rounded p-4 shadow-sm hover:border-accent">
              <div className="font-mono text-[11px] text-fgdim flex gap-2 flex-wrap items-center">
                <span className="text-accent">{p.category}</span><span>{p.date?.replace(/-/g, ".")}</span><span>{p.readingTime} min</span>
                <span className="font-mono text-[10.5px] border border-amber text-amber rounded-full px-2 py-0.5">{p.status}</span>
              </div>
              <h3 className="text-[16px] font-semibold m-0">{p.title}</h3>
              <p className="text-[13.5px] text-fgdim m-0">{p.description}</p>
              <div className="flex gap-1.5 flex-wrap">{p.tags.map((t) => <span key={t} className="font-mono text-[10.5px] border border-line rounded-full px-2 py-0.5 text-fgdim">{t}</span>)}</div>
            </Link>
          ))}
          {list.length === 0 && <p className="text-fgdim">No notes match those filters.</p>}
        </div>
      </div>
    </div>
  );
}
