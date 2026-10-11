import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { POSTS } from "../lib/posts";

export default function SearchModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [q, setQ] = useState("");
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => { if (open) { setQ(""); setTimeout(() => ref.current?.focus(), 10); } }, [open]);
  if (!open) return null;
  const results = q
    ? POSTS.filter((p) => (p.title + p.description + p.tags.join(" ") + p.category).toLowerCase().includes(q.toLowerCase()))
    : [];
  return (
    <div className="fixed inset-0 bg-black/55 z-[80] flex items-start justify-center pt-[12vh]" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="w-[min(560px,92vw)] bg-bg2 border border-line rounded-lg shadow-xl overflow-hidden">
        <input
          ref={ref} value={q} onChange={(e) => setQ(e.target.value)}
          placeholder="Search notes, tags, categories..."
          className="w-full border-b border-line px-4 py-4 font-mono text-[15px] bg-transparent text-fg outline-none"
        />
        <div className="max-h-[50vh] overflow-auto">
          {results.map((p) => (
            <Link key={p.slug} to={`/article/${p.slug}`} onClick={onClose} className="block px-4 py-3 border-b border-line hover:bg-bg">
              <div className="text-[14px]">{p.title}</div>
              <div className="font-mono text-[11px] text-fgdim mt-0.5">{p.category} · {p.tags.join(", ")} · {p.date?.replace(/-/g, ".")}</div>
            </Link>
          ))}
          {q && results.length === 0 && <div className="px-4 py-3 text-fgdim">No results</div>}
        </div>
      </div>
    </div>
  );
}
