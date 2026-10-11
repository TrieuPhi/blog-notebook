import { Link } from "react-router-dom";
import type { Post } from "../lib/posts";

export default function PostCard({ p }: { p: Post }) {
  return (
    <Link to={`/article/${p.slug}`} className="flex flex-col gap-2 bg-bg2 border border-line rounded p-4 shadow-sm hover:border-accent hover:-translate-y-0.5 transition">
      <div className="font-mono text-[11px] text-fgdim flex gap-2 flex-wrap">
        <span className="text-accent">{p.category}</span><span>{p.date?.replace(/-/g, ".")}</span><span>{p.readingTime} min</span>
      </div>
      <h3 className="text-[16px] font-semibold leading-snug m-0">{p.title}</h3>
      <p className="text-[13.5px] text-fgdim flex-1 m-0">{p.description}</p>
      <div className="flex gap-1.5 flex-wrap">
        {p.tags.slice(0, 3).map((t) => <span key={t} className="font-mono text-[10.5px] border border-line rounded-full px-2 py-0.5 text-fgdim">{t}</span>)}
      </div>
    </Link>
  );
}
