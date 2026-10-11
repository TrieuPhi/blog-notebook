import { Link, useNavigate } from "react-router-dom";
import { POSTS, TOPICS } from "../lib/posts";

export default function Topics() {
  const nav = useNavigate();
  return (
    <div>
      <div className="font-mono text-[11.5px] text-fgdim mb-2">TOPICS / EXPLORER</div>
      <h2 className="text-[26px] font-bold mb-5">Browse by topic</h2>
      <div className="grid sm:grid-cols-2 gap-3.5">
        {TOPICS.map((t) => {
          const n = POSTS.filter((p) => p.category === t.name || p.tags.includes(t.name)).length;
          return (
            <button key={t.name} onClick={() => nav("/blog")} className="text-left border border-line bg-bg2 rounded p-4 shadow-sm hover:border-accent">
              <div className="font-mono text-[11px] text-accent">{n} notes</div>
              <h3 className="my-1.5">{t.name}</h3>
              <p className="text-fgdim text-[13.5px] m-0">{t.desc}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
