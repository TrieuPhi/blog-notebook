import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { POSTS, Post } from "../lib/posts";

const W = 800, H = 520;
function nodeColor(cat: string) {
  return ["AI", "RAG", "Learning"].includes(cat) ? "var(--accent)" : "var(--amber)";
}

export default function GraphPage() {
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("");
  const [selected, setSelected] = useState<Post | null>(null);
  const [box, setBox] = useState({ x: 0, y: 0, w: W, h: H });
  const dragging = useRef<{ x: number; y: number } | null>(null);
  const cats = useMemo(() => [...new Set(POSTS.map((p) => p.category))], []);

  const nodes = useMemo(() => {
    const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 70;
    return POSTS.map((p, i) => {
      const ang = (i / POSTS.length) * Math.PI * 2;
      const conns = POSTS.filter((o) => o.slug !== p.slug && o.tags.some((t) => p.tags.includes(t))).length;
      return { p, x: cx + R * Math.cos(ang), y: cy + R * Math.sin(ang), r: 8 + conns * 2.4 };
    });
  }, []);

  const visible = nodes.filter((n) => (!q || n.p.title.toLowerCase().includes(q.toLowerCase())) && (!cat || n.p.category === cat));
  const visSlugs = new Set(visible.map((n) => n.p.slug));

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const scale = e.deltaY > 0 ? 1.1 : 0.9;
    setBox((b) => {
      const w = Math.max(200, Math.min(1600, b.w * scale));
      const h = w * (H / W);
      return { x: b.x + (b.w - w) / 2, y: b.y + (b.h - h) / 2, w, h };
    });
  };
  const onDown = (e: React.MouseEvent) => { dragging.current = { x: e.clientX, y: e.clientY }; };
  const onMove = (e: React.MouseEvent) => {
    if (!dragging.current) return;
    const dx = (e.clientX - dragging.current.x) * (box.w / W);
    const dy = (e.clientY - dragging.current.y) * (box.h / H);
    dragging.current = { x: e.clientX, y: e.clientY };
    setBox((b) => ({ ...b, x: b.x - dx, y: b.y - dy }));
  };
  const onUp = () => { dragging.current = null; };

  return (
    <div>
      <div className="font-mono text-[11.5px] text-fgdim mb-2">GRAPH / INTERACTIVE</div>
      <h2 className="text-[26px] font-bold mb-1.5">Knowledge graph</h2>
      <p className="text-fgdim mt-0 mb-4">Nodes are notes. Edges are shared tags. Click a node for details, scroll to zoom, drag to pan.</p>
      <div className="flex gap-2 flex-wrap mb-3.5">
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="find node..." className="font-mono px-2.5 py-2 border border-line rounded bg-bg2 text-fg" />
        <select value={cat} onChange={(e) => setCat(e.target.value)} className="font-mono px-2.5 py-2 border border-line rounded bg-bg2 text-fg">
          <option value="">All categories</option>
          {cats.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>
      <div className="grid lg:grid-cols-[1fr_260px] gap-4">
        <svg
          viewBox={`${box.x} ${box.y} ${box.w} ${box.h}`}
          className="w-full h-[520px] border border-line rounded bg-bg2 shadow-sm cursor-grab"
          onWheel={onWheel} onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
        >
          {nodes.map((a) => nodes.map((b) => {
            if (a.p.slug < b.p.slug && a.p.tags.some((t) => b.p.tags.includes(t)) && visSlugs.has(a.p.slug) && visSlugs.has(b.p.slug)) {
              return <line key={a.p.slug + b.p.slug} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="var(--line)" strokeWidth={1} />;
            }
            return null;
          }))}
          {visible.map((n) => (
            <g key={n.p.slug} className="cursor-pointer" onClick={() => setSelected(n.p)}>
              <circle cx={n.x} cy={n.y} r={n.r} fill={nodeColor(n.p.category)} opacity={selected?.slug === n.p.slug ? 1 : 0.85}
                stroke={selected?.slug === n.p.slug ? "var(--fg)" : "none"} strokeWidth={2} />
              <text x={n.x} y={n.y + n.r + 14} fontFamily="JetBrains Mono, monospace" fontSize={10.5} fill="var(--fg-dim)" textAnchor="middle">
                {n.p.title.length > 18 ? n.p.title.slice(0, 16) + "…" : n.p.title}
              </text>
            </g>
          ))}
        </svg>
        <div className="border border-line bg-bg2 rounded p-4 shadow-sm font-mono text-xs min-h-[200px]">
          {selected ? (
            <>
              <div className="text-[14px] font-sans font-semibold mb-1.5">{selected.title}</div>
              <div className="text-fgdim mb-2">{selected.category} · {selected.date?.replace(/-/g, ".")}</div>
              <div className="font-sans text-[13px] text-fgdim mb-2.5">{selected.description}</div>
              <div className="flex gap-1.5 flex-wrap">{selected.tags.map((t) => <span key={t} className="border border-line rounded-full px-2 py-0.5 text-fgdim">{t}</span>)}</div>
              <Link to={`/article/${selected.slug}`} className="inline-block mt-3 font-sans border border-accent bg-accent text-[color:var(--accent-fg)] rounded px-3 py-2 text-sm">Open note</Link>
            </>
          ) : <div className="text-fgdim">Select a node to preview its note.</div>}
        </div>
      </div>
    </div>
  );
}
