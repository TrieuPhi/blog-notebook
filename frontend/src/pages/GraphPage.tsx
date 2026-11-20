import { useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { POSTS, Post, backlinksFor, relatedFor, TechnicalLayer } from "../lib/posts";

const W = 900;
const H = 600;

type NodeType = "pillar" | "post" | "tag";

interface GraphNode {
  id: string;
  type: NodeType;
  title: string;
  category?: string;
  radius: number;
  x: number;
  y: number;
  post?: Post;
  layer?: TechnicalLayer;
  tag?: string;
  connectedNodes: string[];
}

interface GraphEdge {
  id: string;
  source: GraphNode;
  target: GraphNode;
  kind: "pillar-post" | "post-tag" | "post-post";
}

const PILLARS: { key: TechnicalLayer; title: string; desc: string; x: number; y: number }[] = [
  { key: "Foundation", title: "Foundation", desc: "Nền tảng kỹ thuật, Python internals, Machine Learning và Deep Learning", x: 230, y: 170 },
  { key: "Core LLM & RAG", title: "Core LLM & RAG", desc: "Kiến trúc Transformer, Fine-Tuning (SFT/LoRA) và Đánh giá RAG định lượng", x: 670, y: 170 },
  { key: "Agent Systems", title: "Agent Systems", desc: "Chu trình ReAct, Tool Calling, MCP Protocols và Autonomous Agents", x: 230, y: 440 },
  { key: "Production / Infra", title: "Production / Infra", desc: "vLLM Serving, Quản trị VRAM, CI/CD Pipeline và Kubernetes", x: 670, y: 440 },
];

const PROMINENT_TAGS = ["Python", "RAGAS", "vLLM", "ReAct", "DeepSpeed", "Kubernetes", "PyTorch", "Docker"];

export default function GraphPage() {
  const nav = useNavigate();
  const [q, setQ] = useState("");
  const [filterLayer, setFilterLayer] = useState("");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [box, setBox] = useState({ x: 0, y: 0, w: W, h: H });
  const dragging = useRef<{ x: number; y: number } | null>(null);

  // Generate Multi-Tier Graph Data
  const { nodes, edges } = useMemo(() => {
    const nodeList: GraphNode[] = [];
    const nodeMap = new Map<string, GraphNode>();

    // 1. Tier 1: Pillars (r = 26px)
    PILLARS.forEach((pil) => {
      const node: GraphNode = {
        id: `pillar-${pil.key}`,
        type: "pillar",
        title: pil.title,
        layer: pil.key,
        radius: 26,
        x: pil.x,
        y: pil.y,
        connectedNodes: [],
      };
      nodeList.push(node);
      nodeMap.set(node.id, node);
    });

    // Helper to map post to pillar
    const getPillarKey = (p: Post): TechnicalLayer => {
      const num = parseInt(p.phaseNumber || "0", 10);
      if (num >= 0 && num <= 4) return "Foundation";
      if (num >= 5 && num <= 11) return "Core LLM & RAG";
      if (num >= 12 && num <= 16) return "Agent Systems";
      return "Production / Infra";
    };

    // 2. Tier 2: Posts (r = 15px)
    const pillarPostCounts: Record<string, number> = {
      Foundation: 0,
      "Core LLM & RAG": 0,
      "Agent Systems": 0,
      "Production / Infra": 0,
    };

    POSTS.forEach((post) => {
      const pKey = getPillarKey(post);
      const pil = PILLARS.find((pl) => pl.key === pKey)!;
      const count = pillarPostCounts[pKey]++;
      const angle = (count * 1.3) + 0.5;
      const dist = 110 + (count % 2) * 25;

      const node: GraphNode = {
        id: `post-${post.slug}`,
        type: "post",
        title: post.title,
        category: post.category,
        radius: 15,
        x: pil.x + dist * Math.cos(angle),
        y: pil.y + dist * Math.sin(angle),
        post,
        layer: pKey,
        connectedNodes: [`pillar-${pKey}`],
      };

      nodeList.push(node);
      nodeMap.set(node.id, node);
      nodeMap.get(`pillar-${pKey}`)?.connectedNodes.push(node.id);
    });

    // 3. Tier 3: Tags / Libraries (r = 8px)
    PROMINENT_TAGS.forEach((tagName, idx) => {
      const angle = (idx / PROMINENT_TAGS.length) * Math.PI * 2;
      const cx = W / 2;
      const cy = H / 2;
      const rx = 130;
      const ry = 80;

      const connectedPostIds = POSTS.filter((p) => p.tags.includes(tagName) || p.category === tagName).map(
        (p) => `post-${p.slug}`
      );

      const node: GraphNode = {
        id: `tag-${tagName}`,
        type: "tag",
        title: tagName,
        tag: tagName,
        radius: 8,
        x: cx + rx * Math.cos(angle),
        y: cy + ry * Math.sin(angle),
        connectedNodes: connectedPostIds,
      };

      nodeList.push(node);
      nodeMap.set(node.id, node);
    });

    // 4. Collision displacement relaxation to guarantee minimum spacing between all node pairs
    const padding = 10;
    for (let iter = 0; iter < 50; iter++) {
      for (let i = 0; i < nodeList.length; i++) {
        for (let j = i + 1; j < nodeList.length; j++) {
          const a = nodeList[i];
          const b = nodeList[j];
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          let dist = Math.sqrt(dx * dx + dy * dy);
          const targetDist = a.radius + b.radius + padding;
          if (dist < targetDist) {
            if (dist < 0.001) {
              dx = 1;
              dy = 0;
              dist = 1;
            }
            const overlap = targetDist - dist;
            const nx = dx / dist;
            const ny = dy / dist;
            if (a.type === "pillar") {
              b.x += nx * overlap;
              b.y += ny * overlap;
            } else if (b.type === "pillar") {
              a.x -= nx * overlap;
              a.y -= ny * overlap;
            } else {
              a.x -= nx * overlap * 0.5;
              a.y -= ny * overlap * 0.5;
              b.x += nx * overlap * 0.5;
              b.y += ny * overlap * 0.5;
            }
          }
        }
      }
    }

    // Edges Calculation
    const edgeList: GraphEdge[] = [];

    // Pillar -> Post edges
    POSTS.forEach((post) => {
      const pKey = getPillarKey(post);
      const pillarNode = nodeMap.get(`pillar-${pKey}`)!;
      const postNode = nodeMap.get(`post-${post.slug}`)!;
      edgeList.push({
        id: `edge-pil-${post.slug}`,
        source: pillarNode,
        target: postNode,
        kind: "pillar-post",
      });
    });

    // Post -> Tag edges
    PROMINENT_TAGS.forEach((tagName) => {
      const tagNode = nodeMap.get(`tag-${tagName}`)!;
      POSTS.forEach((p) => {
        if (p.tags.includes(tagName) || p.category === tagName) {
          const postNode = nodeMap.get(`post-${p.slug}`)!;
          edgeList.push({
            id: `edge-tag-${tagName}-${p.slug}`,
            source: postNode,
            target: tagNode,
            kind: "post-tag",
          });
        }
      });
    });

    // Post <-> Post shared tags edges
    for (let i = 0; i < POSTS.length; i++) {
      for (let j = i + 1; j < POSTS.length; j++) {
        const p1 = POSTS[i];
        const p2 = POSTS[j];
        const shared = p1.tags.filter((t) => p2.tags.includes(t));
        if (shared.length >= 2) {
          const n1 = nodeMap.get(`post-${p1.slug}`)!;
          const n2 = nodeMap.get(`post-${p2.slug}`)!;
          edgeList.push({
            id: `edge-pp-${p1.slug}-${p2.slug}`,
            source: n1,
            target: n2,
            kind: "post-post",
          });
        }
      }
    }

    return { nodes: nodeList, edges: edgeList };
  }, []);

  // Filter nodes based on query and layer
  const visibleNodes = useMemo(() => {
    return nodes.filter((n) => {
      if (filterLayer && n.layer && n.layer !== filterLayer) return false;
      if (q) {
        const query = q.toLowerCase();
        return n.title.toLowerCase().includes(query) || (n.tag && n.tag.toLowerCase().includes(query));
      }
      return true;
    });
  }, [nodes, q, filterLayer]);

  const visNodeIds = useMemo(() => new Set(visibleNodes.map((n) => n.id)), [visibleNodes]);

  // Pan and Zoom
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const scale = e.deltaY > 0 ? 1.08 : 0.92;
    setBox((b) => {
      const w = Math.max(250, Math.min(1800, b.w * scale));
      const h = w * (H / W);
      return { x: b.x + (b.w - w) / 2, y: b.y + (b.h - h) / 2, w, h };
    });
  };

  const onDown = (e: React.MouseEvent) => {
    dragging.current = { x: e.clientX, y: e.clientY };
  };

  const onMove = (e: React.MouseEvent) => {
    if (!dragging.current) return;
    const dx = (e.clientX - dragging.current.x) * (box.w / W);
    const dy = (e.clientY - dragging.current.y) * (box.h / H);
    dragging.current = { x: e.clientX, y: e.clientY };
    setBox((b) => ({ ...b, x: b.x - dx, y: b.y - dy }));
  };

  const onUp = () => {
    dragging.current = null;
  };

  const resetView = () => {
    setBox({ x: 0, y: 0, w: W, h: H });
    setSelectedNode(null);
  };

  return (
    <div className="relative">
      {/* Header breadcrumb */}
      <div className="font-mono text-[11.5px] text-fgdim mb-2 flex items-center gap-2">
        <span>GRAPH</span>
        <span>/</span>
        <span>MULTI-TIER KNOWLEDGE MAP</span>
        <span>/</span>
        <span className="text-accent">{nodes.length} TOTAL NODES</span>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
        <div>
          <h2 className="text-[28px] font-bold m-0">Đồ thị Tri thức Đa tầng (Knowledge Graph)</h2>
          <p className="text-fgdim text-sm mt-1 mb-0">
            Trực quan hóa cấu trúc liên kết tri thức: Node lớn (Tầng công nghệ), Node vừa (Bài viết), Node nhỏ (Thư viện/Thẻ).
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 font-mono text-[11px] text-fgdim bg-bg2 border border-line rounded-lg px-3 py-1.5 self-start sm:self-auto shadow-sm">
          <div className="flex items-center gap-1.5">
            <span className="w-4 h-4 rounded-full border-2 border-accent bg-bg inline-block" />
            <span>Pillar (r~26)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-accent inline-block" />
            <span>Post (r~15)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber inline-block" />
            <span>Tag (r~8)</span>
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex gap-2 flex-wrap mb-4 items-center">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Tìm bài viết, pillar, tag..."
          className="font-mono text-xs px-3 py-2 border border-line rounded-md bg-bg2 text-fg focus:outline-none focus:border-accent w-64"
        />
        <select
          value={filterLayer}
          onChange={(e) => setFilterLayer(e.target.value)}
          className="font-mono text-xs px-3 py-2 border border-line rounded-md bg-bg2 text-fg focus:outline-none focus:border-accent"
        >
          <option value="">Tất cả các tầng kỹ thuật</option>
          {PILLARS.map((p) => (
            <option key={p.key} value={p.key}>
              {p.title}
            </option>
          ))}
        </select>
        <button
          onClick={resetView}
          className="font-mono text-xs px-3 py-2 border border-line rounded-md bg-bg2 text-fgdim hover:text-fg hover:border-fgdim transition"
        >
          Đặt lại góc nhìn
        </button>
      </div>

      {/* Main Canvas & Side Drawer Container */}
      <div className="relative border border-line rounded-xl overflow-hidden bg-bg2 shadow-sm min-h-[560px]">
        {/* SVG Graph Viewport */}
        <svg
          viewBox={`${box.x} ${box.y} ${box.w} ${box.h}`}
          className="w-full h-[580px] cursor-grab active:cursor-grabbing select-none"
          onWheel={onWheel}
          onMouseDown={onDown}
          onMouseMove={onMove}
          onMouseUp={onUp}
          onMouseLeave={onUp}
        >
          {/* Subtle background grid pattern */}
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--line)" strokeWidth="0.5" strokeOpacity="0.4" />
            </pattern>
          </defs>
          <rect x={box.x - 200} y={box.y - 200} width={box.w + 400} height={box.h + 400} fill="url(#grid)" />

          {/* Edges */}
          {edges.map((edge) => {
            if (!visNodeIds.has(edge.source.id) || !visNodeIds.has(edge.target.id)) return null;

            const isHighlighted =
              selectedNode &&
              (selectedNode.id === edge.source.id || selectedNode.id === edge.target.id);

            let strokeColor = "var(--line)";
            let strokeWidth = 1;
            let strokeOpacity = 0.4;

            if (edge.kind === "pillar-post") {
              strokeColor = "var(--accent)";
              strokeWidth = isHighlighted ? 2.5 : 1.5;
              strokeOpacity = isHighlighted ? 0.9 : 0.4;
            } else if (edge.kind === "post-post") {
              strokeColor = "var(--amber)";
              strokeWidth = isHighlighted ? 2 : 1;
              strokeOpacity = isHighlighted ? 0.8 : 0.25;
            } else if (edge.kind === "post-tag") {
              strokeColor = "var(--fg-dim)";
              strokeWidth = isHighlighted ? 1.5 : 0.75;
              strokeOpacity = isHighlighted ? 0.7 : 0.2;
            }

            return (
              <line
                key={edge.id}
                x1={edge.source.x}
                y1={edge.source.y}
                x2={edge.target.x}
                y2={edge.target.y}
                stroke={strokeColor}
                strokeWidth={strokeWidth}
                strokeOpacity={strokeOpacity}
              />
            );
          })}

          {/* Nodes */}
          {visibleNodes.map((n) => {
            const isSelected = selectedNode?.id === n.id;
            const isPillar = n.type === "pillar";
            const isPost = n.type === "post";
            const isTag = n.type === "tag";

            let fill = "var(--bg)";
            let stroke = "var(--line)";
            let strokeWidth = 1.5;

            if (isPillar) {
              fill = isSelected ? "var(--accent)" : "var(--bg)";
              stroke = "var(--accent)";
              strokeWidth = isSelected ? 4 : 2.5;
            } else if (isPost) {
              fill = isSelected ? "var(--fg)" : "var(--accent)";
              stroke = isSelected ? "var(--accent)" : "var(--bg)";
              strokeWidth = 2;
            } else if (isTag) {
              fill = isSelected ? "var(--amber)" : "var(--bg)";
              stroke = isSelected ? "var(--fg)" : "var(--amber)";
              strokeWidth = 1.5;
            }

            return (
              <g
                key={n.id}
                className="cursor-pointer transition-transform duration-100"
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedNode(n);
                }}
              >
                {/* Highlight ring for selected node */}
                {isSelected && (
                  <circle
                    cx={n.x}
                    cy={n.y}
                    r={n.radius + 6}
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    className="animate-pulse"
                  />
                )}

                <circle
                  cx={n.x}
                  cy={n.y}
                  r={n.radius}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={strokeWidth}
                  opacity={isSelected ? 1 : 0.9}
                />

                {/* Node Label */}
                <text
                  x={n.x}
                  y={n.y + n.radius + (isPillar ? 15 : isPost ? 13 : 11)}
                  fontFamily={isPillar || isTag ? "JetBrains Mono, monospace" : "Inter, sans-serif"}
                  fontSize={isPillar ? 12 : isPost ? 10.5 : 9}
                  fontWeight={isPillar ? "bold" : isSelected ? "bold" : "normal"}
                  fill={isSelected ? "var(--accent)" : isPillar ? "var(--fg)" : "var(--fg-dim)"}
                  textAnchor="middle"
                >
                  {isPillar
                    ? n.title
                    : n.title.length > 20
                    ? n.title.slice(0, 18) + "…"
                    : n.title}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Interactive Side Drawer (Slide-Over Panel) */}
        {selectedNode && (
          <aside className="absolute top-0 right-0 w-full sm:w-[380px] h-full bg-bg2/95 backdrop-blur-md border-l border-line p-5 shadow-2xl overflow-y-auto flex flex-col justify-between z-10 transition-transform">
            <div>
              {/* Drawer Top Row: Type Badge & Close Button */}
              <div className="flex items-center justify-between pb-3 border-b border-line mb-4">
                <span className="font-mono text-[10.5px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded border border-accent/40 bg-accent/10 text-accent">
                  {selectedNode.type === "post"
                    ? "Ghi chép (Post Note)"
                    : selectedNode.type === "pillar"
                    ? "Tầng Kỹ thuật (Pillar)"
                    : "Từ khóa / Thư viện (Tag)"}
                </span>
                <button
                  onClick={() => setSelectedNode(null)}
                  className="w-7 h-7 rounded-full border border-line bg-bg flex items-center justify-center text-fgdim hover:text-fg hover:border-accent transition text-xs font-mono"
                  title="Đóng bảng xem trước"
                >
                  ✕
                </button>
              </div>

              {/* DETAILS FOR POST NOTE */}
              {selectedNode.type === "post" && selectedNode.post && (
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-bold m-0 leading-snug">{selectedNode.post.title}</h3>
                    <div className="font-mono text-xs text-fgdim mt-1.5 flex gap-2 flex-wrap items-center">
                      <span className="text-accent">{selectedNode.post.category}</span>
                      <span>•</span>
                      <span>{selectedNode.post.date?.replace(/-/g, ".")}</span>
                      <span>•</span>
                      <span>{selectedNode.post.readingTime} min</span>
                      {selectedNode.post.phaseNumber && (
                        <span className="border border-line rounded px-1.5 py-0.2 text-[10px]">
                          Phase {selectedNode.post.phaseNumber}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Core Motto */}
                  {selectedNode.post.motto && (
                    <div className="border-l-2 border-accent bg-bg/80 px-3 py-2 rounded-r">
                      <div className="font-mono text-[10px] text-accent uppercase font-bold tracking-wider mb-0.5">
                        Châm ngôn cốt lõi (Core Motto):
                      </div>
                      <p className="text-xs italic text-fg/90 m-0 leading-relaxed font-sans">
                        &ldquo;{selectedNode.post.motto}&rdquo;
                      </p>
                    </div>
                  )}

                  {/* 3 Key Points */}
                  {selectedNode.post.keyPoints && selectedNode.post.keyPoints.length > 0 && (
                    <div className="bg-bg/40 border border-line rounded-lg p-3">
                      <div className="font-mono text-[10.5px] text-fg font-semibold uppercase tracking-wider mb-2 flex items-center gap-1.5">
                        <span>🎯</span>
                        <span>3 Ý Mấu Chốt (Key Takeaways):</span>
                      </div>
                      <ul className="text-xs space-y-1.5 m-0 pl-1 list-none font-sans text-fg/85">
                        {selectedNode.post.keyPoints.slice(0, 3).map((pt, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <span className="text-accent shrink-0 font-bold">✓</span>
                            <span className="leading-snug">{pt}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Actionable Artifacts */}
                  {selectedNode.post.artifacts && selectedNode.post.artifacts.length > 0 && (
                    <div>
                      <div className="font-mono text-[10px] text-fgdim uppercase tracking-wider mb-1.5">
                        Actionable Artifacts:
                      </div>
                      <div className="flex gap-1.5 flex-wrap">
                        {selectedNode.post.artifacts.map((art) => (
                          <span
                            key={art}
                            className="font-mono text-[10.5px] border border-amber/40 bg-amber/10 text-amber rounded px-2 py-0.5 font-medium"
                          >
                            {art}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Backlinks */}
                  <div>
                    <div className="font-mono text-[10px] text-fgdim uppercase tracking-wider mb-1.5">
                      Liên kết chéo &amp; Backlinks:
                    </div>
                    {backlinksFor(selectedNode.post).length > 0 ? (
                      <div className="space-y-1">
                        {backlinksFor(selectedNode.post).map((bl) => (
                          <Link
                            key={bl.slug}
                            to={`/article/${bl.slug}`}
                            className="block font-mono text-xs border border-line bg-bg rounded px-2 py-1 text-fgdim hover:border-accent hover:text-accent transition truncate"
                          >
                            ← {bl.title}
                          </Link>
                        ))}
                      </div>
                    ) : (
                      <div className="font-mono text-[11px] text-fgdim italic bg-bg/40 rounded p-2 border border-dashed border-line">
                        Chưa có bài viết nào trỏ trực tiếp đến ghi chép này.
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* DETAILS FOR PILLAR */}
              {selectedNode.type === "pillar" && (
                <div className="space-y-4">
                  <h3 className="text-xl font-bold m-0">{selectedNode.title}</h3>
                  <p className="text-xs text-fgdim leading-relaxed font-sans">
                    {PILLARS.find((p) => p.key === selectedNode.layer)?.desc}
                  </p>

                  <div className="border border-line bg-bg/50 rounded-lg p-3">
                    <div className="font-mono text-xs text-accent font-semibold mb-2">
                      Bài viết trong Tầng này:
                    </div>
                    <div className="space-y-1.5">
                      {POSTS.filter(
                        (p) =>
                          (selectedNode.layer === "Foundation" && parseInt(p.phaseNumber || "0", 10) <= 4) ||
                          (selectedNode.layer === "Core LLM & RAG" &&
                            parseInt(p.phaseNumber || "0", 10) >= 5 &&
                            parseInt(p.phaseNumber || "0", 10) <= 11) ||
                          (selectedNode.layer === "Agent Systems" &&
                            parseInt(p.phaseNumber || "0", 10) >= 12 &&
                            parseInt(p.phaseNumber || "0", 10) <= 16) ||
                          (selectedNode.layer === "Production / Infra" &&
                            parseInt(p.phaseNumber || "0", 10) >= 17)
                      ).map((p) => (
                        <Link
                          key={p.slug}
                          to={`/article/${p.slug}`}
                          className="block text-xs font-sans text-fg hover:text-accent border-b border-line/40 pb-1 truncate"
                        >
                          • {p.title}
                        </Link>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* DETAILS FOR TAG */}
              {selectedNode.type === "tag" && (
                <div className="space-y-4">
                  <h3 className="text-xl font-bold font-mono text-amber m-0">#{selectedNode.tag}</h3>
                  <p className="text-xs text-fgdim font-sans leading-relaxed">
                    Từ khóa kỹ thuật hoặc thư viện trọng tâm được sử dụng xuyên suốt trong các bài toán thực nghiệm.
                  </p>

                  <div className="border border-line bg-bg/50 rounded-lg p-3">
                    <div className="font-mono text-xs text-fg mb-2">
                      Các bài viết gắn thẻ <strong>#{selectedNode.tag}</strong>:
                    </div>
                    <div className="space-y-1.5">
                      {POSTS.filter(
                        (p) => p.tags.includes(selectedNode.tag!) || p.category === selectedNode.tag
                      ).map((p) => (
                        <Link
                          key={p.slug}
                          to={`/article/${p.slug}`}
                          className="block text-xs font-sans text-fg hover:text-accent border-b border-line/40 pb-1 truncate"
                        >
                          • {p.title}
                        </Link>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Drawer Bottom Actions */}
            <div className="pt-4 border-t border-line mt-6">
              {selectedNode.type === "post" && selectedNode.post && (
                <Link
                  to={`/article/${selectedNode.post.slug}`}
                  className="w-full font-mono text-xs border border-accent bg-accent text-[color:var(--accent-fg)] rounded-md py-2.5 px-4 text-center font-semibold block shadow-sm hover:opacity-90 transition"
                >
                  Đọc toàn bộ bài viết (Open Note) →
                </Link>
              )}
              {selectedNode.type === "pillar" && (
                <button
                  onClick={() => nav(`/blog?tab=curriculum`)}
                  className="w-full font-mono text-xs border border-accent bg-accent text-[color:var(--accent-fg)] rounded-md py-2.5 px-4 text-center font-semibold shadow-sm hover:opacity-90 transition"
                >
                  Xem lộ trình đào tạo tầng này →
                </button>
              )}
              {selectedNode.type === "tag" && (
                <button
                  onClick={() => nav(`/blog?tag=${encodeURIComponent(selectedNode.tag!)}`)}
                  className="w-full font-mono text-xs border border-accent bg-accent text-[color:var(--accent-fg)] rounded-md py-2.5 px-4 text-center font-semibold shadow-sm hover:opacity-90 transition"
                >
                  Lọc tất cả bài viết theo Tag này →
                </button>
              )}
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
