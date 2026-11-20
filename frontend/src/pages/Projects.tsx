import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { PROJECTS, Project } from "../lib/posts";

function statusBadgeClass(status: Project["status"]) {
  switch (status) {
    case "Production":
      return "border-emerald-500/50 text-emerald-400 bg-emerald-500/10";
    case "Active":
      return "border-accent text-accent bg-accent/10";
    case "Experimental":
      return "border-amber text-amber bg-amber/10";
    case "Maintained":
      return "border-line text-fgdim bg-bg";
    default:
      return "border-line text-fgdim";
  }
}

export default function Projects() {
  const [selectedCat, setSelectedCat] = useState<string>("All");

  const categories = useMemo(() => {
    return ["All", ...new Set(PROJECTS.map((p) => p.category))];
  }, []);

  const filteredProjects = useMemo(() => {
    if (selectedCat === "All") return PROJECTS;
    return PROJECTS.filter((p) => p.category === selectedCat);
  }, [selectedCat]);

  return (
    <div>
      {/* Header breadcrumb */}
      <div className="font-mono text-[11.5px] text-fgdim mb-2 flex items-center gap-2">
        <span>PROJECTS</span>
        <span>/</span>
        <span>ENGINEERING CASE STUDIES</span>
        <span>/</span>
        <span className="text-accent">{PROJECTS.length} PRODUCTION EXPERIMENTS</span>
      </div>

      <div className="mb-6">
        <h2 className="text-[28px] font-bold m-0">Thực chiến Kỹ thuật (Case Studies)</h2>
        <p className="text-fgdim text-sm mt-1 max-w-[72ch]">
          Không dừng lại ở những bản demo đồ chơi; mỗi dự án là một case study thực tế giải quyết các bài toán hóc búa về chi phí suy luận, độ trễ P99, tối ưu hóa bộ nhớ GPU VRAM và runtime tác tử tự trị.
        </p>
      </div>

      {/* Category filter tabs */}
      <div className="flex gap-2 flex-wrap mb-8 pb-3 border-b border-line">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCat(cat)}
            className={`font-mono text-xs px-3 py-1.5 rounded-full border transition-colors ${
              selectedCat === cat
                ? "bg-accent text-[color:var(--accent-fg)] border-accent font-semibold"
                : "border-line text-fgdim hover:border-fgdim hover:text-fg bg-bg2"
            }`}
          >
            {cat === "All" ? `Tất cả (${PROJECTS.length})` : cat}
          </button>
        ))}
      </div>

      {/* Projects List */}
      <div className="space-y-8">
        {filteredProjects.map((pr) => (
          <article
            key={pr.id}
            className="border border-line bg-bg2 rounded-xl p-5 sm:p-6 shadow-sm hover:border-line/80 transition"
          >
            {/* Top row: Category & Status */}
            <div className="flex items-center justify-between gap-2 mb-2 flex-wrap">
              <span className="font-mono text-[11px] font-semibold text-accent border border-accent/30 bg-accent/10 px-2.5 py-0.5 rounded-full">
                {pr.category}
              </span>
              <span
                className={`font-mono text-[10.5px] font-medium border rounded-full px-2.5 py-0.5 ${statusBadgeClass(
                  pr.status
                )}`}
              >
                ● {pr.status}
              </span>
            </div>

            {/* Title & Tagline */}
            <h3 className="text-xl font-bold m-0 mb-1">{pr.name}</h3>
            <p className="text-xs font-mono text-fgdim mb-4">{pr.tagline}</p>

            {/* Problem Solved Structured Box */}
            <div className="bg-bg/70 border border-line rounded-lg p-4 mb-4 text-xs space-y-3 font-sans">
              <div>
                <span className="font-mono text-amber text-[10.5px] font-bold uppercase tracking-wider block mb-1">
                  ⚠️ Bài toán Thực tế (The Problem):
                </span>
                <p className="text-fg/90 m-0 leading-relaxed">{pr.problemSolved.problem}</p>
              </div>

              <div>
                <span className="font-mono text-accent text-[10.5px] font-bold uppercase tracking-wider block mb-1">
                  💡 Giải pháp Kiến trúc (The Solution):
                </span>
                <p className="text-fg/90 m-0 leading-relaxed">{pr.problemSolved.solution}</p>
              </div>

              <div className="border border-amber/40 bg-amber/10 text-amber rounded-md px-3 py-2 font-mono text-[11.5px] font-semibold flex items-center gap-2">
                <span>⚡ Kết quả đo lường:</span>
                <span>{pr.problemSolved.impact}</span>
              </div>
            </div>

            {/* Architecture Notes & Deep Links */}
            {pr.architectureNotes && pr.architectureNotes.length > 0 && (
              <div className="mb-4">
                <span className="font-mono text-[10.5px] text-fgdim uppercase tracking-wider block mb-2">
                  Ghi chép Kiến trúc &amp; Liên kết Sâu (Architecture Notes):
                </span>
                <div className="grid gap-2 sm:grid-cols-2">
                  {pr.architectureNotes.map((note) => (
                    <Link
                      key={note.postSlug}
                      to={`/article/${note.postSlug}`}
                      className="group/note border border-line bg-bg/40 hover:border-accent hover:bg-bg2/80 rounded-lg p-3 transition flex flex-col justify-between"
                    >
                      <div>
                        <div className="font-semibold text-xs text-fg group-hover/note:text-accent transition-colors flex items-center gap-1.5">
                          <span>📖</span>
                          <span>{note.title}</span>
                        </div>
                        <p className="text-[11.5px] text-fgdim m-0 mt-1 leading-relaxed font-sans">
                          {note.detail}
                        </p>
                      </div>
                      <div className="font-mono text-[10.5px] text-accent mt-2 flex items-center gap-1">
                        <span>Đọc phân tích chi tiết</span>
                        <span className="group-hover/note:translate-x-1 transition-transform">→</span>
                      </div>
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* Tech Stack Badge Cluster */}
            <div className="mb-4">
              <span className="font-mono text-[10.5px] text-fgdim uppercase tracking-wider block mb-2">
                Công nghệ Sử dụng (Tech Stack):
              </span>
              <div className="flex gap-1.5 flex-wrap">
                {pr.stack.map((tech) => (
                  <span
                    key={tech}
                    className="font-mono text-[10.5px] border border-line bg-bg rounded-md px-2 py-0.5 text-fgdim font-medium"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>

            {/* Quick Action Links */}
            <div className="pt-3 border-t border-line/60 flex items-center justify-between flex-wrap gap-2">
              <div className="flex gap-2">
                {pr.quickLinks.github && (
                  <a
                    href={pr.quickLinks.github}
                    target="_blank"
                    rel="noreferrer"
                    className="font-mono text-xs border border-line rounded px-2.5 py-1 text-fg hover:border-accent hover:text-accent transition flex items-center gap-1"
                  >
                    <span>GitHub</span>
                    <span>↗</span>
                  </a>
                )}
                {pr.quickLinks.demo && (
                  <Link
                    to={pr.quickLinks.demo}
                    className="font-mono text-xs border border-line rounded px-2.5 py-1 text-fg hover:border-accent hover:text-accent transition flex items-center gap-1"
                  >
                    <span>Live Demo</span>
                    <span>↗</span>
                  </Link>
                )}
              </div>

              {pr.quickLinks.article && (
                <Link
                  to={pr.quickLinks.article}
                  className="font-mono text-xs font-semibold text-accent hover:underline flex items-center gap-1"
                >
                  <span>Xem Case Study hoàn chỉnh</span>
                  <span>→</span>
                </Link>
              )}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
