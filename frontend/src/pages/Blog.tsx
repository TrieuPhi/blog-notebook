import { useMemo, useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { POSTS, PHASES, TOPICS, TechnicalLayer, Post } from "../lib/posts";
import PostCard from "../components/PostCard";

const LAYERS: { key: TechnicalLayer; title: string; subtitle: string }[] = [
  { key: "Foundation", title: "Tầng 1: Foundation", subtitle: "Nền tảng kỹ thuật, toán học và ngôn ngữ lập trình" },
  { key: "Core LLM & RAG", title: "Tầng 2: Core LLM & RAG", subtitle: "Kiến trúc mô hình lớn, fine-tuning và hệ thống truy xuất" },
  { key: "Agent Systems", title: "Tầng 3: Agent Systems", subtitle: "Vòng lặp tác tử tự trị, ReAct, tool protocols và đa tác tử" },
  { key: "Production / Infra", title: "Tầng 4: Production & Infra", subtitle: "Phục vụ suy luận quy mô lớn, MLOps, CI/CD và Kubernetes" },
];

export default function Blog() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Read URL params
  const tabParam = searchParams.get("tab") || "all";
  const categoryParam = searchParams.get("category") || "";
  const topicParam = searchParams.get("topic") || "";
  const tagParam = searchParams.get("tag") || "";
  const phaseParam = searchParams.get("phase") || "";
  const queryParam = searchParams.get("q") || "";

  // Local state initialized / synced with URL
  const [activeTab, setActiveTab] = useState<"all" | "curriculum">(tabParam === "curriculum" ? "curriculum" : "all");
  const [q, setQ] = useState(queryParam);
  const [sort, setSort] = useState<"new" | "old">("new");

  useEffect(() => {
    if (tabParam === "curriculum" || tabParam === "all") {
      setActiveTab(tabParam);
    }
  }, [tabParam]);

  useEffect(() => {
    if (queryParam !== q) {
      setQ(queryParam);
    }
  }, [queryParam]);

  const setTab = (tab: "all" | "curriculum") => {
    setActiveTab(tab);
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      if (tab === "curriculum") {
        next.set("tab", "curriculum");
      } else {
        next.delete("tab");
      }
      return next;
    });
  };

  const clearAllFilters = () => {
    setQ("");
    setSearchParams((prev) => {
      const next = new URLSearchParams();
      if (activeTab === "curriculum") next.set("tab", "curriculum");
      return next;
    });
  };

  const toggleCategory = (cat: string) => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      if (categoryParam === cat) {
        next.delete("category");
      } else {
        next.set("category", cat);
        next.delete("topic");
      }
      return next;
    });
  };

  const toggleTag = (t: string) => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      if (tagParam === t) {
        next.delete("tag");
      } else {
        next.set("tag", t);
      }
      return next;
    });
  };

  const allCats = useMemo(() => [...new Set(POSTS.map((p) => p.category))], []);
  const tagCounts = useMemo(() => {
    const c: Record<string, number> = {};
    POSTS.forEach((p) => p.tags.forEach((t) => (c[t] = (c[t] || 0) + 1)));
    return c;
  }, []);

  const matchedTopic = useMemo(() => {
    if (!topicParam) return undefined;
    return TOPICS.find((t) => t.name === topicParam || t.tag === topicParam || t.name.toLowerCase() === topicParam.toLowerCase() || t.tag.toLowerCase() === topicParam.toLowerCase());
  }, [topicParam]);

  // Filter list
  const filteredPosts = useMemo(() => {
    return POSTS.filter((p) => {
      if (categoryParam && p.category.toLowerCase() !== categoryParam.toLowerCase()) return false;
      if (tagParam && !p.tags.map((t) => t.toLowerCase()).includes(tagParam.toLowerCase())) return false;
      if (phaseParam && p.phaseNumber !== phaseParam) return false;
      if (matchedTopic) {
        const nameLower = matchedTopic.name.toLowerCase();
        const tagLower = matchedTopic.tag.toLowerCase();
        const matchesCategory = p.category.toLowerCase() === nameLower || p.category.toLowerCase() === tagLower;
        const matchesTag = p.tags.some((t) => {
          const tLower = t.toLowerCase();
          return tLower === nameLower || tLower === tagLower;
        });
        const matchesTitle = p.title.toLowerCase().includes(nameLower) || p.title.toLowerCase().includes(tagLower);
        if (!matchesCategory && !matchesTag && !matchesTitle) return false;
      } else if (topicParam) {
        const term = topicParam.toLowerCase();
        const matchesCategory = p.category.toLowerCase() === term;
        const matchesTag = p.tags.some((t) => t.toLowerCase() === term);
        const matchesTitle = p.title.toLowerCase().includes(term);
        if (!matchesCategory && !matchesTag && !matchesTitle) return false;
      }
      if (q) {
        const query = q.toLowerCase();
        const fullText = (p.title + " " + p.description + " " + (p.motto || "") + " " + p.tags.join(" ")).toLowerCase();
        if (!fullText.includes(query)) return false;
      }
      return true;
    }).sort((a, b) => (sort === "old" ? +new Date(a.date) - +new Date(b.date) : +new Date(b.date) - +new Date(a.date)));
  }, [categoryParam, tagParam, phaseParam, topicParam, matchedTopic, q, sort]);

  const hasActiveFilter = Boolean(categoryParam || topicParam || tagParam || phaseParam || q);

  return (
    <div>
      {/* Header breadcrumb & title */}
      <div className="font-mono text-[11.5px] text-fgdim mb-2 flex items-center gap-2">
        <span>BLOG</span>
        <span>/</span>
        <span>{POSTS.length} NOTES INDEXED</span>
        <span>/</span>
        <span className="text-accent">{PHASES.filter((ph) => ph.status === "active").length} ACTIVE PHASES</span>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h2 className="text-[28px] font-bold m-0">Sổ tay Kỹ thuật &amp; Giáo trình</h2>
          <p className="text-fgdim text-sm mt-1 mb-0">
            Tổng hợp ghi chép thực chiến, giải mã kiến trúc và lộ trình đào tạo AI Engineering từ nguyên lý đệ nhất.
          </p>
        </div>

        {/* Tab switcher */}
        <div className="inline-flex bg-bg2 border border-line rounded-lg p-1 font-mono text-xs shadow-sm self-start sm:self-auto">
          <button
            onClick={() => setTab("all")}
            className={`px-3.5 py-1.5 rounded-md transition-all flex items-center gap-1.5 font-medium ${
              activeTab === "all"
                ? "bg-accent text-[color:var(--accent-fg)] shadow-sm font-semibold"
                : "text-fgdim hover:text-fg"
            }`}
          >
            <span>📝</span>
            <span>Tất cả bài viết ({POSTS.length})</span>
          </button>
          <button
            onClick={() => setTab("curriculum")}
            className={`px-3.5 py-1.5 rounded-md transition-all flex items-center gap-1.5 font-medium ${
              activeTab === "curriculum"
                ? "bg-accent text-[color:var(--accent-fg)] shadow-sm font-semibold"
                : "text-fgdim hover:text-fg"
            }`}
          >
            <span>🗺️</span>
            <span>Lộ trình học tập / Phases ({PHASES.length})</span>
          </button>
        </div>
      </div>

      {/* Active Filter Indicator */}
      {hasActiveFilter && (
        <div className="bg-bg2 border border-accent/40 rounded-lg px-4 py-2.5 mb-6 flex flex-wrap items-center justify-between gap-2 shadow-sm">
          <div className="flex items-center gap-2 flex-wrap font-mono text-xs">
            <span className="text-accent font-semibold">Đang kích hoạt bộ lọc:</span>
            {topicParam && (
              <span className="border border-line bg-bg rounded px-2 py-0.5 text-fg">
                Chủ đề: <strong>{matchedTopic ? matchedTopic.name : topicParam}</strong>
              </span>
            )}
            {categoryParam && (
              <span className="border border-line bg-bg rounded px-2 py-0.5 text-fg">
                Category: <strong>{categoryParam}</strong>
              </span>
            )}
            {tagParam && (
              <span className="border border-line bg-bg rounded px-2 py-0.5 text-fg">
                Tag: <strong>#{tagParam}</strong>
              </span>
            )}
            {phaseParam && (
              <span className="border border-line bg-bg rounded px-2 py-0.5 text-fg">
                Phase: <strong>{phaseParam}</strong>
              </span>
            )}
            {q && (
              <span className="border border-line bg-bg rounded px-2 py-0.5 text-fg">
                Tìm kiếm: <strong>&ldquo;{q}&rdquo;</strong>
              </span>
            )}
            <span className="text-fgdim">({filteredPosts.length} kết quả)</span>
          </div>
          <button
            onClick={clearAllFilters}
            className="font-mono text-xs text-accent hover:underline flex items-center gap-1"
          >
            <span>✕</span>
            <span>Xóa bộ lọc</span>
          </button>
        </div>
      )}

      {/* TAB 1: ALL NOTES */}
      {activeTab === "all" && (
        <div className="grid md:grid-cols-[230px_1fr] gap-8">
          {/* Sidebar Filters */}
          <aside className="border border-line bg-bg2 rounded-lg p-4 shadow-sm h-fit md:sticky md:top-[70px]">
            <h4 className="font-mono text-[11px] text-fgdim uppercase tracking-wider mt-0 mb-2">Tìm kiếm</h4>
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="nhập từ khóa, tags..."
              className="w-full px-2.5 py-2 border border-line rounded bg-bg text-fg font-mono text-[12.5px] focus:outline-none focus:border-accent"
            />

            <h4 className="font-mono text-[11px] text-fgdim uppercase tracking-wider mt-4 mb-2">Chuyên mục</h4>
            <div className="flex flex-wrap gap-1.5">
              {allCats.map((c) => {
                const isSelected = categoryParam === c;
                return (
                  <button
                    key={c}
                    onClick={() => toggleCategory(c)}
                    className={`inline-flex font-mono text-[11px] border rounded-full px-2.5 py-0.5 transition-colors ${
                      isSelected
                        ? "bg-accent text-[color:var(--accent-fg)] border-accent font-semibold"
                        : "border-line text-fgdim hover:border-fgdim hover:text-fg"
                    }`}
                  >
                    {c}
                  </button>
                );
              })}
            </div>

            <h4 className="font-mono text-[11px] text-fgdim uppercase tracking-wider mt-4 mb-2">Thẻ phổ biến</h4>
            <div className="flex flex-wrap gap-1.5">
              {Object.keys(tagCounts)
                .slice(0, 14)
                .map((t) => {
                  const isSelected = tagParam === t;
                  return (
                    <button
                      key={t}
                      onClick={() => toggleTag(t)}
                      className={`inline-flex font-mono text-[10.5px] border rounded-full px-2 py-0.5 transition-colors ${
                        isSelected
                          ? "bg-accent text-[color:var(--accent-fg)] border-accent font-semibold"
                          : "border-line text-fgdim hover:border-fgdim hover:text-fg"
                      }`}
                    >
                      #{t} <span className="opacity-60 ml-1">({tagCounts[t]})</span>
                    </button>
                  );
                })}
            </div>

            <h4 className="font-mono text-[11px] text-fgdim uppercase tracking-wider mt-4 mb-2">Sắp xếp</h4>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as any)}
              className="w-full p-2 border border-line rounded bg-bg text-fg font-mono text-[12px] focus:outline-none focus:border-accent"
            >
              <option value="new">Mới nhất trước</option>
              <option value="old">Cũ nhất trước</option>
            </select>
          </aside>

          {/* Posts List */}
          <div className="flex flex-col gap-4">
            <div className="font-mono text-xs text-fgdim flex justify-between items-center pb-2 border-b border-line">
              <span>HIỂN THỊ {filteredPosts.length} BÀI VIẾT</span>
              <span>TRÊN TỔNG SỐ {POSTS.length} GHI CHÉP</span>
            </div>

            {filteredPosts.map((p) => (
              <PostCard key={p.slug} p={p} />
            ))}

            {filteredPosts.length === 0 && (
              <div className="border border-line bg-bg2 rounded-lg p-8 text-center my-4">
                <p className="text-fgdim font-sans mb-3">Không có bài viết nào khớp với bộ lọc hiện tại.</p>
                <button
                  onClick={clearAllFilters}
                  className="font-mono text-xs border border-accent bg-accent text-[color:var(--accent-fg)] rounded px-3 py-1.5"
                >
                  Xóa tất cả bộ lọc
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: CURRICULUM / PHASES */}
      {activeTab === "curriculum" && (
        <div className="space-y-12">
          {LAYERS.map((layer) => {
            const phasesInLayer = PHASES.filter((ph) => ph.layer === layer.key);

            return (
              <section key={layer.key} className="border-t border-line pt-6">
                <div className="mb-6">
                  <div className="font-mono text-xs text-accent uppercase tracking-wider font-semibold">
                    {layer.title}
                  </div>
                  <h3 className="text-xl font-bold mt-1 mb-1">{layer.subtitle}</h3>
                  <p className="text-xs text-fgdim font-mono">
                    Gồm {phasesInLayer.length} Phase đào tạo từ lý thuyết đến triển khai mã nguồn thực chiến.
                  </p>
                </div>

                <div className="space-y-6">
                  {phasesInLayer.map((phase) => {
                    const postsInPhase = POSTS.filter(
                      (p) => p.phaseNumber === phase.number || p.phase === phase.id
                    );
                    const isActive = phase.status === "active" || postsInPhase.length > 0;

                    return (
                      <div
                        key={phase.id}
                        className={`border rounded-lg p-5 bg-bg2 transition-all ${
                          isActive ? "border-line shadow-sm" : "border-line/40 opacity-75"
                        }`}
                      >
                        {/* Phase Header */}
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3 pb-3 border-b border-line/60">
                          <div>
                            <div className="flex items-center gap-2 flex-wrap mb-1">
                              <span className="font-mono text-xs font-bold text-accent bg-accent/10 border border-accent/30 rounded px-2 py-0.5">
                                Phase {phase.number}
                              </span>
                              <h4 className="text-base font-bold m-0">
                                {phase.titleVi} ({phase.title})
                              </h4>
                            </div>
                            <p className="text-xs text-fgdim m-0 leading-relaxed font-sans">{phase.desc}</p>
                          </div>

                          <div className="flex items-center gap-2 self-start sm:self-auto shrink-0">
                            <span
                              className={`font-mono text-[10.5px] rounded-full px-2.5 py-0.5 border ${
                                isActive
                                  ? "border-accent text-accent bg-accent/5 font-semibold"
                                  : "border-line text-fgdim bg-bg"
                              }`}
                            >
                              {isActive ? `Đang học · ${postsInPhase.length} bài` : "Kế hoạch (Planned)"}
                            </span>
                          </div>
                        </div>

                        {/* Phase Lessons / Notes */}
                        {postsInPhase.length > 0 ? (
                          <div className="grid gap-3 sm:grid-cols-2 mt-4">
                            {postsInPhase.map((p) => (
                              <PostCard key={p.slug} p={p} showPhaseBadge={false} />
                            ))}
                          </div>
                        ) : (
                          <div className="p-3 bg-bg/40 rounded border border-dashed border-line/60 font-mono text-xs text-fgdim flex items-center justify-between">
                            <span>Bài học và mã nguồn của Phase {phase.number} đang được biên soạn theo chuẩn thực chiến.</span>
                            <span className="text-[10px] text-accent">Sắp ra mắt</span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </section>
            );
          })}
        </div>
      )}
    </div>
  );
}
