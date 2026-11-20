import { useNavigate } from "react-router-dom";
import { POSTS, TOPICS, TechnicalLayer } from "../lib/posts";

interface LayerSection {
  key: TechnicalLayer;
  index: string;
  title: string;
  titleVi: string;
  desc: string;
}

const LAYERS: LayerSection[] = [
  {
    key: "Foundation",
    index: "01",
    title: "Foundation",
    titleVi: "Nền tảng Kỹ thuật & Bản chất Máy học",
    desc: "Cấu trúc dữ liệu, giải thuật, toán học ma trận và giải mã các tầng sâu trong Python & Học sâu.",
  },
  {
    key: "Core LLM & RAG",
    index: "02",
    title: "Core LLM & RAG",
    titleVi: "Mô hình Ngôn ngữ Lớn & Hệ thống RAG",
    desc: "Cơ chế Attention, tinh chỉnh tham số hiệu quả (PEFT/LoRA) và đánh giá định lượng pipeline RAG.",
  },
  {
    key: "Agent Systems",
    index: "03",
    title: "Agent Systems",
    titleVi: "Hệ thống Tác tử Tự trị & Vòng lặp ReAct",
    desc: "Tác tử ra quyết định độc lập, giao thức công cụ ngoại vi (MCP) và phối hợp đa tác tử.",
  },
  {
    key: "Production / Infra",
    index: "04",
    title: "Production / Infra",
    titleVi: "Vận hành MLOps & Hạ tầng Phục vụ Suy luận",
    desc: "Triển khai vLLM tốc độ cao, quản trị bộ nhớ GPU VRAM, kiểm soát chất lượng CI/CD và Kubernetes.",
  },
];

export default function Topics() {
  const nav = useNavigate();

  const handleTopicClick = (topicName: string) => {
    nav(`/blog?topic=${encodeURIComponent(topicName)}`);
  };

  return (
    <div>
      {/* Page Header */}
      <div className="font-mono text-[11.5px] text-fgdim mb-2 flex items-center gap-2">
        <span>TOPICS</span>
        <span>/</span>
        <span>4 TECHNICAL LAYERS</span>
        <span>/</span>
        <span>{TOPICS.length} CORE DOMAINS</span>
      </div>

      <div className="mb-8">
        <h2 className="text-[28px] font-bold m-0">Phân tầng Kỹ thuật &amp; Chủ đề</h2>
        <p className="text-fgdim text-sm mt-1 max-w-[70ch]">
          Toàn bộ tri thức và bài viết được sắp xếp theo 4 tầng kiến trúc công nghệ AI thực chiến: từ nền tảng máy học, mô hình ngôn ngữ, hệ thống tác tử cho đến vận hành hạ tầng quy mô lớn.
        </p>
      </div>

      {/* 4 Layers Layout */}
      <div className="space-y-10">
        {LAYERS.map((layer) => {
          const topicsInLayer = TOPICS.filter((t) => t.layer === layer.key);

          return (
            <section key={layer.key} className="border-t border-line pt-6">
              {/* Layer Title and Description */}
              <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 mb-4">
                <div>
                  <div className="font-mono text-xs text-accent font-semibold uppercase tracking-wider flex items-center gap-2">
                    <span>TẦNG {layer.index}</span>
                    <span>•</span>
                    <span>{layer.title}</span>
                  </div>
                  <h3 className="text-lg font-bold mt-1 mb-1">{layer.titleVi}</h3>
                  <p className="text-xs text-fgdim font-sans m-0">{layer.desc}</p>
                </div>
                <div className="font-mono text-[11px] text-fgdim shrink-0">
                  {topicsInLayer.length} chủ đề trọng tâm
                </div>
              </div>

              {/* Topics Grid */}
              <div className="grid sm:grid-cols-2 gap-3.5">
                {topicsInLayer.map((t) => {
                  const matchingCount = POSTS.filter((p) => {
                    const term = t.name.toLowerCase();
                    const tagTerm = t.tag.toLowerCase();
                    return (
                      p.category.toLowerCase() === term ||
                      p.category.toLowerCase() === tagTerm ||
                      p.tags.some((tag) => tag.toLowerCase() === tagTerm) ||
                      p.tags.some((tag) => tag.toLowerCase() === term) ||
                      p.title.toLowerCase().includes(term) ||
                      p.title.toLowerCase().includes(tagTerm)
                    );
                  }).length;

                  return (
                    <button
                      key={t.name}
                      onClick={() => handleTopicClick(t.name)}
                      className="group text-left border border-line bg-bg2 rounded-lg p-4 shadow-sm hover:border-accent hover:-translate-y-0.5 transition duration-150 flex flex-col justify-between"
                    >
                      <div>
                        <div className="flex justify-between items-center mb-1.5">
                          <span className="font-mono text-[11px] text-accent font-semibold">
                            {matchingCount} {matchingCount === 1 ? "bài viết" : "bài viết"}
                          </span>
                          <span className="font-mono text-[10px] text-fgdim border border-line/60 rounded px-1.5 py-0.5">
                            #{t.tag}
                          </span>
                        </div>
                        <h4 className="text-[16px] font-semibold m-0 group-hover:text-accent transition-colors">
                          {t.name}
                        </h4>
                        <p className="text-fgdim text-[13px] my-2 leading-relaxed font-sans">{t.desc}</p>
                      </div>

                      <div className="pt-2 border-t border-line/40 flex items-center justify-between font-mono text-[11px] text-fgdim group-hover:text-accent">
                        <span>Lọc bài viết theo chủ đề này</span>
                        <span className="group-hover:translate-x-1 transition-transform">→</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
