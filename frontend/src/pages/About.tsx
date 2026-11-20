import { Link } from "react-router-dom";
import { getNotebookStats } from "../lib/posts";

export default function About() {
  const stats = getNotebookStats();

  return (
    <div className="space-y-10">
      {/* Header breadcrumb */}
      <div className="font-mono text-[11.5px] text-fgdim mb-2 flex items-center gap-2">
        <span>ABOUT</span>
        <span>/</span>
        <span>ENGINEERING MANIFESTO</span>
        <span>/</span>
        <span className="text-accent">CORE VALUES</span>
      </div>

      {/* Hero Profile Section */}
      <div className="grid sm:grid-cols-[130px_1fr] gap-6 items-start pb-6 border-b border-line">
        <div className="w-[120px] h-[120px] rounded-xl bg-gradient-to-br from-accent to-amber flex flex-col items-center justify-center font-mono font-bold text-[color:var(--accent-fg)] shadow-md">
          <span className="text-[28px] leading-none">AI</span>
          <span className="text-[10px] tracking-widest mt-1 opacity-90">ENG</span>
        </div>

        <div>
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <h2 className="text-[28px] font-bold m-0">Phi Trieu Dai</h2>
            <span className="font-mono text-xs border border-accent/40 bg-accent/10 text-accent rounded-full px-2.5 py-0.5 font-medium">
              AI Engineer &amp; System Builder
            </span>
          </div>

          <p className="font-mono text-xs text-accent mb-3">
            Triết lý cốt lõi: &ldquo;Áp dụng được thực tế, đánh vào trọng tâm, ngắn gọn.&rdquo;
          </p>

          <p className="text-fgdim text-sm leading-relaxed m-0 font-sans">
            Tôi dành phần lớn thời gian xây dựng và đo lường định lượng các hệ thống RAG, fine-tune các mô hình ngôn ngữ nhỏ (SLM) và thiết kế hạ tầng phục vụ suy luận hiệu năng cao. Sổ tay kỹ thuật này là không gian ghi chép có hệ thống: nơi bóc tách bản chất thuật toán, lưu trữ các case study thực chiến và loại bỏ các lớp trừu tượng thừa thãi.
          </p>
        </div>
      </div>

      {/* Live Notebook Stats Widget */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-mono text-xs text-fgdim uppercase tracking-wider font-semibold m-0">
            Notebook Metrics (Số liệu Thời gian thực)
          </h3>
          <span className="font-mono text-[10.5px] text-accent">Vite Build-time Dynamic Computation</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="border border-line bg-bg2 rounded-lg p-4 shadow-sm">
            <div className="text-2xl font-bold font-mono text-accent">{stats.totalPosts}</div>
            <div className="text-xs font-mono text-fgdim mt-1">Bài viết đã lập chỉ mục</div>
          </div>

          <div className="border border-line bg-bg2 rounded-lg p-4 shadow-sm">
            <div className="text-2xl font-bold font-mono text-accent">
              {stats.activePhasesCount} <span className="text-xs font-normal text-fgdim">/ {stats.totalPhasesPlanned}</span>
            </div>
            <div className="text-xs font-mono text-fgdim mt-1">Phases Lộ trình (Active / Total)</div>
          </div>

          <div className="border border-line bg-bg2 rounded-lg p-4 shadow-sm">
            <div className="text-2xl font-bold font-mono text-amber">{stats.totalProjects}</div>
            <div className="text-xs font-mono text-fgdim mt-1">Case Studies Thực chiến</div>
          </div>

          <div className="border border-line bg-bg2 rounded-lg p-4 shadow-sm">
            <div className="text-2xl font-bold font-mono text-accent">~{stats.totalReadingMinutes}m</div>
            <div className="text-xs font-mono text-fgdim mt-1">Thời gian đọc tích lũy</div>
          </div>
        </div>
      </div>

      {/* Core Tech Stack */}
      <div className="border border-line bg-bg2 rounded-xl p-5 shadow-sm">
        <h3 className="font-mono text-xs text-fgdim uppercase tracking-wider font-semibold m-0 mb-3">
          Ngăn xếp Công nghệ Trọng tâm (Core Tech Stack)
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
          {stats.coreStack.map((item) => (
            <div key={item.name} className="border border-line/60 bg-bg/50 rounded-lg p-2.5 flex flex-col justify-between">
              <span className="font-mono text-xs font-bold text-fg">{item.name}</span>
              <span className="font-mono text-[10.5px] text-accent mt-0.5">{item.role}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Engineering Manifesto: 3 Pillars */}
      <div>
        <div className="mb-4">
          <div className="font-mono text-xs text-accent font-semibold uppercase tracking-wider">
            PHILOSOPHY &amp; METHODOLOGY
          </div>
          <h3 className="text-xl font-bold mt-1 mb-1">Tuyên ngôn Kỹ thuật (Engineering Manifesto)</h3>
          <p className="text-xs text-fgdim font-sans m-0">
            Ba nguyên lý bất biến định hình cách tiếp cận công nghệ, kiến trúc mã nguồn và phong cách viết trong sổ tay này.
          </p>
        </div>

        <div className="space-y-4">
          {/* Pillar 1 */}
          <div className="border border-line bg-bg2 rounded-xl p-5 shadow-sm">
            <div className="flex items-center gap-2 mb-2">
              <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-accent/15 text-accent border border-accent/30">
                TRỤ CỘT 01
              </span>
              <h4 className="text-base font-bold m-0">
                First Principles (Nguyên lý Đệ nhất — Hiểu từ Bản chất)
              </h4>
            </div>

            <blockquote className="border-l-2 border-accent bg-bg/60 px-3.5 py-2 my-2.5 rounded-r">
              <p className="text-xs italic text-fg/90 m-0 font-sans leading-relaxed">
                &ldquo;Không bao giờ tin tưởng một abstraction mà bạn chưa thể tự tay xây dựng từ đầu (Build from Scratch).&rdquo;
              </p>
            </blockquote>

            <p className="text-xs text-fgdim leading-relaxed font-sans mb-3">
              Trước khi dùng LangChain hay AutoGen, hãy tự viết vòng lặp ReAct bằng Python stdlib. Trước khi fine-tune một click, hãy hiểu toán học của LoRA rank decomposition và attention masks. Sự hiểu biết ở tầng bản chất giúp cô lập và sửa lỗi production trong 5 phút thay vì mất nhiều ngày lần mò trong các thư viện bên thứ ba.
            </p>

            <div className="font-mono text-[11px] text-accent bg-bg/60 border border-line rounded px-3 py-1.5 inline-block">
              Nguyên tắc: stdlib first → Hiểu toán học → Đo lường hiệu năng → Sau đó mới trừu tượng hóa
            </div>
          </div>

          {/* Pillar 2 */}
          <div className="border border-line bg-bg2 rounded-xl p-5 shadow-sm">
            <div className="flex items-center gap-2 mb-2">
              <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-amber/15 text-amber border border-amber/30">
                TRỤ CỘT 02
              </span>
              <h4 className="text-base font-bold m-0">
                Boring Infra, Reliable AI (Hạ tầng Ổn định, AI Tin cậy)
              </h4>
            </div>

            <blockquote className="border-l-2 border-amber bg-bg/60 px-3.5 py-2 my-2.5 rounded-r">
              <p className="text-xs italic text-fg/90 m-0 font-sans leading-relaxed">
                &ldquo;Mô hình ngôn ngữ là cỗ máy xác suất (non-deterministic); vì vậy hạ tầng bao quanh bắt buộc phải kỷ luật, tất định và có khả năng kiểm chứng.&rdquo;
              </p>
            </blockquote>

            <p className="text-xs text-fgdim leading-relaxed font-sans mb-3">
              Không chạy theo các bản demo hào nhoáng thiếu ổn định. Đặt trọng tâm vào: RAGAS / LLM-as-a-Judge đánh giá định lượng, synthetic golden testsets trong CI/CD pipeline, PagedAttention quản trị bộ nhớ VRAM, Kubernetes container serving có warm-up check và fallback an toàn.
            </p>

            <div className="font-mono text-[11px] text-amber bg-bg/60 border border-line rounded px-3 py-1.5 inline-block">
              Nguyên tắc: Cổng kiểm thử định lượng tự động &gt; Căn chỉnh prompt theo cảm tính
            </div>
          </div>

          {/* Pillar 3 */}
          <div className="border border-line bg-bg2 rounded-xl p-5 shadow-sm">
            <div className="flex items-center gap-2 mb-2">
              <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                TRỤ CỘT 03
              </span>
              <h4 className="text-base font-bold m-0">
                Actionable &amp; Concise (Thực chiến, Trọng tâm, Ngắn gọn)
              </h4>
            </div>

            <blockquote className="border-l-2 border-emerald-500 bg-bg/60 px-3.5 py-2 my-2.5 rounded-r">
              <p className="text-xs italic text-fg/90 m-0 font-sans leading-relaxed">
                &ldquo;Tôn trọng tuyệt đối thời gian của kỹ sư. Đánh thẳng vào trọng tâm, không sáo rỗng, luôn đi kèm Actionable Artifact.&rdquo;
              </p>
            </blockquote>

            <p className="text-xs text-fgdim leading-relaxed font-sans mb-3">
              Mỗi bài viết trong sổ tay này được thiết kế theo cấu trúc nghiêm ngặt: Vấn đề thực tế là gì? (The Problem) → Bản chất kỹ thuật vận hành thế nào? (The Concept) → Tự tay xây dựng từ đầu (Build It) → Đóng gói &amp; Tái sử dụng (Ship It). Không lý thuyết suông; mọi ghi chép đều đi kèm một script kiểm thử chạy được, một bảng checklist hoặc một Dockerfile sẵn sàng dùng ngay.
            </p>

            <div className="font-mono text-[11px] text-emerald-400 bg-bg/60 border border-line rounded px-3 py-1.5 inline-block">
              Nguyên tắc: Mã nguồn thực thi được &amp; Benchmark đo lường &gt; Từ khóa buzzwords sáo rỗng
            </div>
          </div>
        </div>
      </div>

      {/* Connect & Syndication Links */}
      <div className="border-t border-line pt-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h4 className="font-mono text-xs text-fg font-semibold uppercase tracking-wider m-0 mb-1">
            Kết nối &amp; Phân phối Tri thức
          </h4>
          <p className="text-xs text-fgdim m-0 font-sans">
            Theo dõi các bài viết mới qua giao thức RSS Feed hoặc khám phá cấu trúc đồ thị.
          </p>
        </div>

        <div className="flex gap-2 flex-wrap font-mono text-xs">
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="border border-line bg-bg2 rounded px-3 py-1.5 hover:border-accent hover:text-accent transition"
          >
            GitHub ↗
          </a>
          <a
            href="https://linkedin.com"
            target="_blank"
            rel="noreferrer"
            className="border border-line bg-bg2 rounded px-3 py-1.5 hover:border-accent hover:text-accent transition"
          >
            LinkedIn ↗
          </a>
          <a
            href="/rss.xml"
            target="_blank"
            rel="noreferrer"
            className="border border-amber/40 bg-amber/10 text-amber rounded px-3 py-1.5 hover:border-amber transition"
          >
            RSS Feed ⚡
          </a>
          <Link
            to="/graph"
            className="border border-accent bg-accent text-[color:var(--accent-fg)] rounded px-3 py-1.5 shadow-sm hover:opacity-90 transition"
          >
            Khám phá Graph →
          </Link>
        </div>
      </div>
    </div>
  );
}
