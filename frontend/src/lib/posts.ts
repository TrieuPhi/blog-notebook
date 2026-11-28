import { parseFrontmatter, Frontmatter } from "./frontmatter";

export interface Post extends Frontmatter {
  slug: string;
  content: string;
  motto?: string;
  artifacts: string[];
  keyPoints: string[];
}

// Vite loads every markdown file under /posts as raw text at build time (including subdirectories like /posts/phases/**).
const files = import.meta.glob("/posts/**/*.md", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

function slugify(path: string, customSlug?: string) {
  if (customSlug) return customSlug;
  return path.split("/").pop()!.replace(/\.md$/, "");
}

const POST_PHASE_MAP: Record<string, string> = {
  "01-dev-environment": "00-setup-and-tooling",
  "personal-knowledge-graph": "00-setup-and-tooling",
  "learning-python-deeply": "00-setup-and-tooling",
  "Machine Learning Fundamentals Notes": "02-ml-fundamentals",
  "gioi-thieu-deep-learning": "03-deep-learning-core",
  "qwen3-32b-sft": "11-llm-engineering",
  "rag-eval-pipeline": "11-llm-engineering",
  "ragas-evaluators": "11-llm-engineering",
  "01-the-agent-loop": "14-agent-engineering",
  "vllm-production": "17-infrastructure-and-production",
};

function extractPhase(path: string, customPhase?: string, slug?: string) {
  if (customPhase) return customPhase;
  const match = path.match(/\/posts\/phases\/([^/]+)/);
  if (match) return match[1];
  if (slug && POST_PHASE_MAP[slug]) return POST_PHASE_MAP[slug];
  return undefined;
}

function extractMotto(content: string, frontmatterMotto?: string): string {
  if (frontmatterMotto && frontmatterMotto.trim()) return frontmatterMotto.trim();

  const lines = content.split(/\r?\n/);
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (line.startsWith(">") && !line.startsWith(">>")) {
      const clean = line
        .replace(/^>\s*/, "")
        .replace(/^\*\*(?:\[?Motto\]?|Châm ngôn)?[:\s*]*\*\*/i, "")
        .replace(/\*\*/g, "")
        .trim();
      if (clean && !clean.startsWith("#") && !clean.startsWith("-")) {
        if (clean.length > 220) {
          return clean.slice(0, 217) + "…";
        }
        return clean;
      }
    }
  }

  return "Áp dụng được thực tế, đánh vào trọng tâm, ngắn gọn.";
}

function extractArtifacts(content: string, data: Partial<Frontmatter>): string[] {
  if (data.artifacts && Array.isArray(data.artifacts) && data.artifacts.length > 0) {
    return data.artifacts;
  }
  const artifacts: string[] = [];
  const text = content.toLowerCase();

  if (content.includes("```python") || content.includes("```ts") || content.includes("```bash") || content.includes("```json") || (data.type && data.type.toLowerCase().includes("build"))) {
    artifacts.push("Code mẫu");
  }
  if (content.includes("- [ ]") || content.includes("- [x]") || text.includes("checklist") || text.includes("cạm bẫy") || text.includes("pitfall") || text.includes("tiêu chí")) {
    artifacts.push("Checklist");
  }
  if (text.includes("docker") || text.includes("kubernetes") || text.includes("fastapi") || text.includes("cli") || text.includes("tool") || text.includes("vllm") || text.includes("ragas")) {
    artifacts.push("Tool");
  }
  if (text.includes("system prompt") || text.includes("prompt template") || text.includes("few-shot") || text.includes("judge prompt") || text.includes("inst:") || text.includes("instruction:")) {
    artifacts.push("Prompt");
  }
  if (content.includes("```mermaid") || text.includes("triad") || text.includes("kiến trúc") || text.includes("sơ đồ") || text.includes("bảng tra cứu")) {
    artifacts.push("Cheatsheet");
  }

  if (artifacts.length === 0) {
    artifacts.push("Code mẫu", "Checklist");
  }
  return artifacts.slice(0, 3);
}

function extractKeyPoints(content: string, data: Partial<Frontmatter>): string[] {
  if (data.keyPoints && Array.isArray(data.keyPoints) && data.keyPoints.length > 0) {
    return data.keyPoints;
  }
  const points: string[] = [];

  const bulletMatches = content.match(/^[-*]\s+([^\n\r]+)/gm);
  if (bulletMatches) {
    for (const b of bulletMatches) {
      const clean = b.replace(/^[-*]\s+/, "").replace(/\[[ x]\]\s*/, "").replace(/\*\*/g, "").trim();
      if (clean.length > 15 && !clean.startsWith("http") && !clean.includes("```")) {
        points.push(clean);
        if (points.length >= 3) break;
      }
    }
  }

  if (points.length < 3) {
    const headingMatches = content.match(/^#{2,3}\s+([^\n\r]+)/gm);
    if (headingMatches) {
      for (const h of headingMatches) {
        const clean = h.replace(/^#{2,3}\s+/, "").trim();
        if (clean && !["Thông tin nhanh", "Mục tiêu bài học", "The Problem", "The Concept", "Build It", "Ship It", "Tham khảo", "Related"].includes(clean)) {
          if (!points.includes(clean)) {
            points.push(clean);
            if (points.length >= 3) break;
          }
        }
      }
    }
  }

  if (points.length === 0) {
    if (data.description) points.push(data.description);
    points.push("Hiểu từ bản chất kỹ thuật trước khi dùng abstraction");
    points.push("Đo lường định lượng và tối ưu hóa hiệu năng trong production");
  }
  while (points.length < 3) {
    if (points.length === 1) points.push("Kiểm soát chất lượng và xử lý lỗi biên hệ thống");
    else points.push("Tối ưu hóa tài nguyên phần cứng và chi phí vận hành");
  }
  return points.slice(0, 3);
}

export const POSTS: Post[] = Object.entries(files)
  .map(([path, raw]) => {
    const { data, content } = parseFrontmatter(raw);
    const slug = slugify(path, data.slug);
    const phase = extractPhase(path, data.phase, slug);
    const phaseNumber = phase ? phase.split("-")[0] : data.phaseNumber;
    const motto = extractMotto(content, data.motto);
    const artifacts = extractArtifacts(content, data);
    const keyPoints = extractKeyPoints(content, data);

    return {
      ...data,
      slug,
      phase,
      phaseNumber,
      motto,
      artifacts,
      keyPoints,
      content,
    };
  })
  .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

export type TechnicalLayer = "Foundation" | "Core LLM & RAG" | "Agent Systems" | "Production / Infra";

export interface Topic {
  name: string;
  desc: string;
  layer: TechnicalLayer;
  tag: string;
}

export const TOPICS: Topic[] = [
  // 1. Foundation
  { name: "Developer Tools", desc: "CLI, uv package manager, profiling, Docker và môi trường AI chuẩn.", layer: "Foundation", tag: "Developer Tools" },
  { name: "Math & AI Foundations", desc: "Đại số tuyến tính, giải tích ma trận, xác suất, SVD và tối ưu hóa lồi từ bản chất.", layer: "Foundation", tag: "Math & AI Foundations" },
  { name: "Machine Learning", desc: "Nguyên lý máy học từ đầu: Linear/Logistic Regression, Trees, SVM, Ensemble và ML Pipelines.", layer: "Foundation", tag: "Machine Learning" },
  { name: "Python", desc: "Python 3.12 internals, memory model, concurrency và tối ưu hóa hiệu năng.", layer: "Foundation", tag: "Python" },
  { name: "Deep Learning", desc: "Mạng nơ-ron, backpropagation engine, autograd và tensor operations.", layer: "Foundation", tag: "Deep Learning" },

  // 2. Core LLM & RAG
  { name: "LLM", desc: "Kiến trúc Attention, Pre-training, fine-tuning và lượng tử hóa mô hình.", layer: "Core LLM & RAG", tag: "LLM" },
  { name: "RAG", desc: "Retrieval-Augmented Generation: chunking, hybrid search và kiến trúc RAG.", layer: "Core LLM & RAG", tag: "RAG" },
  { name: "RAGAS", desc: "Đánh giá RAG định lượng với LLM-as-a-Judge và synthetic golden testset.", layer: "Core LLM & RAG", tag: "RAGAS" },
  { name: "Fine-tuning (SFT)", desc: "Huấn luyện tinh chỉnh hiệu quả tham số LoRA, QLoRA và DeepSpeed ZeRO.", layer: "Core LLM & RAG", tag: "Fine-tuning" },

  // 3. Agent Systems
  { name: "Agent Engineering", desc: "Vòng lặp ReAct (Observe-Think-Act), turn budget và native reasoning.", layer: "Agent Systems", tag: "Agent Engineering" },
  { name: "ReAct Loop", desc: "Kiến trúc tác tử tự trị tối giản bằng Python stdlib zero-dependency.", layer: "Agent Systems", tag: "ReAct" },
  { name: "Tools & Protocols (MCP)", desc: "Model Context Protocol, function calling và schema validation.", layer: "Agent Systems", tag: "MCP" },
  { name: "Autonomous Systems", desc: "Hệ thống tự động hóa kiểm thử mã nguồn và đa tác tử cộng tác.", layer: "Agent Systems", tag: "Agent" },

  // 4. Production / Infra
  { name: "MLOps", desc: "CI/CD tự động hóa kiểm thử, giám sát độ trễ và model evaluation gates.", layer: "Production / Infra", tag: "MLOps" },
  { name: "vLLM Serving", desc: "PagedAttention, KV Cache memory management và continuous batching.", layer: "Production / Infra", tag: "vLLM" },
  { name: "Backend & API", desc: "FastAPI, asynchronous streaming inference và microservices.", layer: "Production / Infra", tag: "Backend" },
  { name: "DevOps & Infra", desc: "Kubernetes GPU orchestration, HPA autoscaling và Docker runtime.", layer: "Production / Infra", tag: "DevOps" },
];

export interface PhaseInfo {
  id: string;
  number: string;
  title: string;
  titleVi: string;
  desc: string;
  layer: TechnicalLayer;
  status: "active" | "planned";
}

export const PHASES: PhaseInfo[] = [
  // Foundation (Phases 00-04)
  { id: "00-setup-and-tooling", number: "00", title: "Setup & Tooling", titleVi: "Thiết lập Môi trường & Công cụ", desc: "Môi trường phát triển AI hiện đại, uv, Python 3.12, profiling, Docker và GPU CUDA/MPS", layer: "Foundation", status: "active" },
  { id: "01-math-foundations", number: "01", title: "Math Foundations", titleVi: "Toán học Nền tảng cho AI", desc: "Đại số tuyến tính, giải tích ma trận, autodiff, xác suất thống kê, SVD và tối ưu hóa lồi", layer: "Foundation", status: "active" },
  { id: "02-ml-fundamentals", number: "02", title: "ML Fundamentals", titleVi: "Máy học Thực chiến từ Đầu", desc: "Linear/Logistic Regression, Decision Trees, SVM, Ensemble, Pipelines, Drift & Feature Stores thuần NumPy", layer: "Foundation", status: "active" },
  { id: "03-deep-learning-core", number: "03", title: "Deep Learning Core", titleVi: "Học sâu & Động cơ Vi phân", desc: "Xây dựng autograd engine từ đầu, backpropagation, MLP, regularization và normalization", layer: "Foundation", status: "active" },
  { id: "04-computer-vision", number: "04", title: "Computer Vision", titleVi: "Thị giác Máy tính", desc: "Convolution từ đầu, ResNet, Vision Transformers (ViT) và object detection", layer: "Foundation", status: "planned" },

  // Core LLM & RAG (Phases 05-12)
  { id: "05-nlp-foundations-to-advanced", number: "05", title: "NLP Foundations", titleVi: "Xử lý Ngôn ngữ Tự nhiên", desc: "Tokenization, Word Embeddings, RNN/LSTM, Seq2Seq và cơ chế Attention cổ điển", layer: "Core LLM & RAG", status: "planned" },
  { id: "06-speech-and-audio", number: "06", title: "Speech & Audio", titleVi: "Xử lý Âm thanh & Giọng nói", desc: "STFT, Mel Spectrogram, ASR (Whisper) và tổng hợp giọng nói (TTS)", layer: "Core LLM & RAG", status: "planned" },
  { id: "07-transformers-deep-dive", number: "07", title: "Transformers Deep Dive", titleVi: "Giải mã Kiến trúc Transformer", desc: "Scaled Dot-Product Attention, Multi-Head Attention, RoPE, KV Cache và FlashAttention", layer: "Core LLM & RAG", status: "planned" },
  { id: "08-generative-ai", number: "08", title: "Generative AI", titleVi: "AI Tạo sinh & Mô hình Khuếch tán", desc: "VAE, GAN, Diffusion Models (DDPM), Flow Matching và sinh đa phương tiện", layer: "Core LLM & RAG", status: "planned" },
  { id: "09-reinforcement-learning", number: "09", title: "Reinforcement Learning", titleVi: "Học Tăng cường & Tối ưu Chính sách", desc: "MDP, Q-Learning, Policy Gradient, PPO, GRPO và ứng dụng trong RLHF/Reasoning", layer: "Core LLM & RAG", status: "planned" },
  { id: "10-llms-from-scratch", number: "10", title: "LLMs from Scratch", titleVi: "Xây dựng LLM từ Số 0", desc: "BPE Tokenizer, Causal LM Decoder, Pre-training loop, Scaling Laws và Checkpoint", layer: "Core LLM & RAG", status: "planned" },
  { id: "11-llm-engineering", number: "11", title: "LLM Engineering & RAG", titleVi: "Kỹ thuật LLM, Fine-tuning & RAG", desc: "SFT/LoRA/QLoRA, Hybrid RAG Pipelines, RAGAS Evaluation và LLM-as-a-Judge", layer: "Core LLM & RAG", status: "active" },
  { id: "12-multimodal-ai", number: "12", title: "Multimodal AI", titleVi: "AI Đa phương thức", desc: "CLIP, Vision-Language Models (VLM), Audio-Text Alignment và cross-modal attention", layer: "Core LLM & RAG", status: "planned" },

  // Agent Systems (Phases 13-16)
  { id: "13-tools-and-protocols", number: "13", title: "Tools & Protocols (MCP)", titleVi: "Gọi Công cụ & Giao thức MCP", desc: "Model Context Protocol (MCP), function calling, schema validation và sandbox execution", layer: "Agent Systems", status: "planned" },
  { id: "14-agent-engineering", number: "14", title: "Agent Engineering", titleVi: "Kỹ thuật Hệ thống Agent & ReAct", desc: "Vòng lặp ReAct (Observe-Think-Act), ReWOO, Reflexion, turn budget và memory", layer: "Agent Systems", status: "active" },
  { id: "15-autonomous-systems", number: "15", title: "Autonomous Systems", titleVi: "Hệ thống Tác tử Tự trị", desc: "SWE-bench workflows, self-healing code agents, planning trees và long-horizon tasks", layer: "Agent Systems", status: "planned" },
  { id: "16-multi-agent-and-swarms", number: "16", title: "Multi-Agent & Swarms", titleVi: "Hệ thống Đa Tác tử Cộng tác", desc: "Orchestrator-Workers, consensus protocols, debate loops và blackboard pattern", layer: "Agent Systems", status: "planned" },

  // Production / Infra (Phases 17-19)
  { id: "17-infrastructure-and-production", number: "17", title: "MLOps & Production Infra", titleVi: "Hạ tầng Suy luận & Vận hành MLOps", desc: "vLLM PagedAttention, continuous batching, quantization, Kubernetes GPU và CI/CD gates", layer: "Production / Infra", status: "active" },
  { id: "18-ethics-safety-alignment", number: "18", title: "AI Safety & Alignment", titleVi: "An toàn, Căn chỉnh & Bảo mật AI", desc: "Guardrails, prompt injection defense, red-teaming, RLHF/DPO và đánh giá rủi ro", layer: "Production / Infra", status: "planned" },
  { id: "19-capstone-projects", number: "19", title: "Capstone Projects", titleVi: "Dự án Tổng hợp Thực chiến", desc: "Thiết kế và triển khai end-to-end các hệ thống AI Production hoàn chỉnh", layer: "Production / Infra", status: "planned" },
];

export interface ArchitectureNote {
  title: string;
  detail: string;
  postSlug: string;
}

export interface QuickLinks {
  github?: string;
  demo?: string;
  article?: string;
}

export interface ProblemSolved {
  problem: string;
  solution: string;
  impact: string;
}

export interface Project {
  id: string;
  name: string;
  tagline: string;
  desc?: string;
  status: "Active" | "Production" | "Experimental" | "Maintained";
  category: string;
  problemSolved: ProblemSolved;
  stack: string[];
  architectureNotes: ArchitectureNote[];
  quickLinks: QuickLinks;
  posts: string[];
}

export const PROJECTS: Project[] = [
  {
    id: "ragas-slm-evaluator",
    name: "RAGAS SLM Evaluator",
    tagline: "Khung đánh giá RAG tự động hóa dùng SLM chuyên biệt thay thế LLM thương mại",
    desc: "A lightweight evaluator that swaps GPT-4-class judges for a fine-tuned small model, cutting eval cost by ~8x.",
    status: "Active",
    category: "RAG & Evaluation",
    problemSolved: {
      problem: "Đánh giá pipeline RAG trong CI/CD bằng GPT-4 gây tốn kém hàng nghìn USD/tháng và độ trễ P99 > 3.5s cho mỗi testcase.",
      solution: "Fine-tune mô hình nhỏ (Qwen-7B) chuyên trách làm LLM-as-a-Judge theo chuẩn RAG Triad (Faithfulness, Relevancy, Precision).",
      impact: "Giảm 8.2x chi phí suy luận, tăng tốc độ eval 4.5x, đạt 94.2% correlation so với GPT-4.",
    },
    stack: ["Python 3.12", "RAGAS", "Qwen 2.5", "vLLM", "Hugging Face", "PyTorch"],
    architectureNotes: [
      {
        title: "LLM-as-a-Judge & RAG Triad Math",
        detail: "Phân tích toán học đằng sau Faithfulness, Answer Relevancy và hiện tượng ảo giác.",
        postSlug: "ragas-evaluators",
      },
      {
        title: "Synthetic Golden Testset CI/CD Gate",
        detail: "Tự động hóa pipeline đánh giá chất lượng RAG trong GitHub Actions.",
        postSlug: "rag-eval-pipeline",
      },
    ],
    quickLinks: {
      github: "https://github.com",
      demo: "/article/ragas-evaluators",
      article: "/article/ragas-evaluators",
    },
    posts: ["ragas-evaluators", "rag-eval-pipeline"],
  },
  {
    id: "vllm-deployment-kit",
    name: "vLLM Production Deployment Kit",
    tagline: "Bộ kit triển khai vLLM chuẩn hóa trên Kubernetes với PagedAttention và auto-scaling",
    desc: "A reusable Docker + Kubernetes setup for serving open-weight LLMs with autoscaling and warm-pool health checks.",
    status: "Production",
    category: "MLOps & Serving",
    problemSolved: {
      problem: "Cấp phát tĩnh KV Cache lãng phí 60-80% VRAM GPU, gây Out-Of-Memory khi phục vụ đồng thời nhiều phiên hội thoại dài.",
      solution: "Triển khai vLLM với thuật toán PagedAttention, Continuous Batching và Kubernetes HPA tự động co giãn theo tỷ lệ dùng KV Cache.",
      impact: "Tăng 3.8x throughput phục vụ đồng thời, P99 TTFT < 140ms trên cụm GPU NVIDIA A100.",
    },
    stack: ["vLLM", "Docker", "Kubernetes", "FastAPI", "Prometheus", "NVIDIA Triton"],
    architectureNotes: [
      {
        title: "PagedAttention & Quản trị Bộ nhớ KV Cache",
        detail: "Mô hình phân trang bộ nhớ ảo giải quyết triệt để phân mảnh VRAM trong suy luận LLM.",
        postSlug: "vllm-production",
      },
    ],
    quickLinks: {
      github: "https://github.com",
      demo: "/article/vllm-production",
      article: "/article/vllm-production",
    },
    posts: ["vllm-production"],
  },
  {
    id: "stdlib-agent-runtime",
    name: "Pure stdlib Agent Loop Runtime",
    tagline: "Runtime tác tử AI tối giản triển khai chu trình ReAct thuần Python stdlib zero-dependency",
    desc: "A pure Python stdlib implementation of the ReAct autonomous loop with strict turn budgeting and native reasoning parsing.",
    status: "Active",
    category: "Agent Engineering",
    problemSolved: {
      problem: "Các framework Agent bên ngoài quá cồng kềnh, tạo lớp trừu tượng dày đặc khiến việc gỡ lỗi vòng lặp vô tận và rò rỉ token trở nên bất khả thi.",
      solution: "Tự tay xây dựng chu trình ReAct (Observe -> Think -> Act) thuần Python stdlib với cơ chế giới hạn ngân sách bước (turn budget) và tách biệt token suy luận.",
      impact: "100% zero-dependency ngoại vi, thời gian khởi động 0ms, tiêu thụ RAM < 15MB, kiểm soát toàn diện luồng gọi công cụ.",
    },
    stack: ["Python 3.12 stdlib", "ReAct Pattern", "Tool Calling", "JSON Schema"],
    architectureNotes: [
      {
        title: "Vòng lặp ReAct và Turn Budget từ Bản chất",
        detail: "Bóc tách cấu trúc luồng suy luận của Claude Code, Cursor và kỹ thuật xử lý token suy luận nguyên bản.",
        postSlug: "01-the-agent-loop",
      },
    ],
    quickLinks: {
      github: "https://github.com",
      demo: "/article/01-the-agent-loop",
      article: "/article/01-the-agent-loop",
    },
    posts: ["01-the-agent-loop"],
  },
  {
    id: "qwen-sft-peft-lab",
    name: "Qwen SFT & PEFT Optimization Lab",
    tagline: "Quy trình tinh chỉnh mô hình 14B-32B tối ưu VRAM với QLoRA 4-bit và DeepSpeed ZeRO-3",
    desc: "A collection of SFT/LoRA experiments across Qwen model sizes, with eval deltas tracked per run.",
    status: "Experimental",
    category: "Fine-Tuning & Modeling",
    problemSolved: {
      problem: "Huấn luyện tinh chỉnh các mô hình 14B-32B đòi hỏi hạ tầng GPU trung tâm dữ liệu đắt đỏ (> 80GB VRAM mỗi node).",
      solution: "Kết hợp lượng tử hóa 4-bit (QLoRA), Low-Rank Adaptation và kỹ thuật phân tán DeepSpeed ZeRO-3 Memory Offload.",
      impact: "Tiết kiệm 72% VRAM, cho phép fine-tune Qwen-32B hoàn chỉnh trên workstation 2x RTX 4090 (24GB VRAM).",
    },
    stack: ["PyTorch", "Hugging Face PEFT", "DeepSpeed ZeRO-3", "BitsAndBytes", "Qwen-32B"],
    architectureNotes: [
      {
        title: "Toán học LoRA & Tối ưu Bộ nhớ DeepSpeed",
        detail: "Phân tích Rank Decomposition, ma trận A/B và cơ chế phân mảnh trọng số gradient qua GPU.",
        postSlug: "qwen3-32b-sft",
      },
    ],
    quickLinks: {
      github: "https://github.com",
      demo: "/article/qwen3-32b-sft",
      article: "/article/qwen3-32b-sft",
    },
    posts: ["qwen3-32b-sft"],
  },
  {
    id: "notebook-graph-engine",
    name: "Notebook Knowledge Graph Engine",
    tagline: "Động cơ tri thức đồ thị hai chiều phân tích AST Markdown và liên kết Wikilinks thời gian thực",
    desc: "The link-and-backlink engine behind this site — parses frontmatter and wikilinks into a graph.",
    status: "Maintained",
    category: "Knowledge Systems",
    problemSolved: {
      problem: "Ghi chép kỹ thuật phân tán khiến tri thức bị cô lập thành các ốc đảo rời rạc, thiếu liên kết ngữ nghĩa giữa bài toán nền tảng và ứng dụng thực tế.",
      solution: "Bộ phân tích cú pháp AST tối giản trích xuất frontmatter, phân tầng kỹ thuật và đồ thị liên kết hai chiều Wikilinks ([[...]]) thời gian thực.",
      impact: "Hoàn toàn tĩnh (static-first), zero-database, thời gian dựng đồ thị < 50ms cho mạng lưới hàng trăm bài viết.",
    },
    stack: ["TypeScript", "React", "Interactive SVG", "Markdown AST", "Tailwind CSS"],
    architectureNotes: [
      {
        title: "Personal Knowledge Graph Schema & Backlinks",
        detail: "Mô hình dữ liệu đồ thị liên kết, trích xuất hai chiều và trực quan hóa tương tác.",
        postSlug: "personal-knowledge-graph",
      },
    ],
    quickLinks: {
      github: "https://github.com",
      demo: "/graph",
      article: "/article/personal-knowledge-graph",
    },
    posts: ["personal-knowledge-graph"],
  },
];

export function getPost(slug: string) {
  return POSTS.find((p) => p.slug === slug);
}

export function relatedFor(p: Post, limit = 4) {
  return POSTS.filter((o) => o.slug !== p.slug && (o.category === p.category || o.tags.some((t) => p.tags.includes(t))))
    .sort((a, b) => b.tags.filter((t) => p.tags.includes(t)).length - a.tags.filter((t) => p.tags.includes(t)).length)
    .slice(0, limit);
}

// Backlinks: posts whose content contains a [[Wikilink]] or markdown link to this post's title/slug.
export function backlinksFor(p: Post) {
  return POSTS.filter((o) => o.slug !== p.slug && (
    o.content.includes(`[[${p.title}]]`) || o.content.includes(`(/article/${p.slug})`)
  ));
}

// Replace [[Wikilink Title]] with a markdown link to the matching post, resolved by title.
export function resolveWikilinks(content: string): string {
  return content.replace(/\[\[([^\]]+)\]\]/g, (_m, title) => {
    const target = POSTS.find((p) => p.title.toLowerCase() === title.trim().toLowerCase());
    return target ? `[${title}](/article/${target.slug})` : `**${title}**`;
  });
}

export function getNotebookStats() {
  const totalPosts = POSTS.length;
  const activePhases = [...new Set(POSTS.map((p) => p.phase).filter(Boolean))];
  const totalReadingMinutes = POSTS.reduce((sum, p) => sum + (p.readingTime || 0), 0);
  const totalProjects = PROJECTS.length;
  const totalTopics = TOPICS.length;
  const coreStack = [
    { name: "Python 3.12 (uv)", role: "AI Core & Pipeline" },
    { name: "PyTorch & vLLM", role: "Inference Engine" },
    { name: "RAGAS & Judges", role: "Quantitative Eval" },
    { name: "Docker & K8s", role: "Production Infra" },
    { name: "TypeScript & React", role: "UI & Knowledge Graph" },
    { name: "PEFT & DeepSpeed", role: "Model Optimization" },
  ];
  return {
    totalPosts,
    activePhasesCount: activePhases.length,
    totalPhasesPlanned: PHASES.length,
    totalReadingMinutes,
    totalProjects,
    totalTopics,
    coreStack,
  };
}
