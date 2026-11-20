export interface NavRoute {
  path: string;
  changefreq: "always" | "hourly" | "daily" | "weekly" | "monthly" | "yearly" | "never";
  priority: number;
}

export interface SiteConfig {
  title: string;
  tagline: string;
  description: string;
  siteUrl: string;
  author: string;
  navRoutes: NavRoute[];
}

export const SITE_CONFIG: SiteConfig = {
  title: "set.log",
  tagline: "notes on AI, engineering & learning",
  description: "Personal technical notebook on AI, LLM systems, RAG architecture, and engineering.",
  siteUrl: (import.meta.env.VITE_SITE_URL || "https://setlog.dev").replace(/\/$/, ""),
  author: "Phi Trieu Dai",
  navRoutes: [
    { path: "/", changefreq: "daily", priority: 1.0 },
    { path: "/blog", changefreq: "daily", priority: 0.9 },
    { path: "/topics", changefreq: "weekly", priority: 0.8 },
    { path: "/graph", changefreq: "weekly", priority: 0.7 },
    { path: "/projects", changefreq: "weekly", priority: 0.7 },
    { path: "/about", changefreq: "monthly", priority: 0.6 },
  ],
};
