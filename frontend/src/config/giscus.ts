export interface GiscusConfig {
  repo: string;
  repoId: string;
  category: string;
  categoryId: string;
  mapping: "pathname" | "url" | "title" | "og:title" | "specific" | "number";
  strict: "0" | "1";
  reactionsEnabled: "0" | "1";
  emitMetadata: "0" | "1";
  inputPosition: "top" | "bottom";
  theme: string;
  lang: string;
  loading: "lazy" | "eager";
}

export const defaultGiscusConfig: GiscusConfig = {
  repo: import.meta.env.VITE_GISCUS_REPO || "your-github-username/your-blog-repo",
  repoId: import.meta.env.VITE_GISCUS_REPO_ID || "R_kgDOXXXXXXXXXX",
  category: import.meta.env.VITE_GISCUS_CATEGORY || "Announcements",
  categoryId: import.meta.env.VITE_GISCUS_CATEGORY_ID || "DIC_kwDOXXXXXXXXXX",
  mapping: (import.meta.env.VITE_GISCUS_MAPPING as GiscusConfig["mapping"]) || "pathname",
  strict: (import.meta.env.VITE_GISCUS_STRICT as "0" | "1") || "0",
  reactionsEnabled: (import.meta.env.VITE_GISCUS_REACTIONS_ENABLED as "0" | "1") || "1",
  emitMetadata: (import.meta.env.VITE_GISCUS_EMIT_METADATA as "0" | "1") || "0",
  inputPosition: (import.meta.env.VITE_GISCUS_INPUT_POSITION as "top" | "bottom") || "bottom",
  theme: import.meta.env.VITE_GISCUS_THEME || "light",
  lang: import.meta.env.VITE_GISCUS_LANG || "vi",
  loading: "lazy",
};

export function isPlaceholderConfig(config: GiscusConfig = defaultGiscusConfig): boolean {
  return (
    config.repo.includes("your-github-username") ||
    config.repoId.includes("XXXXXX") ||
    config.categoryId.includes("XXXXXX")
  );
}

export function mapBlogThemeToGiscus(theme: string | null): string {
  if (theme === "dark") return "dark";
  return "light";
}
