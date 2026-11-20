import { useEffect, useRef, useState } from "react";
import { defaultGiscusConfig, isPlaceholderConfig, mapBlogThemeToGiscus } from "../config/giscus";

interface GiscusCommentsProps {
  slug: string;
}

export default function GiscusComments({ slug }: GiscusCommentsProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaceholder] = useState(() => isPlaceholderConfig(defaultGiscusConfig));

  useEffect(() => {
    if (!containerRef.current) return;

    // Clear any previous Giscus instance
    containerRef.current.innerHTML = "";

    const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
    const giscusTheme = mapBlogThemeToGiscus(currentTheme);

    const script = document.createElement("script");
    script.src = "https://giscus.app/client.js";
    script.setAttribute("data-repo", defaultGiscusConfig.repo);
    script.setAttribute("data-repo-id", defaultGiscusConfig.repoId);
    script.setAttribute("data-category", defaultGiscusConfig.category);
    script.setAttribute("data-category-id", defaultGiscusConfig.categoryId);
    script.setAttribute("data-mapping", defaultGiscusConfig.mapping);
    script.setAttribute("data-term", slug);
    script.setAttribute("data-strict", defaultGiscusConfig.strict);
    script.setAttribute("data-reactions-enabled", defaultGiscusConfig.reactionsEnabled);
    script.setAttribute("data-emit-metadata", defaultGiscusConfig.emitMetadata);
    script.setAttribute("data-input-position", defaultGiscusConfig.inputPosition);
    script.setAttribute("data-theme", giscusTheme);
    script.setAttribute("data-lang", defaultGiscusConfig.lang);
    script.setAttribute("data-loading", defaultGiscusConfig.loading);
    script.crossOrigin = "anonymous";
    script.async = true;

    containerRef.current.appendChild(script);

    // Dynamic theme change synchronization via MutationObserver
    const observer = new MutationObserver(() => {
      const updatedTheme = document.documentElement.getAttribute("data-theme") || "light";
      const targetGiscusTheme = mapBlogThemeToGiscus(updatedTheme);
      const iframe = containerRef.current?.querySelector<HTMLIFrameElement>("iframe.giscus-frame");
      if (iframe?.contentWindow) {
        iframe.contentWindow.postMessage(
          { giscus: { setConfig: { theme: targetGiscusTheme } } },
          "https://giscus.app"
        );
      }
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    return () => {
      observer.disconnect();
    };
  }, [slug]);

  return (
    <div className="w-full">
      {isPlaceholder && (
        <div className="mb-4 p-3.5 border border-line bg-bg2 rounded text-xs font-mono text-fgdim">
          <span className="font-semibold text-amber">Notice:</span> Giscus comments are running in placeholder mode. Set <code className="text-accent">VITE_GISCUS_REPO</code>, <code className="text-accent">VITE_GISCUS_REPO_ID</code>, <code className="text-accent">VITE_GISCUS_CATEGORY</code>, and <code className="text-accent">VITE_GISCUS_CATEGORY_ID</code> in your <code className="text-accent">.env</code> to connect with your GitHub Discussions.
        </div>
      )}
      <div ref={containerRef} className="giscus-container min-h-[160px]" />
    </div>
  );
}
