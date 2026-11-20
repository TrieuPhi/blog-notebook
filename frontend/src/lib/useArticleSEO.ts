import { useEffect } from "react";
import type { Post } from "./posts";
import { SITE_CONFIG } from "../config/site";

export function useArticleSEO(post: Post | undefined) {
  useEffect(() => {
    if (!post) {
      document.title = `Note not found — ${SITE_CONFIG.title}`;
      return;
    }

    const defaultTitle = `${SITE_CONFIG.title} — ${SITE_CONFIG.tagline}`;
    const articleTitle = `${post.title} — ${SITE_CONFIG.title}`;
    const articleUrl = `${SITE_CONFIG.siteUrl}/article/${encodeURI(post.slug)}`;

    document.title = articleTitle;

    const setMeta = (attr: "name" | "property", key: string, content: string): HTMLMetaElement => {
      let el = document.querySelector<HTMLMetaElement>(`meta[${attr}="${key}"]`);
      if (!el) {
        el = document.createElement("meta");
        el.setAttribute(attr, key);
        document.head.appendChild(el);
      }
      el.setAttribute("content", content);
      return el;
    };

    const setLink = (rel: string, href: string): HTMLLinkElement => {
      let el = document.querySelector<HTMLLinkElement>(`link[rel="${rel}"]`);
      if (!el) {
        el = document.createElement("link");
        el.setAttribute("rel", rel);
        document.head.appendChild(el);
      }
      el.setAttribute("href", href);
      return el;
    };

    // Standard SEO
    setMeta("name", "description", post.description);
    setLink("canonical", articleUrl);

    // OpenGraph
    setMeta("property", "og:type", "article");
    setMeta("property", "og:title", post.title);
    setMeta("property", "og:description", post.description);
    setMeta("property", "og:url", articleUrl);
    setMeta("property", "og:site_name", SITE_CONFIG.title);
    setMeta("property", "article:published_time", post.date);
    if (post.updated) {
      setMeta("property", "article:modified_time", post.updated);
    }
    setMeta("property", "article:section", post.category);

    // Article Tags
    const createdTagElements: HTMLMetaElement[] = [];
    post.tags.forEach((tag) => {
      const tagEl = document.createElement("meta");
      tagEl.setAttribute("property", "article:tag");
      tagEl.setAttribute("content", tag);
      document.head.appendChild(tagEl);
      createdTagElements.push(tagEl);
    });

    // Twitter Card
    setMeta("name", "twitter:card", "summary_large_image");
    setMeta("name", "twitter:title", post.title);
    setMeta("name", "twitter:description", post.description);

    return () => {
      document.title = defaultTitle;
      setMeta("name", "description", SITE_CONFIG.description);
      setMeta("property", "og:type", "website");
      setMeta("property", "og:title", defaultTitle);
      setMeta("property", "og:description", SITE_CONFIG.description);
      setMeta("property", "og:url", SITE_CONFIG.siteUrl);
      setLink("canonical", SITE_CONFIG.siteUrl);

      // Clean up dynamic tags
      createdTagElements.forEach((el) => {
        if (el.parentNode) {
          el.parentNode.removeChild(el);
        }
      });
      const publishedTimeEl = document.querySelector('meta[property="article:published_time"]');
      if (publishedTimeEl?.parentNode) publishedTimeEl.parentNode.removeChild(publishedTimeEl);
      const modifiedTimeEl = document.querySelector('meta[property="article:modified_time"]');
      if (modifiedTimeEl?.parentNode) modifiedTimeEl.parentNode.removeChild(modifiedTimeEl);
      const sectionEl = document.querySelector('meta[property="article:section"]');
      if (sectionEl?.parentNode) sectionEl.parentNode.removeChild(sectionEl);
    };
  }, [post]);
}
