import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendDir = path.resolve(__dirname, "..");
const postsDir = path.resolve(frontendDir, "posts");
const publicDir = path.resolve(frontendDir, "public");
const distDir = path.resolve(frontendDir, "dist");

const siteUrl = (process.env.VITE_SITE_URL || "https://setlog.dev").replace(/\/$/, "");

function escapeXml(unsafe) {
  return String(unsafe || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function getAllMarkdownFiles(dir) {
  let results = [];
  if (!fs.existsSync(dir)) return results;
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results = results.concat(getAllMarkdownFiles(fullPath));
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      results.push(fullPath);
    }
  }
  return results;
}

function parsePosts() {
  if (!fs.existsSync(postsDir)) {
    console.warn(`[generate-rss-sitemap] Warning: Posts directory not found at ${postsDir}`);
    return [];
  }
  const files = getAllMarkdownFiles(postsDir);
  const posts = [];

  for (const file of files) {
    const raw = fs.readFileSync(file, "utf-8");
    const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---/);
    if (!match) continue;

    const fm = match[1];
    const data = {};
    for (const line of fm.split(/\r?\n/)) {
      const kv = line.match(/^([A-Za-z]+):\s*(.*)$/);
      if (kv) {
        const key = kv[1];
        let val = kv[2].trim().replace(/^["']|["']$/g, "");
        if (val.startsWith("[") && val.endsWith("]")) {
          data[key] = val
            .slice(1, -1)
            .split(",")
            .map((s) => s.trim().replace(/^["']|["']$/g, ""));
        } else {
          data[key] = val;
        }
      }
    }

    if (data.status !== "published") continue;
    const slug = data.slug || path.basename(file, ".md");
    posts.push({ ...data, slug });
  }

  return posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

function generateRss(posts) {
  const lastBuildDate = new Date().toUTCString();
  const itemsXml = posts
    .map((post) => {
      const itemUrl = `${siteUrl}/article/${encodeURI(post.slug)}`;
      // Convert YYYY-MM-DD to UTC RFC-822 date
      const pubDate = new Date(post.date + "T00:00:00Z").toUTCString();
      return `    <item>
      <title>${escapeXml(post.title)}</title>
      <link>${itemUrl}</link>
      <guid isPermaLink="true">${itemUrl}</guid>
      <pubDate>${pubDate}</pubDate>
      <description>${escapeXml(post.description)}</description>
      <category>${escapeXml(post.category)}</category>
    </item>`;
    })
    .join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>set.log — notes on AI, engineering &amp; learning</title>
    <link>${siteUrl}</link>
    <description>Personal technical notebook on AI, LLM systems, RAG architecture, and engineering.</description>
    <language>vi</language>
    <lastBuildDate>${lastBuildDate}</lastBuildDate>
    <atom:link href="${siteUrl}/feed.xml" rel="self" type="application/rss+xml"/>
${itemsXml}
  </channel>
</rss>
`;
}

function generateSitemap(posts) {
  const staticRoutes = [
    { loc: `${siteUrl}/`, changefreq: "daily", priority: "1.0" },
    { loc: `${siteUrl}/blog`, changefreq: "daily", priority: "0.9" },
    { loc: `${siteUrl}/topics`, changefreq: "weekly", priority: "0.8" },
    { loc: `${siteUrl}/graph`, changefreq: "weekly", priority: "0.7" },
    { loc: `${siteUrl}/projects`, changefreq: "weekly", priority: "0.7" },
    { loc: `${siteUrl}/about`, changefreq: "monthly", priority: "0.6" },
  ];

  const staticXml = staticRoutes
    .map(
      (r) => `  <url>
    <loc>${r.loc}</loc>
    <changefreq>${r.changefreq}</changefreq>
    <priority>${r.priority}</priority>
  </url>`
    )
    .join("\n");

  const articlesXml = posts
    .map(
      (p) => `  <url>
    <loc>${siteUrl}/article/${encodeURI(p.slug)}</loc>
    <lastmod>${p.updated || p.date}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>`
    )
    .join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${staticXml}
${articlesXml}
</urlset>
`;
}

function generateRobots() {
  return `User-agent: *
Allow: /

Sitemap: ${siteUrl}/sitemap.xml
`;
}

function main() {
  if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir, { recursive: true });
  }

  const posts = parsePosts();
  console.log(`[generate-rss-sitemap] Found ${posts.length} published posts.`);

  const rssContent = generateRss(posts);
  const sitemapContent = generateSitemap(posts);
  const robotsContent = generateRobots();

  // Write to public/
  fs.writeFileSync(path.join(publicDir, "feed.xml"), rssContent, "utf-8");
  fs.writeFileSync(path.join(publicDir, "rss.xml"), rssContent, "utf-8");
  fs.writeFileSync(path.join(publicDir, "sitemap.xml"), sitemapContent, "utf-8");
  fs.writeFileSync(path.join(publicDir, "robots.txt"), robotsContent, "utf-8");
  console.log(`[generate-rss-sitemap] Generated feed.xml, rss.xml, sitemap.xml, robots.txt in public/`);

  // Write to dist/ if dist directory already exists
  if (fs.existsSync(distDir)) {
    fs.writeFileSync(path.join(distDir, "feed.xml"), rssContent, "utf-8");
    fs.writeFileSync(path.join(distDir, "rss.xml"), rssContent, "utf-8");
    fs.writeFileSync(path.join(distDir, "sitemap.xml"), sitemapContent, "utf-8");
    fs.writeFileSync(path.join(distDir, "robots.txt"), robotsContent, "utf-8");
    console.log(`[generate-rss-sitemap] Synchronized syndication files to dist/`);
  }
}

main();
