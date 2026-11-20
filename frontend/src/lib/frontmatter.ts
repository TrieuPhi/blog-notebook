// Minimal YAML-frontmatter parser — enough for this content system's needs,
// avoids pulling in a Node-oriented gray-matter dependency in the browser.
export interface Frontmatter {
  title: string;
  description: string;
  date: string;
  updated?: string;
  category: string;
  tags: string[];
  status: string;
  featured?: boolean;
  readingTime: number;
  slug?: string;
  lang?: string;
  translationKey?: string;
  phase?: string;
  phaseNumber?: string;
  phaseTitle?: string;
  lesson?: string;
  type?: string;
  motto?: string;
  artifacts?: string[];
  keyPoints?: string[];
}

export function parseArrayValue(val: string): string[] {
  try {
    const parsed = JSON.parse(val);
    if (Array.isArray(parsed)) {
      return parsed.map((item) => String(item).trim()).filter(Boolean);
    }
  } catch {
    // fallback to regex-based tokenization
  }
  const inner = val.slice(1, -1).trim();
  if (!inner) return [];

  const items: string[] = [];
  const regex = /"([^"]*)"|'([^']*)'|([^,\s][^,]*?)(?:,\s*|$)/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(inner)) !== null) {
    const item = match[1] ?? match[2] ?? match[3];
    if (item !== undefined) {
      const clean = item.trim();
      if (clean) items.push(clean);
    }
    if (regex.lastIndex === match.index) {
      regex.lastIndex++;
    }
  }
  return items;
}

export function parseFrontmatter(raw: string): { data: Frontmatter; content: string } {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!match) {
    return {
      data: {
        title: "Untitled", description: "", date: "", updated: "", category: "General",
        tags: [], status: "note", featured: false, readingTime: 1,
      },
      content: raw,
    };
  }
  const [, fm, content] = match;
  const data: any = { tags: [] };
  const lines = fm.split(/\r?\n/);
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const kv = line.match(/^([A-Za-z]+):\s*(.*)$/);
    if (kv) {
      const key = kv[1];
      let val: any = kv[2].trim();
      if (val === "" ) {
        // possible list on following lines: "- item"
        const items: string[] = [];
        let j = i + 1;
        while (j < lines.length && /^\s*-\s*/.test(lines[j])) {
          items.push(lines[j].replace(/^\s*-\s*/, "").replace(/^["']|["']$/g, ""));
          j++;
        }
        if (items.length) { data[key] = items; i = j; continue; }
      }
      if (val.startsWith("[") && val.endsWith("]")) {
        data[key] = parseArrayValue(val);
      } else if (val === "true" || val === "false") {
        data[key] = val === "true";
      } else if (/^-?\d+$/.test(val)) {
        data[key] = parseInt(val, 10);
      } else {
        data[key] = val.replace(/^["']|["']$/g, "");
      }
    }
    i++;
  }
  return { data: data as Frontmatter, content: content.trim() };
}
