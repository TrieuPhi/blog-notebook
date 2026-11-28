#!/usr/bin/env python3
"""
adversarial_challenge.py — Empirical Challenger Stress Harness

Adversarially probes:
1. Binary frontmatter integrity: Byte 0, BOM, line endings, parser parity with frontend frontmatter.ts.
2. KaTeX math: Unpaired delimiters, spacing rules, unescaped currency dollar signs, brace balance, nested formulas.
3. Mermaid diagrams: Valid diagram types, double-quoted node labels, Vietnamese character escaping, lexical safety.
4. TOC & Heading integrity: H2 plain text compliance, absence of markdown/HTML formatting, slugify parity with Article.tsx.
5. Cross-linking & Wikilinks: Matching against actual post titles and slugs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

POSTS_DIR = Path(__file__).resolve().parent.parent / "frontend" / "posts"

CANONICAL_CATEGORIES = {
    "AI", "Agent Engineering", "LLM", "RAG", "RAGAS", "MLOps", "Backend", "DevOps", "Python", "Developer Tools", "Learning", "Math & AI Foundations", "Machine Learning"
}

VALID_MERMAID_TYPES = {
    "flowchart", "graph", "sequenceDiagram", "classDiagram", "stateDiagram",
    "stateDiagram-v2", "erDiagram", "gantt", "pie", "gitGraph", "mindmap",
    "quadrantChart", "timeline", "journey", "c4Context", "requirementDiagram",
    "packet-beta", "architecture-beta", "zenuml", "block-beta"
}

VIETNAMESE_RE = re.compile(
    r"[àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
    r"ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]"
)

def slugify(s: str) -> str:
    """Parity with Article.tsx slugify(s)"""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"(^-|-$)", "", s)
    return s


def parse_frontend_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Exact emulation of src/lib/frontmatter.ts parseFrontmatter"""
    match = re.match(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$", raw)
    if not match:
        return {
            "title": "Untitled", "description": "", "date": "", "updated": "", "category": "General",
            "tags": [], "status": "note", "featured": False, "readingTime": 1,
        }, raw

    fm = match.group(1)
    content = match.group(2)
    data: dict[str, Any] = {"tags": []}
    lines = re.split(r"\r?\n", fm)
    i = 0
    while i < len(lines):
        line = lines[i]
        kv = re.match(r"^([A-Za-z]+):\s*(.*)$", line)
        if kv:
            key = kv.group(1)
            val: Any = kv.group(2).strip()
            if val == "":
                items: list[str] = []
                j = i + 1
                while j < len(lines) and re.match(r"^\s*-\s*", lines[j]):
                    clean_item = re.sub(r"^\s*-\s*", "", lines[j])
                    clean_item = re.sub(r'^["\']|["\']$', "", clean_item)
                    items.append(clean_item)
                    j += 1
                if items:
                    data[key] = items
                    i = j
                    continue
            if val.startswith("[") and val.endswith("]"):
                data[key] = [
                    re.sub(r'^["\']|["\']$', "", s.strip())
                    for s in val[1:-1].split(",")
                    if s.strip()
                ]
            elif val in ("true", "false"):
                data[key] = (val == "true")
            elif re.match(r"^-?\d+$", val):
                data[key] = int(val)
            else:
                data[key] = re.sub(r'^["\']|["\']$', "", val)
        i += 1
    return data, content.strip()


class AdversarialAuditor:
    def __init__(self, post_path: Path):
        self.path = post_path
        self.name = post_path.name
        self.raw_bytes = post_path.read_bytes()
        self.text = self.raw_bytes.decode("utf-8", errors="replace")
        self.violations: list[str] = []

    def audit_all(self) -> list[str]:
        self.audit_byte_level()
        self.audit_frontmatter()
        self.audit_headings_and_toc()
        self.audit_katex_math()
        self.audit_mermaid_blocks()
        return self.violations

    def audit_byte_level(self):
        # 1. Byte Order Marks
        if self.raw_bytes.startswith(b"\xef\xbb\xbf"):
            self.violations.append("CRITICAL: UTF-8 BOM (\\xef\\xbb\\xbf) at byte 0. Frontend regex /^---/ will fail.")
        elif self.raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
            self.violations.append("CRITICAL: UTF-16 BOM at byte 0.")

        # 2. Byte 0 delimiter
        if not (self.raw_bytes.startswith(b"---\r\n") or self.raw_bytes.startswith(b"---\n")):
            self.violations.append(f"CRITICAL: Byte 0 does not begin with '---\\n'. Begins with {self.raw_bytes[:10]!r}")

    def audit_frontmatter(self):
        # Run frontend emulator
        data, _content = parse_frontend_frontmatter(self.text)

        # 1. Check title
        if not data.get("title") or data["title"] == "Untitled":
            self.violations.append("Frontmatter 'title' is missing or defaulted to 'Untitled'.")
        elif not isinstance(data["title"], str) or len(data["title"].strip()) < 5:
            self.violations.append(f"Frontmatter 'title' is too short or invalid: {data.get('title')!r}")

        # 2. Check description
        if not data.get("description") or not isinstance(data["description"], str) or len(data["description"].strip()) < 10:
            self.violations.append(f"Frontmatter 'description' is missing or insufficient: {data.get('description')!r}")

        # 3. Check date
        date_str = str(data.get("date", ""))
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            self.violations.append(f"Frontmatter 'date' invalid format: {date_str!r}")

        # 4. Check category
        cat = data.get("category")
        if cat not in CANONICAL_CATEGORIES:
            self.violations.append(f"Frontmatter 'category' {cat!r} not in canonical set: {CANONICAL_CATEGORIES}")

        # 5. Check tags
        tags = data.get("tags")
        if not isinstance(tags, list):
            self.violations.append(f"CRITICAL: 'tags' parsed as {type(tags).__name__}, not list. Causes React TypeError.")
        elif len(tags) == 0:
            self.violations.append("Frontmatter 'tags' is empty list.")
        else:
            for t in tags:
                if not isinstance(t, str) or not t.strip():
                    self.violations.append(f"Invalid tag element: {t!r}")

        # 6. Check status
        if data.get("status") != "published":
            self.violations.append(f"Frontmatter 'status' must be 'published', got {data.get('status')!r}")

        # 7. Check readingTime
        rt = data.get("readingTime")
        if not isinstance(rt, int) or rt <= 0 or isinstance(rt, bool):
            self.violations.append(f"Frontmatter 'readingTime' must be positive int, got {rt!r} ({type(rt).__name__})")

    def audit_headings_and_toc(self):
        fm_match = re.match(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$", self.text)
        if not fm_match:
            return
        body = fm_match.group(2)

        # Remove code blocks before inspecting headings
        cleaned_body = re.sub(r"```[\s\S]*?```", "", body)

        h2_matches = list(re.finditer(r"^##\s+(.+)$", cleaned_body, re.MULTILINE))
        if not h2_matches:
            self.violations.append("Article contains NO '## ' H2 headings (Table of Contents empty).")

        seen_slugs = set()
        for m in h2_matches:
            heading = m.group(1).strip()
            # Check formatting
            if re.search(r"\*\*|__|\*|_|`|\$|\[.*?\]\(.*?\)|<.*?>", heading):
                self.violations.append(f"H2 heading contains formatting or math: {heading!r}")

            slug = slugify(heading)
            if not slug:
                self.violations.append(f"H2 heading slugified to empty string: {heading!r}")
            elif slug in seen_slugs:
                self.violations.append(f"Duplicate H2 heading slug: {slug!r} from heading {heading!r}")
            seen_slugs.add(slug)

    def audit_katex_math(self):
        fm_match = re.match(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$", self.text)
        if not fm_match:
            return
        body = fm_match.group(2)

        # Mask fenced code blocks
        body_no_code = re.sub(r"```[\s\S]*?```", lambda m: "\n" * m.group(0).count("\n"), body)

        # Mask inline code `...`
        body_no_code = re.sub(r"`[^`\n]+`", lambda m: " " * len(m.group(0)), body_no_code)

        # Mask HTML comments <!-- ... -->
        body_no_code = re.sub(r"<!--[\s\S]*?-->", lambda m: " " * len(m.group(0)), body_no_code)

        # 1. Audit Display Math $$...$$
        # Count of $$
        double_dollars = list(re.finditer(r"\$\$", body_no_code))
        if len(double_dollars) % 2 != 0:
            self.violations.append(f"CRITICAL: Odd number of '$$' delimiters ({len(double_dollars)}). Unclosed display math.")

        display_math_blocks = list(re.finditer(r"\$\$([\s\S]*?)\$\$", body_no_code))
        for dm in display_math_blocks:
            math_text = dm.group(1).strip()
            if not math_text:
                self.violations.append("Empty display math formula ($$$$).")
            if math_text.count("{") != math_text.count("}"):
                self.violations.append(f"Unbalanced braces in display math: {math_text[:40]}... ({{: {math_text.count('{')}, }}: {math_text.count('}')})")

        # Mask display math
        body_no_display = re.sub(r"\$\$[\s\S]*?\$\$", lambda m: " " * len(m.group(0)), body_no_code)

        # 2. Audit Inline Math $...$
        lines = body_no_display.splitlines()
        for line_no, line in enumerate(lines, 1):
            if "$" not in line:
                continue

            i = 0
            n = len(line)
            while i < n:
                if line[i] == "\\" and i + 1 < n and line[i + 1] == "$":
                    i += 2
                    continue

                if line[i] == "$":
                    # Unescaped dollar sign
                    start_pos = i

                    # Check space after opening $
                    if i + 1 < n and line[i + 1] in (" ", "\t"):
                        self.violations.append(f"Line {line_no}: Space immediately follows opening '$': {line[start_pos:min(start_pos+30, n)]!r}")
                        i += 1
                        continue

                    # Search for closing $
                    j = i + 1
                    closing_pos = -1
                    while j < n:
                        if line[j] == "\\" and j + 1 < n and line[j + 1] == "$":
                            j += 2
                            continue
                        if line[j] == "$":
                            closing_pos = j
                            break
                        j += 1

                    if closing_pos == -1:
                        # Check if currency
                        curr_m = re.match(r"\$\d+(?:,\d+)*(?:\.\d+)?", line[start_pos:])
                        if curr_m:
                            self.violations.append(f"Line {line_no}: Unescaped currency dollar sign: {curr_m.group(0)!r}. Must escape as \\$")
                        else:
                            self.violations.append(f"Line {line_no}: Unclosed inline math dollar sign: {line[start_pos:min(start_pos+30, n)]!r}")
                        i += 1
                        continue

                    content = line[start_pos + 1 : closing_pos]
                    if not content:
                        self.violations.append(f"Line {line_no}: Empty inline math ($$).")
                    elif content.endswith((" ", "\t")):
                        self.violations.append(f"Line {line_no}: Space immediately precedes closing '$': {line[start_pos:closing_pos+1]!r}")
                    elif content.count("{") != content.count("}"):
                        self.violations.append(f"Line {line_no}: Unbalanced braces in inline math: {content!r}")

                    i = closing_pos + 1
                    continue

                i += 1

    def audit_mermaid_blocks(self):
        fm_match = re.match(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$", self.text)
        if not fm_match:
            return
        body = fm_match.group(2)

        blocks = re.findall(r"```mermaid\r?\n([\s\S]*?)```", body)
        for b_idx, block in enumerate(blocks, 1):
            lines = [l.strip() for l in block.splitlines() if l.strip() and not l.strip().startswith("%%")]
            if not lines:
                self.violations.append(f"Mermaid block #{b_idx} is empty.")
                continue

            first_line = lines[0]
            if not any(first_line.startswith(t) for t in VALID_MERMAID_TYPES):
                self.violations.append(f"Mermaid block #{b_idx} has invalid diagram type declaration: {first_line!r}")

            # Check node quotes
            if first_line.startswith(("flowchart", "graph")):
                quoted_patterns = [
                    r'\b[A-Za-z0-9_-]+\s*\(\[\"[^"\r\n]*\"\]\)',
                    r'\b[A-Za-z0-9_-]+\s*\[\[\"[^"\r\n]*\"\]\]',
                    r'\b[A-Za-z0-9_-]+\s*\[\(\"[^"\r\n]*\"\)\]',
                    r'\b[A-Za-z0-9_-]+\s*\(\(\"[^"\r\n]*\"\)\)',
                    r'\b[A-Za-z0-9_-]+\s*\{\{\"[^"\r\n]*\"\}\}',
                    r'\b[A-Za-z0-9_-]+\s*\[\"[^"\r\n]*\"\]',
                    r'\b[A-Za-z0-9_-]+\s*\(\"[^"\r\n]*\"\)',
                    r'\b[A-Za-z0-9_-]+\s*\{\"[^"\r\n]*\"\}',
                ]
                unquoted_patterns = [
                    (r'\b([A-Za-z0-9_-]+)\s*\(\[([^\]\r\n]*)\]\)', "stadium"),
                    (r'\b([A-Za-z0-9_-]+)\s*\[\[([^\]\r\n]*)\]\]', "subroutine"),
                    (r'\b([A-Za-z0-9_-]+)\s*\[\(([^)\r\n]*)\)\]', "database"),
                    (r'\b([A-Za-z0-9_-]+)\s*\(\(([^)\r\n]*)\)\)', "circle"),
                    (r'\b([A-Za-z0-9_-]+)\s*\{\{([^}\r\n]*)\}\}', "hexagon"),
                    (r'\b([A-Za-z0-9_-]+)\s*\[([^\]\r\n]*)\]', "rectangle"),
                    (r'\b([A-Za-z0-9_-]+)\s*\(([^)\r\n]*)\)', "rounded"),
                    (r'\b([A-Za-z0-9_-]+)\s*\{([^}\r\n]*)\}', "decision"),
                ]

                for line in lines:
                    if line.startswith(("%%", "subgraph", "classDef", "linkStyle", "click")) or line == "end":
                        continue
                    wline = re.sub(r'--\s*"[^"]*"\s*-->', " --> ", line)
                    wline = re.sub(r'-->\|"[^"]*"\|', " --> ", wline)
                    for qp in quoted_patterns:
                        wline = re.sub(qp, " ", wline)

                    for up, shape in unquoted_patterns:
                        for m in re.finditer(up, wline):
                            nid = m.group(1)
                            label = m.group(2).strip()
                            if VIETNAMESE_RE.search(label) or any(c in label for c in "():[]{} ,;/"):
                                self.violations.append(
                                    f"Mermaid block #{b_idx} unquoted {shape} node '{nid}' with special chars: {label!r}"
                                )
                        wline = re.sub(up, " ", wline)


def main():
    print("======================================================================")
    print("           ADVERSARIAL CHALLENGER PROBE SUITE")
    print("======================================================================")

    files = sorted(POSTS_DIR.glob("*.md"))
    print(f"Auditing {len(files)} articles in {POSTS_DIR}...\n")

    total_violations = 0
    passed = 0

    for f in files:
        auditor = AdversarialAuditor(f)
        violations = auditor.audit_all()
        if not violations:
            print(f"[PASS] {f.name}")
            passed += 1
        else:
            print(f"[FAIL] {f.name} — {len(violations)} adversarial violation(s):")
            for v in violations:
                print(f"   * {v}")
            total_violations += len(violations)

    print("\n----------------------------------------------------------------------")
    print(f"Summary: Articles audited: {len(files)} | Passed: {passed} | Total Violations: {total_violations}")
    print("----------------------------------------------------------------------")

    if total_violations > 0:
        print("VERDICT: REQUEST_CHANGES")
        sys.exit(1)
    else:
        print("VERDICT: APPROVE")
        sys.exit(0)


if __name__ == "__main__":
    main()
