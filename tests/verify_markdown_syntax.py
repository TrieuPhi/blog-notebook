#!/usr/bin/env python3
r"""
verify_markdown_syntax.py — Automated Markdown Syntax Validator

Validates markdown files in `blog-notebook/frontend/posts/` for:
  1. KaTeX Math Syntax:
     - Inline math ($...$): Must not have space after opening '$' or before closing '$'
       (remark-math requirement). No empty inline math ($$).
     - Display math ($$...$$): Balanced delimiters, non-empty formulas, balanced braces.
     - Unescaped currency dollar signs (e.g. $100) that risk unintended math spans.
  2. Mermaid Diagram Syntax:
     - Fenced code blocks with `mermaid` tag.
     - Valid diagram type declaration on first line.
     - Node labels containing Vietnamese characters, parentheses, colons, or punctuation
       MUST be wrapped in double quotes (e.g. `A["Dữ liệu (Input)"]`).
  3. Headings (`##` H2):
     - Must be plain text.
     - Must NOT contain bold (** / __), italic (* / _), inline code (`...`), or math ($...$).
     - Preserves exact Table of Contents anchor matching in `Article.tsx`.

Exit code:
  0: All articles pass syntax validation.
  1: One or more syntax violations detected.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Reconfigure stdout/stderr to UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


class SyntaxViolation:
    """Represents a markdown syntax violation."""

    def __init__(self, file_name: str, line_no: int, message: str, snippet: str = ""):
        self.file_name = file_name
        self.line_no = line_no
        self.message = message
        self.snippet = snippet

    def __str__(self) -> str:
        loc = f"line {self.line_no}" if self.line_no > 0 else "general"
        if self.snippet:
            return f"[{loc}] {self.message} -> `{self.snippet}`"
        return f"[{loc}] {self.message}"


# Valid diagram types supported by Mermaid 11.x
VALID_MERMAID_TYPES = (
    "flowchart",
    "graph",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "stateDiagram-v2",
    "erDiagram",
    "gantt",
    "pie",
    "gitGraph",
    "mindmap",
    "quadrantChart",
    "timeline",
    "journey",
    "c4Context",
    "requirementDiagram",
    "packet-beta",
    "architecture-beta",
    "zenuml",
    "block-beta",
)

# Vietnamese characters / accents regex
VIETNAMESE_CHARS = re.compile(
    r"[àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
    r"ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]"
)


def split_frontmatter_and_content(text: str) -> tuple[str, str, int]:
    """
    Separates frontmatter from markdown body.
    Returns (frontmatter_text, body_text, body_start_line_number).
    """
    match = re.match(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$", text)
    if not match:
        return "", text, 1
    fm_lines = match.group(1).count("\n") + 2  # opening --- and closing ---
    return match.group(1), match.group(2), fm_lines + 1


def extract_code_blocks(body: str, start_line: int) -> tuple[list[tuple[int, int, str, str]], list[tuple[int, str]]]:
    """
    Extracts fenced code blocks and plain lines from markdown body.
    Returns:
      fenced_blocks: List of (start_line, end_line, lang, code_content)
      plain_lines: List of (line_no, line_text) outside fenced code blocks
    """
    fenced_blocks: list[tuple[int, int, str, str]] = []
    plain_lines: list[tuple[int, str]] = []

    lines = body.splitlines()
    in_block = False
    fence_char = ""
    block_lang = ""
    block_start = 0
    block_lines: list[str] = []

    for idx, line in enumerate(lines, start=start_line):
        stripped = line.strip()
        fence_match = re.match(r"^(`{3,}|~{3,})(.*)$", stripped)

        if not in_block:
            if fence_match:
                in_block = True
                fence_char = fence_match.group(1)[0]
                block_lang = fence_match.group(2).strip().lower()
                block_start = idx
                block_lines = []
            else:
                plain_lines.append((idx, line))
        else:
            # Check closing fence
            if stripped.startswith(fence_char * 3) and len(stripped.replace(fence_char, "").strip()) == 0:
                in_block = False
                fenced_blocks.append((block_start, idx, block_lang, "\n".join(block_lines)))
            else:
                block_lines.append(line)

    return fenced_blocks, plain_lines


def validate_headings(plain_lines: list[tuple[int, str]], file_name: str) -> list[SyntaxViolation]:
    """
    Validates ## (H2) headings:
    Must be plain text without bold (** or __), italic (* or _), code (`...`), or math ($...$).
    Reason: Article.tsx generates anchor IDs via slugify(children), and ReactMarkdown renders
    formatted text into nested objects resulting in "[object Object]" IDs that break TOC navigation.
    """
    violations: list[SyntaxViolation] = []

    for line_no, line in plain_lines:
        stripped = line.strip()
        if stripped.startswith("## ") and not stripped.startswith("### "):
            heading_text = stripped[3:].strip()

            # Check bold (** or __)
            if re.search(r"\*\*[^*]+\*\*|__[^_]+__", heading_text):
                violations.append(
                    SyntaxViolation(
                        file_name,
                        line_no,
                        "H2 heading contains bold formatting (** or __). "
                        "H2 headings MUST be plain text to ensure Table of Contents anchor matching.",
                        heading_text,
                    )
                )

            # Check inline code (`...`)
            if re.search(r"`[^`]+`", heading_text):
                violations.append(
                    SyntaxViolation(
                        file_name,
                        line_no,
                        "H2 heading contains inline code (`...`). "
                        "H2 headings MUST be plain text to prevent [object Object] TOC anchor failure.",
                        heading_text,
                    )
                )

            # Check math ($...$)
            if "$" in heading_text:
                violations.append(
                    SyntaxViolation(
                        file_name,
                        line_no,
                        "H2 heading contains math delimiter ($). "
                        "Math is not supported in H2 headings because it breaks Table of Contents slug matching.",
                        heading_text,
                    )
                )

            # Check markdown links ([...](...))
            if re.search(r"\[.+?\]\(.+?\)", heading_text):
                violations.append(
                    SyntaxViolation(
                        file_name,
                        line_no,
                        "H2 heading contains markdown link. Headings must be plain text.",
                        heading_text,
                    )
                )

            # Check HTML tags (<...>)
            if re.search(r"<[a-zA-Z\/][^>]*>", heading_text):
                violations.append(
                    SyntaxViolation(
                        file_name,
                        line_no,
                        "H2 heading contains HTML tags. Headings must be plain text.",
                        heading_text,
                    )
                )

    return violations


def validate_katex_syntax(plain_lines: list[tuple[int, str]], file_name: str) -> list[SyntaxViolation]:
    r"""
    Validates KaTeX math formulas in markdown body lines outside code blocks:
    1. Display math ($$...$$): Balanced delimiters, non-empty formulas, balanced braces.
    2. Inline math ($...$): Tokenizer-based validation ensuring:
       - No space after opening '$' (e.g. '$ x$' is invalid).
       - No space before closing '$' (e.g. '$x $' is invalid).
       - No empty inline math ($$).
       - Unescaped currency dollar signs (e.g. $100) are flagged.
       - Balanced braces {} inside math expressions.
    """
    violations: list[SyntaxViolation] = []

    # Join plain lines to validate multi-line display math
    text_buffer: list[str] = []
    line_map: list[int] = []

    for line_no, line in plain_lines:
        text_buffer.append(line)
        line_map.append(line_no)

    full_text = "\n".join(text_buffer)

    # 1. Check total display math ($$) count
    display_matches = list(re.finditer(r"\$\$", full_text))
    if len(display_matches) % 2 != 0:
        last_match = display_matches[-1]
        line_idx = full_text[: last_match.start()].count("\n")
        line_no = line_map[line_idx] if line_idx < len(line_map) else line_map[-1]
        violations.append(
            SyntaxViolation(
                file_name,
                line_no,
                "Unmatched or unclosed display math delimiter '$$'. Count of '$$' in file is odd.",
                "$$",
            )
        )

    # 2. Extract and check display math blocks
    display_block_regex = re.compile(r"\$\$([\s\S]*?)\$\$")
    for m in display_block_regex.finditer(full_text):
        content = m.group(1).strip()
        line_idx = full_text[: m.start()].count("\n")
        line_no = line_map[line_idx] if line_idx < len(line_map) else line_map[-1]

        if not content:
            violations.append(
                SyntaxViolation(
                    file_name,
                    line_no,
                    "Empty display math formula ($$$$).",
                    "$$$$",
                )
            )
        else:
            if content.count("{") != content.count("}"):
                violations.append(
                    SyntaxViolation(
                        file_name,
                        line_no,
                        f"Unbalanced curly braces in display math ({content.count('{')} '{{' vs {content.count('}')} '}}'). "
                        "KaTeX will fail to compile this formula.",
                        content[:60] + ("..." if len(content) > 60 else ""),
                    )
                )

    # Mask display math before checking inline math (preserving newlines to prevent line_map drift)
    masked_text = display_block_regex.sub(lambda m: "".join("\n" if c == "\n" else " " for c in m.group(0)), full_text)

    # Mask inline code spans `...`
    masked_text = re.sub(r"`[^`\n]+`", lambda m: " " * len(m.group(0)), masked_text)

    # Mask HTML comments <!-- ... --> (preserving newlines to prevent line_map drift)
    masked_text = re.sub(r"<!--[\s\S]*?-->", lambda m: "".join("\n" if c == "\n" else " " for c in m.group(0)), masked_text)

    # 3. Tokenize line-by-line for inline math
    masked_lines = masked_text.splitlines()

    for idx, mline in enumerate(masked_lines):
        line_no = line_map[idx] if idx < len(line_map) else line_map[-1]
        if not mline.strip():
            continue

        i = 0
        n = len(mline)
        while i < n:
            # Check escaped dollar: \$
            if mline[i] == "\\" and i + 1 < n and mline[i + 1] == "$":
                i += 2
                continue

            if mline[i] == "$":
                start_pos = i

                # Case A: Space immediately follows opening '$'
                if i + 1 < n and mline[i + 1] in (" ", "\t"):
                    # Look for the next closing $ on this line to report the full snippet
                    closing_pos = -1
                    j = i + 2
                    while j < n:
                        if mline[j] == "\\" and j + 1 < n and mline[j + 1] == "$":
                            j += 2
                            continue
                        if mline[j] == "$":
                            closing_pos = j
                            break
                        j += 1

                    if closing_pos != -1:
                        snippet = mline[start_pos : closing_pos + 1]
                        violations.append(
                            SyntaxViolation(
                                file_name,
                                line_no,
                                "Invalid KaTeX inline math: space immediately follows opening '$' ('$ ...$'). "
                                "remark-math requires no space after opening '$' (e.g. '$x$' not '$ x$').",
                                snippet,
                            )
                        )
                        i = closing_pos + 1
                        continue
                    else:
                        violations.append(
                            SyntaxViolation(
                                file_name,
                                line_no,
                                "Unpaired single '$' delimiter followed by whitespace.",
                                mline[start_pos : min(start_pos + 30, n)],
                            )
                        )
                        i += 1
                        continue

                # Case B: Opening '$' followed by non-space
                closing_pos = -1
                j = i + 1
                while j < n:
                    if mline[j] == "\\" and j + 1 < n and mline[j + 1] == "$":
                        j += 2
                        continue
                    if mline[j] == "$":
                        closing_pos = j
                        break
                    j += 1

                if closing_pos != -1:
                    # Valid math span boundary found!
                    content = mline[start_pos + 1 : closing_pos]
                    if not content:
                        violations.append(
                            SyntaxViolation(
                                file_name,
                                line_no,
                                "Empty inline math formula ($$).",
                                "$$",
                            )
                        )
                    elif content.endswith((" ", "\t")):
                        violations.append(
                            SyntaxViolation(
                                file_name,
                                line_no,
                                "Invalid KaTeX inline math: space immediately precedes closing '$' ('$... $'). "
                                "remark-math requires no space before closing '$'.",
                                mline[start_pos : closing_pos + 1],
                            )
                        )
                    else:
                        # Valid inline formula: check balanced braces
                        if content.count("{") != content.count("}"):
                            violations.append(
                                SyntaxViolation(
                                    file_name,
                                    line_no,
                                    f"Unbalanced curly braces in inline math (${content}$).",
                                    f"${content}$",
                                )
                            )
                    i = closing_pos + 1
                    continue
                else:
                    # No closing $ on this line: unpaired single dollar sign
                    currency_m = re.match(r"\$\d+(?:,\d+)*(?:\.\d+)?", mline[start_pos:])
                    if currency_m:
                        violations.append(
                            SyntaxViolation(
                                file_name,
                                line_no,
                                "Unescaped currency dollar sign detected. Escape literal '$' as '\\$'.",
                                currency_m.group(0),
                            )
                        )
                    else:
                        violations.append(
                            SyntaxViolation(
                                file_name,
                                line_no,
                                "Unpaired single '$' delimiter detected on line.",
                                mline[start_pos : min(start_pos + 30, n)],
                            )
                        )
                    i += 1
                    continue
            i += 1

    return violations


def validate_mermaid_syntax(fenced_blocks: list[tuple[int, int, str, str]], file_name: str) -> list[SyntaxViolation]:
    r"""
    Validates Mermaid diagrams in fenced code blocks:
    1. First line must declare a recognized Mermaid diagram type.
    2. Node labels with Vietnamese characters, parentheses, colons, or punctuation
       MUST be enclosed in double quotes (e.g. `A["Dữ liệu (Input)"]`).
       Without quotes, characters like `(`, `)`, `:`, or accents break Mermaid's lexical parser
       and cause `Mermaid.tsx` to fail with `<pre>Diagram failed to render</pre>`.
    """
    violations: list[SyntaxViolation] = []

    mermaid_blocks = [b for b in fenced_blocks if b[2] == "mermaid"]

    for start_line, end_line, _, code in mermaid_blocks:
        lines = code.splitlines()
        non_empty = [(i, l.strip()) for i, l in enumerate(lines) if l.strip() and not l.strip().startswith("%%")]

        if not non_empty:
            violations.append(
                SyntaxViolation(
                    file_name,
                    start_line,
                    "Empty Mermaid diagram block.",
                    "```mermaid",
                )
            )
            continue

        # 1. Validate diagram type
        first_line_idx, first_line = non_empty[0]
        if not any(first_line.startswith(t) for t in VALID_MERMAID_TYPES):
            valid_types_str = ", ".join(VALID_MERMAID_TYPES[:8]) + "..."
            violations.append(
                SyntaxViolation(
                    file_name,
                    start_line + first_line_idx + 1,
                    f"Unknown or invalid Mermaid diagram type '{first_line}'. "
                    f"Must start with one of: [{valid_types_str}]",
                    first_line,
                )
            )

        # 2. Check node labels in graph / flowchart diagrams
        is_flowchart = first_line.startswith(("flowchart", "graph"))
        if is_flowchart:
            # First, list of double-quoted shape patterns (VALID by definition)
            QUOTED_SHAPES = [
                r'\b[A-Za-z0-9_-]+\s*\(\[\"[^"\r\n]*\"\]\)',  # Stadium: ID(["..."])
                r'\b[A-Za-z0-9_-]+\s*\[\[\"[^"\r\n]*\"\]\]',  # Subroutine: ID[["..."]]
                r'\b[A-Za-z0-9_-]+\s*\[\(\"[^"\r\n]*\"\)\]',  # Database: ID[("...")]
                r'\b[A-Za-z0-9_-]+\s*\(\(\"[^"\r\n]*\"\)\)',  # Circle: ID(("..."))
                r'\b[A-Za-z0-9_-]+\s*\{\{\"[^"\r\n]*\"\}\}',  # Hexagon: ID{{"..."}}
                r'\b[A-Za-z0-9_-]+\s*\[\"[^"\r\n]*\"\]',      # Rectangle: ID["..."]
                r'\b[A-Za-z0-9_-]+\s*\(\"[^"\r\n]*\"\)',      # Rounded: ID("...")
                r'\b[A-Za-z0-9_-]+\s*\{\"[^"\r\n]*\"\}',      # Decision: ID{"..."}
            ]

            # List of unquoted shape patterns to inspect for violations
            UNQUOTED_SHAPES = [
                (r'\b([A-Za-z0-9_-]+)\s*\(\[([^\]\r\n]*)\]\)', "stadium"),
                (r'\b([A-Za-z0-9_-]+)\s*\[\[([^\]\r\n]*)\]\]', "subroutine"),
                (r'\b([A-Za-z0-9_-]+)\s*\[\(([^)\r\n]*)\)\]', "database"),
                (r'\b([A-Za-z0-9_-]+)\s*\(\(([^)\r\n]*)\)\)', "circle"),
                (r'\b([A-Za-z0-9_-]+)\s*\{\{([^}\r\n]*)\}\}', "hexagon"),
                (r'\b([A-Za-z0-9_-]+)\s*\[([^\]\r\n]*)\]', "rectangle"),
                (r'\b([A-Za-z0-9_-]+)\s*\(([^)\r\n]*)\)', "rounded"),
                (r'\b([A-Za-z0-9_-]+)\s*\{([^}\r\n]*)\}', "decision"),
            ]

            for rel_idx, line in enumerate(lines):
                actual_line_no = start_line + rel_idx + 1
                stripped = line.strip()
                if (
                    not stripped
                    or stripped.startswith(("%%", "subgraph", "classDef", "linkStyle", "click"))
                    or stripped == "end"
                ):
                    continue

                # Mask out edge labels like: -- "label" --> or -->|"label"|
                working_line = re.sub(r'--\s*"[^"]*"\s*-->', " --> ", stripped)
                working_line = re.sub(r'-->\|"[^"]*"\|', " --> ", working_line)

                # Mask out all properly double-quoted node definitions
                for qpat in QUOTED_SHAPES:
                    working_line = re.sub(qpat, " ", working_line)

                # Inspect any remaining unquoted node definitions
                for upat, shape_name in UNQUOTED_SHAPES:
                    for m in re.finditer(upat, working_line):
                        node_id = m.group(1)
                        label = m.group(2).strip()

                        # Check for Vietnamese characters or risky punctuation
                        has_vietnamese = bool(VIETNAMESE_CHARS.search(label))
                        has_risky_punct = any(ch in label for ch in [":", "(", ")", "[", "]", "{", "}", ";", "/", ","])

                        if has_vietnamese or has_risky_punct:
                            violations.append(
                                SyntaxViolation(
                                    file_name,
                                    actual_line_no,
                                    f"Mermaid node label with special characters or Vietnamese text must be double-quoted. "
                                    f"Use `{node_id}[\"...\" ]` instead of unquoted label to prevent render failure.",
                                    m.group(0),
                                )
                            )
                    # Mask matched unquoted shape to avoid sub-matching
                    working_line = re.sub(upat, " ", working_line)

    return violations


def validate_file(file_path: Path) -> list[SyntaxViolation]:
    """
    Validates markdown syntax for a single article file.
    Returns list of violations (empty if all checks pass).
    """
    violations: list[SyntaxViolation] = []

    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as e:
        return [SyntaxViolation(file_path.name, 0, f"Failed to read file: {e}")]

    _, body, start_line = split_frontmatter_and_content(text)
    fenced_blocks, plain_lines = extract_code_blocks(body, start_line)

    # 1. Validate H2 headings
    heading_violations = validate_headings(plain_lines, file_path.name)
    violations.extend(heading_violations)

    # 2. Validate KaTeX syntax
    katex_violations = validate_katex_syntax(plain_lines, file_path.name)
    violations.extend(katex_violations)

    # 3. Validate Mermaid syntax
    mermaid_violations = validate_mermaid_syntax(fenced_blocks, file_path.name)
    violations.extend(mermaid_violations)

    return violations


def validate_posts_directory(directory_path: Path) -> tuple[int, int, dict[str, list[SyntaxViolation]]]:
    """
    Validates all .md files in the specified directory.
    Returns (pass_count, fail_count, results_dict).
    """
    if not directory_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    md_files = sorted(directory_path.glob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown (.md) files found in: {directory_path}")

    results: dict[str, list[SyntaxViolation]] = {}
    pass_count = 0
    fail_count = 0

    for f in md_files:
        errs = validate_file(f)
        results[f.name] = errs
        if errs:
            fail_count += 1
        else:
            pass_count += 1

    return pass_count, fail_count, results


def main() -> int:
    default_dir = Path(__file__).resolve().parent.parent / "frontend" / "posts"

    parser = argparse.ArgumentParser(
        description="Verify KaTeX, Mermaid, and heading syntax for blog notebook articles."
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=default_dir,
        help=f"Directory containing post markdown files (default: {default_dir})",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=None,
        help="Validate a single specific markdown file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Display verbose diagnostics.",
    )

    args = parser.parse_args()

    print("======================================================================")
    print("           BLOG NOTEBOOK -- MARKDOWN SYNTAX VALIDATOR")
    print("======================================================================")

    if args.file:
        target = args.file
        print(f"Checking single file: {target}")
        errs = validate_file(target)
        if not errs:
            print(f" [PASS] {target.name} -- All syntax checks passed successfully.")
            return 0
        else:
            print(f" [FAIL] {target.name} -- {len(errs)} violation(s) detected:")
            for e in errs:
                print(f"   * {e}")
            return 1

    target_dir = args.dir
    print(f"Scanning articles in: {target_dir}")

    try:
        pass_count, fail_count, results = validate_posts_directory(target_dir)
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    total = pass_count + fail_count
    print(f"Found {total} markdown files.\n")

    for file_name, errs in results.items():
        if not errs:
            print(f" [PASS] {file_name}")
        else:
            print(f" [FAIL] {file_name} ({len(errs)} violation{'s' if len(errs) > 1 else ''}):")
            for e in errs:
                print(f"   * {e}")
        if args.verbose and not errs:
            print("   -> KaTeX math, Mermaid diagrams, and H2 plain text headings verified clean.")

    print("\n----------------------------------------------------------------------")
    print(f"Summary: Total: {total} | Passed: {pass_count} | Failed: {fail_count}")
    print("----------------------------------------------------------------------")

    if fail_count > 0:
        print(f"RESULT: FAILED ({fail_count} file(s) failed syntax validation).")
        return 1

    print("RESULT: PASSED (100% of articles have valid KaTeX, Mermaid, and heading syntax).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
