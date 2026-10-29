#!/usr/bin/env python3
"""
adversarial_mermaid_deep_scan.py — Deep AST-style Lexical Scanner for Mermaid Blocks
"""

import re
import sys
from pathlib import Path

# Reconfigure stdout/stderr to UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

POSTS_DIR = Path(r"c:\project\blog\blog-notebook\frontend\posts")

VIETNAMESE_RE = re.compile(
    r"[àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
    r"ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]"
)

SUPPORTED_TYPES = (
    "flowchart", "graph", "sequenceDiagram", "classDiagram", "stateDiagram",
    "stateDiagram-v2", "erDiagram", "gantt", "pie", "gitGraph", "mindmap",
    "quadrantChart", "timeline", "journey", "c4Context", "requirementDiagram",
    "packet-beta", "architecture-beta", "zenuml", "block-beta"
)

# Shapes with double-quoted labels (VALID)
QUOTED_SHAPES = [
    r'\b[A-Za-z0-9_-]+\s*\(\[\"[^"\r\n]*\"\]\)',  # Stadium: ID(["..."])
    r'\b[A-Za-z0-9_-]+\s*\[\[\"[^"\r\n]*\"\]\]',  # Subroutine: ID[["..."]]
    r'\b[A-Za-z0-9_-]+\s*\[\(\"[^"\r\n]*\"\)\]',  # Database: ID[("...")]
    r'\b[A-Za-z0-9_-]+\s*\(\(\"[^"\r\n]*\"\)\)',  # Circle: ID(("..."))
    r'\b[A-Za-z0-9_-]+\s*\{\{\"[^"\r\n]*\"\}\}',  # Hexagon: ID{{"..."}}
    r'\b[A-Za-z0-9_-]+\s*\[\"[^"\r\n]*\"\]',      # Rectangle: ID["..."]
    r'\b[A-Za-z0-9_-]+\s*\(\"[^"\r\n]*\"\)',      # Rounded: ID("...")
    r'\b[A-Za-z0-9_-]+\s*\{\"[^"\r\n]*\"\}',      # Decision: ID{"..."}
    r'\bsubgraph\s+[A-Za-z0-9_-]+\s*\[\"[^"\r\n]*\"\]', # Subgraph with quoted label
]

# Unquoted shape patterns to inspect for violations
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

def scan_file(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    fm_match = re.match(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$", text)
    body = fm_match.group(2) if fm_match else text

    blocks = re.findall(r"```mermaid\r?\n([\s\S]*?)```", body)
    print(f"\n--- {file_path.name} ({len(blocks)} Mermaid block{'s' if len(blocks) != 1 else ''}) ---")

    file_violations = 0
    total_nodes = 0
    total_quoted_nodes = 0

    for idx, block in enumerate(blocks, 1):
        lines = [l.strip() for l in block.splitlines() if l.strip() and not l.strip().startswith("%%")]
        if not lines:
            print(f"  [ERROR] Block #{idx} is empty!")
            file_violations += 1
            continue

        diag_type = lines[0]
        valid_type = any(diag_type.startswith(t) for t in SUPPORTED_TYPES)
        if not valid_type:
            print(f"  [ERROR] Block #{idx}: Unknown diagram type: '{diag_type}'")
            file_violations += 1
        else:
            print(f"  Block #{idx}: Type '{diag_type}' | Lines: {len(lines)}")

        if diag_type.startswith(("flowchart", "graph")):
            for line_no, line in enumerate(lines, 1):
                if line.startswith(("%%", "classDef", "linkStyle", "click")) or line == "end":
                    continue

                # Strip edge labels
                working_line = re.sub(r'--\s*"[^"]*"\s*-->', " --> ", line)
                working_line = re.sub(r'-->\|"[^"]*"\|', " --> ", working_line)

                # Count and mask properly quoted nodes
                for qpat in QUOTED_SHAPES:
                    for m in re.finditer(qpat, working_line):
                        total_nodes += 1
                        total_quoted_nodes += 1
                    working_line = re.sub(qpat, " ", working_line)

                # Inspect remaining unquoted definitions
                for upat, shape_name in UNQUOTED_SHAPES:
                    for m in re.finditer(upat, working_line):
                        total_nodes += 1
                        nid = m.group(1)
                        label = m.group(2).strip()
                        has_vi = bool(VIETNAMESE_RE.search(label))
                        has_special = any(c in label for c in "():[]{} ,;/\\$#&`'")
                        if has_vi or has_special:
                            print(f"    [VIOLATION] Line {line_no}: Unquoted {shape_name} '{nid}' label has special chars: {label!r}")
                            file_violations += 1
                    working_line = re.sub(upat, " ", working_line)

    print(f"  Summary: {len(blocks)} blocks scanned, {total_nodes} nodes detected ({total_quoted_nodes} double-quoted). Violations: {file_violations}")
    return file_violations, len(blocks), total_nodes

def main():
    print("======================================================================")
    print("       ADVERSARIAL DEEP SCAN: MERMAID DIAGRAM RESILIENCE")
    print("======================================================================")

    files = sorted(POSTS_DIR.glob("*.md"))
    total_viol = 0
    total_blocks = 0
    total_nodes = 0

    for f in files:
        v, b, n = scan_file(f)
        total_viol += v
        total_blocks += b
        total_nodes += n

    print("\n======================================================================")
    print(f"OVERALL SCAN RESULT: {len(files)} files | {total_blocks} Mermaid blocks | {total_nodes} nodes")
    print(f"Total Violations: {total_viol}")
    print("======================================================================")

    if total_viol > 0:
        print("VERDICT: REQUEST_CHANGES")
        sys.exit(1)
    else:
        print("VERDICT: APPROVE (Mermaid diagrams 100% compliant and lexical safe)")
        sys.exit(0)

if __name__ == "__main__":
    main()
