#!/usr/bin/env python3
"""
Import and scaffold a lesson from ai-engineering-from-scratch into blog-notebook.
Preserves diagrams, code, equations, and sets up frontmatter for Vietnamese localization.
"""

import argparse
import datetime
import os
import re
import shutil
import sys
from pathlib import Path

DEFAULT_SOURCE = Path("/home/trieuphi-admin/projects/ai-engineering-from-scratch")
REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_POSTS = REPO_ROOT / "frontend" / "posts"
PUBLIC_ASSETS = REPO_ROOT / "frontend" / "public" / "assets"

CATEGORY_MAP = {
    "00": "Developer Tools",
    "01": "Math & AI Foundations",
    "02": "Machine Learning",
    "03": "Deep Learning",
    "04": "Computer Vision",
    "05": "NLP & Language",
    "06": "Speech & Audio",
    "07": "Transformers",
    "08": "Generative AI",
    "09": "Reinforcement Learning",
    "10": "LLM",
    "11": "LLM Engineering",
    "12": "Multimodal AI",
    "13": "Tools & Protocols",
    "14": "Agent Engineering",
    "15": "Autonomous Systems",
    "16": "Multi-Agent & Swarms",
    "17": "MLOps & Infrastructure",
    "18": "AI Safety & Alignment",
    "19": "Capstone Projects",
}

def get_phase_category(phase_slug: str) -> str:
    num = phase_slug.split("-")[0]
    return CATEGORY_MAP.get(num, "AI")

def extract_tags(phase_slug: str, lesson_slug: str, content: str) -> list[str]:
    tags = set()
    num = phase_slug.split("-")[0]
    category = CATEGORY_MAP.get(num)
    if category:
        tags.add(category)

    # Clean slugs for tags
    clean_phase = " ".join(phase_slug.split("-")[1:]).title()
    clean_lesson = " ".join(lesson_slug.split("-")[1:]).title()
    tags.add(clean_phase)
    tags.add(clean_lesson)

    # Check common terms
    for term in ["ReAct", "RAG", "Agent", "Tool Calling", "Memory", "LangGraph", "vLLM", "PyTorch", "MCP"]:
        if re.search(rf"\b{re.escape(term)}\b", content, re.IGNORECASE):
            tags.add(term)

    return sorted(list(tags))

def import_lesson(source_root: Path, phase: str, lesson: str, status: str = "draft", lang: str = "vi", force: bool = False):
    source_dir = source_root / "phases" / phase / lesson
    doc_path = source_dir / "docs" / "en.md"

    if not doc_path.exists():
        print(f"Error: Source documentation not found at {doc_path}")
        sys.exit(1)

    target_phase_dir = FRONTEND_POSTS / "phases" / phase
    target_phase_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_phase_dir / f"{lesson}.md"

    if target_file.exists() and not force:
        print(f"Warning: Target file {target_file} already exists. Use --force to overwrite.")
        return target_file

    raw_doc = doc_path.read_text(encoding="utf-8")

    # Extract Title (# Title)
    title_match = re.search(r"^#\s+(.+)$", raw_doc, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else lesson.replace("-", " ").title()

    # Extract Motto (> Motto)
    motto_match = re.search(r"^>\s+(.+)$", raw_doc, re.MULTILINE)
    motto = motto_match.group(1).strip() if motto_match else ""

    # Estimate reading time (approx 200 words per minute)
    words = len(raw_doc.split())
    reading_time = max(5, round(words / 180))

    # Handle assets if present
    source_assets = source_dir / "assets"
    target_asset_public_dir = PUBLIC_ASSETS / "phases" / phase / lesson
    copied_assets = []
    if source_assets.exists():
        target_asset_public_dir.mkdir(parents=True, exist_ok=True)
        for asset in source_assets.glob("*"):
            if asset.is_file():
                dest = target_asset_public_dir / asset.name
                shutil.copy2(asset, dest)
                copied_assets.append(asset.name)

    # Adjust asset references in markdown
    content_body = raw_doc
    # Replace relative asset path with public path
    content_body = re.sub(
        r'!\[(.*?)\]\(assets/([^\)]+)\)',
        rf'![\1](/assets/phases/{phase}/{lesson}/\2)',
        content_body
    )
    content_body = re.sub(
        r'!\[(.*?)\]\(\./assets/([^\)]+)\)',
        rf'![\1](/assets/phases/{phase}/{lesson}/\2)',
        content_body
    )

    # If figure fence is used (```figure ... ```), leave note or convert
    tags = extract_tags(phase, lesson, raw_doc)
    category = get_phase_category(phase)
    today = datetime.date.today().isoformat()
    phase_num = phase.split("-")[0]

    frontmatter = f"""---
title: "{title}"
description: "{motto[:160] if motto else title}"
date: "{today}"
category: "{category}"
tags: {tags}
status: "{status}"
readingTime: {reading_time}
lang: "{lang}"
phase: "{phase}"
phaseNumber: "{phase_num}"
lesson: "{lesson}"
translationKey: "{phase_num}-{lesson}"
slug: "{lesson}"
---

"""

    final_content = frontmatter + content_body
    target_file.write_text(final_content, encoding="utf-8")
    print(f" Successfully imported lesson:")
    print(f"   Source : {doc_path}")
    print(f"   Target : {target_file}")
    if copied_assets:
        print(f"   Assets : Copied {len(copied_assets)} assets to {target_asset_public_dir}")
    print(f"   Status : {status} | Lang: {lang} | Phase: {phase}")

    return target_file

def list_phases(source_root: Path):
    phases_dir = source_root / "phases"
    if not phases_dir.exists():
        print(f"Source phases dir not found at {phases_dir}")
        return
    phases = sorted([p.name for p in phases_dir.iterdir() if p.is_dir()])
    print("Available Phases:")
    for p in phases:
        cat = get_phase_category(p)
        print(f"  - {p} ({cat})")

def list_lessons(source_root: Path, phase: str):
    phase_dir = source_root / "phases" / phase
    if not phase_dir.exists():
        print(f"Phase {phase} not found at {phase_dir}")
        return
    lessons = sorted([l.name for l in phase_dir.iterdir() if l.is_dir()])
    print(f"Lessons in {phase}:")
    for l in lessons:
        has_doc = (phase_dir / l / "docs" / "en.md").exists()
        mark = "✓" if has_doc else "✗ (no docs/en.md)"
        print(f"  - {l} {mark}")

def main():
    parser = argparse.ArgumentParser(description="Import and scaffold lesson into blog-notebook")
    parser.add_argument("--phase", help="Phase directory name, e.g. 14-agent-engineering")
    parser.add_argument("--lesson", help="Lesson directory name, e.g. 01-the-agent-loop")
    parser.add_argument("--status", default="draft", choices=["draft", "published"], help="Post status")
    parser.add_argument("--lang", default="vi", help="Post language (default: vi)")
    parser.add_argument("--all", action="store_true", help="Import all lessons in the given --phase")
    parser.add_argument("--force", action="store_true", help="Overwrite existing target file")
    parser.add_argument("--list-phases", action="store_true", help="List all available phases")
    parser.add_argument("--list-lessons", action="store_true", help="List lessons for given --phase")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Path to ai-engineering-from-scratch")

    args = parser.parse_args()

    if args.list_phases:
        list_phases(args.source)
        return

    if args.list_lessons:
        if not args.phase:
            print("Error: --phase is required to list lessons.")
            sys.exit(1)
        list_lessons(args.source, args.phase)
        return

    if args.all:
        if not args.phase:
            print("Error: --phase is required when using --all.")
            sys.exit(1)
        phase_dir = args.source / "phases" / args.phase
        if not phase_dir.exists():
            print(f"Error: Phase directory not found: {phase_dir}")
            sys.exit(1)
        lessons = sorted([l.name for l in phase_dir.iterdir() if l.is_dir() and (l / "docs" / "en.md").exists()])
        print(f"Found {len(lessons)} lessons in {args.phase}. Importing...")
        for l in lessons:
            import_lesson(args.source, args.phase, l, status=args.status, lang=args.lang, force=args.force)
        print(f"\nDone! Successfully processed {len(lessons)} lessons in {args.phase}.")
        return

    if not args.phase or not args.lesson:
        parser.print_help()
        sys.exit(1)

    import_lesson(args.source, args.phase, args.lesson, status=args.status, lang=args.lang, force=args.force)

if __name__ == "__main__":
    main()
