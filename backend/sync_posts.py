"""
Script to synchronize Markdown files from the local posts directory into Supabase Database.
Workflow (Git-as-CMS):
1. Read all .md files in the specified directory.
2. Parse Frontmatter (title, tags, category, date, etc.) and Markdown content.
3. Automatically UPSERT (Insert if new, Update if exists) into the Supabase 'posts' table.
"""

import os
import re
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load env from backend/.env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL", "").split("/rest/v1")[0].rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

POSTS_DIR = Path(__file__).resolve().parent.parent / "frontend" / "posts"


def parse_markdown_file(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    slug = file_path.stem  # filename without .md

    # Check for YAML frontmatter
    frontmatter = {}
    content = text

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if match:
        raw_yaml, content = match.groups()
        try:
            frontmatter = yaml.safe_load(raw_yaml) or {}
        except Exception as e:
            print(f"⚠️ Error parsing frontmatter in {file_path.name}: {e}")

    # Fallback title if missing
    title = frontmatter.get("title")
    if not title:
        # Try extracting first h1
        h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = h1_match.group(1) if h1_match else slug.replace("-", " ").title()

    # Normalize fields
    excerpt = frontmatter.get("description") or frontmatter.get("excerpt") or ""
    date_val = frontmatter.get("date")
    if isinstance(date_val, datetime):
        date_str = date_val.isoformat()
    elif date_val:
        date_str = str(date_val)
    else:
        date_str = datetime.now().isoformat()

    tags = frontmatter.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    return {
        "slug": slug,
        "title": title,
        "excerpt": excerpt,
        "content": content.strip(),
        "date": date_str,
        "author": frontmatter.get("author", "Phi Trieu Dai"),
        "tags": tags,
        "category": frontmatter.get("category", "General"),
        "series": frontmatter.get("series"),
        "series_index": frontmatter.get("series_index"),
        "pinned": bool(frontmatter.get("pinned", frontmatter.get("featured", False))),
        "published": bool(frontmatter.get("published", True)),
        "updated_at": datetime.now().isoformat(),
    }


def sync_all_posts():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set in backend/.env")
        return

    print(f"🔗 Connecting to Supabase: {SUPABASE_URL}")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    if not POSTS_DIR.exists():
        print(f"❌ Error: Posts directory does not exist: {POSTS_DIR}")
        return

    md_files = list(POSTS_DIR.glob("*.md"))
    print(f"📂 Found {len(md_files)} markdown files in '{POSTS_DIR.name}'\n")

    synced_count = 0
    for file_path in md_files:
        post_data = parse_markdown_file(file_path)
        try:
            # Upsert by unique column 'slug'
            res = supabase.table("posts").upsert(post_data, on_conflict="slug").execute()
            print(f"✅ Synced: [{post_data['slug']}] -> '{post_data['title']}'")
            synced_count += 1
        except Exception as e:
            print(f"❌ Failed to sync [{file_path.name}]: {e}")

    print(f"\n🎉 Successfully synced {synced_count}/{len(md_files)} posts to Supabase Cloud!")


if __name__ == "__main__":
    sync_all_posts()

