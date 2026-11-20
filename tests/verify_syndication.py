#!/usr/bin/env python3
r"""
verify_syndication.py — Standalone Validator for RSS Feed, Sitemap, and Robots.txt

Validates the generation, syntax, schema, and link integrity of:
  - public/dist/feed.xml and rss.xml (RSS 2.0 with RFC-822 dates)
  - public/dist/sitemap.xml (Sitemap 0.9 with static routes and article locs)
  - public/dist/robots.txt (Robot exclusion with sitemap directive)

Exit code:
  0: All syndication artifacts are valid and complete.
  1: One or more validation checks failed.
"""

from __future__ import annotations

import argparse
import email.utils
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

# Reconfigure stdout to UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def validate_rss(file_path: Path, min_items: int = 8) -> list[str]:
    errors = []
    if not file_path.exists():
        return [f"RSS file does not exist: {file_path}"]

    try:
        tree = ET.parse(str(file_path))
        root = tree.getroot()
    except Exception as e:
        return [f"RSS XML parsing error in {file_path.name}: {e}"]

    if root.tag != "rss":
        errors.append(f"Root tag is <{root.tag}>, expected <rss>")
    if root.get("version") != "2.0":
        errors.append(f"RSS version is '{root.get('version')}', expected '2.0'")

    channel = root.find("channel")
    if channel is None:
        errors.append("Missing <channel> element in RSS feed")
        return errors

    title = channel.findtext("title", "").strip()
    link = channel.findtext("link", "").strip()
    description = channel.findtext("description", "").strip()

    if not title:
        errors.append("RSS channel <title> is missing or empty")
    if not link or not (link.startswith("http://") or link.startswith("https://")):
        errors.append(f"RSS channel <link> must be an absolute URL, got '{link}'")
    if not description:
        errors.append("RSS channel <description> is missing or empty")

    items = channel.findall("item")
    if len(items) < min_items:
        errors.append(f"Expected at least {min_items} items in RSS feed, found {len(items)}")

    for i, item in enumerate(items, start=1):
        i_title = item.findtext("title", "").strip()
        i_link = item.findtext("link", "").strip()
        i_guid = item.findtext("guid", "").strip()
        i_pubdate = item.findtext("pubDate", "").strip()
        i_desc = item.findtext("description", "").strip()
        i_cat = item.findtext("category", "").strip()

        prefix = f"Item {i} ('{i_title or 'Untitled'}')"
        if not i_title:
            errors.append(f"{prefix}: missing <title>")
        if not i_link or not (i_link.startswith("http://") or i_link.startswith("https://")):
            errors.append(f"{prefix}: <link> must be absolute URL, got '{i_link}'")
        if not i_guid:
            errors.append(f"{prefix}: missing <guid>")
        if not i_desc:
            errors.append(f"{prefix}: missing <description>")
        if not i_cat:
            errors.append(f"{prefix}: missing <category>")

        if not i_pubdate:
            errors.append(f"{prefix}: missing <pubDate>")
        else:
            try:
                parsed_date = email.utils.parsedate_to_datetime(i_pubdate)
                if not parsed_date:
                    errors.append(f"{prefix}: invalid RFC-822 pubDate '{i_pubdate}'")
            except Exception as e:
                errors.append(f"{prefix}: failed parsing RFC-822 date '{i_pubdate}': {e}")

    return errors


def validate_sitemap(file_path: Path, min_urls: int = 14) -> list[str]:
    errors = []
    if not file_path.exists():
        return [f"Sitemap file does not exist: {file_path}"]

    try:
        tree = ET.parse(str(file_path))
        root = tree.getroot()
    except Exception as e:
        return [f"Sitemap XML parsing error in {file_path.name}: {e}"]

    ns = "http://www.sitemaps.org/schemas/sitemap/0.9"
    expected_root = f"{{{ns}}}urlset"
    if root.tag != expected_root and root.tag != "urlset":
        errors.append(f"Root tag is <{root.tag}>, expected <{expected_root}>")

    urls = root.findall(f"{{{ns}}}url") if root.tag.startswith("{") else root.findall("url")
    if len(urls) < min_urls:
        errors.append(f"Expected at least {min_urls} URLs in sitemap, found {len(urls)}")

    locs: list[str] = []
    date_regex = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    for i, u in enumerate(urls, start=1):
        loc = (u.findtext(f"{{{ns}}}loc") or u.findtext("loc") or "").strip()
        lastmod = (u.findtext(f"{{{ns}}}lastmod") or u.findtext("lastmod") or "").strip()
        changefreq = (u.findtext(f"{{{ns}}}changefreq") or u.findtext("changefreq") or "").strip()
        priority = (u.findtext(f"{{{ns}}}priority") or u.findtext("priority") or "").strip()

        prefix = f"URL {i} ('{loc}')"
        if not loc or not (loc.startswith("http://") or loc.startswith("https://")):
            errors.append(f"{prefix}: <loc> must be an absolute URL, got '{loc}'")
        else:
            locs.append(loc)

        if lastmod and not date_regex.match(lastmod):
            errors.append(f"{prefix}: <lastmod> must be in YYYY-MM-DD format, got '{lastmod}'")

        if changefreq and changefreq not in {"always", "hourly", "daily", "weekly", "monthly", "yearly", "never"}:
            errors.append(f"{prefix}: invalid <changefreq> '{changefreq}'")

        if priority:
            try:
                p_val = float(priority)
                if not (0.0 <= p_val <= 1.0):
                    errors.append(f"{prefix}: <priority> must be between 0.0 and 1.0, got '{priority}'")
            except ValueError:
                errors.append(f"{prefix}: invalid <priority> float '{priority}'")

    # Verify static routes exist
    expected_static = ["/blog", "/topics", "/graph", "/projects", "/about"]
    for path_part in expected_static:
        if not any(loc.endswith(path_part) for loc in locs):
            errors.append(f"Sitemap missing required static route: {path_part}")

    # Verify article routes exist
    article_locs = [loc for loc in locs if "/article/" in loc]
    if len(article_locs) < 8:
        errors.append(f"Expected at least 8 article URLs in sitemap, found {len(article_locs)}")

    return errors


def validate_robots_txt(file_path: Path) -> list[str]:
    errors = []
    if not file_path.exists():
        return [f"robots.txt does not exist: {file_path}"]

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"Failed to read {file_path}: {e}"]

    if "User-agent:" not in content:
        errors.append("robots.txt missing 'User-agent:' directive")
    if "Allow:" not in content:
        errors.append("robots.txt missing 'Allow:' directive")
    if "Sitemap:" not in content or "sitemap.xml" not in content:
        errors.append("robots.txt missing 'Sitemap:' directive pointing to sitemap.xml")

    # Verify sitemap URL format
    sitemap_match = re.search(r"Sitemap:\s*(https?://[^\s]+sitemap\.xml)", content, re.IGNORECASE)
    if not sitemap_match:
        errors.append("robots.txt 'Sitemap:' directive does not contain a valid absolute URL ending with sitemap.xml")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validator for RSS feed, Sitemap, and robots.txt.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory containing syndication files (default: checks dist/ then public/)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    frontend_dir = repo_root / "blog-notebook" / "frontend"

    if args.dir:
        target_dir = args.dir
    elif (frontend_dir / "dist" / "sitemap.xml").exists():
        target_dir = frontend_dir / "dist"
    else:
        target_dir = frontend_dir / "public"

    print(f"Validating syndication files in: {target_dir}")

    all_errors: list[str] = []

    # 1. RSS Feed validation (both feed.xml and rss.xml if present)
    feed_xml = target_dir / "feed.xml"
    rss_xml = target_dir / "rss.xml"
    if feed_xml.exists():
        feed_errors = validate_rss(feed_xml)
        if feed_errors:
            all_errors.extend([f"[feed.xml] {e}" for e in feed_errors])
        else:
            print(f"  [PASS] {feed_xml.name} is valid RSS 2.0 with all articles")
    elif not rss_xml.exists():
        all_errors.append(f"Neither feed.xml nor rss.xml found in {target_dir}")

    if rss_xml.exists():
        rss_errors = validate_rss(rss_xml)
        if rss_errors:
            all_errors.extend([f"[rss.xml] {e}" for e in rss_errors])
        else:
            print(f"  [PASS] {rss_xml.name} is valid RSS 2.0 with all articles")

    # 2. Sitemap validation
    sitemap_xml = target_dir / "sitemap.xml"
    sitemap_errors = validate_sitemap(sitemap_xml)
    if sitemap_errors:
        all_errors.extend([f"[sitemap.xml] {e}" for e in sitemap_errors])
    else:
        print(f"  [PASS] {sitemap_xml.name} is valid Sitemap 0.9 with all routes")

    # 3. Robots.txt validation
    robots_txt = target_dir / "robots.txt"
    robots_errors = validate_robots_txt(robots_txt)
    if robots_errors:
        all_errors.extend([f"[robots.txt] {e}" for e in robots_errors])
    else:
        print(f"  [PASS] {robots_txt.name} is valid with sitemap directive")

    if all_errors:
        print(f"\n[FAIL] Found {len(all_errors)} syndication error(s):")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print("\n[PASS] All syndication files validated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
