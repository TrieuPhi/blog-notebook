#!/usr/bin/env python3
r"""
test_syndication.py — Tier 3 Standard Unit Tests for RSS, Sitemap, and Robots.txt

Validates:
1. RSS 2.0 feed structure, XML validity, RFC-822 dates, and article completeness.
2. Sitemap 0.9 schema, static routes, and article URL formatting.
3. Robots.txt directives and sitemap linkage.
4. Base URL resolution and slug URI encoding.
"""

from __future__ import annotations

import email.utils
import re
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


class TestSyndicationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent.parent.parent
        cls.frontend_dir = cls.repo_root / "blog-notebook" / "frontend"
        cls.public_dir = cls.frontend_dir / "public"
        cls.dist_dir = cls.frontend_dir / "dist"
        cls.posts_dir = cls.frontend_dir / "posts"

        # Determine target directory (prefer dist if available, else public)
        cls.target_dir = cls.dist_dir if (cls.dist_dir / "sitemap.xml").exists() else cls.public_dir

    def test_rss_files_exist_and_parse(self):
        """Verifies both feed.xml and rss.xml exist and are well-formed XML."""
        for filename in ["feed.xml", "rss.xml"]:
            target = self.target_dir / filename
            self.assertTrue(target.exists(), f"{filename} does not exist in {self.target_dir}")
            tree = ET.parse(str(target))
            root = tree.getroot()
            self.assertEqual(root.tag, "rss")
            self.assertEqual(root.get("version"), "2.0")

    def test_rss_channel_metadata(self):
        """Verifies RSS 2.0 channel header elements."""
        feed_path = self.target_dir / "feed.xml"
        tree = ET.parse(str(feed_path))
        channel = tree.getroot().find("channel")
        self.assertIsNotNone(channel, "Channel element missing")

        title = channel.findtext("title", "")
        self.assertIn("set.log", title)

        link = channel.findtext("link", "")
        self.assertTrue(link.startswith("http://") or link.startswith("https://"))

        desc = channel.findtext("description", "")
        self.assertGreater(len(desc), 10)

        lang = channel.findtext("language", "")
        self.assertEqual(lang, "vi")

        last_build = channel.findtext("lastBuildDate", "")
        self.assertTrue(len(last_build) > 0)
        parsed_build_date = email.utils.parsedate_to_datetime(last_build)
        self.assertIsNotNone(parsed_build_date)

    def test_rss_items_completeness_and_rfc822(self):
        """Verifies all published articles are listed with RFC-822 pubDate and valid fields."""
        feed_path = self.target_dir / "feed.xml"
        tree = ET.parse(str(feed_path))
        items = tree.getroot().find("channel").findall("item")

        # Must have at least 8 items
        self.assertGreaterEqual(len(items), 8, f"Expected >= 8 items, found {len(items)}")

        dates = []
        for item in items:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            guid = item.findtext("guid", "")
            pub_date = item.findtext("pubDate", "")
            description = item.findtext("description", "")
            category = item.findtext("category", "")

            self.assertTrue(len(title) > 0, "Item missing title")
            self.assertTrue(link.startswith("https://") or link.startswith("http://"), f"Invalid link: {link}")
            self.assertTrue(guid.startswith("https://") or guid.startswith("http://"), f"Invalid guid: {guid}")
            self.assertTrue(len(description) > 0, f"Item '{title}' missing description")
            self.assertTrue(len(category) > 0, f"Item '{title}' missing category")

            # RFC-822 validation
            parsed_dt = email.utils.parsedate_to_datetime(pub_date)
            self.assertIsNotNone(parsed_dt, f"Failed parsing RFC-822 date for '{title}': {pub_date}")
            dates.append(parsed_dt)

            # Ensure URL spaces are encoded
            self.assertNotIn(" ", link, f"Link contains unencoded spaces: {link}")

        # Check sorting: descending by date
        for i in range(len(dates) - 1):
            self.assertGreaterEqual(dates[i], dates[i + 1], "Items must be sorted descending by publication date")

    def test_sitemap_schema_and_routes(self):
        """Verifies Sitemap 0.9 urlset, static routes, and article URLs."""
        sitemap_path = self.target_dir / "sitemap.xml"
        self.assertTrue(sitemap_path.exists(), "sitemap.xml does not exist")

        tree = ET.parse(str(sitemap_path))
        root = tree.getroot()

        ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = root.findall("ns:url", ns)
        if not urls:
            urls = root.findall("url")

        self.assertGreaterEqual(len(urls), 14, f"Expected >= 14 URLs (6 static + 8 articles), got {len(urls)}")

        locs = []
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

        for u in urls:
            loc = (u.findtext("{http://www.sitemaps.org/schemas/sitemap/0.9}loc") or u.findtext("loc") or "").strip()
            self.assertTrue(loc.startswith("http://") or loc.startswith("https://"), f"Invalid loc: {loc}")
            self.assertNotIn(" ", loc, f"Loc contains unencoded spaces: {loc}")
            locs.append(loc)

            lastmod = (u.findtext("{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod") or u.findtext("lastmod") or "").strip()
            if lastmod:
                self.assertTrue(date_pattern.match(lastmod), f"Invalid lastmod date format: {lastmod}")

        # Verify 6 static routes
        static_suffixes = ["/", "/blog", "/topics", "/graph", "/projects", "/about"]
        for suffix in static_suffixes:
            self.assertTrue(
                any(loc.endswith(suffix) for loc in locs),
                f"Missing static route ending with {suffix} in sitemap",
            )

        # Verify 8 article routes
        article_locs = [loc for loc in locs if "/article/" in loc]
        self.assertGreaterEqual(len(article_locs), 8, f"Expected >= 8 articles, found {len(article_locs)}")

    def test_robots_txt_directives(self):
        """Verifies robots.txt exists and contains standard directives."""
        robots_path = self.target_dir / "robots.txt"
        self.assertTrue(robots_path.exists(), "robots.txt does not exist")

        content = robots_path.read_text(encoding="utf-8")
        self.assertIn("User-agent: *", content)
        self.assertIn("Allow: /", content)
        self.assertIn("Sitemap:", content)
        self.assertIn("sitemap.xml", content)


if __name__ == "__main__":
    unittest.main()
