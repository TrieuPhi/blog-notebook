#!/usr/bin/env python3
r"""
test_validators.py — Unit Tests for Frontmatter & Markdown Syntax Validators

Tests positive paths, edge cases, adversarial inputs, and failure cascades:
- Frontmatter schema validation (BOM, delimiter, types, arrays, ints).
- KaTeX syntax validation (spacing, unbalanced braces, unescaped currency).
- Mermaid diagram validation (node quote requirements for Vietnamese & punct).
- Headings validation (plain text enforcement for TOC anchor stability).
"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# Add parent tests directory to sys.path
TESTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TESTS_DIR))

from verify_frontmatter import (
    UTF8_BOM,
)
from verify_frontmatter import (
    validate_file as validate_frontmatter_file,
)
from verify_markdown_syntax import (
    validate_headings,
    validate_katex_syntax,
    validate_mermaid_syntax,
)


class TestFrontmatterValidator(unittest.TestCase):
    """Unit tests for verify_frontmatter.py."""

    def test_valid_frontmatter_passes(self):
        content = (
            "---\n"
            "title: \"Bản chất Deep Learning\"\n"
            "description: \"Tóm tắt nội dung bài viết kỹ thuật.\"\n"
            "date: \"2026-10-28\"\n"
            "updated: \"2026-10-28\"\n"
            "category: \"AI\"\n"
            "tags: [\"Deep Learning\", \"Neural Networks\"]\n"
            "status: \"published\"\n"
            "featured: true\n"
            "readingTime: 12\n"
            "---\n\n"
            "## 1. Lý thuyết căn bản\n"
        )
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.md"
            file_path.write_bytes(content.encode("utf-8"))
            errors = validate_frontmatter_file(file_path)
            self.assertEqual(len(errors), 0, [str(e) for e in errors])

    def test_utf8_bom_is_rejected(self):
        content = UTF8_BOM + (
            b"---\n"
            b"title: \"Test Title\"\n"
            b"description: \"Test Desc\"\n"
            b"date: \"2026-10-28\"\n"
            b"category: \"AI\"\n"
            b"tags: [\"AI\"]\n"
            b"status: \"published\"\n"
            b"readingTime: 5\n"
            b"---\n"
        )
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "bom.md"
            file_path.write_bytes(content)
            errors = validate_frontmatter_file(file_path)
            self.assertTrue(any("UTF-8 BOM detected" in str(e) for e in errors))

    def test_missing_byte_zero_delimiter_fails(self):
        content = (
            "\n---\n"  # leading newline
            "title: \"Test Title\"\n"
            "description: \"Test Desc\"\n"
            "date: \"2026-10-28\"\n"
            "category: \"AI\"\n"
            "tags: [\"AI\"]\n"
            "status: \"published\"\n"
            "readingTime: 5\n"
            "---\n"
        )
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "leading_nl.md"
            file_path.write_text(content, encoding="utf-8")
            errors = validate_frontmatter_file(file_path)
            self.assertTrue(any("does not start with '---' at byte 0" in str(e) for e in errors))

    def test_string_tags_field_rejected(self):
        """CRITICAL: tags as single string causes React runtime crash."""
        content = (
            "---\n"
            "title: \"Test Title\"\n"
            "description: \"Test Desc\"\n"
            "date: \"2026-10-28\"\n"
            "category: \"AI\"\n"
            "tags: \"Single String Tag\"\n"
            "status: \"published\"\n"
            "readingTime: 5\n"
            "---\n"
        )
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "str_tags.md"
            file_path.write_text(content, encoding="utf-8")
            errors = validate_frontmatter_file(file_path)
            self.assertTrue(any("must be a YAML array" in str(e) for e in errors))

    def test_string_reading_time_rejected(self):
        """readingTime must be int, not string like '8 min read'."""
        content = (
            "---\n"
            "title: \"Test Title\"\n"
            "description: \"Test Desc\"\n"
            "date: \"2026-10-28\"\n"
            "category: \"AI\"\n"
            "tags: [\"AI\"]\n"
            "status: \"published\"\n"
            "readingTime: \"10 min read\"\n"
            "---\n"
        )
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "str_readtime.md"
            file_path.write_text(content, encoding="utf-8")
            errors = validate_frontmatter_file(file_path)
            self.assertTrue(any("must be an integer number" in str(e) for e in errors))

    def test_invalid_status_rejected(self):
        content = (
            "---\n"
            "title: \"Test Title\"\n"
            "description: \"Test Desc\"\n"
            "date: \"2026-10-28\"\n"
            "category: \"AI\"\n"
            "tags: [\"AI\"]\n"
            "status: \"note\"\n"
            "readingTime: 5\n"
            "---\n"
        )
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "bad_status.md"
            file_path.write_text(content, encoding="utf-8")
            errors = validate_frontmatter_file(file_path)
            self.assertTrue(any("must be exactly 'published'" in str(e) for e in errors))

    def test_hyphenated_key_name_rejected(self):
        """Frontend parser regex ^([A-Za-z]+): drops keys with hyphens or underscores."""
        content = (
            "---\n"
            "title: \"Test Title\"\n"
            "description: \"Test Desc\"\n"
            "reading-time: 10\n"
            "date: \"2026-10-28\"\n"
            "category: \"AI\"\n"
            "tags: [\"AI\"]\n"
            "status: \"published\"\n"
            "---\n"
        )
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "hyphen_key.md"
            file_path.write_text(content, encoding="utf-8")
            errors = validate_frontmatter_file(file_path)
            self.assertTrue(any("Invalid frontmatter key 'reading-time'" in str(e) for e in errors))


class TestMarkdownSyntaxValidator(unittest.TestCase):
    """Unit tests for verify_markdown_syntax.py."""

    def test_valid_katex_passes(self):
        lines = [
            (10, "Ta có công thức $E = mc^2$ và $W^T x + b = 0$."),
            (11, "$$"),
            (12, "\\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^N y_i \\log(\\hat{y}_i)"),
            (13, "$$"),
        ]
        violations = validate_katex_syntax(lines, "test.md")
        self.assertEqual(len(violations), 0, [str(v) for v in violations])

    def test_katex_space_after_dollar_fails(self):
        lines = [(15, "Biểu thức $ x + y = z$ có dấu cách sau $ mở.")]
        violations = validate_katex_syntax(lines, "test.md")
        self.assertTrue(any("space immediately follows opening '$'" in str(v) for v in violations))

    def test_katex_space_before_closing_dollar_fails(self):
        lines = [(16, "Biểu thức $x + y = z $ có dấu cách trước $ đóng.")]
        violations = validate_katex_syntax(lines, "test.md")
        self.assertTrue(any("space immediately precedes closing '$'" in str(v) for v in violations))

    def test_katex_unbalanced_curly_braces_fails(self):
        lines = [
            (20, "$$"),
            (21, "\\frac{a}{b"),  # missing }
            (22, "$$"),
        ]
        violations = validate_katex_syntax(lines, "test.md")
        self.assertTrue(any("Unbalanced curly braces in display math" in str(v) for v in violations))

    def test_katex_unescaped_currency_fails(self):
        lines = [(25, "Chi phí server là $500 mỗi tháng.")]
        violations = validate_katex_syntax(lines, "test.md")
        self.assertTrue(any("Unescaped currency dollar sign detected" in str(v) for v in violations))

    def test_valid_mermaid_with_quotes_passes(self):
        code = (
            "flowchart TD\n"
            "    Start([\"Bắt đầu quy trình\"]) --> Process[\"Xử lý dữ liệu (ETL)\"]\n"
            "    Process --> Decision{\"Đạt điều kiện?\"}\n"
            "    Decision -- \"Có\" --> End([\"Kết thúc\"])\n"
        )
        fenced = [(1, 5, "mermaid", code)]
        violations = validate_mermaid_syntax(fenced, "test.md")
        self.assertEqual(len(violations), 0, [str(v) for v in violations])

    def test_mermaid_unquoted_vietnamese_fails(self):
        code = (
            "flowchart TD\n"
            "    Start[Bắt đầu quy trình (Chưa quote)] --> End\n"
        )
        fenced = [(1, 3, "mermaid", code)]
        violations = validate_mermaid_syntax(fenced, "test.md")
        self.assertTrue(any("must be double-quoted" in str(v) for v in violations))

    def test_valid_plain_headings_pass(self):
        lines = [
            (5, "## 1. Lý thuyết Cơ bản về Mạng Nơ-ron"),
            (10, "## 2. Kiến trúc và Tối ưu hóa"),
        ]
        violations = validate_headings(lines, "test.md")
        self.assertEqual(len(violations), 0, [str(v) for v in violations])

    def test_bold_heading_fails(self):
        lines = [(5, "## 1. Lý thuyết về **Deep Learning**")]
        violations = validate_headings(lines, "test.md")
        self.assertTrue(any("H2 heading contains bold formatting" in str(v) for v in violations))

    def test_math_in_heading_fails(self):
        lines = [(12, "## 2. Ý nghĩa Công thức $E=mc^2$")]
        violations = validate_headings(lines, "test.md")
        self.assertTrue(any("H2 heading contains math delimiter" in str(v) for v in violations))

    def test_code_in_heading_fails(self):
        lines = [(15, "## 3. Cấu hình `sys.modules` trong Python")]
        violations = validate_headings(lines, "test.md")
        self.assertTrue(any("H2 heading contains inline code" in str(v) for v in violations))


if __name__ == "__main__":
    unittest.main()
