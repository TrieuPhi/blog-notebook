#!/usr/bin/env python3
"""tests/test_phase02_e2e.py - Standalone E2E Test Suite for blog-notebook Phase 02.

Covers all 21 articles in frontend/posts/phases/02-ml-fundamentals/ across 4 distinct tiers:
  - Tier 1 (TestPhase02Tier1Frontmatter):
      Byte 0 delimiter, BOM rejection, ^[A-Za-z]+$ keys, mandatory core fields
      (title, description, date, category "Machine Learning", tags array, status "published",
      readingTime int), Phase 02 metadata (lang "vi", phase "02-ml-fundamentals", phaseNumber "02",
      lesson, translationKey, slug, motto, clean emoji-free artifacts, keyPoints 3-item array).
  - Tier 2 (TestPhase02Tier2Structure):
      Motto blockquote, Quick Spec, Learning Objectives with checkboxes, all 8 numbered sections
      (The Problem, The Concept with Mermaid & KaTeX, Build It, Use It, Ship It,
      Key Terms with 3-column table, Hands-on Challenges with 3 tiers: Easy/Medium/Hard,
      Further Reading), and plain text H2 headings.
  - Tier 3 (TestPhase02Tier3CodeExecution):
      Section 3 Python blocks extraction, AST syntax validation, anti-blackbox ML import audit
      (only stdlib and numpy allowed; forbids torch, sklearn, scipy, tensorflow, etc.),
      and dynamic execution in isolated subprocess with 10s timeout (skipping gracefully if numpy
      is not installed in local runner).
  - Tier 4 (TestPhase02Tier4Integration):
      KaTeX math syntax (even $$, non-empty, balanced {}, clean inline spacing, no unescaped currency),
      Mermaid diagram syntax (valid diagram type, double-quoted node labels for flowcharts),
      and full production Vite frontend build (npm run build).

Executable via:
  python3 -m unittest tests/test_phase02_e2e.py
"""

from __future__ import annotations

import ast
import datetime
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.verify_frontmatter import (
    DATE_REGEX,
    UTF8_BOM,
    parse_raw_frontmatter_block,
    validate_key_names,
)
from tests.verify_markdown_syntax import (
    extract_code_blocks,
    split_frontmatter_and_content,
    validate_headings,
    validate_katex_syntax,
    validate_mermaid_syntax,
)

PHASE02_DIR = REPO_ROOT / "frontend" / "posts" / "phases" / "02-ml-fundamentals"
FRONTEND_DIR = REPO_ROOT / "frontend"

EXPECTED_ARTICLES: list[str] = [
    "01-what-is-machine-learning.md",
    "02-linear-regression.md",
    "03-logistic-regression.md",
    "04-decision-trees.md",
    "05-support-vector-machines.md",
    "06-knn-and-distances.md",
    "07-unsupervised-learning.md",
    "08-feature-engineering.md",
    "09-model-evaluation.md",
    "10-bias-variance.md",
    "11-ensemble-methods.md",
    "12-hyperparameter-tuning.md",
    "13-ml-pipelines.md",
    "14-naive-bayes.md",
    "15-time-series.md",
    "16-anomaly-detection.md",
    "17-imbalanced-data.md",
    "18-feature-selection.md",
    "19-ml-system-design.md",
    "20-model-monitoring-and-drift.md",
    "21-feature-stores-and-training-serving-skew.md",
]

FORBIDDEN_FRAMEWORKS: set[str] = {
    "torch",
    "pytorch",
    "torchvision",
    "torchaudio",
    "sklearn",
    "scikit_learn",
    "scipy",
    "tensorflow",
    "tf",
    "keras",
    "jax",
    "flax",
    "optax",
    "xgboost",
    "lightgbm",
    "catboost",
    "statsmodels",
    "sympy",
    "cv2",
    "opencv",
    "transformers",
    "huggingface_hub",
    "datasets",
    "accelerate",
    "umap",
    "matplotlib",
    "networkx",
    "langchain",
    "vllm",
    "fastai",
    "pandas",
}

ALLOWED_MODULES: set[str] = set(sys.stdlib_module_names) | set(sys.builtin_module_names) | {"numpy"}

MOTTO_REGEX = re.compile(r"^>\s*\*\*.*?\*\*", re.MULTILINE)
QUICK_SPEC_REGEX = re.compile(r"^##\s+.*?(?:Thông\s+tin\s+nhanh|Quick\s+Spec)", re.IGNORECASE | re.MULTILINE)
OBJECTIVES_REGEX = re.compile(r"^##\s+.*?(?:Mục\s+tiêu\s+bài\s+học|Learning\s+Objectives)", re.IGNORECASE | re.MULTILINE)
PROBLEM_REGEX = re.compile(r"^##\s+1\.\s+.*?(?:Vấn\s+đề\s+thực\s+tế|The\s+Problem)", re.IGNORECASE | re.MULTILINE)
CONCEPT_REGEX = re.compile(r"^##\s+2\.\s+.*?(?:Mô\s+hình\s+tư\s+duy|The\s+Concept)", re.IGNORECASE | re.MULTILINE)
BUILD_IT_REGEX = re.compile(r"^##\s+3\.\s+.*?(?:Tự\s+tay\s+xây\s+dựng|Build\s+It)", re.IGNORECASE | re.MULTILINE)
USE_IT_REGEX = re.compile(r"^##\s+4\.\s+.*?(?:Ứng\s+dụng|Use\s+It)", re.IGNORECASE | re.MULTILINE)
SHIP_IT_REGEX = re.compile(r"^##\s+5\.\s+.*?(?:Đóng\s+gói|Ship\s+It)", re.IGNORECASE | re.MULTILINE)
KEY_TERMS_REGEX = re.compile(r"^##\s+6\.\s+.*?(?:Giải\s+ngố|Key\s+Terms|Thuật\s+ngữ)", re.IGNORECASE | re.MULTILINE)
CHALLENGES_REGEX = re.compile(r"^##\s+7\.\s+.*?(?:Bài\s+tập|Thử\s+thách|Challenges)", re.IGNORECASE | re.MULTILINE)
FURTHER_READING_REGEX = re.compile(r"^##\s+8\.\s+.*?(?:Tài\s+liệu\s+đọc\s+thêm|Further\s+Reading)", re.IGNORECASE | re.MULTILINE)

NEXT_H2_REGEX = re.compile(r"^##\s+[^\n]+$", re.MULTILINE)
PYTHON_FENCE_REGEX = re.compile(r"^```(?:python|py)\s*\r?\n([\s\S]*?)\r?\n```", re.IGNORECASE | re.MULTILINE)
LEADING_EMOJI_REGEX = re.compile(r"^[💻📋📐🛠️💬⚡🔥🚀✅❌]")


def is_numpy_installed() -> bool:
    try:
        res = subprocess.run([sys.executable, "-c", "import numpy"], capture_output=True, timeout=5)
        return res.returncode == 0
    except Exception:
        return False


def load_article_raw(filename: str) -> tuple[bytes, str]:
    file_path = PHASE02_DIR / filename
    raw_bytes = file_path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    return raw_bytes, text


def parse_frontmatter(filename: str) -> tuple[dict[str, Any] | None, str, list[str]]:
    raw_bytes, _ = load_article_raw(filename)
    fm_text, content_text, parse_errors = parse_raw_frontmatter_block(raw_bytes, filename)
    errors = [str(e) for e in parse_errors]
    if not fm_text:
        return None, content_text, errors

    try:
        data = yaml.safe_load(fm_text)
        if not isinstance(data, dict):
            errors.append(f"Frontmatter in {filename} did not parse into a mapping/dict")
            return None, content_text, errors
        return data, content_text, errors
    except yaml.YAMLError as e:
        errors.append(f"YAML parsing error in {filename}: {e}")
        return None, content_text, errors


def get_section_text(content: str, heading_match: re.Match[str] | None) -> str:
    if not heading_match:
        return ""
    start_pos = heading_match.end()
    next_match = NEXT_H2_REGEX.search(content, pos=start_pos)
    end_pos = next_match.start() if next_match else len(content)
    return content[start_pos:end_pos]


def extract_section_3_python_blocks(content: str) -> list[str]:
    m = BUILD_IT_REGEX.search(content)
    if not m:
        return []
    sec3_text = get_section_text(content, m)
    return PYTHON_FENCE_REGEX.findall(sec3_text)


def audit_ast_imports(code_str: str, filename: str) -> tuple[list[str], bool]:
    try:
        tree = ast.parse(code_str, filename=filename)
    except SyntaxError as e:
        return [f"SyntaxError in {filename}:{e.lineno}: {e.msg}"], False

    violations: list[str] = []
    uses_numpy = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top == "numpy":
                    uses_numpy = True
                elif top in FORBIDDEN_FRAMEWORKS:
                    violations.append(
                        f"Forbidden black-box ML framework '{alias.name}' detected in {filename}:{node.lineno}"
                    )
                elif top not in ALLOWED_MODULES:
                    violations.append(
                        f"Disallowed external package '{alias.name}' imported in {filename}:{node.lineno}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top == "numpy":
                    uses_numpy = True
                elif top in FORBIDDEN_FRAMEWORKS:
                    violations.append(
                        f"Forbidden black-box ML framework '{node.module}' detected in {filename}:{node.lineno}"
                    )
                elif top not in ALLOWED_MODULES:
                    violations.append(
                        f"Disallowed external package '{node.module}' imported in {filename}:{node.lineno}"
                    )
            else:
                violations.append(f"Relative import detected in {filename}:{node.lineno}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                violations.append(f"Dynamic import '__import__' detected in {filename}:{node.lineno}")
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                violations.append(f"Dynamic import 'import_module' detected in {filename}:{node.lineno}")

    return violations, uses_numpy


# ==============================================================================
# TIER 1: FRONTMATTER SCHEMA VALIDATION
# ==============================================================================

class TestPhase02Tier1Frontmatter(unittest.TestCase):
    """Tier 1: Feature Coverage & Frontmatter Schema Validation across all 21 articles."""

    def test_01_all_articles_exist(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                file_path = PHASE02_DIR / filename
                self.assertTrue(file_path.is_file(), f"Article missing from disk: {file_path}")

    def test_02_byte0_and_bom_rejection(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                raw_bytes, _ = load_article_raw(filename)
                self.assertFalse(
                    raw_bytes.startswith(UTF8_BOM),
                    f"{filename} contains UTF-8 BOM at byte 0",
                )
                self.assertTrue(
                    raw_bytes.startswith((b"---\n", b"---\r\n")),
                    f"{filename} does not start with '---' delimiter followed by newline at byte 0",
                )
                text = raw_bytes.decode("utf-8", errors="replace")
                self.assertRegex(
                    text,
                    r"^---\r?\n[\s\S]*?\r?\n---\r?\n?",
                    f"{filename} does not contain properly closed '---' frontmatter delimiter",
                )

    def test_03_top_level_key_syntax(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                raw_bytes, _ = load_article_raw(filename)
                fm_text, _, errors = parse_raw_frontmatter_block(raw_bytes, filename)
                if errors:
                    self.fail(f"Cannot extract frontmatter from {filename}: {errors}")
                key_violations = validate_key_names(fm_text, filename)
                self.assertEqual(
                    len(key_violations),
                    0,
                    f"Frontmatter key violations in {filename}:\n" + "\n".join(str(e) for e in key_violations),
                )

    def test_04_mandatory_core_fields(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                data, _, errors = parse_frontmatter(filename)
                if errors or data is None:
                    self.fail(f"Frontmatter parse error in {filename}: {errors}")

                title = data.get("title")
                self.assertIsInstance(title, str)
                self.assertGreater(len(title.strip()), 0)

                description = data.get("description")
                self.assertIsInstance(description, str)
                self.assertGreater(len(description.strip()), 0)

                raw_date = data.get("date")
                if isinstance(raw_date, datetime.date):
                    date_str = raw_date.strftime("%Y-%m-%d")
                elif isinstance(raw_date, str):
                    date_str = raw_date.strip()
                else:
                    self.fail(f"'date' in {filename} must be string or date")
                self.assertTrue(DATE_REGEX.match(date_str))
                datetime.datetime.strptime(date_str, "%Y-%m-%d")

                self.assertEqual(data.get("category"), "Machine Learning")

                tags = data.get("tags")
                self.assertIsInstance(tags, list)
                self.assertGreater(len(tags), 0)
                for tag in tags:
                    self.assertIsInstance(tag, str)
                    self.assertGreater(len(tag.strip()), 0)

                self.assertEqual(data.get("status"), "published")

                rt = data.get("readingTime")
                self.assertTrue(isinstance(rt, int) and not isinstance(rt, bool))
                self.assertGreater(rt, 0)

    def test_05_phase02_metadata_and_clean_artifacts(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                data, _, errors = parse_frontmatter(filename)
                if errors or data is None:
                    self.fail(f"Frontmatter parse error in {filename}: {errors}")

                self.assertEqual(data.get("lang"), "vi")
                self.assertEqual(data.get("phase"), "02-ml-fundamentals")
                self.assertEqual(str(data.get("phaseNumber")), "02")

                for field in ("lesson", "translationKey", "slug", "motto"):
                    val = data.get(field)
                    self.assertIsInstance(val, str, f"'{field}' in {filename} must be string")
                    self.assertGreater(len(val.strip()), 0, f"'{field}' in {filename} cannot be empty")

                artifacts = data.get("artifacts")
                self.assertIsInstance(artifacts, list)
                self.assertGreaterEqual(len(artifacts), 3, f"{filename} must list at least 3 artifacts")
                for art in artifacts:
                    self.assertIsInstance(art, str)
                    self.assertFalse(
                        bool(LEADING_EMOJI_REGEX.match(art.strip())),
                        f"Artifact '{art}' in {filename} has a leading emoji (PostCard.tsx already prepends icons)",
                    )

                key_points = data.get("keyPoints")
                self.assertIsInstance(key_points, list)
                self.assertEqual(len(key_points), 3, f"'keyPoints' in {filename} must contain exactly 3 items")
                for kp in key_points:
                    self.assertIsInstance(kp, str)
                    self.assertGreater(len(kp.strip()), 0)


# ==============================================================================
# TIER 2: STRUCTURAL SECTIONS VALIDATION
# ==============================================================================

class TestPhase02Tier2Structure(unittest.TestCase):
    """Tier 2: Complete 8-section structure, tables, and 3-tier challenges across all 21 articles."""

    def test_01_all_mandatory_sections_present(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)

                self.assertTrue(bool(MOTTO_REGEX.search(content)), f"{filename} missing Motto blockquote")
                self.assertTrue(bool(QUICK_SPEC_REGEX.search(content)), f"{filename} missing Quick Spec")

                obj_match = OBJECTIVES_REGEX.search(content)
                self.assertTrue(bool(obj_match), f"{filename} missing Learning Objectives")
                obj_text = get_section_text(content, obj_match)
                self.assertTrue(bool(re.search(r"-\s*\[[ xX]\]", obj_text)), f"{filename} missing objective checkboxes")

                self.assertTrue(bool(PROBLEM_REGEX.search(content)), f"{filename} missing Section 1 (The Problem)")
                self.assertTrue(bool(CONCEPT_REGEX.search(content)), f"{filename} missing Section 2 (The Concept)")
                self.assertTrue(bool(BUILD_IT_REGEX.search(content)), f"{filename} missing Section 3 (Build It)")
                self.assertTrue(bool(USE_IT_REGEX.search(content)), f"{filename} missing Section 4 (Use It)")
                self.assertTrue(bool(SHIP_IT_REGEX.search(content)), f"{filename} missing Section 5 (Ship It)")
                self.assertTrue(bool(KEY_TERMS_REGEX.search(content)), f"{filename} missing Section 6 (Key Terms)")
                self.assertTrue(bool(CHALLENGES_REGEX.search(content)), f"{filename} missing Section 7 (Challenges)")
                self.assertTrue(bool(FURTHER_READING_REGEX.search(content)), f"{filename} missing Section 8 (Further Reading)")

    def test_02_h2_headings_plain_text(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                _, body, start_line = split_frontmatter_and_content(content)
                _, plain_lines = extract_code_blocks(body, start_line)
                violations = validate_headings(plain_lines, filename)
                self.assertEqual(
                    len(violations),
                    0,
                    f"H2 plain text violations in {filename}:\n"
                    + "\n".join(f"  line {v.line_no}: {v.message}" for v in violations),
                )

    def test_03_concept_has_mermaid_and_display_katex(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                self.assertIn("```mermaid", content, f"{filename} missing Mermaid diagram")
                self.assertTrue(
                    bool(re.search(r"\$\$[\s\S]+?\$\$", content)),
                    f"{filename} missing display KaTeX formula ($$...$$)",
                )

    def test_04_key_terms_table_format(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                m = KEY_TERMS_REGEX.search(content)
                self.assertTrue(bool(m), f"Section 6 heading missing in {filename}")
                sec6_text = get_section_text(content, m)
                has_3col_header = bool(re.search(r"^\s*\|[^|\n]+\|[^|\n]+\|[^|\n]+\|", sec6_text, re.MULTILINE))
                has_separator = bool(re.search(r"^\s*\|(?:\s*:?-+:?\s*\|){3,}", sec6_text, re.MULTILINE))
                self.assertTrue(
                    has_3col_header and has_separator,
                    f"Section 6 in {filename} must contain a 3-column markdown table",
                )

    def test_05_hands_on_challenges_three_tiers(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                m = CHALLENGES_REGEX.search(content)
                self.assertTrue(bool(m), f"Section 7 heading missing in {filename}")
                sec7_text = get_section_text(content, m)
                self.assertIn("🟢", sec7_text, f"Section 7 in {filename} missing 🟢 Easy tier")
                self.assertIn("🟡", sec7_text, f"Section 7 in {filename} missing 🟡 Medium tier")
                self.assertIn("🔴", sec7_text, f"Section 7 in {filename} missing 🔴 Hard tier")


# ==============================================================================
# TIER 3: CODE INTEGRITY & ANTI-BLACKBOX AST AUDIT
# ==============================================================================

class TestPhase02Tier3CodeExecution(unittest.TestCase):
    """Tier 3: Code Extraction, Anti-Blackbox AST Audit & Dynamic Execution."""

    def test_01_section_3_python_blocks_present(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                blocks = extract_section_3_python_blocks(content)
                self.assertGreater(
                    len(blocks),
                    0,
                    f"Section 3 'Build It' in {filename} does not contain any Python code blocks",
                )

    def test_02_section_3_ast_syntax_and_anti_blackbox(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                blocks = extract_section_3_python_blocks(content)
                if not blocks:
                    self.fail(f"No Python code blocks in Section 3 of {filename}")

                combined_code = "\n\n".join(blocks)
                violations, _ = audit_ast_imports(combined_code, filename)
                self.assertEqual(
                    len(violations),
                    0,
                    f"AST anti-blackbox violations in {filename}:\n" + "\n".join(f"  - {v}" for v in violations),
                )

    def test_03_section_3_dynamic_execution(self):
        numpy_installed = is_numpy_installed()

        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                blocks = extract_section_3_python_blocks(content)
                if not blocks:
                    self.fail(f"No Python code blocks in Section 3 of {filename}")

                combined_code = "\n\n".join(blocks)
                _, uses_numpy = audit_ast_imports(combined_code, filename)

                if uses_numpy and not numpy_installed:
                    self.skipTest(f"NumPy not installed in {sys.executable}; skipping execution for {filename}")

                try:
                    res = subprocess.run(
                        [sys.executable, "-c", combined_code],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                except subprocess.TimeoutExpired:
                    self.fail(f"Section 3 execution timed out (10s) in {filename}")

                self.assertEqual(
                    res.returncode,
                    0,
                    f"Section 3 execution failed in {filename} (exit code {res.returncode}):\n"
                    f"STDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}",
                )


# ==============================================================================
# TIER 4: REAL-WORLD INTEGRATION & FRONTEND PRODUCTION BUILD
# ==============================================================================

class TestPhase02Tier4Integration(unittest.TestCase):
    """Tier 4: KaTeX math validation, Mermaid diagram syntax, and Vite production build."""

    def test_01_katex_syntax(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                _, body, start_line = split_frontmatter_and_content(content)
                _, plain_lines = extract_code_blocks(body, start_line)
                violations = validate_katex_syntax(plain_lines, filename)
                self.assertEqual(
                    len(violations),
                    0,
                    f"KaTeX syntax violations in {filename}:\n"
                    + "\n".join(f"  line {v.line_no}: {v.message}" for v in violations),
                )

    def test_02_mermaid_syntax(self):
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                _, body, start_line = split_frontmatter_and_content(content)
                fenced_blocks, _ = extract_code_blocks(body, start_line)
                violations = validate_mermaid_syntax(fenced_blocks, filename)
                self.assertEqual(
                    len(violations),
                    0,
                    f"Mermaid syntax violations in {filename}:\n"
                    + "\n".join(f"  line {v.line_no}: {v.message}" for v in violations),
                )

    def test_03_frontend_production_build(self):
        self.assertTrue(FRONTEND_DIR.is_dir(), f"Frontend directory missing at {FRONTEND_DIR}")
        npm_bin = shutil.which("npm") or shutil.which("npm.cmd")
        self.assertIsNotNone(npm_bin, "npm executable not found in PATH")

        is_windows = sys.platform.startswith("win")
        try:
            res = subprocess.run(
                [npm_bin, "run", "build"],
                cwd=str(FRONTEND_DIR),
                capture_output=True,
                text=True,
                timeout=60,
                shell=is_windows,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self.fail("Frontend build timed out after 60 seconds")

        self.assertEqual(
            res.returncode,
            0,
            f"Frontend build ('npm run build') failed with exit code {res.returncode}:\n"
            f"STDERR:\n{res.stderr[-2000:] if res.stderr else ''}\n"
            f"STDOUT:\n{res.stdout[-2000:] if res.stdout else ''}",
        )

        dist_dir = FRONTEND_DIR / "dist"
        self.assertTrue(dist_dir.is_dir(), "frontend/dist directory was not created")
        self.assertTrue((dist_dir / "index.html").is_file(), "dist/index.html is missing")
        self.assertTrue((dist_dir / "sitemap.xml").is_file(), "dist/sitemap.xml is missing")
        self.assertTrue((dist_dir / "feed.xml").is_file(), "dist/feed.xml is missing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
