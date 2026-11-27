#!/usr/bin/env python3
"""tests/test_phase01_e2e.py - Standalone E2E Test Suite for blog-notebook Phase 01 Rewrite.

Covers all 22 articles in frontend/posts/phases/01-math-foundations/ across 4 distinct tiers:
  - Tier 1 (TestPhase01Tier1Frontmatter):
      Byte 0 delimiter, BOM rejection, ^[A-Za-z]+$ keys, mandatory core fields
      (title, description, date, category "Math & AI Foundations", tags array, status "published",
      readingTime int), Phase 01 metadata (lang "vi", phase "01-math-foundations", phaseNumber "01",
      lesson, translationKey, slug, motto, artifacts, keyPoints 3-item array).
  - Tier 2 (TestPhase01Tier2Structure):
      Motto blockquote, Quick Spec, Learning Objectives with checkboxes, 6 numbered sections
      (The Problem, The Concept, Build It, Use It, Key Terms with 3-column table,
      Hands-on Challenges with 3 tiers: Easy, Medium, Hard), and plain text H2 headings.
  - Tier 3 (TestPhase01Tier3CodeExecution):
      Section 3 Python blocks extraction, AST syntax validation, anti-blackbox ML import audit
      (only stdlib and numpy allowed; forbids torch, sklearn, scipy, tensorflow, etc.),
      and dynamic execution in isolated subprocess with 10s timeout (skipping gracefully if numpy
      is not installed in local runner).
  - Tier 4 (TestPhase01Tier4Integration):
      KaTeX math syntax (even $$, non-empty, balanced {}, clean inline spacing, no unescaped currency),
      Mermaid diagram syntax (valid diagram type, double-quoted node labels for flowcharts),
      and full production Vite frontend build (npm run build).

Executable via:
  python3 -m unittest tests/test_phase01_e2e.py
  python3 -m unittest tests.test_phase01_e2e.TestPhase01Tier1Frontmatter
  pytest tests/test_phase01_e2e.py
"""

from __future__ import annotations

import ast
import datetime
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

import yaml

# Ensure project root is in sys.path so modules in tests/ can be imported cleanly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.verify_frontmatter import (
    DATE_REGEX,
    KEY_REGEX,
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

PHASE01_DIR = REPO_ROOT / "frontend" / "posts" / "phases" / "01-math-foundations"
FRONTEND_DIR = REPO_ROOT / "frontend"

EXPECTED_ARTICLES: list[str] = [
    "01-linear-algebra-intuition.md",
    "02-vectors-matrices-operations.md",
    "03-matrix-transformations.md",
    "04-calculus-for-ml.md",
    "05-chain-rule-and-autodiff.md",
    "06-probability-and-distributions.md",
    "07-bayes-theorem.md",
    "08-optimization.md",
    "09-information-theory.md",
    "10-dimensionality-reduction.md",
    "11-singular-value-decomposition.md",
    "12-tensor-operations.md",
    "13-numerical-stability.md",
    "14-norms-and-distances.md",
    "15-statistics-for-ml.md",
    "16-sampling-methods.md",
    "17-linear-systems.md",
    "18-convex-optimization.md",
    "19-complex-numbers.md",
    "20-fourier-transform.md",
    "21-graph-theory.md",
    "22-stochastic-processes.md",
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
}

ALLOWED_MODULES: set[str] = set(sys.stdlib_module_names) | set(sys.builtin_module_names) | {"numpy"}

# Section regex patterns
MOTTO_REGEX = re.compile(r"^>\s*\*\*.*?\*\*", re.MULTILINE)
QUICK_SPEC_REGEX = re.compile(r"^##\s+.*?(?:Thông\s+tin\s+nhanh|Quick\s+Spec)", re.IGNORECASE | re.MULTILINE)
OBJECTIVES_REGEX = re.compile(r"^##\s+.*?(?:Mục\s+tiêu\s+bài\s+học|Learning\s+Objectives)", re.IGNORECASE | re.MULTILINE)
PROBLEM_REGEX = re.compile(r"^##\s+(?:1\.\s+.*?(?:Vấn\s+đề\s+thực\s+tế|The\s+Problem)|The\s+Problem)", re.IGNORECASE | re.MULTILINE)
CONCEPT_REGEX = re.compile(r"^##\s+(?:2\.\s+.*?(?:Mô\s+hình\s+tư\s+duy|The\s+Concept)|The\s+Concept)", re.IGNORECASE | re.MULTILINE)
BUILD_IT_REGEX = re.compile(r"^##\s+(?:3\.\s+.*?(?:Tự\s+tay\s+xây\s+dựng|Build\s+It)|Build\s+It|Building\s+It)", re.IGNORECASE | re.MULTILINE)
USE_IT_REGEX = re.compile(r"^##\s+(?:4\.\s+.*?(?:Ứng\s+dụng|Use\s+It)|Use\s+It)", re.IGNORECASE | re.MULTILINE)
KEY_TERMS_REGEX = re.compile(r"^##\s+(?:5\.\s+.*?(?:Giải\s+ngố|Key\s+Terms)|Key\s+Terms)", re.IGNORECASE | re.MULTILINE)
CHALLENGES_REGEX = re.compile(r"^##\s+(?:6\.\s+.*?(?:Bài\s+tập|Thử\s+thách|Challenges|Exercises)|Challenges|Exercises)", re.IGNORECASE | re.MULTILINE)

NEXT_H2_REGEX = re.compile(r"^##\s+[^\n]+$", re.MULTILINE)
PYTHON_FENCE_REGEX = re.compile(r"^```(?:python|py)\s*\r?\n([\s\S]*?)\r?\n```", re.IGNORECASE | re.MULTILINE)


# --- Helper Functions ---

def is_numpy_installed() -> bool:
    """Check if numpy is available in the current python runtime."""
    try:
        res = subprocess.run([sys.executable, "-c", "import numpy"], capture_output=True, timeout=5)
        return res.returncode == 0
    except Exception:
        return False


def load_article_raw(filename: str) -> tuple[bytes, str]:
    """Read raw bytes and utf-8 text from an article."""
    file_path = PHASE01_DIR / filename
    raw_bytes = file_path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    return raw_bytes, text


def parse_frontmatter(filename: str) -> tuple[dict[str, Any] | None, str, list[str]]:
    """
    Parses frontmatter using byte-0 delimiter checks and PyYAML safe_load.
    Returns (data_dict, markdown_body, errors_list).
    """
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
    """Extracts section body text from heading match to the next H2 heading or EOF."""
    if not heading_match:
        return ""
    start_pos = heading_match.end()
    next_match = NEXT_H2_REGEX.search(content, pos=start_pos)
    end_pos = next_match.start() if next_match else len(content)
    return content[start_pos:end_pos]


def extract_section_3_python_blocks(content: str) -> list[str]:
    """Extracts all fenced python code blocks strictly from Section 3 'Build It'."""
    m = BUILD_IT_REGEX.search(content)
    if not m:
        return []
    sec3_text = get_section_text(content, m)
    return PYTHON_FENCE_REGEX.findall(sec3_text)


def audit_ast_imports(code_str: str, filename: str) -> tuple[list[str], bool]:
    """
    Audits Python AST for forbidden black-box ML imports and unapproved packages.
    Returns (violations_list, uses_numpy_flag).
    """
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
                violations.append(
                    f"Relative import detected in {filename}:{node.lineno}"
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                violations.append(
                    f"Dynamic import '__import__' detected in {filename}:{node.lineno}"
                )
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                violations.append(
                    f"Dynamic import 'import_module' detected in {filename}:{node.lineno}"
                )

    return violations, uses_numpy


# ==============================================================================
# TIER 1: FRONTMATTER SCHEMA VALIDATION
# ==============================================================================

class TestPhase01Tier1Frontmatter(unittest.TestCase):
    """Tier 1: Feature Coverage & Frontmatter Schema Validation across all 22 articles."""

    def test_01_all_articles_exist(self):
        """Assert all 22 Phase 01 markdown files exist in content directory."""
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                file_path = PHASE01_DIR / filename
                self.assertTrue(file_path.is_file(), f"Article missing from disk: {file_path}")

    def test_02_byte0_and_bom_rejection(self):
        """Assert every article starts at byte 0 with '---' delimiter without UTF-8 BOM."""
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                raw_bytes, _ = load_article_raw(filename)
                self.assertFalse(
                    raw_bytes.startswith(UTF8_BOM),
                    f"{filename} contains UTF-8 BOM (\\xef\\xbb\\xbf) at byte 0"
                )
                self.assertTrue(
                    raw_bytes.startswith((b"---\n", b"---\r\n")),
                    f"{filename} does not start with '---' delimiter followed by newline at byte 0"
                )
                # Check closing delimiter exists
                text = raw_bytes.decode("utf-8", errors="replace")
                self.assertRegex(
                    text,
                    r"^---\r?\n[\s\S]*?\r?\n---\r?\n?",
                    f"{filename} does not contain properly closed '---' frontmatter delimiter"
                )

    def test_03_top_level_key_syntax(self):
        """Assert all top-level frontmatter keys conform to ^[A-Za-z]+$ to prevent silent frontend drop."""
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                raw_bytes, _ = load_article_raw(filename)
                fm_text, _, errors = parse_raw_frontmatter_block(raw_bytes, filename)
                if errors:
                    self.fail(f"Cannot extract frontmatter from {filename}: {errors}")
                key_violations = validate_key_names(fm_text, filename)
                self.assertEqual(
                    len(key_violations), 0,
                    f"Frontmatter key violations in {filename}:\n" + "\n".join(str(e) for e in key_violations)
                )

    def test_04_mandatory_core_fields(self):
        """Assert title, description, date, category, tags, status, readingTime are complete and valid."""
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                data, _, errors = parse_frontmatter(filename)
                if errors or data is None:
                    self.fail(f"Frontmatter parse error in {filename}: {errors}")

                # 1. title: non-empty string
                title = data.get("title")
                self.assertIsInstance(title, str, f"'title' in {filename} must be string")
                self.assertGreater(len(title.strip()), 0, f"'title' in {filename} cannot be empty")

                # 2. description: non-empty string
                description = data.get("description")
                self.assertIsInstance(description, str, f"'description' in {filename} must be string")
                self.assertGreater(len(description.strip()), 0, f"'description' in {filename} cannot be empty")

                # 3. date: valid YYYY-MM-DD
                raw_date = data.get("date")
                if isinstance(raw_date, datetime.date):
                    date_str = raw_date.strftime("%Y-%m-%d")
                elif isinstance(raw_date, str):
                    date_str = raw_date.strip()
                else:
                    self.fail(f"'date' in {filename} must be string or date, got {type(raw_date).__name__}")
                self.assertTrue(DATE_REGEX.match(date_str), f"'date' in {filename} must match YYYY-MM-DD, got '{date_str}'")
                try:
                    datetime.datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError as e:
                    self.fail(f"'date' '{date_str}' in {filename} is not a valid calendar date: {e}")

                # 4. category: must be "Math & AI Foundations"
                self.assertEqual(
                    data.get("category"), "Math & AI Foundations",
                    f"'category' in {filename} must be 'Math & AI Foundations', got '{data.get('category')}'"
                )

                # 5. tags: array of non-empty strings
                tags = data.get("tags")
                self.assertIsInstance(tags, list, f"'tags' in {filename} must be a list/array")
                self.assertGreater(len(tags), 0, f"'tags' in {filename} cannot be empty")
                for tag in tags:
                    self.assertIsInstance(tag, str, f"Tag in {filename} must be a string: {tag}")
                    self.assertGreater(len(tag.strip()), 0, f"Tag in {filename} cannot be blank")

                # 6. status: must be "published"
                self.assertEqual(
                    data.get("status"), "published",
                    f"'status' in {filename} must be 'published', got '{data.get('status')}'"
                )

                # 7. readingTime: positive integer
                rt = data.get("readingTime")
                self.assertTrue(
                    isinstance(rt, int) and not isinstance(rt, bool),
                    f"'readingTime' in {filename} must be an integer, got {type(rt).__name__}"
                )
                self.assertGreater(rt, 0, f"'readingTime' in {filename} must be positive, got {rt}")

    def test_05_phase01_metadata_fields(self):
        """Assert lang, phase, phaseNumber, lesson, translationKey, slug, motto, artifacts, keyPoints."""
        required_artifacts = {"Code mẫu", "Công thức", "Visual Guide"}

        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                data, _, errors = parse_frontmatter(filename)
                if errors or data is None:
                    self.fail(f"Frontmatter parse error in {filename}: {errors}")

                # lang, phase, phaseNumber
                self.assertEqual(data.get("lang"), "vi", f"'lang' in {filename} must be 'vi'")
                self.assertEqual(data.get("phase"), "01-math-foundations", f"'phase' in {filename} must be '01-math-foundations'")
                self.assertEqual(str(data.get("phaseNumber")), "01", f"'phaseNumber' in {filename} must be '01'")

                # lesson, translationKey, slug, motto
                for field in ("lesson", "translationKey", "slug", "motto"):
                    val = data.get(field)
                    self.assertIsInstance(val, str, f"'{field}' in {filename} must be string")
                    self.assertGreater(len(val.strip()), 0, f"'{field}' in {filename} cannot be empty")

                # artifacts
                artifacts = data.get("artifacts")
                self.assertIsInstance(artifacts, list, f"'artifacts' in {filename} must be list")
                missing_artifacts = required_artifacts - set(artifacts)
                self.assertEqual(
                    len(missing_artifacts), 0,
                    f"'artifacts' in {filename} is missing required items: {missing_artifacts}"
                )

                # keyPoints: exactly 3 non-empty strings
                key_points = data.get("keyPoints")
                self.assertIsInstance(key_points, list, f"'keyPoints' in {filename} must be list")
                self.assertEqual(
                    len(key_points), 3,
                    f"'keyPoints' in {filename} must contain exactly 3 items, got {len(key_points)}"
                )
                for i, kp in enumerate(key_points, 1):
                    self.assertIsInstance(kp, str, f"KeyPoint #{i} in {filename} must be string")
                    self.assertGreater(len(kp.strip()), 0, f"KeyPoint #{i} in {filename} cannot be blank")


# ==============================================================================
# TIER 2: BOUNDARY & STRUCTURAL SECTIONS VALIDATION
# ==============================================================================

class TestPhase01Tier2Structure(unittest.TestCase):
    """Tier 2: Boundary & Structural Sections Validation across all 22 articles."""

    def test_01_mandatory_sections_present(self):
        """Assert all 6 numbered sections, Motto callout, Quick Spec, and Objectives exist in order."""
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)

                # 1. Motto blockquote (> **"..."** — ... or > **...**)
                self.assertTrue(
                    bool(MOTTO_REGEX.search(content)),
                    f"{filename} missing core Motto blockquote (> **...**)"
                )

                # 2. Quick Spec (## Thông tin nhanh (Quick Spec) or ## Thông tin nhanh)
                self.assertTrue(
                    bool(QUICK_SPEC_REGEX.search(content)),
                    f"{filename} missing '## Thông tin nhanh (Quick Spec)' heading"
                )

                # 3. Learning Objectives (## Mục tiêu bài học (Learning Objectives))
                obj_match = OBJECTIVES_REGEX.search(content)
                self.assertTrue(
                    bool(obj_match),
                    f"{filename} missing '## Mục tiêu bài học (Learning Objectives)' heading"
                )
                obj_text = get_section_text(content, obj_match)
                self.assertTrue(
                    bool(re.search(r"-\s*\[[ xX]\]", obj_text)),
                    f"{filename} Objectives section must contain '- [ ]' checklist items"
                )

                # 4. Section 1: The Problem
                self.assertTrue(
                    bool(PROBLEM_REGEX.search(content)),
                    f"{filename} missing '## 1. Vấn đề thực tế (The Problem)' heading"
                )

                # 5. Section 2: The Concept
                self.assertTrue(
                    bool(CONCEPT_REGEX.search(content)),
                    f"{filename} missing '## 2. Mô hình tư duy & Bản chất (The Concept)' heading"
                )

                # 6. Section 3: Build It
                self.assertTrue(
                    bool(BUILD_IT_REGEX.search(content)),
                    f"{filename} missing '## 3. Tự tay xây dựng từ đầu (Build It: From Scratch)' heading"
                )

                # 7. Section 4: Use It
                self.assertTrue(
                    bool(USE_IT_REGEX.search(content)),
                    f"{filename} missing '## 4. Ứng dụng AI Engineering & Thực tế Production (Use It)' heading"
                )

                # 8. Section 5: Key Terms
                self.assertTrue(
                    bool(KEY_TERMS_REGEX.search(content)),
                    f"{filename} missing '## 5. Giải ngố thuật ngữ & Ngộ nhận (Key Terms)' heading"
                )

                # 9. Section 6: Hands-on Challenges
                self.assertTrue(
                    bool(CHALLENGES_REGEX.search(content)),
                    f"{filename} missing '## 6. Bài tập thử thách (Hands-on Challenges)' heading"
                )

    def test_02_h2_headings_plain_text(self):
        """Assert all ## headings contain only plain text (NO bold, code, math $, links, or HTML)."""
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                _, body, start_line = split_frontmatter_and_content(content)
                _, plain_lines = extract_code_blocks(body, start_line)
                violations = validate_headings(plain_lines, filename)
                self.assertEqual(
                    len(violations), 0,
                    f"H2 plain text violations in {filename}:\n" + "\n".join(f"  line {v.line_no}: {v.message}" for v in violations)
                )

    def test_03_key_terms_table_format(self):
        """Assert Section 5 contains a 3-column markdown table for terminology."""
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                m = KEY_TERMS_REGEX.search(content)
                self.assertTrue(bool(m), f"Section 5 heading missing in {filename}")
                sec5_text = get_section_text(content, m)
                has_3col_header = bool(re.search(r"^\s*\|[^|\n]+\|[^|\n]+\|[^|\n]+\|", sec5_text, re.MULTILINE))
                has_separator = bool(re.search(r"^\s*\|(?:\s*:?-+:?\s*\|){3,}", sec5_text, re.MULTILINE))
                self.assertTrue(
                    has_3col_header and has_separator,
                    f"Section 5 in {filename} must contain a 3-column markdown table (Thuật ngữ, Ngộ nhận, Bản chất)"
                )

    def test_04_hands_on_challenges_tiers(self):
        """Assert Section 6 includes all 3 challenge tiers: Easy (🟢), Medium (🟡), Hard (🔴)."""
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                m = CHALLENGES_REGEX.search(content)
                self.assertTrue(bool(m), f"Section 6 heading missing in {filename}")
                sec6_text = get_section_text(content, m)
                has_easy = bool(re.search(r"🟢|Dễ|Easy", sec6_text, re.IGNORECASE))
                has_medium = bool(re.search(r"🟡|Vừa|Medium|Trung\s+bình", sec6_text, re.IGNORECASE))
                has_hard = bool(re.search(r"🔴|Khó|Hard|Nâng\s+cao", sec6_text, re.IGNORECASE))

                self.assertTrue(has_easy, f"Section 6 in {filename} missing Easy challenge tier (🟢 Dễ / Easy)")
                self.assertTrue(has_medium, f"Section 6 in {filename} missing Medium challenge tier (🟡 Vừa / Medium)")
                self.assertTrue(has_hard, f"Section 6 in {filename} missing Hard challenge tier (🔴 Khó / Hard)")


# ==============================================================================
# TIER 3: CODE INTEGRITY & ANTI-BLACKBOX AST AUDIT
# ==============================================================================

class TestPhase01Tier3CodeExecution(unittest.TestCase):
    """Tier 3: Code Extraction, Anti-Blackbox AST Audit & Dynamic Execution."""

    def test_01_section_3_python_blocks_present(self):
        """Assert Section 3 'Build It' contains at least one fenced Python code block."""
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                blocks = extract_section_3_python_blocks(content)
                self.assertGreater(
                    len(blocks), 0,
                    f"Section 3 'Build It' in {filename} does not contain any Python code blocks"
                )

    def test_02_section_3_ast_syntax_and_anti_blackbox(self):
        """Audit Python blocks in Section 3 for complete syntax and forbidden black-box ML imports."""
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                blocks = extract_section_3_python_blocks(content)
                if not blocks:
                    self.fail(f"No Python code blocks in Section 3 of {filename} to audit")

                combined_code = "\n\n".join(blocks)
                violations, _ = audit_ast_imports(combined_code, filename)
                self.assertEqual(
                    len(violations), 0,
                    f"AST anti-blackbox violations in {filename}:\n" + "\n".join(f"  - {v}" for v in violations)
                )

    def test_03_section_3_dynamic_execution(self):
        """Execute Section 3 code in isolated subprocess with 10s timeout, asserting exit code 0."""
        numpy_installed = is_numpy_installed()

        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                blocks = extract_section_3_python_blocks(content)
                if not blocks:
                    self.fail(f"No Python code blocks in Section 3 of {filename} to execute")

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
                    res.returncode, 0,
                    f"Section 3 execution failed in {filename} (exit code {res.returncode}):\n"
                    f"STDERR:\n{res.stderr}\n"
                    f"STDOUT:\n{res.stdout}"
                )


# ==============================================================================
# TIER 4: REAL-WORLD INTEGRATION & FRONTEND PRODUCTION BUILD
# ==============================================================================

class TestPhase01Tier4Integration(unittest.TestCase):
    """Tier 4: KaTeX math validation, Mermaid diagram syntax, and Vite production build."""

    def test_01_katex_syntax(self):
        """Assert display and inline KaTeX math formulas are balanced, non-empty, and correctly spaced."""
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                _, body, start_line = split_frontmatter_and_content(content)
                _, plain_lines = extract_code_blocks(body, start_line)
                violations = validate_katex_syntax(plain_lines, filename)
                self.assertEqual(
                    len(violations), 0,
                    f"KaTeX syntax violations in {filename}:\n" + "\n".join(f"  line {v.line_no}: {v.message}" for v in violations)
                )

    def test_02_mermaid_syntax(self):
        """Assert Mermaid diagrams declare valid types and have double-quoted labels for Vietnamese text."""
        for filename in EXPECTED_ARTICLES:
            with self.subTest(article=filename):
                _, content = load_article_raw(filename)
                _, body, start_line = split_frontmatter_and_content(content)
                fenced_blocks, _ = extract_code_blocks(body, start_line)
                violations = validate_mermaid_syntax(fenced_blocks, filename)
                self.assertEqual(
                    len(violations), 0,
                    f"Mermaid syntax violations in {filename}:\n" + "\n".join(f"  line {v.line_no}: {v.message}" for v in violations)
                )

    def test_03_frontend_production_build(self):
        """Assert full static site generation via 'npm run build' generates dist bundle and syndication."""
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
            res.returncode, 0,
            f"Frontend build ('npm run build') failed with exit code {res.returncode}:\n"
            f"STDERR:\n{res.stderr[-2000:] if res.stderr else ''}\n"
            f"STDOUT:\n{res.stdout[-2000:] if res.stdout else ''}"
        )

        dist_dir = FRONTEND_DIR / "dist"
        self.assertTrue(dist_dir.is_dir(), "frontend/dist directory was not created")
        self.assertTrue((dist_dir / "index.html").is_file(), "dist/index.html is missing")
        self.assertTrue((dist_dir / "sitemap.xml").is_file(), "dist/sitemap.xml is missing")
        self.assertTrue((dist_dir / "feed.xml").is_file(), "dist/feed.xml is missing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
