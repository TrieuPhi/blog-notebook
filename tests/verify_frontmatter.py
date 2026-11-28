#!/usr/bin/env python3
"""
verify_frontmatter.py — Automated Frontmatter Schema Validator

Validates YAML frontmatter across all markdown articles in the blog project.
Strictly verifies compatibility with the frontend in-browser parser:
  `blog-notebook/frontend/src/lib/frontmatter.ts`

Rules enforced:
  1. Byte 0 starts with `---` (no UTF-8 BOM, no leading whitespace/newlines).
  2. Frontmatter block closed with `---`.
  3. Key names contain only [A-Za-z]+ (frontend parser regex limitation).
  4. All 7 mandatory fields are present and properly typed:
     - `title`: non-empty string.
     - `description`: non-empty string.
     - `date`: YYYY-MM-DD formatted string representing a valid calendar date.
     - `category`: string matching one of canonical blog categories.
     - `tags`: MUST be a list/array of non-empty strings (NEVER a single string).
     - `status`: MUST be exactly "published".
     - `readingTime`: MUST be a positive integer number (NOT string with 'min').
  5. Optional fields (if present) are validated:
     - `updated`: YYYY-MM-DD formatted valid date string.
     - `featured`: boolean (true/false).

Exit code:
  0: All articles pass validation.
  1: One or more validation violations detected.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Reconfigure stdout/stderr to UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# Canonical categories defined in PROJECT.md and frontend topics
CANONICAL_CATEGORIES = {
    "AI",
    "Agent Engineering",
    "LLM",
    "RAG",
    "RAGAS",
    "MLOps",
    "Backend",
    "DevOps",
    "Python",
    "Developer Tools",
    "Learning",
    "Math & AI Foundations",
    "Machine Learning",
}

DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")
KEY_REGEX = re.compile(r"^[A-Za-z]+$")
UTF8_BOM = b"\xef\xbb\xbf"


class FrontmatterValidationError:
    """Represents a single frontmatter validation violation."""

    def __init__(self, file_path: str | Path, message: str, field: str | None = None):
        self.file_path = str(file_path)
        self.message = message
        self.field = field

    def __str__(self) -> str:
        if self.field:
            return f"[{self.field}] {self.message}"
        return self.message


def parse_raw_frontmatter_block(raw_bytes: bytes, file_name: str) -> tuple[str, str, list[FrontmatterValidationError]]:
    """
    Extracts the YAML frontmatter text and markdown content from raw file bytes.
    Validates byte 0 constraints and delimiter matching.
    """
    errors: list[FrontmatterValidationError] = []

    # 1. Byte 0 BOM check
    if raw_bytes.startswith(UTF8_BOM):
        errors.append(
            FrontmatterValidationError(
                file_name,
                "UTF-8 BOM detected at byte 0 (Byte Order Mark: \\xef\\xbb\\xbf). "
                "Frontend parser regex /^---/ fails on BOM. File must start with clean ASCII '---' at byte 0.",
            )
        )
        # Strip BOM for further parsing so we can detect remaining errors
        raw_bytes = raw_bytes[len(UTF8_BOM) :]

    # 2. Byte 0 starts with '---'
    if not raw_bytes.startswith((b"---\r\n", b"---\n")):
        if raw_bytes.startswith(b"---"):
            errors.append(
                FrontmatterValidationError(
                    file_name,
                    "Opening '---' must be immediately followed by a newline at byte 3.",
                )
            )
        else:
            errors.append(
                FrontmatterValidationError(
                    file_name,
                    "File does not start with '---' at byte 0 (found leading whitespace, newlines, or content).",
                )
            )
        return "", "", errors

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        errors.append(
            FrontmatterValidationError(
                file_name,
                f"File contains invalid UTF-8 encoding: {e}",
            )
        )
        return "", "", errors

    # 3. Match closing delimiter
    match = re.match(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$", text)
    if not match:
        errors.append(
            FrontmatterValidationError(
                file_name,
                "Unclosed frontmatter block. Missing terminating '---' delimiter on its own line.",
            )
        )
        return "", "", errors

    fm_text = match.group(1)
    content_text = match.group(2)
    return fm_text, content_text, errors


def validate_key_names(fm_text: str, file_name: str) -> list[FrontmatterValidationError]:
    r"""
    Validates that all top-level keys in frontmatter conform to ^[A-Za-z]+$.
    The frontend parser (`src/lib/frontmatter.ts` line 32) uses:
      `line.match(/^([A-Za-z]+):\s*(.*)$/)`
    Any key with hyphens, underscores, or numbers is silently ignored by the frontend.
    """
    errors: list[FrontmatterValidationError] = []
    lines = fm_text.splitlines()

    for idx, line in enumerate(lines, start=2):  # line 1 is opening '---'
        trimmed = line.strip()
        if not trimmed or trimmed.startswith(("#", "-")):
            continue
        if ":" in trimmed:
            key_candidate = trimmed.split(":", 1)[0].strip()
            # If top-level key (no leading indentation)
            if not line.startswith((" ", "\t")) and not KEY_REGEX.match(key_candidate):
                errors.append(
                        FrontmatterValidationError(
                            file_name,
                            f"Invalid frontmatter key '{key_candidate}' at line {idx}. "
                            f"Keys must match '^[A-Za-z]+$' (no hyphens, underscores, or digits) "
                            f"to prevent silent rejection by the frontend parser.",
                            field=key_candidate,
                        )
                    )
    return errors


def validate_fields(data: Any, file_name: str) -> list[FrontmatterValidationError]:
    """
    Validates the 7 mandatory fields and any optional fields against the schema.
    """
    errors: list[FrontmatterValidationError] = []

    if not isinstance(data, dict):
        errors.append(
            FrontmatterValidationError(
                file_name,
                f"Frontmatter must parse into a YAML key-value dictionary, got: {type(data).__name__}",
            )
        )
        return errors

    # Check 7 mandatory fields
    # 1. title: str (non-empty)
    if "title" not in data:
        errors.append(FrontmatterValidationError(file_name, "Missing mandatory field 'title'", field="title"))
    elif not isinstance(data["title"], str):
        errors.append(
            FrontmatterValidationError(
                file_name,
                f"Field 'title' must be a string, got {type(data['title']).__name__}",
                field="title",
            )
        )
    elif not data["title"].strip():
        errors.append(FrontmatterValidationError(file_name, "Field 'title' cannot be empty or whitespace", field="title"))

    # 2. description: str (non-empty)
    if "description" not in data:
        errors.append(FrontmatterValidationError(file_name, "Missing mandatory field 'description'", field="description"))
    elif not isinstance(data["description"], str):
        errors.append(
            FrontmatterValidationError(
                file_name,
                f"Field 'description' must be a string, got {type(data['description']).__name__}",
                field="description",
            )
        )
    elif not data["description"].strip():
        errors.append(
            FrontmatterValidationError(
                file_name,
                "Field 'description' cannot be empty or whitespace",
                field="description",
            )
        )

    # 3. date: YYYY-MM-DD string
    if "date" not in data:
        errors.append(FrontmatterValidationError(file_name, "Missing mandatory field 'date'", field="date"))
    else:
        raw_date = data["date"]
        # Handle PyYAML auto-parsing date into datetime.date
        if isinstance(raw_date, datetime.date):
            date_str = raw_date.strftime("%Y-%m-%d")
        elif isinstance(raw_date, str):
            date_str = raw_date.strip().strip("\"'")
        else:
            date_str = str(raw_date)

        if not DATE_REGEX.match(date_str):
            errors.append(
                FrontmatterValidationError(
                    file_name,
                    f"Field 'date' must follow format 'YYYY-MM-DD' (e.g. '2026-10-28'), got '{raw_date}'",
                    field="date",
                )
            )
        else:
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                errors.append(
                    FrontmatterValidationError(
                        file_name,
                        f"Field 'date' contains invalid calendar date '{date_str}'",
                        field="date",
                    )
                )

    # 4. category: str (non-empty & recognized)
    if "category" not in data:
        errors.append(FrontmatterValidationError(file_name, "Missing mandatory field 'category'", field="category"))
    elif not isinstance(data["category"], str):
        errors.append(
            FrontmatterValidationError(
                file_name,
                f"Field 'category' must be a string, got {type(data['category']).__name__}",
                field="category",
            )
        )
    else:
        cat = data["category"].strip()
        if not cat:
            errors.append(FrontmatterValidationError(file_name, "Field 'category' cannot be empty", field="category"))
        elif cat not in CANONICAL_CATEGORIES:
            valid_cats = ", ".join(sorted(CANONICAL_CATEGORIES))
            errors.append(
                FrontmatterValidationError(
                    file_name,
                    f"Field 'category' '{cat}' is not in canonical list: [{valid_cats}]",
                    field="category",
                )
            )

    # 5. tags: MUST be list/array of strings (NEVER a single string)
    if "tags" not in data:
        errors.append(FrontmatterValidationError(file_name, "Missing mandatory field 'tags'", field="tags"))
    elif not isinstance(data["tags"], list):
        errors.append(
            FrontmatterValidationError(
                file_name,
                f"CRITICAL: Field 'tags' must be a YAML array / list of strings (e.g. ['AI', 'ML']), "
                f"got {type(data['tags']).__name__}: '{data['tags']}'. "
                f"A string causes runtime crash 'post.tags.map is not a function' in the frontend React app.",
                field="tags",
            )
        )
    elif len(data["tags"]) == 0:
        errors.append(FrontmatterValidationError(file_name, "Field 'tags' must contain at least 1 tag", field="tags"))
    else:
        for idx, tag in enumerate(data["tags"]):
            if not isinstance(tag, str) or not tag.strip():
                errors.append(
                    FrontmatterValidationError(
                        file_name,
                        f"Tag at index {idx} in 'tags' must be a non-empty string, got: {tag!r}",
                        field="tags",
                    )
                )

    # 6. status: MUST be "published"
    if "status" not in data:
        errors.append(FrontmatterValidationError(file_name, "Missing mandatory field 'status'", field="status"))
    elif not isinstance(data["status"], str):
        errors.append(
            FrontmatterValidationError(
                file_name,
                f"Field 'status' must be a string, got {type(data['status']).__name__}",
                field="status",
            )
        )
    elif data["status"].strip() != "published":
        errors.append(
            FrontmatterValidationError(
                file_name,
                f"Field 'status' must be exactly 'published', got '{data['status']}'",
                field="status",
            )
        )

    # 7. readingTime: MUST be integer number (NOT string)
    if "readingTime" not in data:
        errors.append(FrontmatterValidationError(file_name, "Missing mandatory field 'readingTime'", field="readingTime"))
    elif isinstance(data["readingTime"], bool):
        errors.append(
            FrontmatterValidationError(
                file_name,
                "Field 'readingTime' cannot be a boolean",
                field="readingTime",
            )
        )
    elif not isinstance(data["readingTime"], int):
        errors.append(
            FrontmatterValidationError(
                file_name,
                f"Field 'readingTime' must be an integer number of minutes (e.g. 10, 15), "
                f"got {type(data['readingTime']).__name__}: '{data['readingTime']}'. "
                f"Do NOT include text like 'min' or 'min read' as the UI template automatically appends ' min'.",
                field="readingTime",
            )
        )
    elif data["readingTime"] <= 0:
        errors.append(
            FrontmatterValidationError(
                file_name,
                f"Field 'readingTime' must be a positive integer greater than 0, got {data['readingTime']}",
                field="readingTime",
            )
        )

    # Validate optional fields if present
    # updated: YYYY-MM-DD
    if "updated" in data and data["updated"] is not None:
        raw_up = data["updated"]
        if isinstance(raw_up, datetime.date):
            up_str = raw_up.strftime("%Y-%m-%d")
        elif isinstance(raw_up, str):
            up_str = raw_up.strip().strip("\"'")
        else:
            up_str = str(raw_up)

        if not DATE_REGEX.match(up_str):
            errors.append(
                FrontmatterValidationError(
                    file_name,
                    f"Optional field 'updated' must follow format 'YYYY-MM-DD', got '{raw_up}'",
                    field="updated",
                )
            )
        else:
            try:
                datetime.datetime.strptime(up_str, "%Y-%m-%d")
            except ValueError:
                errors.append(
                    FrontmatterValidationError(
                        file_name,
                        f"Optional field 'updated' contains invalid calendar date '{up_str}'",
                        field="updated",
                    )
                )

    # featured: boolean
    if "featured" in data and data["featured"] is not None and not isinstance(data["featured"], bool):
        errors.append(
            FrontmatterValidationError(
                file_name,
                f"Optional field 'featured' must be a boolean (true or false), got {type(data['featured']).__name__}",
                field="featured",
            )
        )

    return errors


def validate_file(file_path: Path) -> list[FrontmatterValidationError]:
    """
    Validates a single markdown article file.
    Returns a list of validation errors (empty if valid).
    """
    errors: list[FrontmatterValidationError] = []

    try:
        raw_bytes = file_path.read_bytes()
    except OSError as e:
        return [FrontmatterValidationError(file_path.name, f"Failed to read file: {e}")]

    if len(raw_bytes) == 0:
        return [FrontmatterValidationError(file_path.name, "File is completely empty (0 bytes).")]

    fm_text, _, parse_errors = parse_raw_frontmatter_block(raw_bytes, file_path.name)
    errors.extend(parse_errors)

    if parse_errors:
        return errors

    # Check key names
    key_errors = validate_key_names(fm_text, file_path.name)
    errors.extend(key_errors)

    # Parse YAML data
    try:
        data = yaml.safe_load(fm_text)
    except yaml.YAMLError as e:
        errors.append(
            FrontmatterValidationError(
                file_path.name,
                f"YAML parsing error in frontmatter: {e}",
            )
        )
        return errors

    # Validate field schema
    field_errors = validate_fields(data, file_path.name)
    errors.extend(field_errors)

    return errors


def validate_posts_directory(directory_path: Path) -> tuple[int, int, dict[str, list[FrontmatterValidationError]]]:
    """
    Validates all .md files in the specified directory.
    Returns (pass_count, fail_count, error_dict).
    """
    if not directory_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    md_files = sorted(directory_path.glob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown (.md) files found in: {directory_path}")

    results: dict[str, list[FrontmatterValidationError]] = {}
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
        description="Verify YAML frontmatter compliance for blog notebook articles."
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
    print("           BLOG NOTEBOOK -- FRONTMATTER SCHEMA VALIDATOR")
    print("======================================================================")

    if args.file:
        target = args.file
        print(f"Checking single file: {target}")
        errs = validate_file(target)
        if not errs:
            print(f" [PASS] {target.name} -- All frontmatter checks passed successfully.")
            return 0
        else:
            print(f" [FAIL] {target.name} — {len(errs)} violation(s) detected:")
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
            print("   -> All 7 mandatory fields, byte 0, and key formats verified clean.")

    print("\n----------------------------------------------------------------------")
    print(f"Summary: Total: {total} | Passed: {pass_count} | Failed: {fail_count}")
    print("----------------------------------------------------------------------")

    if fail_count > 0:
        print(f"RESULT: FAILED ({fail_count} file(s) failed frontmatter validation).")
        return 1

    print("RESULT: PASSED (100% of articles have valid frontmatter).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
