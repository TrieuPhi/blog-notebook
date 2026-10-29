#!/usr/bin/env python3
r"""
run_all_checks.py — Master E2E Test Runner for Blog Notebook

Orchestrates the entire 5-tier testing suite:
  - Tier 1: Frontmatter Schema Validation (`verify_frontmatter.py`)
  - Tier 2: Markdown & KaTeX / Mermaid Syntax Validation (`verify_markdown_syntax.py`)
  - Tier 3: Standard Unit Test Suite (`blog-notebook/tests/standard/`)
  - Tier 4: Heavy GPU Standalone Dry-Run & Compilation (`blog-notebook/tests/gpu_standalone/`)
  - Tier 5: Frontend Build Verification (`npm run build` in `blog-notebook/frontend`)

Exit code:
  0: All active tiers passed successfully.
  1: One or more tiers detected violations or failures.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Reconfigure stdout/stderr to UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


class TierResult:
    """Stores the execution result for a test tier."""

    def __init__(self, name: str, status: str, details: str, duration: float, error_output: str = ""):
        self.name = name
        self.status = status  # "PASS", "FAIL", "SKIP"
        self.details = details
        self.duration = duration
        self.error_output = error_output


def run_command(cmd: list[str] | str, cwd: Path, shell: bool = False) -> tuple[int, str, str]:
    """Runs a subprocess command and returns (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            shell=shell,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (subprocess.SubprocessError, OSError) as e:
        return 1, "", str(e)


def run_tier_1_frontmatter(python_exe: str, repo_root: Path, posts_dir: Path) -> TierResult:
    """Tier 1: Frontmatter Schema Validation."""
    start = time.time()
    script = repo_root / "blog-notebook" / "tests" / "verify_frontmatter.py"
    cmd = [python_exe, str(script), "--dir", str(posts_dir)]
    code, stdout, stderr = run_command(cmd, repo_root)
    duration = time.time() - start

    if code == 0:
        return TierResult("Tier 1: Frontmatter Schema Validation", "PASS", "All articles compliant", duration)
    else:
        err_msg = stdout + "\n" + stderr
        return TierResult("Tier 1: Frontmatter Schema Validation", "FAIL", "Schema violations detected", duration, err_msg.strip())


def run_tier_2_syntax(python_exe: str, repo_root: Path, posts_dir: Path) -> TierResult:
    """Tier 2: KaTeX, Mermaid & Headings Syntax Validation."""
    start = time.time()
    script = repo_root / "blog-notebook" / "tests" / "verify_markdown_syntax.py"
    cmd = [python_exe, str(script), "--dir", str(posts_dir)]
    code, stdout, stderr = run_command(cmd, repo_root)
    duration = time.time() - start

    if code == 0:
        return TierResult("Tier 2: Markdown & Rendering Syntax", "PASS", "All KaTeX, Mermaid & TOC clean", duration)
    else:
        err_msg = stdout + "\n" + stderr
        return TierResult("Tier 2: Markdown & Rendering Syntax", "FAIL", "Syntax violations detected", duration, err_msg.strip())


def run_tier_3_unit_tests(python_exe: str, repo_root: Path) -> TierResult:
    """Tier 3: Standard Unit Tests in tests/standard/."""
    start = time.time()
    standard_dir = repo_root / "blog-notebook" / "tests" / "standard"

    if not standard_dir.exists():
        return TierResult("Tier 3: Standard Unit Test Suite", "SKIP", "Directory does not exist", 0.0)

    test_files = list(standard_dir.glob("test_*.py"))
    if not test_files:
        return TierResult("Tier 3: Standard Unit Test Suite", "SKIP", "0 test files (pending milestone authoring)", 0.0)

    cmd = [python_exe, "-m", "unittest", "discover", "-s", str(standard_dir), "-p", "test_*.py"]
    code, stdout, stderr = run_command(cmd, repo_root)
    duration = time.time() - start

    if code == 0:
        return TierResult("Tier 3: Standard Unit Test Suite", "PASS", f"Executed {len(test_files)} test suite(s)", duration)
    else:
        err_msg = stdout + "\n" + stderr
        return TierResult("Tier 3: Standard Unit Test Suite", "FAIL", "Unit test failures detected", duration, err_msg.strip())


def run_tier_4_gpu_standalone(python_exe: str, repo_root: Path) -> TierResult:
    """Tier 4: Heavy GPU Standalone Dry-Run & Compilation."""
    start = time.time()
    gpu_dir = repo_root / "blog-notebook" / "tests" / "gpu_standalone"

    if not gpu_dir.exists():
        return TierResult("Tier 4: Heavy GPU Standalone Scripts", "SKIP", "Directory does not exist", 0.0)

    scripts = [f for f in gpu_dir.glob("*.py") if f.name != "__init__.py"]
    if not scripts:
        return TierResult("Tier 4: Heavy GPU Standalone Scripts", "SKIP", "0 standalone scripts (pending milestone authoring)", 0.0)

    failures = []
    for s in scripts:
        # 1. py_compile
        c_code, _, c_err = run_command([python_exe, "-m", "py_compile", str(s)], repo_root)
        if c_code != 0:
            failures.append(f"{s.name} failed py_compile:\n{c_err}")
            continue

        # 2. ruff check if available
        r_code, r_out, _ = run_command([python_exe, "-m", "ruff", "check", str(s)], repo_root)
        if r_code != 0:
            failures.append(f"{s.name} failed ruff lint check:\n{r_out}")

        # 3. execution with --dry-run
        d_code, d_out, d_err = run_command([python_exe, str(s), "--dry-run"], repo_root)
        if d_code != 0:
            failures.append(f"{s.name} failed --dry-run:\n{d_out}\n{d_err}")

    duration = time.time() - start
    if not failures:
        return TierResult("Tier 4: Heavy GPU Standalone Scripts", "PASS", f"Verified {len(scripts)} standalone script(s)", duration)
    else:
        return TierResult("Tier 4: Heavy GPU Standalone Scripts", "FAIL", f"{len(failures)} issue(s) detected", duration, "\n".join(failures))


def run_tier_5_frontend_build(repo_root: Path) -> TierResult:
    """Tier 5: Frontend Build Verification (npm run build)."""
    start = time.time()
    frontend_dir = repo_root / "blog-notebook" / "frontend"

    if not frontend_dir.exists():
        return TierResult("Tier 5: Frontend Build Verification", "SKIP", "Frontend dir not found", 0.0)

    npm_bin = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm_bin:
        # Fallback to shell execution
        code, stdout, stderr = run_command("npm run build", frontend_dir, shell=True)
    else:
        code, stdout, stderr = run_command([npm_bin, "run", "build"], frontend_dir, shell=(os.name == "nt"))

    duration = time.time() - start

    if code == 0:
        return TierResult("Tier 5: Frontend Build Verification", "PASS", f"Built successfully in {duration:.1f}s", duration)
    else:
        err_msg = stdout + "\n" + stderr
        return TierResult("Tier 5: Frontend Build Verification", "FAIL", "Vite/TypeScript build error", duration, err_msg.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Master E2E Test Runner for Blog Notebook Articles & Test Suite."
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Path to Python interpreter (default: current sys.executable)",
    )
    parser.add_argument(
        "--posts-dir",
        type=Path,
        default=None,
        help="Custom path to posts directory.",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Run only a specific test tier (1 to 5).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip Tier 5 (npm run build) for fast Python-only validation.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full error details for any failed checks.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    posts_dir = args.posts_dir or (repo_root / "blog-notebook" / "frontend" / "posts")
    python_exe = args.python

    print("======================================================================")
    print("                 BLOG NOTEBOOK -- E2E TEST RUNNER")
    print("======================================================================")
    print(f"Repository Root : {repo_root}")
    print(f"Posts Directory : {posts_dir}")
    print(f"Python Binary   : {python_exe}")
    print("----------------------------------------------------------------------\n")

    results: list[TierResult] = []
    overall_start = time.time()

    # Tier 1
    if args.tier is None or args.tier == 1:
        print(">> Running Tier 1: Frontmatter Schema Validation...")
        r1 = run_tier_1_frontmatter(python_exe, repo_root, posts_dir)
        results.append(r1)
        print(f"   [{r1.status}] {r1.details} ({r1.duration:.2f}s)")

    # Tier 2
    if args.tier is None or args.tier == 2:
        print(">> Running Tier 2: Markdown & Rendering Syntax...")
        r2 = run_tier_2_syntax(python_exe, repo_root, posts_dir)
        results.append(r2)
        print(f"   [{r2.status}] {r2.details} ({r2.duration:.2f}s)")

    # Tier 3
    if args.tier is None or args.tier == 3:
        print(">> Running Tier 3: Standard Unit Test Suite...")
        r3 = run_tier_3_unit_tests(python_exe, repo_root)
        results.append(r3)
        print(f"   [{r3.status}] {r3.details} ({r3.duration:.2f}s)")

    # Tier 4
    if args.tier is None or args.tier == 4:
        print(">> Running Tier 4: Heavy GPU Standalone Scripts...")
        r4 = run_tier_4_gpu_standalone(python_exe, repo_root)
        results.append(r4)
        print(f"   [{r4.status}] {r4.details} ({r4.duration:.2f}s)")

    # Tier 5
    if (args.tier is None or args.tier == 5) and not args.skip_build:
        print(">> Running Tier 5: Frontend Build Verification (npm run build)...")
        r5 = run_tier_5_frontend_build(repo_root)
        results.append(r5)
        print(f"   [{r5.status}] {r5.details} ({r5.duration:.2f}s)")
    elif args.skip_build:
        results.append(TierResult("Tier 5: Frontend Build Verification", "SKIP", "Skipped via --skip-build", 0.0))

    total_duration = time.time() - overall_start

    # Report Table
    print("\n======================================================================")
    print("                      E2E TEST EXECUTION SUMMARY")
    print("======================================================================")
    print(f"{'Tier':<42} | {'Status':<6} | {'Duration':<8} | {'Details'}")
    print("-" * 70)
    for r in results:
        print(f"{r.name:<42} | {r.status:<6} | {r.duration:>6.2f}s | {r.details}")
    print("======================================================================")

    # Print error details if verbose or on failures
    failures = [r for r in results if r.status == "FAIL"]
    if failures:
        print("\nFAILURE DETAILS:")
        for r in failures:
            print(f"\n[{r.name}]:")
            print(r.error_output)
            print("-" * 70)

        print(f"\nOVERALL STATUS: FAILED ({len(failures)} tier(s) failed in {total_duration:.2f}s)")
        return 1

    print(f"\nOVERALL STATUS: PASSED (All active tiers verified in {total_duration:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
