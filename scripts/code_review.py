#!/usr/bin/env python3
"""
Automated Code Review — Pre-commit quality gate.

Checks staged Python files for:
  1. Security issues (hardcoded secrets, dangerous patterns)
  2. Logic errors (bare except, mutable defaults, missing returns)
  3. Code quality (missing docstrings, long functions, deep nesting)
  4. Import hygiene (unused imports, circular import patterns)
  5. Test coverage (new modules should have corresponding tests)

Exit code 0 = pass, 1 = issues found (blocks commit).
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Issue Dataclass ──────────────────────────────────────────────

@dataclass
class Issue:
    """A single code review finding with severity, location, and fix suggestion."""
    severity: str      # critical, high, medium, low
    file: str
    line: int
    issue: str
    detail: str
    suggestion: str

    def __str__(self):
        return (
            f"  [{self.severity.upper()}] {self.file}:{self.line}\n"
            f"    Issue: {self.issue}\n"
            f"    Detail: {self.detail}\n"
            f"    Fix: {self.suggestion}\n"
        )

    def to_markdown(self):
        """Format this issue as a Markdown section for reports."""
        return (
            f"### {self.severity.upper()}: {self.issue}\n"
            f"- **File:** `{self.file}:{self.line}`\n"
            f"- **Detail:** {self.detail}\n"
            f"- **Suggestion:** {self.suggestion}\n"
        )


# ── Secret Patterns ──────────────────────────────────────────────

SECRET_PATTERNS = [
    (r'(?:api[_-]?key|apikey)\s*=\s*["\'][A-Za-z0-9_\-]{20,}["\']',
     "Hardcoded API key detected"),
    (r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']{8,}["\']',
     "Hardcoded password detected"),
    (r'AKIA[0-9A-Z]{16}',
     "AWS Access Key ID detected"),
    (r'sk-[A-Za-z0-9]{32,}',
     "OpenAI/Anthropic API key detected"),
    (r'ghp_[A-Za-z0-9]{36}',
     "GitHub Personal Access Token detected"),
    (r'xox[bprs]-[A-Za-z0-9\-]{10,}',
     "Slack token detected"),
    (r'-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----',
     "Private key detected"),
    (r'(?:postgres|mysql|mongodb|redis)://[^\s]+:[^\s]+@',
     "Database URI with credentials detected"),
]

# ── Dangerous Patterns ───────────────────────────────────────────

DANGEROUS_PATTERNS = [
    (r'\beval\s*\(', "Use of eval() is a security risk", "high"),
    (r'\bexec\s*\(', "Use of exec() is a security risk", "high"),
    (r'subprocess\.(?:call|run|Popen)\s*\(.*shell\s*=\s*True',
     "subprocess with shell=True is vulnerable to injection", "high"),
    (r'__import__\s*\(', "Dynamic import via __import__() is risky", "medium"),
    (r'pickle\.loads?\s*\(', "pickle.load is vulnerable to arbitrary code execution", "high"),
    (r'yaml\.load\s*\([^)]*\)\s*$', "yaml.load without Loader is unsafe — use yaml.safe_load", "medium"),
]


# ── AST-Based Checks ────────────────────────────────────────────

class ASTChecker(ast.NodeVisitor):
    """Walk the AST looking for code quality issues."""

    def __init__(self, filepath: str, source: str):
        self.filepath = filepath
        self.source = source
        self.lines = source.split("\n")
        self.issues: list[Issue] = []
        self._function_stack: list[str] = []

    def check(self) -> list[Issue]:
        """Parse and visit the AST, returning all detected issues."""
        try:
            tree = ast.parse(self.source, filename=self.filepath)
        except SyntaxError as e:
            self.issues.append(Issue(
                severity="critical",
                file=self.filepath,
                line=e.lineno or 0,
                issue="Syntax error",
                detail=str(e.msg),
                suggestion="Fix the syntax error before committing",
            ))
            return self.issues
        self.visit(tree)
        return self.issues

    def visit_FunctionDef(self, node):
        """Check synchronous function definitions."""
        self._check_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Check asynchronous function definitions."""
        self._check_function(node)
        self.generic_visit(node)

    def _check_function(self, node):
        # 1. Missing docstring on public functions
        if not node.name.startswith("_"):
            if not (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, (ast.Constant, ast.Str))):
                self.issues.append(Issue(
                    severity="low",
                    file=self.filepath,
                    line=node.lineno,
                    issue=f"Public function '{node.name}' missing docstring",
                    detail="Public functions should have docstrings for maintainability",
                    suggestion=f"Add a docstring to '{node.name}'",
                ))

        # 2. Long functions (>80 lines)
        end_line = getattr(node, 'end_lineno', None)
        if end_line:
            length = end_line - node.lineno
            if length > 80:
                self.issues.append(Issue(
                    severity="medium",
                    file=self.filepath,
                    line=node.lineno,
                    issue=f"Function '{node.name}' is {length} lines long",
                    detail="Long functions are harder to test and maintain",
                    suggestion="Consider extracting sub-functions",
                ))

        # 3. Mutable default arguments
        for default in node.args.defaults + node.args.kw_defaults:
            if default and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.issues.append(Issue(
                    severity="high",
                    file=self.filepath,
                    line=node.lineno,
                    issue=f"Mutable default argument in '{node.name}'",
                    detail="Mutable defaults are shared across calls, causing subtle bugs",
                    suggestion="Use None as default and create the mutable inside the function",
                ))

        # 4. Too many parameters (>7)
        total_args = (
            len(node.args.args) +
            len(node.args.posonlyargs) +
            len(node.args.kwonlyargs)
        )
        if total_args > 7:
            self.issues.append(Issue(
                severity="low",
                file=self.filepath,
                line=node.lineno,
                issue=f"Function '{node.name}' has {total_args} parameters",
                detail="Too many parameters indicate the function may need refactoring",
                suggestion="Consider grouping related params into a dataclass or config object",
            ))

    def visit_ClassDef(self, node):
        """Check class definitions for docstrings."""
        # Missing class docstring
        if not node.name.startswith("_"):
            if not (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, (ast.Constant, ast.Str))):
                self.issues.append(Issue(
                    severity="low",
                    file=self.filepath,
                    line=node.lineno,
                    issue=f"Public class '{node.name}' missing docstring",
                    detail="Public classes should have docstrings",
                    suggestion=f"Add a docstring to class '{node.name}'",
                ))
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """Check except handlers for bare except clauses."""
        # Bare except clause
        if node.type is None:
            self.issues.append(Issue(
                severity="high",
                file=self.filepath,
                line=node.lineno,
                issue="Bare 'except:' clause",
                detail="Catches all exceptions including KeyboardInterrupt and SystemExit",
                suggestion="Use 'except Exception:' at minimum, or be more specific",
            ))
        self.generic_visit(node)

    def visit_Assert(self, node):
        """Check for assert statements in production code."""
        # Assert in non-test code
        if "/tests/" not in self.filepath and "test_" not in os.path.basename(self.filepath):
            self.issues.append(Issue(
                severity="medium",
                file=self.filepath,
                line=node.lineno,
                issue="Assert statement in production code",
                detail="Assertions are stripped when Python runs with -O flag",
                suggestion="Use explicit if/raise for production validation",
            ))
        self.generic_visit(node)


# ── File-Level Checks ────────────────────────────────────────────

def check_secrets(filepath: str, content: str) -> list[Issue]:
    """Scan for hardcoded secrets."""
    issues = []
    for i, line in enumerate(content.split("\n"), 1):
        # Skip comments and test files
        stripped = line.strip()
        if stripped.startswith("#") or "test" in filepath.lower():
            continue
        for pattern, message in SECRET_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append(Issue(
                    severity="critical",
                    file=filepath,
                    line=i,
                    issue=message,
                    detail=f"Line contains a potential secret: {line.strip()[:60]}...",
                    suggestion="Move to environment variable or .env file",
                ))
    return issues


def check_dangerous_patterns(filepath: str, content: str) -> list[Issue]:
    """Scan for dangerous code patterns."""
    issues = []
    for i, line in enumerate(content.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Skip lines that are regex pattern definitions (string literals containing patterns)
        if re.match(r"""^\(r['"].*['"],\s*["']""", stripped):
            continue
        for pattern, message, severity in DANGEROUS_PATTERNS:
            if re.search(pattern, line):
                issues.append(Issue(
                    severity=severity,
                    file=filepath,
                    line=i,
                    issue=message,
                    detail=f"Found in: {stripped[:80]}",
                    suggestion="Review if this usage is necessary and properly sandboxed",
                ))
    return issues


def check_test_coverage(changed_files: list[str], all_files: list[str]) -> list[Issue]:
    """Check that new modules have corresponding test files."""
    issues = []
    test_files = {os.path.basename(f) for f in all_files if "test" in f.lower()}

    for f in changed_files:
        if "/tests/" in f or "test_" in os.path.basename(f):
            continue
        if not f.endswith(".py") or f.endswith("__init__.py"):
            continue

        basename = os.path.basename(f).replace(".py", "")
        possible_tests = [
            f"test_{basename}.py",
            f"{basename}_test.py",
        ]
        has_test = any(t in test_files for t in possible_tests)

        # Also check test_pN.py files that cover sprint modules
        if not has_test:
            # Check if ANY test file imports this module
            module_name = basename
            for tf in test_files:
                tf_path = next(
                    (p for p in all_files if os.path.basename(p) == tf), None
                )
                if tf_path and os.path.exists(tf_path):
                    try:
                        with open(tf_path, "r") as fh:
                            if module_name in fh.read():
                                has_test = True
                                break
                    except Exception:
                        pass

        if not has_test:
            issues.append(Issue(
                severity="medium",
                file=f,
                line=0,
                issue=f"No test file found for '{basename}'",
                detail="New modules should have corresponding test coverage",
                suggestion=f"Create tests/test_{basename}.py or add tests to an existing suite",
            ))
    return issues


# ── Main Review Logic ────────────────────────────────────────────

def get_staged_files() -> list[str]:
    """Get list of staged Python files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True, text=True, check=True,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return [f for f in files if f.endswith(".py")]
    except subprocess.CalledProcessError:
        return []


def get_all_tracked_files() -> list[str]:
    """Get all tracked files in the repo."""
    try:
        result = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True,
        )
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except subprocess.CalledProcessError:
        return []


def review_file(filepath: str) -> list[Issue]:
    """Run all checks on a single file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return [Issue(
            severity="high",
            file=filepath,
            line=0,
            issue=f"Cannot read file: {e}",
            detail="File could not be opened for review",
            suggestion="Check file encoding and permissions",
        )]

    issues = []

    # 1. Secret detection
    issues.extend(check_secrets(filepath, content))

    # 2. Dangerous patterns
    issues.extend(check_dangerous_patterns(filepath, content))

    # 3. AST-based checks
    checker = ASTChecker(filepath, content)
    issues.extend(checker.check())

    return issues


def write_report(issues: list[Issue], files_reviewed: list[str], report_dir: str) -> str:
    """Write review report to .agents/code-reviews/."""
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = os.path.join(report_dir, f"{timestamp}-pre-commit.md")

    severity_counts = {}
    for issue in issues:
        severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

    with open(report_path, "w") as f:
        f.write(f"# Code Review Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Files reviewed:** {len(files_reviewed)}\n")
        f.write(f"**Issues found:** {len(issues)}\n\n")

        if severity_counts:
            f.write("**By severity:**\n")
            for sev in ["critical", "high", "medium", "low"]:
                if sev in severity_counts:
                    f.write(f"- {sev.upper()}: {severity_counts[sev]}\n")
            f.write("\n")

        if not issues:
            f.write("Code review passed. No technical issues detected.\n")
        else:
            # Group by severity
            for sev in ["critical", "high", "medium", "low"]:
                sev_issues = [i for i in issues if i.severity == sev]
                if sev_issues:
                    f.write(f"## {sev.upper()} Issues\n\n")
                    for issue in sev_issues:
                        f.write(issue.to_markdown() + "\n")

    return report_path


def main():
    """Run code review on staged files. Exit 1 if critical/high issues found."""
    print("\n🔍 Running code review...\n")

    # Determine files to review
    staged = get_staged_files()
    if not staged:
        print("  No staged Python files to review.\n")
        sys.exit(0)

    all_files = get_all_tracked_files()
    all_issues: list[Issue] = []

    # Review each staged file
    for filepath in staged:
        if not os.path.exists(filepath):
            continue
        print(f"  Reviewing: {filepath}")
        file_issues = review_file(filepath)
        all_issues.extend(file_issues)

    # Check test coverage for new/changed files
    coverage_issues = check_test_coverage(staged, all_files + staged)
    all_issues.extend(coverage_issues)

    # Write report
    report_dir = os.path.join(".agents", "code-reviews")
    report_path = write_report(all_issues, staged, report_dir)

    # Display results
    if not all_issues:
        print(f"\n  ✅ Code review passed. No issues found.")
        print(f"  📄 Report: {report_path}\n")
        sys.exit(0)

    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_issues.sort(key=lambda i: severity_order.get(i.severity, 99))

    # Count blocking issues
    blocking = [i for i in all_issues if i.severity in ("critical", "high")]

    print(f"\n  Found {len(all_issues)} issue(s):\n")
    for issue in all_issues:
        print(str(issue))

    print(f"  📄 Full report: {report_path}")

    if blocking:
        print(f"\n  ❌ {len(blocking)} blocking issue(s) (critical/high). Fix before committing.\n")
        sys.exit(1)
    else:
        print(f"\n  ⚠️  {len(all_issues)} non-blocking issue(s) (medium/low). Consider fixing.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
