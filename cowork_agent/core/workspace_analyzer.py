"""
Sprint 42 · Workspace Analyzer
================================
Analyzes changed files for common issues:
  - JSON syntax errors
  - YAML parse errors
  - Python syntax errors
  - Git merge conflicts
  - Oversized files
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── Data ────────────────────────────────────────────────────────────

@dataclass
class FileIssue:
    """A detected issue in a workspace file."""
    file_path: str
    issue_type: str  # syntax_error | invalid_json | merge_conflict | large_file | broken_yaml
    description: str
    line_number: Optional[int] = None
    severity: str = "warning"  # info | warning | error
    auto_fixable: bool = False

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "issue_type": self.issue_type,
            "description": self.description,
            "line_number": self.line_number,
            "severity": self.severity,
            "auto_fixable": self.auto_fixable,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FileIssue":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Analyzer ────────────────────────────────────────────────────────

MERGE_CONFLICT_RE = re.compile(r"^<{7}\s", re.MULTILINE)
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10 MB


class WorkspaceAnalyzer:
    """Analyzes files for common issues."""

    def __init__(self, workspace_path: str = "."):
        self.workspace_path = workspace_path

    def analyze_file(self, path: str) -> List[FileIssue]:
        """Analyze a single file. *path* may be relative or absolute."""
        full = self._resolve(path)
        if not os.path.isfile(full):
            return []

        issues: List[FileIssue] = []
        ext = os.path.splitext(full)[1].lower()

        # Size check (all files)
        issue = self._check_file_size(full, path)
        if issue:
            issues.append(issue)

        # Merge conflict check (text files)
        if ext in (".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml", ".md", ".html", ".css"):
            issue = self._check_merge_conflicts(full, path)
            if issue:
                issues.append(issue)

        # Type-specific checks
        if ext == ".json":
            issue = self._check_json(full, path)
            if issue:
                issues.append(issue)
        elif ext in (".yaml", ".yml"):
            issue = self._check_yaml(full, path)
            if issue:
                issues.append(issue)
        elif ext == ".py":
            issue = self._check_python(full, path)
            if issue:
                issues.append(issue)

        return issues

    def analyze_batch(self, paths: List[str]) -> List[FileIssue]:
        """Analyze multiple files. Returns all issues found."""
        issues: List[FileIssue] = []
        for p in paths:
            issues.extend(self.analyze_file(p))
        return issues

    # ── checkers ────────────────────────────────────────────────

    def _check_json(self, full_path: str, rel_path: str) -> Optional[FileIssue]:
        try:
            with open(full_path) as f:
                json.load(f)
            return None
        except json.JSONDecodeError as e:
            return FileIssue(
                file_path=rel_path,
                issue_type="invalid_json",
                description=f"JSON parse error: {e.msg}",
                line_number=e.lineno,
                severity="error",
                auto_fixable=False,
            )
        except Exception:
            return None

    def _check_yaml(self, full_path: str, rel_path: str) -> Optional[FileIssue]:
        try:
            import yaml
            with open(full_path) as f:
                yaml.safe_load(f)
            return None
        except ImportError:
            return None  # yaml not available
        except yaml.YAMLError as e:
            line = None
            if hasattr(e, "problem_mark") and e.problem_mark:
                line = e.problem_mark.line + 1
            return FileIssue(
                file_path=rel_path,
                issue_type="broken_yaml",
                description=f"YAML parse error: {e}",
                line_number=line,
                severity="error",
                auto_fixable=False,
            )
        except Exception:
            return None

    def _check_python(self, full_path: str, rel_path: str) -> Optional[FileIssue]:
        try:
            with open(full_path) as f:
                source = f.read()
            ast.parse(source, filename=rel_path)
            return None
        except SyntaxError as e:
            return FileIssue(
                file_path=rel_path,
                issue_type="syntax_error",
                description=f"Python syntax error: {e.msg}",
                line_number=e.lineno,
                severity="error",
                auto_fixable=False,
            )
        except Exception:
            return None

    def _check_merge_conflicts(self, full_path: str, rel_path: str) -> Optional[FileIssue]:
        try:
            with open(full_path) as f:
                content = f.read(512_000)  # cap read
            if MERGE_CONFLICT_RE.search(content):
                # Find first conflict line
                for i, line in enumerate(content.split("\n"), 1):
                    if line.startswith("<<<<<<<"):
                        return FileIssue(
                            file_path=rel_path,
                            issue_type="merge_conflict",
                            description="Unresolved git merge conflict markers found",
                            line_number=i,
                            severity="error",
                            auto_fixable=False,
                        )
            return None
        except Exception:
            return None

    def _check_file_size(self, full_path: str, rel_path: str) -> Optional[FileIssue]:
        try:
            size = os.path.getsize(full_path)
            if size > LARGE_FILE_THRESHOLD:
                mb = size / (1024 * 1024)
                return FileIssue(
                    file_path=rel_path,
                    issue_type="large_file",
                    description=f"File is {mb:.1f} MB (threshold: 10 MB)",
                    severity="warning",
                    auto_fixable=False,
                )
            return None
        except OSError:
            return None

    # ── helpers ─────────────────────────────────────────────────

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_path, path)
