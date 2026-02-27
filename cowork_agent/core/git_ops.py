"""
Git Operations — Subprocess-based git command execution with parsed results.

Provides a structured interface over raw git commands:
  - Status parsing (staged/unstaged/untracked)
  - Branch management (list/create/delete/switch)
  - Commit, diff, log, stash, merge
  - Merge conflict detection

All methods are synchronous (subprocess.run), called from async tool layer.

Sprint 18 (Worktree & Git Integration) Feature 1.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Default timeout for git subprocess calls (seconds)
GIT_TIMEOUT = 30


# ── Dataclasses ───────────────────────────────────────────────────

@dataclass
class GitStatusResult:
    """Parsed output of git status --porcelain."""
    staged: list[str] = field(default_factory=list)
    unstaged: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.staged and not self.unstaged and not self.untracked

    @property
    def total_changes(self) -> int:
        return len(self.staged) + len(self.unstaged) + len(self.untracked)


@dataclass
class GitCommit:
    """A single git commit entry."""
    hash: str
    author: str
    message: str
    timestamp: str = ""  # ISO-like string from git

    def to_dict(self) -> dict:
        return {
            "hash": self.hash,
            "author": self.author,
            "message": self.message,
            "timestamp": self.timestamp,
        }


@dataclass
class GitBranch:
    """A git branch entry."""
    name: str
    is_current: bool = False
    upstream: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "is_current": self.is_current,
            "upstream": self.upstream,
        }


@dataclass
class MergeResult:
    """Result of a git merge operation."""
    success: bool
    conflicts: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0


# ── GitOperations ─────────────────────────────────────────────────

class GitOperations:
    """
    Structured interface over git subprocess calls.

    Usage:
        git = GitOperations(workspace_dir="/path/to/repo")
        status = git.status()
        print(status.staged, status.unstaged)

        git.commit("fix: resolve bug", files=["src/main.py"])
        log = git.log(n=5)
    """

    def __init__(self, workspace_dir: str = ""):
        self.workspace_dir = workspace_dir or os.getcwd()

    # ── Internal helpers ──────────────────────────────────────────

    def _run(self, args: list[str], timeout: int = GIT_TIMEOUT,
             check: bool = False) -> subprocess.CompletedProcess:
        """Execute a git command and return the CompletedProcess."""
        cmd = ["git"] + args
        try:
            return subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Git command timed out after {timeout}s: {' '.join(cmd)}")
            raise
        except FileNotFoundError:
            logger.error("git executable not found")
            raise

    # ── Utility ───────────────────────────────────────────────────

    def is_git_repo(self) -> bool:
        """Check if workspace is inside a git repository."""
        try:
            result = self._run(["rev-parse", "--git-dir"])
            return result.returncode == 0
        except Exception:
            return False

    def git_root(self) -> str:
        """Return the root directory of the git repository."""
        result = self._run(["rev-parse", "--show-toplevel"])
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    # ── Queries ───────────────────────────────────────────────────

    def status(self) -> GitStatusResult:
        """
        Parse git status --porcelain output.

        Porcelain format: XY PATH
          X = index (staged) status
          Y = work-tree (unstaged) status
          ? = untracked
        """
        result = self._run(["status", "--porcelain"])
        if result.returncode != 0:
            logger.warning(f"git status failed: {result.stderr.strip()}")
            return GitStatusResult()

        staged = []
        unstaged = []
        untracked = []

        for line in result.stdout.splitlines():
            if len(line) < 3:
                continue

            x_status = line[0]  # Index (staged) status
            y_status = line[1]  # Work-tree (unstaged) status
            filepath = line[3:]  # Skip "XY "

            if x_status == "?" and y_status == "?":
                untracked.append(filepath)
            else:
                if x_status not in (" ", "?"):
                    staged.append(filepath)
                if y_status not in (" ", "?"):
                    unstaged.append(filepath)

        return GitStatusResult(staged=staged, unstaged=unstaged, untracked=untracked)

    def diff(self, file: str = "", staged: bool = False) -> str:
        """
        Get diff output.

        Args:
            file: Optional specific file path to diff.
            staged: If True, show staged (--cached) diff.
        """
        args = ["diff"]
        if staged:
            args.append("--cached")
        if file:
            args.extend(["--", file])

        result = self._run(args, timeout=60)
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return result.stdout

    def log(self, n: int = 10, branch: str = "") -> list[GitCommit]:
        """
        Get commit log as structured entries.

        Uses --format to parse hash, author, date, message.
        """
        n = min(max(1, n), 200)  # Clamp 1..200
        args = [
            "log", f"-{n}",
            "--format=%H|%an|%aI|%s",  # hash|author|date|subject
        ]
        if branch:
            args.append(branch)

        result = self._run(args)
        if result.returncode != 0:
            logger.warning(f"git log failed: {result.stderr.strip()}")
            return []

        commits = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append(GitCommit(
                    hash=parts[0][:12],  # Short hash
                    author=parts[1],
                    timestamp=parts[2],
                    message=parts[3],
                ))
        return commits

    def current_branch(self) -> str:
        """Get the name of the current branch."""
        result = self._run(["rev-parse", "--abbrev-ref", "HEAD"])
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def branch_list(self) -> list[GitBranch]:
        """List all local branches with current-branch marker."""
        result = self._run(["branch", "--format=%(HEAD)|%(refname:short)|%(upstream:short)"])
        if result.returncode != 0:
            logger.warning(f"git branch failed: {result.stderr.strip()}")
            return []

        branches = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|", 2)
            if len(parts) >= 2:
                branches.append(GitBranch(
                    name=parts[1].strip(),
                    is_current=(parts[0].strip() == "*"),
                    upstream=parts[2].strip() if len(parts) > 2 else "",
                ))
        return branches

    # ── Mutations ─────────────────────────────────────────────────

    def branch_create(self, name: str, from_ref: str = "") -> str:
        """Create a new branch."""
        if not name:
            return "Error: branch name required"

        args = ["branch", name]
        if from_ref:
            args.append(from_ref)

        result = self._run(args)
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return f"Created branch '{name}'"

    def branch_delete(self, name: str, force: bool = False) -> str:
        """Delete a branch."""
        if not name:
            return "Error: branch name required"

        flag = "-D" if force else "-d"
        result = self._run(["branch", flag, name])
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return f"Deleted branch '{name}'"

    def checkout(self, branch: str) -> str:
        """Switch to an existing branch."""
        if not branch:
            return "Error: branch name required"

        result = self._run(["checkout", branch])
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return f"Switched to branch '{branch}'"

    def commit(self, message: str, files: Optional[list[str]] = None) -> str:
        """
        Commit changes.

        Args:
            message: Commit message (required).
            files: Optional list of specific files to stage and commit.
                   If None, commits whatever is currently staged.
        """
        if not message:
            return "Error: commit message required"

        # Stage specific files if provided
        if files:
            add_result = self._run(["add", "--"] + files)
            if add_result.returncode != 0:
                return f"Error staging files: {add_result.stderr.strip()}"

        result = self._run(["commit", "-m", message])
        if result.returncode != 0:
            combined = (result.stderr + result.stdout).strip()
            if "nothing to commit" in combined or "nothing added" in combined:
                return "Nothing to commit — no staged changes."
            return f"Error: {combined or 'commit failed'}"

        # Extract short info from output
        output = result.stdout.strip()
        first_line = output.splitlines()[0] if output else "Committed"
        return first_line

    def stash(self, pop: bool = False) -> str:
        """Stash or pop changes."""
        args = ["stash", "pop"] if pop else ["stash"]
        result = self._run(args)
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return result.stdout.strip() or ("Stash popped" if pop else "Changes stashed")

    def merge(self, branch: str, abort: bool = False) -> MergeResult:
        """
        Merge a branch into current branch.

        Args:
            branch: Branch to merge.
            abort: If True, abort an in-progress merge instead.
        """
        if abort:
            result = self._run(["merge", "--abort"])
            return MergeResult(
                success=result.returncode == 0,
                message="Merge aborted" if result.returncode == 0 else result.stderr.strip(),
            )

        if not branch:
            return MergeResult(success=False, message="Error: branch name required")

        result = self._run(["merge", branch])

        if result.returncode == 0:
            return MergeResult(
                success=True,
                message=result.stdout.strip() or f"Merged '{branch}' successfully",
            )

        # Check for merge conflicts
        conflicts = self._detect_conflicts()
        if conflicts:
            return MergeResult(
                success=False,
                conflicts=conflicts,
                message=f"Merge conflict in {len(conflicts)} file(s)",
            )

        return MergeResult(
            success=False,
            message=f"Merge failed: {result.stderr.strip()}",
        )

    def _detect_conflicts(self) -> list[str]:
        """Detect unmerged (conflicting) files after a failed merge."""
        result = self._run(["diff", "--name-only", "--diff-filter=U"])
        if result.returncode != 0:
            return []
        return [f for f in result.stdout.strip().splitlines() if f]
