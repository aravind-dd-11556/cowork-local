"""
Sprint 42 · Git Monitor
========================
Tracks git repository state changes: branch switches, new commits,
unstaged changes, merge conflicts, and stash operations.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Data ─────────────────────────────────────────────────────────────

@dataclass
class GitChange:
    """A detected change in the git repository state."""
    change_type: str  # new_commit | branch_switch | unstaged_changes | merge_conflict | stash_change
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "change_type": self.change_type,
            "description": self.description,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class GitState:
    """Snapshot of the current git repository state."""
    branch: str = ""
    head_commit: str = ""
    unstaged_count: int = 0
    staged_count: int = 0
    stash_count: int = 0
    has_merge_conflicts: bool = False
    is_git_repo: bool = False

    def to_dict(self) -> dict:
        return {
            "branch": self.branch,
            "head_commit": self.head_commit,
            "unstaged_count": self.unstaged_count,
            "staged_count": self.staged_count,
            "stash_count": self.stash_count,
            "has_merge_conflicts": self.has_merge_conflicts,
            "is_git_repo": self.is_git_repo,
        }


# ── Monitor ──────────────────────────────────────────────────────────

class GitMonitor:
    """
    Monitors a git repository for state changes.

    Compares current git state against the last known state and
    reports any differences as GitChange objects.
    """

    MERGE_CONFLICT_RE = re.compile(r"^(UU|AA|DD)\s", re.MULTILINE)

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self._last_state: Optional[GitState] = None
        self._change_history: List[GitChange] = []

    # ── public API ────────────────────────────────────────────────

    async def check_changes(self) -> List[GitChange]:
        """
        Compare current git state vs last known state.
        Returns list of detected changes.
        """
        current = self._get_current_state()

        if not current.is_git_repo:
            self._last_state = current
            return []

        if self._last_state is None:
            # First check — just record state, no changes to report
            self._last_state = current
            return []

        changes: List[GitChange] = []

        # Branch switch
        if current.branch != self._last_state.branch:
            changes.append(GitChange(
                change_type="branch_switch",
                description=f"Branch switched from '{self._last_state.branch}' to '{current.branch}'",
                details={
                    "old_branch": self._last_state.branch,
                    "new_branch": current.branch,
                },
            ))

        # New commit(s)
        if (current.head_commit != self._last_state.head_commit
                and current.head_commit and self._last_state.head_commit):
            changes.append(GitChange(
                change_type="new_commit",
                description=f"New commit detected: {current.head_commit[:8]}",
                details={
                    "old_commit": self._last_state.head_commit,
                    "new_commit": current.head_commit,
                },
            ))

        # Unstaged changes increase
        if current.unstaged_count > self._last_state.unstaged_count:
            diff = current.unstaged_count - self._last_state.unstaged_count
            changes.append(GitChange(
                change_type="unstaged_changes",
                description=f"{diff} new unstaged change(s) detected ({current.unstaged_count} total)",
                details={
                    "old_count": self._last_state.unstaged_count,
                    "new_count": current.unstaged_count,
                },
            ))

        # Merge conflict appeared
        if current.has_merge_conflicts and not self._last_state.has_merge_conflicts:
            changes.append(GitChange(
                change_type="merge_conflict",
                description="Merge conflict(s) detected in the repository",
                details={"has_conflicts": True},
            ))

        # Stash change
        if current.stash_count != self._last_state.stash_count:
            direction = "added" if current.stash_count > self._last_state.stash_count else "removed"
            changes.append(GitChange(
                change_type="stash_change",
                description=f"Stash {direction} (now {current.stash_count} stash entries)",
                details={
                    "old_count": self._last_state.stash_count,
                    "new_count": current.stash_count,
                },
            ))

        self._last_state = current
        self._change_history.extend(changes)
        return changes

    def get_current_state(self) -> GitState:
        """Return current git state (public wrapper)."""
        return self._get_current_state()

    def reset(self) -> None:
        """Reset last known state so next check reports no diff."""
        self._last_state = None

    @property
    def change_history(self) -> List[GitChange]:
        return list(self._change_history)

    @property
    def last_state(self) -> Optional[GitState]:
        return self._last_state

    # ── internal helpers ──────────────────────────────────────────

    def _get_current_state(self) -> GitState:
        """Build a GitState from actual git commands."""
        state = GitState()

        if not self._is_git_repo():
            return state

        state.is_git_repo = True
        state.branch = self._get_current_branch()
        state.head_commit = self._get_head_commit()
        state.unstaged_count = self._get_unstaged_count()
        state.staged_count = self._get_staged_count()
        state.stash_count = self._get_stash_count()
        state.has_merge_conflicts = self._has_merge_conflicts()
        return state

    def _run_git(self, *args: str) -> Optional[str]:
        """Run a git command and return stdout, or None on failure."""
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=self.workspace_path,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None

    def _is_git_repo(self) -> bool:
        return self._run_git("rev-parse", "--is-inside-work-tree") == "true"

    def _get_current_branch(self) -> str:
        result = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        return result or ""

    def _get_head_commit(self) -> str:
        result = self._run_git("rev-parse", "HEAD")
        return result or ""

    def _get_unstaged_count(self) -> int:
        result = self._run_git("diff", "--name-only")
        if result is None:
            return 0
        return len([l for l in result.splitlines() if l.strip()])

    def _get_staged_count(self) -> int:
        result = self._run_git("diff", "--cached", "--name-only")
        if result is None:
            return 0
        return len([l for l in result.splitlines() if l.strip()])

    def _get_stash_count(self) -> int:
        result = self._run_git("stash", "list")
        if result is None:
            return 0
        return len([l for l in result.splitlines() if l.strip()])

    def _has_merge_conflicts(self) -> bool:
        result = self._run_git("status", "--porcelain")
        if result is None:
            return False
        return bool(self.MERGE_CONFLICT_RE.search(result))
