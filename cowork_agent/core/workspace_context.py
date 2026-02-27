"""
Workspace Context — Git-aware workspace state tracking.

Maintains a snapshot of the current workspace state:
  - Current branch and git status
  - Active worktree (if any)
  - Dirty (uncommitted) files
  - Quick refresh for status bar / prompts

Sprint 18 (Worktree & Git Integration) Feature 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .git_ops import GitOperations
from .worktree import WorktreeInfo

logger = logging.getLogger(__name__)


class WorkspaceContext:
    """
    Git-aware workspace state tracker.

    Combines git status, branch info, and active worktree tracking
    into a single queryable object. Call refresh() to update state.

    Usage:
        ctx = WorkspaceContext("/path/to/repo")
        ctx.refresh()
        print(ctx.current_branch, ctx.is_dirty)
    """

    def __init__(self, workspace_dir: str = "", git_ops: Optional[GitOperations] = None):
        self.workspace_dir = workspace_dir
        self._git_ops = git_ops or GitOperations(workspace_dir=workspace_dir)
        self._current_branch: str = ""
        self._dirty_files: list[str] = []
        self._active_worktree: Optional[WorktreeInfo] = None
        self._is_git_repo: bool = False

    def refresh(self) -> None:
        """Refresh workspace state from git."""
        try:
            self._is_git_repo = self._git_ops.is_git_repo()
            if not self._is_git_repo:
                self._current_branch = ""
                self._dirty_files = []
                return

            self._current_branch = self._git_ops.current_branch()

            status = self._git_ops.status()
            # Combine all modified files (staged + unstaged + untracked)
            self._dirty_files = list(set(
                status.staged + status.unstaged + status.untracked
            ))

        except Exception as e:
            logger.warning(f"Failed to refresh workspace context: {e}")

    @property
    def current_branch(self) -> str:
        """Current branch name (empty if not a git repo or not refreshed)."""
        return self._current_branch

    @property
    def active_worktree(self) -> Optional[WorktreeInfo]:
        """Currently active worktree, if any."""
        return self._active_worktree

    @property
    def dirty_files(self) -> list[str]:
        """List of files with uncommitted changes."""
        return self._dirty_files

    @property
    def is_dirty(self) -> bool:
        """True if there are any uncommitted changes."""
        return len(self._dirty_files) > 0

    @property
    def is_git_repo(self) -> bool:
        """True if workspace is inside a git repository."""
        return self._is_git_repo

    def set_active_worktree(self, info: WorktreeInfo) -> None:
        """Set the currently active worktree."""
        self._active_worktree = info
        logger.info(f"Active worktree set: {info.name} ({info.branch})")

    def clear_active_worktree(self) -> None:
        """Clear the active worktree (back to main workspace)."""
        self._active_worktree = None
        logger.info("Active worktree cleared")

    def to_dict(self) -> dict:
        """Serialize workspace context to a dictionary."""
        result = {
            "is_git_repo": self._is_git_repo,
            "current_branch": self._current_branch,
            "is_dirty": self.is_dirty,
            "dirty_files": self._dirty_files,
            "dirty_file_count": len(self._dirty_files),
        }
        if self._active_worktree:
            result["active_worktree"] = {
                "name": self._active_worktree.name,
                "path": self._active_worktree.path,
                "branch": self._active_worktree.branch,
            }
        else:
            result["active_worktree"] = None
        return result
