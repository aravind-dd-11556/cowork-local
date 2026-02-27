"""
Worktree Manager — Isolated git worktrees for parallel tasks.

Mirrors real Cowork's worktree support:
  - Creates a new git worktree in .claude/worktrees/
  - Each worktree gets its own branch
  - Worktree is cleaned up when done (or kept if changes were made)
  - Agent can work in isolation without affecting main branch
"""

from __future__ import annotations
import logging
import os
import random
import string
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WorktreeInfo:
    """Information about a git worktree."""
    name: str
    path: str
    branch: str
    created_from: str  # Branch/commit it was created from
    has_changes: bool = False


class WorktreeManager:
    """
    Manages git worktrees for isolated task execution.

    Usage:
        wt = WorktreeManager(workspace_dir="/path/to/repo")

        # Create a new worktree
        info = wt.create("my-feature")

        # Work in the worktree...
        # info.path is the worktree directory

        # Clean up when done
        wt.remove(info.name)
    """

    WORKTREE_DIR = ".claude/worktrees"

    def __init__(self, workspace_dir: str = ""):
        self.workspace_dir = workspace_dir

    def is_git_repo(self) -> bool:
        """Check if workspace is a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.workspace_dir,
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def create(self, name: str = "") -> Optional[WorktreeInfo]:
        """
        Create a new git worktree.

        Args:
            name: Optional name. If not provided, a random name is generated.

        Returns:
            WorktreeInfo with the worktree details, or None on failure.
        """
        if not self.is_git_repo():
            logger.warning("Not a git repository — cannot create worktree")
            return None

        if not name:
            name = self._random_name()

        # Sanitize name
        name = name.replace(" ", "-").replace("/", "-").lower()

        # Create worktree directory
        worktree_base = os.path.join(self.workspace_dir, self.WORKTREE_DIR)
        os.makedirs(worktree_base, exist_ok=True)

        worktree_path = os.path.join(worktree_base, name)
        branch_name = f"worktree/{name}"

        if os.path.exists(worktree_path):
            logger.warning(f"Worktree already exists: {worktree_path}")
            return WorktreeInfo(
                name=name, path=worktree_path,
                branch=branch_name, created_from="unknown",
            )

        # Get current branch/HEAD
        try:
            head_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.workspace_dir,
                capture_output=True, text=True, timeout=10,
            )
            created_from = head_result.stdout.strip() or "HEAD"
        except Exception:
            created_from = "HEAD"

        # Create the worktree with a new branch
        try:
            result = subprocess.run(
                ["git", "worktree", "add", "-b", branch_name, worktree_path],
                cwd=self.workspace_dir,
                capture_output=True, text=True, timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"Failed to create worktree: {result.stderr}")
                return None

            logger.info(f"Created worktree: {name} at {worktree_path} (branch: {branch_name})")

            return WorktreeInfo(
                name=name,
                path=worktree_path,
                branch=branch_name,
                created_from=created_from,
            )

        except Exception as e:
            logger.error(f"Worktree creation error: {e}")
            return None

    def remove(self, name: str, force: bool = False) -> str:
        """
        Remove a worktree.

        Args:
            name: Worktree name
            force: Force removal even if there are changes

        Returns:
            Status message
        """
        worktree_path = os.path.join(self.workspace_dir, self.WORKTREE_DIR, name)

        if not os.path.exists(worktree_path):
            return f"Worktree '{name}' not found."

        # Check for uncommitted changes
        if not force:
            has_changes = self._has_changes(worktree_path)
            if has_changes:
                return (
                    f"Worktree '{name}' has uncommitted changes. "
                    f"Use force=True to remove anyway, or commit your changes first."
                )

        try:
            cmd = ["git", "worktree", "remove", worktree_path]
            if force:
                cmd.append("--force")

            result = subprocess.run(
                cmd, cwd=self.workspace_dir,
                capture_output=True, text=True, timeout=30,
            )

            if result.returncode != 0:
                return f"Failed to remove worktree: {result.stderr.strip()}"

            # Also delete the branch
            branch_name = f"worktree/{name}"
            subprocess.run(
                ["git", "branch", "-d" if not force else "-D", branch_name],
                cwd=self.workspace_dir,
                capture_output=True, text=True, timeout=10,
            )

            logger.info(f"Removed worktree: {name}")
            return f"Worktree '{name}' removed."

        except Exception as e:
            return f"Error removing worktree: {str(e)}"

    def list_worktrees(self) -> list[WorktreeInfo]:
        """List all worktrees managed by us."""
        worktree_base = os.path.join(self.workspace_dir, self.WORKTREE_DIR)
        if not os.path.isdir(worktree_base):
            return []

        worktrees = []
        for entry in sorted(os.listdir(worktree_base)):
            path = os.path.join(worktree_base, entry)
            if os.path.isdir(path):
                worktrees.append(WorktreeInfo(
                    name=entry,
                    path=path,
                    branch=f"worktree/{entry}",
                    created_from="unknown",
                    has_changes=self._has_changes(path),
                ))

        return worktrees

    @staticmethod
    def _has_changes(worktree_path: str) -> bool:
        """Check if a worktree has uncommitted changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=worktree_path,
                capture_output=True, text=True, timeout=10,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    @staticmethod
    def _random_name() -> str:
        """Generate a random worktree name."""
        adjectives = ["swift", "calm", "bold", "keen", "fair", "warm", "pure", "wise"]
        nouns = ["hawk", "wolf", "fox", "bear", "deer", "lynx", "crow", "wren"]
        adj = random.choice(adjectives)
        noun = random.choice(nouns)
        suffix = "".join(random.choices(string.ascii_lowercase, k=4))
        return f"{adj}-{noun}-{suffix}"
