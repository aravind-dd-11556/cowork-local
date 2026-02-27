"""
Git Tools — User-facing git operation tools.

Five tools that expose git operations through the standard BaseTool interface:
  1. GitStatusTool — Show staged/unstaged/untracked files
  2. GitDiffTool — Show diffs (staged or unstaged)
  3. GitCommitTool — Commit changes
  4. GitBranchTool — List/create/delete/switch branches
  5. GitLogTool — View commit history

Sprint 18 (Worktree & Git Integration) Feature 4.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import BaseTool
from ..core.git_ops import GitOperations

logger = logging.getLogger(__name__)


class GitStatusTool(BaseTool):
    """Show git status: staged, unstaged, and untracked files."""

    name = "git_status"
    description = (
        "Show the current git status including staged, unstaged, and untracked files. "
        "Returns a summary of the working tree state."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, git_ops: Optional[GitOperations] = None, workspace_dir: str = ""):
        self._git = git_ops or GitOperations(workspace_dir=workspace_dir)

    async def execute(self, tool_id: str = "", **kwargs) -> "ToolResult":
        if not self._git.is_git_repo():
            return self._error("Not a git repository", tool_id)

        status = self._git.status()

        if status.is_clean:
            return self._success("Working tree clean — nothing to commit.", tool_id)

        lines = []
        if status.staged:
            lines.append(f"Staged ({len(status.staged)}):")
            for f in status.staged:
                lines.append(f"  + {f}")
        if status.unstaged:
            lines.append(f"Unstaged ({len(status.unstaged)}):")
            for f in status.unstaged:
                lines.append(f"  M {f}")
        if status.untracked:
            lines.append(f"Untracked ({len(status.untracked)}):")
            for f in status.untracked:
                lines.append(f"  ? {f}")

        lines.append(f"\nTotal: {status.total_changes} change(s)")
        branch = self._git.current_branch()
        if branch:
            lines.insert(0, f"On branch {branch}\n")

        return self._success("\n".join(lines), tool_id)


class GitDiffTool(BaseTool):
    """Show git diffs — staged or unstaged changes."""

    name = "git_diff"
    description = (
        "Show git diff output. By default shows unstaged changes. "
        "Set staged=true for staged diff. Optionally specify a file path."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "Optional file path to diff (all files if omitted)",
            },
            "staged": {
                "type": "boolean",
                "description": "Show staged (--cached) diff instead of unstaged",
            },
        },
        "required": [],
    }

    def __init__(self, git_ops: Optional[GitOperations] = None, workspace_dir: str = ""):
        self._git = git_ops or GitOperations(workspace_dir=workspace_dir)

    async def execute(self, file: str = "", staged: bool = False,
                      tool_id: str = "", **kwargs) -> "ToolResult":
        if not self._git.is_git_repo():
            return self._error("Not a git repository", tool_id)

        diff_output = self._git.diff(file=file, staged=staged)

        if not diff_output.strip():
            label = "staged" if staged else "unstaged"
            target = f" for {file}" if file else ""
            return self._success(f"No {label} changes{target}.", tool_id)

        # Truncate very long diffs
        max_chars = 30000
        if len(diff_output) > max_chars:
            diff_output = diff_output[:max_chars] + f"\n\n... (truncated, {len(diff_output)} total chars)"

        return self._success(diff_output, tool_id)


class GitCommitTool(BaseTool):
    """Commit changes to the repository."""

    name = "git_commit"
    description = (
        "Commit changes. Provide a commit message. Optionally specify files to "
        "stage and commit; otherwise commits whatever is currently staged."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Commit message (required)",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of files to stage and commit",
            },
        },
        "required": ["message"],
    }

    def __init__(self, git_ops: Optional[GitOperations] = None, workspace_dir: str = ""):
        self._git = git_ops or GitOperations(workspace_dir=workspace_dir)

    async def execute(self, message: str = "", files: list[str] | None = None,
                      tool_id: str = "", **kwargs) -> "ToolResult":
        if not self._git.is_git_repo():
            return self._error("Not a git repository", tool_id)

        if not message:
            return self._error("Commit message is required", tool_id)

        result = self._git.commit(message=message, files=files)

        if result.startswith("Error"):
            return self._error(result, tool_id)

        return self._success(result, tool_id)


class GitBranchTool(BaseTool):
    """List, create, delete, or switch branches."""

    name = "git_branch"
    description = (
        "Manage git branches. Actions: 'list' (default), 'create', 'delete', 'switch'. "
        "Provide a branch name for create/delete/switch."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "create", "delete", "switch"],
                "description": "Branch action (default: list)",
            },
            "name": {
                "type": "string",
                "description": "Branch name (required for create/delete/switch)",
            },
            "force": {
                "type": "boolean",
                "description": "Force delete (use -D instead of -d)",
            },
            "from_ref": {
                "type": "string",
                "description": "Create branch from this ref (for create action)",
            },
        },
        "required": [],
    }

    # Branches that cannot be force-deleted via this tool
    PROTECTED_BRANCHES = {"main", "master"}

    def __init__(self, git_ops: Optional[GitOperations] = None, workspace_dir: str = "",
                 protected_branches: set[str] | None = None):
        self._git = git_ops or GitOperations(workspace_dir=workspace_dir)
        if protected_branches is not None:
            self.PROTECTED_BRANCHES = protected_branches

    async def execute(self, action: str = "list", name: str = "",
                      force: bool = False, from_ref: str = "",
                      tool_id: str = "", **kwargs) -> "ToolResult":
        if not self._git.is_git_repo():
            return self._error("Not a git repository", tool_id)

        if action == "list":
            return self._list_branches(tool_id)
        elif action == "create":
            return self._create_branch(name, from_ref, tool_id)
        elif action == "delete":
            return self._delete_branch(name, force, tool_id)
        elif action == "switch":
            return self._switch_branch(name, tool_id)
        else:
            return self._error(f"Unknown action: {action}. Use list/create/delete/switch.", tool_id)

    def _list_branches(self, tool_id: str) -> "ToolResult":
        branches = self._git.branch_list()
        if not branches:
            return self._success("No branches found.", tool_id)

        lines = []
        for b in branches:
            marker = "* " if b.is_current else "  "
            upstream = f" -> {b.upstream}" if b.upstream else ""
            lines.append(f"{marker}{b.name}{upstream}")

        return self._success("\n".join(lines), tool_id)

    def _create_branch(self, name: str, from_ref: str, tool_id: str) -> "ToolResult":
        if not name:
            return self._error("Branch name is required for create", tool_id)
        result = self._git.branch_create(name, from_ref=from_ref)
        if result.startswith("Error"):
            return self._error(result, tool_id)
        return self._success(result, tool_id)

    def _delete_branch(self, name: str, force: bool, tool_id: str) -> "ToolResult":
        if not name:
            return self._error("Branch name is required for delete", tool_id)

        # Protect main/master from force-deletion
        if name in self.PROTECTED_BRANCHES and force:
            return self._error(
                f"Cannot force-delete protected branch '{name}'. "
                f"Protected branches: {', '.join(sorted(self.PROTECTED_BRANCHES))}",
                tool_id,
            )

        result = self._git.branch_delete(name, force=force)
        if result.startswith("Error"):
            return self._error(result, tool_id)
        return self._success(result, tool_id)

    def _switch_branch(self, name: str, tool_id: str) -> "ToolResult":
        if not name:
            return self._error("Branch name is required for switch", tool_id)
        result = self._git.checkout(name)
        if result.startswith("Error"):
            return self._error(result, tool_id)
        return self._success(result, tool_id)


class GitLogTool(BaseTool):
    """View commit history."""

    name = "git_log"
    description = (
        "Show recent commit history. Defaults to 10 entries. "
        "Optionally specify a branch to view its log."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "n": {
                "type": "integer",
                "description": "Number of commits to show (default: 10, max: 50)",
            },
            "branch": {
                "type": "string",
                "description": "Branch to view log for (defaults to current branch)",
            },
        },
        "required": [],
    }

    def __init__(self, git_ops: Optional[GitOperations] = None, workspace_dir: str = "",
                 max_entries: int = 50):
        self._git = git_ops or GitOperations(workspace_dir=workspace_dir)
        self._max_entries = max_entries

    async def execute(self, n: int = 10, branch: str = "",
                      tool_id: str = "", **kwargs) -> "ToolResult":
        if not self._git.is_git_repo():
            return self._error("Not a git repository", tool_id)

        n = min(max(1, n), self._max_entries)
        commits = self._git.log(n=n, branch=branch)

        if not commits:
            label = f" on {branch}" if branch else ""
            return self._success(f"No commits found{label}.", tool_id)

        lines = []
        for c in commits:
            lines.append(f"{c.hash}  {c.author}  {c.timestamp}")
            lines.append(f"  {c.message}")
            lines.append("")

        header = f"Showing {len(commits)} commit(s)"
        if branch:
            header += f" on {branch}"
        lines.insert(0, header + "\n")

        return self._success("\n".join(lines), tool_id)
