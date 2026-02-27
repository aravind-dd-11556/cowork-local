"""
EnterWorktree Tool — Create and enter isolated git worktrees.

Mirrors real Cowork's EnterWorktree:
  - Creates a new git worktree in .claude/worktrees/
  - Optionally names the worktree
  - Returns worktree path and branch info
  - Delegates to WorktreeManager backend
"""

from __future__ import annotations
import logging
from typing import Optional

from .base import BaseTool
from ..core.worktree import WorktreeManager

logger = logging.getLogger(__name__)


class EnterWorktreeTool(BaseTool):
    name = "enter_worktree"
    description = (
        "Create and enter an isolated git worktree for the current task. "
        "The worktree provides an isolated copy of the repository so changes "
        "don't affect the main branch. Must be in a git repository."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Optional name for the worktree. A random name is generated if not provided.",
            },
        },
        "required": [],
    }

    def __init__(self, workspace_dir: str = "", bash_tool=None):
        self._manager = WorktreeManager(workspace_dir=workspace_dir)
        self._active_worktree: Optional[str] = None
        self._bash_tool = bash_tool  # Reference to BashTool for cwd switching

    @property
    def active_worktree(self) -> Optional[str]:
        """Return the name of the currently active worktree, if any."""
        return self._active_worktree

    def set_bash_tool(self, bash_tool) -> None:
        """Late-bind the Bash tool reference (avoids circular init)."""
        self._bash_tool = bash_tool

    async def execute(self, name: str = "", tool_id: str = "", **kwargs) -> "ToolResult":
        # Check if already in a worktree
        if self._active_worktree:
            return self._error(
                f"Already in worktree '{self._active_worktree}'. "
                "Cannot create a nested worktree.",
                tool_id,
            )

        # Check if git repo
        if not self._manager.is_git_repo():
            return self._error(
                "Not in a git repository. Cannot create a worktree.",
                tool_id,
            )

        # Create the worktree
        info = self._manager.create(name=name)
        if info is None:
            return self._error("Failed to create worktree. Check git status.", tool_id)

        self._active_worktree = info.name

        # Actually switch the Bash tool's cwd to the worktree path
        if self._bash_tool and hasattr(self._bash_tool, '_cwd'):
            import os
            if os.path.isdir(info.path):
                self._bash_tool._cwd = info.path
                logger.info(f"Switched Bash cwd to worktree: {info.path}")

        return self._success(
            f"Created worktree '{info.name}' at {info.path}\n"
            f"Branch: {info.branch}\n"
            f"Created from: {info.created_from}\n"
            f"Session working directory switched to worktree.",
            tool_id,
        )


class ListWorktreesTool(BaseTool):
    """List all active worktrees."""
    name = "list_worktrees"
    description = "List all git worktrees managed by the agent."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, workspace_dir: str = ""):
        self._manager = WorktreeManager(workspace_dir=workspace_dir)

    async def execute(self, tool_id: str = "", **kwargs) -> "ToolResult":
        worktrees = self._manager.list_worktrees()

        if not worktrees:
            return self._success("No active worktrees.", tool_id)

        lines = []
        for wt in worktrees:
            status = " (has changes)" if wt.has_changes else ""
            lines.append(f"  {wt.name}: {wt.path} [{wt.branch}]{status}")

        return self._success(
            f"Active worktrees ({len(worktrees)}):\n" + "\n".join(lines),
            tool_id,
        )


class RemoveWorktreeTool(BaseTool):
    """Remove a worktree."""
    name = "remove_worktree"
    description = (
        "Remove a git worktree. Will refuse if there are uncommitted changes "
        "unless force is set to true."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the worktree to remove",
            },
            "force": {
                "type": "boolean",
                "description": "Force removal even with uncommitted changes (default: false)",
                "default": False,
            },
        },
        "required": ["name"],
    }

    def __init__(self, workspace_dir: str = ""):
        self._manager = WorktreeManager(workspace_dir=workspace_dir)

    async def execute(self, name: str = "", force: bool = False,
                      tool_id: str = "", **kwargs) -> "ToolResult":
        if not name:
            return self._error("Worktree name is required.", tool_id)

        result = self._manager.remove(name, force=force)
        if "not found" in result.lower() or "failed" in result.lower():
            return self._error(result, tool_id)

        return self._success(result, tool_id)
