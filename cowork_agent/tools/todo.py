"""
TodoWrite Tool — Task tracking with pending/in_progress/completed states.
Mirrors Cowork's TodoWrite tool behavior.
Supports optional persistence to disk (.cowork/todos.json).
"""

from __future__ import annotations
import json
import logging
import os
from typing import Optional

from .base import BaseTool

logger = logging.getLogger(__name__)


class TodoWriteTool(BaseTool):
    name = "todo_write"
    description = (
        "Create and manage a structured task list for the current session. "
        "Track progress with states: pending, in_progress, completed. "
        "Each todo needs content (imperative: 'Run tests') and "
        "activeForm (continuous: 'Running tests')."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The complete updated todo list",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "What needs to be done (imperative form)",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Current status of the task",
                        },
                        "activeForm": {
                            "type": "string",
                            "description": "Present continuous form (e.g. 'Running tests')",
                        },
                    },
                    "required": ["content", "status", "activeForm"],
                },
            },
        },
        "required": ["todos"],
    }

    def __init__(self, persist_dir: str = ""):
        """
        Args:
            persist_dir: Directory to save todos.json. If empty, no persistence.
                         Typically set to workspace_dir/.cowork/
        """
        self._todos: list[dict] = []
        self._persist_path = ""

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._persist_path = os.path.join(persist_dir, "todos.json")
            self._load_from_disk()

    @property
    def todos(self) -> list[dict]:
        """Return current todo list (read-only copy)."""
        return [t.copy() for t in self._todos]

    async def execute(self, todos: list[dict],
                      tool_id: str = "", **kwargs) -> "ToolResult":
        # Validate each todo
        for i, todo in enumerate(todos):
            if "content" not in todo or "status" not in todo:
                return self._error(
                    f"Todo item {i} missing required 'content' or 'status' field.",
                    tool_id,
                )
            if todo["status"] not in ("pending", "in_progress", "completed"):
                return self._error(
                    f"Todo item {i} has invalid status '{todo['status']}'. "
                    "Must be: pending, in_progress, or completed.",
                    tool_id,
                )

        # Update the todo list
        self._todos = [
            {
                "content": t["content"],
                "status": t["status"],
                "activeForm": t.get("activeForm", t["content"]),
            }
            for t in todos
        ]

        # Persist to disk
        self._save_to_disk()

        # Build summary
        return self._success(self._format_summary(), tool_id)

    def _save_to_disk(self) -> None:
        """Save current todos to disk (if persistence is enabled)."""
        if not self._persist_path:
            return
        try:
            with open(self._persist_path, "w") as f:
                json.dump(self._todos, f, indent=2)
            logger.debug(f"Saved {len(self._todos)} todos to {self._persist_path}")
        except Exception as e:
            logger.warning(f"Failed to save todos to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load todos from disk (if persistence file exists)."""
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._todos = data
                logger.info(f"Loaded {len(self._todos)} todos from {self._persist_path}")
        except Exception as e:
            logger.warning(f"Failed to load todos from disk: {e}")

    def _format_summary(self) -> str:
        """Format the todo list for display."""
        if not self._todos:
            return "Todo list is empty."

        status_icons = {
            "pending": "⬜",
            "in_progress": "🔄",
            "completed": "✅",
        }

        lines = ["## Task Progress\n"]
        for todo in self._todos:
            icon = status_icons.get(todo["status"], "•")
            label = todo["activeForm"] if todo["status"] == "in_progress" else todo["content"]
            lines.append(f"{icon} {label}")

        # Stats
        pending = sum(1 for t in self._todos if t["status"] == "pending")
        active = sum(1 for t in self._todos if t["status"] == "in_progress")
        done = sum(1 for t in self._todos if t["status"] == "completed")
        total = len(self._todos)

        lines.append(f"\n**Progress: {done}/{total}** ({active} active, {pending} pending)")

        return "\n".join(lines)

    def get_context(self) -> list[dict]:
        """Return todos in a format suitable for the prompt builder."""
        return self.todos
