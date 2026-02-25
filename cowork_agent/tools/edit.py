"""
Edit Tool — Exact string replacement in files.
Mirrors Cowork's Edit tool behavior: requires file to have been read first,
fails if old_string is not unique (unless replace_all=True).
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from .base import BaseTool

if TYPE_CHECKING:
    from ..core.tool_registry import ToolRegistry


class EditTool(BaseTool):
    name = "edit"
    description = (
        "Perform exact string replacements in files. "
        "The file must have been read (via the Read tool) before editing. "
        "The edit will FAIL if old_string is not unique in the file — "
        "provide more surrounding context or use replace_all. "
        "Use replace_all for renaming variables across a file."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to modify",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to find and replace",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement text (must differ from old_string)",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences (default: false)",
                "default": False,
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def __init__(self, registry: "ToolRegistry | None" = None):
        self._registry = registry

    def set_registry(self, registry: "ToolRegistry") -> None:
        """Allow late-binding of the registry reference."""
        self._registry = registry

    async def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        tool_id: str = "",
        **kwargs,
    ) -> "ToolResult":
        path = Path(file_path)

        # 1. File must exist
        if not path.exists():
            return self._error(f"File not found: {file_path}", tool_id)

        if path.is_dir():
            return self._error(f"Cannot edit a directory: {file_path}", tool_id)

        # 2. File must have been read first (safety guard)
        if self._registry and not self._registry.has_been_read(file_path):
            return self._error(
                f"File has not been read yet: {file_path}. "
                "You must use the Read tool before editing a file.",
                tool_id,
            )

        # 3. old_string != new_string
        if old_string == new_string:
            return self._error(
                "old_string and new_string are identical — no change to make.",
                tool_id,
            )

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return self._error(f"Error reading file: {str(e)}", tool_id)

        # 4. old_string must be present
        count = content.count(old_string)
        if count == 0:
            # Provide a helpful snippet for debugging
            snippet = content[:200] + "..." if len(content) > 200 else content
            return self._error(
                f"old_string not found in {file_path}. "
                f"File starts with:\n{snippet}",
                tool_id,
            )

        # 5. Uniqueness check (unless replace_all)
        if not replace_all and count > 1:
            return self._error(
                f"old_string appears {count} times in {file_path}. "
                "Provide more surrounding context to make it unique, "
                "or set replace_all=true to replace every occurrence.",
                tool_id,
            )

        # 6. Perform the replacement
        if replace_all:
            new_content = content.replace(old_string, new_string)
            replacements = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replacements = 1

        try:
            path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return self._error(f"Error writing file: {str(e)}", tool_id)

        return self._success(
            f"Replaced {replacements} occurrence(s) in {file_path}",
            tool_id,
        )
