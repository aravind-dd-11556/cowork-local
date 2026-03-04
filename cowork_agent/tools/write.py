"""
Write Tool — Write/create files with automatic parent directory creation.
Mirrors Cowork's Write tool behavior.
"""

from __future__ import annotations
import os
from pathlib import Path

from .base import BaseTool


class WriteTool(BaseTool):
    name = "write"
    description = (
        "Write content to a file. Creates parent directories if needed. "
        "The file_path must be an absolute path. "
        "WARNING: This will overwrite the existing file if one exists at the path."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file",
            },
        },
        "required": ["file_path", "content"],
    }

    # SEC-HIGH-3: Limit directory creation depth to prevent deep tree attacks
    MAX_DIR_DEPTH = 15

    def __init__(self, workspace_dir: str = ""):
        self._workspace_dir = workspace_dir

    async def execute(self, file_path: str, content: str,
                      tool_id: str = "", **kwargs) -> "ToolResult":
        path = Path(file_path)

        # Safety: reject relative paths
        if not path.is_absolute():
            return self._error(
                f"file_path must be absolute, got: {file_path}", tool_id
            )

        # H-11: Resolve symlinks to prevent symlink-based path traversal
        try:
            resolved_path = path.resolve()
        except (OSError, RuntimeError) as e:
            return self._error(
                f"Failed to resolve path: {str(e)}", tool_id
            )

        # H-11: Check for .. in resolved path parts (defense in depth)
        path_str = str(resolved_path)
        if ".." in path_str or path_str != resolved_path.as_posix():
            return self._error(
                f"Path traversal detected in resolved path: {file_path}", tool_id
            )

        # SEC-HIGH-3: Reject excessively deep paths (check resolved path)
        if len(resolved_path.parts) > self.MAX_DIR_DEPTH:
            return self._error(
                f"Path too deep ({len(resolved_path.parts)} levels). "
                f"Maximum allowed depth: {self.MAX_DIR_DEPTH}",
                tool_id,
            )

        # H-11: Verify resolved path is still within allowed directories
        # if workspace_dir is specified
        if self._workspace_dir:
            workspace_resolved = Path(self._workspace_dir).resolve()
            try:
                resolved_path.relative_to(workspace_resolved)
            except ValueError:
                return self._error(
                    f"Path escapes workspace directory: {file_path}", tool_id
                )

        path = resolved_path

        try:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            size = path.stat().st_size
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

            return self._success(
                f"Successfully wrote {size} bytes ({line_count} lines) to {file_path}",
                tool_id,
            )

        except PermissionError:
            return self._error(f"Permission denied: {file_path}", tool_id)
        except Exception as e:
            return self._error(f"Error writing file: {str(e)}", tool_id)
