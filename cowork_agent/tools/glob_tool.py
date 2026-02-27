"""
Glob Tool — Fast file pattern matching.
Mirrors Cowork's Glob tool: supports glob patterns, returns files sorted by mtime.
"""

from __future__ import annotations
import os
from pathlib import Path

from .base import BaseTool


class GlobTool(BaseTool):
    name = "glob"
    description = (
        "Fast file pattern matching tool. "
        "Supports glob patterns like '**/*.py' or 'src/**/*.ts'. "
        "Returns matching file paths sorted by modification time (newest first)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The glob pattern to match files against",
            },
            "path": {
                "type": "string",
                "description": (
                    "Directory to search in. Defaults to current working directory."
                ),
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, default_dir: str = "."):
        resolved = os.path.abspath(default_dir)
        self._default_dir = resolved if os.path.isdir(resolved) else os.getcwd()

    async def execute(self, pattern: str, path: str = "",
                      tool_id: str = "", **kwargs) -> "ToolResult":
        search_dir = Path(path) if path else Path(self._default_dir)

        if not search_dir.exists():
            return self._error(f"Directory not found: {search_dir}", tool_id)
        if not search_dir.is_dir():
            return self._error(f"Not a directory: {search_dir}", tool_id)

        try:
            # pathlib.glob() handles ** patterns natively
            matches = list(search_dir.glob(pattern))

            # Filter to files only (skip directories)
            files = [m for m in matches if m.is_file()]

            # Sort by modification time (newest first)
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            if not files:
                return self._success(
                    f"No files matched pattern '{pattern}' in {search_dir}",
                    tool_id,
                )

            # Format output
            lines = [str(f) for f in files]
            output = "\n".join(lines)

            # Truncate if too many results
            if len(files) > 500:
                output = "\n".join(lines[:500])
                output += f"\n\n[Showing 500 of {len(files)} matches]"

            return self._success(output, tool_id)

        except Exception as e:
            return self._error(f"Glob error: {str(e)}", tool_id)
