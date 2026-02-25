"""
Read Tool — Read files with line numbers, offset/limit support.
Mirrors Cowork's Read tool behavior.
"""

from __future__ import annotations
import os
from pathlib import Path

from .base import BaseTool


class ReadTool(BaseTool):
    name = "read"
    description = (
        "Read a file from the filesystem. Returns content with line numbers. "
        "The file_path must be an absolute path. "
        "By default reads up to 2000 lines. Use offset and limit for large files. "
        "Lines longer than 2000 characters are truncated."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to read",
            },
            "offset": {
                "type": "number",
                "description": "Line number to start reading from (1-based)",
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of lines to read",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, max_lines: int = 2000, max_line_length: int = 2000):
        self._max_lines = max_lines
        self._max_line_length = max_line_length

    async def execute(self, file_path: str, offset: int = 0,
                      limit: int = 0, tool_id: str = "", **kwargs) -> "ToolResult":
        path = Path(file_path)

        if not path.exists():
            return self._error(f"File not found: {file_path}", tool_id)

        if path.is_dir():
            return self._error(
                f"Cannot read directory: {file_path}. Use 'ls' via Bash tool instead.",
                tool_id,
            )

        # Check if binary
        if self._is_binary(path):
            size = path.stat().st_size
            return self._success(
                f"[Binary file: {path.name}, size: {size} bytes]", tool_id
            )

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            start = max(0, offset - 1) if offset > 0 else 0
            end = start + (limit if limit > 0 else self._max_lines)
            selected = lines[start:end]

            # Format with line numbers (cat -n style)
            output_lines = []
            for i, line in enumerate(selected, start=start + 1):
                line = line.rstrip("\n")
                if len(line) > self._max_line_length:
                    line = line[:self._max_line_length] + "..."
                output_lines.append(f"{i:>6}\t{line}")

            output = "\n".join(output_lines)

            if end < total_lines:
                output += f"\n\n[Showing lines {start+1}-{min(end, total_lines)} of {total_lines}]"

            if not output.strip():
                output = "[File is empty]"

            return self._success(output, tool_id)

        except Exception as e:
            return self._error(f"Error reading file: {str(e)}", tool_id)

    @staticmethod
    def _is_binary(path: Path) -> bool:
        """Check if a file appears to be binary."""
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
                return b"\x00" in chunk
        except Exception:
            return False
