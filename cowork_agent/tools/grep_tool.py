"""
Grep Tool — Content search via ripgrep (rg) with Python re fallback.
Mirrors Cowork's Grep tool: regex support, output modes, file type filtering.
"""

from __future__ import annotations
import asyncio
import os
import re
import shutil
from pathlib import Path

from .base import BaseTool


class GrepTool(BaseTool):
    name = "grep"
    description = (
        "Search file contents using regular expressions. Built on ripgrep (rg). "
        "Supports regex patterns, file type filtering, and multiple output modes: "
        "'content' shows matching lines, 'files_with_matches' shows file paths, "
        "'count' shows match counts per file."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search in (defaults to cwd)",
            },
            "glob": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g. '*.py')",
            },
            "type": {
                "type": "string",
                "description": "File type filter (e.g. 'py', 'js', 'rust')",
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": "Output mode (default: files_with_matches)",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Case insensitive search (default: false)",
            },
            "context": {
                "type": "number",
                "description": "Lines of context around matches",
            },
            "head_limit": {
                "type": "number",
                "description": "Limit output to first N results",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, default_dir: str = "."):
        resolved = os.path.abspath(default_dir)
        self._default_dir = resolved if os.path.isdir(resolved) else os.getcwd()
        self._has_rg = shutil.which("rg") is not None

    # SEC-HIGH-2: Cap pattern length to mitigate ReDoS on the Python re fallback
    MAX_PATTERN_LENGTH = 1000

    async def execute(
        self,
        pattern: str,
        path: str = "",
        output_mode: str = "files_with_matches",
        case_insensitive: bool = False,
        context: int = 0,
        head_limit: int = 0,
        tool_id: str = "",
        **kwargs,
    ) -> "ToolResult":
        # Validate pattern length to limit ReDoS surface
        if len(pattern) > self.MAX_PATTERN_LENGTH:
            return self._error(
                f"Pattern too long ({len(pattern)} chars). "
                f"Maximum allowed: {self.MAX_PATTERN_LENGTH}",
                tool_id,
            )

        # Validate regex compiles before executing (fast fail)
        try:
            re.compile(pattern)
        except re.error as e:
            return self._error(f"Invalid regex pattern: {e}", tool_id)

        search_path = path or self._default_dir
        glob_filter = kwargs.get("glob", "")
        type_filter = kwargs.get("type", "")

        if self._has_rg:
            return await self._search_rg(
                pattern, search_path, output_mode,
                case_insensitive, context, head_limit,
                glob_filter, type_filter, tool_id,
            )
        else:
            return await self._search_python(
                pattern, search_path, output_mode,
                case_insensitive, context, head_limit,
                glob_filter, tool_id,
            )

    async def _search_rg(
        self, pattern, path, output_mode, case_insensitive,
        context, head_limit, glob_filter, type_filter, tool_id,
    ) -> "ToolResult":
        """Search using ripgrep (rg)."""
        cmd = ["rg"]

        # Output mode
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")
        else:
            cmd.append("-n")  # line numbers for content mode

        if case_insensitive:
            cmd.append("-i")

        if context and output_mode == "content":
            cmd.extend(["-C", str(context)])

        if glob_filter:
            cmd.extend(["--glob", glob_filter])

        if type_filter:
            cmd.extend(["--type", type_filter])

        cmd.append(pattern)
        cmd.append(path)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30
            )
            output = stdout.decode("utf-8", errors="replace")

            if proc.returncode == 1:
                # rg returns 1 when no matches found
                return self._success(f"No matches found for pattern '{pattern}'", tool_id)

            if proc.returncode not in (0, 1):
                err = stderr.decode("utf-8", errors="replace")
                return self._error(f"rg error: {err}", tool_id)

            # Apply head_limit
            if head_limit > 0:
                lines = output.split("\n")
                output = "\n".join(lines[:head_limit])

            # Truncate large output
            if len(output) > 30000:
                output = output[:30000] + "\n\n[Output truncated at 30000 characters]"

            return self._success(output.strip(), tool_id)

        except asyncio.TimeoutError:
            return self._error("Search timed out after 30 seconds", tool_id)
        except Exception as e:
            return self._error(f"Search error: {str(e)}", tool_id)

    async def _search_python(
        self, pattern, path, output_mode, case_insensitive,
        context, head_limit, glob_filter, tool_id,
    ) -> "ToolResult":
        """Fallback search using Python re module."""
        search_path = Path(path)
        flags = re.IGNORECASE if case_insensitive else 0

        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return self._error(f"Invalid regex: {str(e)}", tool_id)

        results = []
        file_count = 0

        if search_path.is_file():
            files = [search_path]
        else:
            if glob_filter:
                files = list(search_path.rglob(glob_filter))
            else:
                files = [
                    f for f in search_path.rglob("*")
                    if f.is_file() and not self._is_binary_quick(f)
                ]

        for fpath in files:
            if not fpath.is_file():
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
                lines = text.split("\n")
                matches = []

                for i, line in enumerate(lines):
                    if compiled.search(line):
                        matches.append((i + 1, line))

                if matches:
                    file_count += 1
                    if output_mode == "files_with_matches":
                        results.append(str(fpath))
                    elif output_mode == "count":
                        results.append(f"{fpath}:{len(matches)}")
                    else:
                        for lineno, line in matches:
                            results.append(f"{fpath}:{lineno}:{line}")

            except Exception:
                continue

            if head_limit and len(results) >= head_limit:
                results = results[:head_limit]
                break

        if not results:
            return self._success(f"No matches found for pattern '{pattern}'", tool_id)

        output = "\n".join(results)
        if len(output) > 30000:
            output = output[:30000] + "\n\n[Output truncated]"

        return self._success(output, tool_id)

    @staticmethod
    def _is_binary_quick(path: Path) -> bool:
        """Quick binary check — read first 512 bytes."""
        try:
            with open(path, "rb") as f:
                chunk = f.read(512)
                return b"\x00" in chunk
        except Exception:
            return True
