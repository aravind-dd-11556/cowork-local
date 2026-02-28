"""
Bash Tool — Execute shell commands with persistent working directory.
Mirrors Cowork's Bash tool behavior.
"""

from __future__ import annotations
import asyncio
import os
from typing import Optional

from .base import BaseTool


class BashTool(BaseTool):
    name = "bash"
    description = (
        "Execute shell commands. Working directory persists between calls; "
        "shell state (env vars, aliases) does NOT persist. "
        "IMPORTANT: Do NOT use for file operations — use Read, Write, Edit, Glob, Grep instead. "
        "When running multiple independent commands, make multiple tool calls. "
        "When running dependent commands, chain with && in one call."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "description": {
                "type": "string",
                "description": "Brief description of what this command does",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default 120, max 600)",
            },
        },
        "required": ["command"],
    }

    def __init__(self, workspace_dir: str = "."):
        resolved = os.path.abspath(workspace_dir)
        # Validate: if workspace doesn't exist, fall back to cwd
        if os.path.isdir(resolved):
            self._cwd = resolved
        else:
            self._cwd = os.getcwd()
        self._max_output = 30000

    async def execute(self, command: str, description: str = "",
                      timeout: float = 120, tool_id: str = "",
                      progress_callback=None, **kwargs) -> "ToolResult":
        timeout = min(timeout, 600)

        if progress_callback:
            try:
                progress_callback(0, f"Running: {command[:60]}...")
            except Exception:
                pass  # Never let callback errors break tool execution

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
            )

            try:
                if progress_callback:
                    # Poll process with progress updates for long-running commands
                    stdout, stderr = await self._communicate_with_progress(
                        process, timeout, progress_callback,
                    )
                else:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=timeout
                    )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                return self._error(
                    f"Command timed out after {timeout}s: {command}", tool_id
                )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Truncate output if too large
            output = stdout_str
            if stderr_str:
                output += f"\n[stderr]\n{stderr_str}" if output else stderr_str

            if len(output) > self._max_output:
                output = output[:self._max_output] + "\n[Output truncated]"

            # Update cwd if command changed directory
            self._update_cwd(command)

            if progress_callback:
                try:
                    progress_callback(100, "Command completed")
                except Exception:
                    pass

            if process.returncode != 0:
                return self._error(
                    f"Exit code {process.returncode}\n{output}", tool_id
                )

            return self._success(output or "(no output)", tool_id)

        except Exception as e:
            return self._error(f"Failed to execute: {str(e)}", tool_id)

    def _update_cwd(self, command: str):
        """Track cd commands to maintain persistent cwd."""
        import re
        # Split on && and ; to handle chained commands
        parts = re.split(r'&&|;', command.strip())
        for part in parts:
            part = part.strip()
            # Match: cd, cd /path, cd "path", cd 'path', cd ~, cd ~/path
            if part == "cd" or part.startswith("cd "):
                target = part[2:].strip().strip("'\"") if len(part) > 2 else ""

                # Skip unsupported forms: cd -, cd $VAR, cd $(...)
                if target.startswith('-') or target.startswith('$') or target.startswith('`'):
                    continue

                # Handle ~ expansion
                if target == "" or target == "~":
                    target = os.path.expanduser("~")
                elif target.startswith("~/"):
                    target = os.path.expanduser(target)

                if os.path.isabs(target):
                    new_cwd = target
                else:
                    new_cwd = os.path.join(self._cwd, target)
                new_cwd = os.path.normpath(new_cwd)
                if os.path.isdir(new_cwd):
                    self._cwd = new_cwd

    async def _communicate_with_progress(
        self,
        process: asyncio.subprocess.Process,
        timeout: float,
        progress_callback,
    ) -> tuple[bytes, bytes]:
        """Communicate with subprocess while reporting progress at intervals."""
        import time as _time
        start = _time.time()
        poll_interval = 2.0  # report every 2 seconds
        last_report = start

        while True:
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=min(poll_interval, timeout)
                )
                return stdout, stderr
            except asyncio.TimeoutError:
                elapsed = _time.time() - start
                if elapsed >= timeout:
                    raise  # re-raise to let caller handle
                pct = min(int((elapsed / timeout) * 100), 99)
                if _time.time() - last_report >= poll_interval:
                    progress_callback(pct, f"Running... ({elapsed:.0f}s elapsed)")
                    last_report = _time.time()
                # Reduce remaining timeout
                timeout -= elapsed
                start = _time.time()
