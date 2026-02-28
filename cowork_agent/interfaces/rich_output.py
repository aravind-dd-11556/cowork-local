"""
Rich Output — pure ANSI formatting utilities for the CLI.

Provides table rendering, progress bars, enhanced error display (with
ErrorCatalog integration), tool execution timing, and smart truncation.
No external dependencies — only ANSI escape codes.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional


# ── ANSI codes ────────────────────────────────────────────────────

class _C:
    """ANSI color/style codes."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"


# ── Box-drawing characters ────────────────────────────────────────

_H  = "─"   # horizontal
_V  = "│"   # vertical
_TL = "┌"   # top-left
_TR = "┐"   # top-right
_BL = "└"   # bottom-left
_BR = "┘"   # bottom-right
_ML = "├"   # mid-left
_MR = "┤"   # mid-right
_MC = "┼"   # mid-cross
_TM = "┬"   # top-mid
_BM = "┴"   # bottom-mid


# ── Tool icons (reuse from cli.py) ────────────────────────────────

_TOOL_ICONS: Dict[str, str] = {
    "bash": "⚡", "read": "📄", "write": "✏️", "edit": "🔧",
    "glob": "🔍", "grep": "🔎", "web_search": "🌐", "web_fetch": "🌍",
    "todo_write": "📋", "ask_user": "❓", "notebook_edit": "📓",
    "delete_file": "🗑️", "task": "🔀",
}


# ── RichOutput ────────────────────────────────────────────────────

class RichOutput:
    """
    Pure-ANSI formatting utilities for CLI output.

    Usage::

        out = RichOutput(width=80)
        print(out.table(["Name", "Value"], [["foo", "bar"]]))
        print(out.progress_bar(3, 10, label="Processing"))
        print(out.error("Something failed", code="E2001",
                        recovery_hint="Try again with a smaller input"))
    """

    def __init__(self, width: int = 80):
        self._width = width

    # ── Table rendering ────────────────────────────────────────

    def table(
        self,
        headers: List[str],
        rows: List[List[str]],
        col_widths: Optional[List[int]] = None,
    ) -> str:
        """
        Render a bordered ANSI table.

        If *col_widths* is not given, columns are auto-sized based on content.
        """
        if not headers:
            return ""

        n_cols = len(headers)

        # Auto-calculate widths
        if col_widths is None:
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row[:n_cols]):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
            # Cap to available width
            total = sum(col_widths) + n_cols * 3 + 1
            if total > self._width:
                excess = total - self._width
                # Shrink the widest column
                widest = col_widths.index(max(col_widths))
                col_widths[widest] = max(4, col_widths[widest] - excess)

        lines: List[str] = []

        # Top border
        segs = [_H * (w + 2) for w in col_widths]
        lines.append(f"  {_TL}{_TM.join(segs)}{_TR}")

        # Header row
        cells = []
        for i, h in enumerate(headers):
            w = col_widths[i] if i < len(col_widths) else 10
            cells.append(f" {_C.BOLD}{self._pad(h, w)}{_C.RESET} ")
        lines.append(f"  {_V}{'│'.join(cells)}{_V}")

        # Separator
        segs = [_H * (w + 2) for w in col_widths]
        lines.append(f"  {_ML}{_MC.join(segs)}{_MR}")

        # Data rows
        for row in rows:
            cells = []
            for i in range(n_cols):
                w = col_widths[i] if i < len(col_widths) else 10
                val = str(row[i]) if i < len(row) else ""
                cells.append(f" {self._pad(val, w)} ")
            lines.append(f"  {_V}{'│'.join(cells)}{_V}")

        # Bottom border
        segs = [_H * (w + 2) for w in col_widths]
        lines.append(f"  {_BL}{_BM.join(segs)}{_BR}")

        return "\n".join(lines)

    # ── Progress bar ───────────────────────────────────────────

    def progress_bar(
        self,
        current: int,
        total: int,
        width: int = 30,
        label: str = "",
    ) -> str:
        """
        Render a progress bar.

        Example: ``Processing [████████░░░░░░░░] 8/20 (40%)``
        """
        if total <= 0:
            pct = 0.0
        else:
            pct = min(current / total, 1.0)

        filled = int(width * pct)
        empty = width - filled
        bar = "█" * filled + "░" * empty

        pct_str = f"{pct * 100:.0f}%"
        prefix = f"{label} " if label else ""

        return (
            f"  {prefix}{_C.CYAN}[{bar}]{_C.RESET} "
            f"{current}/{total} ({pct_str})"
        )

    # ── Tool execution display ─────────────────────────────────

    def tool_result(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        output_lines: int = 0,
    ) -> str:
        """
        Render a tool execution result line with timing.

        Example: ``  ⚡ bash ✓ 234ms (42 lines)``
        """
        icon = _TOOL_ICONS.get(tool_name, "🔨")
        status = f"{_C.GREEN}✓{_C.RESET}" if success else f"{_C.RED}✗{_C.RESET}"

        # Format duration
        if duration_ms < 1000:
            time_str = f"{duration_ms:.0f}ms"
        else:
            time_str = f"{duration_ms / 1000:.1f}s"

        line = f"  {icon} {_C.CYAN}{tool_name}{_C.RESET} {status} {_C.DIM}{time_str}{_C.RESET}"

        if output_lines > 0:
            line += f" {_C.DIM}({output_lines} lines){_C.RESET}"

        return line

    # ── Enhanced error display ─────────────────────────────────

    def error(
        self,
        message: str,
        code: Optional[str] = None,
        recovery_hint: Optional[str] = None,
    ) -> str:
        """
        Render an enhanced error message with optional code and recovery hint.

        Example::

            ERROR [E2001] Tool execution failed
            Recovery: Try breaking the command into smaller steps
        """
        lines: List[str] = []

        # Error header
        if code:
            lines.append(f"  {_C.RED}{_C.BOLD}ERROR [{code}]{_C.RESET} {message}")
        else:
            lines.append(f"  {_C.RED}{_C.BOLD}ERROR{_C.RESET} {message}")

        # Recovery hint
        if recovery_hint:
            lines.append(f"  {_C.YELLOW}Recovery:{_C.RESET} {recovery_hint}")

        return "\n".join(lines)

    # ── Metrics summary table ──────────────────────────────────

    def metrics_table(self, metrics_summary: dict) -> str:
        """
        Render a metrics summary as a table.

        Expects the dict from ``MetricsCollector.summary()``.
        """
        tools = metrics_summary.get("tools", {})
        if not tools:
            return f"  {_C.DIM}No tool metrics recorded.{_C.RESET}"

        headers = ["Tool", "Calls", "Errors", "Avg (ms)", "Err Rate"]
        rows = []
        for name, m in sorted(tools.items()):
            rows.append([
                name,
                str(m.get("call_count", 0)),
                str(m.get("error_count", 0)),
                str(m.get("avg_ms", 0)),
                f"{m.get('error_rate', 0) * 100:.1f}%",
            ])

        return self.table(headers, rows)

    # ── Streaming progress bar (percent-based) ────────────────

    def stream_progress_bar(
        self,
        percent: int,
        width: int = 30,
        label: str = "",
    ) -> str:
        """
        Render a progress bar from a percentage (0–100).

        For indeterminate progress (percent == -1), shows a pulsing bar.
        Returns a single line suitable for ``\\r`` overwrite.

        Example: ``  Fetching... [████████░░░░░░░░] 40%``
        """
        prefix = f"{label} " if label else ""

        if percent == -1:
            # Indeterminate: pulsing dots
            return f"  {prefix}{_C.CYAN}[{'·' * width}]{_C.RESET} ..."

        pct = max(0, min(100, percent))
        filled = int(width * pct / 100)
        empty = width - filled
        bar = "█" * filled + "░" * empty

        return (
            f"  {prefix}{_C.CYAN}[{bar}]{_C.RESET} {pct}%"
        )

    # ── Smart truncation ───────────────────────────────────────

    def truncate(
        self,
        text: str,
        max_lines: int = 10,
        max_width: int = 100,
    ) -> str:
        """
        Truncate text to *max_lines*, capping each line at *max_width*.

        Adds a "... (N more lines)" indicator if truncated.
        """
        if not text:
            return ""

        lines = text.split("\n")
        total = len(lines)

        # Truncate each line
        result_lines = []
        for line in lines[:max_lines]:
            if len(line) > max_width:
                result_lines.append(line[:max_width - 3] + "...")
            else:
                result_lines.append(line)

        if total > max_lines:
            remaining = total - max_lines
            result_lines.append(
                f"{_C.DIM}... ({remaining} more line{'s' if remaining != 1 else ''}){_C.RESET}"
            )

        return "\n".join(result_lines)

    # ── Internal helpers ───────────────────────────────────────

    @staticmethod
    def _pad(text: str, width: int) -> str:
        """Pad or truncate text to fit *width*."""
        if len(text) > width:
            return text[:width - 1] + "…"
        return text.ljust(width)
