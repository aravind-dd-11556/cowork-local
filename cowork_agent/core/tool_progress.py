"""
Tool Progress — Progress tracking infrastructure for long-running tools.

Provides a typed callback and tracker that tools can use to report
intermediate progress (percentage + message) during execution.

Sprint 14 (Streaming & Partial Output) Module 3.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from .models import ToolCall

logger = logging.getLogger(__name__)

# Type alias for progress callbacks
# (progress_percent: int, message: str) -> None
ProgressCallback = Callable[[int, str], None]


class ProgressTracker:
    """
    Wraps a progress callback for a specific tool execution.

    Clamps percentage values to valid range, handles callback errors
    gracefully, and provides convenience methods.

    Usage:
        tracker = ProgressTracker(callback=my_callback, tool_call=call)
        tracker.start("Beginning task...")
        tracker.update(50, "Halfway done")
        tracker.complete("Finished!")
    """

    def __init__(self, callback: Optional[ProgressCallback] = None,
                 tool_call: Optional[ToolCall] = None):
        self._callback = callback
        self._tool_call = tool_call
        self._last_percent: int = 0
        self._update_count: int = 0

    @property
    def has_callback(self) -> bool:
        """Check if a callback is registered."""
        return self._callback is not None

    @property
    def tool_call(self) -> Optional[ToolCall]:
        return self._tool_call

    @property
    def last_percent(self) -> int:
        return self._last_percent

    @property
    def update_count(self) -> int:
        return self._update_count

    def update(self, percent: int, message: str) -> None:
        """
        Report progress update.

        Args:
            percent: Progress percentage (0–100). Values outside range are
                     clamped. Use -1 for indeterminate progress.
            message: Human-readable status message.
        """
        if self._callback is None:
            return

        # Clamp to valid range (allow -1 for indeterminate)
        if percent != -1:
            percent = max(0, min(100, percent))

        self._last_percent = percent
        self._update_count += 1

        try:
            self._callback(percent, message)
        except Exception as e:
            # Never let a callback error break tool execution
            logger.debug(f"Progress callback error (ignored): {e}")

    def start(self, message: str = "Starting...") -> None:
        """Convenience: report 0% progress."""
        self.update(0, message)

    def complete(self, message: str = "Complete") -> None:
        """Convenience: report 100% progress."""
        self.update(100, message)

    def indeterminate(self, message: str = "Working...") -> None:
        """Convenience: report indeterminate progress (-1)."""
        self.update(-1, message)
