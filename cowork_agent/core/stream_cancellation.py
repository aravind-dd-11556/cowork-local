"""
Stream Cancellation — Graceful cancellation for streaming and tool execution.

Provides a token-based cancellation mechanism that can be shared between
the agent loop, tool execution, and the UI layer (CLI/API).

Sprint 14 (Streaming & Partial Output) Module 2.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class StreamCancelledError(Exception):
    """Raised when a stream or tool execution is cancelled."""

    def __init__(self, message: str = "Stream cancelled by user"):
        super().__init__(message)


class StreamCancellationToken:
    """
    Token for cooperative cancellation of streaming and tool execution.

    Usage:
        token = StreamCancellationToken()

        # In the UI layer (CLI / API):
        token.cancel()

        # In the agent loop or tool:
        if token.is_cancelled:
            raise StreamCancelledError()
        # or:
        token.check()  # raises StreamCancelledError if cancelled
    """

    def __init__(self):
        self._event = asyncio.Event()
        self._cancel_reason: str = ""

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._event.is_set()

    @property
    def cancel_reason(self) -> str:
        """Get the reason for cancellation (empty if not cancelled)."""
        return self._cancel_reason

    def cancel(self, reason: str = "User cancelled") -> None:
        """
        Request cancellation.

        This is thread-safe and can be called from any thread or coroutine.
        """
        self._cancel_reason = reason
        self._event.set()
        logger.info(f"Stream cancellation requested: {reason}")

    def check(self) -> None:
        """
        Check cancellation status and raise if cancelled.

        Raises:
            StreamCancelledError: If cancellation has been requested.
        """
        if self._event.is_set():
            raise StreamCancelledError(self._cancel_reason or "Stream cancelled")

    def reset(self) -> None:
        """Clear the cancellation state for reuse."""
        self._event.clear()
        self._cancel_reason = ""

    async def wait(self, timeout: float | None = None) -> bool:
        """
        Wait for cancellation to be requested.

        Args:
            timeout: Max seconds to wait. None = wait forever.

        Returns:
            True if cancelled, False if timeout expired.
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def to_dict(self) -> dict:
        return {
            "is_cancelled": self.is_cancelled,
            "cancel_reason": self._cancel_reason,
        }
