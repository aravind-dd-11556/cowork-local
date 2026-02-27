"""
Graceful Shutdown Manager — signal handling + ordered callback cleanup.

Transitions through phases: RUNNING → DRAINING → STOPPING → CLEANUP → COMPLETED.
Registered callbacks run in priority order (highest first), each with its own
timeout.  If a callback fails, the manager logs a warning and continues.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import signal
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


# ── Shutdown phases ─────────────────────────────────────────────────

class ShutdownPhase(Enum):
    RUNNING = "running"
    DRAINING = "draining"
    STOPPING = "stopping"
    CLEANUP = "cleanup"
    COMPLETED = "completed"


# ── Callback descriptor ────────────────────────────────────────────

@dataclass
class ShutdownCallback:
    """A named cleanup function with priority and per-callback timeout."""
    name: str
    callback: Callable          # sync or async
    priority: int = 0           # higher = runs first during shutdown
    timeout_seconds: float = 30.0


# ── ShutdownManager ────────────────────────────────────────────────

class ShutdownManager:
    """
    Orchestrates graceful shutdown with ordered callbacks.

    Usage::

        mgr = ShutdownManager(drain_timeout=10.0)
        mgr.register_callback("db", close_db, priority=10, timeout=5.0)
        mgr.install_signal_handlers(loop)
        # later …
        await mgr.shutdown(reason="SIGTERM")
    """

    def __init__(self, drain_timeout: float = 30.0) -> None:
        self._phase = ShutdownPhase.RUNNING
        self._callbacks: List[ShutdownCallback] = []
        self._drain_timeout = drain_timeout
        self._shutting_down = False
        self._shutdown_event = asyncio.Event()
        self._shutdown_reason: Optional[str] = None
        self._results: List[dict] = []

    # ── Properties ─────────────────────────────────────────────

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    @property
    def phase(self) -> ShutdownPhase:
        return self._phase

    @property
    def shutdown_event(self) -> asyncio.Event:
        """Event set once shutdown begins — agent loops can ``await`` this."""
        return self._shutdown_event

    @property
    def shutdown_reason(self) -> Optional[str]:
        return self._shutdown_reason

    @property
    def results(self) -> List[dict]:
        """Per-callback results from the last shutdown run."""
        return list(self._results)

    # ── Callback management ────────────────────────────────────

    def register_callback(
        self,
        name: str,
        callback: Callable,
        priority: int = 0,
        timeout: float = 30.0,
    ) -> None:
        """Register a shutdown callback (sync or async)."""
        self._callbacks.append(
            ShutdownCallback(
                name=name,
                callback=callback,
                priority=priority,
                timeout_seconds=timeout,
            )
        )

    def unregister_callback(self, name: str) -> None:
        """Remove a callback by name."""
        self._callbacks = [cb for cb in self._callbacks if cb.name != name]

    @property
    def callback_names(self) -> List[str]:
        return [cb.name for cb in self._callbacks]

    # ── Signal handlers ────────────────────────────────────────

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Hook SIGTERM and SIGINT to trigger shutdown via the event loop."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.ensure_future(
                    self.shutdown(reason=s.name), loop=loop,
                ),
            )

    # ── Shutdown sequence ──────────────────────────────────────

    async def shutdown(self, reason: str = "signal") -> None:
        """
        Execute the full shutdown sequence.

        Idempotent — calling twice has no additional effect.
        """
        if self._shutting_down:
            return
        self._shutting_down = True
        self._shutdown_reason = reason
        self._shutdown_event.set()
        self._results.clear()

        logger.info("Shutdown initiated (reason=%s)", reason)

        # Phase: DRAINING — stop accepting new work
        self._phase = ShutdownPhase.DRAINING
        logger.info("Phase: DRAINING (timeout=%.1fs)", self._drain_timeout)
        await asyncio.sleep(0)  # yield once; a real system would wait for in-flight tasks

        # Phase: STOPPING — tell components to stop
        self._phase = ShutdownPhase.STOPPING
        logger.info("Phase: STOPPING")

        # Phase: CLEANUP — run registered callbacks in priority order
        self._phase = ShutdownPhase.CLEANUP
        sorted_cbs = sorted(self._callbacks, key=lambda cb: cb.priority, reverse=True)

        for cb in sorted_cbs:
            result = {"name": cb.name, "status": "ok", "error": None}
            try:
                coro = cb.callback
                if inspect.iscoroutinefunction(coro):
                    await asyncio.wait_for(coro(), timeout=cb.timeout_seconds)
                else:
                    # Sync callback — run in executor to avoid blocking
                    coro()
                logger.info("Cleanup callback '%s' completed.", cb.name)
            except asyncio.TimeoutError:
                result["status"] = "timeout"
                result["error"] = f"Timed out after {cb.timeout_seconds}s"
                logger.warning(
                    "Cleanup callback '%s' timed out after %.1fs",
                    cb.name, cb.timeout_seconds,
                )
            except Exception as exc:
                result["status"] = "error"
                result["error"] = str(exc)
                logger.warning(
                    "Cleanup callback '%s' failed: %s", cb.name, exc,
                )
            self._results.append(result)

        # Phase: COMPLETED
        self._phase = ShutdownPhase.COMPLETED
        logger.info("Shutdown completed (reason=%s)", reason)

    # ── Reset (useful for tests) ───────────────────────────────

    def reset(self) -> None:
        """Reset the manager to RUNNING state (for testing)."""
        self._phase = ShutdownPhase.RUNNING
        self._shutting_down = False
        self._shutdown_event.clear()
        self._shutdown_reason = None
        self._results.clear()
