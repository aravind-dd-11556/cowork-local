"""
Correlation ID Manager — request tracing with parent-child relationships.

Propagates trace IDs through async boundaries using ``contextvars``.
Supports depth tracking and header injection/extraction for distributed tracing.

Sprint 16 (Testing & Observability Hardening) Module 2.
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


# ── Context variable for async propagation ───────────────────────

_current_context: contextvars.ContextVar[Optional["CorrelationContext"]] = (
    contextvars.ContextVar("_correlation_context", default=None)
)


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class CorrelationContext:
    """Trace context for a single operation."""
    trace_id: str
    parent_trace_id: Optional[str] = None
    depth: int = 0
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "parent_trace_id": self.parent_trace_id,
            "depth": self.depth,
            "start_time": self.start_time,
            "elapsed_ms": round((time.time() - self.start_time) * 1000, 2),
            "metadata": self.metadata,
        }


# ── CorrelationIdManager ────────────────────────────────────────

class CorrelationIdManager:
    """
    Manages trace-ID generation, context propagation, and header injection.

    Usage::

        mgr = CorrelationIdManager()
        with mgr.trace("my-operation"):
            ctx = mgr.current_context()
            # ctx.trace_id is automatically set
            with mgr.child_trace("sub-operation"):
                child = mgr.current_context()
                assert child.parent_trace_id == ctx.trace_id
    """

    DEFAULT_HEADER_NAME = "X-Correlation-ID"
    PARENT_HEADER_NAME = "X-Parent-Correlation-ID"

    def __init__(self, header_name: Optional[str] = None):
        self._header_name = header_name or self.DEFAULT_HEADER_NAME
        self._total_generated: int = 0
        self._max_depth_seen: int = 0

    # ── ID generation ─────────────────────────────────────────

    def generate_trace_id(self) -> str:
        """Generate a 12-char hex trace ID."""
        self._total_generated += 1
        return uuid.uuid4().hex[:12]

    # ── Context management ────────────────────────────────────

    def current_context(self) -> Optional[CorrelationContext]:
        """Get the current correlation context (or None)."""
        return _current_context.get()

    def set_context(self, context: CorrelationContext) -> None:
        """Set the current context explicitly."""
        _current_context.set(context)
        if context.depth > self._max_depth_seen:
            self._max_depth_seen = context.depth

    def clear_context(self) -> None:
        """Clear the current context."""
        _current_context.set(None)

    @contextmanager
    def trace(
        self,
        operation_name: str = "",
        trace_id: Optional[str] = None,
        **metadata: Any,
    ) -> Generator[CorrelationContext, None, None]:
        """
        Context manager that creates a new root trace context.

        Restores the previous context on exit.
        """
        prev = _current_context.get()
        tid = trace_id or self.generate_trace_id()

        ctx = CorrelationContext(
            trace_id=tid,
            parent_trace_id=None,
            depth=0,
            metadata={"operation": operation_name, **metadata} if operation_name else metadata,
        )
        _current_context.set(ctx)

        try:
            yield ctx
        finally:
            _current_context.set(prev)

    @contextmanager
    def child_trace(
        self,
        operation_name: str = "",
        **metadata: Any,
    ) -> Generator[CorrelationContext, None, None]:
        """
        Context manager for a child trace inheriting the current parent.

        If no parent context exists, creates a root trace instead.
        """
        parent = _current_context.get()

        if parent is None:
            # No parent — create root trace
            with self.trace(operation_name, **metadata) as ctx:
                yield ctx
            return

        child_id = self.generate_trace_id()
        child = CorrelationContext(
            trace_id=child_id,
            parent_trace_id=parent.trace_id,
            depth=parent.depth + 1,
            metadata={"operation": operation_name, **metadata} if operation_name else metadata,
        )
        _current_context.set(child)
        if child.depth > self._max_depth_seen:
            self._max_depth_seen = child.depth

        try:
            yield child
        finally:
            _current_context.set(parent)

    # ── Header injection / extraction ─────────────────────────

    def extract_headers(self) -> Dict[str, str]:
        """Extract trace headers from current context for outgoing requests."""
        ctx = _current_context.get()
        if ctx is None:
            return {}
        headers = {self._header_name: ctx.trace_id}
        if ctx.parent_trace_id:
            headers[self.PARENT_HEADER_NAME] = ctx.parent_trace_id
        return headers

    def inject_from_headers(self, headers: Dict[str, str]) -> Optional[CorrelationContext]:
        """
        Create and set a context from incoming request headers.

        Returns the created context, or None if no trace header is present.
        """
        trace_id = headers.get(self._header_name)
        if not trace_id:
            return None

        parent_id = headers.get(self.PARENT_HEADER_NAME)
        ctx = CorrelationContext(
            trace_id=trace_id,
            parent_trace_id=parent_id,
            depth=0,
        )
        _current_context.set(ctx)
        return ctx

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return manager statistics."""
        ctx = _current_context.get()
        return {
            "total_generated": self._total_generated,
            "max_depth_seen": self._max_depth_seen,
            "current_trace_id": ctx.trace_id if ctx else None,
            "current_depth": ctx.depth if ctx else 0,
            "header_name": self._header_name,
        }
