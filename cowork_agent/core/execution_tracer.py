"""
Execution Tracer — hierarchical span-based tracing for agent operations.

Builds on ``StructuredLogger``'s trace IDs to add parent/child span
relationships and timing, enabling debugging of complex multi-step runs.

Span hierarchy example::

    agent.run (root)
    ├── provider.send  (child)
    ├── tool.bash      (child)
    │   └── provider.send  (grandchild — if tool triggers sub-agent)
    └── provider.send  (child)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Span dataclass ────────────────────────────────────────────────

@dataclass
class Span:
    """A single span representing one logical operation."""
    span_id: str
    parent_id: Optional[str]
    trace_id: str
    operation: str                      # e.g. "agent.run", "tool.bash"
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"             # "running" | "ok" | "error"
    error_message: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "children": self.children,
        }


# ── ExecutionTracer ───────────────────────────────────────────────

class ExecutionTracer:
    """
    Hierarchical span tracer for a single agent run.

    Usage::

        tracer = ExecutionTracer()
        root = tracer.start_span("agent.run", user_input="hello")

        # Child span for provider call
        prov = tracer.start_span("provider.send", parent_id=root, model="llama3")
        tracer.end_span(prov, status="ok")

        # Child span for tool execution
        tool = tracer.start_span("tool.bash", parent_id=root, command="ls")
        tracer.end_span(tool, status="ok")

        tracer.end_span(root, status="ok")

        print(tracer.to_json())
    """

    def __init__(self, trace_id: Optional[str] = None):
        self._trace_id = trace_id or uuid.uuid4().hex[:12]
        self._spans: Dict[str, Span] = {}
        self._root_id: Optional[str] = None

    # ── Properties ─────────────────────────────────────────────

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def root_span_id(self) -> Optional[str]:
        return self._root_id

    @property
    def span_count(self) -> int:
        return len(self._spans)

    # ── Span lifecycle ─────────────────────────────────────────

    def start_span(
        self,
        operation: str,
        parent_id: Optional[str] = None,
        **attributes: Any,
    ) -> str:
        """
        Start a new span.

        Returns the span_id (use it to call ``end_span`` later).
        """
        span_id = f"span_{uuid.uuid4().hex[:8]}"

        span = Span(
            span_id=span_id,
            parent_id=parent_id,
            trace_id=self._trace_id,
            operation=operation,
            start_time=time.time(),
            attributes=dict(attributes),
        )
        self._spans[span_id] = span

        # Track root
        if parent_id is None and self._root_id is None:
            self._root_id = span_id

        # Register as child of parent
        if parent_id and parent_id in self._spans:
            self._spans[parent_id].children.append(span_id)

        return span_id

    def end_span(
        self,
        span_id: str,
        status: str = "ok",
        error: Optional[str] = None,
    ) -> None:
        """End a span, recording its final status and duration."""
        span = self._spans.get(span_id)
        if not span:
            logger.warning("Attempted to end unknown span: %s", span_id)
            return

        span.end_time = time.time()
        span.status = status
        if error:
            span.error_message = error

    # ── Querying ───────────────────────────────────────────────

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a span by ID."""
        return self._spans.get(span_id)

    def get_flat_spans(self) -> List[Span]:
        """Return all spans in chronological order."""
        return sorted(self._spans.values(), key=lambda s: s.start_time)

    def get_trace_tree(self) -> dict:
        """
        Return the full trace as a nested dict tree.

        Each node has the span's data plus a ``children`` list of nested nodes.
        """
        if not self._root_id:
            return {}
        return self._build_tree(self._root_id)

    def get_error_spans(self) -> List[Span]:
        """Return only spans that ended with errors."""
        return [s for s in self._spans.values() if s.status == "error"]

    # ── Export ─────────────────────────────────────────────────

    def to_json(self, indent: int = 2) -> str:
        """Export the full trace as JSON."""
        return json.dumps({
            "trace_id": self._trace_id,
            "span_count": len(self._spans),
            "root_span_id": self._root_id,
            "spans": [s.to_dict() for s in self.get_flat_spans()],
        }, indent=indent)

    def to_summary(self) -> dict:
        """Compact summary for logging."""
        total = len(self._spans)
        errors = len(self.get_error_spans())
        root = self._spans.get(self._root_id) if self._root_id else None
        return {
            "trace_id": self._trace_id,
            "total_spans": total,
            "error_spans": errors,
            "root_duration_ms": root.duration_ms if root else None,
        }

    # ── Internal ───────────────────────────────────────────────

    def _build_tree(self, span_id: str) -> dict:
        """Recursively build a nested dict from a span and its children."""
        span = self._spans.get(span_id)
        if not span:
            return {}
        node = span.to_dict()
        node["children"] = [self._build_tree(cid) for cid in span.children]
        return node
