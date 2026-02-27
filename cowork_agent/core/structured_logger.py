"""
Structured Logger — JSON logging, trace IDs, and contextual fields.

Provides a StructuredLogger wrapper that adds trace_id, agent_name, tool_name,
and arbitrary extra fields to every log record.  Two output modes:

- **JSON mode** (`COWORK_LOG_FORMAT=json`): each line is a JSON object.
- **Human mode** (default): traditional format with `[trace_id]` prefix.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ── LogContext ──────────────────────────────────────────────────────

@dataclass
class LogContext:
    """Immutable bag of contextual fields attached to every log line."""
    trace_id: str = ""
    agent_name: str = ""
    tool_name: str = ""
    provider_name: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def merged_with(self, **kwargs: Any) -> "LogContext":
        """Return a *new* LogContext with the given fields overridden."""
        new_extra = {**self.extra, **kwargs.pop("extra", {})}
        data = {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "provider_name": self.provider_name,
            "extra": new_extra,
        }
        data.update(kwargs)
        return LogContext(**data)


# ── JSON formatter ──────────────────────────────────────────────────

class StructuredFormatter(logging.Formatter):
    """Emits each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject context fields if they exist on the record
        for ctx_field in ("trace_id", "agent_name", "tool_name", "provider_name"):
            val = getattr(record, ctx_field, "")
            if val:
                entry[ctx_field] = val

        # Inject any extra dict
        log_extra = getattr(record, "log_extra", None)
        if log_extra:
            entry["extra"] = log_extra

        # Exception info
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


# ── Human-readable formatter (with trace_id) ───────────────────────

class HumanFormatter(logging.Formatter):
    """Traditional format with optional [trace_id] prefix."""

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(
            fmt=fmt or "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt=datefmt or "%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        trace_id = getattr(record, "trace_id", "")
        if trace_id:
            record.msg = f"[{trace_id}] {record.msg}"
        return super().format(record)


# ── TraceIDFilter ───────────────────────────────────────────────────

class TraceIDFilter(logging.Filter):
    """Injects trace_id (and other context fields) into every log record."""

    def __init__(self, context: Optional[LogContext] = None):
        super().__init__()
        self._context = context or LogContext()

    @property
    def context(self) -> LogContext:
        return self._context

    @context.setter
    def context(self, ctx: LogContext) -> None:
        self._context = ctx

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = getattr(record, "trace_id", "") or self._context.trace_id
        record.agent_name = getattr(record, "agent_name", "") or self._context.agent_name
        record.tool_name = getattr(record, "tool_name", "") or self._context.tool_name
        record.provider_name = getattr(record, "provider_name", "") or self._context.provider_name
        return True


# ── StructuredLogger ────────────────────────────────────────────────

class StructuredLogger:
    """
    High-level logger wrapper that injects context into every log call.

    Usage::

        logger = StructuredLogger("my_module")
        logger = logger.with_context(trace_id="abc123", agent_name="main")
        logger.info("Starting task", task="summarise")
    """

    def __init__(self, name: str, context: Optional[LogContext] = None):
        self._name = name
        self._logger = logging.getLogger(name)
        self._context = context or LogContext()

    # ── Derived loggers ────────────────────────────────────────

    def with_context(self, **kwargs: Any) -> "StructuredLogger":
        """Return a **new** StructuredLogger with merged context."""
        new_ctx = self._context.merged_with(**kwargs)
        return StructuredLogger(self._name, context=new_ctx)

    def bind(self, **kwargs: Any) -> "StructuredLogger":
        """Add fields to the *current* logger's context (mutating)."""
        self._context = self._context.merged_with(**kwargs)
        return self

    # ── Logging methods ────────────────────────────────────────

    def debug(self, msg: str, **extra: Any) -> None:
        self._log(logging.DEBUG, msg, extra)

    def info(self, msg: str, **extra: Any) -> None:
        self._log(logging.INFO, msg, extra)

    def warning(self, msg: str, **extra: Any) -> None:
        self._log(logging.WARNING, msg, extra)

    def error(self, msg: str, **extra: Any) -> None:
        self._log(logging.ERROR, msg, extra)

    def critical(self, msg: str, **extra: Any) -> None:
        self._log(logging.CRITICAL, msg, extra)

    # ── Internal ───────────────────────────────────────────────

    def _log(self, level: int, msg: str, extra: Dict[str, Any]) -> None:
        if not self._logger.isEnabledFor(level):
            return
        record = self._logger.makeRecord(
            name=self._name,
            level=level,
            fn="",
            lno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        # Stamp context fields onto the record
        record.trace_id = self._context.trace_id          # type: ignore[attr-defined]
        record.agent_name = self._context.agent_name      # type: ignore[attr-defined]
        record.tool_name = self._context.tool_name        # type: ignore[attr-defined]
        record.provider_name = self._context.provider_name  # type: ignore[attr-defined]
        if extra:
            record.log_extra = extra                       # type: ignore[attr-defined]
        self._logger.handle(record)

    # ── Utility ────────────────────────────────────────────────

    @staticmethod
    def generate_trace_id() -> str:
        """Generate a short 12-char hex trace ID."""
        return uuid.uuid4().hex[:12]

    @property
    def context(self) -> LogContext:
        return self._context

    def __repr__(self) -> str:
        return f"StructuredLogger({self._name!r}, context={self._context})"


# ── Module-level setup function ─────────────────────────────────────

def setup_structured_logging(
    json_mode: Optional[bool] = None,
    level: str = "WARNING",
) -> None:
    """
    Configure the root logger for structured output.

    Parameters
    ----------
    json_mode : bool or None
        If None, auto-detect from ``COWORK_LOG_FORMAT`` env var
        (set to ``"json"`` to enable JSON mode).
    level : str
        Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    if json_mode is None:
        json_mode = os.getenv("COWORK_LOG_FORMAT", "").lower() == "json"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.WARNING))

    # Remove existing handlers
    root.handlers.clear()

    # Create handler
    handler = logging.StreamHandler()
    if json_mode:
        handler.setFormatter(StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    else:
        handler.setFormatter(HumanFormatter())

    # Add a global TraceIDFilter so any record gets context defaults
    handler.addFilter(TraceIDFilter())

    root.addHandler(handler)
