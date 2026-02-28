"""
Stream Events — Typed event protocol for streaming responses.

Provides structured events for CLI/API consumers to render rich feedback
during streaming: text chunks, tool start/end, progress, and status updates.

Sprint 14 (Streaming & Partial Output) Module 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from .models import ToolCall, ToolResult


# ── Event types ──────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A chunk of streamed text from the LLM."""
    text: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"type": "TextChunk", "text": self.text, "timestamp": self.timestamp}


@dataclass
class ToolStart:
    """Signals that a tool is about to execute."""
    tool_call: ToolCall
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": "ToolStart",
            "tool_name": self.tool_call.name,
            "tool_id": self.tool_call.tool_id,
            "input": self.tool_call.input,
            "timestamp": self.timestamp,
        }


@dataclass
class ToolProgress:
    """Intermediate progress update from a running tool."""
    tool_call: ToolCall
    progress_percent: int  # 0–100, or -1 for indeterminate
    message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": "ToolProgress",
            "tool_name": self.tool_call.name,
            "tool_id": self.tool_call.tool_id,
            "progress_percent": self.progress_percent,
            "message": self.message,
            "timestamp": self.timestamp,
        }


@dataclass
class ToolEnd:
    """Signals that a tool has finished executing."""
    tool_call: ToolCall
    result: ToolResult
    duration_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": "ToolEnd",
            "tool_name": self.tool_call.name,
            "tool_id": self.tool_call.tool_id,
            "success": self.result.success,
            "duration_ms": self.duration_ms,
            "output_lines": len(self.result.output.split("\n")) if self.result.output else 0,
            "timestamp": self.timestamp,
        }


@dataclass
class StatusUpdate:
    """Informational status message from the agent."""
    message: str
    severity: Literal["info", "warning"] = "info"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": "StatusUpdate",
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp,
        }


# ── Union type ───────────────────────────────────────────────────

StreamEvent = Union[TextChunk, ToolStart, ToolProgress, ToolEnd, StatusUpdate]


# ── Serialization helpers ────────────────────────────────────────

def event_to_dict(event: StreamEvent) -> dict:
    """Serialize a StreamEvent to a JSON-compatible dict."""
    return event.to_dict()


def event_from_dict(data: dict) -> StreamEvent:
    """
    Deserialize a dict to a StreamEvent.

    Requires a "type" key matching one of the event class names.
    Tool-related events are reconstructed with minimal ToolCall/ToolResult stubs.
    """
    event_type = data.get("type", "")

    if event_type == "TextChunk":
        return TextChunk(
            text=data.get("text", ""),
            timestamp=data.get("timestamp", 0.0),
        )

    elif event_type == "ToolStart":
        tc = ToolCall(
            name=data.get("tool_name", ""),
            tool_id=data.get("tool_id", ""),
            input=data.get("input", {}),
        )
        return ToolStart(tool_call=tc, timestamp=data.get("timestamp", 0.0))

    elif event_type == "ToolProgress":
        tc = ToolCall(
            name=data.get("tool_name", ""),
            tool_id=data.get("tool_id", ""),
            input={},
        )
        return ToolProgress(
            tool_call=tc,
            progress_percent=data.get("progress_percent", 0),
            message=data.get("message", ""),
            timestamp=data.get("timestamp", 0.0),
        )

    elif event_type == "ToolEnd":
        tc = ToolCall(
            name=data.get("tool_name", ""),
            tool_id=data.get("tool_id", ""),
            input={},
        )
        tr = ToolResult(
            tool_id=data.get("tool_id", ""),
            success=data.get("success", False),
            output="",
        )
        return ToolEnd(
            tool_call=tc,
            result=tr,
            duration_ms=data.get("duration_ms", 0.0),
            timestamp=data.get("timestamp", 0.0),
        )

    elif event_type == "StatusUpdate":
        return StatusUpdate(
            message=data.get("message", ""),
            severity=data.get("severity", "info"),
            timestamp=data.get("timestamp", 0.0),
        )

    raise ValueError(f"Unknown event type: {event_type}")


# ── Type guards ──────────────────────────────────────────────────

def is_text_chunk(event: StreamEvent) -> bool:
    return isinstance(event, TextChunk)

def is_tool_start(event: StreamEvent) -> bool:
    return isinstance(event, ToolStart)

def is_tool_progress(event: StreamEvent) -> bool:
    return isinstance(event, ToolProgress)

def is_tool_end(event: StreamEvent) -> bool:
    return isinstance(event, ToolEnd)

def is_status_update(event: StreamEvent) -> bool:
    return isinstance(event, StatusUpdate)
