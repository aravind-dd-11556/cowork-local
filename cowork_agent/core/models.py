"""
Universal data models for the agent framework.
These are provider-agnostic — each provider converts to/from its native format.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import time
import uuid


@dataclass
class ToolSchema:
    """Universal tool definition for LLM consumption."""
    name: str
    description: str
    input_schema: dict  # JSON Schema format

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""
    name: str
    tool_id: str
    input: dict

    @staticmethod
    def generate_id() -> str:
        return f"tool_{uuid.uuid4().hex[:12]}"


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_id: str
    success: bool
    output: str
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    # Sprint 23: Trust context — tracks content origin and trust level
    trust_context: Optional[Any] = None  # TrustContext instance (optional import)


@dataclass
class Message:
    """A single message in the conversation history."""
    role: Literal["user", "assistant", "tool_result"]
    content: str
    tool_calls: Optional[list[ToolCall]] = None
    tool_results: Optional[list[ToolResult]] = None
    timestamp: float = field(default_factory=time.time)
    # Sprint 11: Advanced Memory System
    importance_score: Optional[float] = None  # 0.0–1.0 weight for pruning
    memory_id: Optional[str] = None           # Unique ID for cross-session tracking
    # Sprint 23: Trust context — tracks content origin and trust level
    trust_context: Optional[Any] = None  # TrustContext instance (optional import)


@dataclass
class AgentResponse:
    """Response from the LLM — either final text or tool calls."""
    text: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "error"] = "end_turn"
    raw_response: Optional[Any] = None
    # Sprint 4: Token usage metadata (populated by providers that report it)
    usage: Optional[dict] = None  # {"input_tokens": N, "output_tokens": N, ...}

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
