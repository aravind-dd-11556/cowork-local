"""
Error Context — Rich error metadata and breadcrumb tracking.

Enhances AgentError with execution context (where, when, what was attempted)
and provides formatting for both LLM consumption and structured logging.

Sprint 13 (Error Recovery & Resilience) Module 1.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .error_catalog import AgentError, ErrorCatalog, ErrorCode

logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Execution context captured at the point of error."""
    iteration: int = 0
    tool_name: str = ""
    provider_name: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    breadcrumbs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "tool_name": self.tool_name,
            "provider_name": self.provider_name,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "breadcrumbs": list(self.breadcrumbs),
        }


class ErrorContextEnricher:
    """
    Enriches exceptions with structured context and breadcrumbs.

    Usage:
        enricher = ErrorContextEnricher()
        ctx = ErrorContext(iteration=3, tool_name="bash", provider_name="ollama")
        agent_error = enricher.enrich(exception, ctx)
        print(enricher.format_for_llm(agent_error))
    """

    def enrich(self, exception: Exception, context: ErrorContext) -> AgentError:
        """
        Classify an exception and attach rich execution context.

        Args:
            exception: The original exception.
            context: Execution context at the point of failure.

        Returns:
            AgentError with merged context dict.
        """
        agent_error = ErrorCatalog.classify_error(exception)

        # Merge execution context into AgentError.context
        agent_error.context.update(context.to_dict())

        return agent_error

    def enrich_from_code(self, code: ErrorCode, context: ErrorContext,
                         exception: Optional[Exception] = None) -> AgentError:
        """Create an AgentError from a known code with context."""
        agent_error = ErrorCatalog.from_code(code, exception=exception)
        agent_error.context.update(context.to_dict())
        return agent_error

    @staticmethod
    def add_breadcrumb(agent_error: AgentError, breadcrumb: str) -> None:
        """Append a breadcrumb string to an AgentError's context."""
        if "breadcrumbs" not in agent_error.context:
            agent_error.context["breadcrumbs"] = []
        agent_error.context["breadcrumbs"].append(breadcrumb)

    @staticmethod
    def get_breadcrumbs(agent_error: AgentError) -> list[str]:
        """Get breadcrumbs from an AgentError."""
        return agent_error.context.get("breadcrumbs", [])

    @staticmethod
    def format_for_llm(agent_error: AgentError) -> str:
        """
        Format an AgentError for inclusion in LLM context.

        Returns a concise, human-readable string the LLM can use to
        understand what went wrong and decide next steps.
        """
        parts = [f"Error [{agent_error.code.value}]: {agent_error.message}"]

        ctx = agent_error.context
        if ctx.get("tool_name"):
            parts.append(f"Tool: {ctx['tool_name']}")
        if ctx.get("provider_name"):
            parts.append(f"Provider: {ctx['provider_name']}")
        if ctx.get("iteration"):
            parts.append(f"Iteration: {ctx['iteration']}")

        if agent_error.recovery_hint:
            parts.append(f"Suggestion: {agent_error.recovery_hint}")

        if agent_error.is_transient:
            parts.append("(This error is transient and may resolve on retry.)")

        breadcrumbs = ctx.get("breadcrumbs", [])
        if breadcrumbs:
            parts.append(f"Path: {' → '.join(breadcrumbs)}")

        return "\n".join(parts)

    @staticmethod
    def format_for_log(agent_error: AgentError) -> str:
        """
        Format an AgentError for structured logging.

        Returns a single-line string with key=value pairs.
        """
        ctx = agent_error.context
        parts = [
            f"code={agent_error.code.value}",
            f"category={agent_error.category.value}",
            f"transient={agent_error.is_transient}",
        ]

        if ctx.get("tool_name"):
            parts.append(f"tool={ctx['tool_name']}")
        if ctx.get("provider_name"):
            parts.append(f"provider={ctx['provider_name']}")
        if ctx.get("iteration"):
            parts.append(f"iteration={ctx['iteration']}")
        if ctx.get("duration_ms"):
            parts.append(f"duration_ms={ctx['duration_ms']:.1f}")

        parts.append(f"msg={agent_error.message}")

        return " ".join(parts)
