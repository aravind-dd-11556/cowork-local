"""
Token Tracker — Per-call and session-level token accounting.

Tracks input/output/cache tokens for every provider call, accumulates
session totals, and enforces configurable budget caps (max tokens and
max estimated cost in USD).

Sprint 4 (P2-Advanced) Feature 1.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token counts for a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    provider: str = ""
    model: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "provider": self.provider,
            "model": self.model,
            "timestamp": self.timestamp,
        }


# ── Cost-per-million-token rates (USD) ──
# Updated as of 2025.  Add new models here as needed.
MODEL_COSTS: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-5-20251101":     {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-5-20250929":   {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5-20251001":    {"input": 0.80,  "output": 4.0},
    # OpenAI
    "gpt-4o":                       {"input": 2.50,  "output": 10.0},
    "gpt-4o-mini":                  {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":                  {"input": 10.0,  "output": 30.0},
    # Ollama (local — zero cost)
    "__ollama_default__":           {"input": 0.0,   "output": 0.0},
}


class BudgetExceededError(Exception):
    """Raised when a budget cap (tokens or cost) is hit."""
    pass


class TokenTracker:
    """
    Session-level token tracker with optional budget enforcement.

    Usage:
        tracker = TokenTracker(max_session_tokens=500_000, max_cost_usd=5.0)
        tracker.record(usage)          # after each LLM call
        tracker.check_budget()         # raises BudgetExceededError if over
        summary = tracker.summary()    # dict for LLM context / UI
    """

    def __init__(
        self,
        max_session_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
    ):
        self.max_session_tokens = max_session_tokens
        self.max_cost_usd = max_cost_usd

        # Per-call history
        self._calls: list[TokenUsage] = []

        # Running totals
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cache_read_tokens: int = 0
        self.total_cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def estimated_cost_usd(self) -> float:
        """Estimate cumulative cost across all recorded calls."""
        total = 0.0
        for usage in self._calls:
            total += self._estimate_call_cost(usage)
        return round(total, 6)

    def record(self, usage: TokenUsage) -> None:
        """Record a single LLM call's token usage."""
        self._calls.append(usage)
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cache_read_tokens += usage.cache_read_tokens
        self.total_cache_write_tokens += usage.cache_write_tokens

        logger.debug(
            f"Token usage: in={usage.input_tokens} out={usage.output_tokens} "
            f"(session total: {self.total_tokens})"
        )

    def check_budget(self) -> None:
        """
        Check if session budget has been exceeded.
        Raises BudgetExceededError if so.
        """
        if self.max_session_tokens and self.total_tokens > self.max_session_tokens:
            raise BudgetExceededError(
                f"Session token budget exceeded: {self.total_tokens} > "
                f"{self.max_session_tokens} max tokens"
            )

        if self.max_cost_usd is not None and self.estimated_cost_usd > self.max_cost_usd:
            raise BudgetExceededError(
                f"Session cost budget exceeded: ${self.estimated_cost_usd:.4f} > "
                f"${self.max_cost_usd:.2f} max"
            )

    def summary(self) -> dict:
        """
        Return a summary dict suitable for injection into the LLM context
        or display in the UI.
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.total_cache_read_tokens,
            "cache_write_tokens": self.total_cache_write_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "call_count": self.call_count,
            "budget": {
                "max_session_tokens": self.max_session_tokens,
                "max_cost_usd": self.max_cost_usd,
                "tokens_remaining": (
                    (self.max_session_tokens - self.total_tokens)
                    if self.max_session_tokens else None
                ),
                "cost_remaining_usd": (
                    round(self.max_cost_usd - self.estimated_cost_usd, 4)
                    if self.max_cost_usd is not None else None
                ),
            },
        }

    def reset(self) -> None:
        """Reset all counters (e.g. new session)."""
        self._calls.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0

    # ── internals ──

    @staticmethod
    def _estimate_call_cost(usage: TokenUsage) -> float:
        """Estimate USD cost for a single call."""
        model = usage.model
        # Try exact model match, then prefix match, then provider default
        rates = MODEL_COSTS.get(model)
        if not rates:
            # Prefix match (e.g. "gpt-4o-2024-..." → "gpt-4o")
            for key in MODEL_COSTS:
                if model.startswith(key):
                    rates = MODEL_COSTS[key]
                    break
        if not rates:
            # Provider-level default
            if usage.provider.lower() == "ollama" or usage.provider.lower() == "ollamaprovider":
                rates = MODEL_COSTS["__ollama_default__"]
            else:
                # Unknown model — assume moderate pricing
                rates = {"input": 3.0, "output": 15.0}

        input_cost = (usage.input_tokens / 1_000_000) * rates["input"]
        output_cost = (usage.output_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost
