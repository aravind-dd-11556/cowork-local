"""
Cost Tracker — Estimates and tracks LLM API costs per session.

Maintains a pricing table for common models and calculates costs from
token usage data. Supports session budget limits with alerts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelPricing:
    """Pricing per 1K tokens for a specific model."""
    provider: str
    model: str
    input_cost_per_1k: float    # USD per 1K input tokens
    output_cost_per_1k: float   # USD per 1K output tokens
    cache_read_discount: float = 0.9   # 90% discount on cache reads
    cache_write_cost_per_1k: float = 0.0  # extra cost for cache writes (usually 0)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_cost_per_1k": self.input_cost_per_1k,
            "output_cost_per_1k": self.output_cost_per_1k,
        }


@dataclass
class CostRecord:
    """A single cost entry from one API call."""
    timestamp: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_cost: float = 0.0
    cache_savings: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost": round(self.estimated_cost, 6),
            "cache_savings": round(self.cache_savings, 6),
        }


# ── Default pricing table (Jan 2025 approximate rates) ───────────

DEFAULT_PRICING: dict[str, ModelPricing] = {
    # Anthropic
    "claude-sonnet-4-5-20250929": ModelPricing(
        "anthropic", "claude-sonnet-4-5-20250929",
        input_cost_per_1k=0.003, output_cost_per_1k=0.015,
        cache_read_discount=0.9,
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        "anthropic", "claude-haiku-4-5-20251001",
        input_cost_per_1k=0.0008, output_cost_per_1k=0.004,
        cache_read_discount=0.9,
    ),
    "claude-opus-4-5-20251101": ModelPricing(
        "anthropic", "claude-opus-4-5-20251101",
        input_cost_per_1k=0.015, output_cost_per_1k=0.075,
        cache_read_discount=0.9,
    ),
    # OpenAI
    "gpt-4o": ModelPricing(
        "openai", "gpt-4o",
        input_cost_per_1k=0.0025, output_cost_per_1k=0.010,
    ),
    "gpt-4o-mini": ModelPricing(
        "openai", "gpt-4o-mini",
        input_cost_per_1k=0.00015, output_cost_per_1k=0.0006,
    ),
    # Ollama (local — zero cost)
    "__ollama_default__": ModelPricing(
        "ollama", "*", input_cost_per_1k=0.0, output_cost_per_1k=0.0,
    ),
}


class BudgetExceededError(Exception):
    """Raised when session cost exceeds the budget limit."""

    def __init__(self, total_cost: float, budget: float):
        self.total_cost = total_cost
        self.budget = budget
        super().__init__(
            f"Session cost ${total_cost:.4f} exceeds budget ${budget:.4f}"
        )


class CostTracker:
    """
    Tracks estimated API costs per session.

    Calculates costs from token usage, maintains a running total,
    and enforces optional budget limits.
    """

    def __init__(
        self,
        budget_limit: Optional[float] = None,
        pricing_table: Optional[dict[str, ModelPricing]] = None,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._budget_limit = budget_limit
        self._pricing = dict(DEFAULT_PRICING)
        if pricing_table:
            self._pricing.update(pricing_table)
        self._records: list[CostRecord] = []
        self._total_cost: float = 0.0
        self._total_cache_savings: float = 0.0

    # ── Pricing lookup ────────────────────────────────────────────

    def get_pricing(self, model: str, provider: str = "") -> Optional[ModelPricing]:
        """
        Look up pricing for a model.

        Tries exact model match first, then falls back to provider default.
        """
        if model in self._pricing:
            return self._pricing[model]
        # Check for Ollama (all local models are free)
        if provider == "ollama" or provider.lower().startswith("ollama"):
            return self._pricing.get("__ollama_default__")
        return None

    def add_pricing(self, model: str, pricing: ModelPricing) -> None:
        """Add or update pricing for a model."""
        self._pricing[model] = pricing

    # ── Recording ─────────────────────────────────────────────────

    def record(
        self,
        usage: dict,
        provider: str,
        model: str,
    ) -> CostRecord:
        """
        Calculate and record cost from a provider's usage dict.

        Args:
            usage: Token counts from AgentResponse.usage
                   Keys: input_tokens, output_tokens,
                         cache_read_input_tokens, cache_creation_input_tokens
            provider: Provider name (e.g., "AnthropicProvider")
            model: Model identifier

        Returns:
            CostRecord with estimated cost
        """
        if not self._enabled or not usage:
            return CostRecord(
                timestamp=time.time(),
                provider=provider,
                model=model,
                input_tokens=0,
                output_tokens=0,
            )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)

        pricing = self.get_pricing(model, provider)
        if not pricing:
            # Unknown model — record tokens but zero cost
            record = CostRecord(
                timestamp=time.time(),
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                estimated_cost=0.0,
                cache_savings=0.0,
            )
            self._records.append(record)
            return record

        # Calculate costs
        # Regular input tokens (excluding cache reads)
        regular_input = max(0, input_tokens - cache_read)
        input_cost = (regular_input / 1000) * pricing.input_cost_per_1k

        # Cache read tokens (discounted)
        cache_cost = (cache_read / 1000) * pricing.input_cost_per_1k * (1 - pricing.cache_read_discount)

        # Output cost
        output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k

        # Cache write cost (if any)
        write_cost = (cache_write / 1000) * pricing.cache_write_cost_per_1k

        total = input_cost + cache_cost + output_cost + write_cost

        # Calculate savings from cache
        full_cache_cost = (cache_read / 1000) * pricing.input_cost_per_1k
        savings = full_cache_cost - cache_cost

        record = CostRecord(
            timestamp=time.time(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            estimated_cost=total,
            cache_savings=savings,
        )
        self._records.append(record)
        self._total_cost += total
        self._total_cache_savings += savings
        return record

    # ── Budget ────────────────────────────────────────────────────

    def check_budget(self) -> None:
        """Raise BudgetExceededError if over budget."""
        if self._budget_limit is not None and self._total_cost > self._budget_limit:
            raise BudgetExceededError(self._total_cost, self._budget_limit)

    @property
    def total_cost(self) -> float:
        return round(self._total_cost, 6)

    @property
    def total_cache_savings(self) -> float:
        return round(self._total_cache_savings, 6)

    def remaining_budget(self) -> Optional[float]:
        """Remaining budget, or None if no limit set."""
        if self._budget_limit is None:
            return None
        return round(max(0, self._budget_limit - self._total_cost), 6)

    def is_over_budget(self) -> bool:
        if self._budget_limit is None:
            return False
        return self._total_cost > self._budget_limit

    @property
    def budget_limit(self) -> Optional[float]:
        return self._budget_limit

    # ── Querying ──────────────────────────────────────────────────

    def get_history(self, last_n: int = 0) -> list[CostRecord]:
        """Get cost history. If last_n > 0, return only the last N records."""
        if last_n > 0:
            return list(self._records[-last_n:])
        return list(self._records)

    @property
    def call_count(self) -> int:
        return len(self._records)

    def per_model_breakdown(self) -> dict[str, dict]:
        """Aggregate costs per model."""
        breakdown: dict[str, dict] = {}
        for r in self._records:
            key = r.model
            if key not in breakdown:
                breakdown[key] = {
                    "model": r.model,
                    "provider": r.provider,
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost": 0.0,
                    "cache_savings": 0.0,
                }
            entry = breakdown[key]
            entry["calls"] += 1
            entry["input_tokens"] += r.input_tokens
            entry["output_tokens"] += r.output_tokens
            entry["total_cost"] += r.estimated_cost
            entry["cache_savings"] += r.cache_savings
        # Round floats
        for entry in breakdown.values():
            entry["total_cost"] = round(entry["total_cost"], 6)
            entry["cache_savings"] = round(entry["cache_savings"], 6)
        return breakdown

    def summary(self) -> dict:
        """Comprehensive cost summary for the session."""
        total_input = sum(r.input_tokens for r in self._records)
        total_output = sum(r.output_tokens for r in self._records)
        return {
            "enabled": self._enabled,
            "total_cost": self.total_cost,
            "total_cache_savings": self.total_cache_savings,
            "budget_limit": self._budget_limit,
            "remaining_budget": self.remaining_budget(),
            "is_over_budget": self.is_over_budget(),
            "call_count": self.call_count,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "per_model": self.per_model_breakdown(),
        }

    def reset(self) -> None:
        """Clear all cost records."""
        self._records.clear()
        self._total_cost = 0.0
        self._total_cache_savings = 0.0
