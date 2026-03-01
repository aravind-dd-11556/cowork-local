"""
Cost Optimizer — Autonomous model selection based on task complexity and budget.

Integrates with ModelRouter and CostTracker to automatically downgrade to
cheaper models for simple tasks, enforce cost budgets, and provide real-time
cost estimates before execution.

Features:
  - Task-aware model selection (simple → FAST tier, complex → POWERFUL)
  - Cost estimation before execution (pre-flight cost check)
  - Automatic downgrade when budget is running low
  - Per-request cost tracking and rolling averages
  - Cost-aware caching decisions

Sprint 24: Production Hardening.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────

class CostDecision(Enum):
    """What the optimizer decided."""
    USE_CURRENT = "use_current"       # Keep current model
    DOWNGRADE = "downgrade"           # Switch to cheaper model
    UPGRADE = "upgrade"               # Switch to more powerful model
    BLOCK = "block"                   # Refuse — budget exceeded


@dataclass
class CostEstimate:
    """Pre-flight cost estimate for a request."""
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    model: str
    provider: str
    budget_remaining: Optional[float] = None
    budget_percentage_used: Optional[float] = None
    within_budget: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "model": self.model,
            "provider": self.provider,
            "budget_remaining": round(self.budget_remaining, 6) if self.budget_remaining is not None else None,
            "budget_percentage_used": round(self.budget_percentage_used, 2) if self.budget_percentage_used is not None else None,
            "within_budget": self.within_budget,
        }


@dataclass
class OptimizationResult:
    """Result of the cost optimizer's decision."""
    decision: CostDecision
    recommended_model: str
    recommended_provider: str
    original_model: str
    original_provider: str
    reasoning: str
    estimate: Optional[CostEstimate] = None
    savings_estimate_usd: float = 0.0

    @property
    def changed(self) -> bool:
        return self.recommended_model != self.original_model

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "recommended_model": self.recommended_model,
            "original_model": self.original_model,
            "changed": self.changed,
            "reasoning": self.reasoning,
            "savings_estimate_usd": round(self.savings_estimate_usd, 6),
            "estimate": self.estimate.to_dict() if self.estimate else None,
        }


# ── Model cost table ─────────────────────────────────────────────

@dataclass
class ModelCostInfo:
    """Cost info for a model, used for comparison."""
    provider: str
    model: str
    input_per_1k: float
    output_per_1k: float
    tier: str  # "fast", "balanced", "powerful"

    @property
    def avg_cost_per_1k(self) -> float:
        """Average of input + output cost per 1K tokens."""
        return (self.input_per_1k + self.output_per_1k) / 2


DEFAULT_MODEL_COSTS: List[ModelCostInfo] = [
    # Fast tier (cheapest)
    ModelCostInfo("anthropic", "claude-haiku-4-5-20251001", 0.0008, 0.004, "fast"),
    ModelCostInfo("openai", "gpt-4o-mini", 0.00015, 0.0006, "fast"),
    # Balanced tier
    ModelCostInfo("anthropic", "claude-sonnet-4-5-20250929", 0.003, 0.015, "balanced"),
    ModelCostInfo("openai", "gpt-4o", 0.0025, 0.010, "balanced"),
    # Powerful tier (most expensive)
    ModelCostInfo("anthropic", "claude-opus-4-5-20251101", 0.015, 0.075, "powerful"),
    # Local (free)
    ModelCostInfo("ollama", "llama3.2", 0.0, 0.0, "fast"),
]


# ── Complexity signals ───────────────────────────────────────────

# Simple tasks that can use FAST tier
_SIMPLE_PATTERNS = [
    r"^\s*(yes|no|ok|sure|thanks|hello|hi)\s*$",
    r"^\s*read\s+(this|the)\s+file",
    r"^\s*(list|show|display)\s+(files|directory)",
    r"^\s*what\s+(is|are)\s+",
    r"^\s*search\s+for\s+",
]

# Complex tasks that need POWERFUL tier
_COMPLEX_PATTERNS = [
    r"\bimplement\b.*\b(system|framework|module)\b",
    r"\brefactor\b.*\b(entire|all|codebase)\b",
    r"\barchitect\b",
    r"\bdesign\s+a\b.*\bsystem\b",
    r"\bmulti.step\b",
    r"\bcomprehensive\b.*\b(analysis|review|audit)\b",
]

import re
_simple_re = [re.compile(p, re.IGNORECASE) for p in _SIMPLE_PATTERNS]
_complex_re = [re.compile(p, re.IGNORECASE) for p in _COMPLEX_PATTERNS]


# ── CostOptimizer ────────────────────────────────────────────────

class CostOptimizer:
    """Autonomous model selection and cost optimization.

    Usage::

        optimizer = CostOptimizer(
            current_provider="anthropic",
            current_model="claude-sonnet-4-5-20250929",
            budget_limit=1.0,
        )
        result = optimizer.optimize("hello, what files are in this dir?")
        if result.changed:
            # switch to recommended model
    """

    def __init__(
        self,
        current_provider: str = "anthropic",
        current_model: str = "claude-sonnet-4-5-20250929",
        budget_limit: Optional[float] = None,
        budget_used: float = 0.0,
        model_costs: Optional[List[ModelCostInfo]] = None,
        auto_downgrade_threshold: float = 0.8,  # downgrade when 80% budget used
        enabled: bool = True,
    ):
        self._current_provider = current_provider
        self._current_model = current_model
        self._budget_limit = budget_limit
        self._budget_used = budget_used
        self._model_costs = model_costs or list(DEFAULT_MODEL_COSTS)
        self._auto_downgrade_threshold = auto_downgrade_threshold
        self._enabled = enabled

        # Build lookup
        self._cost_lookup: Dict[str, ModelCostInfo] = {
            f"{m.provider}:{m.model}": m for m in self._model_costs
        }

        # Stats
        self._total_optimizations = 0
        self._total_downgrades = 0
        self._total_upgrades = 0
        self._total_savings = 0.0
        # Rolling token averages
        self._recent_input_tokens: List[int] = []
        self._recent_output_tokens: List[int] = []
        self._ROLLING_WINDOW = 10

    # ── Core optimization ──────────────────────────────────────

    def optimize(
        self,
        user_input: str,
        tool_count: int = 0,
        message_count: int = 0,
        force_tier: Optional[str] = None,
    ) -> OptimizationResult:
        """Determine optimal model for this request.

        Args:
            user_input: The user's message
            tool_count: Number of tool calls in prior turn
            message_count: Conversation depth
            force_tier: Override tier selection ("fast", "balanced", "powerful")

        Returns:
            OptimizationResult with recommended model
        """
        self._total_optimizations += 1

        if not self._enabled:
            return OptimizationResult(
                decision=CostDecision.USE_CURRENT,
                recommended_model=self._current_model,
                recommended_provider=self._current_provider,
                original_model=self._current_model,
                original_provider=self._current_provider,
                reasoning="Optimizer disabled",
            )

        # 1. Check budget
        if self._budget_limit is not None:
            budget_pct = self._budget_used / self._budget_limit if self._budget_limit > 0 else 1.0
            if budget_pct >= 1.0:
                return OptimizationResult(
                    decision=CostDecision.BLOCK,
                    recommended_model=self._current_model,
                    recommended_provider=self._current_provider,
                    original_model=self._current_model,
                    original_provider=self._current_provider,
                    reasoning=f"Budget exhausted ({budget_pct:.0%} used)",
                )

        # 2. Determine ideal tier
        if force_tier:
            ideal_tier = force_tier
            reasoning = f"Forced tier: {force_tier}"
        else:
            ideal_tier, reasoning = self._classify_complexity(
                user_input, tool_count, message_count
            )

        # 3. Check if budget pressure forces downgrade
        if self._budget_limit is not None:
            budget_pct = self._budget_used / self._budget_limit if self._budget_limit > 0 else 0.0
            if budget_pct >= self._auto_downgrade_threshold and ideal_tier != "fast":
                ideal_tier = "fast"
                reasoning = f"Budget pressure ({budget_pct:.0%} used) — auto-downgrading to fast tier"

        # 4. Find best model for the tier
        recommended = self._find_model_for_tier(ideal_tier)
        if not recommended:
            return OptimizationResult(
                decision=CostDecision.USE_CURRENT,
                recommended_model=self._current_model,
                recommended_provider=self._current_provider,
                original_model=self._current_model,
                original_provider=self._current_provider,
                reasoning=f"No model available for tier '{ideal_tier}' — keeping current",
            )

        # 5. Calculate savings estimate
        current_info = self._cost_lookup.get(f"{self._current_provider}:{self._current_model}")
        savings = 0.0
        if current_info and recommended.avg_cost_per_1k < current_info.avg_cost_per_1k:
            avg_tokens = self._avg_tokens_per_request()
            savings = (current_info.avg_cost_per_1k - recommended.avg_cost_per_1k) * (avg_tokens / 1000)
            self._total_savings += savings

        # 6. Determine decision type
        if recommended.model == self._current_model and recommended.provider == self._current_provider:
            decision = CostDecision.USE_CURRENT
        elif recommended.avg_cost_per_1k < (current_info.avg_cost_per_1k if current_info else float('inf')):
            decision = CostDecision.DOWNGRADE
            self._total_downgrades += 1
        else:
            decision = CostDecision.UPGRADE
            self._total_upgrades += 1

        # 7. Build cost estimate
        estimate = self._estimate_cost(recommended)

        return OptimizationResult(
            decision=decision,
            recommended_model=recommended.model,
            recommended_provider=recommended.provider,
            original_model=self._current_model,
            original_provider=self._current_provider,
            reasoning=reasoning,
            estimate=estimate,
            savings_estimate_usd=savings,
        )

    # ── Complexity classification ──────────────────────────────

    def _classify_complexity(
        self, user_input: str, tool_count: int, message_count: int
    ) -> tuple[str, str]:
        """Classify task complexity into a tier. Returns (tier, reasoning)."""
        reasons: List[str] = []

        # Check simple patterns
        for pattern in _simple_re:
            if pattern.search(user_input):
                reasons.append("simple task pattern")
                return "fast", "; ".join(reasons)

        # Check complex patterns
        for pattern in _complex_re:
            if pattern.search(user_input):
                reasons.append("complex task pattern")
                return "powerful", "; ".join(reasons)

        # Input length heuristic
        input_len = len(user_input)
        if input_len < 30:
            reasons.append(f"very short input ({input_len} chars)")
            return "fast", "; ".join(reasons)
        elif input_len > 500:
            reasons.append(f"long input ({input_len} chars)")
            return "powerful", "; ".join(reasons)

        # Tool count from prior turn
        if tool_count >= 5:
            reasons.append(f"many tool calls ({tool_count})")
            return "powerful", "; ".join(reasons)

        # Deep conversation
        if message_count > 20:
            reasons.append(f"deep conversation ({message_count} msgs)")
            return "balanced", "; ".join(reasons)

        reasons.append("default classification")
        return "balanced", "; ".join(reasons)

    # ── Model selection ────────────────────────────────────────

    def _find_model_for_tier(self, tier: str) -> Optional[ModelCostInfo]:
        """Find the cheapest model for a tier, preferring current provider."""
        candidates = [m for m in self._model_costs if m.tier == tier]
        if not candidates:
            return None

        # Prefer current provider
        same_provider = [m for m in candidates if m.provider == self._current_provider]
        if same_provider:
            return min(same_provider, key=lambda m: m.avg_cost_per_1k)

        return min(candidates, key=lambda m: m.avg_cost_per_1k)

    # ── Cost estimation ────────────────────────────────────────

    def estimate_request_cost(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        estimated_tokens: Optional[int] = None,
    ) -> CostEstimate:
        """Estimate cost for a single request.

        Args:
            model: Model to estimate for (default: current)
            provider: Provider to estimate for (default: current)
            estimated_tokens: Override token estimate

        Returns:
            CostEstimate with predicted cost
        """
        model = model or self._current_model
        provider = provider or self._current_provider
        info = self._cost_lookup.get(f"{provider}:{model}")

        avg_input, avg_output = self._avg_tokens_split()
        if estimated_tokens:
            avg_input = int(estimated_tokens * 0.7)
            avg_output = int(estimated_tokens * 0.3)

        if not info:
            return CostEstimate(
                estimated_input_tokens=avg_input,
                estimated_output_tokens=avg_output,
                estimated_cost_usd=0.0,
                model=model,
                provider=provider,
            )

        cost = (avg_input / 1000) * info.input_per_1k + (avg_output / 1000) * info.output_per_1k

        budget_remaining = None
        budget_pct = None
        within = True
        if self._budget_limit is not None:
            budget_remaining = max(0, self._budget_limit - self._budget_used)
            budget_pct = (self._budget_used / self._budget_limit * 100) if self._budget_limit > 0 else 100.0
            within = (self._budget_used + cost) <= self._budget_limit

        return CostEstimate(
            estimated_input_tokens=avg_input,
            estimated_output_tokens=avg_output,
            estimated_cost_usd=cost,
            model=model,
            provider=provider,
            budget_remaining=budget_remaining,
            budget_percentage_used=budget_pct,
            within_budget=within,
        )

    def _estimate_cost(self, model_info: ModelCostInfo) -> CostEstimate:
        """Internal: estimate cost using model info."""
        return self.estimate_request_cost(
            model=model_info.model,
            provider=model_info.provider,
        )

    # ── Token tracking ─────────────────────────────────────────

    def record_usage(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Record actual token usage for improving future estimates."""
        self._recent_input_tokens.append(input_tokens)
        self._recent_output_tokens.append(output_tokens)
        # Keep rolling window
        if len(self._recent_input_tokens) > self._ROLLING_WINDOW:
            self._recent_input_tokens = self._recent_input_tokens[-self._ROLLING_WINDOW:]
        if len(self._recent_output_tokens) > self._ROLLING_WINDOW:
            self._recent_output_tokens = self._recent_output_tokens[-self._ROLLING_WINDOW:]
        self._budget_used += cost

    def _avg_tokens_per_request(self) -> int:
        """Average total tokens per request."""
        if not self._recent_input_tokens:
            return 2000  # default estimate
        avg_in = sum(self._recent_input_tokens) / len(self._recent_input_tokens)
        avg_out = sum(self._recent_output_tokens) / len(self._recent_output_tokens)
        return int(avg_in + avg_out)

    def _avg_tokens_split(self) -> tuple[int, int]:
        """Average (input, output) token split."""
        if not self._recent_input_tokens:
            return 1400, 600  # default 70/30 split
        avg_in = int(sum(self._recent_input_tokens) / len(self._recent_input_tokens))
        avg_out = int(sum(self._recent_output_tokens) / len(self._recent_output_tokens))
        return avg_in, avg_out

    # ── Budget management ──────────────────────────────────────

    def update_budget(self, budget_limit: Optional[float] = None, budget_used: Optional[float] = None) -> None:
        """Update budget parameters."""
        if budget_limit is not None:
            self._budget_limit = budget_limit
        if budget_used is not None:
            self._budget_used = budget_used

    @property
    def budget_status(self) -> Dict[str, Any]:
        """Current budget status."""
        if self._budget_limit is None:
            return {"has_budget": False}
        pct = (self._budget_used / self._budget_limit * 100) if self._budget_limit > 0 else 100.0
        return {
            "has_budget": True,
            "limit": self._budget_limit,
            "used": round(self._budget_used, 6),
            "remaining": round(max(0, self._budget_limit - self._budget_used), 6),
            "percentage_used": round(pct, 2),
            "over_budget": self._budget_used > self._budget_limit,
        }

    # ── Stats ──────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_optimizations": self._total_optimizations,
            "total_downgrades": self._total_downgrades,
            "total_upgrades": self._total_upgrades,
            "total_savings_usd": round(self._total_savings, 6),
            "avg_tokens_per_request": self._avg_tokens_per_request(),
        }
