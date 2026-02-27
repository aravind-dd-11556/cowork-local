"""
Model Router — Task-aware routing to the optimal model tier.

Classifies user requests by complexity and routes them to the appropriate
model tier (FAST, BALANCED, POWERFUL), with automatic escalation on failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .models import AgentResponse, ToolCall


class ModelTier(Enum):
    """Model capability tiers — maps to cheap/mid/expensive models."""
    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"


@dataclass
class TaskClassification:
    """Result of classifying a user request."""
    tier: ModelTier
    reasoning: str
    confidence: float  # 0.0 – 1.0

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }


@dataclass
class TierConfig:
    """Maps a tier to a specific provider + model."""
    provider: str
    model: str
    priority: int = 0  # lower = preferred when multiple options exist

    def to_dict(self) -> dict:
        return {"provider": self.provider, "model": self.model, "priority": self.priority}


# ── Heuristic keywords and patterns ──────────────────────────────

_POWERFUL_KEYWORDS = re.compile(
    r"\b("
    r"implement|architect|design|refactor|debug|optimize|analyse|analyze|"
    r"create\s+(a\s+)?(class|module|system|framework|api|service)|"
    r"write\s+(a\s+)?(function|script|program|test|spec)|"
    r"plan\s+mode|enter\s+plan|"
    r"explain\s+(how|why|the\s+difference)|"
    r"compare\s+and\s+contrast|"
    r"security\s+audit|code\s+review|"
    r"multi[- ]step|complex|comprehensive"
    r")\b",
    re.IGNORECASE,
)

_FAST_KEYWORDS = re.compile(
    r"\b("
    r"read\s+(this|the|a)\s+file|"
    r"list\s+files|show\s+files|"
    r"what\s+is\s+in|cat\s+|ls\s+|"
    r"search\s+for|find\s+|grep\s+|"
    r"status|hello|hi|thanks|yes|no|ok"
    r")\b",
    re.IGNORECASE,
)


class ModelRouter:
    """
    Routes requests to the optimal model based on task complexity.

    Maintains a tier→config mapping and classifies incoming requests
    using heuristics (input length, keywords, tool call patterns).
    """

    # Escalation chain
    _ESCALATION_ORDER = [ModelTier.FAST, ModelTier.BALANCED, ModelTier.POWERFUL]

    def __init__(
        self,
        tier_configs: Optional[dict[ModelTier, TierConfig]] = None,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._tier_configs: dict[ModelTier, TierConfig] = tier_configs or {}
        self._escalation_count = 0

    # ── Classification ────────────────────────────────────────────

    def classify(
        self,
        user_input: str,
        tool_calls: Optional[list[ToolCall]] = None,
        message_count: int = 0,
    ) -> TaskClassification:
        """
        Classify a task into a model tier using heuristics.

        Factors:
            - Input length (short → FAST, long → POWERFUL)
            - Keyword matching (complexity indicators)
            - Tool call count from prior turn
            - Conversation depth
        """
        if not self._enabled:
            return TaskClassification(
                tier=ModelTier.BALANCED,
                reasoning="Router disabled — using default tier",
                confidence=1.0,
            )

        score = 0.0  # negative → FAST, positive → POWERFUL
        reasons: list[str] = []

        # 1. Input length
        input_len = len(user_input)
        if input_len < 40:
            score -= 1.5
            reasons.append(f"short input ({input_len} chars)")
        elif input_len > 300:
            score += 1.5
            reasons.append(f"long input ({input_len} chars)")
        elif input_len > 150:
            score += 0.5
            reasons.append(f"moderate input ({input_len} chars)")

        # 2. Keyword matching
        powerful_matches = _POWERFUL_KEYWORDS.findall(user_input)
        if powerful_matches:
            score += 2.0
            reasons.append(f"complex keywords: {powerful_matches[:3]}")

        fast_matches = _FAST_KEYWORDS.findall(user_input)
        if fast_matches and not powerful_matches:
            score -= 2.0
            reasons.append(f"simple keywords: {fast_matches[:3]}")

        # 3. Tool call history (many tools → complex task)
        if tool_calls:
            tc_count = len(tool_calls)
            if tc_count >= 4:
                score += 1.5
                reasons.append(f"{tc_count} tool calls in prior turn")
            elif tc_count >= 2:
                score += 0.5

        # 4. Conversation depth (deeper conversations tend to be more complex)
        if message_count > 20:
            score += 1.0
            reasons.append(f"deep conversation ({message_count} messages)")
        elif message_count > 10:
            score += 0.5

        # 5. Code indicators
        if re.search(r'```|def\s+\w+|class\s+\w+|import\s+\w+', user_input):
            score += 1.0
            reasons.append("code detected in input")

        # Map score to tier
        if score <= -1.5:
            tier = ModelTier.FAST
            confidence = min(abs(score) / 4.0, 1.0)
        elif score >= 2.0:
            tier = ModelTier.POWERFUL
            confidence = min(score / 5.0, 1.0)
        else:
            tier = ModelTier.BALANCED
            confidence = 0.6  # default tier, moderate confidence

        return TaskClassification(
            tier=tier,
            reasoning="; ".join(reasons) if reasons else "default classification",
            confidence=round(confidence, 2),
        )

    # ── Provider lookup ───────────────────────────────────────────

    def get_config_for_tier(self, tier: ModelTier) -> Optional[TierConfig]:
        """Get the provider+model config for a given tier."""
        return self._tier_configs.get(tier)

    def set_tier_config(self, tier: ModelTier, config: TierConfig) -> None:
        """Set or update the config for a tier."""
        self._tier_configs[tier] = config

    def has_tier(self, tier: ModelTier) -> bool:
        """Check if a tier has a configured provider."""
        return tier in self._tier_configs

    # ── Escalation ────────────────────────────────────────────────

    def should_escalate(
        self,
        response: AgentResponse,
        current_tier: ModelTier,
        attempt: int = 1,
    ) -> Optional[ModelTier]:
        """
        Determine if the response quality warrants escalation to a higher tier.

        Returns the next tier to try, or None if already at maximum / response OK.
        """
        if not self._enabled:
            return None

        needs_escalation = False
        reason = ""

        # Empty or error response
        if response.stop_reason == "error":
            needs_escalation = True
            reason = "error response"
        elif response.stop_reason == "max_tokens" and not response.has_tool_calls:
            needs_escalation = True
            reason = "truncated without tool calls"
        elif not response.text and not response.has_tool_calls:
            needs_escalation = True
            reason = "empty response"

        # Very short text response on complex query might need escalation
        if response.text and len(response.text) < 20 and attempt > 1:
            needs_escalation = True
            reason = "very short response on retry"

        if not needs_escalation:
            return None

        # Find next tier up
        try:
            current_idx = self._ESCALATION_ORDER.index(current_tier)
        except ValueError:
            return None

        if current_idx >= len(self._ESCALATION_ORDER) - 1:
            return None  # already at POWERFUL

        next_tier = self._ESCALATION_ORDER[current_idx + 1]

        # Only escalate if we have a config for the next tier
        if next_tier not in self._tier_configs:
            return None

        self._escalation_count += 1
        return next_tier

    # ── Stats ─────────────────────────────────────────────────────

    @property
    def escalation_count(self) -> int:
        return self._escalation_count

    def summary(self) -> dict:
        """Return router configuration summary."""
        return {
            "enabled": self._enabled,
            "escalation_count": self._escalation_count,
            "configured_tiers": {
                t.value: c.to_dict() for t, c in self._tier_configs.items()
            },
        }
