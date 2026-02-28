"""
Prompt Budget Manager — Track and manage system prompt size during assembly.

Prevents system prompt bloat by allocating per-section token budgets,
compressing oversized sections, and enforcing priority ordering so
critical sections (behavioral rules, tools) always fit.

Sprint 15 (Prompt Optimization & Context Management) Module 2.
"""

from __future__ import annotations

import logging
from typing import Optional

from .token_estimator import ModelTokenEstimator, FALLBACK_RATIO

logger = logging.getLogger(__name__)


# Section priority (lower = higher priority = allocated first)
DEFAULT_SECTION_PRIORITIES: dict[str, int] = {
    "behavioral_rules": 1,
    "tools": 2,
    "memory": 3,
    "skills": 4,
    "env_context": 5,
}

# Default max percentage of total budget each section may use
DEFAULT_SECTION_LIMITS: dict[str, float] = {
    "behavioral_rules": 0.30,
    "tools": 0.40,
    "memory": 0.15,
    "skills": 0.10,
    "env_context": 0.05,
}


class PromptBudgetManager:
    """
    Tracks system prompt token budget during assembly.

    Usage:
        mgr = PromptBudgetManager(max_system_prompt_tokens=8000)
        content = mgr.allocate("behavioral_rules", rules_text, force=True)
        content = mgr.allocate("tools", tools_text)
        content = mgr.allocate("memory", memory_text)
        print(mgr.report())
    """

    def __init__(
        self,
        max_system_prompt_tokens: int = 8000,
        estimator: Optional[ModelTokenEstimator] = None,
        model: str = "claude",
        section_limits: Optional[dict[str, float]] = None,
    ):
        self.max_tokens = max_system_prompt_tokens
        self.estimator = estimator or ModelTokenEstimator()
        self.model = model

        # Section tracking: name → (content, token_count)
        self._sections: dict[str, tuple[str, int]] = {}

        # Configurable limits
        self._section_limits = dict(DEFAULT_SECTION_LIMITS)
        if section_limits:
            self._section_limits.update(section_limits)

        self._priorities = dict(DEFAULT_SECTION_PRIORITIES)

    @property
    def total_allocated(self) -> int:
        """Total tokens allocated across all sections."""
        return sum(tokens for _, tokens in self._sections.values())

    def remaining(self) -> int:
        """Tokens remaining in the total budget."""
        return max(0, self.max_tokens - self.total_allocated)

    def can_fit(self, additional_tokens: int) -> bool:
        """Check if additional content can fit in the remaining budget."""
        return additional_tokens <= self.remaining()

    def get_section_limit(self, section_name: str) -> int:
        """Get max token allocation for a section."""
        pct = self._section_limits.get(section_name, 0.10)  # Default 10%
        return int(self.max_tokens * pct)

    def allocate(
        self,
        section_name: str,
        content: str,
        force: bool = False,
    ) -> str:
        """
        Allocate budget for a prompt section.

        If the content fits within the section's limit and the total budget,
        returns the original content. Otherwise, compresses it.

        Args:
            section_name: Identifier for this section.
            content: The section content.
            force: If True, always allocate even if over budget (for critical sections).

        Returns:
            The content (original or compressed) that was allocated.
        """
        if not content:
            return ""

        tokens = self.estimator.estimate_tokens(content, self.model)
        section_limit = self.get_section_limit(section_name)

        # Check if content fits within section limit
        if tokens <= section_limit and self.can_fit(tokens):
            self._sections[section_name] = (content, tokens)
            return content

        # Content is too large — try compression
        if not force:
            target = min(section_limit, self.remaining())
            if target <= 0:
                logger.warning(
                    f"No budget remaining for section '{section_name}' "
                    f"({tokens} tokens requested, {self.remaining()} remaining)"
                )
                return ""
            compressed = self.compress(content, target)
            compressed_tokens = self.estimator.estimate_tokens(compressed, self.model)
            self._sections[section_name] = (compressed, compressed_tokens)
            logger.info(
                f"Compressed section '{section_name}': "
                f"{tokens} → {compressed_tokens} tokens"
            )
            return compressed

        # Force allocation (critical section)
        self._sections[section_name] = (content, tokens)
        if tokens > section_limit:
            logger.warning(
                f"Force-allocated section '{section_name}': "
                f"{tokens} tokens exceeds limit {section_limit}"
            )
        return content

    def compress(self, content: str, target_tokens: int) -> str:
        """
        Compress content to fit within a target token count.

        Strategy:
          - Truncate from the end, keeping a small tail for context.
          - Preserve the first portion (usually most important).
        """
        if target_tokens <= 0:
            return ""

        current_tokens = self.estimator.estimate_tokens(content, self.model)
        if current_tokens <= target_tokens:
            return content

        # Calculate approximate character budget
        ratio = self.estimator.get_ratio(self.model)
        target_chars = int(target_tokens * ratio)

        if target_chars <= 0:
            return ""

        # Keep first 80% and last 10% of target, with truncation notice
        head_chars = int(target_chars * 0.80)
        tail_chars = int(target_chars * 0.10)
        notice = f"\n... [compressed from {len(content)} chars] ...\n"

        if head_chars + tail_chars + len(notice) >= len(content):
            # Content is close to fitting — just truncate end
            return content[:target_chars]

        result = content[:head_chars] + notice + content[-tail_chars:]
        return result

    def report(self) -> dict:
        """
        Return a breakdown of budget usage by section.

        Returns:
            {
                "total_allocated": int,
                "max_tokens": int,
                "remaining": int,
                "percent_used": float,
                "sections": {
                    "section_name": {
                        "tokens": int,
                        "limit": int,
                        "percent_of_limit": float,
                        "over_limit": bool,
                    },
                    ...
                }
            }
        """
        sections_report = {}
        for name, (_, tokens) in self._sections.items():
            limit = self.get_section_limit(name)
            sections_report[name] = {
                "tokens": tokens,
                "limit": limit,
                "percent_of_limit": round(tokens / limit * 100, 1) if limit > 0 else 0,
                "over_limit": tokens > limit,
            }

        total = self.total_allocated
        return {
            "total_allocated": total,
            "max_tokens": self.max_tokens,
            "remaining": self.remaining(),
            "percent_used": round(total / self.max_tokens * 100, 1) if self.max_tokens > 0 else 0,
            "sections": sections_report,
        }

    def reset(self) -> None:
        """Clear all section allocations (for reuse across prompt builds)."""
        self._sections.clear()

    def to_dict(self) -> dict:
        """Serialize state for debugging."""
        return {
            "max_tokens": self.max_tokens,
            "model": self.model,
            "total_allocated": self.total_allocated,
            "remaining": self.remaining(),
            "sections": {
                name: {"tokens": tokens}
                for name, (_, tokens) in self._sections.items()
            },
        }
