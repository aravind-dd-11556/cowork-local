"""
Error Budget Tracker — Rate-based failure budget enforcement.

Tracks failure rates across categories (provider, tool, agent) and raises
alerts when the error rate exceeds a configured threshold. Complements
the token-based budget in token_tracker.py with rate-based budgeting.

Sprint 13 (Error Recovery & Resilience) Module 2.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ErrorBudgetConfig:
    """Configuration for error budget enforcement."""
    max_error_rate: float = 0.20  # 20% failure tolerance
    window_seconds: float = 300.0  # 5-minute sliding window


@dataclass
class _BudgetEntry:
    """A single success/failure record."""
    timestamp: float
    category: str
    success: bool


class ErrorBudgetTracker:
    """
    Track failure rates and enforce error budget limits.

    Records success/failure per category and calculates rolling error rates
    within a sliding time window. Raises alert when budget is exceeded.

    Usage:
        tracker = ErrorBudgetTracker()
        tracker.record("tool", success=True)
        tracker.record("tool", success=False)
        rate = tracker.get_error_rate("tool")
        if tracker.is_over_budget():
            ...
    """

    # Valid categories
    CATEGORIES = {"provider", "tool", "agent", "all"}

    def __init__(self, config: Optional[ErrorBudgetConfig] = None):
        self._config = config or ErrorBudgetConfig()
        self._entries: list[_BudgetEntry] = []

    @property
    def max_error_rate(self) -> float:
        return self._config.max_error_rate

    @property
    def window_seconds(self) -> float:
        return self._config.window_seconds

    def record(self, category: str, success: bool) -> None:
        """
        Record a success or failure event.

        Args:
            category: One of 'provider', 'tool', 'agent'.
            success: True for success, False for failure.
        """
        self._entries.append(_BudgetEntry(
            timestamp=time.time(),
            category=category,
            success=success,
        ))

    def get_error_rate(self, category: str = "all") -> float:
        """
        Get the current error rate for a category.

        Args:
            category: 'provider', 'tool', 'agent', or 'all'.

        Returns:
            Error rate as a float between 0.0 and 1.0.
            Returns 0.0 if no entries exist.
        """
        entries = self._window_entries(category)
        if not entries:
            return 0.0
        failures = sum(1 for e in entries if not e.success)
        return failures / len(entries)

    def is_over_budget(self, category: str = "all") -> bool:
        """Check if error rate exceeds the configured maximum."""
        return self.get_error_rate(category) > self._config.max_error_rate

    def get_remaining_budget(self, category: str = "all") -> float:
        """
        Get remaining error budget as a percentage.

        Returns:
            Value between 0.0 (budget exhausted) and 1.0 (no errors used).
        """
        rate = self.get_error_rate(category)
        if self._config.max_error_rate <= 0:
            return 0.0
        remaining = 1.0 - (rate / self._config.max_error_rate)
        return max(0.0, min(1.0, remaining))

    def get_report(self) -> dict:
        """
        Get a full report of error budget status across all categories.

        Returns dict with per-category error rates and overall status.
        """
        now = time.time()
        cutoff = now - self._config.window_seconds

        # Collect unique categories from entries
        categories_seen = set()
        for e in self._entries:
            if e.timestamp >= cutoff:
                categories_seen.add(e.category)

        report: dict = {
            "window_seconds": self._config.window_seconds,
            "max_error_rate": self._config.max_error_rate,
            "overall": {
                "error_rate": self.get_error_rate("all"),
                "over_budget": self.is_over_budget("all"),
                "remaining": self.get_remaining_budget("all"),
            },
            "categories": {},
        }

        for cat in sorted(categories_seen):
            entries = self._window_entries(cat)
            total = len(entries)
            failures = sum(1 for e in entries if not e.success)
            report["categories"][cat] = {
                "total": total,
                "failures": failures,
                "error_rate": failures / total if total > 0 else 0.0,
                "over_budget": self.get_error_rate(cat) > self._config.max_error_rate,
            }

        return report

    def reset(self) -> None:
        """Clear all recorded entries."""
        self._entries.clear()

    def _window_entries(self, category: str = "all") -> list[_BudgetEntry]:
        """Get entries within the current time window for a category."""
        now = time.time()
        cutoff = now - self._config.window_seconds

        if category == "all":
            return [e for e in self._entries if e.timestamp >= cutoff]
        return [
            e for e in self._entries
            if e.timestamp >= cutoff and e.category == category
        ]
