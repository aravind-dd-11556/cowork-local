"""
Error Aggregator — Pattern detection for error events.

Collects error events and detects patterns:
  - Spike: Error rate increases >300% in the last window
  - Recurring: Same error code appears >5 times in 10 minutes
  - Correlated: Multiple providers fail within 30 seconds

Sprint 13 (Error Recovery & Resilience) Module 4.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .error_catalog import AgentError, ErrorCode, ErrorCategory

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of error patterns detected."""
    SPIKE = "spike"              # Sudden increase in error rate
    RECURRING = "recurring"      # Same error repeated many times
    CORRELATED = "correlated"    # Multiple providers failing together


@dataclass
class ErrorEvent:
    """A single recorded error event."""
    timestamp: float
    error_code: ErrorCode
    category: ErrorCategory
    tool_name: str = ""
    provider_name: str = ""
    is_transient: bool = False

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "error_code": self.error_code.value,
            "category": self.category.value,
            "tool_name": self.tool_name,
            "provider_name": self.provider_name,
            "is_transient": self.is_transient,
        }


@dataclass
class ErrorPattern:
    """A detected error pattern."""
    pattern_type: PatternType
    count: int
    first_seen: float
    last_seen: float
    severity: str = "medium"  # low, medium, high
    details: str = ""
    affected_codes: list[str] = field(default_factory=list)
    affected_providers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type.value,
            "count": self.count,
            "severity": self.severity,
            "details": self.details,
            "affected_codes": self.affected_codes,
            "affected_providers": self.affected_providers,
        }


class ErrorAggregator:
    """
    Collects error events and detects patterns.

    Usage:
        agg = ErrorAggregator()
        agg.record_error(agent_error, tool_name="bash")
        patterns = agg.detect_patterns()
    """

    def __init__(self, window_seconds: float = 300.0,
                 spike_threshold: float = 3.0,
                 recurring_threshold: int = 5,
                 correlation_window: float = 30.0):
        self._window_seconds = window_seconds
        self._spike_threshold = spike_threshold       # 300% increase
        self._recurring_threshold = recurring_threshold  # N occurrences
        self._correlation_window = correlation_window  # seconds
        self._events: list[ErrorEvent] = []

    @property
    def event_count(self) -> int:
        return len(self._events)

    def record_error(self, agent_error: AgentError,
                     tool_name: str = "", provider_name: str = "") -> None:
        """Record an error event from an AgentError."""
        event = ErrorEvent(
            timestamp=time.time(),
            error_code=agent_error.code,
            category=agent_error.category,
            tool_name=tool_name,
            provider_name=provider_name,
            is_transient=agent_error.is_transient,
        )
        self._events.append(event)
        logger.debug(
            f"Error recorded: {agent_error.code.value} "
            f"tool={tool_name} provider={provider_name}"
        )

    def detect_patterns(self) -> list[ErrorPattern]:
        """
        Analyze recent events and detect patterns.

        Returns a list of detected patterns (may be empty).
        """
        patterns: list[ErrorPattern] = []
        now = time.time()

        recent = [e for e in self._events if now - e.timestamp <= self._window_seconds]
        if not recent:
            return patterns

        # 1. Spike detection
        spike = self._detect_spike(recent, now)
        if spike:
            patterns.append(spike)

        # 2. Recurring error detection
        recurring = self._detect_recurring(recent)
        patterns.extend(recurring)

        # 3. Correlated failure detection
        correlated = self._detect_correlated(recent)
        if correlated:
            patterns.append(correlated)

        return patterns

    def get_error_rate(self, category: str = "all") -> float:
        """Get the number of errors per minute in the current window."""
        now = time.time()
        recent = [e for e in self._events if now - e.timestamp <= self._window_seconds]
        if category != "all":
            recent = [e for e in recent if e.category.value == category]
        if not recent or self._window_seconds <= 0:
            return 0.0
        minutes = self._window_seconds / 60.0
        return len(recent) / minutes

    def get_window_summary(self, window_seconds: Optional[float] = None) -> dict:
        """Get a summary of errors in the current window."""
        window = window_seconds or self._window_seconds
        now = time.time()
        recent = [e for e in self._events if now - e.timestamp <= window]

        # Count by category
        by_category: dict[str, int] = {}
        by_code: dict[str, int] = {}
        by_provider: dict[str, int] = {}

        for e in recent:
            by_category[e.category.value] = by_category.get(e.category.value, 0) + 1
            by_code[e.error_code.value] = by_code.get(e.error_code.value, 0) + 1
            if e.provider_name:
                by_provider[e.provider_name] = by_provider.get(e.provider_name, 0) + 1

        return {
            "window_seconds": window,
            "total_errors": len(recent),
            "errors_per_minute": len(recent) / (window / 60.0) if window > 0 else 0,
            "by_category": by_category,
            "by_code": by_code,
            "by_provider": by_provider,
            "transient_count": sum(1 for e in recent if e.is_transient),
            "permanent_count": sum(1 for e in recent if not e.is_transient),
        }

    def cleanup_old(self, older_than: float = 3600.0) -> int:
        """Remove events older than the specified age. Returns count removed."""
        cutoff = time.time() - older_than
        before = len(self._events)
        self._events = [e for e in self._events if e.timestamp >= cutoff]
        removed = before - len(self._events)
        if removed:
            logger.debug(f"Cleaned up {removed} old error events")
        return removed

    def reset(self) -> None:
        """Clear all recorded events."""
        self._events.clear()

    # ── Pattern detection internals ──────────────────────────────

    def _detect_spike(self, recent: list[ErrorEvent], now: float) -> Optional[ErrorPattern]:
        """
        Detect spike: compare recent half vs previous half of window.

        A spike is detected when the second half has >threshold× the error
        rate of the first half.
        """
        half = self._window_seconds / 2.0
        midpoint = now - half

        first_half = [e for e in recent if e.timestamp < midpoint]
        second_half = [e for e in recent if e.timestamp >= midpoint]

        if len(first_half) == 0:
            return None

        ratio = len(second_half) / len(first_half)
        if ratio >= self._spike_threshold and len(second_half) >= 3:
            return ErrorPattern(
                pattern_type=PatternType.SPIKE,
                count=len(second_half),
                first_seen=second_half[0].timestamp,
                last_seen=second_half[-1].timestamp,
                severity="high" if ratio >= 5.0 else "medium",
                details=f"Error rate increased {ratio:.1f}x in last {half:.0f}s",
                affected_codes=list(set(e.error_code.value for e in second_half)),
            )
        return None

    def _detect_recurring(self, recent: list[ErrorEvent]) -> list[ErrorPattern]:
        """Detect recurring: same error code appears ≥threshold times."""
        patterns = []
        code_events: dict[str, list[ErrorEvent]] = {}

        for e in recent:
            key = e.error_code.value
            if key not in code_events:
                code_events[key] = []
            code_events[key].append(e)

        for code, events in code_events.items():
            if len(events) >= self._recurring_threshold:
                patterns.append(ErrorPattern(
                    pattern_type=PatternType.RECURRING,
                    count=len(events),
                    first_seen=events[0].timestamp,
                    last_seen=events[-1].timestamp,
                    severity="high" if len(events) >= self._recurring_threshold * 2 else "medium",
                    details=f"Error {code} occurred {len(events)} times",
                    affected_codes=[code],
                ))

        return patterns

    def _detect_correlated(self, recent: list[ErrorEvent]) -> Optional[ErrorPattern]:
        """
        Detect correlated failure: multiple providers failing within
        a short time window.
        """
        # Group events by provider
        by_provider: dict[str, list[ErrorEvent]] = {}
        for e in recent:
            if e.provider_name:
                if e.provider_name not in by_provider:
                    by_provider[e.provider_name] = []
                by_provider[e.provider_name].append(e)

        if len(by_provider) < 2:
            return None

        # Check if multiple providers have failures close together
        providers_with_recent_failures = []
        now = time.time()

        for provider, events in by_provider.items():
            recent_failures = [
                e for e in events
                if now - e.timestamp <= self._correlation_window
            ]
            if recent_failures:
                providers_with_recent_failures.append(provider)

        if len(providers_with_recent_failures) >= 2:
            all_events = []
            for p in providers_with_recent_failures:
                all_events.extend(by_provider[p])
            all_events.sort(key=lambda e: e.timestamp)

            return ErrorPattern(
                pattern_type=PatternType.CORRELATED,
                count=len(all_events),
                first_seen=all_events[0].timestamp if all_events else now,
                last_seen=all_events[-1].timestamp if all_events else now,
                severity="high",
                details=(
                    f"Correlated failures across {len(providers_with_recent_failures)} "
                    f"providers within {self._correlation_window}s"
                ),
                affected_providers=providers_with_recent_failures,
                affected_codes=list(set(e.error_code.value for e in all_events)),
            )

        return None
