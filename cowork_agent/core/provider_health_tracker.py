"""
Provider Health Tracker — Live health scoring for LLM providers.

Maintains real-time health scores based on latency, error rates,
and availability using exponentially-weighted moving averages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProviderHealthScore:
    """Current health snapshot for a single provider."""
    provider_name: str
    score: float = 100.0           # 0 (dead) – 100 (perfect)
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0        # 0.0 – 1.0
    last_success: Optional[float] = None    # timestamp
    last_failure: Optional[float] = None    # timestamp
    consecutive_failures: int = 0
    sample_count: int = 0

    def to_dict(self) -> dict:
        return {
            "provider_name": self.provider_name,
            "score": round(self.score, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "error_rate": round(self.error_rate, 3),
            "consecutive_failures": self.consecutive_failures,
            "sample_count": self.sample_count,
            "status": self.status,
        }

    @property
    def status(self) -> str:
        if self.score >= 70:
            return "healthy"
        elif self.score >= 40:
            return "degraded"
        else:
            return "unhealthy"


class ProviderHealthTracker:
    """
    Tracks real-time health scores for each LLM provider.

    Uses exponentially-weighted moving averages (EWMA) for latency
    and error rate, with score penalties for consecutive failures.

    Score formula:
        score = base_score × availability_factor × latency_factor

    Where:
        base_score = 100 − (consecutive_failures × 15)  (min 0)
        availability_factor = 1.0 − error_rate
        latency_factor = 1.0 − min(avg_latency / latency_ceiling, 0.5)
    """

    def __init__(
        self,
        decay_factor: float = 0.85,
        latency_ceiling_ms: float = 10000.0,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._decay = decay_factor       # EWMA decay (0.85 = recent calls weighted more)
        self._latency_ceiling = latency_ceiling_ms
        self._providers: dict[str, _ProviderState] = {}

    def _get_state(self, name: str) -> _ProviderState:
        if name not in self._providers:
            self._providers[name] = _ProviderState(name=name)
        return self._providers[name]

    # ── Recording ─────────────────────────────────────────────────

    def record_call(
        self,
        provider_name: str,
        duration_ms: float,
        success: bool,
    ) -> None:
        """
        Record a provider call result.

        Updates EWMA latency, error rate, and consecutive failure count.
        """
        if not self._enabled:
            return

        state = self._get_state(provider_name)
        state.sample_count += 1
        now = time.time()

        if success:
            state.consecutive_failures = 0
            state.last_success = now
        else:
            state.consecutive_failures += 1
            state.last_failure = now

        # EWMA for latency
        if state.avg_latency_ms == 0:
            state.avg_latency_ms = duration_ms
        else:
            state.avg_latency_ms = (
                self._decay * state.avg_latency_ms +
                (1 - self._decay) * duration_ms
            )

        # EWMA for error rate
        error_val = 0.0 if success else 1.0
        if state.sample_count == 1:
            state.error_rate = error_val
        else:
            state.error_rate = (
                self._decay * state.error_rate +
                (1 - self._decay) * error_val
            )

    # ── Scoring ───────────────────────────────────────────────────

    def get_score(self, provider_name: str) -> ProviderHealthScore:
        """Calculate and return the current health score for a provider."""
        state = self._get_state(provider_name)

        # Base score — penalize consecutive failures heavily
        base = max(0, 100 - (state.consecutive_failures * 15))

        # Availability factor — penalize high error rates
        availability = max(0, 1.0 - state.error_rate)

        # Latency factor — penalize slow responses (up to 50% penalty)
        if state.avg_latency_ms > 0 and self._latency_ceiling > 0:
            latency_penalty = min(state.avg_latency_ms / self._latency_ceiling, 0.5)
        else:
            latency_penalty = 0.0
        latency_factor = 1.0 - latency_penalty

        score = base * availability * latency_factor

        return ProviderHealthScore(
            provider_name=provider_name,
            score=round(score, 1),
            avg_latency_ms=round(state.avg_latency_ms, 1),
            error_rate=round(state.error_rate, 3),
            last_success=state.last_success,
            last_failure=state.last_failure,
            consecutive_failures=state.consecutive_failures,
            sample_count=state.sample_count,
        )

    def get_rankings(self) -> list[ProviderHealthScore]:
        """Return all providers ranked by health score (best first)."""
        scores = [self.get_score(name) for name in self._providers]
        return sorted(scores, key=lambda s: s.score, reverse=True)

    def is_healthy(self, provider_name: str, threshold: float = 30.0) -> bool:
        """Check if a provider's score is above the threshold."""
        return self.get_score(provider_name).score >= threshold

    def get_best_provider(self) -> Optional[str]:
        """Return the name of the highest-scoring provider."""
        rankings = self.get_rankings()
        return rankings[0].provider_name if rankings else None

    # ── Summary ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Full snapshot of all provider health data."""
        return {
            "enabled": self._enabled,
            "providers": {
                name: self.get_score(name).to_dict()
                for name in self._providers
            },
            "best_provider": self.get_best_provider(),
        }

    def reset(self) -> None:
        """Clear all provider health data."""
        self._providers.clear()

    @property
    def provider_names(self) -> list[str]:
        return list(self._providers.keys())


@dataclass
class _ProviderState:
    """Internal mutable state for EWMA tracking."""
    name: str
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    consecutive_failures: int = 0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    sample_count: int = 0
