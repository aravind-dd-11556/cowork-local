"""
Provider Circuit Breaker — State-machine isolation for failing providers.

Implements the circuit breaker pattern at the provider level:
  CLOSED  → Normal operation; failures counted
  OPEN    → Provider blocked; requests fail-fast after threshold exceeded
  HALF_OPEN → Limited probing; one success resets, one failure re-opens

Prevents cascade failures by isolating unhealthy providers.

Sprint 13 (Error Recovery & Resilience) Module 3.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal — counting failures
    OPEN = "open"           # Tripped — blocking requests
    HALF_OPEN = "half_open" # Probing — allowing limited traffic


@dataclass
class CircuitBreakerConfig:
    """Configuration for provider circuit breaker."""
    failure_threshold: int = 5        # Failures before opening
    timeout_seconds: float = 60.0     # How long OPEN lasts before HALF_OPEN
    half_open_max_calls: int = 2      # Max probe calls in HALF_OPEN


@dataclass
class _ProviderState:
    """Internal state for a single provider."""
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    opened_at: float = 0.0
    half_open_calls: int = 0


class ProviderCircuitBreaker:
    """
    Circuit breaker for LLM providers.

    Tracks per-provider failure/success and manages state transitions.
    Providers in OPEN state are blocked until timeout elapses.

    Usage:
        cb = ProviderCircuitBreaker()
        if cb.is_available("openai"):
            try:
                result = call_openai()
                cb.record_success("openai")
            except Exception as e:
                cb.record_failure("openai", e)
        else:
            # Provider is circuit-broken, use fallback
            ...
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self._config = config or CircuitBreakerConfig()
        self._states: dict[str, _ProviderState] = {}

    def _get_state(self, provider: str) -> _ProviderState:
        """Get or create state for a provider."""
        if provider not in self._states:
            self._states[provider] = _ProviderState()
        return self._states[provider]

    def is_available(self, provider: str) -> bool:
        """
        Check if a provider is available (not circuit-broken).

        Returns True if CLOSED or if OPEN has timed out (transitions to HALF_OPEN).
        """
        state = self._get_state(provider)

        if state.state == CircuitBreakerState.CLOSED:
            return True

        if state.state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed → transition to HALF_OPEN
            elapsed = time.time() - state.opened_at
            if elapsed >= self._config.timeout_seconds:
                state.state = CircuitBreakerState.HALF_OPEN
                state.half_open_calls = 0
                logger.info(
                    f"Circuit breaker for '{provider}' transitioning "
                    f"OPEN → HALF_OPEN after {elapsed:.1f}s"
                )
                return True
            return False

        if state.state == CircuitBreakerState.HALF_OPEN:
            # Allow limited calls in HALF_OPEN
            return state.half_open_calls < self._config.half_open_max_calls

        return False

    def record_success(self, provider: str) -> None:
        """Record a successful call to a provider."""
        state = self._get_state(provider)

        if state.state == CircuitBreakerState.HALF_OPEN:
            # Success in HALF_OPEN → reset to CLOSED
            state.state = CircuitBreakerState.CLOSED
            state.failure_count = 0
            state.success_count = 1
            state.half_open_calls = 0
            logger.info(f"Circuit breaker for '{provider}' HALF_OPEN → CLOSED (recovered)")

        elif state.state == CircuitBreakerState.CLOSED:
            state.success_count += 1
            # Reset failure count on success
            state.failure_count = 0

    def record_failure(self, provider: str, error: Optional[Exception] = None) -> None:
        """Record a failed call to a provider."""
        state = self._get_state(provider)
        now = time.time()

        state.failure_count += 1
        state.last_failure_time = now

        if state.state == CircuitBreakerState.HALF_OPEN:
            state.half_open_calls += 1
            # Failure in HALF_OPEN → back to OPEN
            state.state = CircuitBreakerState.OPEN
            state.opened_at = now
            logger.warning(
                f"Circuit breaker for '{provider}' HALF_OPEN → OPEN "
                f"(probe failed: {error})"
            )

        elif state.state == CircuitBreakerState.CLOSED:
            if state.failure_count >= self._config.failure_threshold:
                # Threshold exceeded → trip to OPEN
                state.state = CircuitBreakerState.OPEN
                state.opened_at = now
                logger.warning(
                    f"Circuit breaker for '{provider}' CLOSED → OPEN "
                    f"(threshold {self._config.failure_threshold} reached)"
                )

    def get_state(self, provider: str) -> CircuitBreakerState:
        """Get the current circuit breaker state for a provider."""
        # Trigger potential OPEN → HALF_OPEN transition
        self.is_available(provider)
        return self._get_state(provider).state

    def reset(self, provider: str) -> None:
        """Reset a provider's circuit breaker to CLOSED."""
        if provider in self._states:
            self._states[provider] = _ProviderState()
            logger.info(f"Circuit breaker for '{provider}' manually reset to CLOSED")

    def reset_all(self) -> None:
        """Reset all providers' circuit breakers."""
        self._states.clear()
        logger.info("All circuit breakers reset")

    def to_dict(self) -> dict:
        """Serialize all provider states for reporting."""
        result = {}
        for provider, state in self._states.items():
            # Trigger potential state transitions
            self.is_available(provider)
            result[provider] = {
                "state": state.state.value,
                "failure_count": state.failure_count,
                "success_count": state.success_count,
                "last_failure_time": state.last_failure_time,
            }
        return result
