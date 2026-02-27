"""
Network Retry Layer — generic async-aware retry with exponential backoff + jitter.

Provides three ways to use retry logic:

1. ``RetryExecutor`` class — full control, returns ``RetryResult``.
2. ``@with_retry`` decorator — wrap async functions.
3. ``retry_async()`` standalone — one-off calls.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


# ── Policy & Result dataclasses ─────────────────────────────────────

@dataclass
class RetryPolicy:
    """Configuration for retry behaviour."""
    max_attempts: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    transient_only: bool = True
    retryable_exceptions: Tuple[Type[BaseException], ...] = (
        ConnectionError, TimeoutError, OSError,
    )


@dataclass
class RetryResult:
    """Outcome of a retry-wrapped execution."""
    success: bool
    result: Any = None
    attempts: int = 0
    total_delay: float = 0.0
    last_error: Optional[Exception] = None
    errors: list = field(default_factory=list)


# ── RetryExecutor ───────────────────────────────────────────────────

class RetryExecutor:
    """Execute an async callable with configurable retry logic."""

    def __init__(self, policy: Optional[RetryPolicy] = None):
        self._policy = policy or RetryPolicy()

    @property
    def policy(self) -> RetryPolicy:
        return self._policy

    async def execute(self, fn: Callable, *args: Any, **kwargs: Any) -> RetryResult:
        """
        Call *fn* up to ``max_attempts`` times, backing off between failures.

        Returns a ``RetryResult`` with full history.
        """
        policy = self._policy
        errors: list[Exception] = []
        total_delay = 0.0

        for attempt in range(1, policy.max_attempts + 1):
            try:
                result = await fn(*args, **kwargs)
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    total_delay=total_delay,
                    errors=errors,
                )
            except Exception as exc:
                errors.append(exc)

                if not self._should_retry(exc, attempt):
                    logger.debug(
                        "Non-retryable error on attempt %d/%d: %s",
                        attempt, policy.max_attempts, exc,
                    )
                    return RetryResult(
                        success=False,
                        attempts=attempt,
                        total_delay=total_delay,
                        last_error=exc,
                        errors=errors,
                    )

                if attempt < policy.max_attempts:
                    delay = self._calculate_delay(attempt)
                    total_delay += delay
                    logger.info(
                        "Retry %d/%d after %.2fs — %s",
                        attempt, policy.max_attempts, delay, exc,
                    )
                    await asyncio.sleep(delay)

        # All attempts exhausted
        last = errors[-1] if errors else None
        return RetryResult(
            success=False,
            attempts=policy.max_attempts,
            total_delay=total_delay,
            last_error=last,
            errors=errors,
        )

    # ── Internal helpers ───────────────────────────────────────

    def _calculate_delay(self, attempt: int) -> float:
        """Exponential backoff with optional jitter."""
        policy = self._policy
        delay = min(
            policy.backoff_base * (policy.backoff_multiplier ** (attempt - 1)),
            policy.backoff_max,
        )
        if policy.jitter:
            # Add 0–50 % random jitter
            delay += delay * random.random() * 0.5
        return delay

    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Decide whether *error* on *attempt* is worth retrying."""
        policy = self._policy
        if attempt >= policy.max_attempts:
            return False

        # Check exception type against allowlist
        if isinstance(error, policy.retryable_exceptions):
            return True

        # If transient_only is set, consult the ErrorCatalog
        if policy.transient_only:
            try:
                from .error_catalog import ErrorCatalog
                return ErrorCatalog.is_transient(error)
            except ImportError:
                pass

        return False


# ── Decorator ───────────────────────────────────────────────────────

def with_retry(policy: Optional[RetryPolicy] = None) -> Callable:
    """
    Decorator that wraps an async function with retry logic.

    Usage::

        @with_retry(RetryPolicy(max_attempts=5))
        async def fetch_data():
            ...
    """
    executor = RetryExecutor(policy)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await executor.execute(fn, *args, **kwargs)
            if result.success:
                return result.result
            raise result.last_error  # type: ignore[misc]
        # Attach executor for introspection in tests
        wrapper._retry_executor = executor  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ── Standalone function ─────────────────────────────────────────────

async def retry_async(
    fn: Callable,
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    **kwargs: Any,
) -> Any:
    """
    One-shot retry wrapper.  Raises the last error on exhaustion.

    Usage::

        result = await retry_async(provider.send_message, messages, policy=policy)
    """
    executor = RetryExecutor(policy)
    result = await executor.execute(fn, *args, **kwargs)
    if result.success:
        return result.result
    raise result.last_error  # type: ignore[misc]
