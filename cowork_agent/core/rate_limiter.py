"""
Rate Limiter — per-tool and per-provider rate limiting.

Implements token bucket and sliding window algorithms for controlling
the rate of tool calls and provider requests.

Sprint 17 (Security & Sandboxing) Module 5.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class RateLimitConfig:
    """Configuration for a rate limit bucket."""
    max_requests: int = 60            # Max requests per window
    window_seconds: float = 60.0      # Time window
    burst_limit: int = 10             # Max burst (token bucket capacity)
    refill_rate: float = 1.0          # Tokens per second

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "burst_limit": self.burst_limit,
            "refill_rate": self.refill_rate,
        }


@dataclass
class RateLimitStatus:
    """Current status of a rate limit bucket."""
    name: str
    allowed: bool
    remaining_requests: int
    remaining_tokens: float
    reset_after_seconds: float
    total_requests: int
    total_rejected: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "allowed": self.allowed,
            "remaining_requests": self.remaining_requests,
            "remaining_tokens": round(self.remaining_tokens, 2),
            "reset_after_seconds": round(self.reset_after_seconds, 2),
            "total_requests": self.total_requests,
            "total_rejected": self.total_rejected,
        }


# ── TokenBucket ──────────────────────────────────────────────────

class TokenBucket:
    """Token bucket rate limiter for burst control."""

    def __init__(self, capacity: int = 10, refill_rate: float = 1.0):
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.time()

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if successful."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def available_tokens(self) -> float:
        """Get current available tokens without consuming."""
        self._refill()
        return self._tokens

    def time_until_available(self, tokens: int = 1) -> float:
        """Seconds until the requested tokens become available."""
        self._refill()
        if self._tokens >= tokens:
            return 0.0
        needed = tokens - self._tokens
        return needed / self._refill_rate if self._refill_rate > 0 else float("inf")

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now


# ── SlidingWindowCounter ─────────────────────────────────────────

class SlidingWindowCounter:
    """Sliding window rate limiter for sustained rate control."""

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._timestamps: List[float] = []

    def try_acquire(self) -> bool:
        """Try to record a request. Returns True if under limit."""
        self._clean()
        if len(self._timestamps) < self._max_requests:
            self._timestamps.append(time.time())
            return True
        return False

    def remaining(self) -> int:
        """Requests remaining in current window."""
        self._clean()
        return max(0, self._max_requests - len(self._timestamps))

    def reset_after(self) -> float:
        """Seconds until the oldest request expires from the window."""
        self._clean()
        if not self._timestamps:
            return 0.0
        oldest = self._timestamps[0]
        return max(0.0, (oldest + self._window_seconds) - time.time())

    def _clean(self) -> None:
        cutoff = time.time() - self._window_seconds
        self._timestamps = [ts for ts in self._timestamps if ts > cutoff]


# ── RateLimiter ──────────────────────────────────────────────────

class RateLimiter:
    """
    Combined rate limiter with per-key token bucket and sliding window.

    Usage::

        rl = RateLimiter()
        rl.configure("bash", RateLimitConfig(max_requests=30, burst_limit=5))

        if rl.allow("bash"):
            # proceed with tool call
            pass
        else:
            status = rl.status("bash")
            print(f"Rate limited. Retry after {status.reset_after_seconds}s")
    """

    def __init__(self, default_config: Optional[RateLimitConfig] = None):
        self._default_config = default_config or RateLimitConfig()
        self._configs: Dict[str, RateLimitConfig] = {}
        self._buckets: Dict[str, TokenBucket] = {}
        self._windows: Dict[str, SlidingWindowCounter] = {}
        self._total_requests: Dict[str, int] = {}
        self._total_rejected: Dict[str, int] = {}

    def configure(self, name: str, config: RateLimitConfig) -> None:
        """Configure rate limits for a named resource (tool or provider)."""
        self._configs[name] = config
        self._buckets[name] = TokenBucket(
            capacity=config.burst_limit,
            refill_rate=config.refill_rate,
        )
        self._windows[name] = SlidingWindowCounter(
            max_requests=config.max_requests,
            window_seconds=config.window_seconds,
        )

    def allow(self, name: str) -> bool:
        """
        Check if a request is allowed for the named resource.

        Initializes with default config if not explicitly configured.
        Both token bucket and sliding window must allow the request.
        """
        self._ensure_initialized(name)
        self._total_requests[name] = self._total_requests.get(name, 0) + 1

        bucket = self._buckets[name]
        window = self._windows[name]

        # Both must allow
        bucket_ok = bucket.try_acquire()
        window_ok = window.try_acquire()

        allowed = bucket_ok and window_ok
        if not allowed:
            self._total_rejected[name] = self._total_rejected.get(name, 0) + 1

        return allowed

    def check(self, name: str) -> bool:
        """Check if a request would be allowed (without consuming tokens)."""
        self._ensure_initialized(name)
        bucket = self._buckets[name]
        window = self._windows[name]
        return bucket.available_tokens() >= 1 and window.remaining() > 0

    def status(self, name: str) -> RateLimitStatus:
        """Get current rate limit status for a resource."""
        self._ensure_initialized(name)
        bucket = self._buckets[name]
        window = self._windows[name]

        return RateLimitStatus(
            name=name,
            allowed=bucket.available_tokens() >= 1 and window.remaining() > 0,
            remaining_requests=window.remaining(),
            remaining_tokens=bucket.available_tokens(),
            reset_after_seconds=max(
                bucket.time_until_available(),
                window.reset_after(),
            ),
            total_requests=self._total_requests.get(name, 0),
            total_rejected=self._total_rejected.get(name, 0),
        )

    def wait_time(self, name: str) -> float:
        """Get seconds to wait before the next request is allowed."""
        self._ensure_initialized(name)
        bucket = self._buckets[name]
        window = self._windows[name]
        return max(bucket.time_until_available(), window.reset_after())

    def _ensure_initialized(self, name: str) -> None:
        """Lazily initialize rate limit state for a name."""
        if name not in self._buckets:
            config = self._configs.get(name, self._default_config)
            self._buckets[name] = TokenBucket(
                capacity=config.burst_limit,
                refill_rate=config.refill_rate,
            )
            self._windows[name] = SlidingWindowCounter(
                max_requests=config.max_requests,
                window_seconds=config.window_seconds,
            )

    # ── Bulk operations ────────────────────────────────────────

    def configure_tools(self, configs: Dict[str, RateLimitConfig]) -> None:
        """Configure rate limits for multiple tools at once."""
        for name, config in configs.items():
            self.configure(name, config)

    def all_statuses(self) -> Dict[str, RateLimitStatus]:
        """Get status for all configured resources."""
        all_names = set(self._configs.keys()) | set(self._buckets.keys())
        return {name: self.status(name) for name in sorted(all_names)}

    # ── Stats ──────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return rate limiter statistics."""
        total_req = sum(self._total_requests.values())
        total_rej = sum(self._total_rejected.values())
        return {
            "configured_resources": sorted(self._configs.keys()),
            "active_resources": sorted(self._buckets.keys()),
            "total_requests": total_req,
            "total_rejected": total_rej,
            "rejection_rate": total_rej / total_req if total_req > 0 else 0.0,
        }

    def reset(self, name: Optional[str] = None) -> None:
        """Reset rate limit state for a resource or all resources."""
        if name:
            self._buckets.pop(name, None)
            self._windows.pop(name, None)
            self._total_requests.pop(name, None)
            self._total_rejected.pop(name, None)
        else:
            self._buckets.clear()
            self._windows.clear()
            self._total_requests.clear()
            self._total_rejected.clear()
