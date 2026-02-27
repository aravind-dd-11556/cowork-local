"""
Metrics Collector — lightweight in-memory per-tool and per-provider metrics.

Tracks call counts, success/error rates, latencies, and exposes percentile
calculations (p50/p95/p99). No external dependencies.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Dataclasses ────────────────────────────────────────────────────

@dataclass
class ToolMetrics:
    """Aggregated metrics for a single tool."""
    name: str
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    last_error: Optional[str] = None
    last_called_at: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.call_count if self.call_count > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "avg_ms": round(self.avg_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "last_error": self.last_error,
        }


@dataclass
class ProviderMetrics:
    """Aggregated metrics for an LLM provider."""
    name: str
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.call_count if self.call_count > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "avg_ms": round(self.avg_ms, 2),
            "error_rate": round(self.error_rate, 4),
        }


# ── MetricsCollector ──────────────────────────────────────────────

class MetricsCollector:
    """
    Lightweight in-memory metrics collector for tool and provider calls.

    Usage::

        collector = MetricsCollector()

        # Record a tool call
        t0 = time.time()
        result = await tool.execute(input)
        duration_ms = (time.time() - t0) * 1000
        collector.record_tool_call("bash", duration_ms, success=True)

        # Query metrics
        p95 = collector.percentile("bash", 95)
        summary = collector.summary()
    """

    def __init__(self, max_latency_samples: int = 500, enabled: bool = True):
        self._max_samples = max_latency_samples
        self._enabled = enabled
        self._tools: Dict[str, ToolMetrics] = {}
        self._providers: Dict[str, ProviderMetrics] = {}
        self._start_time = time.time()

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Recording ──────────────────────────────────────────────

    def record_tool_call(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Record a single tool execution."""
        if not self._enabled:
            return

        m = self._tools.setdefault(tool_name, ToolMetrics(name=tool_name))
        m.call_count += 1
        m.total_ms += duration_ms
        m.last_called_at = time.time()

        if success:
            m.success_count += 1
        else:
            m.error_count += 1
            m.last_error = error

        # Keep bounded latency samples for percentile calculation
        m.latencies.append(duration_ms)
        if len(m.latencies) > self._max_samples:
            m.latencies = m.latencies[-self._max_samples:]

    def record_provider_call(
        self,
        provider_name: str,
        duration_ms: float,
        success: bool,
    ) -> None:
        """Record a single LLM provider call."""
        if not self._enabled:
            return

        p = self._providers.setdefault(provider_name, ProviderMetrics(name=provider_name))
        p.call_count += 1
        p.total_ms += duration_ms

        if success:
            p.success_count += 1
        else:
            p.error_count += 1

        p.latencies.append(duration_ms)
        if len(p.latencies) > self._max_samples:
            p.latencies = p.latencies[-self._max_samples:]

    # ── Querying ───────────────────────────────────────────────

    def get_tool_metrics(self, tool_name: Optional[str] = None) -> Dict[str, ToolMetrics]:
        """Get metrics for a specific tool or all tools."""
        if tool_name:
            m = self._tools.get(tool_name)
            return {tool_name: m} if m else {}
        return dict(self._tools)

    def get_provider_metrics(self, provider_name: Optional[str] = None) -> Dict[str, ProviderMetrics]:
        """Get metrics for a specific provider or all providers."""
        if provider_name:
            p = self._providers.get(provider_name)
            return {provider_name: p} if p else {}
        return dict(self._providers)

    def percentile(self, tool_name: str, p: int = 50) -> float:
        """
        Calculate the p-th percentile latency for a tool.

        Returns 0.0 if no data is available.
        """
        m = self._tools.get(tool_name)
        if not m or not m.latencies:
            return 0.0
        return self._calc_percentile(m.latencies, p)

    def provider_percentile(self, provider_name: str, p: int = 50) -> float:
        """Calculate the p-th percentile latency for a provider."""
        pm = self._providers.get(provider_name)
        if not pm or not pm.latencies:
            return 0.0
        return self._calc_percentile(pm.latencies, p)

    def summary(self) -> dict:
        """Return an overall metrics summary."""
        total_tool_calls = sum(m.call_count for m in self._tools.values())
        total_tool_errors = sum(m.error_count for m in self._tools.values())
        total_provider_calls = sum(p.call_count for p in self._providers.values())

        return {
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "total_tool_calls": total_tool_calls,
            "total_tool_errors": total_tool_errors,
            "tool_error_rate": round(total_tool_errors / total_tool_calls, 4) if total_tool_calls > 0 else 0.0,
            "total_provider_calls": total_provider_calls,
            "tools": {name: m.to_dict() for name, m in self._tools.items()},
            "providers": {name: p.to_dict() for name, p in self._providers.items()},
        }

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._tools.clear()
        self._providers.clear()
        self._start_time = time.time()

    # ── Internal ───────────────────────────────────────────────

    @staticmethod
    def _calc_percentile(data: List[float], p: int) -> float:
        """Calculate the p-th percentile from a list of values."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
