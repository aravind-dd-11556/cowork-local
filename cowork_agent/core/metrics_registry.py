"""
Metrics Registry — extended metrics collection with token, error, and health tracking.

Extends the base MetricsCollector with per-provider token usage, error type
classification, rolling error rates, provider health scores, and export
in JSON or Prometheus format.

Sprint 16 (Testing & Observability Hardening) Module 3.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class TokenUsageMetrics:
    """Snapshot of token usage for a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    provider: str = ""
    model: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "provider": self.provider,
            "model": self.model,
            "timestamp": self.timestamp,
        }


@dataclass
class ProviderErrorRecord:
    """A single provider error record."""
    error_type: str  # rate_limit, auth, timeout, network, api_error, unknown
    error_message: str
    provider: str
    timestamp: float = field(default_factory=time.time)


# ── MetricsRegistry ─────────────────────────────────────────────

class MetricsRegistry(MetricsCollector):
    """
    Extended metrics collection built on MetricsCollector.

    Adds token usage tracking, error type classification, rolling error
    rates, provider health scoring, and multi-format export.

    Usage::

        registry = MetricsRegistry()
        registry.record_token_usage("anthropic", "claude-3", input_tokens=100, output_tokens=50)
        registry.record_error("anthropic", "rate_limit", "429 Too Many Requests")
        health = registry.provider_health_score("anthropic")
    """

    # Error type categories for classification
    ERROR_TYPES = {
        "rate_limit", "auth", "timeout", "network",
        "api_error", "invalid_request", "model_unavailable", "unknown",
    }

    def __init__(
        self,
        max_latency_samples: int = 500,
        enabled: bool = True,
        error_rate_window_seconds: int = 300,
        token_usage_tracking: bool = True,
        detailed_latency_tracking: bool = False,
    ):
        super().__init__(max_latency_samples=max_latency_samples, enabled=enabled)
        self._error_rate_window = error_rate_window_seconds
        self._token_tracking_enabled = token_usage_tracking
        self._detailed_latency = detailed_latency_tracking

        # Token usage per provider
        self._token_usage: Dict[str, List[TokenUsageMetrics]] = {}
        self._max_token_samples = max_latency_samples

        # Error records per provider
        self._error_records: Dict[str, List[ProviderErrorRecord]] = {}
        self._max_error_records = max_latency_samples

        # Provider call history for rolling error rate
        self._provider_call_results: Dict[str, List[tuple]] = {}  # (timestamp, success)
        self._max_call_results = max_latency_samples

    # ── Token usage tracking ──────────────────────────────────

    def record_token_usage(
        self,
        provider: str,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        """Record token usage for a provider call."""
        if not self._enabled or not self._token_tracking_enabled:
            return

        usage = TokenUsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            provider=provider,
            model=model,
        )
        samples = self._token_usage.setdefault(provider, [])
        samples.append(usage)
        if len(samples) > self._max_token_samples:
            self._token_usage[provider] = samples[-self._max_token_samples:]

    def get_token_usage(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated token usage for a provider or all providers."""
        if provider:
            samples = self._token_usage.get(provider, [])
            return self._aggregate_token_usage(provider, samples)

        result: Dict[str, Any] = {}
        for prov, samples in self._token_usage.items():
            result[prov] = self._aggregate_token_usage(prov, samples)
        return result

    def token_efficiency(self, provider: str) -> float:
        """
        Output-to-input token ratio for a provider.

        Higher values indicate the model is generating more output per input
        token.  Returns 0.0 if no data.
        """
        samples = self._token_usage.get(provider, [])
        if not samples:
            return 0.0
        total_in = sum(s.input_tokens for s in samples)
        total_out = sum(s.output_tokens for s in samples)
        return total_out / total_in if total_in > 0 else 0.0

    def _aggregate_token_usage(
        self, provider: str, samples: List[TokenUsageMetrics],
    ) -> Dict[str, Any]:
        if not samples:
            return {"provider": provider, "total_calls": 0}
        return {
            "provider": provider,
            "total_calls": len(samples),
            "total_input_tokens": sum(s.input_tokens for s in samples),
            "total_output_tokens": sum(s.output_tokens for s in samples),
            "total_cache_read": sum(s.cache_read_tokens for s in samples),
            "total_cache_write": sum(s.cache_write_tokens for s in samples),
            "avg_input_tokens": round(
                sum(s.input_tokens for s in samples) / len(samples), 1
            ),
            "avg_output_tokens": round(
                sum(s.output_tokens for s in samples) / len(samples), 1
            ),
        }

    # ── Error tracking ────────────────────────────────────────

    def record_error(
        self,
        provider: str,
        error_type: str,
        error_message: str = "",
    ) -> None:
        """Record a provider error with categorization."""
        if not self._enabled:
            return
        if error_type not in self.ERROR_TYPES:
            error_type = "unknown"

        record = ProviderErrorRecord(
            error_type=error_type,
            error_message=error_message,
            provider=provider,
        )
        records = self._error_records.setdefault(provider, [])
        records.append(record)
        if len(records) > self._max_error_records:
            self._error_records[provider] = records[-self._max_error_records:]

    def get_error_distribution(self, provider: str) -> Dict[str, int]:
        """Get error type distribution for a provider."""
        records = self._error_records.get(provider, [])
        dist: Dict[str, int] = {}
        for rec in records:
            dist[rec.error_type] = dist.get(rec.error_type, 0) + 1
        return dist

    def rolling_error_rate(self, provider: str) -> float:
        """
        Calculate the error rate over the configured time window.

        Uses the provider call results history.
        """
        results = self._provider_call_results.get(provider, [])
        if not results:
            return 0.0

        cutoff = time.time() - self._error_rate_window
        recent = [(ts, success) for ts, success in results if ts >= cutoff]
        if not recent:
            return 0.0

        errors = sum(1 for _, success in recent if not success)
        return errors / len(recent)

    # ── Extended provider call recording ──────────────────────

    def record_provider_call(
        self,
        provider_name: str,
        duration_ms: float,
        success: bool,
    ) -> None:
        """Record a provider call (extends parent with rolling results)."""
        super().record_provider_call(provider_name, duration_ms, success)

        if not self._enabled:
            return

        results = self._provider_call_results.setdefault(provider_name, [])
        results.append((time.time(), success))
        if len(results) > self._max_call_results:
            self._provider_call_results[provider_name] = results[-self._max_call_results:]

    # ── Provider health score ─────────────────────────────────

    def provider_health_score(self, provider: str) -> float:
        """
        Calculate a 0.0–1.0 health score for a provider.

        Combines error rate (60%), latency (20%), and token efficiency (20%).
        """
        # Error rate component (60% weight)
        err_rate = self.rolling_error_rate(provider)
        error_score = max(0.0, 1.0 - err_rate * 2)  # 50% errors → score 0

        # Latency component (20% weight)
        p95 = self.provider_percentile(provider, 95)
        if p95 <= 0:
            latency_score = 1.0  # No data — assume OK
        elif p95 < 1000:
            latency_score = 1.0
        elif p95 < 5000:
            latency_score = 0.7
        elif p95 < 10000:
            latency_score = 0.4
        else:
            latency_score = 0.1

        # Token efficiency component (20% weight)
        eff = self.token_efficiency(provider)
        if eff <= 0:
            token_score = 1.0  # No data
        elif eff > 0.3:
            token_score = 1.0
        elif eff > 0.1:
            token_score = 0.6
        else:
            token_score = 0.3

        score = error_score * 0.6 + latency_score * 0.2 + token_score * 0.2
        return round(min(1.0, max(0.0, score)), 4)

    # ── Export ─────────────────────────────────────────────────

    def export_metrics(self, format: str = "json") -> str:
        """
        Export all metrics in the given format.

        Supported formats: "json", "prometheus"
        """
        if format == "prometheus":
            return self._export_prometheus()
        return self._export_json()

    def _export_json(self) -> str:
        data = self.summary()
        # Add token usage
        data["token_usage"] = self.get_token_usage()
        # Add error distributions
        data["error_distributions"] = {
            prov: self.get_error_distribution(prov)
            for prov in self._error_records
        }
        # Add health scores
        data["provider_health_scores"] = {
            prov: self.provider_health_score(prov)
            for prov in self._providers
        }
        return json.dumps(data, indent=2, default=str)

    def _export_prometheus(self) -> str:
        lines: List[str] = []

        # Tool metrics
        for name, m in self._tools.items():
            safe = name.replace("-", "_").replace(" ", "_")
            lines.append(f'cowork_tool_calls_total{{tool="{safe}"}} {m.call_count}')
            lines.append(f'cowork_tool_errors_total{{tool="{safe}"}} {m.error_count}')
            lines.append(f'cowork_tool_avg_latency_ms{{tool="{safe}"}} {m.avg_ms:.2f}')

        # Provider metrics
        for name, p in self._providers.items():
            safe = name.replace("-", "_").replace(" ", "_")
            lines.append(f'cowork_provider_calls_total{{provider="{safe}"}} {p.call_count}')
            lines.append(f'cowork_provider_errors_total{{provider="{safe}"}} {p.error_count}')
            lines.append(f'cowork_provider_avg_latency_ms{{provider="{safe}"}} {p.avg_ms:.2f}')

        # Token usage
        for prov, samples in self._token_usage.items():
            safe = prov.replace("-", "_").replace(" ", "_")
            total_in = sum(s.input_tokens for s in samples)
            total_out = sum(s.output_tokens for s in samples)
            lines.append(f'cowork_provider_input_tokens_total{{provider="{safe}"}} {total_in}')
            lines.append(f'cowork_provider_output_tokens_total{{provider="{safe}"}} {total_out}')

        return "\n".join(lines)

    # ── Reset ─────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all metrics including extended fields."""
        super().reset()
        self._token_usage.clear()
        self._error_records.clear()
        self._provider_call_results.clear()
