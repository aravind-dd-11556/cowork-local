"""
Persistent Metrics Registry — write-through wrapper for MetricsRegistry.

Extends MetricsRegistry to persist all recorded data to SQLite while
maintaining full in-memory functionality.

Sprint 19 (Persistent Storage) Module 2.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .metrics_registry import MetricsRegistry
from .persistent_store import PersistentStore

logger = logging.getLogger(__name__)


class PersistentMetricsRegistry(MetricsRegistry):
    """
    MetricsRegistry with SQLite write-through persistence.

    All record_*() calls write to both in-memory storage and SQLite.
    Adds query_historical() for time-range DB queries.

    Usage::

        store = PersistentStore("/path/to/db")
        registry = PersistentMetricsRegistry(store=store)
        registry.record_token_usage("anthropic", "claude", 100, 50)
        historical = registry.query_historical(days_back=7)
    """

    def __init__(
        self,
        store: PersistentStore,
        persist_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._store = store
        self._persist_enabled = persist_enabled

    def record_token_usage(
        self,
        provider: str,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        """Record token usage to memory and DB."""
        super().record_token_usage(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

        if self._persist_enabled:
            try:
                self._store.metrics.insert_token_usage(
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                )
            except Exception as exc:
                logger.debug("Failed to persist token usage: %s", exc)

    def record_error(self, provider: str, error_type: str, error_message: str = "") -> None:
        """Record error to memory and DB."""
        super().record_error(provider, error_type, error_message)

        if self._persist_enabled:
            try:
                self._store.metrics.insert_error(
                    provider=provider,
                    error_type=error_type,
                    error_message=error_message,
                )
            except Exception as exc:
                logger.debug("Failed to persist error: %s", exc)

    def record_provider_call(
        self, provider: str, duration_ms: float, success: bool, **kwargs
    ) -> None:
        """Record provider call to memory and DB."""
        super().record_provider_call(provider, duration_ms, success, **kwargs)

        if self._persist_enabled:
            try:
                self._store.metrics.insert_provider_call(
                    provider=provider,
                    duration_ms=duration_ms,
                    success=success,
                )
            except Exception as exc:
                logger.debug("Failed to persist provider call: %s", exc)

    # ── Historical queries ─────────────────────────────────────

    def query_historical(
        self,
        days_back: int = 7,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query historical metrics from DB."""
        since = time.time() - (days_back * 86400)
        return {
            "token_usage": self._store.metrics.query_token_usage(
                provider=provider, since=since
            ),
            "errors": self._store.metrics.query_errors(
                provider=provider, since=since
            ),
            "calls": self._store.metrics.query_provider_calls(
                provider=provider, since=since
            ),
        }

    def query_daily_aggregates(
        self,
        metric: str = "token_usage",
        days_back: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get daily aggregated metrics."""
        since = time.time() - (days_back * 86400)
        return self._store.metrics.aggregate_daily(metric=metric, since=since)

    def export_with_history(
        self,
        format: str = "json",
        days_back: int = 7,
    ) -> str:
        """Export metrics including historical DB data."""
        import json as json_mod

        current = self.export_metrics(format)
        if format != "json":
            return current

        try:
            current_data = json_mod.loads(current)
        except Exception:
            current_data = {}

        historical = self.query_historical(days_back=days_back)
        combined = {
            "current": current_data,
            "historical": {
                "days_back": days_back,
                "token_usage_count": len(historical.get("token_usage", [])),
                "error_count": len(historical.get("errors", [])),
                "call_count": len(historical.get("calls", [])),
            },
        }
        return json_mod.dumps(combined, indent=2, default=str)

    def cleanup(self, retention_days: int = 90) -> int:
        """Delete metrics older than retention_days."""
        cutoff = time.time() - (retention_days * 86400)
        return self._store.metrics.delete_before(cutoff)
