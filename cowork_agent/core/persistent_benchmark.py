"""
Persistent Performance Benchmark — write-through wrapper for PerformanceBenchmark.

Extends PerformanceBenchmark to persist all recorded runs to SQLite while
maintaining full in-memory functionality.

Sprint 19 (Persistent Storage) Module 4.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .performance_benchmark import BenchmarkRun, PerformanceBenchmark
from .persistent_store import PersistentStore

logger = logging.getLogger(__name__)


class PersistentPerformanceBenchmark(PerformanceBenchmark):
    """
    PerformanceBenchmark with SQLite write-through persistence.

    All record() calls write to both in-memory storage and SQLite.
    Adds get_historical_stats() for time-range DB queries.

    Usage::

        store = PersistentStore("/path/to/db")
        bench = PersistentPerformanceBenchmark(store=store)
        bench.record("tool_bash", 42.5, component="tool")
        historical = bench.get_historical_stats("tool_bash", days_back=30)
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

    def record(
        self,
        name: str,
        duration_ms: float,
        component: str = "",
        success: bool = True,
        tags: Optional[List[str]] = None,
        **metadata: Any,
    ) -> BenchmarkRun:
        """Record a benchmark run to memory and DB."""
        run = super().record(
            name=name,
            duration_ms=duration_ms,
            component=component,
            success=success,
            tags=tags,
            **metadata,
        )

        if self._persist_enabled:
            try:
                self._store.benchmarks.insert_run(
                    name=name,
                    duration_ms=duration_ms,
                    component=component,
                    success=success,
                    tags=tags,
                    metadata=metadata if metadata else None,
                    timestamp=run.timestamp,
                )
            except Exception as exc:
                logger.debug("Failed to persist benchmark run: %s", exc)

        return run

    # ── Historical queries ─────────────────────────────────────

    def get_historical_stats(
        self,
        name: str,
        days_back: int = 30,
    ) -> Dict[str, Any]:
        """Get aggregate statistics from DB for a benchmark name."""
        since = time.time() - (days_back * 86400)
        return self._store.benchmarks.get_stats(name=name, since=since)

    def query_historical_runs(
        self,
        name: Optional[str] = None,
        component: Optional[str] = None,
        days_back: int = 7,
        success_only: Optional[bool] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query historical benchmark runs from DB."""
        since = time.time() - (days_back * 86400)
        return self._store.benchmarks.query_runs(
            name=name,
            component=component,
            since=since,
            success_only=success_only,
            limit=limit,
        )

    def cleanup(self, retention_days: int = 90) -> int:
        """Delete benchmark runs older than retention_days."""
        cutoff = time.time() - (retention_days * 86400)
        return self._store.benchmarks.delete_before(cutoff)

    def export_with_history(
        self,
        format: str = "json",
        days_back: int = 7,
    ) -> str:
        """Export benchmark data including historical DB data."""
        import json as json_mod

        current = self.export_report(format)
        if format != "json":
            return current

        try:
            current_data = json_mod.loads(current)
        except Exception:
            current_data = {}

        # Get historical stats for all known benchmarks
        all_names = set(self._runs.keys())
        historical = {}
        since = time.time() - (days_back * 86400)
        for name in all_names:
            stats = self._store.benchmarks.get_stats(name=name, since=since)
            if stats.get("count", 0) > 0:
                historical[name] = stats

        combined = {
            "current": current_data,
            "historical": {
                "days_back": days_back,
                "benchmarks": historical,
            },
        }
        return json_mod.dumps(combined, indent=2, default=str)
