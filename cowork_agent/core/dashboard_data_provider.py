"""
Dashboard Data Provider — aggregates observability data for the web dashboard.

Collects and formats metrics, audit events, health status, and benchmark
data from all observability components into dashboard-ready payloads.

Sprint 20 (Web Dashboard) Module 1.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DashboardDataProvider:
    """
    Aggregates data from observability components into dashboard payloads.

    Accepts references to metrics_registry, audit_log, health_orchestrator,
    benchmark, and optional persistent_store. Provides snapshot methods for
    each dashboard panel.

    Usage::

        provider = DashboardDataProvider(
            metrics_registry=registry,
            audit_log=audit,
            health_orchestrator=health,
            benchmark=bench,
            persistent_store=store,
        )
        dashboard = provider.get_full_dashboard()
    """

    def __init__(
        self,
        metrics_registry=None,
        audit_log=None,
        health_orchestrator=None,
        benchmark=None,
        persistent_store=None,
    ):
        self._metrics = metrics_registry
        self._audit = audit_log
        self._health = health_orchestrator
        self._benchmark = benchmark
        self._store = persistent_store

    # ── Metrics Panel ──────────────────────────────────────────

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot for dashboard."""
        if self._metrics is None:
            return {"available": False}

        try:
            summary = self._metrics.summary()
            token_usage = self._metrics.get_token_usage()
            return {
                "available": True,
                "total_providers": summary.get("total_providers", 0),
                "providers": summary.get("providers", {}),
                "token_usage": token_usage,
                "timestamp": time.time(),
            }
        except Exception as exc:
            logger.debug("Failed to get metrics snapshot: %s", exc)
            return {"available": True, "error": str(exc)}

    def get_metrics_historical(self, days: int = 7) -> Dict[str, Any]:
        """Get historical metrics from persistent store."""
        if self._store is None:
            return {"available": False}

        try:
            token_usage = self._store.metrics.query_token_usage(
                since=time.time() - (days * 86400), limit=500,
            )
            errors = self._store.metrics.query_errors(
                since=time.time() - (days * 86400), limit=500,
            )
            calls = self._store.metrics.query_provider_calls(
                since=time.time() - (days * 86400), limit=500,
            )
            daily_agg = self._store.metrics.aggregate_daily(
                metric="token_usage",
                since=time.time() - (days * 86400),
            )
            return {
                "available": True,
                "days": days,
                "token_usage_count": len(token_usage),
                "error_count": len(errors),
                "call_count": len(calls),
                "daily_aggregates": daily_agg,
                "timestamp": time.time(),
            }
        except Exception as exc:
            logger.debug("Failed to get historical metrics: %s", exc)
            return {"available": True, "error": str(exc)}

    # ── Audit Panel ────────────────────────────────────────────

    def get_audit_feed(
        self,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get recent audit events for dashboard timeline."""
        if self._audit is None:
            return {"available": False, "events": []}

        try:
            # Try DB query if persistent
            if hasattr(self._audit, 'query_db'):
                events = self._audit.query_db(severity=severity, limit=limit)
            else:
                # Fall back to in-memory query
                from ..core.security_audit_log import SecuritySeverity
                sev = SecuritySeverity(severity) if severity else None
                mem_events = self._audit.query(severity=sev, limit=limit)
                events = [e.to_dict() for e in mem_events]

            summary = None
            if hasattr(self._audit, 'summary'):
                s = self._audit.summary()
                summary = s.to_dict() if hasattr(s, 'to_dict') else s

            return {
                "available": True,
                "events": events,
                "event_count": len(events),
                "summary": summary,
                "timestamp": time.time(),
            }
        except Exception as exc:
            logger.debug("Failed to get audit feed: %s", exc)
            return {"available": True, "events": [], "error": str(exc)}

    # ── Health Panel ───────────────────────────────────────────

    def get_health_snapshot(self) -> Dict[str, Any]:
        """Get health status for dashboard."""
        if self._health is None:
            return {"available": False}

        try:
            # IntegratedHealthOrchestrator provides full_report()
            if hasattr(self._health, 'full_report'):
                report = self._health.full_report()
                return {
                    "available": True,
                    "report": report,
                    "timestamp": time.time(),
                }
            # Fallback: minimal health info
            return {
                "available": True,
                "status": "healthy",
                "timestamp": time.time(),
            }
        except Exception as exc:
            logger.debug("Failed to get health snapshot: %s", exc)
            return {"available": True, "error": str(exc)}

    # ── Benchmark Panel ────────────────────────────────────────

    def get_benchmark_data(
        self,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get benchmark data for dashboard charts."""
        if self._benchmark is None:
            return {"available": False}

        try:
            if name:
                stats = self._benchmark.get_stats(name)
                runs = self._benchmark.get_runs(name)
                return {
                    "available": True,
                    "name": name,
                    "stats": stats.to_dict() if stats else None,
                    "recent_runs": [
                        r.to_dict() for r in (runs[-20:] if runs else [])
                    ],
                    "timestamp": time.time(),
                }
            else:
                all_stats = self._benchmark.get_all_stats()
                slowest = self._benchmark.get_slowest(top_n=10)
                return {
                    "available": True,
                    "benchmarks": {
                        n: s.to_dict() for n, s in all_stats.items()
                    },
                    "slowest": slowest,
                    "total_benchmarks": len(all_stats),
                    "timestamp": time.time(),
                }
        except Exception as exc:
            logger.debug("Failed to get benchmark data: %s", exc)
            return {"available": True, "error": str(exc)}

    # ── Full Dashboard ─────────────────────────────────────────

    def get_full_dashboard(self) -> Dict[str, Any]:
        """Get combined data for initial dashboard load."""
        return {
            "metrics": self.get_metrics_snapshot(),
            "audit": self.get_audit_feed(limit=50),
            "health": self.get_health_snapshot(),
            "benchmarks": self.get_benchmark_data(),
            "timestamp": time.time(),
        }

    # ── Store Stats ────────────────────────────────────────────

    def get_store_stats(self) -> Dict[str, Any]:
        """Get persistent store statistics."""
        if self._store is None:
            return {"available": False}
        try:
            return {
                "available": True,
                **self._store.stats(),
            }
        except Exception as exc:
            return {"available": True, "error": str(exc)}
