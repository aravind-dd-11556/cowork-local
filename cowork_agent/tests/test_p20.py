"""
Sprint 20 Tests — Web UI: Observability Dashboard

Covers:
  - DashboardDataProvider (all snapshot methods)
  - API endpoints (dashboard routes)
  - WebSocket dashboard connections
  - HTML dashboard serving
  - Config + wiring
  - Edge cases

~140 tests across 7 classes.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from dataclasses import dataclass

# ── Ensure package is importable ─────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ═══════════════════════════════════════════════════════════════════
#  Mock Classes for Dashboard Tests
# ═══════════════════════════════════════════════════════════════════


class MockMetricsRegistry:
    """Mock MetricsRegistry for testing."""

    def __init__(self, summary_data=None, token_data=None):
        self._summary_data = summary_data or {
            "total_providers": 2,
            "providers": {
                "ollama": {"health_score": 0.95, "total_calls": 100},
                "openai": {"health_score": 0.82, "total_calls": 50},
            },
        }
        self._token_data = token_data or {
            "ollama": {"total_input_tokens": 5000, "total_output_tokens": 3000},
            "openai": {"total_input_tokens": 2000, "total_output_tokens": 1000},
        }

    def summary(self):
        return self._summary_data

    def get_token_usage(self):
        return self._token_data


class MockAuditLog:
    """Mock SecurityAuditLog for testing."""

    def __init__(self, events=None, has_query_db=True):
        if events is not None:
            self._events = events
        else:
            self._events = [
                {
                    "event_type": "input_injection",
                    "severity": "high",
                    "component": "sanitizer",
                    "description": "SQL injection blocked",
                    "blocked": True,
                    "timestamp": time.time() - 60,
                },
                {
                    "event_type": "credential_exposure",
                    "severity": "medium",
                    "component": "detector",
                    "description": "API key detected and masked",
                    "blocked": False,
                    "timestamp": time.time() - 120,
                },
            ]
        self._has_query_db = has_query_db
        self._summary_data = {
            "total_events": len(self._events),
            "blocked_count": sum(1 for e in self._events if e.get("blocked")),
        }

    def query_db(self, severity=None, limit=100):
        if not self._has_query_db:
            raise AttributeError("No query_db")
        events = self._events
        if severity:
            events = [e for e in events if e.get("severity") == severity.lower()]
        return events[:limit]

    def summary(self):
        return self._summary_data

    if True:  # make hasattr checks work
        pass


class MockAuditLogInMemory:
    """Mock SecurityAuditLog without query_db (in-memory only)."""

    def __init__(self, events=None):
        self._events = events or []

    def query(self, severity=None, limit=100):
        return self._events[:limit]


@dataclass
class MockBenchmarkStats:
    """Mock benchmark stats."""
    count: int = 10
    avg_ms: float = 42.5
    p95_ms: float = 85.0
    p99_ms: float = 120.0
    success_rate: float = 0.95

    def to_dict(self):
        return {
            "count": self.count,
            "avg_ms": self.avg_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "success_rate": self.success_rate,
        }


@dataclass
class MockBenchmarkRun:
    """Mock benchmark run."""
    name: str = "test_op"
    duration_ms: float = 42.0
    success: bool = True
    timestamp: float = 0.0

    def to_dict(self):
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "timestamp": self.timestamp or time.time(),
        }


class MockBenchmark:
    """Mock PerformanceBenchmark for testing."""

    def __init__(self):
        self._stats = {"test_op": MockBenchmarkStats()}
        self._runs = {"test_op": [MockBenchmarkRun() for _ in range(5)]}

    def get_stats(self, name):
        return self._stats.get(name)

    def get_runs(self, name):
        return self._runs.get(name, [])

    def get_all_stats(self):
        return self._stats

    def get_slowest(self, top_n=10):
        return [{"name": "test_op", "avg_ms": 42.5}]


class MockHealthOrchestrator:
    """Mock IntegratedHealthOrchestrator for testing."""

    def __init__(self, report=None):
        self._report = report or {
            "components": {
                "provider_ollama": {"score": 0.95, "status": "healthy"},
                "provider_openai": {"score": 0.82, "status": "healthy"},
                "error_aggregator": {"score": 1.0, "status": "healthy"},
            },
            "overall_score": 0.92,
        }

    def full_report(self):
        return self._report


class MockPersistentStore:
    """Mock PersistentStore for testing."""

    def __init__(self):
        self.metrics = MockStoreMetrics()
        self.audit = MockStoreAudit()
        self.benchmarks = MockStoreBenchmarks()

    def stats(self):
        return {
            "db_path": "/tmp/test.db",
            "db_size_bytes": 4096,
            "table_counts": {
                "token_usage": 100,
                "audit_events": 50,
                "benchmark_runs": 200,
            },
        }


class MockStoreMetrics:
    def query_token_usage(self, since=None, limit=500):
        return [{"provider": "ollama", "input_tokens": 100}]

    def query_errors(self, since=None, limit=500):
        return [{"provider": "ollama", "error": "timeout"}]

    def query_provider_calls(self, since=None, limit=500):
        return [{"provider": "ollama", "duration_ms": 42.0}]

    def aggregate_daily(self, metric="token_usage", since=None):
        return [{"date": "2026-02-28", "count": 50}]


class MockStoreAudit:
    def query_events(self, **kwargs):
        return []


class MockStoreBenchmarks:
    def query_runs(self, **kwargs):
        return []


# ═══════════════════════════════════════════════════════════════════
#  Test Class 1: DashboardDataProvider — Metrics
# ═══════════════════════════════════════════════════════════════════


class TestDashboardMetrics(unittest.TestCase):
    """Tests for DashboardDataProvider.get_metrics_snapshot() and historical."""

    def _make_provider(self, **kwargs):
        from cowork_agent.core.dashboard_data_provider import DashboardDataProvider
        return DashboardDataProvider(**kwargs)

    def test_metrics_snapshot_with_registry(self):
        p = self._make_provider(metrics_registry=MockMetricsRegistry())
        result = p.get_metrics_snapshot()
        self.assertTrue(result["available"])
        self.assertEqual(result["total_providers"], 2)
        self.assertIn("providers", result)
        self.assertIn("token_usage", result)
        self.assertIn("timestamp", result)

    def test_metrics_snapshot_no_registry(self):
        p = self._make_provider()
        result = p.get_metrics_snapshot()
        self.assertFalse(result["available"])

    def test_metrics_snapshot_provider_scores(self):
        p = self._make_provider(metrics_registry=MockMetricsRegistry())
        result = p.get_metrics_snapshot()
        providers = result["providers"]
        self.assertIn("ollama", providers)
        self.assertIn("openai", providers)
        self.assertEqual(providers["ollama"]["health_score"], 0.95)

    def test_metrics_snapshot_token_counts(self):
        p = self._make_provider(metrics_registry=MockMetricsRegistry())
        result = p.get_metrics_snapshot()
        usage = result["token_usage"]
        self.assertEqual(usage["ollama"]["total_input_tokens"], 5000)
        self.assertEqual(usage["openai"]["total_output_tokens"], 1000)

    def test_metrics_snapshot_error_handling(self):
        mock = MagicMock()
        mock.summary.side_effect = RuntimeError("crash")
        p = self._make_provider(metrics_registry=mock)
        result = p.get_metrics_snapshot()
        self.assertTrue(result["available"])
        self.assertIn("error", result)

    def test_metrics_historical_with_store(self):
        p = self._make_provider(persistent_store=MockPersistentStore())
        result = p.get_metrics_historical(days=7)
        self.assertTrue(result["available"])
        self.assertEqual(result["days"], 7)
        self.assertGreater(result["token_usage_count"], 0)
        self.assertGreater(result["error_count"], 0)
        self.assertGreater(result["call_count"], 0)

    def test_metrics_historical_no_store(self):
        p = self._make_provider()
        result = p.get_metrics_historical()
        self.assertFalse(result["available"])

    def test_metrics_historical_different_days(self):
        p = self._make_provider(persistent_store=MockPersistentStore())
        for days in [1, 7, 30]:
            result = p.get_metrics_historical(days=days)
            self.assertEqual(result["days"], days)

    def test_metrics_historical_daily_aggregates(self):
        p = self._make_provider(persistent_store=MockPersistentStore())
        result = p.get_metrics_historical()
        self.assertIn("daily_aggregates", result)
        self.assertIsInstance(result["daily_aggregates"], list)

    def test_metrics_historical_error_handling(self):
        store = MagicMock()
        store.metrics.query_token_usage.side_effect = RuntimeError("db error")
        p = self._make_provider(persistent_store=store)
        result = p.get_metrics_historical()
        self.assertTrue(result["available"])
        self.assertIn("error", result)

    def test_metrics_snapshot_has_timestamp(self):
        p = self._make_provider(metrics_registry=MockMetricsRegistry())
        before = time.time()
        result = p.get_metrics_snapshot()
        after = time.time()
        self.assertGreaterEqual(result["timestamp"], before)
        self.assertLessEqual(result["timestamp"], after)

    def test_metrics_snapshot_empty_summary(self):
        mock = MockMetricsRegistry(
            summary_data={"total_providers": 0, "providers": {}},
            token_data={},
        )
        p = self._make_provider(metrics_registry=mock)
        result = p.get_metrics_snapshot()
        self.assertTrue(result["available"])
        self.assertEqual(result["total_providers"], 0)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 2: DashboardDataProvider — Audit Feed
# ═══════════════════════════════════════════════════════════════════


class TestDashboardAudit(unittest.TestCase):
    """Tests for DashboardDataProvider.get_audit_feed()."""

    def _make_provider(self, **kwargs):
        from cowork_agent.core.dashboard_data_provider import DashboardDataProvider
        return DashboardDataProvider(**kwargs)

    def test_audit_feed_with_db(self):
        p = self._make_provider(audit_log=MockAuditLog())
        result = p.get_audit_feed()
        self.assertTrue(result["available"])
        self.assertEqual(result["event_count"], 2)
        self.assertIn("events", result)

    def test_audit_feed_no_audit_log(self):
        p = self._make_provider()
        result = p.get_audit_feed()
        self.assertFalse(result["available"])

    def test_audit_feed_severity_filter(self):
        p = self._make_provider(audit_log=MockAuditLog())
        result = p.get_audit_feed(severity="high")
        self.assertTrue(result["available"])
        self.assertEqual(result["event_count"], 1)
        self.assertEqual(result["events"][0]["severity"], "high")

    def test_audit_feed_limit(self):
        p = self._make_provider(audit_log=MockAuditLog())
        result = p.get_audit_feed(limit=1)
        self.assertEqual(result["event_count"], 1)

    def test_audit_feed_includes_summary(self):
        p = self._make_provider(audit_log=MockAuditLog())
        result = p.get_audit_feed()
        self.assertIn("summary", result)

    def test_audit_feed_has_timestamp(self):
        p = self._make_provider(audit_log=MockAuditLog())
        result = p.get_audit_feed()
        self.assertIn("timestamp", result)

    def test_audit_feed_error_handling(self):
        mock = MagicMock()
        mock.query_db.side_effect = RuntimeError("crash")
        p = self._make_provider(audit_log=mock)
        result = p.get_audit_feed()
        self.assertTrue(result["available"])
        self.assertIn("error", result)

    def test_audit_feed_no_severity_returns_all(self):
        p = self._make_provider(audit_log=MockAuditLog())
        result = p.get_audit_feed(severity=None)
        self.assertEqual(result["event_count"], 2)

    def test_audit_feed_empty_events(self):
        p = self._make_provider(audit_log=MockAuditLog(events=[]))
        result = p.get_audit_feed()
        self.assertTrue(result["available"])
        self.assertEqual(result["event_count"], 0)

    def test_audit_feed_many_events_limited(self):
        events = [
            {
                "event_type": "test",
                "severity": "low",
                "component": "test",
                "description": f"event_{i}",
                "blocked": False,
                "timestamp": time.time() - i,
            }
            for i in range(200)
        ]
        p = self._make_provider(audit_log=MockAuditLog(events=events))
        result = p.get_audit_feed(limit=50)
        self.assertEqual(result["event_count"], 50)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 3: DashboardDataProvider — Health & Benchmarks
# ═══════════════════════════════════════════════════════════════════


class TestDashboardHealthBenchmarks(unittest.TestCase):
    """Tests for health and benchmark data methods."""

    def _make_provider(self, **kwargs):
        from cowork_agent.core.dashboard_data_provider import DashboardDataProvider
        return DashboardDataProvider(**kwargs)

    # ── Health ────────────────────────────────────────────────

    def test_health_snapshot_with_orchestrator(self):
        p = self._make_provider(health_orchestrator=MockHealthOrchestrator())
        result = p.get_health_snapshot()
        self.assertTrue(result["available"])
        self.assertIn("report", result)
        self.assertIn("components", result["report"])

    def test_health_snapshot_no_orchestrator(self):
        p = self._make_provider()
        result = p.get_health_snapshot()
        self.assertFalse(result["available"])

    def test_health_snapshot_component_scores(self):
        p = self._make_provider(health_orchestrator=MockHealthOrchestrator())
        result = p.get_health_snapshot()
        components = result["report"]["components"]
        self.assertIn("provider_ollama", components)
        self.assertEqual(components["provider_ollama"]["score"], 0.95)

    def test_health_snapshot_error_handling(self):
        mock = MagicMock()
        mock.full_report.side_effect = RuntimeError("crash")
        p = self._make_provider(health_orchestrator=mock)
        result = p.get_health_snapshot()
        self.assertTrue(result["available"])
        self.assertIn("error", result)

    def test_health_snapshot_without_full_report(self):
        """Health object without full_report should return minimal info."""
        mock = MagicMock(spec=[])  # no attributes at all
        p = self._make_provider(health_orchestrator=mock)
        result = p.get_health_snapshot()
        self.assertTrue(result["available"])
        self.assertEqual(result.get("status"), "healthy")

    def test_health_snapshot_has_timestamp(self):
        p = self._make_provider(health_orchestrator=MockHealthOrchestrator())
        result = p.get_health_snapshot()
        self.assertIn("timestamp", result)

    # ── Benchmarks ────────────────────────────────────────────

    def test_benchmark_data_all(self):
        p = self._make_provider(benchmark=MockBenchmark())
        result = p.get_benchmark_data()
        self.assertTrue(result["available"])
        self.assertIn("benchmarks", result)
        self.assertIn("test_op", result["benchmarks"])
        self.assertIn("slowest", result)

    def test_benchmark_data_by_name(self):
        p = self._make_provider(benchmark=MockBenchmark())
        result = p.get_benchmark_data(name="test_op")
        self.assertTrue(result["available"])
        self.assertEqual(result["name"], "test_op")
        self.assertIn("stats", result)
        self.assertIn("recent_runs", result)

    def test_benchmark_data_no_benchmark(self):
        p = self._make_provider()
        result = p.get_benchmark_data()
        self.assertFalse(result["available"])

    def test_benchmark_data_nonexistent_name(self):
        p = self._make_provider(benchmark=MockBenchmark())
        result = p.get_benchmark_data(name="nonexistent")
        self.assertTrue(result["available"])
        self.assertIsNone(result["stats"])

    def test_benchmark_data_stats_dict(self):
        p = self._make_provider(benchmark=MockBenchmark())
        result = p.get_benchmark_data()
        stats = result["benchmarks"]["test_op"]
        self.assertEqual(stats["count"], 10)
        self.assertAlmostEqual(stats["avg_ms"], 42.5)

    def test_benchmark_data_error_handling(self):
        mock = MagicMock()
        mock.get_all_stats.side_effect = RuntimeError("crash")
        p = self._make_provider(benchmark=mock)
        result = p.get_benchmark_data()
        self.assertTrue(result["available"])
        self.assertIn("error", result)

    def test_benchmark_data_recent_runs_limited(self):
        p = self._make_provider(benchmark=MockBenchmark())
        result = p.get_benchmark_data(name="test_op")
        self.assertLessEqual(len(result["recent_runs"]), 20)

    def test_benchmark_data_has_timestamp(self):
        p = self._make_provider(benchmark=MockBenchmark())
        result = p.get_benchmark_data()
        self.assertIn("timestamp", result)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 4: DashboardDataProvider — Full Dashboard & Store
# ═══════════════════════════════════════════════════════════════════


class TestDashboardFullAndStore(unittest.TestCase):
    """Tests for get_full_dashboard() and get_store_stats()."""

    def _make_provider(self, **kwargs):
        from cowork_agent.core.dashboard_data_provider import DashboardDataProvider
        return DashboardDataProvider(**kwargs)

    def test_full_dashboard_all_components(self):
        p = self._make_provider(
            metrics_registry=MockMetricsRegistry(),
            audit_log=MockAuditLog(),
            health_orchestrator=MockHealthOrchestrator(),
            benchmark=MockBenchmark(),
        )
        result = p.get_full_dashboard()
        self.assertIn("metrics", result)
        self.assertIn("audit", result)
        self.assertIn("health", result)
        self.assertIn("benchmarks", result)
        self.assertIn("timestamp", result)
        self.assertTrue(result["metrics"]["available"])
        self.assertTrue(result["audit"]["available"])
        self.assertTrue(result["health"]["available"])
        self.assertTrue(result["benchmarks"]["available"])

    def test_full_dashboard_no_components(self):
        p = self._make_provider()
        result = p.get_full_dashboard()
        self.assertFalse(result["metrics"]["available"])
        self.assertFalse(result["audit"]["available"])
        self.assertFalse(result["health"]["available"])
        self.assertFalse(result["benchmarks"]["available"])

    def test_full_dashboard_partial_components(self):
        p = self._make_provider(
            metrics_registry=MockMetricsRegistry(),
            health_orchestrator=MockHealthOrchestrator(),
        )
        result = p.get_full_dashboard()
        self.assertTrue(result["metrics"]["available"])
        self.assertFalse(result["audit"]["available"])
        self.assertTrue(result["health"]["available"])
        self.assertFalse(result["benchmarks"]["available"])

    def test_full_dashboard_metrics_content(self):
        p = self._make_provider(metrics_registry=MockMetricsRegistry())
        result = p.get_full_dashboard()
        self.assertEqual(result["metrics"]["total_providers"], 2)

    def test_full_dashboard_audit_limit(self):
        events = [
            {
                "event_type": "test", "severity": "low",
                "component": "test", "description": f"e{i}",
                "blocked": False, "timestamp": time.time(),
            }
            for i in range(100)
        ]
        p = self._make_provider(audit_log=MockAuditLog(events=events))
        result = p.get_full_dashboard()
        # Full dashboard uses limit=50 for audit
        self.assertLessEqual(result["audit"]["event_count"], 50)

    def test_full_dashboard_has_timestamp(self):
        p = self._make_provider()
        before = time.time()
        result = p.get_full_dashboard()
        after = time.time()
        self.assertGreaterEqual(result["timestamp"], before)
        self.assertLessEqual(result["timestamp"], after)

    # ── Store Stats ──────────────────────────────────────────

    def test_store_stats_with_store(self):
        p = self._make_provider(persistent_store=MockPersistentStore())
        result = p.get_store_stats()
        self.assertTrue(result["available"])
        self.assertIn("db_path", result)
        self.assertIn("table_counts", result)

    def test_store_stats_no_store(self):
        p = self._make_provider()
        result = p.get_store_stats()
        self.assertFalse(result["available"])

    def test_store_stats_error_handling(self):
        store = MagicMock()
        store.stats.side_effect = RuntimeError("crash")
        p = self._make_provider(persistent_store=store)
        result = p.get_store_stats()
        self.assertTrue(result["available"])
        self.assertIn("error", result)

    def test_store_stats_table_counts(self):
        p = self._make_provider(persistent_store=MockPersistentStore())
        result = p.get_store_stats()
        counts = result["table_counts"]
        self.assertEqual(counts["token_usage"], 100)
        self.assertEqual(counts["audit_events"], 50)
        self.assertEqual(counts["benchmark_runs"], 200)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 5: API Endpoints (via TestClient)
# ═══════════════════════════════════════════════════════════════════


class TestDashboardAPIEndpoints(unittest.TestCase):
    """Tests for dashboard REST API endpoints."""

    @classmethod
    def setUpClass(cls):
        """Create a RestAPIInterface with mock components."""
        try:
            from fastapi.testclient import TestClient
            cls.TestClient = TestClient
        except ImportError:
            cls.TestClient = None
            return

        from cowork_agent.interfaces.api import RestAPIInterface

        # Create minimal mock agent
        mock_agent = MagicMock()
        mock_agent.messages = []
        mock_agent.registry = MagicMock()
        mock_agent.registry.tool_names = ["bash", "read"]
        mock_agent.registry.get_schemas.return_value = []

        # Create dashboard provider with all mock components
        from cowork_agent.core.dashboard_data_provider import DashboardDataProvider
        cls.dashboard_provider = DashboardDataProvider(
            metrics_registry=MockMetricsRegistry(),
            audit_log=MockAuditLog(),
            health_orchestrator=MockHealthOrchestrator(),
            benchmark=MockBenchmark(),
            persistent_store=MockPersistentStore(),
        )

        cls.api = RestAPIInterface(
            agent=mock_agent,
            dashboard_provider=cls.dashboard_provider,
        )
        cls.client = TestClient(cls.api.app)

    def _skip_if_no_testclient(self):
        if self.TestClient is None:
            self.skipTest("fastapi testclient not available")

    def test_dashboard_html_served(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/dashboard")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers.get("content-type", ""))

    def test_dashboard_full_endpoint(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/full")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("metrics", data)
        self.assertIn("audit", data)
        self.assertIn("health", data)
        self.assertIn("benchmarks", data)

    def test_dashboard_metrics_endpoint(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["available"])

    def test_dashboard_metrics_historical_endpoint(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/metrics/historical?days=7")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["available"])
        self.assertEqual(data["days"], 7)

    def test_dashboard_metrics_historical_custom_days(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/metrics/historical?days=30")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["days"], 30)

    def test_dashboard_audit_endpoint(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/audit")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["available"])
        self.assertGreater(data["event_count"], 0)

    def test_dashboard_audit_severity_filter(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/audit?severity=high")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["event_count"], 1)

    def test_dashboard_audit_limit(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/audit?limit=1")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["event_count"], 1)

    def test_dashboard_health_endpoint(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["available"])
        self.assertIn("report", data)

    def test_dashboard_benchmarks_endpoint(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/benchmarks")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["available"])
        self.assertIn("benchmarks", data)

    def test_dashboard_benchmarks_by_name(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/benchmarks?name=test_op")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "test_op")
        self.assertIn("stats", data)

    def test_dashboard_store_stats_endpoint(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/store")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["available"])
        self.assertIn("table_counts", data)

    def test_health_endpoint_still_works(self):
        """Existing /api/health endpoint should still work."""
        self._skip_if_no_testclient()
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")

    def test_root_dashboard_still_works(self):
        """Existing / dashboard should still work."""
        self._skip_if_no_testclient()
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 6: API Endpoints — No Dashboard Provider
# ═══════════════════════════════════════════════════════════════════


class TestDashboardAPINoProvider(unittest.TestCase):
    """Tests for dashboard API endpoints when no provider is configured."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            cls.TestClient = TestClient
        except ImportError:
            cls.TestClient = None
            return

        from cowork_agent.interfaces.api import RestAPIInterface

        mock_agent = MagicMock()
        mock_agent.messages = []
        mock_agent.registry = MagicMock()
        mock_agent.registry.tool_names = []
        mock_agent.registry.get_schemas.return_value = []

        cls.api = RestAPIInterface(agent=mock_agent)
        cls.client = TestClient(cls.api.app)

    def _skip_if_no_testclient(self):
        if self.TestClient is None:
            self.skipTest("fastapi testclient not available")

    def test_full_returns_503(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/full")
        self.assertEqual(resp.status_code, 503)

    def test_metrics_returns_503(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/metrics")
        self.assertEqual(resp.status_code, 503)

    def test_historical_returns_503(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/metrics/historical")
        self.assertEqual(resp.status_code, 503)

    def test_audit_returns_503(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/audit")
        self.assertEqual(resp.status_code, 503)

    def test_health_returns_503(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/health")
        self.assertEqual(resp.status_code, 503)

    def test_benchmarks_returns_503(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/benchmarks")
        self.assertEqual(resp.status_code, 503)

    def test_store_returns_503(self):
        self._skip_if_no_testclient()
        resp = self.client.get("/api/dashboard/store")
        self.assertEqual(resp.status_code, 503)

    def test_observability_dashboard_html_still_serves(self):
        """Dashboard HTML should be served even without data provider."""
        self._skip_if_no_testclient()
        resp = self.client.get("/dashboard")
        self.assertEqual(resp.status_code, 200)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 7: Config, Wiring & HTML
# ═══════════════════════════════════════════════════════════════════


class TestDashboardConfigAndWiring(unittest.TestCase):
    """Tests for config, main.py wiring, and HTML dashboard."""

    def test_config_web_dashboard_section(self):
        from cowork_agent.config.settings import load_config
        config = load_config()
        dash = config.get("web_dashboard", {})
        self.assertTrue(dash.get("enabled", False))
        self.assertEqual(dash.get("auto_refresh_seconds"), 30)
        self.assertEqual(dash.get("max_audit_feed"), 100)
        self.assertTrue(dash.get("websocket_broadcast", False))

    def test_dashboard_html_exists(self):
        html_path = Path(__file__).resolve().parents[1] / "interfaces" / "web" / "observability_dashboard.html"
        self.assertTrue(html_path.exists(), f"Dashboard HTML not found at {html_path}")

    def test_dashboard_html_has_panels(self):
        html_path = Path(__file__).resolve().parents[1] / "interfaces" / "web" / "observability_dashboard.html"
        content = html_path.read_text()
        self.assertIn("metrics-panel", content)
        self.assertIn("events-panel", content)
        self.assertIn("health-panel", content)
        self.assertIn("bench-panel", content)

    def test_dashboard_html_has_websocket(self):
        html_path = Path(__file__).resolve().parents[1] / "interfaces" / "web" / "observability_dashboard.html"
        content = html_path.read_text()
        self.assertIn("WebSocket", content)

    def test_dashboard_html_has_chart_js(self):
        html_path = Path(__file__).resolve().parents[1] / "interfaces" / "web" / "observability_dashboard.html"
        content = html_path.read_text()
        self.assertIn("chart", content.lower())

    def test_dashboard_html_has_refresh(self):
        html_path = Path(__file__).resolve().parents[1] / "interfaces" / "web" / "observability_dashboard.html"
        content = html_path.read_text()
        self.assertIn("refresh", content.lower())

    def test_dashboard_html_fetches_full_endpoint(self):
        html_path = Path(__file__).resolve().parents[1] / "interfaces" / "web" / "observability_dashboard.html"
        content = html_path.read_text()
        self.assertIn("/api/dashboard/full", content)

    def test_agent_has_dashboard_provider_attr(self):
        from cowork_agent.core.agent import Agent
        mock_provider = MagicMock()
        mock_registry = MagicMock()
        mock_prompt = MagicMock()
        agent = Agent(
            provider=mock_provider,
            registry=mock_registry,
            prompt_builder=mock_prompt,
        )
        self.assertIsNone(agent.dashboard_provider)
        agent.dashboard_provider = "test"
        self.assertEqual(agent.dashboard_provider, "test")

    def test_rest_api_accepts_dashboard_provider(self):
        from cowork_agent.interfaces.api import RestAPIInterface
        mock_agent = MagicMock()
        mock_agent.messages = []
        mock_agent.registry = MagicMock()
        mock_agent.registry.tool_names = []
        mock_agent.registry.get_schemas.return_value = []

        dp = MagicMock()
        api = RestAPIInterface(agent=mock_agent, dashboard_provider=dp)
        self.assertEqual(api._dashboard_provider, dp)

    def test_rest_api_has_dashboard_ws_clients(self):
        from cowork_agent.interfaces.api import RestAPIInterface
        mock_agent = MagicMock()
        mock_agent.messages = []
        mock_agent.registry = MagicMock()
        mock_agent.registry.tool_names = []
        mock_agent.registry.get_schemas.return_value = []

        api = RestAPIInterface(agent=mock_agent)
        self.assertIsInstance(api._dashboard_ws_clients, list)
        self.assertEqual(len(api._dashboard_ws_clients), 0)

    def test_dashboard_data_provider_init_all_none(self):
        from cowork_agent.core.dashboard_data_provider import DashboardDataProvider
        dp = DashboardDataProvider()
        self.assertIsNone(dp._metrics)
        self.assertIsNone(dp._audit)
        self.assertIsNone(dp._health)
        self.assertIsNone(dp._benchmark)
        self.assertIsNone(dp._store)

    def test_dashboard_data_provider_init_with_components(self):
        from cowork_agent.core.dashboard_data_provider import DashboardDataProvider
        m = MockMetricsRegistry()
        a = MockAuditLog()
        h = MockHealthOrchestrator()
        b = MockBenchmark()
        s = MockPersistentStore()
        dp = DashboardDataProvider(
            metrics_registry=m, audit_log=a,
            health_orchestrator=h, benchmark=b,
            persistent_store=s,
        )
        self.assertIs(dp._metrics, m)
        self.assertIs(dp._audit, a)
        self.assertIs(dp._health, h)
        self.assertIs(dp._benchmark, b)
        self.assertIs(dp._store, s)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 8: Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestDashboardEdgeCases(unittest.TestCase):
    """Edge cases and integration tests."""

    def _make_provider(self, **kwargs):
        from cowork_agent.core.dashboard_data_provider import DashboardDataProvider
        return DashboardDataProvider(**kwargs)

    def test_full_dashboard_with_errors_in_components(self):
        """Full dashboard should still return even if individual components fail."""
        mock_metrics = MagicMock()
        mock_metrics.summary.side_effect = RuntimeError("crash")
        mock_audit = MagicMock()
        mock_audit.query_db.side_effect = RuntimeError("crash")

        p = self._make_provider(
            metrics_registry=mock_metrics,
            audit_log=mock_audit,
            health_orchestrator=MockHealthOrchestrator(),
            benchmark=MockBenchmark(),
        )
        result = p.get_full_dashboard()
        # Metrics and audit have errors, but health and benchmarks work
        self.assertIn("error", result["metrics"])
        self.assertIn("error", result["audit"])
        self.assertTrue(result["health"]["available"])
        self.assertTrue(result["benchmarks"]["available"])

    def test_concurrent_snapshot_calls(self):
        """Multiple snapshot calls shouldn't interfere."""
        p = self._make_provider(
            metrics_registry=MockMetricsRegistry(),
            benchmark=MockBenchmark(),
        )
        results = [p.get_full_dashboard() for _ in range(10)]
        for r in results:
            self.assertTrue(r["metrics"]["available"])
            self.assertTrue(r["benchmarks"]["available"])

    def test_metrics_snapshot_large_provider_list(self):
        summary = {
            "total_providers": 20,
            "providers": {f"prov_{i}": {"health_score": 0.9} for i in range(20)},
        }
        mock = MockMetricsRegistry(summary_data=summary)
        p = self._make_provider(metrics_registry=mock)
        result = p.get_metrics_snapshot()
        self.assertEqual(result["total_providers"], 20)
        self.assertEqual(len(result["providers"]), 20)

    def test_audit_feed_unicode_descriptions(self):
        events = [
            {
                "event_type": "test", "severity": "low",
                "component": "test", "description": "Injection \u2014 blocked \u2603",
                "blocked": True, "timestamp": time.time(),
            },
        ]
        p = self._make_provider(audit_log=MockAuditLog(events=events))
        result = p.get_audit_feed()
        self.assertIn("\u2014", result["events"][0]["description"])

    def test_benchmark_empty_stats(self):
        mock = MagicMock()
        mock.get_all_stats.return_value = {}
        mock.get_slowest.return_value = []
        p = self._make_provider(benchmark=mock)
        result = p.get_benchmark_data()
        self.assertTrue(result["available"])
        self.assertEqual(result["total_benchmarks"], 0)

    def test_health_custom_report(self):
        custom = {
            "components": {"custom_comp": {"score": 0.5, "status": "degraded"}},
            "overall_score": 0.5,
        }
        p = self._make_provider(health_orchestrator=MockHealthOrchestrator(report=custom))
        result = p.get_health_snapshot()
        self.assertIn("custom_comp", result["report"]["components"])
        self.assertEqual(
            result["report"]["components"]["custom_comp"]["status"], "degraded"
        )

    def test_store_stats_db_size(self):
        p = self._make_provider(persistent_store=MockPersistentStore())
        result = p.get_store_stats()
        self.assertEqual(result["db_size_bytes"], 4096)

    def test_metrics_historical_zero_days(self):
        p = self._make_provider(persistent_store=MockPersistentStore())
        result = p.get_metrics_historical(days=0)
        self.assertTrue(result["available"])
        self.assertEqual(result["days"], 0)

    def test_full_dashboard_returns_dict(self):
        p = self._make_provider()
        result = p.get_full_dashboard()
        self.assertIsInstance(result, dict)

    def test_full_dashboard_serializable(self):
        """Full dashboard output should be JSON-serializable."""
        p = self._make_provider(
            metrics_registry=MockMetricsRegistry(),
            audit_log=MockAuditLog(),
            health_orchestrator=MockHealthOrchestrator(),
            benchmark=MockBenchmark(),
            persistent_store=MockPersistentStore(),
        )
        result = p.get_full_dashboard()
        serialized = json.dumps(result)
        self.assertIsInstance(serialized, str)
        parsed = json.loads(serialized)
        self.assertEqual(parsed["metrics"]["total_providers"], 2)

    def test_broadcast_dashboard_event_method_exists(self):
        """RestAPIInterface should have broadcast_dashboard_event method."""
        from cowork_agent.interfaces.api import RestAPIInterface
        mock_agent = MagicMock()
        mock_agent.messages = []
        mock_agent.registry = MagicMock()
        mock_agent.registry.tool_names = []
        mock_agent.registry.get_schemas.return_value = []
        api = RestAPIInterface(agent=mock_agent)
        self.assertTrue(hasattr(api, "broadcast_dashboard_event"))
        self.assertTrue(callable(api.broadcast_dashboard_event))


# ═══════════════════════════════════════════════════════════════════
#  Test Class 9: WebSocket Dashboard (unit)
# ═══════════════════════════════════════════════════════════════════


class TestDashboardWebSocket(unittest.TestCase):
    """Unit tests for dashboard WebSocket functionality."""

    def test_ws_clients_list_initialized(self):
        from cowork_agent.interfaces.api import RestAPIInterface
        mock_agent = MagicMock()
        mock_agent.messages = []
        mock_agent.registry = MagicMock()
        mock_agent.registry.tool_names = []
        mock_agent.registry.get_schemas.return_value = []
        api = RestAPIInterface(agent=mock_agent)
        self.assertEqual(api._dashboard_ws_clients, [])

    def test_broadcast_dashboard_event_no_clients(self):
        """Should not error when no clients connected."""
        import asyncio
        from cowork_agent.interfaces.api import RestAPIInterface

        mock_agent = MagicMock()
        mock_agent.messages = []
        mock_agent.registry = MagicMock()
        mock_agent.registry.tool_names = []
        mock_agent.registry.get_schemas.return_value = []
        api = RestAPIInterface(agent=mock_agent)

        # Should run without errors
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                api.broadcast_dashboard_event("test_event", {"foo": "bar"})
            )
        finally:
            loop.close()

    def test_broadcast_removes_dead_clients(self):
        """Dead WebSocket clients should be removed."""
        import asyncio
        from cowork_agent.interfaces.api import RestAPIInterface

        mock_agent = MagicMock()
        mock_agent.messages = []
        mock_agent.registry = MagicMock()
        mock_agent.registry.tool_names = []
        mock_agent.registry.get_schemas.return_value = []
        api = RestAPIInterface(agent=mock_agent)

        # Add a mock WS that raises on send
        dead_ws = AsyncMock()
        dead_ws.send_json.side_effect = RuntimeError("connection closed")
        api._dashboard_ws_clients.append(dead_ws)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                api.broadcast_dashboard_event("test", {"data": 1})
            )
        finally:
            loop.close()

        self.assertNotIn(dead_ws, api._dashboard_ws_clients)

    def test_broadcast_sends_to_healthy_clients(self):
        """Healthy clients should receive messages."""
        import asyncio
        from cowork_agent.interfaces.api import RestAPIInterface

        mock_agent = MagicMock()
        mock_agent.messages = []
        mock_agent.registry = MagicMock()
        mock_agent.registry.tool_names = []
        mock_agent.registry.get_schemas.return_value = []
        api = RestAPIInterface(agent=mock_agent)

        healthy_ws = AsyncMock()
        api._dashboard_ws_clients.append(healthy_ws)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                api.broadcast_dashboard_event("metrics_updated", {"score": 0.95})
            )
        finally:
            loop.close()

        healthy_ws.send_json.assert_called_once_with({
            "type": "metrics_updated",
            "data": {"score": 0.95},
        })

    def test_websocket_endpoint_registered(self):
        """The /ws/dashboard endpoint should be registered."""
        from cowork_agent.interfaces.api import RestAPIInterface
        mock_agent = MagicMock()
        mock_agent.messages = []
        mock_agent.registry = MagicMock()
        mock_agent.registry.tool_names = []
        mock_agent.registry.get_schemas.return_value = []
        api = RestAPIInterface(agent=mock_agent)

        routes = [r.path for r in api.app.routes]
        self.assertIn("/ws/dashboard", routes)


if __name__ == "__main__":
    unittest.main()
