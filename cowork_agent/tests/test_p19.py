"""
Sprint 19 — Persistent Storage for Metrics & Audit Logs.

Tests for:
  - PersistentStore (SQLite backend with MetricsTable, AuditTable, BenchmarkTable)
  - PersistentMetricsRegistry (write-through MetricsRegistry wrapper)
  - PersistentAuditLog (write-through SecurityAuditLog wrapper)
  - PersistentPerformanceBenchmark (write-through PerformanceBenchmark wrapper)
  - Config wiring & integration

~150 tests across 8 test classes.
"""

import json
import math
import os
import shutil
import tempfile
import time

import pytest

# ── PersistentStore ─────────────────────────────────────────────

from cowork_agent.core.persistent_store import (
    PersistentStore,
    MetricsTable,
    AuditTable,
    BenchmarkTable,
    SCHEMA_VERSION,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="test_p19_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(tmp_dir):
    s = PersistentStore(base_path=tmp_dir)
    yield s
    s.close()


# ── TestPersistentStore ──────────────────────────────────────────

class TestPersistentStore:
    """Tests for PersistentStore initialization and management."""

    def test_creates_db_file(self, tmp_dir):
        store = PersistentStore(base_path=tmp_dir)
        assert os.path.exists(store.db_path)
        store.close()

    def test_creates_directory_if_missing(self):
        d = tempfile.mkdtemp(prefix="test_p19_")
        sub = os.path.join(d, "sub", "deep")
        store = PersistentStore(base_path=sub)
        assert os.path.isdir(sub)
        assert os.path.exists(store.db_path)
        store.close()
        shutil.rmtree(d, ignore_errors=True)

    def test_custom_db_name(self, tmp_dir):
        store = PersistentStore(base_path=tmp_dir, db_name="custom.db")
        assert store.db_path.endswith("custom.db")
        store.close()

    def test_schema_version_recorded(self, store):
        cur = store._conn.cursor()
        cur.execute("SELECT MAX(version) FROM schema_version")
        assert cur.fetchone()[0] == SCHEMA_VERSION

    def test_idempotent_schema_creation(self, tmp_dir):
        s1 = PersistentStore(base_path=tmp_dir)
        s1.close()
        s2 = PersistentStore(base_path=tmp_dir)
        cur = s2._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM schema_version")
        assert cur.fetchone()[0] == 1
        s2.close()

    def test_stats_empty(self, store):
        stats = store.stats()
        assert stats["token_usage_rows"] == 0
        assert stats["error_rows"] == 0
        assert stats["call_rows"] == 0
        assert stats["audit_event_rows"] == 0
        assert stats["benchmark_run_rows"] == 0
        assert stats["schema_version"] == SCHEMA_VERSION

    def test_stats_after_inserts(self, store):
        store.metrics.insert_token_usage("a", "m", 10, 5)
        store.metrics.insert_error("a", "rate_limit")
        store.metrics.insert_provider_call("a", 100.0, True)
        store.audit.insert_event("input_injection", "high", "san", "test")
        store.benchmarks.insert_run("bench", 50.0)
        stats = store.stats()
        assert stats["token_usage_rows"] == 1
        assert stats["error_rows"] == 1
        assert stats["call_rows"] == 1
        assert stats["audit_event_rows"] == 1
        assert stats["benchmark_run_rows"] == 1

    def test_db_path_property(self, store, tmp_dir):
        assert store.db_path == os.path.join(tmp_dir, "metrics.db")

    def test_close_and_reopen(self, tmp_dir):
        s = PersistentStore(base_path=tmp_dir)
        s.metrics.insert_token_usage("p", "m", 100, 50)
        s.close()
        s2 = PersistentStore(base_path=tmp_dir)
        rows = s2.metrics.query_token_usage()
        assert len(rows) == 1
        s2.close()

    def test_transaction_rollback(self, store):
        try:
            with store.transaction() as cur:
                cur.execute(
                    "INSERT INTO token_usage (provider, model, input_tokens, output_tokens, timestamp) VALUES (?, ?, ?, ?, ?)",
                    ("p", "m", 10, 5, time.time()),
                )
                raise ValueError("force rollback")
        except ValueError:
            pass
        rows = store.metrics.query_token_usage()
        assert len(rows) == 0

    def test_wal_mode(self, store):
        cur = store._conn.cursor()
        cur.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        assert mode.lower() == "wal"


# ── TestMetricsTable ─────────────────────────────────────────────

class TestMetricsTable:
    """Tests for MetricsTable (token usage, errors, provider calls)."""

    def test_insert_token_usage(self, store):
        row_id = store.metrics.insert_token_usage("anthropic", "claude", 100, 50)
        assert row_id >= 1

    def test_insert_token_usage_with_cache(self, store):
        store.metrics.insert_token_usage("a", "m", 100, 50, cache_read_tokens=20, cache_write_tokens=10)
        rows = store.metrics.query_token_usage()
        assert rows[0]["cache_read_tokens"] == 20
        assert rows[0]["cache_write_tokens"] == 10

    def test_insert_token_usage_custom_timestamp(self, store):
        ts = 1000000.0
        store.metrics.insert_token_usage("a", "m", 10, 5, timestamp=ts)
        rows = store.metrics.query_token_usage()
        assert rows[0]["timestamp"] == ts

    def test_query_token_usage_all(self, store):
        store.metrics.insert_token_usage("a", "m1", 10, 5)
        store.metrics.insert_token_usage("b", "m2", 20, 10)
        rows = store.metrics.query_token_usage()
        assert len(rows) == 2

    def test_query_token_usage_by_provider(self, store):
        store.metrics.insert_token_usage("a", "m1", 10, 5)
        store.metrics.insert_token_usage("b", "m2", 20, 10)
        rows = store.metrics.query_token_usage(provider="a")
        assert len(rows) == 1
        assert rows[0]["provider"] == "a"

    def test_query_token_usage_since(self, store):
        old_ts = time.time() - 10000
        new_ts = time.time()
        store.metrics.insert_token_usage("a", "m", 10, 5, timestamp=old_ts)
        store.metrics.insert_token_usage("a", "m", 20, 10, timestamp=new_ts)
        rows = store.metrics.query_token_usage(since=new_ts - 1)
        assert len(rows) == 1
        assert rows[0]["input_tokens"] == 20

    def test_query_token_usage_until(self, store):
        old_ts = time.time() - 10000
        new_ts = time.time()
        store.metrics.insert_token_usage("a", "m", 10, 5, timestamp=old_ts)
        store.metrics.insert_token_usage("a", "m", 20, 10, timestamp=new_ts)
        rows = store.metrics.query_token_usage(until=old_ts + 1)
        assert len(rows) == 1
        assert rows[0]["input_tokens"] == 10

    def test_query_token_usage_limit(self, store):
        for i in range(10):
            store.metrics.insert_token_usage("a", "m", i, i)
        rows = store.metrics.query_token_usage(limit=3)
        assert len(rows) == 3

    def test_insert_error(self, store):
        row_id = store.metrics.insert_error("anthropic", "timeout", "Connection timed out")
        assert row_id >= 1

    def test_query_errors_all(self, store):
        store.metrics.insert_error("a", "timeout")
        store.metrics.insert_error("b", "rate_limit")
        rows = store.metrics.query_errors()
        assert len(rows) == 2

    def test_query_errors_by_provider(self, store):
        store.metrics.insert_error("a", "timeout")
        store.metrics.insert_error("b", "rate_limit")
        rows = store.metrics.query_errors(provider="b")
        assert len(rows) == 1

    def test_query_errors_by_type(self, store):
        store.metrics.insert_error("a", "timeout")
        store.metrics.insert_error("a", "rate_limit")
        rows = store.metrics.query_errors(error_type="timeout")
        assert len(rows) == 1

    def test_query_errors_since(self, store):
        old_ts = time.time() - 10000
        store.metrics.insert_error("a", "old", timestamp=old_ts)
        store.metrics.insert_error("a", "new")
        rows = store.metrics.query_errors(since=time.time() - 5)
        assert len(rows) == 1

    def test_insert_provider_call(self, store):
        row_id = store.metrics.insert_provider_call("anthropic", 150.5, True)
        assert row_id >= 1

    def test_query_provider_calls(self, store):
        store.metrics.insert_provider_call("a", 100.0, True)
        store.metrics.insert_provider_call("a", 200.0, False)
        rows = store.metrics.query_provider_calls()
        assert len(rows) == 2

    def test_query_provider_calls_by_provider(self, store):
        store.metrics.insert_provider_call("a", 100.0, True)
        store.metrics.insert_provider_call("b", 200.0, True)
        rows = store.metrics.query_provider_calls(provider="a")
        assert len(rows) == 1

    def test_query_provider_calls_success_only(self, store):
        store.metrics.insert_provider_call("a", 100.0, True)
        store.metrics.insert_provider_call("a", 200.0, False)
        rows = store.metrics.query_provider_calls(success_only=True)
        assert len(rows) == 1
        assert rows[0]["success"] == 1

    def test_query_provider_calls_failures_only(self, store):
        store.metrics.insert_provider_call("a", 100.0, True)
        store.metrics.insert_provider_call("a", 200.0, False)
        rows = store.metrics.query_provider_calls(success_only=False)
        assert len(rows) == 1

    def test_aggregate_daily_token_usage(self, store):
        ts = time.time()
        store.metrics.insert_token_usage("a", "m", 100, 50, timestamp=ts)
        store.metrics.insert_token_usage("a", "m", 200, 100, timestamp=ts)
        rows = store.metrics.aggregate_daily("token_usage")
        assert len(rows) >= 1
        assert rows[0]["total_input"] == 300
        assert rows[0]["total_output"] == 150

    def test_aggregate_daily_errors(self, store):
        ts = time.time()
        store.metrics.insert_error("a", "timeout", timestamp=ts)
        store.metrics.insert_error("a", "timeout", timestamp=ts)
        rows = store.metrics.aggregate_daily("errors")
        assert len(rows) >= 1

    def test_aggregate_daily_calls(self, store):
        ts = time.time()
        store.metrics.insert_provider_call("a", 100.0, True, timestamp=ts)
        store.metrics.insert_provider_call("a", 200.0, False, timestamp=ts)
        rows = store.metrics.aggregate_daily("calls")
        assert len(rows) >= 1
        assert rows[0]["total_calls"] == 2

    def test_aggregate_daily_unknown_metric(self, store):
        rows = store.metrics.aggregate_daily("unknown")
        assert rows == []

    def test_aggregate_daily_since(self, store):
        old_ts = time.time() - 200000
        new_ts = time.time()
        store.metrics.insert_token_usage("a", "m", 10, 5, timestamp=old_ts)
        store.metrics.insert_token_usage("a", "m", 20, 10, timestamp=new_ts)
        rows = store.metrics.aggregate_daily("token_usage", since=new_ts - 100)
        # Should only include the recent one
        total = sum(r["total_input"] for r in rows)
        assert total == 20

    def test_delete_before(self, store):
        old_ts = time.time() - 10000
        store.metrics.insert_token_usage("a", "m", 10, 5, timestamp=old_ts)
        store.metrics.insert_error("a", "err", timestamp=old_ts)
        store.metrics.insert_provider_call("a", 100.0, True, timestamp=old_ts)
        store.metrics.insert_token_usage("a", "m", 20, 10)
        deleted = store.metrics.delete_before(time.time() - 5000)
        assert deleted == 3
        assert len(store.metrics.query_token_usage()) == 1

    def test_delete_before_none(self, store):
        store.metrics.insert_token_usage("a", "m", 10, 5)
        deleted = store.metrics.delete_before(time.time() - 100000)
        assert deleted == 0


# ── TestAuditTable ───────────────────────────────────────────────

class TestAuditTable:
    """Tests for AuditTable persistence."""

    def test_insert_event(self, store):
        row_id = store.audit.insert_event(
            "input_injection", "high", "sanitizer", "SQL injection detected",
            tool_name="read", blocked=True,
        )
        assert row_id >= 1

    def test_insert_event_with_metadata(self, store):
        store.audit.insert_event(
            "credential_detected", "high", "detector", "API key found",
            metadata={"type": "aws_key"},
        )
        rows = store.audit.query_events()
        assert rows[0]["metadata"] == {"type": "aws_key"}

    def test_insert_event_custom_timestamp(self, store):
        ts = 1234567890.0
        store.audit.insert_event("test", "low", "comp", "desc", timestamp=ts)
        rows = store.audit.query_events()
        assert rows[0]["timestamp"] == ts

    def test_query_events_all(self, store):
        store.audit.insert_event("type_a", "high", "comp", "desc1")
        store.audit.insert_event("type_b", "low", "comp", "desc2")
        rows = store.audit.query_events()
        assert len(rows) == 2

    def test_query_events_by_type(self, store):
        store.audit.insert_event("type_a", "high", "comp", "desc1")
        store.audit.insert_event("type_b", "low", "comp", "desc2")
        rows = store.audit.query_events(event_type="type_a")
        assert len(rows) == 1
        assert rows[0]["event_type"] == "type_a"

    def test_query_events_by_severity(self, store):
        store.audit.insert_event("type_a", "high", "comp", "desc1")
        store.audit.insert_event("type_b", "low", "comp", "desc2")
        rows = store.audit.query_events(severity="high")
        assert len(rows) == 1

    def test_query_events_by_component(self, store):
        store.audit.insert_event("t", "h", "comp_a", "desc1")
        store.audit.insert_event("t", "h", "comp_b", "desc2")
        rows = store.audit.query_events(component="comp_a")
        assert len(rows) == 1

    def test_query_events_since(self, store):
        old_ts = time.time() - 10000
        store.audit.insert_event("t", "h", "c", "old", timestamp=old_ts)
        store.audit.insert_event("t", "h", "c", "new")
        rows = store.audit.query_events(since=time.time() - 5)
        assert len(rows) == 1

    def test_query_events_blocked_only(self, store):
        store.audit.insert_event("t", "h", "c", "blocked", blocked=True)
        store.audit.insert_event("t", "h", "c", "allowed", blocked=False)
        rows = store.audit.query_events(blocked_only=True)
        assert len(rows) == 1
        assert rows[0]["blocked"] is True

    def test_query_events_limit(self, store):
        for i in range(10):
            store.audit.insert_event("t", "h", "c", f"desc_{i}")
        rows = store.audit.query_events(limit=3)
        assert len(rows) == 3

    def test_count_by_severity(self, store):
        store.audit.insert_event("t", "high", "c", "d1")
        store.audit.insert_event("t", "high", "c", "d2")
        store.audit.insert_event("t", "low", "c", "d3")
        counts = store.audit.count_by_severity()
        assert counts["high"] == 2
        assert counts["low"] == 1

    def test_count_by_severity_since(self, store):
        old_ts = time.time() - 10000
        store.audit.insert_event("t", "high", "c", "old", timestamp=old_ts)
        store.audit.insert_event("t", "high", "c", "new")
        counts = store.audit.count_by_severity(since=time.time() - 5)
        assert counts.get("high", 0) == 1

    def test_delete_before(self, store):
        old_ts = time.time() - 10000
        store.audit.insert_event("t", "h", "c", "old", timestamp=old_ts)
        store.audit.insert_event("t", "h", "c", "new")
        deleted = store.audit.delete_before(time.time() - 5000)
        assert deleted == 1
        assert len(store.audit.query_events()) == 1

    def test_blocked_boolean_conversion(self, store):
        store.audit.insert_event("t", "h", "c", "d", blocked=True)
        rows = store.audit.query_events()
        assert rows[0]["blocked"] is True

    def test_metadata_json_roundtrip(self, store):
        meta = {"key1": "val1", "nested": {"a": 1}}
        store.audit.insert_event("t", "h", "c", "d", metadata=meta)
        rows = store.audit.query_events()
        assert rows[0]["metadata"] == meta

    def test_correlation_id_stored(self, store):
        store.audit.insert_event("t", "h", "c", "d", correlation_id="abc-123")
        rows = store.audit.query_events()
        assert rows[0]["correlation_id"] == "abc-123"


# ── TestBenchmarkTable ───────────────────────────────────────────

class TestBenchmarkTable:
    """Tests for BenchmarkTable persistence."""

    def test_insert_run(self, store):
        row_id = store.benchmarks.insert_run("bench_a", 42.5)
        assert row_id >= 1

    def test_insert_run_with_details(self, store):
        store.benchmarks.insert_run(
            "bench_a", 42.5, component="tool", success=True,
            tags=["fast", "tool"], metadata={"version": "1.0"},
        )
        rows = store.benchmarks.query_runs(name="bench_a")
        assert len(rows) == 1
        assert rows[0]["component"] == "tool"
        assert rows[0]["tags"] == ["fast", "tool"]
        assert rows[0]["metadata"] == {"version": "1.0"}

    def test_insert_run_failure(self, store):
        store.benchmarks.insert_run("bench_a", 42.5, success=False)
        rows = store.benchmarks.query_runs()
        assert rows[0]["success"] is False

    def test_query_runs_all(self, store):
        store.benchmarks.insert_run("a", 10.0)
        store.benchmarks.insert_run("b", 20.0)
        rows = store.benchmarks.query_runs()
        assert len(rows) == 2

    def test_query_runs_by_name(self, store):
        store.benchmarks.insert_run("a", 10.0)
        store.benchmarks.insert_run("b", 20.0)
        rows = store.benchmarks.query_runs(name="a")
        assert len(rows) == 1

    def test_query_runs_by_component(self, store):
        store.benchmarks.insert_run("a", 10.0, component="tool")
        store.benchmarks.insert_run("b", 20.0, component="provider")
        rows = store.benchmarks.query_runs(component="tool")
        assert len(rows) == 1

    def test_query_runs_since(self, store):
        old_ts = time.time() - 10000
        store.benchmarks.insert_run("a", 10.0, timestamp=old_ts)
        store.benchmarks.insert_run("a", 20.0)
        rows = store.benchmarks.query_runs(since=time.time() - 5)
        assert len(rows) == 1

    def test_query_runs_success_only(self, store):
        store.benchmarks.insert_run("a", 10.0, success=True)
        store.benchmarks.insert_run("a", 20.0, success=False)
        rows = store.benchmarks.query_runs(success_only=True)
        assert len(rows) == 1

    def test_query_runs_limit(self, store):
        for i in range(10):
            store.benchmarks.insert_run("a", float(i))
        rows = store.benchmarks.query_runs(limit=3)
        assert len(rows) == 3

    def test_get_stats_basic(self, store):
        for d in [10.0, 20.0, 30.0, 40.0, 50.0]:
            store.benchmarks.insert_run("bench", d)
        stats = store.benchmarks.get_stats("bench")
        assert stats["count"] == 5
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 50.0
        assert stats["avg_ms"] == 30.0

    def test_get_stats_empty(self, store):
        stats = store.benchmarks.get_stats("nonexistent")
        assert stats["count"] == 0

    def test_get_stats_with_since(self, store):
        old_ts = time.time() - 10000
        store.benchmarks.insert_run("b", 10.0, timestamp=old_ts)
        store.benchmarks.insert_run("b", 50.0)
        stats = store.benchmarks.get_stats("b", since=time.time() - 5)
        assert stats["count"] == 1
        assert stats["avg_ms"] == 50.0

    def test_get_stats_success_rate(self, store):
        store.benchmarks.insert_run("b", 10.0, success=True)
        store.benchmarks.insert_run("b", 20.0, success=True)
        store.benchmarks.insert_run("b", 30.0, success=False)
        stats = store.benchmarks.get_stats("b")
        assert abs(stats["success_rate"] - 2.0/3.0) < 0.01

    def test_get_stats_percentiles(self, store):
        for i in range(100):
            store.benchmarks.insert_run("b", float(i + 1))
        stats = store.benchmarks.get_stats("b")
        assert stats["p95_ms"] >= 90.0
        assert stats["p99_ms"] >= 95.0

    def test_delete_before(self, store):
        old_ts = time.time() - 10000
        store.benchmarks.insert_run("a", 10.0, timestamp=old_ts)
        store.benchmarks.insert_run("a", 20.0)
        deleted = store.benchmarks.delete_before(time.time() - 5000)
        assert deleted == 1

    def test_tags_json_roundtrip(self, store):
        store.benchmarks.insert_run("a", 10.0, tags=["x", "y", "z"])
        rows = store.benchmarks.query_runs()
        assert rows[0]["tags"] == ["x", "y", "z"]

    def test_metadata_json_roundtrip(self, store):
        meta = {"key": "value", "num": 42}
        store.benchmarks.insert_run("a", 10.0, metadata=meta)
        rows = store.benchmarks.query_runs()
        assert rows[0]["metadata"] == meta


# ── TestPersistentMetricsRegistry ────────────────────────────────

from cowork_agent.core.persistent_metrics_registry import PersistentMetricsRegistry


class TestPersistentMetricsRegistry:
    """Tests for PersistentMetricsRegistry write-through behavior."""

    @pytest.fixture
    def registry(self, store):
        return PersistentMetricsRegistry(store=store)

    def test_record_token_usage_persists(self, registry, store):
        registry.record_token_usage("test", "m", input_tokens=100, output_tokens=50)
        rows = store.metrics.query_token_usage()
        assert len(rows) == 1
        assert rows[0]["provider"] == "test"
        assert rows[0]["input_tokens"] == 100

    def test_record_token_usage_with_cache(self, registry, store):
        registry.record_token_usage("test", "m", 100, 50, cache_read_tokens=20, cache_write_tokens=10)
        rows = store.metrics.query_token_usage()
        assert rows[0]["cache_read_tokens"] == 20

    def test_record_error_persists(self, registry, store):
        registry.record_error("anthropic", "timeout", "Connection timed out")
        rows = store.metrics.query_errors()
        assert len(rows) == 1
        assert rows[0]["provider"] == "anthropic"
        assert rows[0]["error_type"] == "timeout"

    def test_record_provider_call_persists(self, registry, store):
        registry.record_provider_call("openai", 150.0, True)
        rows = store.metrics.query_provider_calls()
        assert len(rows) == 1
        assert rows[0]["duration_ms"] == 150.0

    def test_persist_disabled(self, store):
        reg = PersistentMetricsRegistry(store=store, persist_enabled=False)
        reg.record_token_usage("test", "m", 100, 50)
        rows = store.metrics.query_token_usage()
        assert len(rows) == 0  # Not persisted

    def test_query_historical(self, registry, store):
        registry.record_token_usage("test", "m", 100, 50)
        registry.record_error("a", "err")
        registry.record_provider_call("a", 100.0, True)
        result = registry.query_historical(days_back=1)
        assert "token_usage" in result
        assert "errors" in result
        assert "calls" in result
        assert len(result["token_usage"]) == 1

    def test_query_historical_with_provider(self, registry, store):
        registry.record_error("a", "err1")
        registry.record_error("b", "err2")
        result = registry.query_historical(days_back=1, provider="a")
        assert len(result["errors"]) == 1

    def test_query_daily_aggregates(self, registry, store):
        registry.record_token_usage("test", "m", 100, 50)
        registry.record_token_usage("test", "m", 200, 100)
        agg = registry.query_daily_aggregates(metric="token_usage", days_back=1)
        assert len(agg) >= 1

    def test_export_with_history(self, registry, store):
        registry.record_token_usage("test", "m", 100, 50)
        exported = registry.export_with_history(format="json", days_back=7)
        data = json.loads(exported)
        assert "current" in data
        assert "historical" in data

    def test_cleanup(self, registry, store):
        old_ts = time.time() - 200 * 86400
        store.metrics.insert_token_usage("a", "m", 10, 5, timestamp=old_ts)
        deleted = registry.cleanup(retention_days=90)
        assert deleted >= 1

    def test_multiple_records_persist(self, registry, store):
        for i in range(5):
            registry.record_token_usage("test", "m", i * 10, i * 5)
        rows = store.metrics.query_token_usage()
        assert len(rows) == 5

    def test_db_failure_does_not_crash(self, store):
        """If DB write fails, in-memory should still work."""
        reg = PersistentMetricsRegistry(store=store)
        # Close DB to force failure
        store.close()
        # Should not raise
        reg.record_token_usage("test", "m", 100, 50)
        reg.record_error("a", "err")
        reg.record_provider_call("a", 100.0, True)


# ── TestPersistentAuditLog ───────────────────────────────────────

from cowork_agent.core.persistent_audit_log import PersistentAuditLog
from cowork_agent.core.security_audit_log import SecurityEventType, SecuritySeverity


class TestPersistentAuditLog:
    """Tests for PersistentAuditLog write-through behavior."""

    @pytest.fixture
    def audit(self, store):
        return PersistentAuditLog(store=store)

    def test_log_persists(self, audit, store):
        audit.log(
            SecurityEventType.INPUT_INJECTION,
            SecuritySeverity.HIGH,
            "sanitizer",
            "SQL injection blocked",
            blocked=True,
        )
        rows = store.audit.query_events()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "input_injection"
        assert rows[0]["severity"] == "high"
        assert rows[0]["blocked"] is True

    def test_log_with_metadata(self, audit, store):
        audit.log(
            SecurityEventType.CREDENTIAL_DETECTED,
            SecuritySeverity.HIGH,
            "detector",
            "API key",
            metadata={"type": "aws"},
        )
        rows = store.audit.query_events()
        assert rows[0]["metadata"]["type"] == "aws"

    def test_log_with_correlation_id(self, audit, store):
        audit.log(
            SecurityEventType.PROMPT_INJECTION,
            SecuritySeverity.CRITICAL,
            "detector",
            "Prompt injection",
            correlation_id="corr-123",
        )
        rows = store.audit.query_events()
        assert rows[0]["correlation_id"] == "corr-123"

    def test_log_with_tool_name(self, audit, store):
        audit.log(
            SecurityEventType.SANDBOX_VIOLATION,
            SecuritySeverity.HIGH,
            "sandbox",
            "Violation",
            tool_name="bash",
        )
        rows = store.audit.query_events()
        assert rows[0]["tool_name"] == "bash"

    def test_persist_disabled(self, store):
        audit = PersistentAuditLog(store=store, persist_enabled=False)
        audit.log(
            SecurityEventType.INPUT_INJECTION,
            SecuritySeverity.LOW,
            "comp",
            "desc",
        )
        rows = store.audit.query_events()
        assert len(rows) == 0

    def test_in_memory_still_works(self, audit):
        audit.log(
            SecurityEventType.INPUT_INJECTION,
            SecuritySeverity.HIGH,
            "comp",
            "test",
        )
        assert audit.event_count == 1

    def test_query_db(self, audit, store):
        audit.log(SecurityEventType.INPUT_INJECTION, SecuritySeverity.HIGH, "c", "d1", blocked=True)
        audit.log(SecurityEventType.RATE_LIMIT_EXCEEDED, SecuritySeverity.MEDIUM, "c", "d2")
        rows = audit.query_db(severity="high")
        assert len(rows) == 1

    def test_query_db_blocked_only(self, audit, store):
        audit.log(SecurityEventType.INPUT_INJECTION, SecuritySeverity.HIGH, "c", "d1", blocked=True)
        audit.log(SecurityEventType.INPUT_INJECTION, SecuritySeverity.HIGH, "c", "d2", blocked=False)
        rows = audit.query_db(blocked_only=True)
        assert len(rows) == 1

    def test_count_by_severity_db(self, audit, store):
        audit.log(SecurityEventType.INPUT_INJECTION, SecuritySeverity.HIGH, "c", "d1")
        audit.log(SecurityEventType.PROMPT_INJECTION, SecuritySeverity.CRITICAL, "c", "d2")
        audit.log(SecurityEventType.RATE_LIMIT_EXCEEDED, SecuritySeverity.MEDIUM, "c", "d3")
        counts = audit.count_by_severity_db()
        assert counts["high"] == 1
        assert counts["critical"] == 1
        assert counts["medium"] == 1

    def test_query_historical(self, audit, store):
        audit.log(SecurityEventType.INPUT_INJECTION, SecuritySeverity.HIGH, "c", "d")
        rows = audit.query_historical(days_back=1)
        assert len(rows) == 1

    def test_cleanup_old(self, audit, store):
        old_ts = time.time() - 200 * 86400
        store.audit.insert_event("t", "h", "c", "old", timestamp=old_ts)
        deleted = audit.cleanup_old(retention_days=90)
        assert deleted == 1

    def test_export_with_history(self, audit, store):
        audit.log(SecurityEventType.INPUT_INJECTION, SecuritySeverity.HIGH, "c", "d")
        exported = audit.export_with_history(format="json", days_back=7)
        data = json.loads(exported)
        assert "current" in data
        assert "historical" in data
        assert "severity_counts" in data["historical"]

    def test_export_with_history_csv_passthrough(self, audit, store):
        audit.log(SecurityEventType.INPUT_INJECTION, SecuritySeverity.HIGH, "c", "d")
        exported = audit.export_with_history(format="csv")
        assert "timestamp" in exported  # CSV header

    def test_convenience_methods_persist(self, audit, store):
        """Convenience methods (log_injection, etc.) should also persist."""
        audit.log_injection("input", "sanitizer", "SQL detected", tool_name="read")
        rows = store.audit.query_events()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "input_injection"

    def test_log_credential_persists(self, audit, store):
        audit.log_credential("aws_key", tool_name="bash")
        rows = store.audit.query_events()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "credential_detected"

    def test_log_rate_limit_persists(self, audit, store):
        audit.log_rate_limit("bash")
        rows = store.audit.query_events()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "rate_limit_exceeded"

    def test_log_sandbox_violation_persists(self, audit, store):
        audit.log_sandbox_violation("bash", "Memory limit exceeded")
        rows = store.audit.query_events()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "sandbox_violation"

    def test_db_failure_does_not_crash(self, store):
        audit = PersistentAuditLog(store=store)
        store.close()
        # Should not raise — just fails to persist
        audit.log(SecurityEventType.INPUT_INJECTION, SecuritySeverity.HIGH, "c", "d")
        assert audit.event_count == 1


# ── TestPersistentPerformanceBenchmark ───────────────────────────

from cowork_agent.core.persistent_benchmark import PersistentPerformanceBenchmark


class TestPersistentPerformanceBenchmark:
    """Tests for PersistentPerformanceBenchmark write-through behavior."""

    @pytest.fixture
    def bench(self, store):
        return PersistentPerformanceBenchmark(store=store)

    def test_record_persists(self, bench, store):
        bench.record("bench_a", 42.5, component="tool")
        rows = store.benchmarks.query_runs(name="bench_a")
        assert len(rows) == 1
        assert rows[0]["duration_ms"] == 42.5

    def test_record_with_tags(self, bench, store):
        bench.record("bench_a", 42.5, tags=["fast", "tool"])
        rows = store.benchmarks.query_runs()
        assert rows[0]["tags"] == ["fast", "tool"]

    def test_record_failure(self, bench, store):
        bench.record("bench_a", 42.5, success=False)
        rows = store.benchmarks.query_runs()
        assert rows[0]["success"] is False

    def test_persist_disabled(self, store):
        bench = PersistentPerformanceBenchmark(store=store, persist_enabled=False)
        bench.record("a", 10.0)
        rows = store.benchmarks.query_runs()
        assert len(rows) == 0

    def test_in_memory_still_works(self, bench):
        bench.record("a", 10.0)
        stats = bench.get_stats("a")
        assert stats is not None
        assert stats.count == 1

    def test_get_historical_stats(self, bench, store):
        for d in [10.0, 20.0, 30.0]:
            bench.record("b", d)
        stats = bench.get_historical_stats("b", days_back=1)
        assert stats["count"] == 3
        assert stats["avg_ms"] == 20.0

    def test_query_historical_runs(self, bench, store):
        bench.record("a", 10.0, component="tool")
        bench.record("b", 20.0, component="provider")
        runs = bench.query_historical_runs(component="tool", days_back=1)
        assert len(runs) == 1

    def test_query_historical_runs_by_name(self, bench, store):
        bench.record("a", 10.0)
        bench.record("b", 20.0)
        runs = bench.query_historical_runs(name="a", days_back=1)
        assert len(runs) == 1

    def test_cleanup(self, bench, store):
        old_ts = time.time() - 200 * 86400
        store.benchmarks.insert_run("old", 10.0, timestamp=old_ts)
        deleted = bench.cleanup(retention_days=90)
        assert deleted == 1

    def test_export_with_history(self, bench, store):
        bench.record("a", 10.0)
        bench.record("a", 20.0)
        exported = bench.export_with_history(format="json", days_back=7)
        data = json.loads(exported)
        assert "current" in data
        assert "historical" in data

    def test_export_with_history_markdown_passthrough(self, bench):
        bench.record("a", 10.0)
        exported = bench.export_with_history(format="markdown")
        assert "Performance Benchmark Report" in exported

    def test_multiple_records_persist(self, bench, store):
        for i in range(10):
            bench.record("b", float(i + 1))
        rows = store.benchmarks.query_runs(name="b")
        assert len(rows) == 10

    def test_db_failure_does_not_crash(self, store):
        bench = PersistentPerformanceBenchmark(store=store)
        store.close()
        # Should not raise
        run = bench.record("a", 10.0)
        assert run is not None

    def test_record_with_metadata(self, bench, store):
        bench.record("a", 10.0, version="1.0")
        rows = store.benchmarks.query_runs()
        assert rows[0]["metadata"].get("version") == "1.0"


# ── TestConfigWiring ─────────────────────────────────────────────

from cowork_agent.config.settings import load_config


class TestConfigWiring:
    """Tests for Sprint 19 config section."""

    def test_persistent_storage_section_exists(self):
        config = load_config()
        ps = config.get("persistent_storage", {})
        assert isinstance(ps, dict)

    def test_persistent_storage_enabled(self):
        config = load_config()
        assert config.get("persistent_storage.enabled", True) is True

    def test_persistent_storage_retention_days(self):
        config = load_config()
        assert config.get("persistent_storage.retention_days", 90) == 90

    def test_persistent_storage_auto_cleanup(self):
        config = load_config()
        assert config.get("persistent_storage.auto_cleanup", True) is True

    def test_persistent_storage_db_name(self):
        config = load_config()
        assert config.get("persistent_storage.db_name", "metrics.db") == "metrics.db"


# ── TestAgentIntegration ─────────────────────────────────────────

from cowork_agent.core.agent import Agent
from cowork_agent.core.prompt_builder import PromptBuilder


class TestAgentIntegration:
    """Tests for Sprint 19 agent integration."""

    def _make_agent(self):
        builder = PromptBuilder(config={"agent": {"workspace_dir": "/tmp"}})
        return Agent(
            provider=None,
            registry=None,
            prompt_builder=builder,
            workspace_dir="/tmp",
        )

    def test_agent_has_persistent_store_attr(self):
        agent = self._make_agent()
        assert hasattr(agent, "persistent_store")
        assert agent.persistent_store is None

    def test_agent_persistent_store_assignable(self, tmp_dir):
        agent = self._make_agent()
        store = PersistentStore(base_path=tmp_dir)
        agent.persistent_store = store
        assert agent.persistent_store is store
        store.close()

    def test_persistent_metrics_registry_assignable(self, tmp_dir):
        agent = self._make_agent()
        store = PersistentStore(base_path=tmp_dir)
        pmr = PersistentMetricsRegistry(store=store)
        agent.metrics_registry = pmr
        assert isinstance(agent.metrics_registry, PersistentMetricsRegistry)
        store.close()

    def test_persistent_audit_log_assignable(self, tmp_dir):
        agent = self._make_agent()
        store = PersistentStore(base_path=tmp_dir)
        pal = PersistentAuditLog(store=store)
        agent.security_audit_log = pal
        assert isinstance(agent.security_audit_log, PersistentAuditLog)
        store.close()

    def test_persistent_benchmark_assignable(self, tmp_dir):
        agent = self._make_agent()
        store = PersistentStore(base_path=tmp_dir)
        ppb = PersistentPerformanceBenchmark(store=store)
        agent.benchmark = ppb
        assert isinstance(agent.benchmark, PersistentPerformanceBenchmark)
        store.close()


# ── TestEdgeCases ────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and stress tests for persistent storage."""

    def test_concurrent_inserts(self, store):
        """Multiple rapid inserts should all succeed."""
        for i in range(100):
            store.metrics.insert_token_usage("p", "m", i, i)
        assert len(store.metrics.query_token_usage(limit=200)) == 100

    def test_large_metadata(self, store):
        """Large metadata blobs should round-trip correctly."""
        meta = {"key_" + str(i): "value_" * 100 for i in range(50)}
        store.audit.insert_event("t", "h", "c", "d", metadata=meta)
        rows = store.audit.query_events()
        assert len(rows[0]["metadata"]) == 50

    def test_unicode_content(self, store):
        """Unicode in descriptions and metadata."""
        store.audit.insert_event("t", "h", "c", "日本語テスト 🔒")
        rows = store.audit.query_events()
        assert "日本語" in rows[0]["description"]

    def test_empty_strings(self, store):
        """Empty strings handled gracefully."""
        store.metrics.insert_error("", "", "")
        rows = store.metrics.query_errors()
        assert len(rows) == 1
        assert rows[0]["provider"] == ""

    def test_very_large_numbers(self, store):
        store.metrics.insert_token_usage("p", "m", 999999999, 999999999)
        rows = store.metrics.query_token_usage()
        assert rows[0]["input_tokens"] == 999999999

    def test_zero_duration(self, store):
        store.benchmarks.insert_run("b", 0.0)
        rows = store.benchmarks.query_runs()
        assert rows[0]["duration_ms"] == 0.0

    def test_negative_duration(self, store):
        """Negative durations should be stored (validation is caller's job)."""
        store.benchmarks.insert_run("b", -1.0)
        rows = store.benchmarks.query_runs()
        assert rows[0]["duration_ms"] == -1.0

    def test_delete_all(self, store):
        """Deleting everything should leave tables empty."""
        for i in range(5):
            store.metrics.insert_token_usage("p", "m", i, i)
            store.audit.insert_event("t", "h", "c", f"d{i}")
            store.benchmarks.insert_run("b", float(i))
        future = time.time() + 10000
        store.metrics.delete_before(future)
        store.audit.delete_before(future)
        store.benchmarks.delete_before(future)
        stats = store.stats()
        assert stats["token_usage_rows"] == 0
        assert stats["audit_event_rows"] == 0
        assert stats["benchmark_run_rows"] == 0

    def test_query_empty_tables(self, store):
        """Queries on empty tables should return empty lists."""
        assert store.metrics.query_token_usage() == []
        assert store.metrics.query_errors() == []
        assert store.metrics.query_provider_calls() == []
        assert store.audit.query_events() == []
        assert store.benchmarks.query_runs() == []

    def test_stats_db_size(self, store):
        """DB size should be positive after inserts."""
        store.metrics.insert_token_usage("p", "m", 10, 5)
        stats = store.stats()
        assert stats["db_size_bytes"] > 0

    def test_special_characters_in_strings(self, store):
        """Strings with quotes and special chars."""
        store.audit.insert_event("t", "h", "c", 'desc with "quotes" and \'apostrophes\'')
        rows = store.audit.query_events()
        assert '"quotes"' in rows[0]["description"]

    def test_null_metadata(self, store):
        """None metadata should become empty dict."""
        store.audit.insert_event("t", "h", "c", "d", metadata=None)
        rows = store.audit.query_events()
        assert rows[0]["metadata"] == {}

    def test_empty_tags(self, store):
        store.benchmarks.insert_run("b", 10.0, tags=[])
        rows = store.benchmarks.query_runs()
        assert rows[0]["tags"] == []

    def test_persistence_across_reopens(self, tmp_dir):
        """Data survives close/reopen cycles."""
        s1 = PersistentStore(base_path=tmp_dir)
        s1.metrics.insert_token_usage("p", "m", 100, 50)
        s1.audit.insert_event("t", "h", "c", "d")
        s1.benchmarks.insert_run("b", 42.0)
        s1.close()

        s2 = PersistentStore(base_path=tmp_dir)
        assert len(s2.metrics.query_token_usage()) == 1
        assert len(s2.audit.query_events()) == 1
        assert len(s2.benchmarks.query_runs()) == 1
        s2.close()
