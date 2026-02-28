"""
Sprint 16 — Testing & Observability Hardening

Tests for ObservabilityEventBus, CorrelationIdManager, MetricsRegistry,
PerformanceBenchmark, TestCoverageCollector, IntegratedHealthOrchestrator,
and integration with agent/config.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# ── Imports ──────────────────────────────────────────────────────

from cowork_agent.core.observability_event_bus import (
    ObservabilityEventBus, ObservabilityEvent, EventType,
)
from cowork_agent.core.correlation_id_manager import (
    CorrelationIdManager, CorrelationContext,
)
from cowork_agent.core.metrics_registry import (
    MetricsRegistry, TokenUsageMetrics, ProviderErrorRecord,
)
from cowork_agent.core.performance_benchmark import (
    PerformanceBenchmark, BenchmarkRun, BenchmarkStats, ComparisonReport,
)
from cowork_agent.core.test_coverage_collector import (
    CoverageCollector, CheckResult, CheckSuite, CheckStatus, CoverageDetail,
)
from cowork_agent.core.integrated_health_orchestrator import (
    IntegratedHealthOrchestrator, HealthTrend, FailurePrediction,
)
from cowork_agent.core.health_monitor import HealthStatus, HealthReport


# ═══════════════════════════════════════════════════════════════════
# TestObservabilityEventBus
# ═══════════════════════════════════════════════════════════════════

class TestObservabilityEventBus:

    def test_init_defaults(self):
        bus = ObservabilityEventBus()
        assert bus.subscriber_count() == 0
        stats = bus.stats()
        assert stats["total_events_emitted"] == 0

    def test_subscribe_and_emit(self):
        bus = ObservabilityEventBus()
        received = []
        bus.subscribe(EventType.TOOL_CALL_INITIATED, lambda e: received.append(e))
        event = ObservabilityEvent(event_type=EventType.TOOL_CALL_INITIATED, component="bash")
        count = bus.emit(event)
        assert count == 1
        assert len(received) == 1
        assert received[0].component == "bash"

    def test_subscribe_returns_id(self):
        bus = ObservabilityEventBus()
        sub_id = bus.subscribe(EventType.AGENT_STARTED, lambda e: None)
        assert isinstance(sub_id, str)
        assert len(sub_id) == 12

    def test_unsubscribe(self):
        bus = ObservabilityEventBus()
        sub_id = bus.subscribe(EventType.AGENT_STARTED, lambda e: None)
        assert bus.subscriber_count(EventType.AGENT_STARTED) == 1
        assert bus.unsubscribe(sub_id) is True
        assert bus.subscriber_count(EventType.AGENT_STARTED) == 0

    def test_unsubscribe_nonexistent(self):
        bus = ObservabilityEventBus()
        assert bus.unsubscribe("nonexistent") is False

    def test_emit_no_subscribers(self):
        bus = ObservabilityEventBus()
        event = ObservabilityEvent(event_type=EventType.AGENT_STARTED)
        count = bus.emit(event)
        assert count == 0

    def test_emit_multiple_subscribers(self):
        bus = ObservabilityEventBus()
        count_a = []
        count_b = []
        bus.subscribe(EventType.AGENT_STARTED, lambda e: count_a.append(1))
        bus.subscribe(EventType.AGENT_STARTED, lambda e: count_b.append(1))
        event = ObservabilityEvent(event_type=EventType.AGENT_STARTED)
        result = bus.emit(event)
        assert result == 2
        assert len(count_a) == 1
        assert len(count_b) == 1

    def test_event_type_filtering(self):
        bus = ObservabilityEventBus()
        received = []
        bus.subscribe(EventType.TOOL_CALL_INITIATED, lambda e: received.append("tool"))
        bus.subscribe(EventType.AGENT_STARTED, lambda e: received.append("agent"))
        bus.emit(ObservabilityEvent(event_type=EventType.TOOL_CALL_INITIATED))
        assert received == ["tool"]

    def test_callback_exception_handled(self):
        bus = ObservabilityEventBus()
        bus.subscribe(EventType.AGENT_STARTED, lambda e: 1/0)
        event = ObservabilityEvent(event_type=EventType.AGENT_STARTED)
        count = bus.emit(event)
        assert count == 0  # Failed callbacks don't count
        assert bus.stats()["error_count"] == 1

    def test_event_history(self):
        bus = ObservabilityEventBus(event_buffer_size=5)
        for i in range(10):
            bus.emit(ObservabilityEvent(event_type=EventType.AGENT_STARTED, component=str(i)))
        history = bus.get_event_history()
        assert len(history) == 5
        assert history[0].component == "5"  # Oldest in buffer

    def test_event_history_filtered(self):
        bus = ObservabilityEventBus()
        bus.emit(ObservabilityEvent(event_type=EventType.AGENT_STARTED))
        bus.emit(ObservabilityEvent(event_type=EventType.TOOL_CALL_INITIATED))
        bus.emit(ObservabilityEvent(event_type=EventType.AGENT_STARTED))
        history = bus.get_event_history(event_type=EventType.AGENT_STARTED)
        assert len(history) == 2

    def test_max_subscribers_limit(self):
        bus = ObservabilityEventBus(max_subscribers_per_event=2)
        bus.subscribe(EventType.AGENT_STARTED, lambda e: None)
        bus.subscribe(EventType.AGENT_STARTED, lambda e: None)
        with pytest.raises(ValueError):
            bus.subscribe(EventType.AGENT_STARTED, lambda e: None)

    def test_subscribe_all(self):
        bus = ObservabilityEventBus()
        ids = bus.subscribe_all(lambda e: None)
        assert len(ids) == len(EventType)

    def test_stats(self):
        bus = ObservabilityEventBus()
        bus.subscribe(EventType.AGENT_STARTED, lambda e: None)
        bus.subscribe(EventType.TOOL_CALL_INITIATED, lambda e: None)
        bus.emit(ObservabilityEvent(event_type=EventType.AGENT_STARTED))
        stats = bus.stats()
        assert stats["total_subscribers"] == 2
        assert stats["total_events_emitted"] == 1

    def test_clear_history(self):
        bus = ObservabilityEventBus()
        bus.emit(ObservabilityEvent(event_type=EventType.AGENT_STARTED))
        bus.clear_history()
        assert len(bus.get_event_history()) == 0

    def test_reset(self):
        bus = ObservabilityEventBus()
        bus.subscribe(EventType.AGENT_STARTED, lambda e: None)
        bus.emit(ObservabilityEvent(event_type=EventType.AGENT_STARTED))
        bus.reset()
        assert bus.subscriber_count() == 0
        assert bus.stats()["total_events_emitted"] == 0

    def test_event_to_dict(self):
        event = ObservabilityEvent(
            event_type=EventType.TOOL_CALL_COMPLETED,
            component="bash",
            trace_id="abc123",
            severity="info",
            metadata={"duration_ms": 42},
        )
        d = event.to_dict()
        assert d["event_type"] == "tool_call_completed"
        assert d["component"] == "bash"
        assert d["trace_id"] == "abc123"

    @pytest.mark.asyncio
    async def test_emit_async(self):
        bus = ObservabilityEventBus()
        received = []
        async def on_event(e):
            received.append(e)
        bus.subscribe(EventType.AGENT_STARTED, on_event)
        count = await bus.emit_async(ObservabilityEvent(event_type=EventType.AGENT_STARTED))
        assert count == 1
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_emit_async_exception_handled(self):
        bus = ObservabilityEventBus()
        async def bad_callback(e):
            raise RuntimeError("boom")
        bus.subscribe(EventType.AGENT_STARTED, bad_callback)
        count = await bus.emit_async(ObservabilityEvent(event_type=EventType.AGENT_STARTED))
        assert count == 0
        assert bus.stats()["error_count"] == 1


# ═══════════════════════════════════════════════════════════════════
# TestCorrelationIdManager
# ═══════════════════════════════════════════════════════════════════

class TestCorrelationIdManager:

    def test_generate_trace_id(self):
        mgr = CorrelationIdManager()
        tid = mgr.generate_trace_id()
        assert isinstance(tid, str)
        assert len(tid) == 12

    def test_generate_unique_ids(self):
        mgr = CorrelationIdManager()
        ids = {mgr.generate_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_current_context_none_by_default(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()  # Reset any state
        assert mgr.current_context() is None

    def test_trace_context_manager(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()
        with mgr.trace("my-op") as ctx:
            assert ctx.trace_id
            assert ctx.depth == 0
            assert ctx.parent_trace_id is None
            current = mgr.current_context()
            assert current.trace_id == ctx.trace_id
        # Context restored after exit
        assert mgr.current_context() is None

    def test_child_trace(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()
        with mgr.trace("parent") as parent:
            with mgr.child_trace("child") as child:
                assert child.parent_trace_id == parent.trace_id
                assert child.depth == 1
                assert child.trace_id != parent.trace_id
            # Back to parent
            assert mgr.current_context().trace_id == parent.trace_id

    def test_deeply_nested_traces(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()
        with mgr.trace("root"):
            with mgr.child_trace("l1"):
                with mgr.child_trace("l2"):
                    with mgr.child_trace("l3"):
                        ctx = mgr.current_context()
                        assert ctx.depth == 3

    def test_child_trace_without_parent(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()
        with mgr.child_trace("orphan") as ctx:
            assert ctx.depth == 0  # Becomes root
            assert ctx.parent_trace_id is None

    def test_set_and_clear_context(self):
        mgr = CorrelationIdManager()
        ctx = CorrelationContext(trace_id="test123", depth=2)
        mgr.set_context(ctx)
        assert mgr.current_context().trace_id == "test123"
        mgr.clear_context()
        assert mgr.current_context() is None

    def test_extract_headers(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()
        with mgr.trace("op") as ctx:
            headers = mgr.extract_headers()
            assert headers["X-Correlation-ID"] == ctx.trace_id

    def test_extract_headers_with_parent(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()
        with mgr.trace("parent"):
            with mgr.child_trace("child"):
                headers = mgr.extract_headers()
                assert "X-Correlation-ID" in headers
                assert "X-Parent-Correlation-ID" in headers

    def test_extract_headers_empty(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()
        assert mgr.extract_headers() == {}

    def test_inject_from_headers(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()
        ctx = mgr.inject_from_headers({"X-Correlation-ID": "incoming123"})
        assert ctx.trace_id == "incoming123"
        assert mgr.current_context().trace_id == "incoming123"

    def test_inject_from_headers_no_trace(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()
        result = mgr.inject_from_headers({"Other-Header": "value"})
        assert result is None

    def test_custom_header_name(self):
        mgr = CorrelationIdManager(header_name="X-Request-ID")
        mgr.clear_context()
        with mgr.trace("op"):
            headers = mgr.extract_headers()
            assert "X-Request-ID" in headers

    def test_stats(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()
        mgr.generate_trace_id()
        mgr.generate_trace_id()
        stats = mgr.stats()
        assert stats["total_generated"] == 2

    def test_context_to_dict(self):
        ctx = CorrelationContext(trace_id="test", parent_trace_id="parent", depth=1)
        d = ctx.to_dict()
        assert d["trace_id"] == "test"
        assert d["parent_trace_id"] == "parent"
        assert d["depth"] == 1
        assert "elapsed_ms" in d

    def test_max_depth_tracked(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()
        with mgr.trace("root"):
            with mgr.child_trace("l1"):
                with mgr.child_trace("l2"):
                    pass
        assert mgr.stats()["max_depth_seen"] == 2


# ═══════════════════════════════════════════════════════════════════
# TestMetricsRegistry
# ═══════════════════════════════════════════════════════════════════

class TestMetricsRegistry:

    def test_inherits_metrics_collector(self):
        reg = MetricsRegistry()
        reg.record_tool_call("bash", 100.0, True)
        assert reg.percentile("bash", 50) == 100.0

    def test_record_token_usage(self):
        reg = MetricsRegistry()
        reg.record_token_usage("anthropic", "claude-3", input_tokens=100, output_tokens=50)
        usage = reg.get_token_usage("anthropic")
        assert usage["total_calls"] == 1
        assert usage["total_input_tokens"] == 100
        assert usage["total_output_tokens"] == 50

    def test_token_usage_aggregation(self):
        reg = MetricsRegistry()
        reg.record_token_usage("anthropic", input_tokens=100, output_tokens=50)
        reg.record_token_usage("anthropic", input_tokens=200, output_tokens=100)
        usage = reg.get_token_usage("anthropic")
        assert usage["total_calls"] == 2
        assert usage["total_input_tokens"] == 300
        assert usage["avg_input_tokens"] == 150.0

    def test_token_usage_all_providers(self):
        reg = MetricsRegistry()
        reg.record_token_usage("anthropic", input_tokens=100)
        reg.record_token_usage("openai", input_tokens=200)
        all_usage = reg.get_token_usage()
        assert "anthropic" in all_usage
        assert "openai" in all_usage

    def test_token_efficiency(self):
        reg = MetricsRegistry()
        reg.record_token_usage("anthropic", input_tokens=100, output_tokens=50)
        eff = reg.token_efficiency("anthropic")
        assert eff == 0.5

    def test_token_efficiency_no_data(self):
        reg = MetricsRegistry()
        assert reg.token_efficiency("unknown") == 0.0

    def test_record_error(self):
        reg = MetricsRegistry()
        reg.record_error("anthropic", "rate_limit", "429 Too Many Requests")
        dist = reg.get_error_distribution("anthropic")
        assert dist["rate_limit"] == 1

    def test_error_type_validation(self):
        reg = MetricsRegistry()
        reg.record_error("anthropic", "made_up_type")
        dist = reg.get_error_distribution("anthropic")
        assert "unknown" in dist  # Invalid type maps to unknown

    def test_multiple_error_types(self):
        reg = MetricsRegistry()
        reg.record_error("anthropic", "rate_limit")
        reg.record_error("anthropic", "rate_limit")
        reg.record_error("anthropic", "timeout")
        dist = reg.get_error_distribution("anthropic")
        assert dist["rate_limit"] == 2
        assert dist["timeout"] == 1

    def test_rolling_error_rate(self):
        reg = MetricsRegistry(error_rate_window_seconds=300)
        reg.record_provider_call("anthropic", 100.0, True)
        reg.record_provider_call("anthropic", 100.0, False)
        rate = reg.rolling_error_rate("anthropic")
        assert rate == 0.5

    def test_rolling_error_rate_no_data(self):
        reg = MetricsRegistry()
        assert reg.rolling_error_rate("unknown") == 0.0

    def test_provider_health_score_healthy(self):
        reg = MetricsRegistry()
        for _ in range(10):
            reg.record_provider_call("anthropic", 100.0, True)
        reg.record_token_usage("anthropic", input_tokens=100, output_tokens=50)
        score = reg.provider_health_score("anthropic")
        assert score >= 0.8

    def test_provider_health_score_unhealthy(self):
        reg = MetricsRegistry()
        for _ in range(10):
            reg.record_provider_call("bad_provider", 100.0, False)
        score = reg.provider_health_score("bad_provider")
        assert score < 0.5

    def test_provider_health_score_no_data(self):
        reg = MetricsRegistry()
        score = reg.provider_health_score("unknown")
        # No data → assumes healthy (1.0 for each component)
        assert score >= 0.8

    def test_export_json(self):
        reg = MetricsRegistry()
        reg.record_tool_call("bash", 50.0, True)
        reg.record_token_usage("anthropic", input_tokens=100)
        exported = reg.export_metrics("json")
        data = json.loads(exported)
        assert "tools" in data
        assert "token_usage" in data

    def test_export_prometheus(self):
        reg = MetricsRegistry()
        reg.record_tool_call("bash", 50.0, True)
        reg.record_token_usage("anthropic", input_tokens=100, output_tokens=50)
        exported = reg.export_metrics("prometheus")
        assert "cowork_tool_calls_total" in exported
        assert "cowork_provider_input_tokens_total" in exported

    def test_reset_clears_extended(self):
        reg = MetricsRegistry()
        reg.record_token_usage("anthropic", input_tokens=100)
        reg.record_error("anthropic", "rate_limit")
        reg.reset()
        assert reg.get_token_usage("anthropic") == {"provider": "anthropic", "total_calls": 0}
        assert reg.get_error_distribution("anthropic") == {}

    def test_token_usage_disabled(self):
        reg = MetricsRegistry(token_usage_tracking=False)
        reg.record_token_usage("anthropic", input_tokens=100)
        assert reg.get_token_usage("anthropic") == {"provider": "anthropic", "total_calls": 0}

    def test_disabled_registry(self):
        reg = MetricsRegistry(enabled=False)
        reg.record_tool_call("bash", 100.0, True)
        reg.record_token_usage("anthropic", input_tokens=100)
        assert reg.summary()["total_tool_calls"] == 0

    def test_token_usage_metrics_dataclass(self):
        t = TokenUsageMetrics(input_tokens=100, output_tokens=50)
        assert t.total_tokens == 150
        d = t.to_dict()
        assert d["total_tokens"] == 150


# ═══════════════════════════════════════════════════════════════════
# TestPerformanceBenchmark
# ═══════════════════════════════════════════════════════════════════

class TestPerformanceBenchmark:

    def test_record_run(self):
        bench = PerformanceBenchmark()
        run = bench.record("tool_bash", 42.5, component="tool")
        assert run.name == "tool_bash"
        assert run.duration_ms == 42.5

    def test_get_stats(self):
        bench = PerformanceBenchmark()
        bench.record("bash", 40.0)
        bench.record("bash", 50.0)
        bench.record("bash", 60.0)
        stats = bench.get_stats("bash")
        assert stats.count == 3
        assert stats.avg_ms == 50.0
        assert stats.min_ms == 40.0
        assert stats.max_ms == 60.0

    def test_get_stats_no_data(self):
        bench = PerformanceBenchmark()
        assert bench.get_stats("nonexistent") is None

    def test_percentiles(self):
        bench = PerformanceBenchmark()
        for i in range(100):
            bench.record("test", float(i))
        stats = bench.get_stats("test")
        assert stats.p95_ms > stats.median_ms
        assert stats.p99_ms >= stats.p95_ms

    def test_std_dev(self):
        bench = PerformanceBenchmark()
        # All same values → std_dev = 0
        for _ in range(10):
            bench.record("constant", 50.0)
        stats = bench.get_stats("constant")
        assert stats.std_dev_ms == 0.0

    def test_success_rate(self):
        bench = PerformanceBenchmark()
        bench.record("test", 10.0, success=True)
        bench.record("test", 10.0, success=True)
        bench.record("test", 10.0, success=False)
        stats = bench.get_stats("test")
        assert abs(stats.success_rate - 2/3) < 0.01

    def test_compare(self):
        bench = PerformanceBenchmark()
        for _ in range(10):
            bench.record("fast", 10.0)
            bench.record("slow", 100.0)
        report = bench.compare("slow", "fast")
        assert report is not None
        assert report.faster == "fast"
        assert report.avg_diff_ms > 0

    def test_compare_insufficient_data(self):
        bench = PerformanceBenchmark()
        bench.record("only_one", 10.0)
        assert bench.compare("only_one", "nonexistent") is None

    def test_get_slowest(self):
        bench = PerformanceBenchmark()
        bench.record("fast", 10.0)
        bench.record("medium", 50.0)
        bench.record("slow", 100.0)
        slowest = bench.get_slowest(2)
        assert len(slowest) == 2
        assert slowest[0]["name"] == "slow"

    def test_get_by_tag(self):
        bench = PerformanceBenchmark()
        bench.record("a", 10.0, tags=["provider"])
        bench.record("b", 20.0, tags=["tool"])
        bench.record("c", 30.0, tags=["provider"])
        by_tag = bench.get_by_tag("provider")
        assert "a" in by_tag
        assert "c" in by_tag
        assert "b" not in by_tag

    def test_get_by_component(self):
        bench = PerformanceBenchmark()
        bench.record("a", 10.0, component="tool")
        bench.record("b", 20.0, component="provider")
        by_comp = bench.get_by_component("tool")
        assert "a" in by_comp
        assert "b" not in by_comp

    def test_detect_regression(self):
        bench = PerformanceBenchmark()
        # 20 historical runs at 50ms
        for _ in range(10):
            bench.record("test", 50.0)
        # 10 recent runs at 100ms (100% regression)
        for _ in range(10):
            bench.record("test", 100.0)
        result = bench.detect_regression("test", window_size=10, threshold_percent=20.0)
        assert result is not None
        assert result["regression"] is True

    def test_detect_no_regression(self):
        bench = PerformanceBenchmark()
        for _ in range(20):
            bench.record("test", 50.0)
        result = bench.detect_regression("test", window_size=10)
        assert result is None  # No regression

    def test_export_json(self):
        bench = PerformanceBenchmark()
        bench.record("test", 42.0)
        exported = bench.export_report("json")
        data = json.loads(exported)
        assert "benchmarks" in data
        assert "test" in data["benchmarks"]

    def test_export_markdown(self):
        bench = PerformanceBenchmark()
        bench.record("test", 42.0)
        md = bench.export_report("markdown")
        assert "# Performance Benchmark Report" in md
        assert "test" in md

    def test_export_markdown_empty(self):
        bench = PerformanceBenchmark()
        md = bench.export_report("markdown")
        assert "No benchmark data" in md

    def test_get_runs(self):
        bench = PerformanceBenchmark()
        bench.record("a", 10.0)
        bench.record("b", 20.0)
        all_runs = bench.get_runs()
        assert len(all_runs) == 2
        a_runs = bench.get_runs("a")
        assert len(a_runs) == 1

    def test_reset(self):
        bench = PerformanceBenchmark()
        bench.record("test", 42.0)
        bench.reset()
        assert bench.get_stats("test") is None

    def test_max_runs_limit(self):
        bench = PerformanceBenchmark(max_runs=5)
        for i in range(10):
            bench.record("test", float(i))
        runs = bench.get_runs("test")
        assert len(runs) == 5

    def test_benchmark_run_to_dict(self):
        run = BenchmarkRun(name="test", duration_ms=42.5, component="tool")
        d = run.to_dict()
        assert d["name"] == "test"
        assert d["duration_ms"] == 42.5

    def test_stats_to_dict(self):
        bench = PerformanceBenchmark()
        bench.record("test", 42.0)
        stats = bench.get_stats("test")
        d = stats.to_dict()
        assert d["count"] == 1
        assert d["avg_ms"] == 42.0

    def test_comparison_report_to_dict(self):
        bench = PerformanceBenchmark()
        for _ in range(5):
            bench.record("a", 10.0)
            bench.record("b", 20.0)
        report = bench.compare("a", "b")
        d = report.to_dict()
        assert "name_a" in d
        assert "faster" in d


# ═══════════════════════════════════════════════════════════════════
# TestTestCoverageCollector
# ═══════════════════════════════════════════════════════════════════

class TestCoverageCollector:

    def _sample_report(self):
        return {
            "root": "cowork_agent",
            "tests": [
                {"nodeid": "tests/test_agent.py::TestAgent::test_run", "outcome": "passed", "duration": 0.1},
                {"nodeid": "tests/test_agent.py::TestAgent::test_fail", "outcome": "failed", "duration": 0.2, "call": {"longrepr": "AssertionError"}},
                {"nodeid": "tests/test_tools.py::TestTools::test_bash", "outcome": "passed", "duration": 0.05},
                {"nodeid": "tests/test_tools.py::TestTools::test_skip", "outcome": "skipped", "duration": 0.0},
            ],
        }

    def test_collect_from_pytest_json(self):
        collector = CoverageCollector()
        suite = collector.collect_from_pytest_json(self._sample_report())
        assert suite.test_count == 4
        assert suite.passed == 2
        assert suite.failed == 1
        assert suite.skipped == 1

    def test_generate_summary(self):
        collector = CoverageCollector()
        collector.collect_from_pytest_json(self._sample_report())
        summary = collector.generate_summary()
        assert summary["total_tests"] == 4
        assert summary["passed"] == 2
        assert summary["pass_rate"] == 0.5

    def test_generate_summary_empty(self):
        collector = CoverageCollector()
        summary = collector.generate_summary()
        assert summary["total_suites"] == 0

    def test_coverage_by_module(self):
        collector = CoverageCollector()
        collector.collect_from_pytest_json(self._sample_report())
        detail = collector.get_coverage_by_module("agent")
        assert detail.test_count >= 1

    def test_identify_uncovered_modules(self):
        collector = CoverageCollector()
        collector.collect_from_pytest_json(self._sample_report())
        uncovered = collector.identify_uncovered_modules(
            ["agent", "tools", "unknown_module"], threshold=0.5,
        )
        assert "unknown_module" in uncovered

    def test_collect_results_directly(self):
        collector = CoverageCollector()
        results = [
            CheckResult(test_id="test_1", status=CheckStatus.PASSED, duration_ms=10),
            CheckResult(test_id="test_2", status=CheckStatus.FAILED, duration_ms=20),
        ]
        suite = collector.collect_results(results, name="manual")
        assert suite.test_count == 2
        assert suite.pass_rate == 0.5

    def test_slowest_tests(self):
        collector = CoverageCollector()
        collector.collect_from_pytest_json(self._sample_report())
        slowest = collector.get_slowest_tests(2)
        assert len(slowest) == 2
        assert slowest[0]["duration_ms"] >= slowest[1]["duration_ms"]

    def test_failed_tests(self):
        collector = CoverageCollector()
        collector.collect_from_pytest_json(self._sample_report())
        failed = collector.get_failed_tests()
        assert len(failed) == 1
        assert "test_fail" in failed[0]["test_id"]

    def test_compare_suites(self):
        collector = CoverageCollector()
        suite_a = collector.collect_from_pytest_json(self._sample_report())
        # Create a bigger suite
        report_b = self._sample_report()
        report_b["tests"].append(
            {"nodeid": "tests/test_new.py::test_new", "outcome": "passed", "duration": 0.01}
        )
        suite_b = collector.collect_from_pytest_json(report_b)
        comparison = collector.compare_suites(suite_a, suite_b)
        assert comparison["test_count_delta"] == 1

    def test_export_json(self):
        collector = CoverageCollector()
        collector.collect_from_pytest_json(self._sample_report())
        exported = collector.export_report("json")
        data = json.loads(exported)
        assert "total_tests" in data
        assert "suites" in data

    def test_export_markdown(self):
        collector = CoverageCollector()
        collector.collect_from_pytest_json(self._sample_report())
        md = collector.export_report("markdown")
        assert "# Test Coverage Report" in md

    def test_reset(self):
        collector = CoverageCollector()
        collector.collect_from_pytest_json(self._sample_report())
        collector.reset()
        assert collector.generate_summary()["total_suites"] == 0

    def test_test_suite_to_dict(self):
        collector = CoverageCollector()
        suite = collector.collect_from_pytest_json(self._sample_report())
        d = suite.to_dict()
        assert d["test_count"] == 4
        assert "modules_covered" in d

    def test_test_result_to_dict(self):
        r = CheckResult(test_id="t1", class_name="Cls", method_name="m", status=CheckStatus.PASSED)
        d = r.to_dict()
        assert d["status"] == "passed"
        assert d["test_id"] == "t1"


# ═══════════════════════════════════════════════════════════════════
# TestIntegratedHealthOrchestrator
# ═══════════════════════════════════════════════════════════════════

class TestIntegratedHealthOrchestrator:

    @pytest.mark.asyncio
    async def test_run_full_check(self):
        orch = IntegratedHealthOrchestrator()
        async def healthy_check():
            return {"status": "ok"}
        orch.register_component("provider", healthy_check)
        report = await orch.run_full_check()
        assert report.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_run_full_check_tracks_trends(self):
        orch = IntegratedHealthOrchestrator()
        async def ok():
            return {"status": "ok"}
        orch.register_component("comp", ok)
        await orch.run_full_check()
        trends = orch.get_trends("comp")
        assert "comp" in trends
        assert trends["comp"]["total_checks"] == 1

    @pytest.mark.asyncio
    async def test_run_component_check(self):
        orch = IntegratedHealthOrchestrator()
        async def ok():
            return {"status": "ok"}
        orch.register_component("comp", ok)
        result = await orch.run_component_check("comp")
        assert result is not None
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_run_component_check_unknown(self):
        orch = IntegratedHealthOrchestrator()
        result = await orch.run_component_check("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_health_check_with_event_bus(self):
        bus = ObservabilityEventBus()
        received = []
        bus.subscribe(EventType.HEALTH_CHECK_RUN, lambda e: received.append(e))
        orch = IntegratedHealthOrchestrator(event_bus=bus)
        async def ok():
            return {"status": "ok"}
        orch.register_component("comp", ok)
        await orch.run_full_check(correlation_id="test123")
        assert len(received) == 1
        assert received[0].trace_id == "test123"

    @pytest.mark.asyncio
    async def test_trend_tracking_multiple_checks(self):
        orch = IntegratedHealthOrchestrator()
        async def ok():
            return {"status": "ok"}
        orch.register_component("comp", ok)
        for _ in range(5):
            await orch.run_full_check()
        trends = orch.get_trends("comp")
        assert trends["comp"]["total_checks"] == 5

    @pytest.mark.asyncio
    async def test_get_check_history(self):
        orch = IntegratedHealthOrchestrator()
        async def ok():
            return {"status": "ok"}
        orch.register_component("comp", ok)
        await orch.run_full_check()
        await orch.run_full_check()
        history = orch.get_check_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_predict_failure_insufficient_data(self):
        orch = IntegratedHealthOrchestrator()
        pred = orch.predict_failure("unknown")
        assert pred is None

    @pytest.mark.asyncio
    async def test_predict_failure_healthy(self):
        orch = IntegratedHealthOrchestrator()
        async def ok():
            return {"status": "ok"}
        orch.register_component("comp", ok)
        for _ in range(10):
            await orch.run_full_check()
        pred = orch.predict_failure("comp")
        assert pred is not None
        assert pred.failure_probability < 0.1
        assert pred.trend_direction == "stable"

    @pytest.mark.asyncio
    async def test_predict_failure_degrading(self):
        orch = IntegratedHealthOrchestrator()
        call_count = [0]
        async def degrading_check():
            call_count[0] += 1
            if call_count[0] > 5:
                return {"status": "unhealthy"}
            return {"status": "ok"}
        orch.register_component("comp", degrading_check)
        for _ in range(10):
            await orch.run_full_check()
        pred = orch.predict_failure("comp")
        assert pred is not None
        assert pred.trend_direction == "degrading"

    def test_enable_events(self):
        orch = IntegratedHealthOrchestrator()
        assert orch._events_enabled is False
        bus = ObservabilityEventBus()
        orch.set_event_bus(bus)
        assert orch._events_enabled is True
        orch.enable_events(False)
        assert orch._events_enabled is False

    def test_orchestrator_stats(self):
        orch = IntegratedHealthOrchestrator()
        async def ok():
            return {"status": "ok"}
        orch.register_component("comp", ok)
        stats = orch.orchestrator_stats()
        assert "comp" in stats["registered_components"]

    def test_reset_trends(self):
        orch = IntegratedHealthOrchestrator()
        orch._trends["comp"] = HealthTrend(component_name="comp")
        orch._trends["comp"].add("healthy", 10.0, time.time())
        orch.reset_trends()
        assert orch.get_trends() == {}

    def test_health_trend_to_dict(self):
        trend = HealthTrend(component_name="test")
        trend.add("healthy", 10.0, time.time())
        trend.add("unhealthy", 50.0, time.time())
        d = trend.to_dict()
        assert d["total_checks"] == 2
        assert d["failure_rate"] == 0.5

    def test_failure_prediction_to_dict(self):
        pred = FailurePrediction(
            component_name="comp", failure_probability=0.3,
            trend_direction="stable", confidence=0.8,
            recommendation="Monitor",
        )
        d = pred.to_dict()
        assert d["failure_probability"] == 0.3


# ═══════════════════════════════════════════════════════════════════
# TestAgentIntegration
# ═══════════════════════════════════════════════════════════════════

class TestAgentIntegration:

    def _make_agent(self):
        from cowork_agent.core.agent import Agent
        from cowork_agent.core.prompt_builder import PromptBuilder
        from cowork_agent.core.tool_registry import ToolRegistry
        provider = MagicMock()
        registry = ToolRegistry()
        prompt_builder = PromptBuilder(config={"agent": {"workspace_dir": "/tmp"}})
        agent = Agent(provider=provider, registry=registry, prompt_builder=prompt_builder)
        return agent

    def test_sprint16_attributes_default_none(self):
        agent = self._make_agent()
        assert agent.event_bus is None
        assert agent.correlation_manager is None
        assert agent.metrics_registry is None
        assert agent.benchmark is None
        assert agent.health_orchestrator is None

    def test_attach_event_bus(self):
        agent = self._make_agent()
        bus = ObservabilityEventBus()
        agent.event_bus = bus
        assert agent.event_bus is bus

    def test_attach_correlation_manager(self):
        agent = self._make_agent()
        mgr = CorrelationIdManager()
        agent.correlation_manager = mgr
        assert agent.correlation_manager is mgr

    def test_attach_metrics_registry(self):
        agent = self._make_agent()
        reg = MetricsRegistry()
        agent.metrics_registry = reg
        assert agent.metrics_registry is reg

    def test_attach_benchmark(self):
        agent = self._make_agent()
        bench = PerformanceBenchmark()
        agent.benchmark = bench
        assert agent.benchmark is bench

    def test_attach_health_orchestrator(self):
        agent = self._make_agent()
        orch = IntegratedHealthOrchestrator()
        agent.health_orchestrator = orch
        assert agent.health_orchestrator is orch

    def test_emit_event_with_bus(self):
        agent = self._make_agent()
        bus = ObservabilityEventBus()
        received = []
        bus.subscribe(EventType.AGENT_STARTED, lambda e: received.append(e))
        agent.event_bus = bus
        agent._emit_observability_event("agent_started", metadata={"test": True})
        assert len(received) == 1
        assert received[0].metadata["test"] is True

    def test_emit_event_without_bus(self):
        agent = self._make_agent()
        # Should not raise
        agent._emit_observability_event("agent_started")

    def test_emit_event_custom_type(self):
        agent = self._make_agent()
        bus = ObservabilityEventBus()
        received = []
        bus.subscribe(EventType.CUSTOM, lambda e: received.append(e))
        agent.event_bus = bus
        agent._emit_observability_event("not_a_real_type")
        assert len(received) == 1

    def test_clear_history_preserves_sprint16(self):
        agent = self._make_agent()
        bus = ObservabilityEventBus()
        agent.event_bus = bus
        agent.clear_history()
        assert agent.event_bus is bus


# ═══════════════════════════════════════════════════════════════════
# TestConfigWiring
# ═══════════════════════════════════════════════════════════════════

class TestConfigWiring:

    def _load_config(self):
        import os
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "default_config.yaml"
        )
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_config_has_observability_section(self):
        cfg = self._load_config()
        assert "observability" in cfg

    def test_config_event_bus(self):
        cfg = self._load_config()
        eb = cfg["observability"]["event_bus"]
        assert eb["enabled"] is True
        assert eb["max_subscribers_per_event"] == 100

    def test_config_correlation_ids(self):
        cfg = self._load_config()
        cid = cfg["observability"]["correlation_ids"]
        assert cid["enabled"] is True
        assert cid["header_name"] == "X-Correlation-ID"

    def test_config_metrics_registry(self):
        cfg = self._load_config()
        mr = cfg["observability"]["metrics_registry"]
        assert mr["enabled"] is True
        assert mr["error_rate_window_seconds"] == 300

    def test_config_performance_benchmarking(self):
        cfg = self._load_config()
        pb = cfg["observability"]["performance_benchmarking"]
        assert pb["enabled"] is True

    def test_config_health_orchestrator(self):
        cfg = self._load_config()
        ho = cfg["observability"]["health_orchestrator"]
        assert ho["enabled"] is True

    def test_config_test_coverage(self):
        cfg = self._load_config()
        tc = cfg["observability"]["test_coverage"]
        assert tc["enabled"] is False  # Disabled by default

    def test_event_bus_created_from_config(self):
        cfg = self._load_config()
        eb_cfg = cfg["observability"]["event_bus"]
        bus = ObservabilityEventBus(
            max_subscribers_per_event=eb_cfg["max_subscribers_per_event"],
            async_emit=eb_cfg["async_emit"],
            event_buffer_size=eb_cfg["event_buffer_size"],
        )
        assert bus._max_per_event == 100

    def test_metrics_registry_from_config(self):
        cfg = self._load_config()
        mr_cfg = cfg["observability"]["metrics_registry"]
        reg = MetricsRegistry(
            error_rate_window_seconds=mr_cfg["error_rate_window_seconds"],
            token_usage_tracking=mr_cfg["token_usage_tracking"],
            detailed_latency_tracking=mr_cfg["detailed_latency_tracking"],
        )
        assert reg._error_rate_window == 300

    def test_health_orchestrator_from_config(self):
        cfg = self._load_config()
        ho_cfg = cfg["observability"]["health_orchestrator"]
        orch = IntegratedHealthOrchestrator(
            max_trend_history=ho_cfg["max_trend_history"],
        )
        assert orch._max_trend_history == 100


# ═══════════════════════════════════════════════════════════════════
# TestEdgeCases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_event_bus_emit_with_sync_callback_for_async(self):
        """Sync emit with async callback should not crash."""
        bus = ObservabilityEventBus()
        async def async_cb(e):
            pass
        bus.subscribe(EventType.AGENT_STARTED, async_cb)
        # Should handle gracefully (close coroutine)
        count = bus.emit(ObservabilityEvent(event_type=EventType.AGENT_STARTED))
        assert count == 1  # Callback invoked but coroutine closed

    def test_correlation_context_metadata(self):
        mgr = CorrelationIdManager()
        mgr.clear_context()
        with mgr.trace("op", request_id="req-1") as ctx:
            assert ctx.metadata["request_id"] == "req-1"

    def test_metrics_registry_bounded_samples(self):
        reg = MetricsRegistry(max_latency_samples=5)
        for i in range(10):
            reg.record_token_usage("prov", input_tokens=i)
        samples = reg._token_usage["prov"]
        assert len(samples) == 5

    def test_benchmark_compare_identical(self):
        bench = PerformanceBenchmark()
        for _ in range(5):
            bench.record("a", 50.0)
            bench.record("b", 50.0)
        report = bench.compare("a", "b")
        assert report.statistically_significant is False

    def test_coverage_collector_malformed_report(self):
        collector = CoverageCollector()
        suite = collector.collect_from_pytest_json({})  # Empty report
        assert suite.test_count == 0

    def test_health_trend_max_history(self):
        trend = HealthTrend(component_name="test", max_history=3)
        for i in range(10):
            trend.add("healthy", float(i), time.time())
        assert len(trend.reports) == 3

    def test_health_trend_failure_rate(self):
        trend = HealthTrend(component_name="test")
        trend.add("healthy", 10.0, time.time())
        trend.add("unhealthy", 50.0, time.time())
        assert trend.failure_rate == 0.5

    def test_metrics_error_distribution_empty(self):
        reg = MetricsRegistry()
        assert reg.get_error_distribution("nonexistent") == {}

    def test_benchmark_detect_regression_insufficient_data(self):
        bench = PerformanceBenchmark()
        bench.record("test", 50.0)
        result = bench.detect_regression("test")
        assert result is None

    def test_coverage_detail_score(self):
        detail = CoverageDetail(module_name="test", test_count=4, pass_count=3, fail_count=1)
        assert detail.coverage_score == 0.75

    def test_coverage_detail_empty(self):
        detail = CoverageDetail(module_name="test")
        assert detail.coverage_score == 0.0

    @pytest.mark.asyncio
    async def test_orchestrator_unhealthy_component(self):
        orch = IntegratedHealthOrchestrator()
        async def unhealthy():
            return {"status": "error", "detail": "connection refused"}
        orch.register_component("failing", unhealthy)
        report = await orch.run_full_check()
        assert report.status == HealthStatus.UNHEALTHY
