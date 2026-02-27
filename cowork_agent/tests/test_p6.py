"""
Sprint 6 — Production Hardening Tests

~70 tests covering:
  - TestErrorCatalog         (14 tests)
  - TestStructuredLogger     (14 tests)
  - TestRetryLayer           (12 tests)
  - TestHealthMonitor        (12 tests)
  - TestShutdownManager      (12 tests)
  - TestSprint6Integration   (6 tests)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Path setup ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cowork_agent.core.error_catalog import (
    AgentError,
    ErrorCatalog,
    ErrorCategory,
    ErrorCode,
)
from cowork_agent.core.structured_logger import (
    HumanFormatter,
    LogContext,
    StructuredFormatter,
    StructuredLogger,
    TraceIDFilter,
    setup_structured_logging,
)
from cowork_agent.core.retry import (
    RetryExecutor,
    RetryPolicy,
    RetryResult,
    retry_async,
    with_retry,
)
from cowork_agent.core.health_monitor import (
    ComponentHealth,
    HealthMonitor,
    HealthReport,
    HealthStatus,
)
from cowork_agent.core.shutdown_manager import (
    ShutdownCallback,
    ShutdownManager,
    ShutdownPhase,
)


# ── Async test helper ───────────────────────────────────────────────

def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════
#  1. ERROR CATALOG — 14 tests
# ═══════════════════════════════════════════════════════════════════

class TestErrorCatalog(unittest.TestCase):
    """Tests for ErrorCatalog, ErrorCode, AgentError."""

    # ── classify_error: network-level ──────────────────────────

    def test_classify_connection_refused(self):
        err = ErrorCatalog.classify_error(ConnectionRefusedError("conn refused"))
        self.assertEqual(err.code, ErrorCode.NETWORK_CONNECTION_REFUSED)
        self.assertEqual(err.category, ErrorCategory.NETWORK)
        self.assertTrue(err.is_transient)

    def test_classify_timeout_error(self):
        err = ErrorCatalog.classify_error(TimeoutError("timed out"))
        self.assertEqual(err.code, ErrorCode.NETWORK_TIMEOUT)
        self.assertTrue(err.is_transient)

    def test_classify_dns_failure(self):
        err = ErrorCatalog.classify_error(OSError("name resolution failed"))
        self.assertEqual(err.code, ErrorCode.NETWORK_DNS_FAILED)
        self.assertTrue(err.is_transient)

    def test_classify_ssl_error(self):
        err = ErrorCatalog.classify_error(OSError("SSL: CERTIFICATE_VERIFY_FAILED"))
        self.assertEqual(err.code, ErrorCode.NETWORK_SSL_ERROR)
        self.assertFalse(err.is_transient)

    # ── classify_error: provider-level heuristics ──────────────

    def test_classify_rate_limit(self):
        err = ErrorCatalog.classify_error(Exception("rate limit exceeded (429)"))
        self.assertEqual(err.code, ErrorCode.PROVIDER_RATE_LIMITED)
        self.assertTrue(err.is_transient)

    def test_classify_auth_failed(self):
        err = ErrorCatalog.classify_error(Exception("authentication failed 401"))
        self.assertEqual(err.code, ErrorCode.PROVIDER_AUTH_FAILED)
        self.assertFalse(err.is_transient)

    def test_classify_model_not_found(self):
        err = ErrorCatalog.classify_error(Exception("model not found: gpt-99"))
        self.assertEqual(err.code, ErrorCode.PROVIDER_MODEL_NOT_FOUND)
        self.assertFalse(err.is_transient)

    def test_classify_overloaded(self):
        err = ErrorCatalog.classify_error(Exception("server overloaded 503"))
        self.assertEqual(err.code, ErrorCode.PROVIDER_OVERLOADED)
        self.assertTrue(err.is_transient)

    def test_classify_fallback_invalid_response(self):
        err = ErrorCatalog.classify_error(Exception("something weird happened"))
        self.assertEqual(err.code, ErrorCode.PROVIDER_INVALID_RESPONSE)

    # ── get_recovery_hint ──────────────────────────────────────

    def test_get_recovery_hint(self):
        hint = ErrorCatalog.get_recovery_hint(ErrorCode.PROVIDER_RATE_LIMITED)
        self.assertIn("retry", hint.lower())

    # ── is_transient ───────────────────────────────────────────

    def test_is_transient_true(self):
        self.assertTrue(ErrorCatalog.is_transient(ConnectionRefusedError("x")))

    def test_is_transient_false(self):
        self.assertFalse(ErrorCatalog.is_transient(Exception("authentication failed")))

    # ── wrap ───────────────────────────────────────────────────

    def test_wrap_with_context(self):
        err = ErrorCatalog.wrap(
            ConnectionRefusedError("refused"),
            context={"agent": "main", "attempt": 2},
        )
        self.assertEqual(err.code, ErrorCode.NETWORK_CONNECTION_REFUSED)
        self.assertEqual(err.context["agent"], "main")
        self.assertEqual(err.context["attempt"], 2)
        self.assertIsNotNone(err.original_exception)

    # ── from_code ──────────────────────────────────────────────

    def test_from_code(self):
        err = ErrorCatalog.from_code(ErrorCode.AGENT_MAX_ITERATIONS)
        self.assertEqual(err.code, ErrorCode.AGENT_MAX_ITERATIONS)
        self.assertEqual(err.category, ErrorCategory.AGENT)
        self.assertFalse(err.is_transient)
        self.assertIn("max_iterations", err.recovery_hint.lower().replace(" ", "_"))


# ═══════════════════════════════════════════════════════════════════
#  2. STRUCTURED LOGGER — 14 tests
# ═══════════════════════════════════════════════════════════════════

class TestStructuredLogger(unittest.TestCase):
    """Tests for StructuredLogger, formatters, filters."""

    def setUp(self):
        # Ensure clean root logger state
        root = logging.getLogger()
        root.handlers.clear()
        root.filters.clear()

    def tearDown(self):
        root = logging.getLogger()
        root.handlers.clear()
        root.filters.clear()

    # ── generate_trace_id ──────────────────────────────────────

    def test_trace_id_length(self):
        tid = StructuredLogger.generate_trace_id()
        self.assertEqual(len(tid), 12)

    def test_trace_id_hex(self):
        tid = StructuredLogger.generate_trace_id()
        int(tid, 16)  # Should not raise

    def test_trace_id_unique(self):
        ids = {StructuredLogger.generate_trace_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    # ── LogContext ─────────────────────────────────────────────

    def test_log_context_merged_with(self):
        ctx = LogContext(trace_id="abc", agent_name="main")
        ctx2 = ctx.merged_with(tool_name="bash")
        self.assertEqual(ctx2.trace_id, "abc")
        self.assertEqual(ctx2.agent_name, "main")
        self.assertEqual(ctx2.tool_name, "bash")
        # Original unchanged
        self.assertEqual(ctx.tool_name, "")

    # ── with_context / bind ────────────────────────────────────

    def test_with_context_returns_new_logger(self):
        lg1 = StructuredLogger("test1")
        lg2 = lg1.with_context(trace_id="t1")
        self.assertIsNot(lg1, lg2)
        self.assertEqual(lg2.context.trace_id, "t1")
        self.assertEqual(lg1.context.trace_id, "")

    def test_bind_mutates_current(self):
        lg = StructuredLogger("test2")
        lg.bind(agent_name="worker")
        self.assertEqual(lg.context.agent_name, "worker")

    # ── StructuredFormatter (JSON) ─────────────────────────────

    def test_json_formatter_output(self):
        fmt = StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        record.trace_id = "abc123def456"
        record.agent_name = "main"
        record.tool_name = ""
        record.provider_name = ""
        output = fmt.format(record)
        data = json.loads(output)
        self.assertEqual(data["message"], "hello world")
        self.assertEqual(data["trace_id"], "abc123def456")
        self.assertEqual(data["agent_name"], "main")
        self.assertNotIn("tool_name", data)  # Empty fields omitted

    def test_json_formatter_extra(self):
        fmt = StructuredFormatter()
        record = logging.LogRecord(
            name="t", level=logging.WARNING, pathname="", lineno=0,
            msg="warn", args=(), exc_info=None,
        )
        record.trace_id = ""
        record.log_extra = {"key": "val"}
        output = fmt.format(record)
        data = json.loads(output)
        self.assertEqual(data["extra"]["key"], "val")

    # ── HumanFormatter ─────────────────────────────────────────

    def test_human_formatter_with_trace(self):
        fmt = HumanFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        record.trace_id = "abc"
        output = fmt.format(record)
        self.assertIn("[abc]", output)
        self.assertIn("hello", output)

    def test_human_formatter_without_trace(self):
        fmt = HumanFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        record.trace_id = ""
        output = fmt.format(record)
        # Without trace_id, the message should NOT have a [trace_id] prefix
        # The output looks like: "HH:MM:SS [INFO] test: hello"
        # We just check that "hello" is NOT prefixed with a trace bracket
        self.assertNotIn("[hello", output)  # No trace_id wrapping the message
        self.assertIn("hello", output)

    # ── TraceIDFilter ──────────────────────────────────────────

    def test_trace_id_filter_injects(self):
        filt = TraceIDFilter(context=LogContext(trace_id="xyz"))
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="", lineno=0,
            msg="m", args=(), exc_info=None,
        )
        filt.filter(record)
        self.assertEqual(record.trace_id, "xyz")

    # ── setup_structured_logging ───────────────────────────────

    def test_setup_json_mode(self):
        setup_structured_logging(json_mode=True, level="DEBUG")
        root = logging.getLogger()
        self.assertEqual(len(root.handlers), 1)
        self.assertIsInstance(root.handlers[0].formatter, StructuredFormatter)

    def test_setup_human_mode(self):
        setup_structured_logging(json_mode=False, level="INFO")
        root = logging.getLogger()
        self.assertEqual(len(root.handlers), 1)
        self.assertIsInstance(root.handlers[0].formatter, HumanFormatter)


# ═══════════════════════════════════════════════════════════════════
#  3. RETRY LAYER — 12 tests
# ═══════════════════════════════════════════════════════════════════

class TestRetryLayer(unittest.TestCase):
    """Tests for RetryExecutor, with_retry decorator, retry_async."""

    # ── Basic success ──────────────────────────────────────────

    def test_success_first_attempt(self):
        async def ok():
            return "done"
        executor = RetryExecutor(RetryPolicy(max_attempts=3))
        result = _run(executor.execute(ok))
        self.assertTrue(result.success)
        self.assertEqual(result.result, "done")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(result.errors), 0)

    # ── Retry then succeed ─────────────────────────────────────

    def test_retry_then_succeed(self):
        call_count = [0]
        async def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("down")
            return "recovered"

        policy = RetryPolicy(max_attempts=5, backoff_base=0.01, jitter=False)
        result = _run(RetryExecutor(policy).execute(flaky))
        self.assertTrue(result.success)
        self.assertEqual(result.result, "recovered")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(len(result.errors), 2)

    # ── Max attempts exhausted ─────────────────────────────────

    def test_max_attempts_exhausted(self):
        async def always_fail():
            raise ConnectionError("nope")

        policy = RetryPolicy(max_attempts=3, backoff_base=0.01, jitter=False)
        result = _run(RetryExecutor(policy).execute(always_fail))
        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(len(result.errors), 3)
        self.assertIsInstance(result.last_error, ConnectionError)

    # ── Non-retryable error stops immediately ──────────────────

    def test_non_retryable_stops(self):
        async def auth_fail():
            raise ValueError("authentication failed 401")

        policy = RetryPolicy(
            max_attempts=5,
            backoff_base=0.01,
            transient_only=True,
            retryable_exceptions=(ConnectionError,),
        )
        result = _run(RetryExecutor(policy).execute(auth_fail))
        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 1)  # Stopped after first

    # ── Backoff calculation ────────────────────────────────────

    def test_backoff_calculation(self):
        policy = RetryPolicy(
            backoff_base=1.0, backoff_multiplier=2.0, backoff_max=10.0, jitter=False,
        )
        executor = RetryExecutor(policy)
        self.assertAlmostEqual(executor._calculate_delay(1), 1.0)
        self.assertAlmostEqual(executor._calculate_delay(2), 2.0)
        self.assertAlmostEqual(executor._calculate_delay(3), 4.0)
        self.assertAlmostEqual(executor._calculate_delay(4), 8.0)
        self.assertAlmostEqual(executor._calculate_delay(5), 10.0)  # capped

    def test_backoff_with_jitter(self):
        policy = RetryPolicy(
            backoff_base=1.0, backoff_multiplier=2.0, jitter=True,
        )
        executor = RetryExecutor(policy)
        delays = [executor._calculate_delay(1) for _ in range(20)]
        # With jitter, delays should vary (1.0 to 1.5)
        self.assertTrue(min(delays) >= 1.0)
        self.assertTrue(max(delays) <= 1.5)
        # Not all the same
        self.assertTrue(len(set(delays)) > 1)

    # ── with_retry decorator ───────────────────────────────────

    def test_decorator_success(self):
        @with_retry(RetryPolicy(max_attempts=3, backoff_base=0.01))
        async def greet():
            return "hi"

        result = _run(greet())
        self.assertEqual(result, "hi")

    def test_decorator_raises_on_exhaustion(self):
        @with_retry(RetryPolicy(max_attempts=2, backoff_base=0.01, jitter=False))
        async def fail():
            raise ConnectionError("boom")

        with self.assertRaises(ConnectionError):
            _run(fail())

    # ── retry_async standalone ─────────────────────────────────

    def test_retry_async_success(self):
        async def ok():
            return 42
        result = _run(retry_async(ok, policy=RetryPolicy(max_attempts=2)))
        self.assertEqual(result, 42)

    def test_retry_async_raises(self):
        async def fail():
            raise TimeoutError("slow")
        with self.assertRaises(TimeoutError):
            _run(retry_async(fail, policy=RetryPolicy(max_attempts=2, backoff_base=0.01, jitter=False)))

    # ── RetryResult tracks total_delay ─────────────────────────

    def test_total_delay_tracked(self):
        call_count = [0]
        async def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("x")
            return "ok"

        policy = RetryPolicy(max_attempts=3, backoff_base=0.05, jitter=False)
        result = _run(RetryExecutor(policy).execute(flaky))
        self.assertTrue(result.success)
        self.assertGreater(result.total_delay, 0.0)

    # ── should_retry respects retryable_exceptions ─────────────

    def test_should_retry_exception_types(self):
        policy = RetryPolicy(
            max_attempts=5,
            retryable_exceptions=(ConnectionError,),
            transient_only=False,
        )
        executor = RetryExecutor(policy)
        self.assertTrue(executor._should_retry(ConnectionError("x"), 1))
        self.assertFalse(executor._should_retry(ValueError("x"), 1))


# ═══════════════════════════════════════════════════════════════════
#  4. HEALTH MONITOR — 12 tests
# ═══════════════════════════════════════════════════════════════════

class TestHealthMonitor(unittest.TestCase):
    """Tests for HealthMonitor, HealthReport, HealthStatus."""

    def setUp(self):
        self.monitor = HealthMonitor()

    # ── Registration ───────────────────────────────────────────

    def test_register_component(self):
        async def check():
            return {"status": "ok"}
        self.monitor.register_component("comp1", check)
        self.assertIn("comp1", self.monitor.component_names)

    def test_unregister_component(self):
        async def check():
            return {"status": "ok"}
        self.monitor.register_component("comp1", check)
        self.monitor.unregister_component("comp1")
        self.assertNotIn("comp1", self.monitor.component_names)

    def test_unregister_nonexistent(self):
        # Should not raise
        self.monitor.unregister_component("ghost")

    # ── check_health: all healthy ──────────────────────────────

    def test_all_healthy(self):
        async def ok():
            return {"status": "ok"}
        self.monitor.register_component("a", ok)
        self.monitor.register_component("b", ok)
        report = _run(self.monitor.check_health())
        self.assertEqual(report.status, HealthStatus.HEALTHY)
        self.assertEqual(len(report.components), 2)

    # ── check_health: degraded ─────────────────────────────────

    def test_degraded_status(self):
        async def ok():
            return {"status": "ok"}
        async def bad():
            return {"status": "error"}
        self.monitor.register_component("good", ok)
        self.monitor.register_component("bad", bad)
        report = _run(self.monitor.check_health())
        self.assertEqual(report.status, HealthStatus.DEGRADED)

    # ── check_health: all unhealthy ────────────────────────────

    def test_all_unhealthy(self):
        async def bad():
            return {"status": "error"}
        self.monitor.register_component("a", bad)
        self.monitor.register_component("b", bad)
        report = _run(self.monitor.check_health())
        self.assertEqual(report.status, HealthStatus.UNHEALTHY)

    # ── check_health: component exception ──────────────────────

    def test_component_exception(self):
        async def explode():
            raise RuntimeError("kaboom")
        self.monitor.register_component("broken", explode)
        report = _run(self.monitor.check_health())
        self.assertEqual(report.status, HealthStatus.UNHEALTHY)
        self.assertIn("kaboom", report.components[0].details.get("error", ""))

    # ── check_health: component timeout ────────────────────────

    def test_component_timeout(self):
        async def slow():
            await asyncio.sleep(60)
            return {"status": "ok"}
        self.monitor.register_component("slow", slow)
        report = _run(self.monitor.check_health())
        self.assertEqual(report.components[0].status, HealthStatus.UNHEALTHY)
        self.assertIn("timed out", report.components[0].details.get("error", "").lower())

    # ── response time tracking ─────────────────────────────────

    def test_response_time_tracked(self):
        async def ok():
            await asyncio.sleep(0.05)
            return {"status": "ok"}
        self.monitor.register_component("slow-ish", ok)
        report = _run(self.monitor.check_health())
        self.assertGreater(report.components[0].response_time_ms, 0)

    # ── uptime ─────────────────────────────────────────────────

    def test_uptime_seconds(self):
        time.sleep(0.05)
        report = _run(self.monitor.check_health())
        self.assertGreater(report.uptime_seconds, 0.0)

    # ── liveness ───────────────────────────────────────────────

    def test_liveness_always_true(self):
        self.assertTrue(_run(self.monitor.check_liveness()))

    # ── readiness ──────────────────────────────────────────────

    def test_readiness_healthy(self):
        async def ok():
            return {"status": "ok"}
        self.monitor.register_component("provider", ok)
        self.assertTrue(_run(self.monitor.check_readiness()))

    def test_readiness_false_during_shutdown(self):
        self.monitor.set_shutting_down(True)
        self.assertFalse(_run(self.monitor.check_readiness()))

    # ── to_dict ────────────────────────────────────────────────

    def test_report_to_dict(self):
        async def ok():
            return {"status": "ok"}
        self.monitor.register_component("x", ok)
        report = _run(self.monitor.check_health())
        d = report.to_dict()
        self.assertEqual(d["status"], "healthy")
        self.assertIn("components", d)
        self.assertIn("uptime_seconds", d)

    # ── get_last_report ────────────────────────────────────────

    def test_get_last_report(self):
        self.assertIsNone(self.monitor.get_last_report())
        _run(self.monitor.check_health())
        self.assertIsNotNone(self.monitor.get_last_report())


# ═══════════════════════════════════════════════════════════════════
#  5. SHUTDOWN MANAGER — 12 tests
# ═══════════════════════════════════════════════════════════════════

class TestShutdownManager(unittest.TestCase):
    """Tests for ShutdownManager, phases, callbacks."""

    def setUp(self):
        self.mgr = ShutdownManager(drain_timeout=1.0)

    # ── Initial state ──────────────────────────────────────────

    def test_initial_phase(self):
        self.assertEqual(self.mgr.phase, ShutdownPhase.RUNNING)
        self.assertFalse(self.mgr.is_shutting_down)

    # ── Register / unregister ──────────────────────────────────

    def test_register_callback(self):
        self.mgr.register_callback("test", lambda: None)
        self.assertIn("test", self.mgr.callback_names)

    def test_unregister_callback(self):
        self.mgr.register_callback("test", lambda: None)
        self.mgr.unregister_callback("test")
        self.assertNotIn("test", self.mgr.callback_names)

    # ── Shutdown completes ─────────────────────────────────────

    def test_shutdown_completes(self):
        _run(self.mgr.shutdown(reason="test"))
        self.assertEqual(self.mgr.phase, ShutdownPhase.COMPLETED)
        self.assertTrue(self.mgr.is_shutting_down)
        self.assertEqual(self.mgr.shutdown_reason, "test")

    # ── Shutdown is idempotent ─────────────────────────────────

    def test_shutdown_idempotent(self):
        _run(self.mgr.shutdown(reason="first"))
        _run(self.mgr.shutdown(reason="second"))
        self.assertEqual(self.mgr.shutdown_reason, "first")

    # ── Callback execution order (priority) ────────────────────

    def test_callback_priority_order(self):
        order = []
        self.mgr.register_callback("low", lambda: order.append("low"), priority=1)
        self.mgr.register_callback("high", lambda: order.append("high"), priority=10)
        self.mgr.register_callback("mid", lambda: order.append("mid"), priority=5)
        _run(self.mgr.shutdown())
        self.assertEqual(order, ["high", "mid", "low"])

    # ── Async callback ─────────────────────────────────────────

    def test_async_callback(self):
        results = []
        async def cleanup():
            results.append("cleaned")
        self.mgr.register_callback("async_cb", cleanup)
        _run(self.mgr.shutdown())
        self.assertEqual(results, ["cleaned"])

    # ── Callback timeout ───────────────────────────────────────

    def test_callback_timeout(self):
        async def slow():
            await asyncio.sleep(60)
        self.mgr.register_callback("slow", slow, timeout=0.1)
        _run(self.mgr.shutdown())
        # Should complete despite timeout
        self.assertEqual(self.mgr.phase, ShutdownPhase.COMPLETED)
        self.assertEqual(self.mgr.results[0]["status"], "timeout")

    # ── Callback error doesn't stop others ─────────────────────

    def test_callback_error_continues(self):
        order = []
        def good():
            order.append("good")
        def bad():
            raise RuntimeError("fail")
        self.mgr.register_callback("bad", bad, priority=10)
        self.mgr.register_callback("good", good, priority=1)
        _run(self.mgr.shutdown())
        self.assertIn("good", order)
        self.assertEqual(self.mgr.results[0]["status"], "error")
        self.assertEqual(self.mgr.results[1]["status"], "ok")

    # ── shutdown_event is set ──────────────────────────────────

    def test_shutdown_event_set(self):
        self.assertFalse(self.mgr.shutdown_event.is_set())
        _run(self.mgr.shutdown())
        self.assertTrue(self.mgr.shutdown_event.is_set())

    # ── reset ──────────────────────────────────────────────────

    def test_reset(self):
        _run(self.mgr.shutdown())
        self.mgr.reset()
        self.assertEqual(self.mgr.phase, ShutdownPhase.RUNNING)
        self.assertFalse(self.mgr.is_shutting_down)
        self.assertFalse(self.mgr.shutdown_event.is_set())

    # ── Results tracking ───────────────────────────────────────

    def test_results_tracking(self):
        self.mgr.register_callback("a", lambda: None, priority=2)
        self.mgr.register_callback("b", lambda: None, priority=1)
        _run(self.mgr.shutdown())
        self.assertEqual(len(self.mgr.results), 2)
        self.assertEqual(self.mgr.results[0]["name"], "a")
        self.assertEqual(self.mgr.results[1]["name"], "b")


# ═══════════════════════════════════════════════════════════════════
#  6. INTEGRATION — 6 tests
# ═══════════════════════════════════════════════════════════════════

class TestSprint6Integration(unittest.TestCase):
    """Cross-feature integration tests."""

    # ── Error catalog + retry chain ────────────────────────────

    def test_error_catalog_drives_retry(self):
        """ErrorCatalog.is_transient should allow retry for connection errors."""
        call_count = [0]
        async def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionRefusedError("refused")
            return "ok"

        policy = RetryPolicy(
            max_attempts=3,
            backoff_base=0.01,
            jitter=False,
            transient_only=True,
        )
        result = _run(RetryExecutor(policy).execute(flaky))
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)

    def test_non_transient_error_stops_retry(self):
        """Non-transient error (auth) should not be retried."""
        async def auth_fail():
            raise Exception("authentication failed 401")

        policy = RetryPolicy(
            max_attempts=5,
            backoff_base=0.01,
            transient_only=True,
            retryable_exceptions=(),  # Only rely on ErrorCatalog
        )
        result = _run(RetryExecutor(policy).execute(auth_fail))
        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 1)

    # ── Health monitor + shutdown integration ──────────────────

    def test_health_readiness_after_shutdown(self):
        """Readiness should be False once shutdown manager signals shutdown."""
        monitor = HealthMonitor()
        async def ok():
            return {"status": "ok"}
        monitor.register_component("provider", ok)

        self.assertTrue(_run(monitor.check_readiness()))

        # Signal shutdown
        monitor.set_shutting_down(True)
        self.assertFalse(_run(monitor.check_readiness()))

    # ── Shutdown + health wiring ───────────────────────────────

    def test_shutdown_triggers_health_update(self):
        """Shutdown manager callback can update health monitor."""
        monitor = HealthMonitor()
        mgr = ShutdownManager()

        mgr.register_callback(
            "health_shutdown",
            lambda: monitor.set_shutting_down(True),
            priority=100,
        )
        _run(mgr.shutdown(reason="test"))
        self.assertFalse(_run(monitor.check_readiness()))

    # ── Structured logger + error catalog ──────────────────────

    def test_logger_with_error_context(self):
        """StructuredLogger can incorporate error catalog context."""
        err = ErrorCatalog.classify_error(ConnectionRefusedError("refused"))
        lg = StructuredLogger("integration").with_context(
            trace_id="abc123",
            agent_name="main",
        )
        # Just verify no exceptions and context is set
        self.assertEqual(lg.context.trace_id, "abc123")
        self.assertEqual(err.code, ErrorCode.NETWORK_CONNECTION_REFUSED)

    # ── Full startup-shutdown cycle ────────────────────────────

    def test_full_startup_shutdown_cycle(self):
        """Simulate a mini lifecycle: create components, check health, shutdown."""
        # Setup
        monitor = HealthMonitor()
        mgr = ShutdownManager()
        cleanup_log = []

        async def provider_check():
            return {"status": "ok", "model": "test"}

        monitor.register_component("provider", provider_check)
        mgr.register_callback(
            "mark_shutdown",
            lambda: monitor.set_shutting_down(True),
            priority=100,
        )
        mgr.register_callback(
            "cleanup_log",
            lambda: cleanup_log.append("done"),
            priority=50,
        )

        # Pre-shutdown: healthy + ready
        report = _run(monitor.check_health())
        self.assertEqual(report.status, HealthStatus.HEALTHY)
        self.assertTrue(_run(monitor.check_readiness()))

        # Shutdown
        _run(mgr.shutdown(reason="test_cycle"))
        self.assertEqual(mgr.phase, ShutdownPhase.COMPLETED)
        self.assertFalse(_run(monitor.check_readiness()))
        self.assertEqual(cleanup_log, ["done"])


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
