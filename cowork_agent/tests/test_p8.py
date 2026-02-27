"""
Sprint 8 — Cross-Theme Hardening Tests

~68 tests covering:
  - TestOutputSanitizer      (14 tests)
  - TestMetricsCollector     (12 tests)
  - TestExecutionTracer      (12 tests)
  - TestToolPermissions      (12 tests)
  - TestRichOutput           (12 tests)
  - TestSprint8Integration   (6 tests)
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest

# ── Path setup ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cowork_agent.core.output_sanitizer import OutputSanitizer, SanitizationResult
from cowork_agent.core.metrics_collector import (
    MetricsCollector,
    ProviderMetrics,
    ToolMetrics,
)
from cowork_agent.core.execution_tracer import ExecutionTracer, Span
from cowork_agent.core.tool_permissions import (
    PermissionProfile,
    ToolPermissionManager,
    ToolQuota,
)
from cowork_agent.interfaces.rich_output import RichOutput


# ═══════════════════════════════════════════════════════════════════
#  1. OUTPUT SANITIZER — 14 tests
# ═══════════════════════════════════════════════════════════════════

class TestOutputSanitizer(unittest.TestCase):
    """Tests for OutputSanitizer: pattern detection, masking, false positives."""

    def setUp(self):
        self.san = OutputSanitizer(show_last_n=4)

    # ── AWS keys ───────────────────────────────────────────────

    def test_detect_aws_access_key(self):
        text = "Found key: AKIAIOSFODNN7EXAMPLE"
        result = self.san.sanitize(text)
        self.assertTrue(result.had_secrets)
        self.assertIn("aws_key", result.detected_types)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", result.sanitized)

    def test_detect_aws_temp_key(self):
        text = "ASIAIOSFODNN7EXAMPLE1"
        result = self.san.sanitize(text)
        self.assertTrue(result.had_secrets)
        self.assertIn("aws_temp_key", result.detected_types)

    # ── API tokens ─────────────────────────────────────────────

    def test_detect_openai_key(self):
        text = "api_key = sk-abcdefghijklmnopqrstuvwx1234567890"
        result = self.san.sanitize(text)
        self.assertTrue(result.had_secrets)
        self.assertIn("api_secret_key", result.detected_types)
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwx", result.sanitized)

    def test_detect_github_pat(self):
        text = "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        result = self.san.sanitize(text)
        self.assertTrue(result.had_secrets)
        self.assertIn("github_pat", result.detected_types)

    # ── JWTs ───────────────────────────────────────────────────

    def test_detect_jwt(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = self.san.sanitize(f"Bearer {jwt}")
        self.assertTrue(result.had_secrets)
        self.assertIn("jwt", result.detected_types)

    # ── Database URIs ──────────────────────────────────────────

    def test_detect_postgres_uri(self):
        text = "DATABASE_URL=postgres://admin:s3cr3t@db.example.com:5432/mydb"
        result = self.san.sanitize(text)
        self.assertTrue(result.had_secrets)
        self.assertIn("db_uri", result.detected_types)

    # ── Password assignments ───────────────────────────────────

    def test_detect_password_assignment(self):
        text = "config.password = 'my_super_secret'"
        result = self.san.sanitize(text)
        self.assertTrue(result.had_secrets)
        self.assertIn("password_assignment", result.detected_types)

    def test_detect_api_key_assignment(self):
        text = 'api_key = "abcdef1234567890abcdef"'
        result = self.san.sanitize(text)
        self.assertTrue(result.had_secrets)

    # ── Masking ────────────────────────────────────────────────

    def test_mask_shows_last_n(self):
        masked = OutputSanitizer.mask("AKIAIOSFODNN7EXAMPLE", 4)
        self.assertEqual(masked, "***MPLE")

    def test_mask_short_secret(self):
        masked = OutputSanitizer.mask("ab", 4)
        self.assertEqual(masked, "***")

    # ── False positives ────────────────────────────────────────

    def test_clean_text_no_false_positive(self):
        text = "This is a normal log message with no secrets."
        result = self.san.sanitize(text)
        self.assertFalse(result.had_secrets)
        self.assertEqual(result.sanitized, text)

    def test_short_hex_not_detected(self):
        # Short hex strings should NOT be flagged
        text = "commit abc123def456"
        self.assertTrue(self.san.is_clean(text))

    # ── Disabled sanitizer ─────────────────────────────────────

    def test_disabled_sanitizer(self):
        san = OutputSanitizer(enabled=False)
        text = "sk-abcdefghijklmnopqrstuvwxyz1234"
        result = san.sanitize(text)
        self.assertFalse(result.had_secrets)
        self.assertEqual(result.sanitized, text)

    # ── detect_secrets method ──────────────────────────────────

    def test_detect_secrets_returns_types(self):
        text = "key: AKIAIOSFODNN7EXAMPLE, token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        types = self.san.detect_secrets(text)
        self.assertIn("aws_key", types)
        self.assertIn("github_pat", types)


# ═══════════════════════════════════════════════════════════════════
#  2. METRICS COLLECTOR — 12 tests
# ═══════════════════════════════════════════════════════════════════

class TestMetricsCollector(unittest.TestCase):
    """Tests for MetricsCollector: recording, percentiles, summaries."""

    def setUp(self):
        self.mc = MetricsCollector()

    # ── Tool recording ─────────────────────────────────────────

    def test_record_tool_call_success(self):
        self.mc.record_tool_call("bash", 100.0, success=True)
        m = self.mc.get_tool_metrics("bash")
        self.assertIn("bash", m)
        self.assertEqual(m["bash"].call_count, 1)
        self.assertEqual(m["bash"].success_count, 1)

    def test_record_tool_call_error(self):
        self.mc.record_tool_call("bash", 50.0, success=False, error="timeout")
        m = self.mc.get_tool_metrics("bash")
        self.assertEqual(m["bash"].error_count, 1)
        self.assertEqual(m["bash"].last_error, "timeout")

    def test_multiple_tool_calls(self):
        for i in range(5):
            self.mc.record_tool_call("read", float(i * 10), success=True)
        m = self.mc.get_tool_metrics("read")
        self.assertEqual(m["read"].call_count, 5)
        self.assertAlmostEqual(m["read"].avg_ms, 20.0)

    def test_get_all_tool_metrics(self):
        self.mc.record_tool_call("bash", 10, True)
        self.mc.record_tool_call("read", 5, True)
        all_m = self.mc.get_tool_metrics()
        self.assertEqual(len(all_m), 2)

    # ── Provider recording ─────────────────────────────────────

    def test_record_provider_call(self):
        self.mc.record_provider_call("ollama", 500.0, success=True)
        p = self.mc.get_provider_metrics("ollama")
        self.assertIn("ollama", p)
        self.assertEqual(p["ollama"].call_count, 1)

    def test_provider_error_rate(self):
        self.mc.record_provider_call("openai", 100, True)
        self.mc.record_provider_call("openai", 200, False)
        p = self.mc.get_provider_metrics("openai")
        self.assertAlmostEqual(p["openai"].error_rate, 0.5)

    # ── Percentiles ────────────────────────────────────────────

    def test_percentile_p50(self):
        for i in range(100):
            self.mc.record_tool_call("bash", float(i), True)
        p50 = self.mc.percentile("bash", 50)
        self.assertGreater(p50, 40)
        self.assertLess(p50, 60)

    def test_percentile_p99(self):
        for i in range(100):
            self.mc.record_tool_call("bash", float(i), True)
        p99 = self.mc.percentile("bash", 99)
        self.assertGreater(p99, 95)

    def test_percentile_no_data(self):
        p = self.mc.percentile("nonexistent", 50)
        self.assertEqual(p, 0.0)

    # ── Summary ────────────────────────────────────────────────

    def test_summary(self):
        self.mc.record_tool_call("bash", 100, True)
        self.mc.record_provider_call("ollama", 500, True)
        s = self.mc.summary()
        self.assertEqual(s["total_tool_calls"], 1)
        self.assertEqual(s["total_provider_calls"], 1)
        self.assertIn("tools", s)
        self.assertIn("providers", s)

    # ── Reset ──────────────────────────────────────────────────

    def test_reset(self):
        self.mc.record_tool_call("bash", 100, True)
        self.mc.reset()
        self.assertEqual(len(self.mc.get_tool_metrics()), 0)

    # ── Disabled ───────────────────────────────────────────────

    def test_disabled_collector(self):
        mc = MetricsCollector(enabled=False)
        mc.record_tool_call("bash", 100, True)
        self.assertEqual(len(mc.get_tool_metrics()), 0)


# ═══════════════════════════════════════════════════════════════════
#  3. EXECUTION TRACER — 12 tests
# ═══════════════════════════════════════════════════════════════════

class TestExecutionTracer(unittest.TestCase):
    """Tests for ExecutionTracer: spans, hierarchy, export."""

    def setUp(self):
        self.tracer = ExecutionTracer()

    # ── Span lifecycle ─────────────────────────────────────────

    def test_start_span(self):
        sid = self.tracer.start_span("agent.run")
        self.assertTrue(sid.startswith("span_"))
        self.assertEqual(self.tracer.span_count, 1)

    def test_end_span_ok(self):
        sid = self.tracer.start_span("agent.run")
        self.tracer.end_span(sid, status="ok")
        span = self.tracer.get_span(sid)
        self.assertEqual(span.status, "ok")
        self.assertIsNotNone(span.end_time)
        self.assertIsNotNone(span.duration_ms)

    def test_end_span_error(self):
        sid = self.tracer.start_span("tool.bash")
        self.tracer.end_span(sid, status="error", error="timeout")
        span = self.tracer.get_span(sid)
        self.assertEqual(span.status, "error")
        self.assertEqual(span.error_message, "timeout")

    def test_end_unknown_span(self):
        # Should not raise, just log warning
        self.tracer.end_span("nonexistent")
        self.assertEqual(self.tracer.span_count, 0)

    # ── Parent-child hierarchy ─────────────────────────────────

    def test_parent_child(self):
        root = self.tracer.start_span("agent.run")
        child = self.tracer.start_span("tool.bash", parent_id=root)
        root_span = self.tracer.get_span(root)
        self.assertIn(child, root_span.children)

    def test_root_span_auto_detected(self):
        root = self.tracer.start_span("agent.run")
        self.assertEqual(self.tracer.root_span_id, root)

    def test_nested_hierarchy(self):
        root = self.tracer.start_span("agent.run")
        prov = self.tracer.start_span("provider.send", parent_id=root)
        tool = self.tracer.start_span("tool.bash", parent_id=root)
        self.assertEqual(self.tracer.span_count, 3)

    # ── Trace tree ─────────────────────────────────────────────

    def test_get_trace_tree(self):
        root = self.tracer.start_span("agent.run")
        child = self.tracer.start_span("tool.bash", parent_id=root)
        self.tracer.end_span(child, "ok")
        self.tracer.end_span(root, "ok")
        tree = self.tracer.get_trace_tree()
        self.assertEqual(tree["operation"], "agent.run")
        self.assertEqual(len(tree["children"]), 1)

    def test_empty_trace_tree(self):
        tree = self.tracer.get_trace_tree()
        self.assertEqual(tree, {})

    # ── Export ─────────────────────────────────────────────────

    def test_to_json(self):
        root = self.tracer.start_span("agent.run")
        self.tracer.end_span(root, "ok")
        j = self.tracer.to_json()
        data = json.loads(j)
        self.assertEqual(data["span_count"], 1)
        self.assertEqual(data["trace_id"], self.tracer.trace_id)

    def test_to_summary(self):
        root = self.tracer.start_span("agent.run")
        self.tracer.end_span(root, "ok")
        s = self.tracer.to_summary()
        self.assertEqual(s["total_spans"], 1)
        self.assertEqual(s["error_spans"], 0)

    # ── Error spans ────────────────────────────────────────────

    def test_get_error_spans(self):
        s1 = self.tracer.start_span("tool.a")
        s2 = self.tracer.start_span("tool.b")
        self.tracer.end_span(s1, "ok")
        self.tracer.end_span(s2, "error", error="fail")
        errors = self.tracer.get_error_spans()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].span_id, s2)


# ═══════════════════════════════════════════════════════════════════
#  4. TOOL PERMISSIONS — 12 tests
# ═══════════════════════════════════════════════════════════════════

class TestToolPermissions(unittest.TestCase):
    """Tests for ToolPermissionManager: profiles, checks, quotas."""

    def setUp(self):
        self.pm = ToolPermissionManager(default_profile="full_access")

    # ── Built-in profiles ──────────────────────────────────────

    def test_full_access_allows_all(self):
        allowed, reason = self.pm.check_permission("bash")
        self.assertTrue(allowed)
        self.assertEqual(reason, "")

    def test_read_only_denies_bash(self):
        self.pm.set_profile_by_name("read_only")
        allowed, reason = self.pm.check_permission("bash")
        self.assertFalse(allowed)
        self.assertIn("denied", reason)

    def test_read_only_allows_read(self):
        self.pm.set_profile_by_name("read_only")
        allowed, _ = self.pm.check_permission("read")
        self.assertTrue(allowed)

    def test_safe_mode_only_allows_listed(self):
        self.pm.set_profile_by_name("safe_mode")
        allowed, _ = self.pm.check_permission("read")
        self.assertTrue(allowed)
        allowed, reason = self.pm.check_permission("bash")
        self.assertFalse(allowed)
        self.assertIn("allow-list", reason)

    # ── Custom profiles ────────────────────────────────────────

    def test_create_custom_profile(self):
        profile = self.pm.create_profile(
            "custom", denied=["bash", "write"],
        )
        self.pm.set_profile(profile)
        allowed, _ = self.pm.check_permission("bash")
        self.assertFalse(allowed)
        allowed, _ = self.pm.check_permission("read")
        self.assertTrue(allowed)

    def test_deny_takes_precedence_over_allow(self):
        profile = self.pm.create_profile(
            "conflict",
            allowed=["bash", "read"],
            denied=["bash"],
        )
        self.pm.set_profile(profile)
        allowed, _ = self.pm.check_permission("bash")
        self.assertFalse(allowed)

    # ── Quotas ─────────────────────────────────────────────────

    def test_quota_enforcement(self):
        profile = self.pm.create_profile("quota", quotas={"bash": 2})
        self.pm.set_profile(profile)

        # First 2 calls should be within quota
        ok1, _ = self.pm.check_quota("bash")
        self.assertTrue(ok1)
        self.pm.record_call("bash")

        ok2, _ = self.pm.check_quota("bash")
        self.assertTrue(ok2)
        self.pm.record_call("bash")

        # Third call should be over quota
        ok3, reason = self.pm.check_quota("bash")
        self.assertFalse(ok3)
        self.assertIn("quota", reason)

    def test_no_quota_unlimited(self):
        # Default profile has no quotas
        for _ in range(100):
            ok, _ = self.pm.check_quota("bash")
            self.assertTrue(ok)
            self.pm.record_call("bash")

    def test_check_all_combined(self):
        profile = self.pm.create_profile("combined", denied=["write"])
        self.pm.set_profile(profile)
        allowed, reason = self.pm.check_all("write")
        self.assertFalse(allowed)
        allowed, _ = self.pm.check_all("read")
        self.assertTrue(allowed)

    # ── Profile management ─────────────────────────────────────

    def test_list_profiles(self):
        profiles = self.pm.list_profiles()
        names = [p.name for p in profiles]
        self.assertIn("full_access", names)
        self.assertIn("read_only", names)
        self.assertIn("safe_mode", names)

    def test_set_profile_by_name_invalid(self):
        ok = self.pm.set_profile_by_name("nonexistent")
        self.assertFalse(ok)

    # ── Reset ──────────────────────────────────────────────────

    def test_reset_quotas(self):
        profile = self.pm.create_profile("resetable", quotas={"bash": 5})
        self.pm.set_profile(profile)
        self.pm.record_call("bash")
        self.pm.record_call("bash")
        self.pm.reset_quotas()
        quota = profile.quotas["bash"]
        self.assertEqual(quota.current_calls, 0)


# ═══════════════════════════════════════════════════════════════════
#  5. RICH OUTPUT — 12 tests
# ═══════════════════════════════════════════════════════════════════

class TestRichOutput(unittest.TestCase):
    """Tests for RichOutput: tables, progress bars, error display, truncation."""

    def setUp(self):
        self.out = RichOutput(width=80)

    # ── Tables ─────────────────────────────────────────────────

    def test_table_renders(self):
        result = self.out.table(["Name", "Value"], [["foo", "bar"]])
        self.assertIn("foo", result)
        self.assertIn("bar", result)
        self.assertIn("┌", result)
        self.assertIn("┘", result)

    def test_table_multiple_rows(self):
        rows = [["a", "1"], ["b", "2"], ["c", "3"]]
        result = self.out.table(["Key", "Val"], rows)
        self.assertIn("a", result)
        self.assertIn("c", result)

    def test_table_empty_headers(self):
        result = self.out.table([], [])
        self.assertEqual(result, "")

    def test_table_auto_width(self):
        result = self.out.table(
            ["Short", "A Much Longer Header"],
            [["x", "y"]],
        )
        self.assertIn("A Much Longer Header", result)

    # ── Progress bar ───────────────────────────────────────────

    def test_progress_bar_zero(self):
        result = self.out.progress_bar(0, 10)
        self.assertIn("0/10", result)
        self.assertIn("0%", result)

    def test_progress_bar_complete(self):
        result = self.out.progress_bar(10, 10)
        self.assertIn("100%", result)

    def test_progress_bar_with_label(self):
        result = self.out.progress_bar(5, 10, label="Loading")
        self.assertIn("Loading", result)
        self.assertIn("50%", result)

    # ── Tool result ────────────────────────────────────────────

    def test_tool_result_success(self):
        result = self.out.tool_result("bash", True, 234.0, output_lines=42)
        self.assertIn("bash", result)
        self.assertIn("234ms", result)
        self.assertIn("42 lines", result)

    def test_tool_result_slow(self):
        result = self.out.tool_result("bash", True, 2500.0)
        self.assertIn("2.5s", result)

    # ── Error display ──────────────────────────────────────────

    def test_error_with_code(self):
        result = self.out.error("Tool failed", code="E2001")
        self.assertIn("E2001", result)
        self.assertIn("Tool failed", result)

    def test_error_with_recovery_hint(self):
        result = self.out.error("Timed out", recovery_hint="Try a smaller command")
        self.assertIn("Recovery:", result)
        self.assertIn("Try a smaller command", result)

    # ── Truncation ─────────────────────────────────────────────

    def test_truncate_long_text(self):
        text = "\n".join(f"line {i}" for i in range(50))
        result = self.out.truncate(text, max_lines=5)
        lines = result.split("\n")
        self.assertEqual(len(lines), 6)  # 5 lines + "... (45 more lines)"
        self.assertIn("45 more lines", result)


# ═══════════════════════════════════════════════════════════════════
#  6. SPRINT 8 INTEGRATION — 6 tests
# ═══════════════════════════════════════════════════════════════════

class TestSprint8Integration(unittest.TestCase):
    """Integration tests combining Sprint 8 features."""

    def test_sanitizer_plus_metrics(self):
        """Sanitizer cleans output, metrics track the call."""
        san = OutputSanitizer()
        mc = MetricsCollector()

        # Simulate tool execution with secret in output
        output = "Result: sk-abcdefghijklmnopqrstuvwxyz1234"
        t0 = time.time()
        result = san.sanitize(output)
        duration_ms = (time.time() - t0) * 1000
        mc.record_tool_call("bash", duration_ms, success=True)

        self.assertTrue(result.had_secrets)
        self.assertEqual(mc.get_tool_metrics("bash")["bash"].call_count, 1)

    def test_tracer_with_metrics(self):
        """Tracer spans align with metrics recording."""
        tracer = ExecutionTracer()
        mc = MetricsCollector()

        root = tracer.start_span("agent.run")
        tool_span = tracer.start_span("tool.bash", parent_id=root)
        time.sleep(0.01)
        tracer.end_span(tool_span, "ok")
        mc.record_tool_call("bash", tracer.get_span(tool_span).duration_ms, True)
        tracer.end_span(root, "ok")

        self.assertEqual(mc.get_tool_metrics("bash")["bash"].call_count, 1)
        self.assertEqual(tracer.span_count, 2)

    def test_permissions_block_before_execution(self):
        """Permission check prevents tool execution."""
        pm = ToolPermissionManager(default_profile="read_only")
        mc = MetricsCollector()

        allowed, reason = pm.check_permission("bash")
        if not allowed:
            mc.record_tool_call("bash", 0.0, success=False, error=reason)

        m = mc.get_tool_metrics("bash")
        self.assertEqual(m["bash"].error_count, 1)

    def test_rich_output_renders_metrics(self):
        """RichOutput renders a MetricsCollector summary as a table."""
        mc = MetricsCollector()
        mc.record_tool_call("bash", 100, True)
        mc.record_tool_call("read", 5, True)
        mc.record_tool_call("bash", 200, False, error="timeout")

        out = RichOutput()
        table = out.metrics_table(mc.summary())
        self.assertIn("bash", table)
        self.assertIn("read", table)

    def test_rich_output_renders_error_with_catalog(self):
        """RichOutput renders enhanced error with code and hint."""
        out = RichOutput()
        err = out.error(
            "Tool execution timed out",
            code="E2003",
            recovery_hint="Increase the timeout or break the command into steps.",
        )
        self.assertIn("E2003", err)
        self.assertIn("Recovery:", err)

    def test_full_pipeline(self):
        """
        Full pipeline: check permission → start trace → execute with sanitize →
        record metrics → render with rich output.
        """
        pm = ToolPermissionManager()
        tracer = ExecutionTracer()
        san = OutputSanitizer()
        mc = MetricsCollector()
        out = RichOutput()

        # Check permission
        allowed, _ = pm.check_all("bash")
        self.assertTrue(allowed)

        # Start trace
        root = tracer.start_span("agent.run")
        tool_span = tracer.start_span("tool.bash", parent_id=root, command="env")

        # Simulate execution
        raw_output = "HOME=/home/user\nAPI_KEY=sk-abcdefghijklmnopqrstuvwxyz1234"
        t0 = time.time()
        sanitized = san.sanitize(raw_output)
        duration_ms = (time.time() - t0) * 1000

        # End span, record metrics
        tracer.end_span(tool_span, "ok")
        mc.record_tool_call("bash", duration_ms, True)
        pm.record_call("bash")

        tracer.end_span(root, "ok")

        # Verify
        self.assertTrue(sanitized.had_secrets)
        self.assertEqual(mc.get_tool_metrics("bash")["bash"].call_count, 1)
        self.assertEqual(tracer.span_count, 2)
        self.assertIsNotNone(tracer.get_trace_tree())

        # Render
        result_line = out.tool_result("bash", True, duration_ms, output_lines=2)
        self.assertIn("bash", result_line)


# ═══════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
