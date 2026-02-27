"""
Sprint 9 Tests — Multi-Provider Intelligence

Covers: ModelRouter, CostTracker, ProviderHealthTracker, ProviderPool, UsageAnalytics
70 tests across 6 test classes.
"""

import asyncio
import time
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from cowork_agent.core.model_router import (
    ModelRouter, ModelTier, TaskClassification, TierConfig,
)
from cowork_agent.core.cost_tracker import (
    CostTracker, CostRecord, ModelPricing, BudgetExceededError,
)
from cowork_agent.core.provider_health_tracker import (
    ProviderHealthTracker, ProviderHealthScore,
)
from cowork_agent.core.provider_pool import ProviderPool, ProviderEntry
from cowork_agent.core.usage_analytics import UsageAnalytics, RoutingDecision
from cowork_agent.core.models import AgentResponse, ToolCall


def _run(coro):
    """Helper to run async code in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_response(text="ok", tool_calls=None, stop_reason="end_turn", usage=None):
    return AgentResponse(
        text=text,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
        usage=usage,
    )


def _make_tool_calls(n=1):
    return [
        ToolCall(name=f"tool_{i}", tool_id=f"tid_{i}", input={})
        for i in range(n)
    ]


# ══════════════════════════════════════════════════════════════════
# 1. Model Router (14 tests)
# ══════════════════════════════════════════════════════════════════

class TestModelRouter(unittest.TestCase):
    def setUp(self):
        self.router = ModelRouter(
            tier_configs={
                ModelTier.FAST: TierConfig("openai", "gpt-4o-mini"),
                ModelTier.BALANCED: TierConfig("anthropic", "claude-sonnet"),
                ModelTier.POWERFUL: TierConfig("anthropic", "claude-opus"),
            },
        )

    def test_classify_short_simple_input(self):
        c = self.router.classify("hi")
        self.assertEqual(c.tier, ModelTier.FAST)

    def test_classify_moderate_input(self):
        c = self.router.classify("Can you help me understand how the config system works?")
        self.assertEqual(c.tier, ModelTier.BALANCED)

    def test_classify_complex_input(self):
        c = self.router.classify(
            "I need you to implement a comprehensive authentication system "
            "with OAuth2, JWT tokens, and role-based access control. Also "
            "refactor the existing user module to support multi-tenancy."
        )
        self.assertEqual(c.tier, ModelTier.POWERFUL)

    def test_classify_with_code_keywords(self):
        c = self.router.classify("implement a function that sorts items by priority")
        self.assertEqual(c.tier, ModelTier.POWERFUL)

    def test_classify_simple_file_read(self):
        c = self.router.classify("read the config file")
        self.assertEqual(c.tier, ModelTier.FAST)

    def test_classify_with_tool_calls(self):
        tools = _make_tool_calls(5)
        c = self.router.classify("continue", tool_calls=tools)
        # Many tool calls should push toward BALANCED or POWERFUL
        self.assertIn(c.tier, [ModelTier.BALANCED, ModelTier.POWERFUL])

    def test_classify_deep_conversation(self):
        c = self.router.classify("continue working on the current task", message_count=25)
        # Deep conversation adds complexity
        self.assertNotEqual(c.tier, ModelTier.FAST)

    def test_classify_disabled_returns_balanced(self):
        router = ModelRouter(enabled=False)
        c = router.classify("implement a complex system")
        self.assertEqual(c.tier, ModelTier.BALANCED)

    def test_classification_has_reasoning(self):
        c = self.router.classify("debug the authentication module")
        self.assertTrue(len(c.reasoning) > 0)

    def test_get_config_for_tier(self):
        cfg = self.router.get_config_for_tier(ModelTier.FAST)
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.model, "gpt-4o-mini")

    def test_get_config_missing_tier(self):
        router = ModelRouter()
        self.assertIsNone(router.get_config_for_tier(ModelTier.FAST))

    def test_should_escalate_on_error(self):
        resp = _make_response(text="", stop_reason="error")
        next_tier = self.router.should_escalate(resp, ModelTier.FAST)
        self.assertEqual(next_tier, ModelTier.BALANCED)

    def test_should_escalate_at_powerful_returns_none(self):
        resp = _make_response(text="", stop_reason="error")
        next_tier = self.router.should_escalate(resp, ModelTier.POWERFUL)
        self.assertIsNone(next_tier)

    def test_no_escalation_on_good_response(self):
        resp = _make_response(text="Here is your answer")
        next_tier = self.router.should_escalate(resp, ModelTier.FAST)
        self.assertIsNone(next_tier)


# ══════════════════════════════════════════════════════════════════
# 2. Cost Tracker (14 tests)
# ══════════════════════════════════════════════════════════════════

class TestCostTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = CostTracker(budget_limit=1.0)

    def test_record_basic_cost(self):
        usage = {"input_tokens": 1000, "output_tokens": 500}
        record = self.tracker.record(usage, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        self.assertGreater(record.estimated_cost, 0)

    def test_cost_calculation_accuracy(self):
        # Claude Sonnet: $3/1M input, $15/1M output → $0.003/1K, $0.015/1K
        usage = {"input_tokens": 1000, "output_tokens": 1000}
        record = self.tracker.record(usage, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        expected = (1000 / 1000 * 0.003) + (1000 / 1000 * 0.015)  # 0.018
        self.assertAlmostEqual(record.estimated_cost, expected, places=4)

    def test_cache_savings_calculation(self):
        usage = {
            "input_tokens": 2000,
            "output_tokens": 500,
            "cache_read_input_tokens": 1500,
        }
        record = self.tracker.record(usage, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        self.assertGreater(record.cache_savings, 0)

    def test_ollama_zero_cost(self):
        usage = {"input_tokens": 10000, "output_tokens": 5000}
        record = self.tracker.record(usage, "ollama", "llama3")
        self.assertEqual(record.estimated_cost, 0.0)

    def test_unknown_model_zero_cost(self):
        usage = {"input_tokens": 1000, "output_tokens": 500}
        record = self.tracker.record(usage, "unknown", "mystery-model-7b")
        self.assertEqual(record.estimated_cost, 0.0)

    def test_budget_enforcement(self):
        tracker = CostTracker(budget_limit=0.001)
        usage = {"input_tokens": 10000, "output_tokens": 10000}
        tracker.record(usage, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        self.assertTrue(tracker.is_over_budget())
        with self.assertRaises(BudgetExceededError):
            tracker.check_budget()

    def test_remaining_budget(self):
        self.assertEqual(self.tracker.remaining_budget(), 1.0)
        usage = {"input_tokens": 1000, "output_tokens": 1000}
        self.tracker.record(usage, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        remaining = self.tracker.remaining_budget()
        self.assertIsNotNone(remaining)
        self.assertLess(remaining, 1.0)

    def test_no_budget_limit(self):
        tracker = CostTracker(budget_limit=None)
        self.assertIsNone(tracker.remaining_budget())
        self.assertFalse(tracker.is_over_budget())

    def test_per_model_breakdown(self):
        usage1 = {"input_tokens": 1000, "output_tokens": 500}
        usage2 = {"input_tokens": 2000, "output_tokens": 1000}
        self.tracker.record(usage1, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        self.tracker.record(usage2, "openai", "gpt-4o")
        breakdown = self.tracker.per_model_breakdown()
        self.assertEqual(len(breakdown), 2)
        self.assertIn("claude-sonnet-4-5-20250929", breakdown)
        self.assertIn("gpt-4o", breakdown)

    def test_summary(self):
        usage = {"input_tokens": 1000, "output_tokens": 500}
        self.tracker.record(usage, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        s = self.tracker.summary()
        self.assertIn("total_cost", s)
        self.assertIn("per_model", s)
        self.assertEqual(s["call_count"], 1)

    def test_get_history(self):
        usage = {"input_tokens": 100, "output_tokens": 50}
        for _ in range(5):
            self.tracker.record(usage, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        self.assertEqual(len(self.tracker.get_history()), 5)
        self.assertEqual(len(self.tracker.get_history(last_n=3)), 3)

    def test_add_custom_pricing(self):
        self.tracker.add_pricing(
            "custom-model",
            ModelPricing("custom", "custom-model", 0.001, 0.002),
        )
        usage = {"input_tokens": 1000, "output_tokens": 1000}
        record = self.tracker.record(usage, "custom", "custom-model")
        expected = 0.001 + 0.002
        self.assertAlmostEqual(record.estimated_cost, expected, places=4)

    def test_disabled_tracker(self):
        tracker = CostTracker(enabled=False)
        usage = {"input_tokens": 1000, "output_tokens": 500}
        record = tracker.record(usage, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        self.assertEqual(record.estimated_cost, 0.0)

    def test_reset_clears_all(self):
        usage = {"input_tokens": 1000, "output_tokens": 500}
        self.tracker.record(usage, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        self.tracker.reset()
        self.assertEqual(self.tracker.total_cost, 0)
        self.assertEqual(self.tracker.call_count, 0)


# ══════════════════════════════════════════════════════════════════
# 3. Provider Health Tracker (14 tests)
# ══════════════════════════════════════════════════════════════════

class TestProviderHealthTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = ProviderHealthTracker()

    def test_initial_score_after_success(self):
        self.tracker.record_call("anthropic", 200, True)
        score = self.tracker.get_score("anthropic")
        self.assertGreater(score.score, 80)

    def test_score_degrades_on_failure(self):
        self.tracker.record_call("openai", 200, True)
        good_score = self.tracker.get_score("openai").score
        self.tracker.record_call("openai", 500, False)
        bad_score = self.tracker.get_score("openai").score
        self.assertLess(bad_score, good_score)

    def test_consecutive_failures_heavy_penalty(self):
        for _ in range(5):
            self.tracker.record_call("bad_provider", 1000, False)
        score = self.tracker.get_score("bad_provider")
        self.assertLess(score.score, 30)
        self.assertEqual(score.consecutive_failures, 5)

    def test_recovery_after_success(self):
        for _ in range(3):
            self.tracker.record_call("provider", 500, False)
        low_score = self.tracker.get_score("provider").score
        self.tracker.record_call("provider", 200, True)
        recovered = self.tracker.get_score("provider")
        self.assertEqual(recovered.consecutive_failures, 0)
        self.assertGreater(recovered.score, low_score)

    def test_high_latency_reduces_score(self):
        self.tracker.record_call("fast", 100, True)
        self.tracker.record_call("slow", 8000, True)
        fast_score = self.tracker.get_score("fast").score
        slow_score = self.tracker.get_score("slow").score
        self.assertGreater(fast_score, slow_score)

    def test_ewma_latency(self):
        self.tracker.record_call("p", 100, True)
        self.tracker.record_call("p", 300, True)
        avg = self.tracker.get_score("p").avg_latency_ms
        # EWMA: 0.85 * 100 + 0.15 * 300 = 130
        self.assertAlmostEqual(avg, 130, delta=5)

    def test_ewma_error_rate(self):
        self.tracker.record_call("p", 100, True)   # error_rate = 0
        self.tracker.record_call("p", 100, False)  # EWMA: 0.85*0 + 0.15*1 = 0.15
        rate = self.tracker.get_score("p").error_rate
        self.assertAlmostEqual(rate, 0.15, delta=0.02)

    def test_rankings_order(self):
        self.tracker.record_call("good", 100, True)
        self.tracker.record_call("bad", 5000, False)
        self.tracker.record_call("bad", 5000, False)
        rankings = self.tracker.get_rankings()
        self.assertEqual(rankings[0].provider_name, "good")

    def test_is_healthy(self):
        self.tracker.record_call("healthy", 100, True)
        self.assertTrue(self.tracker.is_healthy("healthy"))
        for _ in range(5):
            self.tracker.record_call("unhealthy", 1000, False)
        self.assertFalse(self.tracker.is_healthy("unhealthy"))

    def test_get_best_provider(self):
        self.tracker.record_call("alpha", 100, True)
        self.tracker.record_call("beta", 5000, False)
        best = self.tracker.get_best_provider()
        self.assertEqual(best, "alpha")

    def test_status_property(self):
        self.tracker.record_call("p", 100, True)
        score = self.tracker.get_score("p")
        self.assertEqual(score.status, "healthy")

    def test_disabled_tracker(self):
        tracker = ProviderHealthTracker(enabled=False)
        tracker.record_call("p", 100, True)
        score = tracker.get_score("p")
        self.assertEqual(score.sample_count, 0)

    def test_to_dict(self):
        self.tracker.record_call("p1", 100, True)
        d = self.tracker.to_dict()
        self.assertIn("providers", d)
        self.assertIn("p1", d["providers"])

    def test_reset(self):
        self.tracker.record_call("p", 100, True)
        self.tracker.reset()
        self.assertEqual(len(self.tracker.provider_names), 0)


# ══════════════════════════════════════════════════════════════════
# 4. Provider Pool (12 tests)
# ══════════════════════════════════════════════════════════════════

class TestProviderPool(unittest.TestCase):
    def _make_provider(self, name="test", model="test-model"):
        p = MagicMock()
        p.provider_name = name
        p.model = model
        return p

    def setUp(self):
        self.pool = ProviderPool()

    def test_register_and_get(self):
        p = self._make_provider()
        self.pool.register("test", p, ModelTier.BALANCED)
        self.assertEqual(self.pool.get("test"), p)

    def test_first_registered_is_active(self):
        p1 = self._make_provider("p1")
        p2 = self._make_provider("p2")
        self.pool.register("p1", p1)
        self.pool.register("p2", p2)
        self.assertEqual(self.pool.active, p1)
        self.assertEqual(self.pool.active_name, "p1")

    def test_swap_active(self):
        p1 = self._make_provider("p1")
        p2 = self._make_provider("p2")
        self.pool.register("p1", p1)
        self.pool.register("p2", p2)
        self.assertTrue(self.pool.swap_active("p2"))
        self.assertEqual(self.pool.active, p2)

    def test_swap_unknown_returns_false(self):
        self.assertFalse(self.pool.swap_active("nonexistent"))

    def test_get_for_tier(self):
        p_fast = self._make_provider("fast")
        p_power = self._make_provider("power")
        self.pool.register("fast", p_fast, ModelTier.FAST)
        self.pool.register("power", p_power, ModelTier.POWERFUL)
        self.assertEqual(self.pool.get_for_tier(ModelTier.FAST), p_fast)
        self.assertEqual(self.pool.get_for_tier(ModelTier.POWERFUL), p_power)

    def test_get_for_tier_empty(self):
        self.assertIsNone(self.pool.get_for_tier(ModelTier.FAST))

    def test_get_for_tier_with_health(self):
        health = ProviderHealthTracker()
        pool = ProviderPool(health_tracker=health)
        p1 = self._make_provider("p1")
        p2 = self._make_provider("p2")
        pool.register("p1", p1, ModelTier.BALANCED)
        pool.register("p2", p2, ModelTier.BALANCED)
        # Make p2 healthier
        health.record_call("p1", 5000, False)
        health.record_call("p2", 100, True)
        result = pool.get_for_tier(ModelTier.BALANCED)
        self.assertEqual(result, p2)

    def test_list_providers(self):
        self.pool.register("a", self._make_provider("a"))
        self.pool.register("b", self._make_provider("b"))
        entries = self.pool.list_providers()
        self.assertEqual(len(entries), 2)

    def test_list_for_tier(self):
        self.pool.register("f1", self._make_provider("f1"), ModelTier.FAST)
        self.pool.register("f2", self._make_provider("f2"), ModelTier.FAST)
        self.pool.register("b1", self._make_provider("b1"), ModelTier.BALANCED)
        self.assertEqual(len(self.pool.list_for_tier(ModelTier.FAST)), 2)
        self.assertEqual(len(self.pool.list_for_tier(ModelTier.BALANCED)), 1)

    def test_size_property(self):
        self.assertEqual(self.pool.size, 0)
        self.pool.register("a", self._make_provider())
        self.assertEqual(self.pool.size, 1)

    def test_summary(self):
        self.pool.register("a", self._make_provider(), ModelTier.FAST)
        s = self.pool.summary()
        self.assertIn("providers", s)
        self.assertIn("tiers", s)
        self.assertEqual(s["provider_count"], 1)

    def test_health_check_all(self):
        p = self._make_provider()
        p.health_check = AsyncMock(return_value={"model": "test"})
        self.pool.register("p", p)
        results = _run(self.pool.health_check_all())
        self.assertIn("p", results)
        self.assertEqual(results["p"]["status"], "ok")


# ══════════════════════════════════════════════════════════════════
# 5. Usage Analytics (10 tests)
# ══════════════════════════════════════════════════════════════════

class TestUsageAnalytics(unittest.TestCase):
    def setUp(self):
        self.cost_tracker = CostTracker()
        self.metrics = MagicMock()
        self.metrics.summary.return_value = {
            "total_tool_calls": 10,
            "total_errors": 1,
            "tools": {},
        }
        self.health = ProviderHealthTracker()
        self.analytics = UsageAnalytics(
            cost_tracker=self.cost_tracker,
            metrics_collector=self.metrics,
            health_tracker=self.health,
        )

    def test_record_routing_decision(self):
        self.analytics.record_routing_decision(
            ModelTier.FAST, "openai", "gpt-4o-mini", "short input",
        )
        self.assertEqual(len(self.analytics.routing_decisions), 1)
        self.assertEqual(self.analytics.routing_decisions[0].tier, "fast")

    def test_session_report_structure(self):
        self.analytics.record_routing_decision(
            ModelTier.BALANCED, "anthropic", "sonnet", "default",
        )
        report = self.analytics.session_report()
        self.assertIn("session", report)
        self.assertIn("cost", report)
        self.assertIn("routing", report)
        self.assertIn("tools", report)
        self.assertIn("providers", report)
        self.assertIn("recommendations", report)

    def test_routing_distribution(self):
        for _ in range(3):
            self.analytics.record_routing_decision(
                ModelTier.FAST, "openai", "mini", "simple",
            )
        for _ in range(7):
            self.analytics.record_routing_decision(
                ModelTier.BALANCED, "anthropic", "sonnet", "moderate",
            )
        report = self.analytics.session_report()
        dist = report["routing"]["tier_distribution"]
        self.assertEqual(dist["fast"], 3)
        self.assertEqual(dist["balanced"], 7)

    def test_escalation_count(self):
        self.analytics.record_routing_decision(
            ModelTier.FAST, "openai", "mini", "simple",
        )
        self.analytics.record_routing_decision(
            ModelTier.BALANCED, "anthropic", "sonnet", "escalated", escalated=True,
        )
        report = self.analytics.session_report()
        self.assertEqual(report["routing"]["escalation_count"], 1)

    def test_efficiency_score_default(self):
        score = self.analytics.efficiency_score()
        self.assertEqual(score, 50.0)  # no data

    def test_efficiency_score_with_data(self):
        for _ in range(5):
            self.analytics.record_routing_decision(
                ModelTier.FAST, "openai", "mini", "simple",
            )
        score = self.analytics.efficiency_score()
        self.assertGreater(score, 50)

    def test_recommendations_high_powerful_usage(self):
        for _ in range(10):
            self.analytics.record_routing_decision(
                ModelTier.POWERFUL, "anthropic", "opus", "complex",
            )
        report = self.analytics.session_report()
        recs = report["recommendations"]
        self.assertTrue(any("POWERFUL" in r for r in recs))

    def test_recommendations_healthy_session(self):
        for _ in range(3):
            self.analytics.record_routing_decision(
                ModelTier.BALANCED, "anthropic", "sonnet", "normal",
            )
        report = self.analytics.session_report()
        recs = report["recommendations"]
        self.assertTrue(any("healthy" in r.lower() or "no optimization" in r.lower() for r in recs))

    def test_disabled_analytics(self):
        analytics = UsageAnalytics(enabled=False)
        analytics.record_routing_decision(
            ModelTier.FAST, "openai", "mini", "test",
        )
        self.assertEqual(len(analytics.routing_decisions), 0)

    def test_reset(self):
        self.analytics.record_routing_decision(
            ModelTier.FAST, "openai", "mini", "test",
        )
        self.analytics.reset()
        self.assertEqual(len(self.analytics.routing_decisions), 0)


# ══════════════════════════════════════════════════════════════════
# 6. Integration Tests (6 tests)
# ══════════════════════════════════════════════════════════════════

class TestSprint9Integration(unittest.TestCase):
    """End-to-end tests combining multiple Sprint 9 features."""

    def test_router_to_pool_pipeline(self):
        """Router classifies → pool returns provider for tier."""
        health = ProviderHealthTracker()
        pool = ProviderPool(health_tracker=health)

        fast_provider = MagicMock()
        fast_provider.provider_name = "fast"
        fast_provider.model = "mini"

        pool.register("fast", fast_provider, ModelTier.FAST)

        router = ModelRouter(tier_configs={
            ModelTier.FAST: TierConfig("openai", "gpt-4o-mini"),
        })

        classification = router.classify("hi")
        self.assertEqual(classification.tier, ModelTier.FAST)

        provider = pool.get_for_tier(classification.tier)
        self.assertEqual(provider, fast_provider)

    def test_cost_plus_health_tracking(self):
        """Record call updates both cost tracker and health tracker."""
        cost = CostTracker()
        health = ProviderHealthTracker()

        usage = {"input_tokens": 1000, "output_tokens": 500}
        cost.record(usage, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        health.record_call("anthropic", 250, True)

        self.assertGreater(cost.total_cost, 0)
        self.assertGreater(health.get_score("anthropic").score, 70)

    def test_escalation_with_cost_awareness(self):
        """Escalation happens, cost is tracked at each tier."""
        router = ModelRouter(tier_configs={
            ModelTier.FAST: TierConfig("openai", "gpt-4o-mini"),
            ModelTier.BALANCED: TierConfig("anthropic", "claude-sonnet-4-5-20250929"),
        })
        cost = CostTracker()

        # First attempt at FAST tier fails
        error_resp = _make_response(text="", stop_reason="error")
        next_tier = router.should_escalate(error_resp, ModelTier.FAST)
        self.assertEqual(next_tier, ModelTier.BALANCED)

        # Record cost for the escalated call
        usage = {"input_tokens": 2000, "output_tokens": 1000}
        cost.record(usage, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        self.assertGreater(cost.total_cost, 0)

    def test_analytics_aggregates_all(self):
        """Analytics combines cost, metrics, health, and routing."""
        cost = CostTracker()
        health = ProviderHealthTracker()

        metrics = MagicMock()
        metrics.summary.return_value = {"total_tool_calls": 5, "total_errors": 0}

        analytics = UsageAnalytics(
            cost_tracker=cost,
            metrics_collector=metrics,
            health_tracker=health,
        )

        # Simulate a session
        usage = {"input_tokens": 1000, "output_tokens": 500}
        cost.record(usage, "anthropic", "claude-sonnet-4-5-20250929")
        health.record_call("anthropic", 200, True)
        analytics.record_routing_decision(
            ModelTier.BALANCED, "anthropic", "claude-sonnet-4-5-20250929", "default",
        )

        report = analytics.session_report()
        self.assertGreater(report["cost"]["total_cost"], 0)
        self.assertEqual(report["routing"]["total_decisions"], 1)

    def test_pool_health_aware_routing(self):
        """Pool selects healthier provider when multiple exist for a tier."""
        health = ProviderHealthTracker()
        pool = ProviderPool(health_tracker=health)

        p1 = MagicMock()
        p1.provider_name = "slow"
        p2 = MagicMock()
        p2.provider_name = "fast"

        pool.register("slow", p1, ModelTier.BALANCED)
        pool.register("fast", p2, ModelTier.BALANCED)

        # Make "fast" healthier
        health.record_call("slow", 5000, False)
        health.record_call("slow", 5000, False)
        health.record_call("fast", 100, True)

        selected = pool.get_for_tier(ModelTier.BALANCED)
        self.assertEqual(selected, p2)

    def test_budget_blocks_before_costly_call(self):
        """Budget enforcement prevents expensive calls."""
        cost = CostTracker(budget_limit=0.01)
        # Burn through budget
        usage = {"input_tokens": 50000, "output_tokens": 50000}
        cost.record(usage, "AnthropicProvider", "claude-sonnet-4-5-20250929")
        self.assertTrue(cost.is_over_budget())
        with self.assertRaises(BudgetExceededError):
            cost.check_budget()


if __name__ == "__main__":
    unittest.main()
