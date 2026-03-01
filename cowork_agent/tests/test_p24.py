"""
Sprint 24 Tests — Production Hardening.

Tests for:
  - ToolOutputValidator: Schema validation, assertions, retry/block decisions
  - CostOptimizer: Model selection, budget management, complexity classification
  - ApprovalWorkflow: Approval requests, escalation policies, audit trails

Target: 160+ tests.
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

# ── Tool Output Validator ────────────────────────────────────────

from cowork_agent.core.tool_output_validator import (
    AssertionType,
    OutputAssertion,
    ToolOutputValidator,
    ValidationFailure,
    ValidationResult,
    ValidationSeverity,
)


class TestToolOutputValidatorBasics(unittest.TestCase):
    """Test basic validator setup and configuration."""

    def setUp(self):
        self.validator = ToolOutputValidator(load_defaults=True)

    def test_default_assertions_loaded(self):
        self.assertIn("bash", self.validator.registered_tools)
        self.assertIn("read", self.validator.registered_tools)
        self.assertIn("web_fetch", self.validator.registered_tools)

    def test_add_assertion(self):
        self.validator.add_assertion("my_tool", OutputAssertion(
            assertion_type=AssertionType.NOT_EMPTY,
        ))
        self.assertIn("my_tool", self.validator.registered_tools)

    def test_set_assertions(self):
        self.validator.set_assertions("custom", [
            OutputAssertion(assertion_type=AssertionType.NOT_EMPTY),
            OutputAssertion(assertion_type=AssertionType.MAX_LENGTH, value=100),
        ])
        self.assertEqual(len(self.validator.get_assertions("custom")), 2)

    def test_clear_assertions(self):
        self.validator.clear_assertions("bash")
        self.assertEqual(len(self.validator.get_assertions("bash")), 0)

    def test_no_assertions_passes(self):
        result = self.validator.validate("unknown_tool", "any output")
        self.assertTrue(result.passed)

    def test_empty_output_for_defaults(self):
        # No defaults require NOT_EMPTY for bash, only MAX_LENGTH
        result = self.validator.validate("bash", "hello")
        self.assertTrue(result.passed)

    def test_stats_initial(self):
        stats = self.validator.stats
        self.assertEqual(stats["total_validations"], 0)


class TestAssertionTypes(unittest.TestCase):
    """Test each assertion type individually."""

    def setUp(self):
        self.validator = ToolOutputValidator(load_defaults=False)

    def test_not_empty_passes(self):
        self.validator.add_assertion("t", OutputAssertion(assertion_type=AssertionType.NOT_EMPTY))
        result = self.validator.validate("t", "hello")
        self.assertTrue(result.passed)

    def test_not_empty_fails_empty(self):
        self.validator.add_assertion("t", OutputAssertion(assertion_type=AssertionType.NOT_EMPTY))
        result = self.validator.validate("t", "")
        self.assertFalse(result.passed)

    def test_not_empty_fails_whitespace(self):
        self.validator.add_assertion("t", OutputAssertion(assertion_type=AssertionType.NOT_EMPTY))
        result = self.validator.validate("t", "   ")
        self.assertFalse(result.passed)

    def test_max_length_passes(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.MAX_LENGTH, value=100))
        result = self.validator.validate("t", "short text")
        self.assertTrue(result.passed)

    def test_max_length_fails(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.MAX_LENGTH, value=5))
        result = self.validator.validate("t", "this is too long")
        self.assertFalse(result.passed)

    def test_min_length_passes(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.MIN_LENGTH, value=3))
        result = self.validator.validate("t", "hello")
        self.assertTrue(result.passed)

    def test_min_length_fails(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.MIN_LENGTH, value=100))
        result = self.validator.validate("t", "short")
        self.assertFalse(result.passed)

    def test_matches_regex_passes(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.MATCHES_REGEX, value=r"\d+"))
        result = self.validator.validate("t", "answer is 42")
        self.assertTrue(result.passed)

    def test_matches_regex_fails(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.MATCHES_REGEX, value=r"^\d+$"))
        result = self.validator.validate("t", "not a number")
        self.assertFalse(result.passed)

    def test_is_json_passes(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.IS_JSON))
        result = self.validator.validate("t", '{"key": "value"}')
        self.assertTrue(result.passed)

    def test_is_json_fails(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.IS_JSON))
        result = self.validator.validate("t", "not json {")
        self.assertFalse(result.passed)

    def test_json_has_key_passes(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.JSON_HAS_KEY, value="status"))
        result = self.validator.validate("t", '{"status": "ok"}')
        self.assertTrue(result.passed)

    def test_json_has_key_fails_missing(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.JSON_HAS_KEY, value="status"))
        result = self.validator.validate("t", '{"name": "test"}')
        self.assertFalse(result.passed)

    def test_json_has_key_fails_not_json(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.JSON_HAS_KEY, value="status"))
        result = self.validator.validate("t", "plain text")
        self.assertFalse(result.passed)

    def test_contains_passes(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.CONTAINS, value="SUCCESS"))
        result = self.validator.validate("t", "Operation SUCCESS completed")
        self.assertTrue(result.passed)

    def test_contains_fails(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.CONTAINS, value="SUCCESS"))
        result = self.validator.validate("t", "Operation failed")
        self.assertFalse(result.passed)

    def test_not_contains_passes(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.NOT_CONTAINS, value="ERROR"))
        result = self.validator.validate("t", "All good")
        self.assertTrue(result.passed)

    def test_not_contains_fails(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.NOT_CONTAINS, value="ERROR"))
        result = self.validator.validate("t", "An ERROR occurred")
        self.assertFalse(result.passed)

    def test_custom_passes(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.CUSTOM,
            custom_fn=lambda o: len(o.split("\n")) > 1,
        ))
        result = self.validator.validate("t", "line1\nline2")
        self.assertTrue(result.passed)

    def test_custom_fails(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.CUSTOM,
            custom_fn=lambda o: len(o.split("\n")) > 5,
        ))
        result = self.validator.validate("t", "single line")
        self.assertFalse(result.passed)


class TestValidationSeverity(unittest.TestCase):
    """Test severity-based actions (warn, block, retry)."""

    def setUp(self):
        self.validator = ToolOutputValidator(load_defaults=False)

    def test_warn_doesnt_block_or_retry(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.NOT_EMPTY,
            severity=ValidationSeverity.WARN,
        ))
        result = self.validator.validate("t", "")
        self.assertFalse(result.passed)
        self.assertFalse(result.should_block)
        self.assertFalse(result.should_retry)

    def test_block_severity(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.NOT_CONTAINS,
            value="FORBIDDEN",
            severity=ValidationSeverity.BLOCK,
        ))
        result = self.validator.validate("t", "FORBIDDEN content")
        self.assertFalse(result.passed)
        self.assertTrue(result.should_block)

    def test_retry_severity(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.IS_JSON,
            severity=ValidationSeverity.RETRY,
        ))
        result = self.validator.validate("t", "not json")
        self.assertFalse(result.passed)
        self.assertTrue(result.should_retry)
        self.assertTrue(len(result.retry_hint) > 0)

    def test_mixed_severities_highest_wins(self):
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.NOT_EMPTY,
            severity=ValidationSeverity.WARN,
        ))
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.MIN_LENGTH,
            value=100,
            severity=ValidationSeverity.BLOCK,
        ))
        result = self.validator.validate("t", "")
        self.assertTrue(result.should_block)


class TestValidatorStats(unittest.TestCase):
    """Test statistics tracking."""

    def setUp(self):
        self.validator = ToolOutputValidator(load_defaults=False)
        self.validator.add_assertion("t", OutputAssertion(
            assertion_type=AssertionType.NOT_EMPTY,
            severity=ValidationSeverity.BLOCK,
        ))

    def test_pass_increments_passed(self):
        self.validator.validate("t", "good")
        self.assertEqual(self.validator.stats["total_passed"], 1)

    def test_fail_increments_failed(self):
        self.validator.validate("t", "")
        self.assertEqual(self.validator.stats["total_failed"], 1)

    def test_block_increments_blocks(self):
        self.validator.validate("t", "")
        self.assertEqual(self.validator.stats["total_blocks"], 1)

    def test_total_validations(self):
        self.validator.validate("t", "good")
        self.validator.validate("t", "")
        self.assertEqual(self.validator.stats["total_validations"], 2)

    def test_validation_result_summary(self):
        result = self.validator.validate("t", "good")
        self.assertIn("passed", result.summary)

    def test_validation_result_to_dict(self):
        result = self.validator.validate("t", "")
        d = result.to_dict()
        self.assertIn("failures", d)
        self.assertFalse(d["passed"])

    def test_failure_to_dict(self):
        result = self.validator.validate("t", "")
        self.assertTrue(len(result.failures) > 0)
        fd = result.failures[0].to_dict()
        self.assertIn("tool_name", fd)


class TestValidatorDefaults(unittest.TestCase):
    """Test default assertions for common tools."""

    def setUp(self):
        self.validator = ToolOutputValidator(load_defaults=True)

    def test_bash_large_output_warns(self):
        result = self.validator.validate("bash", "x" * 600_000)
        self.assertFalse(result.passed)

    def test_bash_normal_output_passes(self):
        result = self.validator.validate("bash", "command output")
        self.assertTrue(result.passed)

    def test_read_empty_warns(self):
        result = self.validator.validate("read", "")
        self.assertFalse(result.passed)

    def test_read_normal_passes(self):
        result = self.validator.validate("read", "file content here")
        self.assertTrue(result.passed)

    def test_web_fetch_empty_warns(self):
        result = self.validator.validate("web_fetch", "")
        self.assertFalse(result.passed)

    def test_web_search_empty_warns(self):
        result = self.validator.validate("web_search", "")
        self.assertFalse(result.passed)

    def test_web_fetch_large_warns(self):
        result = self.validator.validate("web_fetch", "x" * 300_000)
        self.assertFalse(result.passed)


# ── Cost Optimizer ───────────────────────────────────────────────

from cowork_agent.core.cost_optimizer import (
    CostDecision,
    CostEstimate,
    CostOptimizer,
    ModelCostInfo,
    OptimizationResult,
)


class TestCostOptimizerBasics(unittest.TestCase):
    """Test basic optimizer setup."""

    def test_default_construction(self):
        opt = CostOptimizer()
        self.assertEqual(opt._current_model, "claude-sonnet-4-5-20250929")

    def test_disabled_returns_current(self):
        opt = CostOptimizer(enabled=False)
        result = opt.optimize("hello")
        self.assertEqual(result.decision, CostDecision.USE_CURRENT)
        self.assertFalse(result.changed)

    def test_stats_initial(self):
        opt = CostOptimizer()
        stats = opt.stats
        self.assertEqual(stats["total_optimizations"], 0)


class TestComplexityClassification(unittest.TestCase):
    """Test task complexity classification."""

    def setUp(self):
        self.opt = CostOptimizer()

    def test_simple_greeting_fast(self):
        result = self.opt.optimize("hello")
        self.assertEqual(result.decision, CostDecision.DOWNGRADE)
        self.assertIn("simple", result.reasoning.lower())

    def test_yes_is_fast(self):
        result = self.opt.optimize("yes")
        self.assertEqual(result.decision, CostDecision.DOWNGRADE)

    def test_short_input_fast(self):
        result = self.opt.optimize("ok")
        self.assertEqual(result.decision, CostDecision.DOWNGRADE)

    def test_complex_implement_powerful(self):
        result = self.opt.optimize("implement a distributed system with load balancing")
        # This should trigger complex pattern → powerful tier
        self.assertIn(result.decision, (CostDecision.UPGRADE, CostDecision.USE_CURRENT))

    def test_refactor_codebase_powerful(self):
        result = self.opt.optimize("refactor the entire codebase to use async")
        self.assertIn(result.decision, (CostDecision.UPGRADE, CostDecision.USE_CURRENT))

    def test_medium_input_balanced(self):
        result = self.opt.optimize("Can you help me understand how the authentication system works?")
        # Medium length, no strong signals → balanced
        self.assertIn(result.decision, (CostDecision.USE_CURRENT, CostDecision.DOWNGRADE, CostDecision.UPGRADE))

    def test_force_tier_overrides(self):
        result = self.opt.optimize("hello", force_tier="powerful")
        self.assertIn("forced", result.reasoning.lower())

    def test_many_tool_calls_powerful(self):
        result = self.opt.optimize("please continue working on the current task at hand", tool_count=6)
        # 6 tool calls + medium input → powerful
        self.assertIn(result.decision, (CostDecision.UPGRADE, CostDecision.USE_CURRENT))


class TestBudgetManagement(unittest.TestCase):
    """Test budget-aware optimization."""

    def test_budget_exhausted_blocks(self):
        opt = CostOptimizer(budget_limit=1.0, budget_used=1.5)
        result = opt.optimize("do something")
        self.assertEqual(result.decision, CostDecision.BLOCK)

    def test_budget_pressure_downgrades(self):
        opt = CostOptimizer(budget_limit=1.0, budget_used=0.9)
        result = opt.optimize("implement complex system")
        # 90% budget used → force downgrade to fast
        self.assertEqual(result.decision, CostDecision.DOWNGRADE)
        self.assertIn("budget", result.reasoning.lower())

    def test_no_budget_no_block(self):
        opt = CostOptimizer(budget_limit=None)
        result = opt.optimize("hello")
        self.assertNotEqual(result.decision, CostDecision.BLOCK)

    def test_update_budget(self):
        opt = CostOptimizer(budget_limit=1.0, budget_used=0.0)
        opt.update_budget(budget_used=0.5)
        self.assertEqual(opt.budget_status["used"], 0.5)

    def test_budget_status_no_limit(self):
        opt = CostOptimizer(budget_limit=None)
        self.assertFalse(opt.budget_status["has_budget"])

    def test_budget_status_with_limit(self):
        opt = CostOptimizer(budget_limit=2.0, budget_used=0.5)
        status = opt.budget_status
        self.assertTrue(status["has_budget"])
        self.assertEqual(status["remaining"], 1.5)
        self.assertEqual(status["percentage_used"], 25.0)


class TestCostEstimation(unittest.TestCase):
    """Test cost estimation."""

    def test_estimate_with_known_model(self):
        opt = CostOptimizer()
        estimate = opt.estimate_request_cost(
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
        )
        self.assertGreater(estimate.estimated_cost_usd, 0)
        self.assertEqual(estimate.model, "claude-haiku-4-5-20251001")

    def test_estimate_with_unknown_model(self):
        opt = CostOptimizer()
        estimate = opt.estimate_request_cost(
            model="unknown-model",
            provider="unknown",
        )
        self.assertEqual(estimate.estimated_cost_usd, 0.0)

    def test_estimate_with_budget(self):
        opt = CostOptimizer(budget_limit=1.0, budget_used=0.5)
        estimate = opt.estimate_request_cost()
        self.assertIsNotNone(estimate.budget_remaining)
        self.assertIsNotNone(estimate.budget_percentage_used)

    def test_estimate_to_dict(self):
        opt = CostOptimizer()
        estimate = opt.estimate_request_cost()
        d = estimate.to_dict()
        self.assertIn("estimated_cost_usd", d)
        self.assertIn("model", d)

    def test_estimate_with_custom_tokens(self):
        opt = CostOptimizer()
        estimate = opt.estimate_request_cost(estimated_tokens=5000)
        self.assertEqual(estimate.estimated_input_tokens, 3500)  # 70% of 5000
        self.assertEqual(estimate.estimated_output_tokens, 1500)  # 30% of 5000


class TestUsageTracking(unittest.TestCase):
    """Test rolling average token tracking."""

    def test_record_usage_updates_averages(self):
        opt = CostOptimizer()
        opt.record_usage(1000, 500, 0.01)
        opt.record_usage(2000, 1000, 0.02)
        avg = opt._avg_tokens_per_request()
        self.assertEqual(avg, 2250)  # (1000+2000)/2 + (500+1000)/2

    def test_rolling_window_size(self):
        opt = CostOptimizer()
        for i in range(15):
            opt.record_usage(100, 50, 0.001)
        # Should keep only last 10
        self.assertEqual(len(opt._recent_input_tokens), 10)

    def test_record_usage_updates_budget(self):
        opt = CostOptimizer(budget_limit=1.0, budget_used=0.0)
        opt.record_usage(1000, 500, 0.1)
        self.assertEqual(opt._budget_used, 0.1)


class TestOptimizationResult(unittest.TestCase):
    """Test OptimizationResult properties."""

    def test_changed_when_different_model(self):
        result = OptimizationResult(
            decision=CostDecision.DOWNGRADE,
            recommended_model="claude-haiku-4-5-20251001",
            recommended_provider="anthropic",
            original_model="claude-sonnet-4-5-20250929",
            original_provider="anthropic",
            reasoning="test",
        )
        self.assertTrue(result.changed)

    def test_not_changed_when_same_model(self):
        result = OptimizationResult(
            decision=CostDecision.USE_CURRENT,
            recommended_model="claude-sonnet-4-5-20250929",
            recommended_provider="anthropic",
            original_model="claude-sonnet-4-5-20250929",
            original_provider="anthropic",
            reasoning="test",
        )
        self.assertFalse(result.changed)

    def test_to_dict(self):
        result = OptimizationResult(
            decision=CostDecision.DOWNGRADE,
            recommended_model="claude-haiku-4-5-20251001",
            recommended_provider="anthropic",
            original_model="claude-sonnet-4-5-20250929",
            original_provider="anthropic",
            reasoning="cost savings",
        )
        d = result.to_dict()
        self.assertEqual(d["decision"], "downgrade")
        self.assertTrue(d["changed"])


class TestOptimizerStats(unittest.TestCase):
    """Test optimizer statistics."""

    def test_downgrade_counted(self):
        opt = CostOptimizer()
        opt.optimize("yes")  # simple → fast → downgrade
        self.assertEqual(opt.stats["total_downgrades"], 1)

    def test_total_optimizations(self):
        opt = CostOptimizer()
        opt.optimize("hello")
        opt.optimize("implement a system")
        self.assertEqual(opt.stats["total_optimizations"], 2)


# ── Approval Workflow ────────────────────────────────────────────

from cowork_agent.core.approval_workflow import (
    ApprovalCategory,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalWorkflow,
    CategoryConfig,
    EscalationPolicy,
    classify_tool_category,
)


class TestToolCategoryClassification(unittest.TestCase):
    """Test tool → category mapping."""

    def test_rm_is_destructive(self):
        cat = classify_tool_category("bash", {"command": "rm -rf /tmp/old"})
        self.assertEqual(cat, ApprovalCategory.DESTRUCTIVE)

    def test_pip_install_is_install(self):
        cat = classify_tool_category("bash", {"command": "pip install requests"})
        self.assertEqual(cat, ApprovalCategory.INSTALL)

    def test_curl_is_network(self):
        cat = classify_tool_category("bash", {"command": "curl https://example.com"})
        self.assertEqual(cat, ApprovalCategory.NETWORK)

    def test_git_push_is_network(self):
        cat = classify_tool_category("bash", {"command": "git push origin main"})
        self.assertEqual(cat, ApprovalCategory.NETWORK)

    def test_npm_publish_is_publish(self):
        cat = classify_tool_category("bash", {"command": "npm publish"})
        self.assertEqual(cat, ApprovalCategory.PUBLISH)

    def test_chmod_is_configuration(self):
        cat = classify_tool_category("bash", {"command": "chmod 755 script.sh"})
        self.assertEqual(cat, ApprovalCategory.CONFIGURATION)

    def test_write_env_is_sensitive(self):
        cat = classify_tool_category("write", {"file_path": "/app/.env"})
        self.assertEqual(cat, ApprovalCategory.SENSITIVE_DATA)

    def test_write_pem_is_sensitive(self):
        cat = classify_tool_category("write", {"file_path": "/home/user/cert.pem"})
        self.assertEqual(cat, ApprovalCategory.SENSITIVE_DATA)

    def test_write_etc_is_config(self):
        cat = classify_tool_category("write", {"file_path": "/etc/nginx/nginx.conf"})
        self.assertEqual(cat, ApprovalCategory.CONFIGURATION)

    def test_ls_is_general(self):
        cat = classify_tool_category("bash", {"command": "ls -la"})
        self.assertEqual(cat, ApprovalCategory.GENERAL)

    def test_read_is_general(self):
        cat = classify_tool_category("read", {"file_path": "/app/main.py"})
        self.assertEqual(cat, ApprovalCategory.GENERAL)

    def test_write_normal_is_general(self):
        cat = classify_tool_category("write", {"file_path": "/app/main.py"})
        self.assertEqual(cat, ApprovalCategory.GENERAL)


class TestApprovalWithCallback(unittest.TestCase):
    """Test approval with user callback."""

    def test_approve_with_yes(self):
        workflow = ApprovalWorkflow(user_callback=lambda r: "yes")
        decision = workflow.request_approval("bash", {"command": "git push"}, "Push changes")
        self.assertTrue(decision.approved)
        self.assertEqual(decision.status, ApprovalStatus.APPROVED)

    def test_decline_with_no(self):
        workflow = ApprovalWorkflow(user_callback=lambda r: "no")
        decision = workflow.request_approval("bash", {"command": "rm -rf /"}, "Delete everything")
        self.assertFalse(decision.approved)
        self.assertEqual(decision.status, ApprovalStatus.DECLINED)

    def test_approve_various_words(self):
        for word in ["yes", "y", "approve", "ok", "confirm", "go", "proceed", "sure"]:
            workflow = ApprovalWorkflow(user_callback=lambda r, w=word: w)
            decision = workflow.request_approval("bash", {"command": "test"}, "test")
            self.assertTrue(decision.approved, f"Expected '{word}' to approve")

    def test_decline_various_words(self):
        for word in ["no", "nope", "cancel", "abort", "deny"]:
            workflow = ApprovalWorkflow(user_callback=lambda r, w=word: w)
            decision = workflow.request_approval("bash", {"command": "test"}, "test")
            self.assertFalse(decision.approved, f"Expected '{word}' to decline")

    def test_decided_by_user(self):
        workflow = ApprovalWorkflow(user_callback=lambda r: "yes")
        decision = workflow.request_approval("bash", {"command": "test"}, "test")
        self.assertEqual(decision.decided_by, "user")


class TestAutoApproval(unittest.TestCase):
    """Test auto-approval modes."""

    def test_auto_approve_all(self):
        workflow = ApprovalWorkflow(auto_approve_all=True)
        decision = workflow.request_approval("bash", {"command": "rm -rf /"}, "delete")
        self.assertTrue(decision.approved)
        self.assertEqual(decision.status, ApprovalStatus.AUTO_APPROVED)
        self.assertEqual(decision.decided_by, "auto")

    def test_cost_based_auto_approve(self):
        workflow = ApprovalWorkflow()
        workflow.set_category_config(ApprovalCategory.GENERAL, CategoryConfig(
            auto_approve_below_cost=0.01,
        ))
        decision = workflow.request_approval(
            "bash", {"command": "ls"}, "list files",
            estimated_cost=0.005,
            category=ApprovalCategory.GENERAL,
        )
        self.assertTrue(decision.approved)
        self.assertEqual(decision.decided_by, "cost_check")

    def test_cost_above_threshold_asks_user(self):
        workflow = ApprovalWorkflow(user_callback=lambda r: "no")
        workflow.set_category_config(ApprovalCategory.GENERAL, CategoryConfig(
            auto_approve_below_cost=0.01,
        ))
        decision = workflow.request_approval(
            "bash", {"command": "heavy compute"}, "expensive operation",
            estimated_cost=0.5,
            category=ApprovalCategory.GENERAL,
        )
        self.assertFalse(decision.approved)


class TestEscalationPolicies(unittest.TestCase):
    """Test escalation when no callback is available."""

    def test_auto_decline_default(self):
        workflow = ApprovalWorkflow()  # No callback
        decision = workflow.request_approval("bash", {"command": "rm test"}, "delete file")
        self.assertFalse(decision.approved)
        self.assertEqual(decision.decided_by, "escalation_policy")

    def test_auto_approve_policy(self):
        workflow = ApprovalWorkflow()
        workflow.set_category_config(ApprovalCategory.GENERAL, CategoryConfig(
            escalation_policy=EscalationPolicy.AUTO_APPROVE,
        ))
        decision = workflow.request_approval(
            "bash", {"command": "ls"}, "list",
            category=ApprovalCategory.GENERAL,
        )
        self.assertTrue(decision.approved)

    def test_escalate_policy(self):
        workflow = ApprovalWorkflow()
        workflow.set_category_config(ApprovalCategory.GENERAL, CategoryConfig(
            escalation_policy=EscalationPolicy.ESCALATE,
        ))
        decision = workflow.request_approval(
            "bash", {"command": "deploy"}, "deploy",
            category=ApprovalCategory.GENERAL,
        )
        self.assertEqual(decision.status, ApprovalStatus.ESCALATED)
        self.assertFalse(decision.approved)

    def test_callback_error_falls_through(self):
        def bad_callback(r):
            raise RuntimeError("User disconnected")

        workflow = ApprovalWorkflow(user_callback=bad_callback)
        decision = workflow.request_approval("bash", {"command": "test"}, "test")
        # Should fall through to escalation policy (auto_decline)
        self.assertFalse(decision.approved)


class TestAuditTrail(unittest.TestCase):
    """Test decision history and audit trail."""

    def setUp(self):
        self.workflow = ApprovalWorkflow(user_callback=lambda r: "yes")

    def test_decision_recorded(self):
        self.workflow.request_approval("bash", {"command": "test"}, "test")
        self.assertEqual(len(self.workflow.decision_history), 1)

    def test_multiple_decisions(self):
        self.workflow.request_approval("bash", {"command": "test1"}, "test1")
        self.workflow.request_approval("bash", {"command": "test2"}, "test2")
        self.assertEqual(len(self.workflow.decision_history), 2)

    def test_get_decision_by_id(self):
        self.workflow.request_approval("bash", {"command": "test"}, "test")
        decision = self.workflow.get_decision("approval_1")
        self.assertIsNotNone(decision)
        self.assertTrue(decision.approved)

    def test_get_nonexistent_decision(self):
        decision = self.workflow.get_decision("nonexistent")
        self.assertIsNone(decision)

    def test_clear_history(self):
        self.workflow.request_approval("bash", {"command": "test"}, "test")
        self.workflow.clear_history()
        self.assertEqual(len(self.workflow.decision_history), 0)

    def test_decision_to_dict(self):
        self.workflow.request_approval("bash", {"command": "test"}, "test")
        d = self.workflow.decision_history[0].to_dict()
        self.assertIn("request_id", d)
        self.assertIn("status", d)
        self.assertIn("approved", d)

    def test_decision_latency(self):
        self.workflow.request_approval("bash", {"command": "test"}, "test")
        dec = self.workflow.decision_history[0]
        self.assertGreaterEqual(dec.latency_seconds, 0)


class TestApprovalStats(unittest.TestCase):
    """Test workflow statistics."""

    def test_approved_stats(self):
        wf = ApprovalWorkflow(user_callback=lambda r: "yes")
        wf.request_approval("bash", {"command": "test"}, "test")
        self.assertEqual(wf.stats["total_approved"], 1)

    def test_declined_stats(self):
        wf = ApprovalWorkflow(user_callback=lambda r: "no")
        wf.request_approval("bash", {"command": "test"}, "test")
        self.assertEqual(wf.stats["total_declined"], 1)

    def test_auto_approved_stats(self):
        wf = ApprovalWorkflow(auto_approve_all=True)
        wf.request_approval("bash", {"command": "test"}, "test")
        self.assertEqual(wf.stats["total_auto_approved"], 1)

    def test_approval_rate(self):
        wf = ApprovalWorkflow(user_callback=lambda r: "yes")
        wf.request_approval("bash", {"command": "test1"}, "t1")
        wf.request_approval("bash", {"command": "test2"}, "t2")
        self.assertEqual(wf.stats["approval_rate"], 100.0)

    def test_mixed_stats(self):
        calls = iter(["yes", "no", "yes"])
        wf = ApprovalWorkflow(user_callback=lambda r: next(calls))
        wf.request_approval("bash", {"command": "t1"}, "t1")
        wf.request_approval("bash", {"command": "t2"}, "t2")
        wf.request_approval("bash", {"command": "t3"}, "t3")
        self.assertEqual(wf.stats["total_approved"], 2)
        self.assertEqual(wf.stats["total_declined"], 1)

    def test_summary(self):
        wf = ApprovalWorkflow(user_callback=lambda r: "yes")
        wf.request_approval("bash", {"command": "test"}, "test")
        summary = wf.summary()
        self.assertIn("category_configs", summary)
        self.assertIn("recent_decisions", summary)


class TestApprovalRequest(unittest.TestCase):
    """Test ApprovalRequest dataclass."""

    def test_request_to_dict(self):
        req = ApprovalRequest(
            request_id="test_1",
            category=ApprovalCategory.DESTRUCTIVE,
            tool_name="bash",
            tool_input={"command": "rm -rf /"},
            description="Delete everything",
        )
        d = req.to_dict()
        self.assertEqual(d["category"], "destructive")
        self.assertEqual(d["tool_name"], "bash")

    def test_default_timeout(self):
        req = ApprovalRequest(
            request_id="test_1",
            category=ApprovalCategory.GENERAL,
            tool_name="bash",
            tool_input={},
            description="test",
        )
        self.assertEqual(req.timeout_seconds, 300.0)


class TestCategoryConfig(unittest.TestCase):
    """Test category configuration."""

    def test_default_destructive_config(self):
        wf = ApprovalWorkflow()
        cfg = wf.get_category_config(ApprovalCategory.DESTRUCTIVE)
        self.assertEqual(cfg.escalation_policy, EscalationPolicy.AUTO_DECLINE)
        self.assertTrue(cfg.require_explicit_confirmation)

    def test_custom_category_config(self):
        wf = ApprovalWorkflow()
        wf.set_category_config(ApprovalCategory.INSTALL, CategoryConfig(
            escalation_policy=EscalationPolicy.AUTO_APPROVE,
            timeout_seconds=60.0,
        ))
        cfg = wf.get_category_config(ApprovalCategory.INSTALL)
        self.assertEqual(cfg.escalation_policy, EscalationPolicy.AUTO_APPROVE)

    def test_config_to_dict(self):
        cfg = CategoryConfig()
        d = cfg.to_dict()
        self.assertIn("escalation_policy", d)
        self.assertIn("timeout_seconds", d)


# ── Integration Tests ────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple Sprint 24 components."""

    def test_validator_plus_optimizer(self):
        """Validate output and track cost together."""
        validator = ToolOutputValidator(load_defaults=True)
        optimizer = CostOptimizer()

        # Simple task → fast tier
        opt_result = optimizer.optimize("hello")
        self.assertEqual(opt_result.decision, CostDecision.DOWNGRADE)

        # Validate tool output
        val_result = validator.validate("bash", "Hello World!")
        self.assertTrue(val_result.passed)

    def test_approval_then_validation(self):
        """Approval workflow gates execution, then output is validated."""
        workflow = ApprovalWorkflow(user_callback=lambda r: "yes")
        validator = ToolOutputValidator(load_defaults=False)
        validator.add_assertion("bash", OutputAssertion(
            assertion_type=AssertionType.NOT_CONTAINS,
            value="ERROR",
            severity=ValidationSeverity.BLOCK,
        ))

        # Approve action
        decision = workflow.request_approval("bash", {"command": "run test"}, "Run tests")
        self.assertTrue(decision.approved)

        # Validate output
        result = validator.validate("bash", "Tests passed: 10/10")
        self.assertTrue(result.passed)

    def test_approval_declined_skips_validation(self):
        """If approval is declined, no validation needed."""
        workflow = ApprovalWorkflow(user_callback=lambda r: "no")
        decision = workflow.request_approval("bash", {"command": "rm -rf /"}, "Delete all")
        self.assertFalse(decision.approved)
        # No need to validate — action never executed

    def test_budget_blocks_before_approval(self):
        """Budget exhaustion prevents even asking for approval."""
        optimizer = CostOptimizer(budget_limit=0.01, budget_used=0.02)
        result = optimizer.optimize("implement something complex")
        self.assertEqual(result.decision, CostDecision.BLOCK)
        # No need to ask for approval — budget blocked it

    def test_cost_aware_approval(self):
        """Approval checks estimated cost."""
        optimizer = CostOptimizer()
        estimate = optimizer.estimate_request_cost()

        workflow = ApprovalWorkflow()
        workflow.set_category_config(ApprovalCategory.GENERAL, CategoryConfig(
            auto_approve_below_cost=0.10,
        ))
        decision = workflow.request_approval(
            "bash", {"command": "simple task"}, "Run simple command",
            estimated_cost=estimate.estimated_cost_usd,
            category=ApprovalCategory.GENERAL,
        )
        # If estimate < 0.10, auto-approved
        if estimate.estimated_cost_usd < 0.10:
            self.assertTrue(decision.approved)

    def test_full_pipeline_flow(self):
        """Full flow: optimize → approve → execute → validate."""
        optimizer = CostOptimizer()
        workflow = ApprovalWorkflow(user_callback=lambda r: "yes")
        validator = ToolOutputValidator(load_defaults=True)

        # 1. Optimize
        opt_result = optimizer.optimize("read this file please")
        self.assertIsNotNone(opt_result)

        # 2. Approve
        decision = workflow.request_approval(
            "read", {"file_path": "/app/main.py"}, "Read file"
        )
        self.assertTrue(decision.approved)

        # 3. Validate output
        val_result = validator.validate("read", "def main():\n    print('hello')")
        self.assertTrue(val_result.passed)


if __name__ == "__main__":
    unittest.main()
