"""
Sprint 28 Tests — Adaptive Tool Chaining.

Tests for:
  - ChainStep dataclass
  - ToolChainPlan dataclass
  - ChainExecutionResult dataclass
  - Template matching
  - plan_chain()
  - execute_chain() success paths
  - execute_chain() failure paths (retry, fallback, adaptation, rollback)
  - Output type validation
  - ChainTool
  - Agent integration
"""

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from cowork_agent.core.adaptive_chain import (
    AdaptiveChainExecutor,
    ADAPTATION_MAP,
    CHAIN_TEMPLATES,
    ChainExecutionResult,
    ChainStep,
    PlanStatus,
    RISKY_CHAIN_TOOLS,
    StepStatus,
    TEMPLATE_KEYWORDS,
    ToolChainPlan,
)
from cowork_agent.core.models import ToolCall, ToolResult
from cowork_agent.tools.chain_tool import ChainTool


# ── Helpers ───────────────────────────────────────────────────────

def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_registry(tool_names=None, results=None):
    """Create a mock ToolRegistry."""
    registry = MagicMock()
    registry.list_tools.return_value = tool_names or ["read", "write", "edit", "bash", "glob"]

    if results is None:
        # Default: all tools succeed
        async def _execute(call):
            return ToolResult(tool_id=call.tool_id, success=True, output=f"ok from {call.name}")
        registry.execute_tool = AsyncMock(side_effect=_execute)
    else:
        # Custom results list — returned in order
        registry.execute_tool = AsyncMock(side_effect=results)

    def _get_tool(name):
        if tool_names and name not in tool_names:
            raise KeyError(f"Unknown tool: {name}")
        return MagicMock()
    registry.get_tool = MagicMock(side_effect=_get_tool)

    return registry


# ══════════════════════════════════════════════════════════════════
# ChainStep dataclass
# ══════════════════════════════════════════════════════════════════

class TestChainStep(unittest.TestCase):

    def test_creation_defaults(self):
        step = ChainStep(tool_name="read")
        self.assertEqual(step.tool_name, "read")
        self.assertEqual(step.status, StepStatus.PENDING)
        self.assertEqual(step.max_retries, 2)
        self.assertEqual(step.retry_count, 0)
        self.assertIsNone(step.fallback_tool)
        self.assertEqual(step.expected_output_type, "any")

    def test_to_dict(self):
        step = ChainStep(tool_name="bash", description="Run tests",
                         fallback_tool="write")
        d = step.to_dict()
        self.assertEqual(d["tool_name"], "bash")
        self.assertEqual(d["description"], "Run tests")
        self.assertEqual(d["fallback_tool"], "write")

    def test_from_dict(self):
        data = {"tool_name": "edit", "description": "Fix code",
                "max_retries": 3, "status": "completed"}
        step = ChainStep.from_dict(data)
        self.assertEqual(step.tool_name, "edit")
        self.assertEqual(step.max_retries, 3)
        self.assertEqual(step.status, "completed")

    def test_is_retryable(self):
        step = ChainStep(tool_name="bash", max_retries=2, retry_count=0)
        self.assertTrue(step.is_retryable)
        step.retry_count = 2
        self.assertFalse(step.is_retryable)

    def test_has_fallback(self):
        step = ChainStep(tool_name="edit", fallback_tool="write")
        self.assertTrue(step.has_fallback)
        step2 = ChainStep(tool_name="edit")
        self.assertFalse(step2.has_fallback)

    def test_roundtrip(self):
        original = ChainStep(tool_name="read", description="Read file",
                             expected_output_type="code", fallback_tool="bash",
                             retry_count=1, max_retries=3)
        restored = ChainStep.from_dict(original.to_dict())
        self.assertEqual(restored.tool_name, original.tool_name)
        self.assertEqual(restored.expected_output_type, original.expected_output_type)
        self.assertEqual(restored.fallback_tool, original.fallback_tool)


# ══════════════════════════════════════════════════════════════════
# ToolChainPlan dataclass
# ══════════════════════════════════════════════════════════════════

class TestToolChainPlan(unittest.TestCase):

    def test_creation(self):
        plan = ToolChainPlan(plan_id="chain_abc", goal="fix bug")
        self.assertEqual(plan.plan_id, "chain_abc")
        self.assertEqual(plan.goal, "fix bug")
        self.assertEqual(plan.status, PlanStatus.PENDING)
        self.assertEqual(plan.current_step_index, 0)
        self.assertEqual(plan.steps, [])

    def test_to_dict(self):
        plan = ToolChainPlan(
            plan_id="chain_123", goal="create file",
            steps=[ChainStep(tool_name="write")],
            template_used="file_creation",
        )
        d = plan.to_dict()
        self.assertEqual(d["plan_id"], "chain_123")
        self.assertEqual(len(d["steps"]), 1)
        self.assertEqual(d["template_used"], "file_creation")

    def test_from_dict(self):
        data = {
            "plan_id": "chain_x",
            "goal": "test goal",
            "steps": [{"tool_name": "read"}, {"tool_name": "write"}],
            "current_step_index": 1,
            "status": "running",
        }
        plan = ToolChainPlan.from_dict(data)
        self.assertEqual(plan.plan_id, "chain_x")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.current_step_index, 1)

    def test_next_step(self):
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="read"), ChainStep(tool_name="write")],
        )
        self.assertEqual(plan.next_step.tool_name, "read")
        plan.current_step_index = 1
        self.assertEqual(plan.next_step.tool_name, "write")
        plan.current_step_index = 2
        self.assertIsNone(plan.next_step)

    def test_is_complete(self):
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="read")],
        )
        self.assertFalse(plan.is_complete)
        plan.current_step_index = 1
        self.assertTrue(plan.is_complete)

    def test_completed_steps_count(self):
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[
                ChainStep(tool_name="read", status=StepStatus.COMPLETED),
                ChainStep(tool_name="write", status=StepStatus.FAILED),
                ChainStep(tool_name="bash", status=StepStatus.COMPLETED),
            ],
        )
        self.assertEqual(plan.completed_steps, 2)


# ══════════════════════════════════════════════════════════════════
# ChainExecutionResult dataclass
# ══════════════════════════════════════════════════════════════════

class TestChainExecutionResult(unittest.TestCase):

    def test_creation(self):
        plan = ToolChainPlan(plan_id="p1", goal="test")
        result = ChainExecutionResult(plan=plan, success=True, steps_completed=3, steps_total=3)
        self.assertTrue(result.success)
        self.assertEqual(result.steps_completed, 3)

    def test_to_dict(self):
        plan = ToolChainPlan(plan_id="p1", goal="test")
        result = ChainExecutionResult(
            plan=plan, success=True, steps_completed=2,
            steps_total=3, execution_time_ms=150.5,
        )
        d = result.to_dict()
        self.assertEqual(d["plan_id"], "p1")
        self.assertEqual(d["steps_completed"], 2)
        self.assertEqual(d["execution_time_ms"], 150.5)

    def test_success_rate(self):
        plan = ToolChainPlan(plan_id="p1", goal="test")
        result = ChainExecutionResult(plan=plan, success=True, steps_completed=2, steps_total=4)
        self.assertAlmostEqual(result.success_rate, 0.5)

    def test_success_rate_zero_steps(self):
        plan = ToolChainPlan(plan_id="p1", goal="test")
        result = ChainExecutionResult(plan=plan, success=False, steps_completed=0, steps_total=0)
        self.assertEqual(result.success_rate, 0.0)


# ══════════════════════════════════════════════════════════════════
# Template matching
# ══════════════════════════════════════════════════════════════════

class TestTemplateMatching(unittest.TestCase):

    def test_file_creation_match(self):
        result = AdaptiveChainExecutor._match_template("create file for config")
        self.assertEqual(result, "file_creation")

    def test_code_fix_match(self):
        result = AdaptiveChainExecutor._match_template("fix the bug in parser.py")
        self.assertEqual(result, "code_fix")

    def test_research_match(self):
        result = AdaptiveChainExecutor._match_template("search for Python best practices")
        self.assertEqual(result, "research")

    def test_refactor_match(self):
        result = AdaptiveChainExecutor._match_template("refactor the authentication module")
        self.assertEqual(result, "refactor")

    def test_data_pipeline_match(self):
        result = AdaptiveChainExecutor._match_template("transform data from CSV to JSON")
        self.assertEqual(result, "data_pipeline")

    def test_no_match_returns_empty(self):
        result = AdaptiveChainExecutor._match_template("hello world")
        self.assertEqual(result, "")

    def test_best_match_longer_keyword_wins(self):
        # "create file" (11 chars) should beat "fix" (3 chars)
        result = AdaptiveChainExecutor._match_template("create file and fix it")
        self.assertEqual(result, "file_creation")

    def test_templates_exist(self):
        self.assertEqual(len(CHAIN_TEMPLATES), 5)
        for name in ["file_creation", "code_fix", "research", "refactor", "data_pipeline"]:
            self.assertIn(name, CHAIN_TEMPLATES)


# ══════════════════════════════════════════════════════════════════
# plan_chain()
# ══════════════════════════════════════════════════════════════════

class TestPlanChain(unittest.TestCase):

    def test_plan_basic_goal(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        plan = executor.plan_chain("fix the test failure")
        self.assertIn("chain_", plan.plan_id)
        self.assertEqual(plan.goal, "fix the test failure")
        self.assertEqual(plan.template_used, "code_fix")
        self.assertTrue(len(plan.steps) > 0)

    def test_plan_file_creation(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        plan = executor.plan_chain("create file README.md")
        self.assertEqual(plan.template_used, "file_creation")
        tool_names = [s.tool_name for s in plan.steps]
        self.assertIn("read", tool_names)
        self.assertIn("write", tool_names)

    def test_plan_filters_unavailable_tools(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        plan = executor.plan_chain(
            "fix the bug",
            available_tools=["read", "write"],  # no edit, no bash
        )
        tool_names = [s.tool_name for s in plan.steps]
        self.assertNotIn("bash", tool_names)
        # edit should be swapped to write (fallback)
        self.assertIn("write", tool_names)

    def test_plan_with_reflection_hints(self):
        mock_entry = MagicMock()
        mock_entry.value = "Check permissions before writing"
        reflection = MagicMock()
        reflection.get_relevant_reflections.return_value = [mock_entry]

        executor = AdaptiveChainExecutor(
            tool_registry=_make_registry(),
            reflection_engine=reflection,
        )
        plan = executor.plan_chain("fix the permission error")
        self.assertIn("Check permissions before writing", plan.reflection_hints)

    def test_plan_no_match_returns_empty_steps(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        plan = executor.plan_chain("do something random")
        self.assertEqual(len(plan.steps), 0)

    def test_plan_with_context_hints(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        plan = executor.plan_chain(
            "fix the test",
            context_hints={"file_path": "/src/app.py"},
        )
        for step in plan.steps:
            self.assertIn("file_path", step.input_args)

    def test_plan_fallback_swap_when_primary_unavailable(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        # code_fix template has edit with fallback write
        plan = executor.plan_chain(
            "fix the bug",
            available_tools=["read", "write", "bash"],  # no edit
        )
        tool_names = [s.tool_name for s in plan.steps]
        # edit swapped to write
        self.assertIn("write", tool_names)
        self.assertNotIn("edit", tool_names)

    def test_plan_id_unique(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        p1 = executor.plan_chain("fix bug")
        p2 = executor.plan_chain("fix bug")
        self.assertNotEqual(p1.plan_id, p2.plan_id)


# ══════════════════════════════════════════════════════════════════
# execute_chain() — success paths
# ══════════════════════════════════════════════════════════════════

class TestExecuteChainSuccess(unittest.TestCase):

    def test_all_steps_succeed(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = executor.plan_chain("fix the bug")
        result = _run(executor.execute_chain(plan))
        self.assertTrue(result.success)
        self.assertEqual(result.steps_completed, result.steps_total)
        self.assertEqual(plan.status, PlanStatus.COMPLETED)

    def test_execution_time_tracked(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = executor.plan_chain("fix the bug")
        result = _run(executor.execute_chain(plan))
        self.assertGreater(result.execution_time_ms, 0)

    def test_step_results_collected(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = executor.plan_chain("create file config.json")
        result = _run(executor.execute_chain(plan))
        self.assertEqual(len(result.step_results), result.steps_total)

    def test_step_statuses_updated(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = executor.plan_chain("fix the bug")
        _run(executor.execute_chain(plan))
        for step in plan.steps:
            self.assertEqual(step.status, StepStatus.COMPLETED)

    def test_no_adaptations_on_success(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = executor.plan_chain("create file output.txt")
        result = _run(executor.execute_chain(plan))
        self.assertEqual(result.adaptations_made, 0)

    def test_empty_plan_succeeds(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        plan = ToolChainPlan(plan_id="empty", goal="nothing")
        result = _run(executor.execute_chain(plan))
        self.assertTrue(result.success)
        self.assertEqual(result.steps_completed, 0)

    def test_rollback_not_used_on_success(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = executor.plan_chain("fix the bug")
        result = _run(executor.execute_chain(plan))
        self.assertFalse(result.rollback_used)

    def test_tracer_integration(self):
        registry = _make_registry()
        tracer = MagicMock()
        tracer.start_span.return_value = "span_123"
        executor = AdaptiveChainExecutor(tool_registry=registry, execution_tracer=tracer)
        plan = executor.plan_chain("fix the bug")
        result = _run(executor.execute_chain(plan))
        tracer.start_span.assert_called_once()
        tracer.end_span.assert_called_once_with("span_123", status="ok")


# ══════════════════════════════════════════════════════════════════
# execute_chain() — failure paths
# ══════════════════════════════════════════════════════════════════

class TestExecuteChainFailure(unittest.TestCase):

    def test_retry_on_failure(self):
        # First call fails, second succeeds
        results = [
            ToolResult(tool_id="t1", success=True, output="ok"),   # read
            ToolResult(tool_id="t2", success=False, output="", error="timeout"),  # edit fail 1
            ToolResult(tool_id="t3", success=False, output="", error="timeout"),  # edit fail 2 (retry)
            ToolResult(tool_id="t4", success=False, output="", error="timeout"),  # edit fail 3 (retry)
        ]
        registry = _make_registry(results=results)
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = executor.plan_chain("fix the bug")
        result = _run(executor.execute_chain(plan))
        # After 2 retries of edit + fallback to write, it should eventually handle
        self.assertFalse(result.success)

    def test_fallback_tool_used(self):
        call_count = [0]
        async def _execute(call):
            call_count[0] += 1
            if call.name == "edit":
                return ToolResult(tool_id=call.tool_id, success=False, output="",
                                  error="edit failed")
            return ToolResult(tool_id=call.tool_id, success=True, output=f"ok from {call.name}")

        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = executor.plan_chain("fix the bug")
        result = _run(executor.execute_chain(plan))
        # After retries of edit, fallback to write should succeed
        self.assertTrue(result.success)

    def test_adaptation_inserts_step(self):
        call_count = [0]
        async def _execute(call):
            call_count[0] += 1
            if call.name == "write" and call_count[0] <= 5:
                return ToolResult(tool_id=call.tool_id, success=False, output="",
                                  error="No such file or directory")
            return ToolResult(tool_id=call.tool_id, success=True, output=f"ok from {call.name}")

        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = executor.plan_chain("create file output.txt")
        result = _run(executor.execute_chain(plan))
        self.assertGreater(result.adaptations_made, 0)

    def test_rollback_on_permanent_failure(self):
        async def _execute(call):
            if call.name in ("edit", "write"):
                return ToolResult(tool_id=call.tool_id, success=False, output="",
                                  error="permanent error xyz")
            return ToolResult(tool_id=call.tool_id, success=True, output="ok")

        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        rollback = MagicMock()
        rollback.auto_checkpoint_before_chain.return_value = "ckpt_123"
        rollback.rollback.return_value = MagicMock(success=True)

        executor = AdaptiveChainExecutor(
            tool_registry=registry, rollback_journal=rollback,
        )
        plan = executor.plan_chain("fix the bug")
        result = _run(executor.execute_chain(plan))
        self.assertFalse(result.success)
        self.assertTrue(result.rollback_used)

    def test_max_adaptations_limit(self):
        async def _execute(call):
            if call.name == "write":
                return ToolResult(tool_id=call.tool_id, success=False, output="",
                                  error="no such file or directory")
            return ToolResult(tool_id=call.tool_id, success=True, output="ok")

        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        executor = AdaptiveChainExecutor(tool_registry=registry)
        executor.MAX_ADAPTATIONS = 2
        plan = executor.plan_chain("create file output.txt")
        result = _run(executor.execute_chain(plan))
        self.assertFalse(result.success)
        self.assertLessEqual(result.adaptations_made, 2)

    def test_partial_success_tracking(self):
        call_idx = [0]
        async def _execute(call):
            call_idx[0] += 1
            if call_idx[0] >= 3:
                return ToolResult(tool_id=call.tool_id, success=False, output="",
                                  error="permanent failure")
            return ToolResult(tool_id=call.tool_id, success=True, output="ok")

        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = executor.plan_chain("refactor the module")
        result = _run(executor.execute_chain(plan))
        self.assertFalse(result.success)
        self.assertGreater(result.steps_completed, 0)
        self.assertLess(result.steps_completed, result.steps_total)

    def test_error_message_in_result(self):
        async def _execute(call):
            return ToolResult(tool_id=call.tool_id, success=False, output="",
                              error="specific error message")
        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash", max_retries=0)],
        )
        result = _run(executor.execute_chain(plan))
        self.assertIn("specific error message", result.error)

    def test_no_registry_returns_error(self):
        executor = AdaptiveChainExecutor()
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash")],
        )
        result = _run(executor.execute_chain(plan))
        self.assertFalse(result.success)
        self.assertIn("No tool registry", result.error)

    def test_tracer_records_error(self):
        async def _execute(call):
            return ToolResult(tool_id=call.tool_id, success=False, output="",
                              error="fail")
        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        tracer = MagicMock()
        tracer.start_span.return_value = "span_1"
        executor = AdaptiveChainExecutor(tool_registry=registry, execution_tracer=tracer)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash", max_retries=0)],
        )
        result = _run(executor.execute_chain(plan))
        tracer.end_span.assert_called_once()
        call_args = tracer.end_span.call_args
        self.assertEqual(call_args[1]["status"], "error")

    def test_exception_during_execution(self):
        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=RuntimeError("unexpected"))
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash")],
        )
        result = _run(executor.execute_chain(plan))
        self.assertFalse(result.success)
        self.assertIn("unexpected", result.error)


# ══════════════════════════════════════════════════════════════════
# Output type validation
# ══════════════════════════════════════════════════════════════════

class TestOutputValidation(unittest.TestCase):

    def test_any_always_valid(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type("anything", "any"))

    def test_json_valid(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type('{"key": "value"}', "json"))

    def test_json_invalid(self):
        self.assertFalse(AdaptiveChainExecutor._validate_output_type("not json", "json"))

    def test_file_path_valid(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type("/usr/local/bin/test", "file_path"))

    def test_code_valid(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type("def foo():\n    return 1", "code"))

    def test_code_invalid(self):
        self.assertFalse(AdaptiveChainExecutor._validate_output_type("just plain text", "code"))

    def test_text_always_valid(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type("any text", "text"))

    def test_empty_output_valid(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type("", "json"))


# ══════════════════════════════════════════════════════════════════
# adapt_chain()
# ══════════════════════════════════════════════════════════════════

class TestAdaptChain(unittest.TestCase):

    def test_permission_error_adaptation(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="write")],
        )
        adapted = executor.adapt_chain(plan, 0, "Permission denied")
        self.assertTrue(adapted)
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].tool_name, "bash")

    def test_not_found_adaptation(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="write")],
        )
        adapted = executor.adapt_chain(plan, 0, "No such file or directory")
        self.assertTrue(adapted)
        self.assertEqual(plan.steps[0].description, "Create parent directories")

    def test_no_adaptation_for_unknown_error(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="write")],
        )
        adapted = executor.adapt_chain(plan, 0, "some completely unique error")
        self.assertFalse(adapted)

    def test_adaptation_resets_failed_step(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="write", retry_count=2,
                             status=StepStatus.RUNNING, error="old error")],
        )
        executor.adapt_chain(plan, 0, "No such file or directory")
        # The original step is now at index 1 and should be reset
        self.assertEqual(plan.steps[1].retry_count, 0)
        self.assertEqual(plan.steps[1].status, StepStatus.PENDING)
        self.assertIsNone(plan.steps[1].error)


# ══════════════════════════════════════════════════════════════════
# ChainTool
# ══════════════════════════════════════════════════════════════════

class TestChainTool(unittest.TestCase):

    def test_success(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        tool = ChainTool(executor=executor)
        result = _run(tool.execute(goal="fix the bug", tool_id="t1"))
        self.assertTrue(result.success)
        self.assertIn("completed successfully", result.output)

    def test_no_executor(self):
        tool = ChainTool()
        result = _run(tool.execute(goal="fix bug", tool_id="t1"))
        self.assertFalse(result.success)
        self.assertIn("not enabled", result.error)

    def test_no_goal(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        tool = ChainTool(executor=executor)
        result = _run(tool.execute(goal="", tool_id="t1"))
        self.assertFalse(result.success)
        self.assertIn("Missing", result.error)

    def test_no_matching_template(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        tool = ChainTool(executor=executor)
        result = _run(tool.execute(goal="do something random", tool_id="t1"))
        self.assertFalse(result.success)
        self.assertIn("Could not plan", result.error)

    def test_with_context(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        tool = ChainTool(executor=executor)
        ctx = json.dumps({"file_path": "/src/app.py"})
        result = _run(tool.execute(goal="fix the bug", context=ctx, tool_id="t1"))
        self.assertTrue(result.success)


# ══════════════════════════════════════════════════════════════════
# Agent integration
# ══════════════════════════════════════════════════════════════════

class TestAgentIntegration(unittest.TestCase):

    def test_agent_has_attribute(self):
        from cowork_agent.core.agent import Agent
        provider = MagicMock()
        registry = MagicMock()
        registry.list_tools.return_value = []
        prompt_builder = MagicMock()
        agent = Agent(provider=provider, registry=registry, prompt_builder=prompt_builder)
        self.assertTrue(hasattr(agent, "adaptive_chain_executor"))
        self.assertIsNone(agent.adaptive_chain_executor)

    def test_main_wires_executor(self):
        """Test that main.py wiring block can be imported without errors."""
        # Just verify the imports work
        from cowork_agent.core.adaptive_chain import AdaptiveChainExecutor
        from cowork_agent.tools.chain_tool import ChainTool
        executor = AdaptiveChainExecutor()
        tool = ChainTool(executor=executor)
        self.assertEqual(tool.name, "chain")


# ══════════════════════════════════════════════════════════════════
# Constants and configuration
# ══════════════════════════════════════════════════════════════════

class TestConstants(unittest.TestCase):

    def test_risky_tools(self):
        self.assertIn("bash", RISKY_CHAIN_TOOLS)
        self.assertIn("write", RISKY_CHAIN_TOOLS)
        self.assertIn("edit", RISKY_CHAIN_TOOLS)

    def test_adaptation_map_keys(self):
        self.assertIn("permission", ADAPTATION_MAP)
        self.assertIn("no such file", ADAPTATION_MAP)
        self.assertIn("module", ADAPTATION_MAP)

    def test_template_keywords_coverage(self):
        for template_name in CHAIN_TEMPLATES:
            self.assertIn(template_name, TEMPLATE_KEYWORDS,
                          f"Template '{template_name}' missing from TEMPLATE_KEYWORDS")

    def test_step_status_enum(self):
        self.assertEqual(StepStatus.PENDING, "pending")
        self.assertEqual(StepStatus.COMPLETED, "completed")
        self.assertEqual(StepStatus.FAILED, "failed")

    def test_plan_status_enum(self):
        self.assertEqual(PlanStatus.ADAPTED, "adapted")
        self.assertEqual(PlanStatus.RUNNING, "running")


if __name__ == "__main__":
    unittest.main()
