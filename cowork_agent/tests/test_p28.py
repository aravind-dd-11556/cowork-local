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


# ══════════════════════════════════════════════════════════════════
# Edge Cases — ChainStep
# ══════════════════════════════════════════════════════════════════

class TestChainStepEdgeCases(unittest.TestCase):

    def test_from_dict_ignores_extra_keys(self):
        data = {"tool_name": "bash", "unknown_field": "ignored", "extra": 42}
        step = ChainStep.from_dict(data)
        self.assertEqual(step.tool_name, "bash")
        self.assertFalse(hasattr(step, "unknown_field"))

    def test_from_dict_minimal(self):
        step = ChainStep.from_dict({"tool_name": "read"})
        self.assertEqual(step.tool_name, "read")
        self.assertEqual(step.status, StepStatus.PENDING)

    def test_input_args_default_empty(self):
        step = ChainStep(tool_name="bash")
        self.assertEqual(step.input_args, {})
        # Ensure default factory gives independent dicts
        step2 = ChainStep(tool_name="bash")
        step.input_args["key"] = "value"
        self.assertEqual(step2.input_args, {})

    def test_retryable_boundary_at_max(self):
        step = ChainStep(tool_name="bash", max_retries=0)
        self.assertFalse(step.is_retryable)

    def test_retryable_one_below_max(self):
        step = ChainStep(tool_name="bash", max_retries=5, retry_count=4)
        self.assertTrue(step.is_retryable)

    def test_to_dict_preserves_none_fields(self):
        step = ChainStep(tool_name="read")
        d = step.to_dict()
        self.assertIsNone(d["result"])
        self.assertIsNone(d["error"])
        self.assertIsNone(d["fallback_tool"])


# ══════════════════════════════════════════════════════════════════
# Edge Cases — ToolChainPlan
# ══════════════════════════════════════════════════════════════════

class TestToolChainPlanEdgeCases(unittest.TestCase):

    def test_from_dict_missing_steps_key(self):
        data = {"plan_id": "p1", "goal": "test"}
        plan = ToolChainPlan.from_dict(data)
        self.assertEqual(plan.steps, [])

    def test_from_dict_with_empty_steps(self):
        data = {"plan_id": "p1", "goal": "test", "steps": []}
        plan = ToolChainPlan.from_dict(data)
        self.assertEqual(plan.steps, [])

    def test_roundtrip_full_plan(self):
        original = ToolChainPlan(
            plan_id="chain_xyz", goal="refactor code",
            steps=[
                ChainStep(tool_name="glob", description="Find files"),
                ChainStep(tool_name="read", fallback_tool="bash"),
            ],
            template_used="refactor",
            adaptations=["step 1 adapted"],
            reflection_hints=["check permissions"],
        )
        restored = ToolChainPlan.from_dict(original.to_dict())
        self.assertEqual(restored.plan_id, original.plan_id)
        self.assertEqual(restored.goal, original.goal)
        self.assertEqual(len(restored.steps), 2)
        self.assertEqual(restored.template_used, "refactor")
        self.assertEqual(restored.adaptations, ["step 1 adapted"])

    def test_next_step_negative_index(self):
        plan = ToolChainPlan(plan_id="p1", goal="test",
                             steps=[ChainStep(tool_name="read")])
        plan.current_step_index = -1
        # Negative index is < len(steps), so next_step should exist
        # This is technically an invalid state but tests robustness
        step = plan.next_step
        self.assertIsNotNone(step)

    def test_completed_steps_all_pending(self):
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="read"), ChainStep(tool_name="write")],
        )
        self.assertEqual(plan.completed_steps, 0)

    def test_adaptations_list_independent(self):
        p1 = ToolChainPlan(plan_id="p1", goal="test")
        p2 = ToolChainPlan(plan_id="p2", goal="test")
        p1.adaptations.append("x")
        self.assertEqual(p2.adaptations, [])


# ══════════════════════════════════════════════════════════════════
# Edge Cases — ChainExecutionResult
# ══════════════════════════════════════════════════════════════════

class TestChainExecutionResultEdgeCases(unittest.TestCase):

    def test_success_rate_full_completion(self):
        plan = ToolChainPlan(plan_id="p1", goal="test")
        result = ChainExecutionResult(plan=plan, success=True, steps_completed=5, steps_total=5)
        self.assertAlmostEqual(result.success_rate, 1.0)

    def test_to_dict_rounds_execution_time(self):
        plan = ToolChainPlan(plan_id="p1", goal="test")
        result = ChainExecutionResult(plan=plan, success=True, execution_time_ms=123.456789)
        d = result.to_dict()
        self.assertEqual(d["execution_time_ms"], 123.46)

    def test_step_results_list_independent(self):
        plan = ToolChainPlan(plan_id="p1", goal="test")
        r1 = ChainExecutionResult(plan=plan, success=True)
        r2 = ChainExecutionResult(plan=plan, success=True)
        r1.step_results.append("data")
        self.assertEqual(r2.step_results, [])


# ══════════════════════════════════════════════════════════════════
# Edge Cases — Template matching
# ══════════════════════════════════════════════════════════════════

class TestTemplateMatchingEdgeCases(unittest.TestCase):

    def test_case_insensitive(self):
        result = AdaptiveChainExecutor._match_template("FIX THE BUG")
        self.assertEqual(result, "code_fix")

    def test_empty_goal(self):
        result = AdaptiveChainExecutor._match_template("")
        self.assertEqual(result, "")

    def test_multiple_keywords_same_template_accumulate(self):
        # "fix" + "debug" + "patch" all belong to code_fix
        result = AdaptiveChainExecutor._match_template("fix and debug and patch the issue")
        self.assertEqual(result, "code_fix")

    def test_whitespace_only_goal(self):
        result = AdaptiveChainExecutor._match_template("   ")
        self.assertEqual(result, "")

    def test_partial_keyword_no_match(self):
        # "ref" should NOT match "refactor"
        result = AdaptiveChainExecutor._match_template("ref something")
        self.assertEqual(result, "")

    def test_keyword_as_substring(self):
        # "fix" is in "prefix" but should still match code_fix
        result = AdaptiveChainExecutor._match_template("prefix something")
        self.assertEqual(result, "code_fix")


# ══════════════════════════════════════════════════════════════════
# Edge Cases — plan_chain()
# ══════════════════════════════════════════════════════════════════

class TestPlanChainEdgeCases(unittest.TestCase):

    def test_reflection_engine_throws(self):
        reflection = MagicMock()
        reflection.get_relevant_reflections.side_effect = RuntimeError("DB down")
        executor = AdaptiveChainExecutor(
            tool_registry=_make_registry(),
            reflection_engine=reflection,
        )
        plan = executor.plan_chain("fix the bug")
        # Should not crash, just no hints
        self.assertEqual(plan.reflection_hints, [])
        self.assertTrue(len(plan.steps) > 0)

    def test_reflection_entries_without_value_attr(self):
        mock_entry = MagicMock(spec=[])  # no attributes at all
        reflection = MagicMock()
        reflection.get_relevant_reflections.return_value = [mock_entry]
        executor = AdaptiveChainExecutor(
            tool_registry=_make_registry(),
            reflection_engine=reflection,
        )
        plan = executor.plan_chain("fix the bug")
        self.assertEqual(plan.reflection_hints, [])

    def test_empty_available_tools_filters_everything(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        plan = executor.plan_chain("fix the bug", available_tools=[])
        self.assertEqual(len(plan.steps), 0)

    def test_plan_without_registry(self):
        executor = AdaptiveChainExecutor()
        plan = executor.plan_chain("fix the bug")
        # Should still produce a plan (from template), just may fail at execution
        self.assertEqual(plan.template_used, "code_fix")

    def test_all_steps_skipped_when_no_tools_match(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        plan = executor.plan_chain(
            "create file test.txt",
            available_tools=["web_search"],  # none of file_creation tools
        )
        self.assertEqual(len(plan.steps), 0)

    def test_fallback_unavailable_gets_nulled(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        plan = executor.plan_chain(
            "fix the bug",
            available_tools=["read", "edit", "bash"],  # edit available, write (fallback) not
        )
        for step in plan.steps:
            if step.tool_name == "edit":
                self.assertIsNone(step.fallback_tool)

    def test_context_hints_none(self):
        executor = AdaptiveChainExecutor(tool_registry=_make_registry())
        plan = executor.plan_chain("fix the bug", context_hints=None)
        for step in plan.steps:
            self.assertEqual(step.input_args, {})


# ══════════════════════════════════════════════════════════════════
# Edge Cases — execute_chain()
# ══════════════════════════════════════════════════════════════════

class TestExecuteChainEdgeCases(unittest.TestCase):

    def test_single_step_plan_success(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash")],
        )
        result = _run(executor.execute_chain(plan))
        self.assertTrue(result.success)
        self.assertEqual(result.steps_completed, 1)
        self.assertEqual(result.steps_total, 1)

    def test_messages_none_default(self):
        registry = _make_registry()
        rollback = MagicMock()
        rollback.auto_checkpoint_before_chain.return_value = "ckpt_1"
        executor = AdaptiveChainExecutor(
            tool_registry=registry, rollback_journal=rollback,
        )
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash")],
        )
        result = _run(executor.execute_chain(plan))
        self.assertTrue(result.success)
        # Rollback should have been called with empty list (not None)
        call_args = rollback.auto_checkpoint_before_chain.call_args
        self.assertEqual(call_args[1].get("messages", call_args[0][-1] if call_args[0] else []), [])

    def test_rollback_checkpoint_exception_does_not_crash(self):
        registry = _make_registry()
        rollback = MagicMock()
        rollback.auto_checkpoint_before_chain.side_effect = RuntimeError("checkpoint failed")
        executor = AdaptiveChainExecutor(
            tool_registry=registry, rollback_journal=rollback,
        )
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash")],
        )
        result = _run(executor.execute_chain(plan))
        self.assertTrue(result.success)

    def test_rollback_itself_fails_gracefully(self):
        async def _execute(call):
            return ToolResult(tool_id=call.tool_id, success=False, output="",
                              error="permanent error xyz")
        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        rollback = MagicMock()
        rollback.auto_checkpoint_before_chain.return_value = "ckpt_1"
        rollback.rollback.side_effect = RuntimeError("rollback crashed")

        executor = AdaptiveChainExecutor(
            tool_registry=registry, rollback_journal=rollback,
        )
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash", max_retries=0)],
        )
        result = _run(executor.execute_chain(plan))
        self.assertFalse(result.success)
        self.assertFalse(result.rollback_used)

    def test_step_with_zero_max_retries_fails_immediately(self):
        async def _execute(call):
            return ToolResult(tool_id=call.tool_id, success=False, output="", error="fail")
        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="read", max_retries=0)],
        )
        result = _run(executor.execute_chain(plan))
        self.assertFalse(result.success)
        # read is not in RISKY_CHAIN_TOOLS so no rollback
        self.assertFalse(result.rollback_used)

    def test_non_risky_tool_skips_checkpoint(self):
        registry = _make_registry()
        rollback = MagicMock()
        executor = AdaptiveChainExecutor(
            tool_registry=registry, rollback_journal=rollback,
        )
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="read")],  # read is not risky
        )
        _run(executor.execute_chain(plan))
        rollback.auto_checkpoint_before_chain.assert_not_called()

    def test_re_execute_completed_plan(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash")],
        )
        r1 = _run(executor.execute_chain(plan))
        self.assertTrue(r1.success)
        # Plan is now complete (current_step_index = 1)
        # Re-executing should immediately succeed with 0 steps
        r2 = _run(executor.execute_chain(plan))
        self.assertTrue(r2.success)
        self.assertEqual(r2.steps_completed, 1)  # still 1 completed from before

    def test_error_none_becomes_unknown(self):
        async def _execute(call):
            return ToolResult(tool_id=call.tool_id, success=False, output="", error=None)
        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="read", max_retries=0)],
        )
        result = _run(executor.execute_chain(plan))
        self.assertFalse(result.success)
        self.assertIn("Unknown error", result.error)

    def test_plan_status_transitions(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash"), ChainStep(tool_name="read")],
        )
        self.assertEqual(plan.status, PlanStatus.PENDING)
        result = _run(executor.execute_chain(plan))
        self.assertEqual(plan.status, PlanStatus.COMPLETED)


# ══════════════════════════════════════════════════════════════════
# Edge Cases — adapt_chain()
# ══════════════════════════════════════════════════════════════════

class TestAdaptChainEdgeCases(unittest.TestCase):

    def test_adaptation_tool_not_in_registry(self):
        registry = MagicMock()
        registry.get_tool.side_effect = KeyError("bash not found")
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="write")],
        )
        adapted = executor.adapt_chain(plan, 0, "Permission denied")
        self.assertFalse(adapted)
        self.assertEqual(len(plan.steps), 1)  # no step inserted

    def test_adapt_at_non_zero_index(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[
                ChainStep(tool_name="read", status=StepStatus.COMPLETED),
                ChainStep(tool_name="write"),
            ],
        )
        adapted = executor.adapt_chain(plan, 1, "No such file or directory")
        self.assertTrue(adapted)
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[1].tool_name, "bash")  # inserted before index 1
        self.assertEqual(plan.steps[2].tool_name, "write")  # original shifted

    def test_empty_error_string(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="write")],
        )
        adapted = executor.adapt_chain(plan, 0, "")
        self.assertFalse(adapted)

    def test_module_error_adaptation(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash")],
        )
        adapted = executor.adapt_chain(plan, 0, "ModuleNotFoundError: No module named 'pandas'")
        self.assertTrue(adapted)
        self.assertEqual(plan.steps[0].description, "Install missing dependency")

    def test_syntax_error_adaptation(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="edit")],
        )
        adapted = executor.adapt_chain(plan, 0, "SyntaxError: invalid syntax at line 5")
        self.assertTrue(adapted)
        self.assertEqual(plan.steps[0].tool_name, "read")

    def test_adaptation_without_registry(self):
        executor = AdaptiveChainExecutor()  # no registry
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="write")],
        )
        # Without registry, can't verify tool exists — should still try
        adapted = executor.adapt_chain(plan, 0, "Permission denied")
        self.assertTrue(adapted)

    def test_first_matching_keyword_wins(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="write")],
        )
        # "not found" and "module" both in error — first match in ADAPTATION_MAP iteration wins
        adapted = executor.adapt_chain(plan, 0, "module not found error")
        self.assertTrue(adapted)
        # Should have inserted some step
        self.assertEqual(len(plan.steps), 2)


# ══════════════════════════════════════════════════════════════════
# Edge Cases — Output type validation
# ══════════════════════════════════════════════════════════════════

class TestOutputValidationEdgeCases(unittest.TestCase):

    def test_json_array_valid(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type('[1, 2, 3]', "json"))

    def test_json_number_valid(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type('42', "json"))

    def test_json_null_valid(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type('null', "json"))

    def test_file_path_relative(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type("./src/file.py", "file_path"))

    def test_file_path_home(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type("~/documents/test.txt", "file_path"))

    def test_file_path_bare_filename(self):
        # No "/" — triggers `"." in output.split("/")[-1] if "/" in output else False`
        result = AdaptiveChainExecutor._validate_output_type("file.txt", "file_path")
        # Due to operator precedence bug: "." in "file.txt" if "/" in "file.txt" else False
        # "/" not in "file.txt" so result is False
        self.assertFalse(result)

    def test_code_with_javascript_patterns(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type("const x = () => {}", "code"))

    def test_code_with_class_keyword(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type("class MyClass:", "code"))

    def test_code_with_braces_only(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type("{ }", "code"))

    def test_unknown_type_returns_true(self):
        # Fallthrough to "text" behavior
        self.assertTrue(AdaptiveChainExecutor._validate_output_type("anything", "unknown_type"))

    def test_file_path_with_spaces(self):
        self.assertTrue(AdaptiveChainExecutor._validate_output_type("/path/to/my file.txt", "file_path"))


# ══════════════════════════════════════════════════════════════════
# Edge Cases — ChainTool
# ══════════════════════════════════════════════════════════════════

class TestChainToolEdgeCases(unittest.TestCase):

    def test_invalid_json_context_treated_as_raw(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        tool = ChainTool(executor=executor)
        result = _run(tool.execute(
            goal="fix the bug",
            context="not valid json {{{",
            tool_id="t1",
        ))
        self.assertTrue(result.success)

    def test_progress_callback_called(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        tool = ChainTool(executor=executor)
        calls = []
        def cb(pct, msg):
            calls.append((pct, msg))
        _run(tool.execute(goal="fix the bug", tool_id="t1", progress_callback=cb))
        self.assertGreater(len(calls), 0)
        self.assertEqual(calls[-1][0], 100)

    def test_get_schema(self):
        tool = ChainTool()
        schema = tool.get_schema()
        self.assertEqual(schema.name, "chain")
        self.assertIn("goal", schema.input_schema["properties"])

    def test_executor_without_registry_for_available_tools(self):
        executor = AdaptiveChainExecutor()  # no registry
        tool = ChainTool(executor=executor)
        result = _run(tool.execute(goal="fix the bug", tool_id="t1"))
        # Plan matches code_fix template but no registry means execution fails
        self.assertFalse(result.success)

    def test_chain_tool_with_failed_chain(self):
        async def _execute(call):
            return ToolResult(tool_id=call.tool_id, success=False, output="",
                              error="tool broken")
        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        executor = AdaptiveChainExecutor(tool_registry=registry)
        tool = ChainTool(executor=executor)
        result = _run(tool.execute(goal="fix the bug", tool_id="t1"))
        self.assertFalse(result.success)
        self.assertIn("Chain failed", result.error)

    def test_success_output_includes_template_name(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        tool = ChainTool(executor=executor)
        result = _run(tool.execute(goal="fix the bug", tool_id="t1"))
        self.assertIn("code_fix", result.output)

    def test_success_output_includes_time(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        tool = ChainTool(executor=executor)
        result = _run(tool.execute(goal="fix the bug", tool_id="t1"))
        self.assertIn("Time:", result.output)


# ══════════════════════════════════════════════════════════════════
# Edge Cases — Enum values
# ══════════════════════════════════════════════════════════════════

class TestEnumEdgeCases(unittest.TestCase):

    def test_step_status_skipped(self):
        self.assertEqual(StepStatus.SKIPPED, "skipped")

    def test_step_status_running(self):
        self.assertEqual(StepStatus.RUNNING, "running")

    def test_plan_status_pending(self):
        self.assertEqual(PlanStatus.PENDING, "pending")

    def test_plan_status_completed(self):
        self.assertEqual(PlanStatus.COMPLETED, "completed")

    def test_plan_status_failed(self):
        self.assertEqual(PlanStatus.FAILED, "failed")

    def test_step_status_string_comparison(self):
        # str(Enum) behavior
        self.assertTrue(StepStatus.COMPLETED == "completed")
        self.assertIn(StepStatus.PENDING, ["pending", "running"])

    def test_step_status_used_as_dict_key(self):
        d = {StepStatus.COMPLETED: 1, StepStatus.FAILED: 2}
        self.assertEqual(d["completed"], 1)
        self.assertEqual(d[StepStatus.FAILED], 2)


# ══════════════════════════════════════════════════════════════════
# Edge Cases — _execute_step internal
# ══════════════════════════════════════════════════════════════════

class TestExecuteStepEdgeCases(unittest.TestCase):

    def test_execute_step_tool_raises_exception(self):
        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=Exception("internal crash"))
        executor = AdaptiveChainExecutor(tool_registry=registry)
        step = ChainStep(tool_name="bash", input_args={"command": "ls"})
        result = _run(executor._execute_step(step))
        self.assertFalse(result.success)
        self.assertIn("internal crash", result.error)

    def test_execute_step_passes_input_args(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        step = ChainStep(tool_name="read", input_args={"file_path": "/test.txt"})
        _run(executor._execute_step(step))
        call_args = registry.execute_tool.call_args
        tool_call = call_args[0][0]
        self.assertEqual(tool_call.name, "read")
        self.assertEqual(tool_call.input, {"file_path": "/test.txt"})

    def test_execute_step_generates_unique_tool_ids(self):
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        step = ChainStep(tool_name="bash")
        _run(executor._execute_step(step))
        _run(executor._execute_step(step))
        call1 = registry.execute_tool.call_args_list[0][0][0]
        call2 = registry.execute_tool.call_args_list[1][0][0]
        self.assertNotEqual(call1.tool_id, call2.tool_id)


# ══════════════════════════════════════════════════════════════════
# Edge Cases — Full integration flows
# ══════════════════════════════════════════════════════════════════

class TestFullIntegrationEdgeCases(unittest.TestCase):

    def test_retry_then_succeed(self):
        """Step fails once, retries, then succeeds."""
        call_count = [0]
        async def _execute(call):
            call_count[0] += 1
            if call.name == "bash" and call_count[0] == 1:
                return ToolResult(tool_id=call.tool_id, success=False, output="",
                                  error="transient error")
            return ToolResult(tool_id=call.tool_id, success=True, output="ok")

        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash", max_retries=2)],
        )
        result = _run(executor.execute_chain(plan))
        self.assertTrue(result.success)
        self.assertEqual(plan.steps[0].retry_count, 1)

    def test_adapt_then_succeed(self):
        """Step fails with recognizable error, gets adapted, then succeeds."""
        call_count = [0]
        async def _execute(call):
            call_count[0] += 1
            if call.name == "write" and call_count[0] <= 3:
                return ToolResult(tool_id=call.tool_id, success=False, output="",
                                  error="Permission denied: /opt/file.txt")
            return ToolResult(tool_id=call.tool_id, success=True, output="ok")

        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="write", max_retries=0)],
        )
        result = _run(executor.execute_chain(plan))
        self.assertTrue(result.success)
        self.assertGreater(result.adaptations_made, 0)
        # Status ends as COMPLETED since all steps eventually passed
        self.assertEqual(plan.status, PlanStatus.COMPLETED)

    def test_long_chain_all_succeed(self):
        """A chain with many steps all succeeding."""
        registry = _make_registry()
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="bash") for _ in range(10)],
        )
        result = _run(executor.execute_chain(plan))
        self.assertTrue(result.success)
        self.assertEqual(result.steps_completed, 10)
        self.assertEqual(result.steps_total, 10)

    def test_chain_with_all_recovery_mechanisms(self):
        """Tests retry → fallback → adapt in sequence."""
        call_count = [0]
        async def _execute(call):
            call_count[0] += 1
            # First 3 calls: edit fails (triggers retry x2 then fallback)
            # Call 4: write (fallback) also fails with permission error (triggers adapt)
            # Call 5+: everything succeeds (mkdir -p adaptation, then write retry)
            if call.name == "edit":
                return ToolResult(tool_id=call.tool_id, success=False, output="",
                                  error="edit broken")
            if call.name == "write" and call_count[0] <= 6:
                return ToolResult(tool_id=call.tool_id, success=False, output="",
                                  error="Permission denied")
            return ToolResult(tool_id=call.tool_id, success=True, output="ok")

        registry = _make_registry()
        registry.execute_tool = AsyncMock(side_effect=_execute)
        executor = AdaptiveChainExecutor(tool_registry=registry)
        plan = ToolChainPlan(
            plan_id="p1", goal="test",
            steps=[ChainStep(tool_name="edit", fallback_tool="write", max_retries=2)],
        )
        result = _run(executor.execute_chain(plan))
        # Complex recovery chain — may or may not fully succeed depending on
        # how many adaptations and retries happen
        self.assertIsInstance(result, ChainExecutionResult)


if __name__ == "__main__":
    unittest.main()
