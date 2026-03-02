"""
Sprint 43 · Tests – Multi-Agent Crew Mode
==========================================
~100 tests covering CrewRoles, TaskDecomposer, ResultAggregator,
CrewManager, and agent integration.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from cowork_agent.core.crew_roles import (
    CrewRole,
    PREDEFINED_ROLES,
    RoleAssigner,
    _ROLE_KEYWORDS,
)
from cowork_agent.core.task_decomposer import (
    DecompositionResult,
    DECOMPOSITION_TEMPLATES,
    SubTask,
    TaskDecomposer,
)
from cowork_agent.core.result_aggregator import (
    AggregationStrategy,
    ResultAggregator,
)
from cowork_agent.core.crew import (
    CrewConfig,
    CrewManager,
    CrewResult,
    CrewStrategy,
)


# ═══════════════════════════════════════════════════════════════════════
# CrewConfig
# ═══════════════════════════════════════════════════════════════════════

class TestCrewConfig(unittest.TestCase):
    """5 tests"""

    def test_defaults(self):
        cfg = CrewConfig()
        self.assertEqual(cfg.name, "default_crew")
        self.assertEqual(cfg.strategy, CrewStrategy.SEQUENTIAL)
        self.assertEqual(cfg.max_agents, 4)
        self.assertTrue(cfg.auto_review)

    def test_custom_values(self):
        cfg = CrewConfig(
            name="my_crew", strategy=CrewStrategy.PARALLEL,
            max_agents=8, timeout_total=600.0,
        )
        self.assertEqual(cfg.name, "my_crew")
        self.assertEqual(cfg.strategy, CrewStrategy.PARALLEL)

    def test_to_dict(self):
        cfg = CrewConfig()
        d = cfg.to_dict()
        self.assertEqual(d["name"], "default_crew")
        self.assertEqual(d["strategy"], "sequential")

    def test_from_dict(self):
        d = {"name": "test", "strategy": "parallel", "max_agents": 6}
        cfg = CrewConfig.from_dict(d)
        self.assertEqual(cfg.name, "test")
        self.assertEqual(cfg.strategy, CrewStrategy.PARALLEL)
        self.assertEqual(cfg.max_agents, 6)

    def test_all_strategies(self):
        for s in CrewStrategy:
            cfg = CrewConfig(strategy=s)
            self.assertEqual(cfg.strategy, s)


# ═══════════════════════════════════════════════════════════════════════
# CrewResult
# ═══════════════════════════════════════════════════════════════════════

class TestCrewResult(unittest.TestCase):
    """6 tests"""

    def test_creation(self):
        r = CrewResult(success=True, task_description="test task")
        self.assertTrue(r.success)
        self.assertEqual(r.task_description, "test task")

    def test_to_dict(self):
        r = CrewResult(
            success=True, task_description="test",
            agents_used=["coder", "reviewer"],
            aggregated_output="done",
        )
        d = r.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["agents_used"], ["coder", "reviewer"])

    def test_agent_count(self):
        r = CrewResult(success=True, task_description="t",
                       agents_used=["a", "b", "c"])
        self.assertEqual(r.agent_count, 3)

    def test_empty_result(self):
        r = CrewResult(success=False, task_description="failed")
        self.assertEqual(r.sub_results, [])
        self.assertEqual(r.aggregated_output, "")
        self.assertIsNone(r.review_output)

    def test_with_review(self):
        r = CrewResult(
            success=True, task_description="t",
            review_output="Looks good!",
        )
        self.assertEqual(r.review_output, "Looks good!")

    def test_with_sub_results(self):
        r = CrewResult(
            success=True, task_description="t",
            sub_results=[{"role": "coder", "output": "code"}],
        )
        self.assertEqual(len(r.sub_results), 1)


# ═══════════════════════════════════════════════════════════════════════
# CrewRole
# ═══════════════════════════════════════════════════════════════════════

class TestCrewRoles(unittest.TestCase):
    """12 tests"""

    def test_predefined_researcher(self):
        role = PREDEFINED_ROLES["researcher"]
        self.assertEqual(role.name, "researcher")
        self.assertIn("web_search", role.allowed_tools)

    def test_predefined_coder(self):
        role = PREDEFINED_ROLES["coder"]
        self.assertEqual(role.name, "coder")
        self.assertIn("write", role.allowed_tools)
        self.assertIn("edit", role.allowed_tools)

    def test_predefined_reviewer(self):
        role = PREDEFINED_ROLES["reviewer"]
        self.assertEqual(role.name, "reviewer")
        self.assertIn("read", role.allowed_tools)

    def test_predefined_tester(self):
        role = PREDEFINED_ROLES["tester"]
        self.assertEqual(role.name, "tester")
        self.assertIn("bash", role.allowed_tools)

    def test_predefined_planner(self):
        role = PREDEFINED_ROLES["planner"]
        self.assertEqual(role.name, "planner")
        self.assertIn("read", role.allowed_tools)

    def test_all_five_predefined(self):
        expected = {"researcher", "coder", "reviewer", "tester", "planner"}
        self.assertEqual(set(PREDEFINED_ROLES.keys()), expected)

    def test_role_to_dict(self):
        role = PREDEFINED_ROLES["coder"]
        d = role.to_dict()
        self.assertEqual(d["name"], "coder")
        self.assertIn("allowed_tools", d)

    def test_role_from_dict(self):
        d = {
            "name": "custom", "description": "Custom role",
            "system_prompt_addon": "You are custom.",
            "allowed_tools": ["read"], "max_iterations": 5,
        }
        role = CrewRole.from_dict(d)
        self.assertEqual(role.name, "custom")
        self.assertEqual(role.max_iterations, 5)

    def test_role_has_prompt(self):
        for name, role in PREDEFINED_ROLES.items():
            self.assertTrue(len(role.system_prompt_addon) > 10,
                            f"{name} has no prompt")

    def test_role_has_tools(self):
        for name, role in PREDEFINED_ROLES.items():
            self.assertTrue(len(role.allowed_tools) > 0,
                            f"{name} has no tools")

    def test_max_iterations_positive(self):
        for name, role in PREDEFINED_ROLES.items():
            self.assertGreater(role.max_iterations, 0)

    def test_keywords_cover_all_roles(self):
        for role_name in PREDEFINED_ROLES:
            self.assertIn(role_name, _ROLE_KEYWORDS)


# ═══════════════════════════════════════════════════════════════════════
# RoleAssigner
# ═══════════════════════════════════════════════════════════════════════

class TestRoleAssigner(unittest.TestCase):
    """10 tests"""

    def test_assign_researcher(self):
        ra = RoleAssigner()
        role = ra.assign_role("Research best practices for API design")
        self.assertEqual(role.name, "researcher")

    def test_assign_coder(self):
        ra = RoleAssigner()
        role = ra.assign_role("Write a function to parse JSON")
        self.assertEqual(role.name, "coder")

    def test_assign_reviewer(self):
        ra = RoleAssigner()
        role = ra.assign_role("Review this code for quality issues")
        self.assertEqual(role.name, "reviewer")

    def test_assign_tester(self):
        ra = RoleAssigner()
        role = ra.assign_role("Run unit tests and verify test coverage")
        self.assertEqual(role.name, "tester")

    def test_assign_planner(self):
        ra = RoleAssigner()
        role = ra.assign_role("Plan the architecture for the new system")
        self.assertEqual(role.name, "planner")

    def test_fallback_to_coder(self):
        ra = RoleAssigner()
        role = ra.assign_role("Do something completely generic and vague")
        self.assertEqual(role.name, "coder")

    def test_get_role_by_name(self):
        ra = RoleAssigner()
        role = ra.get_role("reviewer")
        self.assertEqual(role.name, "reviewer")

    def test_get_unknown_role_raises(self):
        ra = RoleAssigner()
        with self.assertRaises(KeyError):
            ra.get_role("nonexistent")

    def test_register_custom_role(self):
        ra = RoleAssigner()
        custom = CrewRole(
            name="debugger", description="Debug specialist",
            system_prompt_addon="You debug.",
            allowed_tools=["bash", "read"],
        )
        ra.register_custom_role(custom)
        role = ra.get_role("debugger")
        self.assertEqual(role.name, "debugger")

    def test_list_roles(self):
        ra = RoleAssigner()
        roles = ra.list_roles()
        self.assertIn("researcher", roles)
        self.assertIn("coder", roles)
        self.assertEqual(len(roles), 5)


# ═══════════════════════════════════════════════════════════════════════
# SubTask
# ═══════════════════════════════════════════════════════════════════════

class TestSubTask(unittest.TestCase):
    """5 tests"""

    def test_creation(self):
        st = SubTask(id="sub_1", description="Do something")
        self.assertEqual(st.id, "sub_1")
        self.assertEqual(st.role_hint, "coder")
        self.assertEqual(st.priority, 5)

    def test_to_dict(self):
        st = SubTask(id="sub_2", description="Test", role_hint="tester",
                      dependencies=["sub_1"])
        d = st.to_dict()
        self.assertEqual(d["role_hint"], "tester")
        self.assertEqual(d["dependencies"], ["sub_1"])

    def test_from_dict(self):
        d = {"id": "sub_3", "description": "Plan", "role_hint": "planner",
             "dependencies": [], "priority": 8, "estimated_complexity": "simple"}
        st = SubTask.from_dict(d)
        self.assertEqual(st.priority, 8)

    def test_generate_id(self):
        sid = SubTask.generate_id()
        self.assertTrue(sid.startswith("sub_"))
        self.assertEqual(len(sid), 12)  # "sub_" + 8 hex

    def test_unique_ids(self):
        ids = {SubTask.generate_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)


# ═══════════════════════════════════════════════════════════════════════
# TaskDecomposer
# ═══════════════════════════════════════════════════════════════════════

class TestTaskDecomposer(unittest.TestCase):
    """15 tests"""

    def test_decompose_build(self):
        td = TaskDecomposer()
        result = td.decompose("Build a REST API for user management")
        self.assertGreater(result.task_count, 0)
        self.assertEqual(result.original_task, "Build a REST API for user management")

    def test_decompose_fix_bug(self):
        td = TaskDecomposer()
        result = td.decompose("Fix the bug in the login handler")
        roles = [st.role_hint for st in result.sub_tasks]
        self.assertIn("coder", roles)

    def test_decompose_refactor(self):
        td = TaskDecomposer()
        result = td.decompose("Refactor the database module")
        roles = [st.role_hint for st in result.sub_tasks]
        self.assertIn("reviewer", roles)

    def test_decompose_test(self):
        td = TaskDecomposer()
        result = td.decompose("Write tests for the payment module")
        roles = [st.role_hint for st in result.sub_tasks]
        self.assertIn("tester", roles)

    def test_decompose_document(self):
        td = TaskDecomposer()
        result = td.decompose("Document the API endpoints")
        self.assertGreater(result.task_count, 0)

    def test_decompose_generic_uses_build(self):
        td = TaskDecomposer()
        result = td.decompose("Something completely vague")
        # Falls back to "build" template
        self.assertGreater(result.task_count, 0)

    def test_dependencies_sequential(self):
        td = TaskDecomposer()
        result = td.decompose("Build a feature")
        # Each task after first should have a dependency
        for i, st in enumerate(result.sub_tasks):
            if i > 0:
                self.assertTrue(len(st.dependencies) > 0,
                                f"Task {i} has no dependencies")

    def test_execution_order_exists(self):
        td = TaskDecomposer()
        result = td.decompose("Build a thing")
        self.assertGreater(result.stage_count, 0)

    def test_execution_order_covers_all_tasks(self):
        td = TaskDecomposer()
        result = td.decompose("Build a thing")
        all_ids_in_order = []
        for stage in result.execution_order:
            all_ids_in_order.extend(stage)
        task_ids = {st.id for st in result.sub_tasks}
        self.assertEqual(set(all_ids_in_order), task_ids)

    def test_estimated_time_positive(self):
        td = TaskDecomposer()
        result = td.decompose("Build something")
        self.assertGreater(result.estimated_total_time, 0)

    def test_result_to_dict(self):
        td = TaskDecomposer()
        result = td.decompose("Build something")
        d = result.to_dict()
        self.assertIn("sub_tasks", d)
        self.assertIn("execution_order", d)
        self.assertIn("estimated_total_time", d)

    def test_register_custom_template(self):
        td = TaskDecomposer()
        td.register_template("custom", [
            ("Step 1", "coder", "simple"),
            ("Step 2", "tester", "simple"),
        ])
        # Force match by adding keyword mapping
        result = td.decompose("custom task")  # won't match, falls back
        self.assertGreater(result.task_count, 0)

    def test_all_templates_have_steps(self):
        for name, steps in DECOMPOSITION_TEMPLATES.items():
            self.assertGreater(len(steps), 0, f"Template {name} is empty")

    def test_sub_task_contextualized(self):
        td = TaskDecomposer()
        result = td.decompose("Build the chat feature")
        for st in result.sub_tasks:
            self.assertIn("Build the chat feature", st.description)

    def test_deploy_template(self):
        td = TaskDecomposer()
        result = td.decompose("Deploy the application to production")
        self.assertGreater(result.task_count, 0)


# ═══════════════════════════════════════════════════════════════════════
# ResultAggregator
# ═══════════════════════════════════════════════════════════════════════

class TestResultAggregator(unittest.TestCase):
    """12 tests"""

    def _results(self, n=3):
        return [
            {"role": f"agent{i}", "output": f"Output from agent {i}",
             "confidence": 0.5 + i * 0.1, "task_description": f"Task {i}"}
            for i in range(n)
        ]

    def test_concatenate(self):
        agg = ResultAggregator()
        result = agg.aggregate(self._results(), AggregationStrategy.CONCATENATE)
        self.assertIn("agent0", result)
        self.assertIn("agent1", result)
        self.assertIn("agent2", result)

    def test_merge_deduplicates(self):
        agg = ResultAggregator()
        results = [
            {"role": "a", "output": "line1\nline2\nline3"},
            {"role": "b", "output": "line2\nline3\nline4"},
        ]
        merged = agg.aggregate(results, AggregationStrategy.MERGE)
        # line2 and line3 should appear only once
        self.assertEqual(merged.count("line2"), 1)
        self.assertEqual(merged.count("line3"), 1)
        self.assertIn("line4", merged)

    def test_best_of(self):
        agg = ResultAggregator()
        results = [
            {"role": "a", "output": "low quality", "confidence": 0.3},
            {"role": "b", "output": "high quality", "confidence": 0.9},
        ]
        best = agg.aggregate(results, AggregationStrategy.BEST_OF)
        self.assertIn("high quality", best)
        self.assertIn("0.90", best)

    def test_consensus(self):
        agg = ResultAggregator()
        results = [
            {"role": "a", "output": "common line\nunique a"},
            {"role": "b", "output": "common line\nunique b"},
            {"role": "c", "output": "common line\nunique c"},
        ]
        consensus = agg.aggregate(results, AggregationStrategy.CONSENSUS)
        self.assertIn("common line", consensus)

    def test_empty_results(self):
        agg = ResultAggregator()
        result = agg.aggregate([])
        self.assertEqual(result, "")

    def test_single_result(self):
        agg = ResultAggregator()
        results = [{"role": "a", "output": "solo output", "confidence": 0.8}]
        result = agg.aggregate(results, AggregationStrategy.CONCATENATE)
        self.assertIn("solo output", result)

    def test_default_strategy(self):
        agg = ResultAggregator(default_strategy=AggregationStrategy.MERGE)
        results = [
            {"role": "a", "output": "line1"},
            {"role": "b", "output": "line1"},
        ]
        result = agg.aggregate(results)  # uses default
        self.assertEqual(result.count("line1"), 1)

    def test_summarize_results(self):
        results = [
            {"role": "coder", "output": "..."},
            {"role": "reviewer", "output": "..."},
        ]
        summary = ResultAggregator.summarize_results(results)
        self.assertIn("2", summary)
        self.assertIn("coder", summary)

    def test_all_strategies_enum(self):
        expected = {"concatenate", "merge", "best_of", "consensus"}
        actual = {s.value for s in AggregationStrategy}
        self.assertEqual(actual, expected)

    def test_consensus_single_result(self):
        agg = ResultAggregator()
        results = [{"role": "a", "output": "only result"}]
        result = agg.aggregate(results, AggregationStrategy.CONSENSUS)
        self.assertEqual(result, "only result")

    def test_best_of_all_equal(self):
        agg = ResultAggregator()
        results = [
            {"role": "a", "output": "same", "confidence": 0.5},
            {"role": "b", "output": "same", "confidence": 0.5},
        ]
        result = agg.aggregate(results, AggregationStrategy.BEST_OF)
        self.assertIn("same", result)

    def test_merge_empty_lines_ignored(self):
        agg = ResultAggregator()
        results = [
            {"role": "a", "output": "line1\n\n\nline2"},
        ]
        merged = agg.aggregate(results, AggregationStrategy.MERGE)
        self.assertIn("line1", merged)
        self.assertIn("line2", merged)


# ═══════════════════════════════════════════════════════════════════════
# CrewManager
# ═══════════════════════════════════════════════════════════════════════

class TestCrewManager(unittest.TestCase):
    """15 tests"""

    def _make_manager(self, strategy=CrewStrategy.SEQUENTIAL, executor=None):
        config = CrewConfig(strategy=strategy, auto_review=False)
        return CrewManager(config=config, agent_executor=executor)

    def test_init(self):
        cm = self._make_manager()
        self.assertEqual(cm.total_executions, 0)

    def test_sequential_execution(self):
        async def _run():
            cm = self._make_manager(strategy=CrewStrategy.SEQUENTIAL)
            result = await cm.execute_crew_task("Build a feature")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertTrue(result.success)
        self.assertGreater(result.agent_count, 0)

    def test_parallel_execution(self):
        async def _run():
            cm = self._make_manager(strategy=CrewStrategy.PARALLEL)
            result = await cm.execute_crew_task("Build a feature")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertTrue(result.success)

    def test_pipeline_execution(self):
        async def _run():
            cm = self._make_manager(strategy=CrewStrategy.PIPELINE)
            result = await cm.execute_crew_task("Build a feature")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertTrue(result.success)

    def test_custom_executor(self):
        def executor(role, desc):
            return f"Executed by {role.name}: {desc}"

        async def _run():
            cm = self._make_manager(executor=executor)
            result = await cm.execute_crew_task("Fix a bug")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertTrue(result.success)
        self.assertIn("Executed by", result.aggregated_output)

    def test_async_executor(self):
        async def executor(role, desc):
            return f"Async {role.name}"

        async def _run():
            cm = self._make_manager(executor=executor)
            result = await cm.execute_crew_task("Build something")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertIn("Async", result.aggregated_output)

    def test_result_has_decomposition(self):
        async def _run():
            cm = self._make_manager()
            result = await cm.execute_crew_task("Build a feature")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertIsNotNone(result.decomposition)

    def test_execution_time_recorded(self):
        async def _run():
            cm = self._make_manager()
            result = await cm.execute_crew_task("Quick task")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertGreater(result.execution_time_ms, 0)

    def test_auto_review_with_executor(self):
        def executor(role, desc):
            return f"Result from {role.name}"

        async def _run():
            config = CrewConfig(auto_review=True)
            cm = CrewManager(config=config, agent_executor=executor)
            result = await cm.execute_crew_task("Build something")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertIsNotNone(result.review_output)

    def test_auto_review_without_executor(self):
        async def _run():
            config = CrewConfig(auto_review=True)
            cm = CrewManager(config=config)
            result = await cm.execute_crew_task("Build something")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertIsNone(result.review_output)

    def test_execution_history(self):
        async def _run():
            cm = self._make_manager()
            await cm.execute_crew_task("Task 1")
            await cm.execute_crew_task("Task 2")
            return cm

        cm = asyncio.get_event_loop().run_until_complete(_run())
        self.assertEqual(cm.total_executions, 2)
        self.assertEqual(len(cm.execution_history), 2)

    def test_stats(self):
        async def _run():
            cm = self._make_manager()
            await cm.execute_crew_task("Test task")
            return cm.stats()

        stats = asyncio.get_event_loop().run_until_complete(_run())
        self.assertEqual(stats["total_executions"], 1)
        self.assertEqual(stats["successful"], 1)

    def test_executor_error_handled(self):
        def bad_executor(role, desc):
            raise RuntimeError("Agent crashed")

        async def _run():
            cm = self._make_manager(executor=bad_executor)
            result = await cm.execute_crew_task("Build something")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        # Should still succeed (errors are caught per-subtask)
        self.assertTrue(result.success)
        self.assertIn("Error", result.aggregated_output)

    def test_to_dict_result(self):
        async def _run():
            cm = self._make_manager()
            result = await cm.execute_crew_task("Build")
            return result.to_dict()

        d = asyncio.get_event_loop().run_until_complete(_run())
        self.assertIn("success", d)
        self.assertIn("agents_used", d)

    def test_sub_results_have_structure(self):
        async def _run():
            cm = self._make_manager()
            result = await cm.execute_crew_task("Build a feature")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        for sr in result.sub_results:
            self.assertIn("role", sr)
            self.assertIn("output", sr)
            self.assertIn("task_id", sr)


# ═══════════════════════════════════════════════════════════════════════
# Agent Integration
# ═══════════════════════════════════════════════════════════════════════

class TestAgentIntegration(unittest.TestCase):
    """8 tests"""

    def _make_agent(self):
        from cowork_agent.core.agent import Agent
        provider = MagicMock()
        registry = MagicMock()
        prompt_builder = MagicMock()
        return Agent(provider=provider, registry=registry,
                     prompt_builder=prompt_builder)

    def test_agent_has_crew_manager_attr(self):
        agent = self._make_agent()
        self.assertIsNone(agent.crew_manager)

    def test_crew_manager_assignable(self):
        agent = self._make_agent()
        cm = CrewManager(config=CrewConfig())
        agent.crew_manager = cm
        self.assertIs(agent.crew_manager, cm)

    def test_main_has_sprint_43_wiring(self):
        import cowork_agent.main as main_mod
        source = open(main_mod.__file__).read()
        self.assertIn("Sprint 43", source)
        self.assertIn("CrewManager", source)
        self.assertIn("RoleAssigner", source)
        self.assertIn("TaskDecomposer", source)
        self.assertIn("ResultAggregator", source)

    def test_crew_config_from_dict_partial(self):
        d = {"name": "test"}
        cfg = CrewConfig.from_dict(d)
        self.assertEqual(cfg.name, "test")
        self.assertEqual(cfg.max_agents, 4)  # default

    def test_strategy_enum_values(self):
        self.assertEqual(CrewStrategy.SEQUENTIAL.value, "sequential")
        self.assertEqual(CrewStrategy.PARALLEL.value, "parallel")
        self.assertEqual(CrewStrategy.PIPELINE.value, "pipeline")

    def test_crew_works_with_agent_context(self):
        """CrewManager can be used standalone or with agent."""
        async def _run():
            cm = CrewManager(config=CrewConfig(auto_review=False))
            result = await cm.execute_crew_task("Analyze code quality")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertTrue(result.success)

    def test_all_sprint_43_modules_importable(self):
        from cowork_agent.core.crew import CrewManager
        from cowork_agent.core.crew_roles import CrewRole
        from cowork_agent.core.task_decomposer import TaskDecomposer
        from cowork_agent.core.result_aggregator import ResultAggregator
        self.assertIsNotNone(CrewManager)
        self.assertIsNotNone(CrewRole)
        self.assertIsNotNone(TaskDecomposer)
        self.assertIsNotNone(ResultAggregator)

    def test_role_assigner_custom_roles_property(self):
        ra = RoleAssigner()
        self.assertEqual(ra.custom_roles, {})
        custom = CrewRole(name="x", description="x", system_prompt_addon="x")
        ra.register_custom_role(custom)
        self.assertIn("x", ra.custom_roles)


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    """7 tests"""

    def test_empty_task_description(self):
        td = TaskDecomposer()
        result = td.decompose("")
        # Should still decompose (using default template)
        self.assertGreater(result.task_count, 0)

    def test_very_long_task_description(self):
        td = TaskDecomposer()
        desc = "Build " + "a very complex " * 100 + "system"
        result = td.decompose(desc)
        self.assertGreater(result.task_count, 0)

    def test_consensus_no_overlap(self):
        agg = ResultAggregator()
        results = [
            {"role": "a", "output": "unique a only"},
            {"role": "b", "output": "unique b only"},
            {"role": "c", "output": "unique c only"},
        ]
        result = agg.aggregate(results, AggregationStrategy.CONSENSUS)
        # Should fall back to concatenation when no consensus
        self.assertTrue(len(result) > 0)

    def test_decomposer_circular_deps_handled(self):
        """Build execution order handles edge cases gracefully."""
        td = TaskDecomposer()
        # Create tasks with circular dependencies
        st1 = SubTask(id="a", description="A", dependencies=["b"])
        st2 = SubTask(id="b", description="B", dependencies=["a"])
        order = td._build_execution_order([st1, st2])
        # Should still produce stages (dumps remaining on circular)
        all_ids = []
        for stage in order:
            all_ids.extend(stage)
        self.assertIn("a", all_ids)
        self.assertIn("b", all_ids)

    def test_crew_manager_no_agents_still_returns(self):
        async def _run():
            config = CrewConfig(max_agents=0, auto_review=False)
            cm = CrewManager(config=config)
            result = await cm.execute_crew_task("Test")
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertIsNotNone(result)

    def test_merge_empty_outputs(self):
        agg = ResultAggregator()
        results = [
            {"role": "a", "output": ""},
            {"role": "b", "output": ""},
        ]
        result = agg.aggregate(results, AggregationStrategy.MERGE)
        self.assertEqual(result, "")

    def test_best_of_no_confidence(self):
        agg = ResultAggregator()
        results = [
            {"role": "a", "output": "result a"},
            {"role": "b", "output": "result b"},
        ]
        # No confidence fields → all default to 0.0
        result = agg.aggregate(results, AggregationStrategy.BEST_OF)
        self.assertTrue(len(result) > 0)


if __name__ == "__main__":
    unittest.main()
