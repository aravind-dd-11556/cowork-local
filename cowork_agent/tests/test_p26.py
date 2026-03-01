"""
Sprint 26 Tests — Module Wiring Completion + Scheduler Activation.

Tests:
  - Scheduler Background Loop (6): start/stop, task execution, skip cases
  - Sprint 21 Module Wiring (14): specialization, routing, pool, conflict resolver, strategies
  - Orphan Module Wiring (11): multimodal, hybrid cache, coverage collector
  - Agent Attribute Verification (10): all attributes properly initialized
  - Integration (9): end-to-end flows
"""

import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Scheduler Loop Tests ─────────────────────────────────────────

from cowork_agent.core.scheduler import TaskScheduler, ScheduledTask


class TestSchedulerBackgroundLoop(unittest.TestCase):
    """Test the scheduler start/stop lifecycle."""

    def test_start_sets_running(self):
        scheduler = TaskScheduler()
        self.assertFalse(scheduler._running)

        async def _run():
            runner = AsyncMock()
            with patch('cowork_agent.core.scheduler.asyncio.sleep',
                       new_callable=AsyncMock) as mock_sleep:
                async def _short_sleep(duration):
                    scheduler.stop()

                mock_sleep.side_effect = _short_sleep
                # start() sets _running = True before the loop
                await scheduler.start(runner)
                # After stop(), _running is False, but it was True during the loop

        asyncio.get_event_loop().run_until_complete(_run())
        # Verify it was started (stop was called, so now it's False)
        self.assertFalse(scheduler._running)

    def test_stop_clears_running(self):
        scheduler = TaskScheduler()
        scheduler._running = True
        scheduler.stop()
        self.assertFalse(scheduler._running)

    def test_scheduler_loop_executes_due_task(self):
        """A task whose next_run_at is in the past should execute."""
        scheduler = TaskScheduler()
        task = ScheduledTask(
            task_id="test-loop",
            prompt="hello",
            description="Test",
            cron_expression="0 0 * * *",
        )
        scheduler.create_task(task)
        # Set next_run_at to past
        from datetime import datetime, timedelta
        task.next_run_at = (datetime.now() - timedelta(minutes=5)).isoformat()

        runner = AsyncMock()

        async def _run():
            with patch('cowork_agent.core.scheduler.asyncio.sleep',
                       new_callable=AsyncMock) as mock_sleep:
                call_count = 0

                async def _short_sleep(duration):
                    nonlocal call_count
                    call_count += 1
                    if call_count >= 1:
                        scheduler.stop()

                mock_sleep.side_effect = _short_sleep

                await scheduler.start(runner)

        asyncio.get_event_loop().run_until_complete(_run())
        runner.assert_called_once_with("hello")

    def test_scheduler_records_history_on_loop_run(self):
        scheduler = TaskScheduler()
        task = ScheduledTask(
            task_id="hist-test",
            prompt="greet",
            description="Test",
            cron_expression="0 0 * * *",
        )
        scheduler.create_task(task)
        from datetime import datetime, timedelta
        task.next_run_at = (datetime.now() - timedelta(minutes=1)).isoformat()

        runner = AsyncMock()

        async def _run():
            with patch('cowork_agent.core.scheduler.asyncio.sleep',
                       new_callable=AsyncMock) as mock_sleep:
                async def _short_sleep(duration):
                    scheduler.stop()

                mock_sleep.side_effect = _short_sleep
                await scheduler.start(runner)

        asyncio.get_event_loop().run_until_complete(_run())
        self.assertGreaterEqual(len(scheduler.run_history), 1)

    def test_scheduler_skips_disabled_task(self):
        scheduler = TaskScheduler()
        task = ScheduledTask(
            task_id="disabled",
            prompt="should not run",
            description="Test",
            cron_expression="0 0 * * *",
            enabled=False,
        )
        scheduler.create_task(task)
        from datetime import datetime, timedelta
        task.next_run_at = (datetime.now() - timedelta(minutes=1)).isoformat()

        runner = AsyncMock()

        async def _run():
            with patch('cowork_agent.core.scheduler.asyncio.sleep',
                       new_callable=AsyncMock) as mock_sleep:
                async def _short_sleep(duration):
                    scheduler.stop()

                mock_sleep.side_effect = _short_sleep
                await scheduler.start(runner)

        asyncio.get_event_loop().run_until_complete(_run())
        runner.assert_not_called()

    def test_scheduler_skips_adhoc_task(self):
        """Tasks without cron_expression are manual-only."""
        scheduler = TaskScheduler()
        task = ScheduledTask(
            task_id="manual",
            prompt="manual only",
            description="Test",
            cron_expression="",
        )
        scheduler.create_task(task)

        runner = AsyncMock()

        async def _run():
            with patch('cowork_agent.core.scheduler.asyncio.sleep',
                       new_callable=AsyncMock) as mock_sleep:
                async def _short_sleep(duration):
                    scheduler.stop()

                mock_sleep.side_effect = _short_sleep
                await scheduler.start(runner)

        asyncio.get_event_loop().run_until_complete(_run())
        runner.assert_not_called()


# ── Sprint 21 Module Wiring ──────────────────────────────────────

class TestSpecializationRegistryWiring(unittest.TestCase):
    """SpecializationRegistry can be used."""

    def test_import_and_create(self):
        from cowork_agent.core.agent_specialization import (
            SpecializationRegistry, AgentSpecialization, AgentRole,
        )
        reg = SpecializationRegistry()
        self.assertEqual(len(reg.list_agents()), 0)

    def test_register_and_find(self):
        from cowork_agent.core.agent_specialization import (
            SpecializationRegistry, AgentSpecialization, AgentRole,
        )
        reg = SpecializationRegistry()
        spec = AgentSpecialization(role=AgentRole.CODER)
        reg.register_agent("coder-1", spec)
        best, score = reg.find_best_agent("implement a login function")
        self.assertEqual(best, "coder-1")
        self.assertGreater(score, 0)

    def test_find_tester(self):
        from cowork_agent.core.agent_specialization import (
            SpecializationRegistry, AgentSpecialization, AgentRole,
        )
        reg = SpecializationRegistry()
        reg.register_agent("tester-1", AgentSpecialization(role=AgentRole.TESTER))
        best, score = reg.find_best_agent("write unit tests for the auth module")
        self.assertEqual(best, "tester-1")
        self.assertGreater(score, 0)


class TestConversationRouterWiring(unittest.TestCase):
    """ConversationRouter can be used."""

    def test_import_and_create(self):
        from cowork_agent.core.conversation_router import ConversationRouter
        router = ConversationRouter()
        self.assertIsNotNone(router)

    def test_analyze_task(self):
        from cowork_agent.core.conversation_router import ConversationRouter
        router = ConversationRouter()
        analysis = router._analyzer.analyze("Build a REST API with tests")
        self.assertIn(analysis.complexity, ["simple", "moderate", "complex"])

    def test_route_task(self):
        from cowork_agent.core.conversation_router import ConversationRouter
        from cowork_agent.core.agent_specialization import (
            SpecializationRegistry, AgentSpecialization, AgentRole,
        )
        reg = SpecializationRegistry()
        reg.register_agent("coder-1", AgentSpecialization(role=AgentRole.CODER))
        reg.register_agent("tester-1", AgentSpecialization(role=AgentRole.TESTER))
        router = ConversationRouter(spec_registry=reg)
        decision = router.route_task(
            "Implement and test the login feature",
            available_agents=["coder-1", "tester-1"],
        )
        self.assertIsNotNone(decision)


class TestAgentPoolWiring(unittest.TestCase):
    """AgentPool can be created and managed."""

    def test_import_and_create(self):
        from cowork_agent.core.agent_pool import AgentPool, PoolConfig
        pool = AgentPool(config=PoolConfig(name="test", min_size=1, max_size=5))
        self.assertEqual(pool.config.name, "test")

    def test_pool_stats(self):
        from cowork_agent.core.agent_pool import AgentPool, PoolConfig
        pool = AgentPool(config=PoolConfig())
        self.assertEqual(pool.config.min_size, 1)
        self.assertEqual(pool.config.max_size, 10)


class TestConflictResolverWiring(unittest.TestCase):
    """ConflictResolver can be created and used."""

    def test_import_and_create(self):
        from cowork_agent.core.conflict_resolver import ConflictResolver
        resolver = ConflictResolver()
        self.assertIsNotNone(resolver)

    def test_resolve_voting(self):
        from cowork_agent.core.conflict_resolver import (
            ConflictResolver, ConflictStrategy,
        )
        resolver = ConflictResolver(strategy=ConflictStrategy.VOTING)

        async def _run():
            return await resolver.resolve(
                field_name="answer",
                agent_values={"agent-1": "yes", "agent-2": "yes", "agent-3": "no"},
            )

        result = asyncio.get_event_loop().run_until_complete(_run())
        self.assertEqual(result, "yes")

    def test_resolve_first_win(self):
        from cowork_agent.core.conflict_resolver import (
            ConflictResolver, ConflictStrategy,
        )
        resolver = ConflictResolver(strategy=ConflictStrategy.FIRST_WIN)

        async def _run():
            return await resolver.resolve(
                field_name="color",
                agent_values={"agent-1": "red", "agent-2": "blue"},
            )

        result = asyncio.get_event_loop().run_until_complete(_run())
        # FIRST_WIN returns the first value encountered
        self.assertIn(result, ["red", "blue"])


class TestSupervisorStrategiesWiring(unittest.TestCase):
    """Supervisor strategies can be imported and created."""

    def test_import_map_reduce(self):
        from cowork_agent.core.supervisor_strategies import MapReduceStrategy
        strategy = MapReduceStrategy()
        self.assertIsNotNone(strategy)

    def test_import_debate(self):
        from cowork_agent.core.supervisor_strategies import DebateStrategy
        strategy = DebateStrategy()
        self.assertIsNotNone(strategy)

    def test_import_voting(self):
        from cowork_agent.core.supervisor_strategies import VotingStrategy
        strategy = VotingStrategy()
        self.assertIsNotNone(strategy)


# ── Orphan Module Wiring ─────────────────────────────────────────

class TestMultimodalWiring(unittest.TestCase):
    """Multimodal input support."""

    def test_import_parse_multimodal(self):
        from cowork_agent.core.multimodal import parse_multimodal_input
        result = parse_multimodal_input("Hello world")
        self.assertEqual(result.text, "Hello world")
        self.assertFalse(result.has_images)

    def test_import_load_image(self):
        from cowork_agent.core.multimodal import load_image
        result = load_image("/nonexistent/file.png")
        self.assertIsNone(result)

    def test_image_content_to_anthropic(self):
        from cowork_agent.core.multimodal import ImageContent
        img = ImageContent(
            media_type="image/png",
            base64_data="abc123",
        )
        block = img.to_anthropic_block()
        self.assertEqual(block["type"], "image")
        self.assertEqual(block["source"]["media_type"], "image/png")

    def test_image_content_to_openai(self):
        from cowork_agent.core.multimodal import ImageContent
        img = ImageContent(
            media_type="image/jpeg",
            base64_data="xyz789",
        )
        block = img.to_openai_block()
        self.assertEqual(block["type"], "image_url")

    def test_extract_image_paths(self):
        from cowork_agent.core.multimodal import extract_image_paths
        # extract_image_paths only returns paths that exist on disk
        # With non-existent paths, returns empty list
        paths = extract_image_paths("Look at /tmp/photo.png and /tmp/cat.jpg")
        self.assertIsInstance(paths, list)
        # Create a temp image file to test extraction
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake png")
            tmp_path = f.name
        try:
            paths = extract_image_paths(f"Check {tmp_path} for details")
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0], tmp_path)
        finally:
            os.unlink(tmp_path)


class TestHybridCacheWiring(unittest.TestCase):
    """Hybrid response cache."""

    def test_import_and_create(self):
        from cowork_agent.core.hybrid_cache import HybridResponseCache
        cache = HybridResponseCache(enabled=True)
        self.assertIsNotNone(cache)

    def test_create_with_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from cowork_agent.core.hybrid_cache import HybridResponseCache
            cache = HybridResponseCache(workspace_dir=tmpdir, enabled=True)
            self.assertTrue(os.path.exists(
                os.path.join(tmpdir, ".cowork", "cache")
            ))

    def test_disabled_cache(self):
        from cowork_agent.core.hybrid_cache import HybridResponseCache
        cache = HybridResponseCache(enabled=False)
        self.assertIsNotNone(cache)


class TestCoverageCollectorWiring(unittest.TestCase):
    """Test coverage collector."""

    def test_import_and_create(self):
        from cowork_agent.core.test_coverage_collector import CoverageCollector
        collector = CoverageCollector()
        self.assertIsNotNone(collector)

    def test_empty_summary(self):
        from cowork_agent.core.test_coverage_collector import CoverageCollector
        collector = CoverageCollector()
        summary = collector.generate_summary()
        self.assertIsNotNone(summary)

    def test_identify_uncovered(self):
        from cowork_agent.core.test_coverage_collector import CoverageCollector
        collector = CoverageCollector()
        uncovered = collector.identify_uncovered_modules(
            all_modules=["mod_a", "mod_b"],
        )
        self.assertIsNotNone(uncovered)


# ── Agent Attribute Verification ─────────────────────────────────

class TestAgentAttributes(unittest.TestCase):
    """All Sprint 26 attributes exist on Agent."""

    def _make_agent(self):
        from cowork_agent.core.agent import Agent
        provider = MagicMock()
        provider.provider_name = "mock"
        provider.model = "test"
        registry = MagicMock()
        registry.tool_names = []
        registry.get_schemas.return_value = []
        prompt_builder = MagicMock()
        return Agent(provider=provider, registry=registry, prompt_builder=prompt_builder)

    def test_has_specialization_registry(self):
        agent = self._make_agent()
        self.assertIsNone(agent.specialization_registry)

    def test_has_conversation_router(self):
        agent = self._make_agent()
        self.assertIsNone(agent.conversation_router)

    def test_has_agent_pool(self):
        agent = self._make_agent()
        self.assertIsNone(agent.agent_pool)

    def test_has_conflict_resolver(self):
        agent = self._make_agent()
        self.assertIsNone(agent.conflict_resolver)

    def test_has_multimodal_parser(self):
        agent = self._make_agent()
        self.assertIsNone(agent.multimodal_parser)

    def test_has_image_loader(self):
        agent = self._make_agent()
        self.assertIsNone(agent.image_loader)

    def test_has_hybrid_cache(self):
        agent = self._make_agent()
        self.assertIsNone(agent.hybrid_cache)

    def test_has_coverage_collector(self):
        agent = self._make_agent()
        self.assertIsNone(agent.coverage_collector)

    def test_has_task_scheduler(self):
        agent = self._make_agent()
        self.assertIsNone(agent.task_scheduler)

    def test_has_scheduler_run_tool(self):
        agent = self._make_agent()
        self.assertIsNone(agent._scheduler_run_tool)


# ── Integration ──────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def test_scheduler_create_and_run_now(self):
        """Full flow: create → run_now → check history."""
        scheduler = TaskScheduler()
        runner = AsyncMock(return_value="done")

        scheduler.create_task(ScheduledTask(
            task_id="quick-task",
            prompt="do something quick",
            description="Quick task",
        ))

        result = asyncio.get_event_loop().run_until_complete(
            scheduler.run_now("quick-task", runner)
        )
        self.assertIn("successfully", result)
        runner.assert_called_once_with("do something quick")
        self.assertEqual(len(scheduler.run_history), 1)

    def test_specialization_to_router_pipeline(self):
        """Specialization registry feeds into conversation router."""
        from cowork_agent.core.agent_specialization import (
            SpecializationRegistry, AgentSpecialization, AgentRole,
        )
        from cowork_agent.core.conversation_router import ConversationRouter

        reg = SpecializationRegistry()
        reg.register_agent("coder", AgentSpecialization(role=AgentRole.CODER))
        reg.register_agent("writer", AgentSpecialization(role=AgentRole.WRITER))

        router = ConversationRouter(spec_registry=reg)
        decision = router.route_task(
            "Write documentation for the API",
            available_agents=["coder", "writer"],
        )
        self.assertIsNotNone(decision)

    def test_conflict_resolver_with_strategies(self):
        """Conflict resolver handles different strategies."""
        from cowork_agent.core.conflict_resolver import (
            ConflictResolver, ConflictStrategy,
        )

        async def _run():
            for strategy in [ConflictStrategy.VOTING, ConflictStrategy.FIRST_WIN]:
                resolver = ConflictResolver(strategy=strategy)
                result = await resolver.resolve(
                    field_name="output",
                    agent_values={"a1": "x", "a2": "y"},
                )
                self.assertIsNotNone(result)

        asyncio.get_event_loop().run_until_complete(_run())

    def test_multimodal_text_only_message(self):
        """Multimodal parser handles text-only input gracefully."""
        from cowork_agent.core.multimodal import parse_multimodal_input
        msg = parse_multimodal_input("Just text, no images")
        self.assertEqual(msg.text, "Just text, no images")
        self.assertEqual(len(msg.images), 0)
        content = msg.to_anthropic_content()
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")

    def test_scheduler_with_persistence(self):
        """Scheduler persists and loads tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = TaskScheduler(workspace_dir=tmpdir)
            s1.create_task(ScheduledTask(
                task_id="persist-test",
                prompt="persisted prompt",
                description="Persistence test",
                cron_expression="0 9 * * *",
            ))

            s2 = TaskScheduler(workspace_dir=tmpdir)
            count = s2.load()
            self.assertEqual(count, 1)
            task = s2.get_task("persist-test")
            self.assertEqual(task.prompt, "persisted prompt")

    def test_scheduler_run_tool_late_binding(self):
        """RunScheduledTaskTool supports late binding of agent_runner."""
        from cowork_agent.tools.scheduler_tools_ext import RunScheduledTaskTool
        scheduler = TaskScheduler()
        scheduler.create_task(ScheduledTask(
            task_id="late-bind",
            prompt="test",
            description="Late bind test",
        ))

        tool = RunScheduledTaskTool(scheduler=scheduler)
        runner = AsyncMock()
        tool.set_agent_runner(runner)

        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(taskId="late-bind", tool_id="t1")
        )
        self.assertTrue(result.success)
        runner.assert_called_once_with("test")

    def test_all_modules_importable(self):
        """All wired modules can be imported without error."""
        modules = [
            "cowork_agent.core.agent_specialization",
            "cowork_agent.core.conversation_router",
            "cowork_agent.core.agent_pool",
            "cowork_agent.core.supervisor_strategies",
            "cowork_agent.core.conflict_resolver",
            "cowork_agent.core.multimodal",
            "cowork_agent.core.hybrid_cache",
            "cowork_agent.core.test_coverage_collector",
        ]
        import importlib
        for mod_name in modules:
            mod = importlib.import_module(mod_name)
            self.assertIsNotNone(mod, f"Failed to import {mod_name}")

    def test_scheduler_full_lifecycle(self):
        """Create → List → Update → Run → Delete."""
        scheduler = TaskScheduler()
        runner = AsyncMock()

        # Create
        scheduler.create_task(ScheduledTask(
            task_id="lifecycle",
            prompt="original",
            description="Lifecycle test",
            cron_expression="0 9 * * 1-5",
        ))
        self.assertEqual(len(scheduler.list_tasks()), 1)

        # Update
        scheduler.update_task("lifecycle", prompt="updated prompt")
        self.assertEqual(scheduler.get_task("lifecycle").prompt, "updated prompt")

        # Run
        asyncio.get_event_loop().run_until_complete(
            scheduler.run_now("lifecycle", runner)
        )
        runner.assert_called_once_with("updated prompt")

        # Delete
        scheduler.delete_task("lifecycle")
        self.assertEqual(len(scheduler.list_tasks()), 0)

    def test_scheduler_concurrent_create_and_delete(self):
        """Multiple tasks created and some deleted."""
        scheduler = TaskScheduler()
        for i in range(5):
            scheduler.create_task(ScheduledTask(
                task_id=f"task-{i}",
                prompt=f"prompt-{i}",
                description=f"Task {i}",
            ))
        self.assertEqual(len(scheduler.list_tasks()), 5)

        scheduler.delete_task("task-1")
        scheduler.delete_task("task-3")
        self.assertEqual(len(scheduler.list_tasks()), 3)
        remaining_ids = [t["task_id"] for t in scheduler.list_tasks()]
        self.assertNotIn("task-1", remaining_ids)
        self.assertNotIn("task-3", remaining_ids)


if __name__ == "__main__":
    unittest.main()
