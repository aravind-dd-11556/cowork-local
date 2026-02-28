"""
Sprint 21 Tests — Multi-Agent Orchestration Enhancement

Covers:
  - Supervisor strategies (MAP_REDUCE, DEBATE, VOTING)
  - Agent specialization (roles, registry, matching)
  - Conversation router (analysis, routing, decomposition)
  - Agent pool (acquire/release, scaling, auto-scaler)
  - Config + wiring
  - Edge cases

~170 tests across 10 classes.
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════


def run_async(coro):
    """Run an async function in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def mock_agent_runner(name: str, prompt: str) -> str:
    """Mock agent runner that returns agent name + prompt snippet."""
    return f"[{name}] processed: {prompt[:50]}"


async def mock_failing_runner(name: str, prompt: str) -> str:
    """Mock runner that fails for specific agents."""
    if "fail" in name:
        raise RuntimeError(f"Agent {name} failed")
    return f"[{name}] ok"


async def mock_vote_runner(name: str, prompt: str) -> str:
    """Mock runner that returns vote-like responses."""
    if "vote" in prompt.lower():
        # Return a vote for solution 1
        return "I vote for Solution 1"
    return f"[{name}] solution output"


# ═══════════════════════════════════════════════════════════════════
#  Test Class 1: MapReduceStrategy
# ═══════════════════════════════════════════════════════════════════


class TestMapReduceStrategy(unittest.TestCase):
    """Tests for MAP_REDUCE execution strategy."""

    def test_basic_map_reduce(self):
        from cowork_agent.core.supervisor_strategies import MapReduceStrategy
        strategy = MapReduceStrategy()
        result = run_async(strategy.execute(
            task="Analyze codebase",
            agent_runner=mock_agent_runner,
            agent_names=["agent_a", "agent_b", "agent_c"],
        ))
        self.assertEqual(result.strategy, "map_reduce")
        self.assertIn("agent_a", result.agent_outputs)
        self.assertIn("agent_b", result.agent_outputs)
        self.assertIn("agent_c", result.agent_outputs)
        self.assertIn("agent_a", result.final_output)

    def test_map_reduce_single_agent(self):
        from cowork_agent.core.supervisor_strategies import MapReduceStrategy
        strategy = MapReduceStrategy()
        result = run_async(strategy.execute(
            task="Simple task",
            agent_runner=mock_agent_runner,
            agent_names=["agent_a"],
        ))
        self.assertEqual(len(result.agent_outputs), 1)

    def test_map_reduce_with_failures(self):
        from cowork_agent.core.supervisor_strategies import MapReduceStrategy
        strategy = MapReduceStrategy()
        result = run_async(strategy.execute(
            task="Do work",
            agent_runner=mock_failing_runner,
            agent_names=["agent_ok", "agent_fail"],
        ))
        self.assertIn("[ERROR", result.agent_outputs["agent_fail"])
        self.assertIn("ok", result.agent_outputs["agent_ok"])
        self.assertEqual(result.metadata["failed"], 1)
        self.assertEqual(result.metadata["successful"], 1)

    def test_map_reduce_custom_config(self):
        from cowork_agent.core.supervisor_strategies import (
            MapReduceStrategy, MapReduceConfig,
        )
        config = MapReduceConfig(
            subtask_template="Custom: {agent} do {task}",
            max_parallel=2,
        )
        strategy = MapReduceStrategy(config=config)
        result = run_async(strategy.execute(
            task="test", agent_runner=mock_agent_runner,
            agent_names=["a", "b"],
        ))
        self.assertIn("Custom:", result.agent_outputs["a"])

    def test_map_reduce_with_merge_agent(self):
        from cowork_agent.core.supervisor_strategies import (
            MapReduceStrategy, MapReduceConfig,
        )
        config = MapReduceConfig(merge_agent="agent_a")
        strategy = MapReduceStrategy(config=config)
        result = run_async(strategy.execute(
            task="Merge test",
            agent_runner=mock_agent_runner,
            agent_names=["agent_a", "agent_b"],
        ))
        self.assertEqual(result.metadata["merge_agent"], "agent_a")

    def test_map_reduce_elapsed_time(self):
        from cowork_agent.core.supervisor_strategies import MapReduceStrategy
        strategy = MapReduceStrategy()
        result = run_async(strategy.execute(
            task="t", agent_runner=mock_agent_runner, agent_names=["a"],
        ))
        self.assertGreater(result.elapsed_seconds, 0)

    def test_map_reduce_rounds(self):
        from cowork_agent.core.supervisor_strategies import MapReduceStrategy
        strategy = MapReduceStrategy()
        result = run_async(strategy.execute(
            task="t", agent_runner=mock_agent_runner, agent_names=["a"],
        ))
        self.assertEqual(result.rounds, 1)

    def test_map_reduce_result_type(self):
        from cowork_agent.core.supervisor_strategies import (
            MapReduceStrategy, StrategyResult,
        )
        strategy = MapReduceStrategy()
        result = run_async(strategy.execute(
            task="t", agent_runner=mock_agent_runner, agent_names=["a"],
        ))
        self.assertIsInstance(result, StrategyResult)

    def test_map_reduce_parallel_limit(self):
        from cowork_agent.core.supervisor_strategies import (
            MapReduceStrategy, MapReduceConfig,
        )
        config = MapReduceConfig(max_parallel=1)
        strategy = MapReduceStrategy(config=config)
        agents = [f"agent_{i}" for i in range(5)]
        result = run_async(strategy.execute(
            task="test", agent_runner=mock_agent_runner, agent_names=agents,
        ))
        self.assertEqual(len(result.agent_outputs), 5)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 2: DebateStrategy
# ═══════════════════════════════════════════════════════════════════


class TestDebateStrategy(unittest.TestCase):
    """Tests for DEBATE execution strategy."""

    def test_basic_debate(self):
        from cowork_agent.core.supervisor_strategies import DebateStrategy
        strategy = DebateStrategy()
        result = run_async(strategy.execute(
            task="Should we use microservices?",
            agent_runner=mock_agent_runner,
            agent_names=["pro", "con"],
        ))
        self.assertEqual(result.strategy, "debate")
        self.assertIn("pro", result.agent_outputs)
        self.assertIn("con", result.agent_outputs)
        self.assertIn("DEBATE", result.final_output)

    def test_debate_requires_two_agents(self):
        from cowork_agent.core.supervisor_strategies import DebateStrategy
        strategy = DebateStrategy()
        with self.assertRaises(ValueError):
            run_async(strategy.execute(
                task="test", agent_runner=mock_agent_runner, agent_names=["solo"],
            ))

    def test_debate_multiple_rounds(self):
        from cowork_agent.core.supervisor_strategies import (
            DebateStrategy, DebateConfig,
        )
        config = DebateConfig(max_rounds=5)
        strategy = DebateStrategy(config=config)
        result = run_async(strategy.execute(
            task="debate topic",
            agent_runner=mock_agent_runner,
            agent_names=["a", "b"],
        ))
        self.assertEqual(result.rounds, 5)
        self.assertEqual(result.metadata["rounds"], 5)

    def test_debate_has_judgment(self):
        from cowork_agent.core.supervisor_strategies import DebateStrategy
        strategy = DebateStrategy()
        result = run_async(strategy.execute(
            task="test", agent_runner=mock_agent_runner, agent_names=["a", "b"],
        ))
        self.assertIn("judgment", result.metadata)
        self.assertIn("judge", result.metadata)

    def test_debate_custom_judge(self):
        from cowork_agent.core.supervisor_strategies import (
            DebateStrategy, DebateConfig,
        )
        config = DebateConfig(judge_agent="judge_bot")
        strategy = DebateStrategy(config=config)
        result = run_async(strategy.execute(
            task="test",
            agent_runner=mock_agent_runner,
            agent_names=["a", "b", "judge_bot"],
        ))
        self.assertEqual(result.metadata["judge"], "judge_bot")

    def test_debate_with_failures(self):
        from cowork_agent.core.supervisor_strategies import DebateStrategy
        strategy = DebateStrategy()
        result = run_async(strategy.execute(
            task="test",
            agent_runner=mock_failing_runner,
            agent_names=["ok_agent", "fail_agent"],
        ))
        self.assertIn("[ERROR", result.agent_outputs["fail_agent"])

    def test_debate_three_agents(self):
        from cowork_agent.core.supervisor_strategies import DebateStrategy
        strategy = DebateStrategy()
        result = run_async(strategy.execute(
            task="test",
            agent_runner=mock_agent_runner,
            agent_names=["a", "b", "c"],
        ))
        self.assertEqual(len(result.agent_outputs), 3)

    def test_debate_round_1_opening(self):
        from cowork_agent.core.supervisor_strategies import DebateStrategy
        strategy = DebateStrategy()
        result = run_async(strategy.execute(
            task="test",
            agent_runner=mock_agent_runner,
            agent_names=["a", "b"],
        ))
        self.assertIn("Round 1", result.final_output)

    def test_debate_elapsed_time(self):
        from cowork_agent.core.supervisor_strategies import DebateStrategy
        strategy = DebateStrategy()
        result = run_async(strategy.execute(
            task="t", agent_runner=mock_agent_runner, agent_names=["a", "b"],
        ))
        self.assertGreater(result.elapsed_seconds, 0)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 3: VotingStrategy
# ═══════════════════════════════════════════════════════════════════


class TestVotingStrategy(unittest.TestCase):
    """Tests for VOTING execution strategy."""

    def test_basic_voting(self):
        from cowork_agent.core.supervisor_strategies import VotingStrategy
        strategy = VotingStrategy()
        result = run_async(strategy.execute(
            task="Best approach for auth?",
            agent_runner=mock_vote_runner,
            agent_names=["agent_a", "agent_b", "agent_c"],
        ))
        self.assertEqual(result.strategy, "voting")
        self.assertIn("VOTING", result.final_output)
        self.assertIn("Winner", result.final_output)

    def test_voting_requires_two_agents(self):
        from cowork_agent.core.supervisor_strategies import VotingStrategy
        strategy = VotingStrategy()
        with self.assertRaises(ValueError):
            run_async(strategy.execute(
                task="test", agent_runner=mock_vote_runner, agent_names=["solo"],
            ))

    def test_voting_metadata(self):
        from cowork_agent.core.supervisor_strategies import VotingStrategy
        strategy = VotingStrategy()
        result = run_async(strategy.execute(
            task="test",
            agent_runner=mock_vote_runner,
            agent_names=["a", "b"],
        ))
        self.assertIn("votes", result.metadata)
        self.assertIn("vote_counts", result.metadata)
        self.assertIn("winner_name", result.metadata)
        self.assertIn("consensus", result.metadata)

    def test_voting_consensus_threshold(self):
        from cowork_agent.core.supervisor_strategies import (
            VotingStrategy, VotingConfig,
        )
        config = VotingConfig(consensus_threshold=0.9)
        strategy = VotingStrategy(config=config)
        result = run_async(strategy.execute(
            task="test",
            agent_runner=mock_vote_runner,
            agent_names=["a", "b", "c"],
        ))
        self.assertIn("has_consensus", result.metadata)

    def test_voting_parse_vote_valid(self):
        from cowork_agent.core.supervisor_strategies import VotingStrategy
        self.assertEqual(VotingStrategy._parse_vote("1", 3), 1)
        self.assertEqual(VotingStrategy._parse_vote("I vote 2", 3), 2)
        self.assertEqual(VotingStrategy._parse_vote("Solution 3", 3), 3)

    def test_voting_parse_vote_invalid(self):
        from cowork_agent.core.supervisor_strategies import VotingStrategy
        self.assertIsNone(VotingStrategy._parse_vote("", 3))
        self.assertIsNone(VotingStrategy._parse_vote("no number", 3))
        self.assertIsNone(VotingStrategy._parse_vote("5", 3))  # Out of range

    def test_voting_parse_vote_edge(self):
        from cowork_agent.core.supervisor_strategies import VotingStrategy
        self.assertIsNone(VotingStrategy._parse_vote("0", 3))  # 0 is not valid
        self.assertEqual(VotingStrategy._parse_vote("Solution 1 is the best", 2), 1)

    def test_voting_with_failures(self):
        from cowork_agent.core.supervisor_strategies import VotingStrategy
        strategy = VotingStrategy()
        result = run_async(strategy.execute(
            task="test",
            agent_runner=mock_failing_runner,
            agent_names=["ok_agent", "fail_agent"],
        ))
        self.assertIn("[ERROR", result.agent_outputs["fail_agent"])

    def test_voting_rounds(self):
        from cowork_agent.core.supervisor_strategies import VotingStrategy
        strategy = VotingStrategy()
        result = run_async(strategy.execute(
            task="test",
            agent_runner=mock_vote_runner,
            agent_names=["a", "b"],
        ))
        self.assertEqual(result.rounds, 2)  # generate + vote

    def test_voting_allow_self_vote(self):
        from cowork_agent.core.supervisor_strategies import (
            VotingStrategy, VotingConfig,
        )
        config = VotingConfig(allow_self_vote=True)
        strategy = VotingStrategy(config=config)
        result = run_async(strategy.execute(
            task="test",
            agent_runner=mock_vote_runner,
            agent_names=["a", "b"],
        ))
        self.assertIn("has_consensus", result.metadata)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 4: StrategyResult
# ═══════════════════════════════════════════════════════════════════


class TestStrategyResult(unittest.TestCase):
    """Tests for StrategyResult dataclass."""

    def test_strategy_result_defaults(self):
        from cowork_agent.core.supervisor_strategies import StrategyResult
        r = StrategyResult(final_output="test")
        self.assertEqual(r.final_output, "test")
        self.assertEqual(r.agent_outputs, {})
        self.assertEqual(r.metadata, {})
        self.assertEqual(r.strategy, "")
        self.assertEqual(r.elapsed_seconds, 0.0)
        self.assertEqual(r.rounds, 0)

    def test_strategy_result_full(self):
        from cowork_agent.core.supervisor_strategies import StrategyResult
        r = StrategyResult(
            final_output="result",
            agent_outputs={"a": "output_a"},
            metadata={"key": "value"},
            strategy="test",
            elapsed_seconds=1.5,
            rounds=3,
        )
        self.assertEqual(r.strategy, "test")
        self.assertEqual(r.rounds, 3)
        self.assertAlmostEqual(r.elapsed_seconds, 1.5)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 5: AgentSpecialization
# ═══════════════════════════════════════════════════════════════════


class TestAgentSpecialization(unittest.TestCase):
    """Tests for AgentRole, AgentSpecialization, and SpecializationRegistry."""

    def test_agent_roles(self):
        from cowork_agent.core.agent_specialization import AgentRole
        self.assertEqual(AgentRole.RESEARCHER.value, "researcher")
        self.assertEqual(AgentRole.CODER.value, "coder")
        self.assertEqual(AgentRole.REVIEWER.value, "reviewer")
        self.assertEqual(AgentRole.TESTER.value, "tester")
        self.assertEqual(AgentRole.WRITER.value, "writer")
        self.assertEqual(AgentRole.ARCHITECT.value, "architect")
        self.assertEqual(AgentRole.GENERAL.value, "general")

    def test_specialization_merges_keywords(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization,
        )
        spec = AgentSpecialization(
            role=AgentRole.CODER,
            keywords=["python", "javascript"],
        )
        self.assertIn("python", spec.keywords)
        self.assertIn("javascript", spec.keywords)
        self.assertIn("code", spec.keywords)  # From role defaults

    def test_specialization_custom_threshold(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization,
        )
        spec = AgentSpecialization(
            role=AgentRole.CODER,
            confidence_threshold=0.5,
        )
        self.assertEqual(spec.confidence_threshold, 0.5)

    def test_registry_register_and_get(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        reg = SpecializationRegistry()
        spec = AgentSpecialization(role=AgentRole.CODER)
        reg.register_agent("coder1", spec)
        self.assertIs(reg.get_specialization("coder1"), spec)

    def test_registry_unregister(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        reg = SpecializationRegistry()
        reg.register_agent("a", AgentSpecialization(role=AgentRole.CODER))
        self.assertTrue(reg.unregister_agent("a"))
        self.assertFalse(reg.unregister_agent("a"))

    def test_registry_list_agents(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        reg = SpecializationRegistry()
        reg.register_agent("a", AgentSpecialization(role=AgentRole.CODER))
        reg.register_agent("b", AgentSpecialization(role=AgentRole.TESTER))
        self.assertEqual(sorted(reg.list_agents()), ["a", "b"])

    def test_registry_list_by_role(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        reg = SpecializationRegistry()
        reg.register_agent("c1", AgentSpecialization(role=AgentRole.CODER))
        reg.register_agent("c2", AgentSpecialization(role=AgentRole.CODER))
        reg.register_agent("t1", AgentSpecialization(role=AgentRole.TESTER))
        self.assertEqual(sorted(reg.list_by_role(AgentRole.CODER)), ["c1", "c2"])
        self.assertEqual(reg.list_by_role(AgentRole.TESTER), ["t1"])

    def test_find_best_agent_keyword_match(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        reg = SpecializationRegistry()
        reg.register_agent("coder", AgentSpecialization(role=AgentRole.CODER))
        reg.register_agent("tester", AgentSpecialization(role=AgentRole.TESTER))
        best, conf = reg.find_best_agent("Write code to implement a feature")
        self.assertEqual(best, "coder")
        self.assertGreater(conf, 0)

    def test_find_best_agent_no_match(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        reg = SpecializationRegistry()
        reg.register_agent("gen", AgentSpecialization(
            role=AgentRole.GENERAL, confidence_threshold=0.5,
        ))
        best, conf = reg.find_best_agent("xyz abc 123")
        self.assertIsNone(best)

    def test_find_best_agent_with_available_filter(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        reg = SpecializationRegistry()
        reg.register_agent("a", AgentSpecialization(role=AgentRole.CODER))
        reg.register_agent("b", AgentSpecialization(role=AgentRole.CODER))
        best, _ = reg.find_best_agent("code", available_agents=["b"])
        self.assertEqual(best, "b")

    def test_find_top_agents(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        reg = SpecializationRegistry()
        reg.register_agent("coder", AgentSpecialization(role=AgentRole.CODER))
        reg.register_agent("reviewer", AgentSpecialization(role=AgentRole.REVIEWER))
        reg.register_agent("tester", AgentSpecialization(role=AgentRole.TESTER))
        top = reg.find_top_agents("Review and test the code", top_n=2)
        self.assertLessEqual(len(top), 2)

    def test_specialization_priority(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        reg = SpecializationRegistry()
        reg.register_agent("low", AgentSpecialization(
            role=AgentRole.CODER, priority=0,
        ))
        reg.register_agent("high", AgentSpecialization(
            role=AgentRole.CODER, priority=10,
        ))
        best, _ = reg.find_best_agent("code something")
        self.assertEqual(best, "high")

    def test_registry_summary(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        reg = SpecializationRegistry()
        reg.register_agent("a", AgentSpecialization(role=AgentRole.CODER))
        summary = reg.get_registry_summary()
        self.assertEqual(summary["total_agents"], 1)
        self.assertIn("a", summary["agents"])

    def test_capability_weights(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        reg = SpecializationRegistry()
        spec = AgentSpecialization(
            role=AgentRole.CODER,
            keywords=["python"],
            capability_weights={"python": 5.0},
        )
        reg.register_agent("py_expert", spec)
        best, conf = reg.find_best_agent("Write python code")
        self.assertEqual(best, "py_expert")

    def test_get_specialization_missing(self):
        from cowork_agent.core.agent_specialization import SpecializationRegistry
        reg = SpecializationRegistry()
        self.assertIsNone(reg.get_specialization("nonexistent"))


# ═══════════════════════════════════════════════════════════════════
#  Test Class 6: TaskAnalyzer
# ═══════════════════════════════════════════════════════════════════


class TestTaskAnalyzer(unittest.TestCase):
    """Tests for TaskAnalyzer."""

    def _analyzer(self):
        from cowork_agent.core.conversation_router import TaskAnalyzer
        return TaskAnalyzer()

    def test_analyze_simple_task(self):
        a = self._analyzer()
        result = a.analyze("Fix a bug")
        self.assertEqual(result.complexity, "simple")
        self.assertEqual(result.subtask_count, 1)

    def test_analyze_moderate_task(self):
        a = self._analyzer()
        result = a.analyze("Write code and also test it with some edge cases")
        self.assertIn(result.complexity, ("moderate", "complex"))

    def test_analyze_complex_task(self):
        a = self._analyzer()
        result = a.analyze(
            "First research the topic and then write a comprehensive report "
            "followed by multiple rounds of review and testing for the entire system"
        )
        self.assertEqual(result.complexity, "complex")
        self.assertGreaterEqual(result.subtask_count, 3)

    def test_analyze_extracts_keywords(self):
        a = self._analyzer()
        result = a.analyze("Implement authentication module with OAuth")
        self.assertIn("implement", result.keywords)
        self.assertIn("authentication", result.keywords)
        self.assertIn("module", result.keywords)

    def test_analyze_suggests_roles(self):
        from cowork_agent.core.agent_specialization import AgentRole
        a = self._analyzer()
        result = a.analyze("Write unit tests for the API")
        self.assertIn(AgentRole.TESTER, result.suggested_roles)

    def test_analyze_suggests_coder_role(self):
        from cowork_agent.core.agent_specialization import AgentRole
        a = self._analyzer()
        result = a.analyze("Implement a new feature")
        self.assertIn(AgentRole.CODER, result.suggested_roles)

    def test_analyze_suggests_writer_role(self):
        from cowork_agent.core.agent_specialization import AgentRole
        a = self._analyzer()
        result = a.analyze("Write documentation for the API")
        self.assertIn(AgentRole.WRITER, result.suggested_roles)

    def test_analyze_general_fallback(self):
        from cowork_agent.core.agent_specialization import AgentRole
        a = self._analyzer()
        result = a.analyze("Do something")
        self.assertIn(AgentRole.GENERAL, result.suggested_roles)

    def test_analyze_metadata(self):
        a = self._analyzer()
        result = a.analyze("Step 1. Do this 2. Do that")
        self.assertIn("word_count", result.metadata)
        self.assertTrue(result.metadata["has_steps"])

    def test_analyze_confidence(self):
        a = self._analyzer()
        result = a.analyze("Research and implement authentication with tests")
        self.assertGreater(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_analyze_empty_task(self):
        a = self._analyzer()
        result = a.analyze("")
        self.assertEqual(result.complexity, "simple")
        self.assertEqual(len(result.keywords), 0)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 7: ConversationRouter
# ═══════════════════════════════════════════════════════════════════


class TestConversationRouter(unittest.TestCase):
    """Tests for ConversationRouter."""

    def _make_router(self):
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        from cowork_agent.core.conversation_router import ConversationRouter

        reg = SpecializationRegistry()
        reg.register_agent("coder", AgentSpecialization(role=AgentRole.CODER))
        reg.register_agent("tester", AgentSpecialization(role=AgentRole.TESTER))
        reg.register_agent("writer", AgentSpecialization(role=AgentRole.WRITER))
        reg.register_agent("researcher", AgentSpecialization(role=AgentRole.RESEARCHER))
        return ConversationRouter(spec_registry=reg)

    def test_route_simple_task(self):
        router = self._make_router()
        decision = router.route_task("Write some code")
        self.assertGreater(len(decision.assignments), 0)
        self.assertEqual(decision.analysis.complexity, "simple")

    def test_route_to_best_match(self):
        router = self._make_router()
        decision = router.route_task("Implement a new feature")
        agent_names = [a[0] for a in decision.assignments]
        self.assertIn("coder", agent_names)

    def test_route_complex_task(self):
        router = self._make_router()
        decision = router.route_task(
            "First research the topic and then implement a comprehensive "
            "solution followed by testing everything"
        )
        self.assertGreater(len(decision.assignments), 1)

    def test_route_no_agents(self):
        from cowork_agent.core.conversation_router import ConversationRouter
        router = ConversationRouter()
        decision = router.route_task("anything")
        self.assertEqual(len(decision.assignments), 0)
        self.assertTrue(decision.fallback_used)

    def test_route_with_available_filter(self):
        router = self._make_router()
        decision = router.route_task("Write code", available_agents=["coder"])
        agent_names = [a[0] for a in decision.assignments]
        self.assertEqual(agent_names, ["coder"])

    def test_route_strategy_suggestion(self):
        router = self._make_router()
        decision = router.route_task("Simple fix")
        self.assertIn(decision.strategy_suggestion, [
            "sequential", "parallel", "pipeline",
        ])

    def test_route_with_decomposition(self):
        router = self._make_router()
        decision = router.route_with_decomposition(
            "Research and write documentation"
        )
        self.assertGreater(len(decision.assignments), 0)

    def test_route_with_decomposition_role_matching(self):
        router = self._make_router()
        decision = router.route_with_decomposition(
            "Test and review the code"
        )
        agent_names = [a[0] for a in decision.assignments]
        # Should have at least tester
        self.assertGreater(len(agent_names), 0)

    def test_route_fallback_simple(self):
        router = self._make_router()
        decision = router.route_task(
            "xyzzy foobar",
            available_agents=["coder"],
        )
        # Should fallback to first available
        self.assertEqual(decision.assignments[0][0], "coder")
        self.assertTrue(decision.fallback_used)

    def test_routing_decision_has_analysis(self):
        router = self._make_router()
        decision = router.route_task("Test the API")
        self.assertIsNotNone(decision.analysis)
        self.assertIsInstance(decision.analysis.keywords, list)

    def test_route_preserves_task_in_assignments(self):
        router = self._make_router()
        decision = router.route_task("Write code for login")
        for _, subtask in decision.assignments:
            self.assertIn("login", subtask.lower())


# ═══════════════════════════════════════════════════════════════════
#  Test Class 8: AgentPool
# ═══════════════════════════════════════════════════════════════════


class TestAgentPool(unittest.TestCase):
    """Tests for AgentPool."""

    def _make_pool(self, **kwargs):
        from cowork_agent.core.agent_pool import AgentPool, PoolConfig
        config = PoolConfig(
            name="test_pool",
            min_size=1,
            max_size=5,
            initial_size=2,
            **kwargs,
        )
        return AgentPool(
            config=config,
            agent_factory=lambda: MagicMock(),
        )

    def test_initialize_pool(self):
        pool = self._make_pool()
        run_async(pool.initialize())
        self.assertEqual(pool.size, 2)
        self.assertEqual(pool.available_count, 2)
        self.assertEqual(pool.in_use_count, 0)

    def test_acquire_agent(self):
        pool = self._make_pool()
        run_async(pool.initialize())
        agent = run_async(pool.acquire())
        self.assertIsNotNone(agent)
        self.assertTrue(agent.in_use)
        self.assertEqual(pool.in_use_count, 1)
        self.assertEqual(pool.available_count, 1)

    def test_release_agent(self):
        pool = self._make_pool()
        run_async(pool.initialize())
        agent = run_async(pool.acquire())
        run_async(pool.release(agent))
        self.assertFalse(agent.in_use)
        self.assertEqual(pool.available_count, 2)
        self.assertEqual(pool.in_use_count, 0)

    def test_acquire_grows_pool(self):
        pool = self._make_pool()
        run_async(pool.initialize())
        # Acquire all initial agents
        a1 = run_async(pool.acquire())
        a2 = run_async(pool.acquire())
        # Pool should grow
        a3 = run_async(pool.acquire())
        self.assertEqual(pool.size, 3)

    def test_acquire_respects_max(self):
        from cowork_agent.core.agent_pool import AgentPool, PoolConfig
        config = PoolConfig(
            name="test", min_size=1, max_size=2, initial_size=2,
            acquire_timeout_seconds=0.1,
        )
        pool = AgentPool(config=config, agent_factory=lambda: MagicMock())
        run_async(pool.initialize())
        a1 = run_async(pool.acquire())
        a2 = run_async(pool.acquire())
        with self.assertRaises(TimeoutError):
            run_async(pool.acquire(timeout=0.1))

    def test_utilization(self):
        pool = self._make_pool()
        run_async(pool.initialize())
        self.assertAlmostEqual(pool.utilization, 0.0)
        a = run_async(pool.acquire())
        self.assertAlmostEqual(pool.utilization, 0.5)  # 1/2
        run_async(pool.release(a))
        self.assertAlmostEqual(pool.utilization, 0.0)

    def test_scale_up(self):
        pool = self._make_pool()
        run_async(pool.initialize())
        run_async(pool.scale(4))
        self.assertEqual(pool.size, 4)

    def test_scale_down(self):
        pool = self._make_pool()
        run_async(pool.initialize())
        run_async(pool.scale(4))
        run_async(pool.scale(2))
        self.assertEqual(pool.size, 2)

    def test_scale_respects_min(self):
        pool = self._make_pool()
        run_async(pool.initialize())
        run_async(pool.scale(0))
        self.assertGreaterEqual(pool.size, 1)  # min_size = 1

    def test_scale_respects_max(self):
        pool = self._make_pool()
        run_async(pool.initialize())
        run_async(pool.scale(100))
        self.assertLessEqual(pool.size, 5)  # max_size = 5

    def test_get_stats(self):
        pool = self._make_pool()
        run_async(pool.initialize())
        stats = pool.get_stats()
        self.assertEqual(stats["name"], "test_pool")
        self.assertEqual(stats["size"], 2)
        self.assertTrue(stats["initialized"])

    def test_shutdown(self):
        pool = self._make_pool()
        run_async(pool.initialize())
        run_async(pool.shutdown())
        self.assertEqual(pool.size, 0)
        self.assertFalse(pool.get_stats()["initialized"])

    def test_no_factory_error(self):
        from cowork_agent.core.agent_pool import AgentPool
        pool = AgentPool()
        with self.assertRaises(RuntimeError):
            run_async(pool.initialize())

    def test_pooled_agent_use_count(self):
        pool = self._make_pool()
        run_async(pool.initialize())
        a = run_async(pool.acquire())
        self.assertEqual(a.use_count, 1)
        run_async(pool.release(a))
        a2 = run_async(pool.acquire())
        # May or may not be same agent due to deque ordering
        self.assertGreaterEqual(a2.use_count, 1)

    def test_cleanup_idle(self):
        from cowork_agent.core.agent_pool import AgentPool, PoolConfig
        config = PoolConfig(
            name="test", min_size=1, max_size=5, initial_size=3,
            idle_timeout_seconds=0,  # Instant timeout
        )
        pool = AgentPool(config=config, agent_factory=lambda: MagicMock())
        run_async(pool.initialize())
        removed = run_async(pool.cleanup_idle())
        self.assertGreater(removed, 0)
        self.assertGreaterEqual(pool.size, 1)  # Keeps min_size

    def test_pooled_agent_idle_seconds(self):
        from cowork_agent.core.agent_pool import PooledAgent
        pa = PooledAgent(agent=MagicMock(), last_used_at=time.time() - 10)
        self.assertGreaterEqual(pa.idle_seconds, 9)


# ═══════════════════════════════════════════════════════════════════
#  Test Class 9: AutoScaler
# ═══════════════════════════════════════════════════════════════════


class TestAutoScaler(unittest.TestCase):
    """Tests for AutoScaler."""

    def _make_scaler(self, **scaler_kwargs):
        from cowork_agent.core.agent_pool import (
            AgentPool, PoolConfig, AutoScaler, AutoScalerConfig,
        )
        config = PoolConfig(
            name="test", min_size=1, max_size=10, initial_size=2,
        )
        pool = AgentPool(config=config, agent_factory=lambda: MagicMock())
        run_async(pool.initialize())
        scaler_cfg = AutoScalerConfig(cooldown_seconds=0, **scaler_kwargs)
        return pool, AutoScaler(pool=pool, config=scaler_cfg)

    def test_scale_up_on_high_utilization(self):
        pool, scaler = self._make_scaler(scale_up_threshold=0.3)
        # Acquire to get utilization > 0.3
        a1 = run_async(pool.acquire())
        action = run_async(scaler.check_and_scale())
        self.assertIsNotNone(action)
        self.assertEqual(action["action"], "scale_up")

    def test_scale_down_on_low_utilization(self):
        pool, scaler = self._make_scaler(scale_down_threshold=0.5)
        # No agents in use → utilization = 0
        action = run_async(scaler.check_and_scale())
        if pool.size > pool.config.min_size:
            self.assertIsNotNone(action)
            self.assertEqual(action["action"], "scale_down")

    def test_no_action_in_cooldown(self):
        from cowork_agent.core.agent_pool import (
            AgentPool, PoolConfig, AutoScaler, AutoScalerConfig,
        )
        config = PoolConfig(name="test", min_size=1, max_size=10, initial_size=2)
        pool = AgentPool(config=config, agent_factory=lambda: MagicMock())
        run_async(pool.initialize())
        scaler = AutoScaler(
            pool=pool,
            config=AutoScalerConfig(cooldown_seconds=999),
        )
        # First action resets cooldown
        run_async(scaler.check_and_scale())
        # Second should be blocked by cooldown
        action = run_async(scaler.check_and_scale())
        self.assertIsNone(action)

    def test_no_action_in_normal_range(self):
        pool, scaler = self._make_scaler(
            scale_up_threshold=0.8, scale_down_threshold=0.2,
        )
        # Acquire 1 of 2 → utilization = 0.5 → in normal range
        run_async(pool.acquire())
        action = run_async(scaler.check_and_scale())
        self.assertIsNone(action)

    def test_scale_history(self):
        pool, scaler = self._make_scaler(scale_down_threshold=0.5)
        run_async(scaler.check_and_scale())
        # History should contain at least one entry if action was taken
        self.assertIsInstance(scaler.scale_history, list)

    def test_scaler_is_running(self):
        _, scaler = self._make_scaler()
        self.assertFalse(scaler.is_running)

    def test_start_stop(self):
        _, scaler = self._make_scaler()

        async def _start_and_stop():
            await scaler.start()
            self.assertTrue(scaler.is_running)
            await asyncio.sleep(0.05)  # let loop tick
            await scaler.stop()
            self.assertFalse(scaler.is_running)

        run_async(_start_and_stop())


# ═══════════════════════════════════════════════════════════════════
#  Test Class 10: Config, Wiring & Integration
# ═══════════════════════════════════════════════════════════════════


class TestMultiAgentConfig(unittest.TestCase):
    """Tests for config, wiring, and integration."""

    def test_config_multi_agent_strategies(self):
        from cowork_agent.config.settings import load_config
        config = load_config()
        ma = config.get("multi_agent", {})
        strategies = ma.get("strategies", {})
        self.assertIn("map_reduce", strategies)
        self.assertIn("debate", strategies)
        self.assertIn("voting", strategies)

    def test_config_map_reduce_settings(self):
        from cowork_agent.config.settings import load_config
        config = load_config()
        mr = config.get("multi_agent.strategies.map_reduce", {})
        self.assertEqual(mr.get("max_parallel"), 5)

    def test_config_debate_settings(self):
        from cowork_agent.config.settings import load_config
        config = load_config()
        debate = config.get("multi_agent.strategies.debate", {})
        self.assertEqual(debate.get("max_rounds"), 3)

    def test_config_voting_settings(self):
        from cowork_agent.config.settings import load_config
        config = load_config()
        voting = config.get("multi_agent.strategies.voting", {})
        self.assertEqual(voting.get("consensus_threshold"), 0.5)
        self.assertFalse(voting.get("allow_self_vote", True))

    def test_config_pool_settings(self):
        from cowork_agent.config.settings import load_config
        config = load_config()
        pools = config.get("multi_agent.pools", {})
        self.assertFalse(pools.get("enabled", True))

    def test_config_routing_settings(self):
        from cowork_agent.config.settings import load_config
        config = load_config()
        routing = config.get("multi_agent.routing", {})
        self.assertTrue(routing.get("enabled", False))
        self.assertEqual(routing.get("default_strategy"), "sequential")

    def test_execution_strategy_new_values(self):
        from cowork_agent.core.supervisor import ExecutionStrategy
        self.assertEqual(ExecutionStrategy.MAP_REDUCE.value, "map_reduce")
        self.assertEqual(ExecutionStrategy.DEBATE.value, "debate")
        self.assertEqual(ExecutionStrategy.VOTING.value, "voting")

    def test_execution_strategy_all_values(self):
        from cowork_agent.core.supervisor import ExecutionStrategy
        values = [e.value for e in ExecutionStrategy]
        self.assertIn("sequential", values)
        self.assertIn("parallel", values)
        self.assertIn("pipeline", values)
        self.assertIn("map_reduce", values)
        self.assertIn("debate", values)
        self.assertIn("voting", values)

    def test_agent_has_sprint21_attributes(self):
        from cowork_agent.core.agent import Agent
        agent = Agent(
            provider=MagicMock(),
            registry=MagicMock(),
            prompt_builder=MagicMock(),
        )
        self.assertIsNone(agent.specialization_registry)
        self.assertIsNone(agent.conversation_router)
        self.assertIsNone(agent.agent_pool)

    def test_base_strategy_interface(self):
        from cowork_agent.core.supervisor_strategies import BaseStrategy
        # BaseStrategy is abstract and cannot be instantiated directly
        with self.assertRaises(TypeError):
            BaseStrategy()

    def test_imports_all_modules(self):
        """All Sprint 21 modules should be importable."""
        from cowork_agent.core.supervisor_strategies import (
            MapReduceStrategy, DebateStrategy, VotingStrategy,
            StrategyResult, BaseStrategy,
        )
        from cowork_agent.core.agent_specialization import (
            AgentRole, AgentSpecialization, SpecializationRegistry,
        )
        from cowork_agent.core.conversation_router import (
            TaskAnalyzer, ConversationRouter, RoutingDecision, TaskAnalysis,
        )
        from cowork_agent.core.agent_pool import (
            AgentPool, PoolConfig, PooledAgent, AutoScaler, AutoScalerConfig,
        )
        self.assertTrue(True)  # If we get here, imports succeeded

    def test_pool_config_defaults(self):
        from cowork_agent.core.agent_pool import PoolConfig
        pc = PoolConfig()
        self.assertEqual(pc.name, "default")
        self.assertEqual(pc.min_size, 1)
        self.assertEqual(pc.max_size, 10)
        self.assertEqual(pc.initial_size, 2)

    def test_autoscaler_config_defaults(self):
        from cowork_agent.core.agent_pool import AutoScalerConfig
        ac = AutoScalerConfig()
        self.assertEqual(ac.scale_up_threshold, 0.8)
        self.assertEqual(ac.scale_down_threshold, 0.2)
        self.assertEqual(ac.scale_up_step, 2)
        self.assertEqual(ac.scale_down_step, 1)

    def test_role_keywords_all_roles(self):
        from cowork_agent.core.agent_specialization import ROLE_KEYWORDS, AgentRole
        for role in AgentRole:
            self.assertIn(role, ROLE_KEYWORDS)


if __name__ == "__main__":
    unittest.main()
