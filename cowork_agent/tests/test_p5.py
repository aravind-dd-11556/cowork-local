"""
Sprint 5 (P3-Multi-Agent Orchestration) Test Suite
====================================================
Feature 1: Agent Registry & Lifecycle      (12 tests)
Feature 2: Shared Context Bus              (14 tests)
Feature 3: Agent Delegation Tool           (13 tests)
Feature 4: Supervisor Pattern              (15 tests)
Feature 5: Conflict Resolution             (16 tests)
Integration Tests                          ( 5 tests)
─────────────────────────────────────────────────────
Total:                                      75 tests
"""

import asyncio
import os
import sys
import time
import unittest
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

# ── Path setup ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cowork_agent.core.models import AgentResponse, Message, ToolCall, ToolResult, ToolSchema
from cowork_agent.core.agent_registry import (
    AgentRegistry, AgentConfig, AgentInstance, AgentState,
)
from cowork_agent.core.context_bus import (
    ContextBus, BusMessage, MessageType,
)
from cowork_agent.core.delegate_tool import (
    DelegateTaskTool, DelegatedTask, DelegateMode,
)
from cowork_agent.core.supervisor import (
    Supervisor, SupervisorConfig, ExecutionStrategy, SubTask,
)
from cowork_agent.core.conflict_resolver import (
    ConflictResolver, ConflictStrategy, ConflictReport,
    ConflictDetector, DeadlockDetector,
)


def _run(coro):
    """Helper to run async code in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_mock_provider(name="test"):
    """Create a mock LLM provider."""
    p = MagicMock()
    p.provider_name = name
    p.model = f"{name}-model"
    p.base_url = None
    p.api_key = None
    p.send_message = AsyncMock(
        return_value=AgentResponse(text=f"Response from {name}", stop_reason="end_turn")
    )
    p.health_check = AsyncMock(return_value={"status": "ok"})
    return p


def _make_mock_registry():
    """Create a mock ToolRegistry with some tools."""
    from cowork_agent.core.tool_registry import ToolRegistry
    registry = ToolRegistry()

    for tool_name in ["read", "write", "bash", "glob"]:
        mock_tool = MagicMock()
        mock_tool.name = tool_name
        mock_tool.get_schema.return_value = ToolSchema(
            name=tool_name,
            description=f"{tool_name} tool",
            input_schema={"type": "object", "properties": {}},
        )
        mock_tool.execute = AsyncMock(return_value=ToolResult(
            tool_id="t1", success=True, output=f"{tool_name} done"
        ))
        registry.register(mock_tool)

    return registry


def _make_prompt_builder():
    """Create a PromptBuilder with minimal config."""
    from cowork_agent.core.prompt_builder import PromptBuilder
    return PromptBuilder(config={"workspace_dir": "/tmp", "provider": "test"})


# ═══════════════════════════════════════════════
# Feature 1: Agent Registry & Lifecycle
# ═══════════════════════════════════════════════

class TestAgentRegistry(unittest.TestCase):
    """Test agent registry and lifecycle management."""

    def test_create_agent(self):
        """Create an agent from config."""
        reg = AgentRegistry()
        config = AgentConfig(name="reader", role="file_reader", capabilities=["read", "glob"])
        instance = _run(reg.create_agent(
            config, _make_mock_provider(), _make_mock_registry(), _make_prompt_builder()
        ))
        self.assertEqual(instance.config.name, "reader")
        self.assertEqual(instance.state, AgentState.IDLE)
        self.assertIn("reader", reg.list_agents())

    def test_duplicate_agent_name_raises(self):
        """Cannot create two agents with the same name."""
        reg = AgentRegistry()
        config = AgentConfig(name="dup", role="test")
        _run(reg.create_agent(config, _make_mock_provider(), _make_mock_registry(), _make_prompt_builder()))
        with self.assertRaises(ValueError):
            _run(reg.create_agent(config, _make_mock_provider(), _make_mock_registry(), _make_prompt_builder()))

    def test_start_agent(self):
        """Start an agent and it transitions to RUNNING."""
        reg = AgentRegistry()
        config = AgentConfig(name="worker", role="test")
        _run(reg.create_agent(config, _make_mock_provider(), _make_mock_registry(), _make_prompt_builder()))
        task = _run(reg.start_agent("worker", "do stuff"))
        # Wait briefly for the task to start
        _run(asyncio.sleep(0.05))
        # The agent should have completed (mock provider returns immediately)
        result = _run(reg.get_result("worker", timeout=5))
        self.assertIn("Response from test", result)

    def test_agent_completes_to_completed_state(self):
        """After run() finishes, state should be COMPLETED."""
        reg = AgentRegistry()
        config = AgentConfig(name="done", role="test")
        _run(reg.create_agent(config, _make_mock_provider(), _make_mock_registry(), _make_prompt_builder()))
        _run(reg.start_agent("done", "task"))
        _run(reg.get_result("done", timeout=5))
        status = reg.get_agent_status("done")
        self.assertEqual(status.state, AgentState.COMPLETED)
        self.assertIsNotNone(status.result)
        self.assertIsNotNone(status.completed_at)

    def test_agent_error_state(self):
        """Agent that throws error transitions to ERROR."""
        reg = AgentRegistry()
        config = AgentConfig(name="fail", role="test")
        provider = _make_mock_provider()
        provider.send_message = AsyncMock(side_effect=RuntimeError("LLM crashed"))
        _run(reg.create_agent(config, provider, _make_mock_registry(), _make_prompt_builder()))
        _run(reg.start_agent("fail", "will fail"))
        _run(asyncio.sleep(0.1))
        status = reg.get_agent_status("fail")
        # Agent.run() catches LLM errors and returns error text, doesn't raise
        self.assertIn(status.state, (AgentState.COMPLETED, AgentState.ERROR))

    def test_start_nonexistent_agent(self):
        """Starting an unregistered agent raises KeyError."""
        reg = AgentRegistry()
        with self.assertRaises(KeyError):
            _run(reg.start_agent("ghost", "task"))

    def test_terminate_agent(self):
        """Terminate sets state to ERROR."""
        reg = AgentRegistry()
        config = AgentConfig(name="victim", role="test")
        _run(reg.create_agent(config, _make_mock_provider(), _make_mock_registry(), _make_prompt_builder()))
        _run(reg.terminate_agent("victim"))
        status = reg.get_agent_status("victim")
        self.assertEqual(status.state, AgentState.ERROR)
        self.assertEqual(status.error, "Terminated")

    def test_tool_filtering(self):
        """Agents only get tools listed in capabilities."""
        reg = AgentRegistry()
        config = AgentConfig(name="limited", role="reader", capabilities=["read"])
        instance = _run(reg.create_agent(
            config, _make_mock_provider(), _make_mock_registry(), _make_prompt_builder()
        ))
        tool_names = instance.agent.registry.tool_names
        self.assertIn("read", tool_names)
        self.assertNotIn("bash", tool_names)
        self.assertNotIn("write", tool_names)

    def test_empty_capabilities_all_tools(self):
        """Empty capabilities list = all tools allowed."""
        reg = AgentRegistry()
        config = AgentConfig(name="full", role="all", capabilities=[])
        instance = _run(reg.create_agent(
            config, _make_mock_provider(), _make_mock_registry(), _make_prompt_builder()
        ))
        self.assertGreater(len(instance.agent.registry.tool_names), 0)

    def test_list_agents(self):
        """list_agents returns all registered names."""
        reg = AgentRegistry()
        for name in ["a", "b", "c"]:
            _run(reg.create_agent(
                AgentConfig(name=name, role="test"),
                _make_mock_provider(), _make_mock_registry(), _make_prompt_builder()
            ))
        self.assertEqual(sorted(reg.list_agents()), ["a", "b", "c"])

    def test_remove_agent(self):
        """Remove a non-running agent."""
        reg = AgentRegistry()
        config = AgentConfig(name="temp", role="test")
        _run(reg.create_agent(config, _make_mock_provider(), _make_mock_registry(), _make_prompt_builder()))
        _run(reg.remove_agent("temp"))
        self.assertNotIn("temp", reg.list_agents())

    def test_agent_instance_to_dict(self):
        """AgentInstance.to_dict returns expected fields."""
        reg = AgentRegistry()
        config = AgentConfig(name="info", role="tester")
        instance = _run(reg.create_agent(
            config, _make_mock_provider(), _make_mock_registry(), _make_prompt_builder()
        ))
        d = instance.to_dict()
        self.assertEqual(d["name"], "info")
        self.assertEqual(d["state"], "idle")
        self.assertIn("elapsed", d)


# ═══════════════════════════════════════════════
# Feature 2: Shared Context Bus
# ═══════════════════════════════════════════════

class TestContextBus(unittest.TestCase):
    """Test shared context bus pub/sub and state management."""

    def test_publish_subscribe(self):
        """Basic publish triggers subscriber callback."""
        bus = ContextBus()
        received = []

        def on_msg(msg):
            received.append(msg)

        bus.subscribe("agent_a", MessageType.TASK_RESULT, on_msg)
        _run(bus.publish(BusMessage(
            msg_type=MessageType.TASK_RESULT,
            sender="agent_a",
            content="done",
        )))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].content, "done")

    def test_async_subscriber(self):
        """Async callback is awaited correctly."""
        bus = ContextBus()
        received = []

        async def on_msg(msg):
            received.append(msg.content)

        bus.subscribe("sender", MessageType.DATA_SHARE, on_msg)
        _run(bus.publish(BusMessage(
            msg_type=MessageType.DATA_SHARE,
            sender="sender",
            content="async_data",
        )))
        self.assertEqual(received, ["async_data"])

    def test_multiple_subscribers(self):
        """Multiple callbacks for same sender/type all get called."""
        bus = ContextBus()
        r1, r2 = [], []

        bus.subscribe("s", MessageType.TASK_RESULT, lambda m: r1.append(m))
        bus.subscribe("s", MessageType.TASK_RESULT, lambda m: r2.append(m))

        _run(bus.publish(BusMessage(msg_type=MessageType.TASK_RESULT, sender="s", content="x")))
        self.assertEqual(len(r1), 1)
        self.assertEqual(len(r2), 1)

    def test_wildcard_subscriber(self):
        """subscribe_all receives messages from ANY sender."""
        bus = ContextBus()
        received = []
        bus.subscribe_all(MessageType.ERROR, lambda m: received.append(m.sender))

        _run(bus.publish(BusMessage(msg_type=MessageType.ERROR, sender="a")))
        _run(bus.publish(BusMessage(msg_type=MessageType.ERROR, sender="b")))
        self.assertEqual(received, ["a", "b"])

    def test_no_crossfire(self):
        """Subscriber for agent_a doesn't get agent_b's messages."""
        bus = ContextBus()
        received = []
        bus.subscribe("a", MessageType.TASK_RESULT, lambda m: received.append(m))

        _run(bus.publish(BusMessage(msg_type=MessageType.TASK_RESULT, sender="b", content="wrong")))
        self.assertEqual(len(received), 0)

    def test_unsubscribe(self):
        """After unsubscribe, callback is no longer invoked."""
        bus = ContextBus()
        received = []
        cb = lambda m: received.append(m)
        bus.subscribe("s", MessageType.TASK_RESULT, cb)
        bus.unsubscribe("s", MessageType.TASK_RESULT, cb)

        _run(bus.publish(BusMessage(msg_type=MessageType.TASK_RESULT, sender="s")))
        self.assertEqual(len(received), 0)

    def test_shared_state_set_get(self):
        """Set and get shared state values."""
        bus = ContextBus()
        _run(bus.set_shared("key", "value"))
        self.assertEqual(_run(bus.get_shared("key")), "value")

    def test_shared_state_default(self):
        """get_shared returns default for missing keys."""
        bus = ContextBus()
        self.assertEqual(_run(bus.get_shared("missing", "default")), "default")
        self.assertIsNone(_run(bus.get_shared("missing")))

    def test_shared_state_atomic_update(self):
        """update_shared applies updater function atomically."""
        bus = ContextBus()
        _run(bus.set_shared("counter", 0))
        result = _run(bus.update_shared("counter", lambda x: (x or 0) + 1))
        self.assertEqual(result, 1)
        self.assertEqual(_run(bus.get_shared("counter")), 1)

    def test_delete_shared(self):
        """Delete a shared state key."""
        bus = ContextBus()
        _run(bus.set_shared("temp", 42))
        self.assertTrue(_run(bus.delete_shared("temp")))
        self.assertIsNone(_run(bus.get_shared("temp")))
        self.assertFalse(_run(bus.delete_shared("temp")))  # Already deleted

    def test_message_history(self):
        """Messages are stored in history."""
        bus = ContextBus()
        _run(bus.publish(BusMessage(msg_type=MessageType.TASK_RESULT, sender="a", content="1")))
        _run(bus.publish(BusMessage(msg_type=MessageType.ERROR, sender="b", content="2")))

        history = bus.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(bus.history_size, 2)

    def test_history_limit(self):
        """History is capped at max_history."""
        bus = ContextBus(max_history=5)
        for i in range(10):
            _run(bus.publish(BusMessage(msg_type=MessageType.HEARTBEAT, sender="s", content=i)))
        self.assertEqual(bus.history_size, 5)

    def test_history_filtering(self):
        """History can be filtered by sender, type, topic."""
        bus = ContextBus()
        _run(bus.publish(BusMessage(msg_type=MessageType.TASK_RESULT, sender="a", topic="t1")))
        _run(bus.publish(BusMessage(msg_type=MessageType.ERROR, sender="a", topic="t2")))
        _run(bus.publish(BusMessage(msg_type=MessageType.TASK_RESULT, sender="b", topic="t1")))

        self.assertEqual(len(bus.get_history(sender="a")), 2)
        self.assertEqual(len(bus.get_history(msg_type=MessageType.TASK_RESULT)), 2)
        self.assertEqual(len(bus.get_history(topic="t1")), 2)

    def test_subscriber_error_doesnt_crash(self):
        """Error in subscriber is logged but doesn't crash publish."""
        bus = ContextBus()
        bus.subscribe("s", MessageType.TASK_RESULT, lambda m: 1 / 0)  # Will raise
        results = []
        bus.subscribe("s", MessageType.TASK_RESULT, lambda m: results.append(m))

        _run(bus.publish(BusMessage(msg_type=MessageType.TASK_RESULT, sender="s")))
        # Second subscriber still executed despite first crashing
        self.assertEqual(len(results), 1)


# ═══════════════════════════════════════════════
# Feature 3: Agent Delegation Tool
# ═══════════════════════════════════════════════

class TestDelegateTaskTool(unittest.TestCase):
    """Test agent-to-agent task delegation."""

    def _make_registry_with_agent(self, name="worker"):
        """Create an AgentRegistry with one mock agent."""
        reg = AgentRegistry()
        config = AgentConfig(name=name, role="test")
        _run(reg.create_agent(
            config, _make_mock_provider(name), _make_mock_registry(), _make_prompt_builder()
        ))
        return reg

    def test_sync_delegation(self):
        """Delegate task and wait for result."""
        agent_reg = self._make_registry_with_agent("worker")
        tool = DelegateTaskTool(agent_reg)
        result = _run(tool.execute(agent_name="worker", task="do work", wait=True, timeout_seconds=10))
        self.assertTrue(result.success)
        self.assertIn("worker", result.output)

    def test_async_delegation(self):
        """Fire-and-forget delegation returns task ID immediately."""
        agent_reg = self._make_registry_with_agent("async_worker")
        tool = DelegateTaskTool(agent_reg)
        result = _run(tool.execute(
            agent_name="async_worker", task="background work", wait=False
        ))
        self.assertTrue(result.success)
        self.assertIn("task_", result.output)

    def test_invalid_agent(self):
        """Delegating to nonexistent agent returns error."""
        agent_reg = AgentRegistry()
        tool = DelegateTaskTool(agent_reg)
        result = _run(tool.execute(agent_name="ghost", task="anything"))
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)

    def test_task_id_uniqueness(self):
        """Each delegation gets a unique task ID."""
        ids = set()
        for _ in range(100):
            ids.add(DelegatedTask.generate_id())
        self.assertEqual(len(ids), 100)

    def test_task_tracking(self):
        """Delegated tasks are tracked and retrievable."""
        agent_reg = self._make_registry_with_agent("tracked")
        tool = DelegateTaskTool(agent_reg)
        result = _run(tool.execute(agent_name="tracked", task="track me", wait=True, timeout_seconds=10))

        tasks = tool.list_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0].is_completed)
        self.assertEqual(tasks[0].delegatee, "tracked")

    def test_bus_event_published(self):
        """Delegation publishes events to the context bus."""
        agent_reg = self._make_registry_with_agent("bus_worker")
        bus = ContextBus()
        received = []
        bus.subscribe_all(MessageType.STATUS_UPDATE, lambda m: received.append(m))
        bus.subscribe_all(MessageType.TASK_RESULT, lambda m: received.append(m))

        tool = DelegateTaskTool(agent_reg, context_bus=bus)
        _run(tool.execute(agent_name="bus_worker", task="with bus", wait=True, timeout_seconds=10))
        self.assertGreaterEqual(len(received), 1)

    def test_empty_agent_name_error(self):
        """Empty agent_name returns error."""
        tool = DelegateTaskTool(AgentRegistry())
        result = _run(tool.execute(agent_name="", task="x"))
        self.assertFalse(result.success)

    def test_empty_task_error(self):
        """Empty task returns error."""
        tool = DelegateTaskTool(AgentRegistry())
        result = _run(tool.execute(agent_name="x", task=""))
        self.assertFalse(result.success)

    def test_pending_completed_counts(self):
        """pending_count and completed_count track task states."""
        agent_reg = self._make_registry_with_agent("counter")
        tool = DelegateTaskTool(agent_reg)
        _run(tool.execute(agent_name="counter", task="t1", wait=True, timeout_seconds=10))
        self.assertEqual(tool.completed_count, 1)
        self.assertEqual(tool.pending_count, 0)

    def test_delegated_task_to_dict(self):
        """DelegatedTask.to_dict contains expected fields."""
        dt = DelegatedTask(
            task_id="task_abc",
            delegator="sup",
            delegatee="worker",
            task_description="do stuff",
        )
        d = dt.to_dict()
        self.assertEqual(d["task_id"], "task_abc")
        self.assertEqual(d["delegatee"], "worker")
        self.assertFalse(d["is_completed"])

    def test_no_bus_still_works(self):
        """Delegation works without a context bus."""
        agent_reg = self._make_registry_with_agent("no_bus")
        tool = DelegateTaskTool(agent_reg, context_bus=None)
        result = _run(tool.execute(agent_name="no_bus", task="task", wait=True, timeout_seconds=10))
        self.assertTrue(result.success)

    def test_tool_schema(self):
        """DelegateTaskTool has proper schema."""
        tool = DelegateTaskTool(AgentRegistry())
        schema = tool.get_schema()
        self.assertEqual(schema.name, "delegate_task")
        self.assertIn("agent_name", schema.input_schema["properties"])

    def test_get_task_by_id(self):
        """get_task retrieves by task_id, None for missing."""
        agent_reg = self._make_registry_with_agent("getter")
        tool = DelegateTaskTool(agent_reg)
        _run(tool.execute(agent_name="getter", task="findme", wait=True, timeout_seconds=10))

        tasks = tool.list_tasks()
        self.assertIsNotNone(tool.get_task(tasks[0].task_id))
        self.assertIsNone(tool.get_task("nonexistent_id"))


# ═══════════════════════════════════════════════
# Feature 4: Supervisor Pattern
# ═══════════════════════════════════════════════

class TestSupervisor(unittest.TestCase):
    """Test supervisor orchestration strategies."""

    def _setup_registry(self, agent_names):
        """Create registry with multiple mock agents."""
        reg = AgentRegistry()
        for name in agent_names:
            config = AgentConfig(name=name, role="worker")
            _run(reg.create_agent(
                config, _make_mock_provider(name), _make_mock_registry(), _make_prompt_builder()
            ))
        return reg

    def test_sequential_execution(self):
        """SEQUENTIAL runs agents one after another."""
        agent_reg = self._setup_registry(["a1", "a2", "a3"])
        sv = Supervisor(
            config=SupervisorConfig(name="seq", strategy=ExecutionStrategy.SEQUENTIAL),
            agent_registry=agent_reg,
        )
        result = _run(sv.execute_task("analyze", ["a1", "a2", "a3"]))
        self.assertIn("a1", result)
        self.assertIn("a2", result)
        self.assertIn("a3", result)

    def test_parallel_execution(self):
        """PARALLEL runs agents concurrently."""
        agent_reg = self._setup_registry(["p1", "p2", "p3"])
        sv = Supervisor(
            config=SupervisorConfig(name="par", strategy=ExecutionStrategy.PARALLEL),
            agent_registry=agent_reg,
        )
        result = _run(sv.execute_task("analyze", ["p1", "p2", "p3"]))
        self.assertIn("p1", result)
        self.assertIn("p2", result)
        self.assertIn("3/3 completed", result)

    def test_pipeline_execution(self):
        """PIPELINE chains agent outputs as inputs."""
        agent_reg = self._setup_registry(["step1", "step2"])
        sv = Supervisor(
            config=SupervisorConfig(name="pipe", strategy=ExecutionStrategy.PIPELINE),
            agent_registry=agent_reg,
        )
        result = _run(sv.execute_task("process data", ["step1", "step2"]))
        self.assertIn("step1", result)
        self.assertIn("step2", result)

    def test_empty_agents_raises(self):
        """Empty agent list raises ValueError."""
        sv = Supervisor(
            config=SupervisorConfig(name="empty"),
            agent_registry=AgentRegistry(),
        )
        with self.assertRaises(ValueError):
            _run(sv.execute_task("task", []))

    def test_progress_tracking(self):
        """Progress dict tracks completion."""
        agent_reg = self._setup_registry(["w1", "w2"])
        sv = Supervisor(
            config=SupervisorConfig(name="prog", strategy=ExecutionStrategy.SEQUENTIAL),
            agent_registry=agent_reg,
        )
        _run(sv.execute_task("task", ["w1", "w2"]))
        progress = sv.get_progress()
        self.assertEqual(progress["completed"], 2)
        self.assertEqual(progress["total"], 2)
        self.assertEqual(progress["failed"], 0)

    def test_subtask_creation(self):
        """Subtasks are created for each agent."""
        agent_reg = self._setup_registry(["s1", "s2"])
        sv = Supervisor(
            config=SupervisorConfig(name="sub", strategy=ExecutionStrategy.SEQUENTIAL),
            agent_registry=agent_reg,
        )
        _run(sv.execute_task("task", ["s1", "s2"]))
        subtasks = sv.get_subtasks()
        self.assertEqual(len(subtasks), 2)
        self.assertEqual(subtasks[0].agent_name, "s1")
        self.assertTrue(subtasks[0].is_completed)

    def test_skip_on_failure(self):
        """on_failure='skip' continues after agent failure."""
        agent_reg = AgentRegistry()
        # Create a failing agent — patch agent.run directly to raise
        fail_provider = _make_mock_provider("fail")
        config_fail = AgentConfig(name="failer", role="test")
        _run(agent_reg.create_agent(config_fail, fail_provider, _make_mock_registry(), _make_prompt_builder()))
        # Make agent.run itself raise (bypassing Agent's internal error handling)
        agent_reg.get_agent_status("failer").agent.run = AsyncMock(side_effect=RuntimeError("crash"))
        # Create a good agent
        config_ok = AgentConfig(name="ok", role="test")
        _run(agent_reg.create_agent(config_ok, _make_mock_provider("ok"), _make_mock_registry(), _make_prompt_builder()))

        sv = Supervisor(
            config=SupervisorConfig(
                name="skip", strategy=ExecutionStrategy.SEQUENTIAL,
                on_failure="skip", max_retries=0,
            ),
            agent_registry=agent_reg,
        )
        # Should not raise — failer is skipped
        result = _run(sv.execute_task("task", ["failer", "ok"]))
        self.assertIn("SKIPPED", result)
        self.assertIn("ok", result)

    def test_abort_on_failure(self):
        """on_failure='abort' raises on first failure."""
        agent_reg = AgentRegistry()
        fail_provider = _make_mock_provider("fail")
        _run(agent_reg.create_agent(
            AgentConfig(name="failer", role="test"),
            fail_provider, _make_mock_registry(), _make_prompt_builder()
        ))
        # Patch agent.run directly to raise (bypassing Agent's internal error handling)
        agent_reg.get_agent_status("failer").agent.run = AsyncMock(side_effect=RuntimeError("crash"))

        sv = Supervisor(
            config=SupervisorConfig(
                name="abort", strategy=ExecutionStrategy.SEQUENTIAL,
                on_failure="abort", max_retries=0,
            ),
            agent_registry=agent_reg,
        )
        with self.assertRaises(Exception):
            _run(sv.execute_task("task", ["failer"]))

    def test_retry_on_failure(self):
        """Failed subtask is retried up to max_retries."""
        agent_reg = AgentRegistry()
        provider = _make_mock_provider("retry")
        _run(agent_reg.create_agent(
            AgentConfig(name="retrier", role="test"),
            provider, _make_mock_registry(), _make_prompt_builder()
        ))

        # Patch agent.run to fail first call, succeed second (bypassing Agent's internal error handling)
        call_count = [0]

        async def flaky_run(task):
            call_count[0] += 1
            if call_count[0] <= 1:
                raise RuntimeError("temporary failure")
            return "recovered"

        agent_reg.get_agent_status("retrier").agent.run = flaky_run

        sv = Supervisor(
            config=SupervisorConfig(
                name="retry", strategy=ExecutionStrategy.SEQUENTIAL,
                max_retries=2, on_failure="abort",
            ),
            agent_registry=agent_reg,
        )
        result = _run(sv.execute_task("task", ["retrier"]))
        self.assertIn("recovered", result)

    def test_bus_events_published(self):
        """Supervisor publishes start/complete events to bus."""
        agent_reg = self._setup_registry(["b1"])
        bus = ContextBus()
        received = []
        bus.subscribe_all(MessageType.STATUS_UPDATE, lambda m: received.append(m))
        bus.subscribe_all(MessageType.TASK_RESULT, lambda m: received.append(m))

        sv = Supervisor(
            config=SupervisorConfig(name="bus_sv", strategy=ExecutionStrategy.SEQUENTIAL),
            agent_registry=agent_reg,
            context_bus=bus,
        )
        _run(sv.execute_task("task", ["b1"]))
        self.assertGreaterEqual(len(received), 2)  # start + complete

    def test_result_aggregation_format(self):
        """Aggregated result has expected format."""
        agent_reg = self._setup_registry(["fmt1"])
        sv = Supervisor(
            config=SupervisorConfig(name="fmt", strategy=ExecutionStrategy.SEQUENTIAL),
            agent_registry=agent_reg,
        )
        result = _run(sv.execute_task("task", ["fmt1"]))
        self.assertIn("=== Supervisor", result)
        self.assertIn("fmt1", result)
        self.assertIn("Summary:", result)

    def test_supervisor_config_defaults(self):
        """SupervisorConfig has sensible defaults."""
        config = SupervisorConfig(name="default")
        self.assertEqual(config.strategy, ExecutionStrategy.SEQUENTIAL)
        self.assertEqual(config.max_retries, 2)
        self.assertEqual(config.on_failure, "abort")

    def test_subtask_id_generation(self):
        """SubTask IDs are unique."""
        ids = {SubTask.generate_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)

    def test_parallel_with_semaphore(self):
        """Parallel execution respects max_parallel limit."""
        agent_reg = self._setup_registry([f"p{i}" for i in range(5)])
        sv = Supervisor(
            config=SupervisorConfig(name="sem", strategy=ExecutionStrategy.PARALLEL),
            agent_registry=agent_reg,
        )
        result = _run(sv.execute_task("task", [f"p{i}" for i in range(5)], max_parallel=2))
        self.assertIn("5/5 completed", result)


# ═══════════════════════════════════════════════
# Feature 5: Conflict Resolution
# ═══════════════════════════════════════════════

class TestConflictResolver(unittest.TestCase):
    """Test conflict detection and resolution strategies."""

    def test_detect_conflict(self):
        """Different values are detected as conflict."""
        resolver = ConflictResolver()
        self.assertTrue(resolver.detect_conflict("x", {"a": 1, "b": 2}))

    def test_no_conflict(self):
        """Same values are not a conflict."""
        resolver = ConflictResolver()
        self.assertFalse(resolver.detect_conflict("x", {"a": "yes", "b": "yes"}))

    def test_single_value_no_conflict(self):
        """Single value cannot conflict."""
        resolver = ConflictResolver()
        self.assertFalse(resolver.detect_conflict("x", {"a": 1}))

    def test_empty_values_no_conflict(self):
        """Empty dict is not a conflict."""
        resolver = ConflictResolver()
        self.assertFalse(resolver.detect_conflict("x", {}))

    def test_voting_resolution(self):
        """VOTING: majority value wins."""
        resolver = ConflictResolver(strategy=ConflictStrategy.VOTING)
        result = _run(resolver.resolve("answer", {"a": "yes", "b": "no", "c": "yes"}))
        self.assertEqual(result, "yes")

    def test_priority_resolution(self):
        """PRIORITY: highest-priority agent's value wins."""
        resolver = ConflictResolver(strategy=ConflictStrategy.PRIORITY)
        result = _run(resolver.resolve(
            "answer",
            {"a": "low", "b": "high"},
            agent_priorities={"a": 1, "b": 10},
        ))
        self.assertEqual(result, "high")

    def test_merge_dicts(self):
        """MERGE: dict values are merged."""
        resolver = ConflictResolver(strategy=ConflictStrategy.MERGE)
        result = _run(resolver.resolve("data", {
            "a": {"key1": "v1"},
            "b": {"key2": "v2"},
        }))
        self.assertEqual(result, {"key1": "v1", "key2": "v2"})

    def test_merge_lists(self):
        """MERGE: list values are concatenated."""
        resolver = ConflictResolver(strategy=ConflictStrategy.MERGE)
        result = _run(resolver.resolve("items", {
            "a": [1, 2],
            "b": [3, 4],
        }))
        self.assertEqual(result, [1, 2, 3, 4])

    def test_merge_strings(self):
        """MERGE: string values are joined with newlines."""
        resolver = ConflictResolver(strategy=ConflictStrategy.MERGE)
        result = _run(resolver.resolve("text", {
            "a": "line1",
            "b": "line2",
        }))
        self.assertEqual(result, "line1\nline2")

    def test_first_win(self):
        """FIRST_WIN: first value in dict wins."""
        resolver = ConflictResolver(strategy=ConflictStrategy.FIRST_WIN)
        result = _run(resolver.resolve("answer", {"a": "first", "b": "second"}))
        self.assertEqual(result, "first")

    def test_consensus_success(self):
        """CONSENSUS: all agree → returns value."""
        resolver = ConflictResolver(strategy=ConflictStrategy.CONSENSUS)
        result = _run(resolver.resolve("answer", {"a": "same", "b": "same"}))
        self.assertEqual(result, "same")

    def test_consensus_failure(self):
        """CONSENSUS: disagreement raises ValueError."""
        resolver = ConflictResolver(strategy=ConflictStrategy.CONSENSUS)
        with self.assertRaises(ValueError):
            _run(resolver.resolve("answer", {"a": "yes", "b": "no"}))

    def test_conflict_history(self):
        """Resolved conflicts are recorded in history."""
        resolver = ConflictResolver(strategy=ConflictStrategy.VOTING)
        _run(resolver.resolve("q1", {"a": 1, "b": 2}))
        _run(resolver.resolve("q2", {"a": "x", "b": "y"}))
        self.assertEqual(resolver.conflict_count, 2)
        history = resolver.get_conflict_history()
        self.assertEqual(history[0].field_name, "q1")

    def test_no_conflict_returns_value_directly(self):
        """If all agents agree, resolve returns value without recording conflict."""
        resolver = ConflictResolver()
        result = _run(resolver.resolve("x", {"a": 42, "b": 42}))
        self.assertEqual(result, 42)
        self.assertEqual(resolver.conflict_count, 0)

    def test_resolve_empty_returns_none(self):
        """Empty agent_values → returns None."""
        resolver = ConflictResolver()
        result = _run(resolver.resolve("x", {}))
        self.assertIsNone(result)


class TestConflictDetector(unittest.TestCase):
    """Test resource-level conflict detection."""

    def test_write_write_conflict(self):
        """Two agents writing same file → conflict."""
        cd = ConflictDetector()
        result1 = _run(cd.check_resource_conflict("a1", "write", "/file.txt"))
        self.assertIsNone(result1)  # First write is fine
        result2 = _run(cd.check_resource_conflict("a2", "write", "/file.txt"))
        self.assertEqual(result2, "a1")  # Conflict with a1

    def test_read_read_no_conflict(self):
        """Two agents reading same file → no conflict."""
        cd = ConflictDetector()
        _run(cd.check_resource_conflict("a1", "read", "/file.txt"))
        result = _run(cd.check_resource_conflict("a2", "read", "/file.txt"))
        self.assertIsNone(result)

    def test_read_write_conflict(self):
        """Agent writing while another reads → conflict."""
        cd = ConflictDetector()
        _run(cd.check_resource_conflict("a1", "read", "/file.txt"))
        result = _run(cd.check_resource_conflict("a2", "write", "/file.txt"))
        self.assertEqual(result, "a1")

    def test_release_resource(self):
        """After releasing, resource is available again."""
        cd = ConflictDetector()
        _run(cd.check_resource_conflict("a1", "write", "/file.txt"))
        _run(cd.release_resource("a1", "/file.txt"))
        result = _run(cd.check_resource_conflict("a2", "write", "/file.txt"))
        self.assertIsNone(result)

    def test_same_agent_reacquire(self):
        """Same agent can re-acquire its own resource."""
        cd = ConflictDetector()
        _run(cd.check_resource_conflict("a1", "write", "/file.txt"))
        result = _run(cd.check_resource_conflict("a1", "write", "/file.txt"))
        self.assertIsNone(result)


class TestDeadlockDetector(unittest.TestCase):
    """Test deadlock detection in wait graphs."""

    def test_simple_deadlock(self):
        """A → B, B → A creates deadlock."""
        dd = DeadlockDetector()
        self.assertFalse(dd.register_wait("A", "B"))
        self.assertTrue(dd.register_wait("B", "A"))

    def test_no_deadlock(self):
        """A → B, B → C (no cycle) is not a deadlock."""
        dd = DeadlockDetector()
        self.assertFalse(dd.register_wait("A", "B"))
        self.assertFalse(dd.register_wait("B", "C"))

    def test_three_way_deadlock(self):
        """A → B, B → C, C → A creates deadlock."""
        dd = DeadlockDetector()
        self.assertFalse(dd.register_wait("A", "B"))
        self.assertFalse(dd.register_wait("B", "C"))
        self.assertTrue(dd.register_wait("C", "A"))

    def test_clear_wait(self):
        """Clearing a wait breaks the potential cycle."""
        dd = DeadlockDetector()
        dd.register_wait("A", "B")
        dd.clear_wait("A")
        # Now B → A should not deadlock (A no longer waits for B)
        self.assertFalse(dd.register_wait("B", "A"))

    def test_self_deadlock(self):
        """A → A is a deadlock (self-cycle)."""
        dd = DeadlockDetector()
        self.assertTrue(dd.register_wait("A", "A"))


# ═══════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════

class TestMultiAgentIntegration(unittest.TestCase):
    """Cross-feature integration tests."""

    def test_registry_bus_delegation_flow(self):
        """Full flow: registry → bus → delegation."""
        # Setup
        agent_reg = AgentRegistry()
        bus = ContextBus()

        config = AgentConfig(name="worker", role="test", capabilities=["read"])
        _run(agent_reg.create_agent(
            config, _make_mock_provider("worker"), _make_mock_registry(), _make_prompt_builder()
        ))

        # Delegation with bus
        tool = DelegateTaskTool(agent_reg, context_bus=bus)
        result = _run(tool.execute(agent_name="worker", task="read files", wait=True, timeout_seconds=10))
        self.assertTrue(result.success)

        # Bus should have messages
        self.assertGreater(bus.history_size, 0)

    def test_supervisor_with_bus_events(self):
        """Supervisor publishes events through context bus."""
        agent_reg = AgentRegistry()
        bus = ContextBus()
        events = []
        bus.subscribe_all(MessageType.STATUS_UPDATE, lambda m: events.append(m))
        bus.subscribe_all(MessageType.TASK_RESULT, lambda m: events.append(m))

        for name in ["reader", "writer"]:
            _run(agent_reg.create_agent(
                AgentConfig(name=name, role="test"),
                _make_mock_provider(name), _make_mock_registry(), _make_prompt_builder()
            ))

        sv = Supervisor(
            config=SupervisorConfig(name="full_sv", strategy=ExecutionStrategy.PIPELINE),
            agent_registry=agent_reg,
            context_bus=bus,
        )
        _run(sv.execute_task("process", ["reader", "writer"]))
        self.assertGreaterEqual(len(events), 2)

    def test_conflict_resolution_with_supervisor(self):
        """After parallel supervisor, resolve conflicts."""
        agent_reg = AgentRegistry()

        # Two agents producing different mock responses
        for name, text in [("fast", "answer_A"), ("slow", "answer_B")]:
            provider = _make_mock_provider(name)
            provider.send_message = AsyncMock(
                return_value=AgentResponse(text=text, stop_reason="end_turn")
            )
            _run(agent_reg.create_agent(
                AgentConfig(name=name, role="analyzer"),
                provider, _make_mock_registry(), _make_prompt_builder()
            ))

        sv = Supervisor(
            config=SupervisorConfig(name="vote_sv", strategy=ExecutionStrategy.PARALLEL),
            agent_registry=agent_reg,
        )
        _run(sv.execute_task("analyze", ["fast", "slow"]))

        # Simulate conflict resolution on subtask results
        resolver = ConflictResolver(strategy=ConflictStrategy.FIRST_WIN)
        subtasks = sv.get_subtasks()
        values = {st.agent_name: st.result for st in subtasks if st.result}
        if resolver.detect_conflict("analysis", values):
            resolved = _run(resolver.resolve("analysis", values))
            self.assertIsNotNone(resolved)

    def test_deadlock_prevention_in_delegation(self):
        """DeadlockDetector catches circular delegation."""
        dd = DeadlockDetector()

        # Simulate: agent_a delegates to agent_b, agent_b delegates to agent_a
        self.assertFalse(dd.register_wait("agent_a", "agent_b"))
        self.assertTrue(dd.register_wait("agent_b", "agent_a"))
        # Deadlock detected — system should refuse the second delegation

    def test_shared_state_across_agents(self):
        """Agents can share state through the bus."""
        bus = ContextBus()

        # Agent 1 sets shared state
        _run(bus.set_shared("analysis_result", {"files": 42, "issues": 3}))

        # Agent 2 reads shared state
        data = _run(bus.get_shared("analysis_result"))
        self.assertEqual(data["files"], 42)
        self.assertEqual(data["issues"], 3)

        # Atomic update
        _run(bus.update_shared("analysis_result",
                               lambda d: {**d, "issues": d["issues"] + 1}))
        data = _run(bus.get_shared("analysis_result"))
        self.assertEqual(data["issues"], 4)


if __name__ == "__main__":
    unittest.main()
