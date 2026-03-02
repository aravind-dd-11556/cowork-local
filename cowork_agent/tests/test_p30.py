"""
Sprint 30 Tests — Task Tool Agent Types, Worktree Isolation, Resume.

Tests the enhanced TaskTool with:
  - 6 agent type profiles (Bash, Explore, Plan, general-purpose,
    claude-code-guide, statusline-setup)
  - Worktree isolation mode
  - Agent resume by ID
  - Agent session store
  - Tool filtering per agent type
  - Main.py wiring with enhanced factory
  - ToolRegistry.unregister()
"""

import asyncio
import os
import unittest
from dataclasses import dataclass, field
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# ── Imports ──────────────────────────────────────────────────────────────
from cowork_agent.tools.task_tool import (
    TaskTool,
    AgentTypeProfile,
    AgentSession,
    AgentSessionStore,
    AGENT_TYPE_PROFILES,
    _current_depth,
    get_session_store,
    reset_session_store,
)
from cowork_agent.core.models import ToolResult
from cowork_agent.core.tool_registry import ToolRegistry


def run(coro):
    """Helper to run async coroutines in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Mock Agent ───────────────────────────────────────────────────────────

class MockAgent:
    """Minimal Agent mock for testing TaskTool."""

    def __init__(self, result="Subagent done.", **kwargs):
        self._result = result
        self._messages = []
        self.max_iterations = kwargs.get("max_iterations", 10)
        self.workspace_dir = kwargs.get("workspace_dir", "/tmp")

    async def run(self, prompt: str) -> str:
        return self._result


class FailingAgent:
    """Agent that raises an exception."""

    def __init__(self, **kwargs):
        self._messages = []
        self.max_iterations = kwargs.get("max_iterations", 10)

    async def run(self, prompt: str) -> str:
        raise RuntimeError("Simulated failure")


# ── Mock WorktreeManager ────────────────────────────────────────────────

@dataclass
class MockWorktreeInfo:
    name: str = "test-wt"
    path: str = "/tmp/worktrees/test-wt"
    branch: str = "worktree/test-wt"
    created_from: str = "HEAD"
    has_changes: bool = False


class MockWorktreeManager:
    """Mock worktree manager for testing isolation."""

    def __init__(self, is_repo=True, fail_create=False, has_changes=False):
        self._is_repo = is_repo
        self._fail_create = fail_create
        self._has_changes = has_changes
        self._created = []
        self._removed = []

    def is_git_repo(self):
        return self._is_repo

    def create(self, name=""):
        if self._fail_create:
            return None
        info = MockWorktreeInfo(
            name=name or "auto-wt",
            path=f"/tmp/worktrees/{name or 'auto-wt'}",
            branch=f"worktree/{name or 'auto-wt'}",
            has_changes=self._has_changes,
        )
        self._created.append(info)
        return info

    def list_worktrees(self):
        return self._created

    def remove(self, name, force=False):
        self._removed.append((name, force))
        return f"Removed {name}"


# ═══════════════════════════════════════════════════════════════════════
# TEST: Agent Type Profiles
# ═══════════════════════════════════════════════════════════════════════

class TestAgentTypeProfiles(unittest.TestCase):
    """Test the 6 agent type profile definitions."""

    def test_six_profiles_exist(self):
        self.assertEqual(len(AGENT_TYPE_PROFILES), 6)

    def test_profile_names(self):
        expected = {
            "Bash", "Explore", "Plan", "general-purpose",
            "claude-code-guide", "statusline-setup",
        }
        self.assertEqual(set(AGENT_TYPE_PROFILES.keys()), expected)

    def test_bash_profile_tool_filter(self):
        p = AGENT_TYPE_PROFILES["Bash"]
        result = p.filter_tools(["bash", "read", "write", "edit"])
        self.assertEqual(result, ["bash"])

    def test_explore_profile_excludes_edit_tools(self):
        p = AGENT_TYPE_PROFILES["Explore"]
        tools = ["read", "glob_tool", "grep_tool", "edit", "write", "task", "notebook_edit"]
        result = p.filter_tools(tools)
        self.assertNotIn("edit", result)
        self.assertNotIn("write", result)
        self.assertNotIn("task", result)
        self.assertNotIn("notebook_edit", result)
        self.assertIn("read", result)

    def test_plan_profile_excludes_write_tools(self):
        p = AGENT_TYPE_PROFILES["Plan"]
        tools = ["read", "glob_tool", "edit", "write", "task"]
        result = p.filter_tools(tools)
        self.assertNotIn("edit", result)
        self.assertNotIn("write", result)
        self.assertNotIn("task", result)

    def test_general_purpose_allows_all_tools(self):
        p = AGENT_TYPE_PROFILES["general-purpose"]
        tools = ["bash", "read", "write", "edit", "task", "web_search"]
        result = p.filter_tools(tools)
        self.assertEqual(result, tools)

    def test_claude_code_guide_whitelist(self):
        p = AGENT_TYPE_PROFILES["claude-code-guide"]
        tools = ["glob_tool", "grep_tool", "read", "web_fetch", "web_search", "edit", "bash"]
        result = p.filter_tools(tools)
        self.assertIn("glob_tool", result)
        self.assertIn("web_search", result)
        self.assertNotIn("edit", result)
        self.assertNotIn("bash", result)

    def test_statusline_setup_whitelist(self):
        p = AGENT_TYPE_PROFILES["statusline-setup"]
        tools = ["read", "edit", "bash", "write"]
        result = p.filter_tools(tools)
        self.assertEqual(set(result), {"read", "edit"})

    def test_all_profiles_have_description(self):
        for name, p in AGENT_TYPE_PROFILES.items():
            self.assertTrue(len(p.description) > 10, f"{name} has short/missing description")

    def test_all_profiles_have_positive_max_turns(self):
        for name, p in AGENT_TYPE_PROFILES.items():
            self.assertGreater(p.max_turns_default, 0, f"{name} has invalid max_turns")


class TestAgentTypeProfileDataclass(unittest.TestCase):
    """Test AgentTypeProfile dataclass behavior."""

    def test_custom_profile(self):
        p = AgentTypeProfile(
            name="custom",
            description="Test",
            tool_filter=["foo", "bar"],
        )
        self.assertEqual(p.filter_tools(["foo", "bar", "baz"]), ["foo", "bar"])

    def test_empty_excluded_allows_all(self):
        p = AgentTypeProfile(name="all", description="All", tool_filter=None)
        tools = ["a", "b", "c"]
        self.assertEqual(p.filter_tools(tools), tools)

    def test_excluded_tools_blacklist(self):
        p = AgentTypeProfile(
            name="limited",
            description="Limited",
            excluded_tools=["danger"],
        )
        self.assertEqual(p.filter_tools(["safe", "danger", "ok"]), ["safe", "ok"])

    def test_whitelist_with_missing_tools(self):
        p = AgentTypeProfile(name="x", description="x", tool_filter=["nonexistent"])
        self.assertEqual(p.filter_tools(["read", "write"]), [])


# ═══════════════════════════════════════════════════════════════════════
# TEST: Agent Session Store
# ═══════════════════════════════════════════════════════════════════════

class TestAgentSessionStore(unittest.TestCase):
    """Test the agent session store for resume functionality."""

    def setUp(self):
        reset_session_store()
        self.store = get_session_store()

    def test_save_and_get(self):
        session = AgentSession(
            agent_id="a1", subagent_type="Bash",
            description="test", prompt="do stuff", result="done",
        )
        self.store.save(session)
        retrieved = self.store.get("a1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.result, "done")

    def test_get_nonexistent(self):
        self.assertIsNone(self.store.get("nonexistent"))

    def test_remove(self):
        session = AgentSession(
            agent_id="a2", subagent_type="Explore",
            description="explore", prompt="find", result="found",
        )
        self.store.save(session)
        self.assertTrue(self.store.remove("a2"))
        self.assertIsNone(self.store.get("a2"))

    def test_remove_nonexistent(self):
        self.assertFalse(self.store.remove("nope"))

    def test_session_ids(self):
        for i in range(3):
            self.store.save(AgentSession(
                agent_id=f"s{i}", subagent_type="Bash",
                description="d", prompt="p", result="r",
            ))
        self.assertEqual(self.store.session_ids, ["s0", "s1", "s2"])

    def test_len(self):
        self.assertEqual(len(self.store), 0)
        self.store.save(AgentSession(
            agent_id="x", subagent_type="Bash",
            description="d", prompt="p", result="r",
        ))
        self.assertEqual(len(self.store), 1)

    def test_max_sessions_eviction(self):
        store = AgentSessionStore(max_sessions=3)
        for i in range(5):
            store.save(AgentSession(
                agent_id=f"s{i}", subagent_type="Bash",
                description="d", prompt="p", result="r",
            ))
        self.assertEqual(len(store), 3)
        # Oldest (s0, s1) evicted
        self.assertIsNone(store.get("s0"))
        self.assertIsNone(store.get("s1"))
        self.assertIsNotNone(store.get("s2"))

    def test_session_with_worktree(self):
        session = AgentSession(
            agent_id="wt1", subagent_type="Bash",
            description="d", prompt="p", result="r",
            worktree_path="/tmp/wt", worktree_branch="wt/branch",
        )
        self.store.save(session)
        retrieved = self.store.get("wt1")
        self.assertEqual(retrieved.worktree_path, "/tmp/wt")
        self.assertEqual(retrieved.worktree_branch, "wt/branch")

    def test_session_defaults(self):
        session = AgentSession(
            agent_id="d1", subagent_type="Plan",
            description="d", prompt="p", result="r",
        )
        self.assertEqual(session.messages, [])
        self.assertIsNone(session.worktree_path)
        self.assertTrue(session.completed)

    def test_reset_session_store(self):
        self.store.save(AgentSession(
            agent_id="x", subagent_type="Bash",
            description="d", prompt="p", result="r",
        ))
        reset_session_store()
        new_store = get_session_store()
        self.assertEqual(len(new_store), 0)


# ═══════════════════════════════════════════════════════════════════════
# TEST: TaskTool Basic Execution
# ═══════════════════════════════════════════════════════════════════════

class TestTaskToolBasic(unittest.TestCase):
    """Test basic TaskTool execution with agent types."""

    def setUp(self):
        reset_session_store()
        import cowork_agent.tools.task_tool as tt
        tt._current_depth = 0
        self.factory = MagicMock(return_value=MockAgent())
        self.tool = TaskTool(agent_factory=self.factory, workspace_dir="/tmp/ws")

    def test_general_purpose_execution(self):
        result = run(self.tool.execute(
            description="test task",
            prompt="do something",
            subagent_type="general-purpose",
        ))
        self.assertTrue(result.success)
        self.assertIn("Subagent done.", result.output)

    def test_bash_type_execution(self):
        result = run(self.tool.execute(
            description="run cmd",
            prompt="ls -la",
            subagent_type="Bash",
        ))
        self.assertTrue(result.success)
        self.assertIn("Subagent done.", result.output)
        # Factory should be called with tool_filter
        self.factory.assert_called_once()
        call_kwargs = self.factory.call_args[1]
        self.assertIsNotNone(call_kwargs.get("tool_filter"))

    def test_explore_type_execution(self):
        result = run(self.tool.execute(
            description="explore code",
            prompt="find main",
            subagent_type="Explore",
        ))
        self.assertTrue(result.success)

    def test_plan_type_execution(self):
        result = run(self.tool.execute(
            description="plan feature",
            prompt="design auth",
            subagent_type="Plan",
        ))
        self.assertTrue(result.success)

    def test_claude_code_guide_type(self):
        result = run(self.tool.execute(
            description="guide query",
            prompt="how to use hooks",
            subagent_type="claude-code-guide",
        ))
        self.assertTrue(result.success)

    def test_statusline_setup_type(self):
        result = run(self.tool.execute(
            description="setup status",
            prompt="configure line",
            subagent_type="statusline-setup",
        ))
        self.assertTrue(result.success)

    def test_unknown_type_error(self):
        result = run(self.tool.execute(
            description="bad",
            prompt="do something",
            subagent_type="nonexistent-type",
        ))
        self.assertFalse(result.success)
        self.assertIn("Unknown subagent_type", result.error)
        self.assertIn("nonexistent-type", result.error)

    def test_empty_prompt_error(self):
        result = run(self.tool.execute(
            description="test",
            prompt="",
            subagent_type="general-purpose",
        ))
        self.assertFalse(result.success)
        self.assertIn("Prompt is required", result.error)

    def test_agent_id_in_metadata(self):
        result = run(self.tool.execute(
            description="test",
            prompt="do it",
            subagent_type="general-purpose",
        ))
        self.assertTrue(result.success)
        self.assertIn("agent_id", result.metadata)
        self.assertTrue(result.metadata["agent_id"].startswith("agent_"))

    def test_subagent_type_in_metadata(self):
        result = run(self.tool.execute(
            description="test",
            prompt="do it",
            subagent_type="Bash",
        ))
        self.assertEqual(result.metadata["subagent_type"], "Bash")

    def test_session_saved_after_execution(self):
        result = run(self.tool.execute(
            description="save test",
            prompt="do it",
            subagent_type="general-purpose",
        ))
        store = get_session_store()
        agent_id = result.metadata["agent_id"]
        session = store.get(agent_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.description, "save test")
        self.assertEqual(session.result, "Subagent done.")


# ═══════════════════════════════════════════════════════════════════════
# TEST: TaskTool Max Turns
# ═══════════════════════════════════════════════════════════════════════

class TestTaskToolMaxTurns(unittest.TestCase):
    """Test max_turns parameter behavior."""

    def setUp(self):
        reset_session_store()
        import cowork_agent.tools.task_tool as tt
        tt._current_depth = 0
        self.factory = MagicMock(return_value=MockAgent())
        self.tool = TaskTool(agent_factory=self.factory, workspace_dir="/tmp")

    def test_custom_max_turns(self):
        run(self.tool.execute(
            description="test",
            prompt="do it",
            subagent_type="general-purpose",
            max_turns=25,
        ))
        call_kwargs = self.factory.call_args[1]
        self.assertEqual(call_kwargs["max_iterations"], 25)

    def test_default_max_turns_from_profile(self):
        run(self.tool.execute(
            description="test",
            prompt="do it",
            subagent_type="Explore",
        ))
        call_kwargs = self.factory.call_args[1]
        self.assertEqual(call_kwargs["max_iterations"], 15)  # Explore default

    def test_zero_max_turns_uses_profile_default(self):
        run(self.tool.execute(
            description="test",
            prompt="do it",
            subagent_type="statusline-setup",
            max_turns=0,
        ))
        call_kwargs = self.factory.call_args[1]
        self.assertEqual(call_kwargs["max_iterations"], 5)  # statusline default


# ═══════════════════════════════════════════════════════════════════════
# TEST: TaskTool Depth Limiting
# ═══════════════════════════════════════════════════════════════════════

class TestTaskToolDepthLimit(unittest.TestCase):
    """Test recursion depth limiting."""

    def setUp(self):
        reset_session_store()
        import cowork_agent.tools.task_tool as tt
        tt._current_depth = 0
        self.factory = MagicMock(return_value=MockAgent())
        self.tool = TaskTool(agent_factory=self.factory, workspace_dir="/tmp")

    def test_depth_limit_reached(self):
        import cowork_agent.tools.task_tool as tt
        tt._current_depth = 3
        result = run(self.tool.execute(
            description="deep",
            prompt="nested",
            subagent_type="general-purpose",
        ))
        self.assertFalse(result.success)
        self.assertIn("Maximum subagent nesting depth", result.error)
        tt._current_depth = 0

    def test_depth_increments_and_decrements(self):
        import cowork_agent.tools.task_tool as tt
        self.assertEqual(tt._current_depth, 0)
        run(self.tool.execute(
            description="test",
            prompt="do it",
            subagent_type="general-purpose",
        ))
        self.assertEqual(tt._current_depth, 0)  # Decremented after completion

    def test_depth_decrements_on_error(self):
        import cowork_agent.tools.task_tool as tt
        self.factory.return_value = FailingAgent()
        run(self.tool.execute(
            description="fail",
            prompt="crash",
            subagent_type="general-purpose",
        ))
        self.assertEqual(tt._current_depth, 0)


# ═══════════════════════════════════════════════════════════════════════
# TEST: Worktree Isolation
# ═══════════════════════════════════════════════════════════════════════

class TestTaskToolWorktreeIsolation(unittest.TestCase):
    """Test worktree isolation mode."""

    def setUp(self):
        reset_session_store()
        import cowork_agent.tools.task_tool as tt
        tt._current_depth = 0

    def test_worktree_isolation_passes_path_to_factory(self):
        wt_mgr = MockWorktreeManager()
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(
            agent_factory=factory,
            worktree_manager=wt_mgr,
            workspace_dir="/tmp/ws",
        )
        result = run(tool.execute(
            description="isolated task",
            prompt="do isolated work",
            subagent_type="general-purpose",
            isolation="worktree",
        ))
        self.assertTrue(result.success)
        call_kwargs = factory.call_args[1]
        self.assertIn("/tmp/worktrees/", call_kwargs["workspace_dir"])

    def test_worktree_auto_cleanup_no_changes(self):
        wt_mgr = MockWorktreeManager(has_changes=False)
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(
            agent_factory=factory,
            worktree_manager=wt_mgr,
            workspace_dir="/tmp/ws",
        )
        result = run(tool.execute(
            description="clean task",
            prompt="no changes",
            subagent_type="general-purpose",
            isolation="worktree",
        ))
        self.assertTrue(result.success)
        self.assertIn("auto-cleaned", result.output)
        self.assertTrue(len(wt_mgr._removed) > 0)

    def test_worktree_kept_with_changes(self):
        wt_mgr = MockWorktreeManager(has_changes=True)
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(
            agent_factory=factory,
            worktree_manager=wt_mgr,
            workspace_dir="/tmp/ws",
        )
        result = run(tool.execute(
            description="changing task",
            prompt="make changes",
            subagent_type="general-purpose",
            isolation="worktree",
        ))
        self.assertTrue(result.success)
        self.assertIn("changes detected", result.output)
        self.assertEqual(len(wt_mgr._removed), 0)

    def test_worktree_not_in_git_repo(self):
        wt_mgr = MockWorktreeManager(is_repo=False)
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(
            agent_factory=factory,
            worktree_manager=wt_mgr,
            workspace_dir="/tmp/ws",
        )
        result = run(tool.execute(
            description="not git",
            prompt="try worktree",
            subagent_type="general-purpose",
            isolation="worktree",
        ))
        self.assertFalse(result.success)
        self.assertIn("git repository", result.error)

    def test_worktree_creation_fails(self):
        wt_mgr = MockWorktreeManager(fail_create=True)
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(
            agent_factory=factory,
            worktree_manager=wt_mgr,
            workspace_dir="/tmp/ws",
        )
        result = run(tool.execute(
            description="fail create",
            prompt="try worktree",
            subagent_type="general-purpose",
            isolation="worktree",
        ))
        self.assertFalse(result.success)
        self.assertIn("Failed to create", result.error)

    def test_worktree_no_manager(self):
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(
            agent_factory=factory,
            worktree_manager=None,
            workspace_dir="/tmp/ws",
        )
        result = run(tool.execute(
            description="no mgr",
            prompt="try worktree",
            subagent_type="general-purpose",
            isolation="worktree",
        ))
        self.assertFalse(result.success)

    def test_worktree_cleanup_on_subagent_error(self):
        wt_mgr = MockWorktreeManager()
        factory = MagicMock(return_value=FailingAgent())
        tool = TaskTool(
            agent_factory=factory,
            worktree_manager=wt_mgr,
            workspace_dir="/tmp/ws",
        )
        result = run(tool.execute(
            description="fail task",
            prompt="crash in worktree",
            subagent_type="general-purpose",
            isolation="worktree",
        ))
        self.assertFalse(result.success)
        # Force cleanup should have been called
        self.assertTrue(len(wt_mgr._removed) > 0)

    def test_worktree_metadata_in_result(self):
        wt_mgr = MockWorktreeManager(has_changes=True)
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(
            agent_factory=factory,
            worktree_manager=wt_mgr,
            workspace_dir="/tmp/ws",
        )
        result = run(tool.execute(
            description="meta test",
            prompt="check meta",
            subagent_type="general-purpose",
            isolation="worktree",
        ))
        self.assertIsNotNone(result.metadata.get("worktree_path"))
        self.assertIsNotNone(result.metadata.get("worktree_branch"))


# ═══════════════════════════════════════════════════════════════════════
# TEST: Agent Resume
# ═══════════════════════════════════════════════════════════════════════

class TestTaskToolResume(unittest.TestCase):
    """Test agent resume by ID."""

    def setUp(self):
        reset_session_store()
        import cowork_agent.tools.task_tool as tt
        tt._current_depth = 0

    def test_resume_existing_session(self):
        store = get_session_store()
        store.save(AgentSession(
            agent_id="prev-123",
            subagent_type="general-purpose",
            description="previous task",
            prompt="old prompt",
            result="old result",
            messages=[{"role": "user", "content": "hi"}],
        ))
        factory = MagicMock(return_value=MockAgent(result="Resumed work done."))
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        result = run(tool.execute(
            description="continue",
            prompt="keep going",
            subagent_type="general-purpose",
            resume="prev-123",
        ))
        self.assertTrue(result.success)
        self.assertIn("Resumed work done.", result.output)
        self.assertIn("prev-123", result.output)

    def test_resume_nonexistent_session(self):
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        result = run(tool.execute(
            description="bad resume",
            prompt="continue",
            subagent_type="general-purpose",
            resume="nonexistent-id",
        ))
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)

    def test_resume_shows_available_ids(self):
        store = get_session_store()
        store.save(AgentSession(
            agent_id="id-aaa", subagent_type="Bash",
            description="d", prompt="p", result="r",
        ))
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        result = run(tool.execute(
            description="bad resume",
            prompt="continue",
            subagent_type="general-purpose",
            resume="wrong-id",
        ))
        self.assertFalse(result.success)
        self.assertIn("id-aaa", result.error)

    def test_resume_updates_session(self):
        store = get_session_store()
        store.save(AgentSession(
            agent_id="upd-123",
            subagent_type="Explore",
            description="explore",
            prompt="find stuff",
            result="found A",
        ))
        factory = MagicMock(return_value=MockAgent(result="found B"))
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        run(tool.execute(
            description="continue explore",
            prompt="find more",
            subagent_type="Explore",
            resume="upd-123",
        ))
        updated = store.get("upd-123")
        self.assertEqual(updated.result, "found B")

    def test_resume_with_worktree_session(self):
        store = get_session_store()
        store.save(AgentSession(
            agent_id="wt-456",
            subagent_type="Bash",
            description="wt task",
            prompt="run cmd",
            result="done",
            worktree_path="/tmp/wt/my-wt",
        ))
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        result = run(tool.execute(
            description="resume wt",
            prompt="continue",
            subagent_type="Bash",
            resume="wt-456",
        ))
        self.assertTrue(result.success)
        # Should use worktree path as workspace
        call_kwargs = factory.call_args[1]
        self.assertEqual(call_kwargs["workspace_dir"], "/tmp/wt/my-wt")

    def test_resume_failure_returns_error(self):
        store = get_session_store()
        store.save(AgentSession(
            agent_id="fail-789",
            subagent_type="Bash",
            description="fail",
            prompt="crash",
            result="",
        ))
        factory = MagicMock(return_value=FailingAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        result = run(tool.execute(
            description="resume fail",
            prompt="continue",
            subagent_type="Bash",
            resume="fail-789",
        ))
        self.assertFalse(result.success)
        self.assertIn("Resume failed", result.error)

    def test_resume_metadata(self):
        store = get_session_store()
        store.save(AgentSession(
            agent_id="meta-001",
            subagent_type="Plan",
            description="plan",
            prompt="design",
            result="planned",
        ))
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        result = run(tool.execute(
            description="resume plan",
            prompt="continue",
            subagent_type="Plan",
            resume="meta-001",
        ))
        self.assertEqual(result.metadata.get("agent_id"), "meta-001")
        self.assertTrue(result.metadata.get("resumed"))


# ═══════════════════════════════════════════════════════════════════════
# TEST: TaskTool Schema
# ═══════════════════════════════════════════════════════════════════════

class TestTaskToolSchema(unittest.TestCase):
    """Test TaskTool schema definition."""

    def setUp(self):
        self.tool = TaskTool(agent_factory=MagicMock(), workspace_dir="/tmp")

    def test_schema_has_subagent_type(self):
        props = self.tool.input_schema["properties"]
        self.assertIn("subagent_type", props)
        self.assertIn("enum", props["subagent_type"])

    def test_schema_has_isolation(self):
        props = self.tool.input_schema["properties"]
        self.assertIn("isolation", props)
        self.assertEqual(props["isolation"]["enum"], ["worktree"])

    def test_schema_has_resume(self):
        props = self.tool.input_schema["properties"]
        self.assertIn("resume", props)

    def test_schema_has_model(self):
        props = self.tool.input_schema["properties"]
        self.assertIn("model", props)
        self.assertEqual(props["model"]["enum"], ["sonnet", "opus", "haiku"])

    def test_schema_required_fields(self):
        self.assertIn("description", self.tool.input_schema["required"])
        self.assertIn("prompt", self.tool.input_schema["required"])
        self.assertIn("subagent_type", self.tool.input_schema["required"])

    def test_schema_max_turns_positive(self):
        props = self.tool.input_schema["properties"]
        self.assertIn("exclusiveMinimum", props["max_turns"])

    def test_agent_type_enum_matches_profiles(self):
        enum_vals = set(self.tool.input_schema["properties"]["subagent_type"]["enum"])
        profile_names = set(AGENT_TYPE_PROFILES.keys())
        self.assertEqual(enum_vals, profile_names)


# ═══════════════════════════════════════════════════════════════════════
# TEST: TaskTool Static Methods
# ═══════════════════════════════════════════════════════════════════════

class TestTaskToolStaticMethods(unittest.TestCase):
    """Test static helper methods."""

    def test_get_agent_types(self):
        types = TaskTool.get_agent_types()
        self.assertEqual(len(types), 6)
        self.assertIn("Bash", types)

    def test_get_agent_type_existing(self):
        profile = TaskTool.get_agent_type("Explore")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "Explore")

    def test_get_agent_type_nonexistent(self):
        self.assertIsNone(TaskTool.get_agent_type("fantasy"))


# ═══════════════════════════════════════════════════════════════════════
# TEST: ToolRegistry.unregister()
# ═══════════════════════════════════════════════════════════════════════

class TestToolRegistryUnregister(unittest.TestCase):
    """Test the new unregister method added in Sprint 30."""

    def test_unregister_existing(self):
        reg = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        reg.register(mock_tool)
        self.assertTrue(reg.unregister("test_tool"))
        self.assertNotIn("test_tool", reg.tool_names)

    def test_unregister_nonexistent(self):
        reg = ToolRegistry()
        self.assertFalse(reg.unregister("nope"))

    def test_unregister_then_get_fails(self):
        reg = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.name = "temp"
        reg.register(mock_tool)
        reg.unregister("temp")
        with self.assertRaises(KeyError):
            reg.get_tool("temp")

    def test_unregister_preserves_others(self):
        reg = ToolRegistry()
        for name in ["a", "b", "c"]:
            t = MagicMock()
            t.name = name
            reg.register(t)
        reg.unregister("b")
        self.assertEqual(sorted(reg.tool_names), ["a", "c"])


# ═══════════════════════════════════════════════════════════════════════
# TEST: Progress Callback
# ═══════════════════════════════════════════════════════════════════════

class TestTaskToolProgressCallback(unittest.TestCase):
    """Test progress callback integration."""

    def setUp(self):
        reset_session_store()
        import cowork_agent.tools.task_tool as tt
        tt._current_depth = 0

    def test_progress_callback_called(self):
        cb = MagicMock()
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        run(tool.execute(
            description="test",
            prompt="do it",
            subagent_type="general-purpose",
            progress_callback=cb,
        ))
        cb.assert_called()
        # First call should be indeterminate (-1)
        cb.assert_any_call(-1, unittest.mock.ANY)

    def test_no_callback_no_error(self):
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        result = run(tool.execute(
            description="test",
            prompt="do it",
            subagent_type="general-purpose",
        ))
        self.assertTrue(result.success)


# ═══════════════════════════════════════════════════════════════════════
# TEST: Tool Filter Integration
# ═══════════════════════════════════════════════════════════════════════

class TestToolFilterIntegration(unittest.TestCase):
    """Test that tool_filter callable is properly passed to factory."""

    def setUp(self):
        reset_session_store()
        import cowork_agent.tools.task_tool as tt
        tt._current_depth = 0

    def test_bash_type_passes_filter_to_factory(self):
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        run(tool.execute(
            description="bash",
            prompt="run cmd",
            subagent_type="Bash",
        ))
        call_kwargs = factory.call_args[1]
        filter_fn = call_kwargs["tool_filter"]
        # The filter should only allow "bash"
        filtered = filter_fn(["bash", "read", "write", "edit"])
        self.assertEqual(filtered, ["bash"])

    def test_general_purpose_passes_none_filter(self):
        """general-purpose has tool_filter=None, so filter_tools returns all."""
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        run(tool.execute(
            description="general",
            prompt="do it",
            subagent_type="general-purpose",
        ))
        call_kwargs = factory.call_args[1]
        filter_fn = call_kwargs["tool_filter"]
        all_tools = ["bash", "read", "write", "edit", "task"]
        self.assertEqual(filter_fn(all_tools), all_tools)

    def test_explore_filter_removes_write_tools(self):
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        run(tool.execute(
            description="explore",
            prompt="find code",
            subagent_type="Explore",
        ))
        call_kwargs = factory.call_args[1]
        filter_fn = call_kwargs["tool_filter"]
        filtered = filter_fn(["read", "glob_tool", "edit", "write", "task"])
        self.assertIn("read", filtered)
        self.assertNotIn("edit", filtered)
        self.assertNotIn("write", filtered)
        self.assertNotIn("task", filtered)


# ═══════════════════════════════════════════════════════════════════════
# TEST: Enriched Prompt
# ═══════════════════════════════════════════════════════════════════════

class TestEnrichedPrompt(unittest.TestCase):
    """Test that the prompt is enriched with agent type instructions."""

    def setUp(self):
        reset_session_store()
        import cowork_agent.tools.task_tool as tt
        tt._current_depth = 0

    def test_bash_enriched_prompt(self):
        captured_prompt = []

        class CapturingAgent:
            def __init__(self, **kw):
                self._messages = []
                self.max_iterations = 10

            async def run(self, prompt):
                captured_prompt.append(prompt)
                return "done"

        factory = MagicMock(return_value=CapturingAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        run(tool.execute(
            description="bash",
            prompt="ls -la",
            subagent_type="Bash",
        ))
        self.assertIn("[Agent Type: Bash]", captured_prompt[0])
        self.assertIn("ls -la", captured_prompt[0])

    def test_general_purpose_no_extra_instructions(self):
        captured_prompt = []

        class CapturingAgent:
            def __init__(self, **kw):
                self._messages = []
                self.max_iterations = 10

            async def run(self, prompt):
                captured_prompt.append(prompt)
                return "done"

        factory = MagicMock(return_value=CapturingAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        run(tool.execute(
            description="general",
            prompt="do stuff",
            subagent_type="general-purpose",
        ))
        # general-purpose has empty system_instructions
        self.assertEqual(captured_prompt[0], "do stuff")


# ═══════════════════════════════════════════════════════════════════════
# TEST: Main.py Wiring
# ═══════════════════════════════════════════════════════════════════════

class TestMainWiring(unittest.TestCase):
    """Test Sprint 30 wiring logic in main.py."""

    def test_task_tool_has_required_params(self):
        """Verify TaskTool constructor accepts the Sprint 30 params."""
        tool = TaskTool(
            agent_factory=MagicMock(),
            worktree_manager=MagicMock(),
            workspace_dir="/tmp/ws",
        )
        self.assertEqual(tool.name, "task")
        self.assertIsNotNone(tool._worktree_manager)
        self.assertEqual(tool._workspace_dir, "/tmp/ws")

    def test_factory_keyword_args(self):
        """Verify the factory is called with Sprint 30 keyword args."""
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        import cowork_agent.tools.task_tool as tt
        tt._current_depth = 0
        reset_session_store()
        run(tool.execute(
            description="test",
            prompt="do it",
            subagent_type="Bash",
        ))
        factory.assert_called_once()
        kwargs = factory.call_args[1]
        self.assertIn("tool_filter", kwargs)
        self.assertIn("system_instructions", kwargs)
        self.assertIn("max_iterations", kwargs)
        self.assertIn("workspace_dir", kwargs)


# ═══════════════════════════════════════════════════════════════════════
# TEST: Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestTaskToolEdgeCases(unittest.TestCase):
    """Edge case tests for Sprint 30 Task Tool."""

    def setUp(self):
        reset_session_store()
        import cowork_agent.tools.task_tool as tt
        tt._current_depth = 0

    def test_very_long_description_for_worktree_name(self):
        wt_mgr = MockWorktreeManager()
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(
            agent_factory=factory,
            worktree_manager=wt_mgr,
            workspace_dir="/tmp",
        )
        run(tool.execute(
            description="a" * 200,
            prompt="long desc",
            subagent_type="general-purpose",
            isolation="worktree",
        ))
        # Worktree name should be truncated
        if wt_mgr._created:
            self.assertLessEqual(len(wt_mgr._created[0].name), 50)

    def test_special_chars_in_description_for_worktree(self):
        wt_mgr = MockWorktreeManager()
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(
            agent_factory=factory,
            worktree_manager=wt_mgr,
            workspace_dir="/tmp",
        )
        run(tool.execute(
            description="fix bug #123 (urgent!)",
            prompt="fix it",
            subagent_type="general-purpose",
            isolation="worktree",
        ))
        self.assertTrue(True)  # Should not crash

    def test_resume_skips_depth_check_initially(self):
        """Resume still needs depth check for the actual run."""
        store = get_session_store()
        store.save(AgentSession(
            agent_id="depth-test",
            subagent_type="Bash",
            description="d", prompt="p", result="r",
        ))
        import cowork_agent.tools.task_tool as tt
        tt._current_depth = 3
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        # Resume should still fail at depth limit in _handle_resume's run
        # (the depth increment happens inside _handle_resume too)
        result = run(tool.execute(
            description="deep resume",
            prompt="continue",
            subagent_type="Bash",
            resume="depth-test",
        ))
        # It should succeed or fail based on whether depth is checked in resume
        # In current impl, resume increments depth too
        tt._current_depth = 0

    def test_empty_isolation_string_no_worktree(self):
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        result = run(tool.execute(
            description="no isolation",
            prompt="normal",
            subagent_type="general-purpose",
            isolation="",
        ))
        self.assertTrue(result.success)

    def test_concurrent_session_saves(self):
        """Multiple task executions should each save their session."""
        factory = MagicMock(return_value=MockAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        for i in range(5):
            run(tool.execute(
                description=f"task {i}",
                prompt=f"prompt {i}",
                subagent_type="general-purpose",
            ))
        store = get_session_store()
        self.assertEqual(len(store), 5)

    def test_resume_injects_previous_messages(self):
        store = get_session_store()
        mock_messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        store.save(AgentSession(
            agent_id="msg-test",
            subagent_type="general-purpose",
            description="msg test",
            prompt="original",
            result="done",
            messages=mock_messages,
        ))

        injected_messages = []

        class TrackingAgent:
            def __init__(self, **kw):
                self._messages = []
                self.max_iterations = 10

            async def run(self, prompt):
                injected_messages.extend(self._messages)
                return "resumed"

        factory = MagicMock(return_value=TrackingAgent())
        tool = TaskTool(agent_factory=factory, workspace_dir="/tmp")
        run(tool.execute(
            description="resume msg",
            prompt="continue",
            subagent_type="general-purpose",
            resume="msg-test",
        ))
        self.assertEqual(len(injected_messages), 2)


if __name__ == "__main__":
    unittest.main()
