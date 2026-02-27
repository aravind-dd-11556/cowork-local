"""
Tests for P2 features:
  1. Skill Registry
  2. Plan Mode
  3. Plan Tools (EnterPlanModeTool / ExitPlanModeTool)
  4. Task Tool (subagent delegation)
  5. MCP Client (unit-level, no real servers)
  6. MCP Bridge
  7. Plugin System
  8. Scheduled Tasks (TaskScheduler)
  9. Worktree Manager (unit-level, no real git repo)
"""

import asyncio
import json
import os
import sys
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from unittest.mock import AsyncMock, MagicMock, patch

# ── 1. Skill Registry ────────────────────────────────────────────────────────

from cowork_agent.core.skill_registry import Skill, SkillRegistry


class TestSkillRegistry(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a fake skill directory:  workspace/.skills/skills/docx/SKILL.md
        self.skills_base = os.path.join(self.tmpdir, ".skills", "skills")
        docx_dir = os.path.join(self.skills_base, "docx")
        os.makedirs(docx_dir)
        with open(os.path.join(docx_dir, "SKILL.md"), "w") as f:
            f.write(
                "# Word Document Skill\n"
                "MANDATORY TRIGGERS: Word, document, .docx, report\n\n"
                "Instructions for creating Word documents.\n"
            )

        # Another skill with no explicit triggers
        pdf_dir = os.path.join(self.skills_base, "pdf")
        os.makedirs(pdf_dir)
        with open(os.path.join(pdf_dir, "SKILL.md"), "w") as f:
            f.write("# PDF Skill\nInstructions for working with PDFs.\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_discover_finds_skills(self):
        reg = SkillRegistry(workspace_dir=self.tmpdir, user_skills_dir="/nonexistent")
        count = reg.discover()
        self.assertEqual(count, 2)
        self.assertIn("docx", reg.skill_names)
        self.assertIn("pdf", reg.skill_names)

    def test_discover_loads_content(self):
        reg = SkillRegistry(workspace_dir=self.tmpdir, user_skills_dir="/nonexistent")
        reg.discover()
        skill = reg.get_skill("docx")
        self.assertIsNotNone(skill)
        self.assertIn("Word Document Skill", skill.content)

    def test_trigger_extraction(self):
        reg = SkillRegistry(workspace_dir=self.tmpdir, user_skills_dir="/nonexistent")
        reg.discover()
        skill = reg.get_skill("docx")
        # Name-based triggers + mandatory triggers
        self.assertIn("docx", skill.triggers)
        self.assertIn("word", skill.triggers)
        self.assertIn("document", skill.triggers)
        self.assertIn("report", skill.triggers)

    def test_match_skills(self):
        reg = SkillRegistry(workspace_dir=self.tmpdir, user_skills_dir="/nonexistent")
        reg.discover()
        matches = reg.match_skills("Create a Word document report")
        self.assertTrue(len(matches) > 0)
        self.assertEqual(matches[0].name, "docx")

    def test_match_no_results(self):
        reg = SkillRegistry(workspace_dir=self.tmpdir, user_skills_dir="/nonexistent")
        reg.discover()
        matches = reg.match_skills("hello world")
        self.assertEqual(len(matches), 0)

    def test_available_skills_section(self):
        reg = SkillRegistry(workspace_dir=self.tmpdir, user_skills_dir="/nonexistent")
        reg.discover()
        section = reg.get_available_skills_section()
        self.assertIn("<available_skills>", section)
        self.assertIn("docx", section)
        self.assertIn("pdf", section)

    def test_skill_prompt_section(self):
        reg = SkillRegistry(workspace_dir=self.tmpdir, user_skills_dir="/nonexistent")
        reg.discover()
        skill = reg.get_skill("docx")
        section = reg.get_skill_prompt_section([skill])
        self.assertIn("<skills_instructions>", section)
        self.assertIn("Word Document Skill", section)

    def test_empty_skills(self):
        reg = SkillRegistry(workspace_dir="/nonexistent", user_skills_dir="/nonexistent")
        count = reg.discover()
        self.assertEqual(count, 0)
        self.assertEqual(reg.get_available_skills_section(), "")


# ── 2. Plan Mode ─────────────────────────────────────────────────────────────

from cowork_agent.core.plan_mode import AgentMode, PlanManager, PLAN_MODE_ALLOWED_TOOLS, PLAN_MODE_BLOCKED_TOOLS


class TestPlanMode(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pm = PlanManager(workspace_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_initial_state_is_normal(self):
        self.assertEqual(self.pm.mode, AgentMode.NORMAL)
        self.assertFalse(self.pm.is_plan_mode)

    def test_enter_plan_mode(self):
        result = self.pm.enter_plan_mode()
        self.assertIn("plan mode", result.lower())
        self.assertTrue(self.pm.is_plan_mode)
        self.assertEqual(self.pm.mode, AgentMode.PLAN)

    def test_enter_already_in_plan_mode(self):
        self.pm.enter_plan_mode()
        result = self.pm.enter_plan_mode()
        self.assertIn("already", result.lower())

    def test_exit_plan_mode_with_plan(self):
        self.pm.enter_plan_mode()
        plan = "# My Plan\n1. Step one\n2. Step two"
        result = self.pm.exit_plan_mode(plan)
        self.assertFalse(self.pm.is_plan_mode)
        self.assertEqual(self.pm.plan_content, plan)
        # Check plan saved to disk
        plan_file = os.path.join(self.tmpdir, ".cowork", "plan.md")
        self.assertTrue(os.path.exists(plan_file))
        with open(plan_file) as f:
            self.assertEqual(f.read(), plan)

    def test_exit_not_in_plan_mode(self):
        result = self.pm.exit_plan_mode("some plan")
        self.assertIn("not in plan mode", result.lower())

    def test_tool_allowed_normal_mode(self):
        allowed, reason = self.pm.is_tool_allowed("bash")
        self.assertTrue(allowed)
        self.assertEqual(reason, "")

    def test_tool_blocked_in_plan_mode(self):
        self.pm.enter_plan_mode()
        for tool in PLAN_MODE_BLOCKED_TOOLS:
            allowed, reason = self.pm.is_tool_allowed(tool)
            self.assertFalse(allowed, f"{tool} should be blocked in plan mode")
            self.assertIn("not available in plan mode", reason)

    def test_tool_allowed_in_plan_mode(self):
        self.pm.enter_plan_mode()
        for tool in PLAN_MODE_ALLOWED_TOOLS:
            allowed, reason = self.pm.is_tool_allowed(tool)
            self.assertTrue(allowed, f"{tool} should be allowed in plan mode")

    def test_unknown_tool_blocked_in_plan_mode(self):
        self.pm.enter_plan_mode()
        allowed, reason = self.pm.is_tool_allowed("some_unknown_tool")
        self.assertFalse(allowed)

    def test_plan_mode_prompt(self):
        # Should return empty when not in plan mode
        self.assertEqual(self.pm.get_plan_mode_prompt(), "")
        # Should return instructions when in plan mode
        self.pm.enter_plan_mode()
        prompt = self.pm.get_plan_mode_prompt()
        self.assertIn("<plan_mode>", prompt)
        self.assertIn("read-only", prompt)


# ── 3. Plan Tools ────────────────────────────────────────────────────────────

from cowork_agent.tools.plan_tools import EnterPlanModeTool, ExitPlanModeTool


class TestPlanTools(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pm = PlanManager(workspace_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_enter_plan_mode_tool(self):
        tool = EnterPlanModeTool(self.pm)
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(tool_id="t1")
        )
        self.assertTrue(result.success)
        self.assertTrue(self.pm.is_plan_mode)

    def test_exit_plan_mode_tool(self):
        self.pm.enter_plan_mode()
        tool = ExitPlanModeTool(self.pm)
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(plan="My plan content", tool_id="t2")
        )
        self.assertTrue(result.success)
        self.assertFalse(self.pm.is_plan_mode)
        self.assertIn("My plan content", result.output)


# ── 4. Task Tool ─────────────────────────────────────────────────────────────

from cowork_agent.tools.task_tool import TaskTool


class TestTaskTool(unittest.TestCase):
    def test_task_tool_runs_subagent(self):
        # Create a mock agent that returns a result
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="Subagent completed the task.")
        mock_agent.max_iterations = 10

        factory = MagicMock(return_value=mock_agent)
        tool = TaskTool(agent_factory=factory)

        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(
                description="Test task",
                prompt="Do something",
                max_turns=5,
                tool_id="t3",
            )
        )
        self.assertTrue(result.success)
        self.assertIn("Subagent completed the task", result.output)
        factory.assert_called_once()
        mock_agent.run.assert_awaited_once_with("Do something")

    def test_task_tool_no_prompt(self):
        factory = MagicMock()
        tool = TaskTool(agent_factory=factory)

        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(description="Test", prompt="", tool_id="t4")
        )
        self.assertFalse(result.success)
        self.assertIn("required", result.error.lower())

    def test_task_tool_handles_exception(self):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        mock_agent.max_iterations = 10
        factory = MagicMock(return_value=mock_agent)
        tool = TaskTool(agent_factory=factory)

        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(description="Fail task", prompt="Crash", tool_id="t5")
        )
        self.assertFalse(result.success)
        self.assertIn("boom", result.error)


# ── 5. MCP Client (unit-level) ───────────────────────────────────────────────

from cowork_agent.core.mcp_client import MCPClient, MCPServerConfig, MCPTool


class TestMCPClient(unittest.TestCase):
    def test_add_server(self):
        client = MCPClient()
        client.add_server(MCPServerConfig(name="test", command="echo"))
        self.assertIn("test", client._servers)

    def test_get_tools_empty(self):
        client = MCPClient()
        self.assertEqual(client.get_tools(), [])

    def test_call_unknown_tool(self):
        client = MCPClient()
        result = asyncio.get_event_loop().run_until_complete(
            client.call_tool("mcp__nonexistent__tool", {})
        )
        self.assertIn("error", result)

    def test_tool_schemas_format(self):
        client = MCPClient()
        # Manually inject a tool for testing
        client._tools["mcp__test__hello"] = MCPTool(
            name="mcp__test__hello",
            description="Say hello",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
            server_name="test",
        )
        schemas = client.get_tool_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["name"], "mcp__test__hello")
        self.assertIn("[MCP:test]", schemas[0]["description"])


# ── 6. MCP Bridge ────────────────────────────────────────────────────────────

from cowork_agent.tools.mcp_bridge import MCPBridgeTool, register_mcp_tools


class TestMCPBridge(unittest.TestCase):
    def test_bridge_tool_creation(self):
        mcp_tool = MCPTool(
            name="mcp__github__list_repos",
            description="List repos",
            input_schema={"type": "object", "properties": {}},
            server_name="github",
        )
        mock_client = MagicMock()
        bridge = MCPBridgeTool(mcp_tool, mock_client)
        self.assertEqual(bridge.name, "mcp__github__list_repos")
        self.assertIn("[MCP:github]", bridge.description)

    def test_bridge_tool_execute_success(self):
        mcp_tool = MCPTool(
            name="mcp__test__greet",
            description="Greet",
            input_schema={"type": "object", "properties": {}},
            server_name="test",
        )
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(return_value={
            "content": [{"type": "text", "text": "Hello, world!"}]
        })
        bridge = MCPBridgeTool(mcp_tool, mock_client)

        result = asyncio.get_event_loop().run_until_complete(
            bridge.execute(tool_id="t6")
        )
        self.assertTrue(result.success)
        self.assertIn("Hello, world!", result.output)

    def test_bridge_tool_execute_error(self):
        mcp_tool = MCPTool(
            name="mcp__test__fail",
            description="Fail",
            input_schema={},
            server_name="test",
        )
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(return_value={"error": "Something went wrong"})
        bridge = MCPBridgeTool(mcp_tool, mock_client)

        result = asyncio.get_event_loop().run_until_complete(
            bridge.execute(tool_id="t7")
        )
        self.assertFalse(result.success)

    def test_register_mcp_tools(self):
        mock_client = MagicMock()
        mock_client.get_tools.return_value = [
            MCPTool(name="mcp__s__t1", description="T1", input_schema={}, server_name="s"),
            MCPTool(name="mcp__s__t2", description="T2", input_schema={}, server_name="s"),
        ]
        mock_registry = MagicMock()
        count = register_mcp_tools(mock_registry, mock_client)
        self.assertEqual(count, 2)
        self.assertEqual(mock_registry.register.call_count, 2)


# ── 7. Plugin System ─────────────────────────────────────────────────────────

from cowork_agent.core.plugin_system import PluginSystem, PluginInfo


class TestPluginSystem(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.plugins_dir = os.path.join(self.tmpdir, ".cowork", "plugins")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_plugins_dir(self):
        ps = PluginSystem(workspace_dir=self.tmpdir, user_plugins_dir="/nonexistent")
        results = ps.discover_and_load()
        self.assertEqual(results, [])

    def test_discover_valid_plugin(self):
        # Create a minimal plugin
        plugin_dir = os.path.join(self.plugins_dir, "sample")
        os.makedirs(plugin_dir)

        with open(os.path.join(plugin_dir, "plugin.json"), "w") as f:
            json.dump({"name": "Sample Plugin", "description": "A test plugin", "version": "1.0.0"}, f)

        # __init__.py exports TOOLS = []
        with open(os.path.join(plugin_dir, "__init__.py"), "w") as f:
            f.write("TOOLS = []\n")

        ps = PluginSystem(workspace_dir=self.tmpdir, user_plugins_dir="/nonexistent")
        results = ps.discover_and_load()
        self.assertEqual(len(results), 1)
        info, tools = results[0]
        self.assertEqual(info.name, "Sample Plugin")
        self.assertEqual(info.version, "1.0.0")
        self.assertTrue(info.enabled)
        self.assertEqual(tools, [])

    def test_plugin_with_bad_init(self):
        plugin_dir = os.path.join(self.plugins_dir, "broken")
        os.makedirs(plugin_dir)
        with open(os.path.join(plugin_dir, "__init__.py"), "w") as f:
            f.write("raise RuntimeError('plugin init crash')\n")

        ps = PluginSystem(workspace_dir=self.tmpdir, user_plugins_dir="/nonexistent")
        results = ps.discover_and_load()
        self.assertEqual(len(results), 1)
        info, tools = results[0]
        self.assertFalse(info.enabled)
        self.assertIn("plugin init crash", info.error)

    def test_plugins_summary(self):
        ps = PluginSystem(workspace_dir="/nonexistent", user_plugins_dir="/nonexistent")
        ps.discover_and_load()
        self.assertEqual(ps.get_plugins_summary(), "No plugins loaded.")

    def test_plugin_with_tools_export(self):
        plugin_dir = os.path.join(self.plugins_dir, "withtool")
        os.makedirs(plugin_dir)
        with open(os.path.join(plugin_dir, "__init__.py"), "w") as f:
            f.write(
                "class FakeTool:\n"
                "    name = 'fake_tool'\n"
                "TOOLS = [FakeTool()]\n"
            )
        ps = PluginSystem(workspace_dir=self.tmpdir, user_plugins_dir="/nonexistent")
        results = ps.discover_and_load()
        info, tools = results[0]
        self.assertEqual(len(tools), 1)
        self.assertEqual(info.tool_names, ["fake_tool"])


# ── 8. Scheduled Tasks ───────────────────────────────────────────────────────

from cowork_agent.core.scheduler import ScheduledTask, TaskScheduler


class TestScheduledTasks(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.scheduler = TaskScheduler(workspace_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_task(self):
        task = ScheduledTask(
            task_id="daily-standup",
            prompt="Summarize tasks",
            description="Daily standup summary",
            cron_expression="0 9 * * 1-5",
        )
        result = self.scheduler.create_task(task)
        self.assertIn("daily-standup", result)
        self.assertIn("daily-standup", self.scheduler.tasks)

    def test_task_persistence(self):
        task = ScheduledTask(
            task_id="persist-test",
            prompt="Test prompt",
            description="Test desc",
        )
        self.scheduler.create_task(task)

        # Create new scheduler and load
        scheduler2 = TaskScheduler(workspace_dir=self.tmpdir)
        count = scheduler2.load()
        self.assertEqual(count, 1)
        self.assertIn("persist-test", scheduler2.tasks)

    def test_update_task(self):
        task = ScheduledTask(
            task_id="update-test",
            prompt="Old prompt",
            description="Old desc",
        )
        self.scheduler.create_task(task)
        result = self.scheduler.update_task("update-test", prompt="New prompt")
        self.assertIn("updated", result.lower())
        self.assertEqual(self.scheduler.tasks["update-test"].prompt, "New prompt")

    def test_update_nonexistent(self):
        result = self.scheduler.update_task("nonexistent", prompt="x")
        self.assertIn("not found", result.lower())

    def test_delete_task(self):
        task = ScheduledTask(task_id="del-test", prompt="X", description="X")
        self.scheduler.create_task(task)
        result = self.scheduler.delete_task("del-test")
        self.assertIn("deleted", result.lower())
        self.assertNotIn("del-test", self.scheduler.tasks)

    def test_delete_nonexistent(self):
        result = self.scheduler.delete_task("nope")
        self.assertIn("not found", result.lower())

    def test_list_tasks(self):
        self.scheduler.create_task(ScheduledTask(task_id="a", prompt="A", description="A"))
        self.scheduler.create_task(ScheduledTask(task_id="b", prompt="B", description="B"))
        tasks = self.scheduler.list_tasks()
        self.assertEqual(len(tasks), 2)
        ids = [t["task_id"] for t in tasks]
        self.assertIn("a", ids)
        self.assertIn("b", ids)

    def test_cron_next_time(self):
        # Should compute a next run time for a valid cron expression
        next_time = TaskScheduler._next_cron_time("30 8 * * *")
        self.assertIsNotNone(next_time)
        parsed = datetime.fromisoformat(next_time)
        self.assertEqual(parsed.hour, 8)
        self.assertEqual(parsed.minute, 30)

    def test_cron_invalid(self):
        next_time = TaskScheduler._next_cron_time("not a cron")
        self.assertIsNone(next_time)

    def test_cron_with_dow(self):
        # Weekday-only: "0 9 * * 1-5"
        next_time = TaskScheduler._next_cron_time("0 9 * * 1-5")
        self.assertIsNotNone(next_time)
        parsed = datetime.fromisoformat(next_time)
        # Should be a weekday (Mon=0 through Fri=4 in Python)
        self.assertIn(parsed.weekday(), [0, 1, 2, 3, 4])

    def test_task_to_dict(self):
        task = ScheduledTask(
            task_id="dict-test",
            prompt="P",
            description="D",
            cron_expression="0 0 * * *",
        )
        d = task.to_dict()
        self.assertEqual(d["task_id"], "dict-test")
        self.assertEqual(d["prompt"], "P")
        # Round-trip
        task2 = ScheduledTask.from_dict(d)
        self.assertEqual(task2.task_id, "dict-test")


# ── 9. Worktree Manager ──────────────────────────────────────────────────────

from cowork_agent.core.worktree import WorktreeManager, WorktreeInfo


class TestWorktreeManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_not_git_repo(self):
        wm = WorktreeManager(workspace_dir=self.tmpdir)
        self.assertFalse(wm.is_git_repo())

    def test_create_fails_without_git(self):
        wm = WorktreeManager(workspace_dir=self.tmpdir)
        result = wm.create("test")
        self.assertIsNone(result)

    def test_list_empty(self):
        wm = WorktreeManager(workspace_dir=self.tmpdir)
        self.assertEqual(wm.list_worktrees(), [])

    def test_remove_nonexistent(self):
        wm = WorktreeManager(workspace_dir=self.tmpdir)
        result = wm.remove("nonexistent")
        self.assertIn("not found", result.lower())

    def test_random_name_format(self):
        name = WorktreeManager._random_name()
        parts = name.split("-")
        self.assertEqual(len(parts), 3)  # adjective-noun-suffix
        self.assertEqual(len(parts[2]), 4)  # suffix is 4 chars

    def test_worktree_info_dataclass(self):
        info = WorktreeInfo(
            name="test", path="/tmp/test", branch="worktree/test",
            created_from="main", has_changes=True,
        )
        self.assertEqual(info.name, "test")
        self.assertTrue(info.has_changes)


# ── 10. Prompt Builder (P2 integration) ──────────────────────────────────────

from cowork_agent.core.prompt_builder import PromptBuilder


class TestPromptBuilderP2(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        skills_dir = os.path.join(self.tmpdir, ".skills", "skills", "xlsx")
        os.makedirs(skills_dir)
        with open(os.path.join(skills_dir, "SKILL.md"), "w") as f:
            f.write("# Excel Skill\nMANDATORY TRIGGERS: excel, spreadsheet\nUse this for Excel.\n")

        self.skill_registry = SkillRegistry(workspace_dir=self.tmpdir, user_skills_dir="/nonexistent")
        self.skill_registry.discover()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_prompt_includes_available_skills(self):
        config = {"agent": {"workspace_dir": self.tmpdir}, "llm": {"model": "test", "provider": "test"}}
        builder = PromptBuilder(config, skill_registry=self.skill_registry)
        prompt = builder.build(tools=[], context={})
        self.assertIn("<available_skills>", prompt)
        self.assertIn("xlsx", prompt)

    def test_prompt_includes_plan_mode(self):
        config = {"agent": {"workspace_dir": self.tmpdir}, "llm": {"model": "test", "provider": "test"}}
        builder = PromptBuilder(config, skill_registry=self.skill_registry)
        prompt = builder.build(tools=[], context={
            "plan_mode_prompt": "<plan_mode>You are in plan mode</plan_mode>"
        })
        self.assertIn("<plan_mode>", prompt)

    def test_prompt_includes_active_skills(self):
        config = {"agent": {"workspace_dir": self.tmpdir}, "llm": {"model": "test", "provider": "test"}}
        builder = PromptBuilder(config, skill_registry=self.skill_registry)
        xlsx_skill = self.skill_registry.get_skill("xlsx")
        prompt = builder.build(tools=[], context={
            "active_skills": [xlsx_skill],
        })
        self.assertIn("Excel Skill", prompt)
        self.assertIn("<skills_instructions>", prompt)


if __name__ == "__main__":
    unittest.main()
