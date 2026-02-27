"""
Tests for P3 features:
  1. AskUserQuestion structured format (header, options, multiSelect)
  2. NotebookEdit tool (Jupyter .ipynb cell editing)
  3. Worktree tools (enter, list, remove)
  4. Scheduler tools (create, list, update)
  5. Anthropic provider streaming (method exists, event handling)
  6. OpenAI provider streaming (method exists, chunk handling)
  7. Session persistence (create, save, load, list, delete)
"""

import asyncio
import json
import os
import sys
import shutil
import tempfile
import time
import unittest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── 1. AskUserQuestion Structured ───────────────────────────────────────────

from cowork_agent.tools.ask_user import AskUserTool


class TestAskUserStructured(unittest.TestCase):
    def setUp(self):
        self.tool = AskUserTool()

    def test_schema_has_questions_array(self):
        schema = self.tool.input_schema
        self.assertIn("questions", schema["properties"])
        q_schema = schema["properties"]["questions"]
        self.assertEqual(q_schema["type"], "array")
        self.assertEqual(q_schema["minItems"], 1)
        self.assertEqual(q_schema["maxItems"], 4)

    def test_questions_item_schema(self):
        items = self.tool.input_schema["properties"]["questions"]["items"]
        props = items["properties"]
        self.assertIn("question", props)
        self.assertIn("header", props)
        self.assertIn("options", props)
        self.assertIn("multiSelect", props)
        # Options have label + description
        opt_props = props["options"]["items"]["properties"]
        self.assertIn("label", opt_props)
        self.assertIn("description", opt_props)
        self.assertIn("markdown", opt_props)

    def test_structured_question_with_callback(self):
        """Test structured question dispatches to callback correctly."""
        captured = {}

        def mock_callback(question, options, multi_select):
            captured["question"] = question
            captured["options"] = options
            captured["multi_select"] = multi_select
            return "Option A"

        self.tool.set_input_callback(mock_callback)

        result = asyncio.get_event_loop().run_until_complete(
            self.tool.execute(
                questions=[{
                    "question": "Which approach?",
                    "header": "Approach",
                    "options": [
                        {"label": "Option A", "description": "First approach"},
                        {"label": "Option B", "description": "Second approach"},
                    ],
                    "multiSelect": False,
                }],
                tool_id="t1",
            )
        )

        self.assertTrue(result.success)
        self.assertIn("Option A", result.output)
        self.assertEqual(captured["question"], "[Approach] Which approach?")
        self.assertFalse(captured["multi_select"])

    def test_multi_select_response(self):
        """Test multiSelect returns comma-separated answers."""
        def mock_callback(question, options, multi_select):
            return ["Feature A", "Feature C"]

        self.tool.set_input_callback(mock_callback)

        result = asyncio.get_event_loop().run_until_complete(
            self.tool.execute(
                questions=[{
                    "question": "Which features?",
                    "header": "Features",
                    "options": [
                        {"label": "Feature A", "description": "First"},
                        {"label": "Feature B", "description": "Second"},
                        {"label": "Feature C", "description": "Third"},
                    ],
                    "multiSelect": True,
                }],
                tool_id="t2",
            )
        )

        self.assertTrue(result.success)
        self.assertIn("Feature A", result.output)
        self.assertIn("Feature C", result.output)

    def test_legacy_simple_format(self):
        """Test backward-compatible simple question format."""
        def mock_callback(question, options):
            return "yes"

        self.tool.set_input_callback(mock_callback)

        result = asyncio.get_event_loop().run_until_complete(
            self.tool.execute(question="Continue?", options=["yes", "no"], tool_id="t3")
        )

        self.assertTrue(result.success)
        self.assertIn("yes", result.output)

    def test_no_callback_error(self):
        """Tool should error if no input callback is set."""
        result = asyncio.get_event_loop().run_until_complete(
            self.tool.execute(
                questions=[{
                    "question": "Test?",
                    "header": "Test",
                    "options": [{"label": "A", "description": "a"}, {"label": "B", "description": "b"}],
                    "multiSelect": False,
                }],
                tool_id="t4",
            )
        )
        self.assertFalse(result.success)
        self.assertIn("callback", result.error.lower())

    def test_multiple_questions(self):
        """Test asking multiple questions in one call."""
        call_count = [0]

        def mock_callback(question, options, multi_select):
            call_count[0] += 1
            return f"Answer {call_count[0]}"

        self.tool.set_input_callback(mock_callback)

        result = asyncio.get_event_loop().run_until_complete(
            self.tool.execute(
                questions=[
                    {
                        "question": "Q1?", "header": "First",
                        "options": [{"label": "A", "description": "a"}, {"label": "B", "description": "b"}],
                        "multiSelect": False,
                    },
                    {
                        "question": "Q2?", "header": "Second",
                        "options": [{"label": "X", "description": "x"}, {"label": "Y", "description": "y"}],
                        "multiSelect": False,
                    },
                ],
                tool_id="t5",
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(call_count[0], 2)
        self.assertIn("Answer 1", result.output)
        self.assertIn("Answer 2", result.output)


# ── 2. NotebookEdit ─────────────────────────────────────────────────────────

from cowork_agent.tools.notebook_edit import NotebookEditTool


class TestNotebookEdit(unittest.TestCase):
    def setUp(self):
        self.tool = NotebookEditTool()
        self.tmpdir = tempfile.mkdtemp()
        self.nb_path = os.path.join(self.tmpdir, "test.ipynb")

        # Create a minimal valid notebook
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {
                    "cell_type": "code",
                    "id": "cell-001",
                    "metadata": {},
                    "source": ["print('hello')\n"],
                    "outputs": [],
                    "execution_count": 1,
                },
                {
                    "cell_type": "markdown",
                    "id": "cell-002",
                    "metadata": {},
                    "source": ["# Title\n"],
                },
            ],
        }
        with open(self.nb_path, "w") as f:
            json.dump(notebook, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _load(self):
        with open(self.nb_path) as f:
            return json.load(f)

    def test_replace_cell(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.tool.execute(
                notebook_path=self.nb_path,
                cell_number=0,
                new_source="print('world')",
                tool_id="t1",
            )
        )
        self.assertTrue(result.success)
        nb = self._load()
        self.assertEqual(nb["cells"][0]["source"], ["print('world')"])
        # Outputs should be cleared
        self.assertEqual(nb["cells"][0]["outputs"], [])
        self.assertIsNone(nb["cells"][0]["execution_count"])

    def test_replace_by_cell_id(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.tool.execute(
                notebook_path=self.nb_path,
                cell_id="cell-002",
                new_source="# New Title\nSome text",
                tool_id="t2",
            )
        )
        self.assertTrue(result.success)
        nb = self._load()
        self.assertEqual(nb["cells"][1]["source"], ["# New Title\n", "Some text"])

    def test_insert_cell(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.tool.execute(
                notebook_path=self.nb_path,
                cell_number=1,
                new_source="x = 42",
                cell_type="code",
                edit_mode="insert",
                tool_id="t3",
            )
        )
        self.assertTrue(result.success)
        nb = self._load()
        self.assertEqual(len(nb["cells"]), 3)
        self.assertEqual(nb["cells"][1]["source"], ["x = 42"])
        self.assertEqual(nb["cells"][1]["cell_type"], "code")

    def test_insert_requires_cell_type(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.tool.execute(
                notebook_path=self.nb_path,
                cell_number=0,
                new_source="test",
                edit_mode="insert",
                tool_id="t4",
            )
        )
        self.assertFalse(result.success)
        self.assertIn("cell_type", result.error)

    def test_delete_cell(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.tool.execute(
                notebook_path=self.nb_path,
                cell_number=0,
                new_source="",
                edit_mode="delete",
                tool_id="t5",
            )
        )
        self.assertTrue(result.success)
        nb = self._load()
        self.assertEqual(len(nb["cells"]), 1)
        self.assertEqual(nb["cells"][0]["cell_type"], "markdown")

    def test_out_of_range_error(self):
        result = asyncio.get_event_loop().run_until_complete(
            self.tool.execute(
                notebook_path=self.nb_path,
                cell_number=99,
                new_source="test",
                tool_id="t6",
            )
        )
        self.assertFalse(result.success)
        self.assertIn("out of range", result.error)

    def test_not_ipynb_error(self):
        txt_path = os.path.join(self.tmpdir, "test.txt")
        with open(txt_path, "w") as f:
            f.write("hello")

        result = asyncio.get_event_loop().run_until_complete(
            self.tool.execute(notebook_path=txt_path, new_source="x", tool_id="t7")
        )
        self.assertFalse(result.success)
        self.assertIn("Not a Jupyter", result.error)

    def test_source_line_formatting(self):
        """Verify source is split into lines with newlines (ipynb format)."""
        lines = NotebookEditTool._to_source_lines("line1\nline2\nline3")
        self.assertEqual(lines, ["line1\n", "line2\n", "line3"])

    def test_empty_source(self):
        lines = NotebookEditTool._to_source_lines("")
        self.assertEqual(lines, [])


# ── 3. Worktree Tools ───────────────────────────────────────────────────────

from cowork_agent.tools.worktree_tool import (
    EnterWorktreeTool, ListWorktreesTool, RemoveWorktreeTool,
)
from cowork_agent.core.worktree import WorktreeManager


class TestWorktreeTools(unittest.TestCase):
    def test_enter_not_git_repo(self):
        """EnterWorktree should error if not in a git repo."""
        tmpdir = tempfile.mkdtemp()
        try:
            tool = EnterWorktreeTool(workspace_dir=tmpdir)
            result = asyncio.get_event_loop().run_until_complete(
                tool.execute(tool_id="t1")
            )
            self.assertFalse(result.success)
            self.assertIn("git repository", result.error.lower())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_enter_prevents_nesting(self):
        """Cannot create a worktree when already in one."""
        tool = EnterWorktreeTool(workspace_dir="/tmp")
        tool._active_worktree = "existing-wt"
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(tool_id="t2")
        )
        self.assertFalse(result.success)
        self.assertIn("already in worktree", result.error.lower())

    def test_active_worktree_property(self):
        tool = EnterWorktreeTool(workspace_dir="/tmp")
        self.assertIsNone(tool.active_worktree)
        tool._active_worktree = "test-wt"
        self.assertEqual(tool.active_worktree, "test-wt")

    def test_list_empty(self):
        """ListWorktrees on non-git dir returns empty."""
        tmpdir = tempfile.mkdtemp()
        try:
            tool = ListWorktreesTool(workspace_dir=tmpdir)
            result = asyncio.get_event_loop().run_until_complete(
                tool.execute(tool_id="t3")
            )
            self.assertTrue(result.success)
            self.assertIn("no active", result.output.lower())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_remove_not_found(self):
        """RemoveWorktree for nonexistent name should error."""
        tmpdir = tempfile.mkdtemp()
        try:
            tool = RemoveWorktreeTool(workspace_dir=tmpdir)
            result = asyncio.get_event_loop().run_until_complete(
                tool.execute(name="nonexistent", tool_id="t4")
            )
            self.assertFalse(result.success)
            self.assertIn("not found", result.error.lower())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_remove_requires_name(self):
        tool = RemoveWorktreeTool(workspace_dir="/tmp")
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(name="", tool_id="t5")
        )
        self.assertFalse(result.success)
        self.assertIn("required", result.error.lower())

    def test_tool_schemas(self):
        """All three worktree tools should have valid schemas."""
        for ToolClass in [EnterWorktreeTool, ListWorktreesTool, RemoveWorktreeTool]:
            tool = ToolClass(workspace_dir="/tmp")
            schema = tool.get_schema()
            self.assertTrue(len(schema.name) > 0)
            self.assertTrue(len(schema.description) > 0)


# ── 4. Scheduler Tools ──────────────────────────────────────────────────────

from cowork_agent.tools.scheduler_tools import (
    CreateScheduledTaskTool, ListScheduledTasksTool, UpdateScheduledTaskTool,
)
from cowork_agent.core.scheduler import TaskScheduler


class TestSchedulerTools(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.scheduler = TaskScheduler(workspace_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_task(self):
        tool = CreateScheduledTaskTool(scheduler=self.scheduler)
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(
                taskId="daily-check",
                prompt="Check email and summarize",
                description="Daily email check",
                cronExpression="0 9 * * *",
                tool_id="t1",
            )
        )
        self.assertTrue(result.success)
        self.assertIn("daily-check", result.output)
        self.assertEqual(len(self.scheduler.list_tasks()), 1)

    def test_create_task_requires_fields(self):
        tool = CreateScheduledTaskTool(scheduler=self.scheduler)
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(taskId="", prompt="test", description="test", tool_id="t2")
        )
        self.assertFalse(result.success)

        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(taskId="test", prompt="", description="test", tool_id="t3")
        )
        self.assertFalse(result.success)

    def test_list_empty(self):
        tool = ListScheduledTasksTool(scheduler=self.scheduler)
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(tool_id="t4")
        )
        self.assertTrue(result.success)
        self.assertIn("no scheduled", result.output.lower())

    def test_list_with_tasks(self):
        create = CreateScheduledTaskTool(scheduler=self.scheduler)
        asyncio.get_event_loop().run_until_complete(
            create.execute(taskId="t1", prompt="Do X", description="Task 1", tool_id="c1")
        )
        asyncio.get_event_loop().run_until_complete(
            create.execute(taskId="t2", prompt="Do Y", description="Task 2",
                           cronExpression="0 10 * * 1-5", tool_id="c2")
        )

        tool = ListScheduledTasksTool(scheduler=self.scheduler)
        result = asyncio.get_event_loop().run_until_complete(
            tool.execute(tool_id="t5")
        )
        self.assertTrue(result.success)
        self.assertIn("t1", result.output)
        self.assertIn("t2", result.output)
        self.assertIn("Task 1", result.output)

    def test_update_task(self):
        create = CreateScheduledTaskTool(scheduler=self.scheduler)
        asyncio.get_event_loop().run_until_complete(
            create.execute(taskId="updatable", prompt="Old", description="Old desc", tool_id="c1")
        )

        update = UpdateScheduledTaskTool(scheduler=self.scheduler)
        result = asyncio.get_event_loop().run_until_complete(
            update.execute(taskId="updatable", description="New desc", tool_id="t6")
        )
        self.assertTrue(result.success)

        tasks = self.scheduler.list_tasks()
        self.assertEqual(tasks[0]["description"], "New desc")

    def test_update_not_found(self):
        update = UpdateScheduledTaskTool(scheduler=self.scheduler)
        result = asyncio.get_event_loop().run_until_complete(
            update.execute(taskId="nonexistent", description="test", tool_id="t7")
        )
        self.assertFalse(result.success)
        self.assertIn("not found", result.error.lower())

    def test_update_no_fields(self):
        create = CreateScheduledTaskTool(scheduler=self.scheduler)
        asyncio.get_event_loop().run_until_complete(
            create.execute(taskId="nochange", prompt="X", description="X", tool_id="c1")
        )

        update = UpdateScheduledTaskTool(scheduler=self.scheduler)
        result = asyncio.get_event_loop().run_until_complete(
            update.execute(taskId="nochange", tool_id="t8")
        )
        self.assertFalse(result.success)
        self.assertIn("no fields", result.error.lower())

    def test_update_disable_enable(self):
        create = CreateScheduledTaskTool(scheduler=self.scheduler)
        asyncio.get_event_loop().run_until_complete(
            create.execute(taskId="toggle", prompt="X", description="X",
                           cronExpression="0 9 * * *", tool_id="c1")
        )

        update = UpdateScheduledTaskTool(scheduler=self.scheduler)
        result = asyncio.get_event_loop().run_until_complete(
            update.execute(taskId="toggle", enabled=False, tool_id="t9")
        )
        self.assertTrue(result.success)

        tasks = self.scheduler.list_tasks()
        self.assertFalse(tasks[0]["enabled"])

    def test_tool_schemas(self):
        for ToolClass in [CreateScheduledTaskTool, ListScheduledTasksTool, UpdateScheduledTaskTool]:
            tool = ToolClass(scheduler=self.scheduler)
            schema = tool.get_schema()
            self.assertTrue(len(schema.name) > 0)
            self.assertTrue(len(schema.description) > 0)


# ── 5. Anthropic Provider Streaming ─────────────────────────────────────────

from cowork_agent.core.providers.anthropic_provider import AnthropicProvider
from cowork_agent.core.models import AgentResponse, ToolCall


class TestAnthropicStreaming(unittest.TestCase):
    def test_has_stream_method(self):
        """AnthropicProvider should override send_message_stream."""
        provider = AnthropicProvider(api_key="test-key")
        # Must have its own send_message_stream (not just inherited)
        self.assertTrue(
            hasattr(provider, "send_message_stream"),
            "AnthropicProvider missing send_message_stream",
        )
        # Check it's actually overridden (not the base class default)
        from cowork_agent.core.providers.base import BaseLLMProvider
        self.assertIsNot(
            type(provider).send_message_stream,
            BaseLLMProvider.send_message_stream,
            "send_message_stream should be overridden, not inherited",
        )

    def test_last_stream_response_property(self):
        """last_stream_response should be accessible."""
        provider = AnthropicProvider(api_key="test-key")
        # Before any call, should be None
        self.assertIsNone(provider.last_stream_response)

    def test_stream_method_is_async_generator(self):
        """send_message_stream should be an async generator function."""
        import inspect
        provider = AnthropicProvider(api_key="test-key")
        self.assertTrue(
            inspect.isasyncgenfunction(provider.send_message_stream),
            "send_message_stream must be an async generator",
        )


# ── 6. OpenAI Provider Streaming ────────────────────────────────────────────

from cowork_agent.core.providers.openai_provider import OpenAIProvider


class TestOpenAIStreaming(unittest.TestCase):
    def test_has_stream_method(self):
        """OpenAIProvider should override send_message_stream."""
        provider = OpenAIProvider(api_key="test-key")
        self.assertTrue(hasattr(provider, "send_message_stream"))
        from cowork_agent.core.providers.base import BaseLLMProvider
        self.assertIsNot(
            type(provider).send_message_stream,
            BaseLLMProvider.send_message_stream,
            "send_message_stream should be overridden",
        )

    def test_last_stream_response_property(self):
        provider = OpenAIProvider(api_key="test-key")
        self.assertIsNone(provider.last_stream_response)

    def test_stream_method_is_async_generator(self):
        import inspect
        provider = OpenAIProvider(api_key="test-key")
        self.assertTrue(
            inspect.isasyncgenfunction(provider.send_message_stream),
            "send_message_stream must be an async generator",
        )


# ── 7. Session Persistence ──────────────────────────────────────────────────

from cowork_agent.core.session_manager import SessionManager, SessionMetadata
from cowork_agent.core.models import Message, ToolCall, ToolResult


class TestSessionPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manager = SessionManager(workspace_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_session(self):
        sid = self.manager.create_session(provider="test", model="test-model")
        self.assertTrue(len(sid) > 0)

        # Session directory should exist
        session_dir = os.path.join(self.tmpdir, ".cowork", "sessions", sid)
        self.assertTrue(os.path.isdir(session_dir))

        # Metadata file should exist
        self.assertTrue(os.path.exists(os.path.join(session_dir, "metadata.json")))
        self.assertTrue(os.path.exists(os.path.join(session_dir, "messages.jsonl")))

    def test_save_and_load_messages(self):
        sid = self.manager.create_session()

        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="assistant", content="Hi there!")
        self.manager.save_message(sid, msg1)
        self.manager.save_message(sid, msg2)

        loaded = self.manager.load_messages(sid)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].role, "user")
        self.assertEqual(loaded[0].content, "Hello")
        self.assertEqual(loaded[1].role, "assistant")
        self.assertEqual(loaded[1].content, "Hi there!")

    def test_save_message_with_tool_calls(self):
        sid = self.manager.create_session()

        msg = Message(
            role="assistant",
            content="Let me read that.",
            tool_calls=[
                ToolCall(name="read", tool_id="tc_001", input={"file_path": "/test.py"}),
            ],
        )
        self.manager.save_message(sid, msg)

        loaded = self.manager.load_messages(sid)
        self.assertEqual(len(loaded), 1)
        self.assertIsNotNone(loaded[0].tool_calls)
        self.assertEqual(len(loaded[0].tool_calls), 1)
        self.assertEqual(loaded[0].tool_calls[0].name, "read")
        self.assertEqual(loaded[0].tool_calls[0].input["file_path"], "/test.py")

    def test_save_message_with_tool_results(self):
        sid = self.manager.create_session()

        msg = Message(
            role="tool_result",
            content="",
            tool_results=[
                ToolResult(tool_id="tc_001", success=True, output="file contents here"),
            ],
        )
        self.manager.save_message(sid, msg)

        loaded = self.manager.load_messages(sid)
        self.assertEqual(len(loaded), 1)
        self.assertIsNotNone(loaded[0].tool_results)
        self.assertTrue(loaded[0].tool_results[0].success)
        self.assertEqual(loaded[0].tool_results[0].output, "file contents here")

    def test_list_sessions(self):
        sid1 = self.manager.create_session(title="Session 1")
        time.sleep(0.05)
        sid2 = self.manager.create_session(title="Session 2")

        sessions = self.manager.list_sessions()
        self.assertEqual(len(sessions), 2)
        # Newest first
        self.assertEqual(sessions[0].session_id, sid2)
        self.assertEqual(sessions[1].session_id, sid1)

    def test_delete_session(self):
        sid = self.manager.create_session()
        result = self.manager.delete_session(sid)
        self.assertIn("deleted", result.lower())

        # Should be gone
        sessions = self.manager.list_sessions()
        self.assertEqual(len(sessions), 0)

    def test_delete_nonexistent(self):
        result = self.manager.delete_session("nonexistent-id")
        self.assertIn("not found", result.lower())

    def test_update_title(self):
        sid = self.manager.create_session(title="Original")
        self.manager.update_title(sid, "Updated Title")

        meta = self.manager.get_metadata(sid)
        self.assertEqual(meta.title, "Updated Title")

    def test_metadata_message_count(self):
        sid = self.manager.create_session()
        self.manager.save_message(sid, Message(role="user", content="A"))
        self.manager.save_message(sid, Message(role="assistant", content="B"))

        meta = self.manager.get_metadata(sid)
        self.assertEqual(meta.message_count, 2)

    def test_load_nonexistent_session(self):
        loaded = self.manager.load_messages("nonexistent-id")
        self.assertEqual(loaded, [])

    def test_session_metadata_serialization(self):
        meta = SessionMetadata(
            session_id="abc123",
            created_at=1000.0,
            updated_at=2000.0,
            message_count=5,
            provider="ollama",
            model="qwen3",
            title="Test Session",
        )
        d = meta.to_dict()
        restored = SessionMetadata.from_dict(d)
        self.assertEqual(restored.session_id, "abc123")
        self.assertEqual(restored.message_count, 5)
        self.assertEqual(restored.model, "qwen3")


if __name__ == "__main__":
    unittest.main()
