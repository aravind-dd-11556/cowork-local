"""
Sprint 12 Tests — Wire All Unconnected Modules into Working System

Tests for: ResponseCache, StreamHardener, RetryExecutor, Multimodal,
NotebookEditTool, TaskScheduler, SchedulerTools, WorktreeManager,
WorktreeTools, PluginSystem, MCPClient, AgentRegistry, Supervisor,
DelegateTaskTool, and Agent integration with cache/hardener.

Run: python -m pytest cowork_agent/tests/test_p12.py -v
"""

import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from collections import OrderedDict

# ── Ensure project root is on path ──────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cowork_agent.core.models import (
    Message, ToolCall, ToolResult, ToolSchema, AgentResponse,
)
from cowork_agent.core.response_cache import ResponseCache, CacheEntry
from cowork_agent.core.stream_hardener import StreamHardener, StreamTimeoutError
from cowork_agent.core.retry import (
    RetryExecutor, RetryPolicy, RetryResult, with_retry, retry_async,
)
from cowork_agent.core.multimodal import (
    ImageContent, MultiModalMessage, load_image, extract_image_paths,
    parse_multimodal_input, SUPPORTED_IMAGE_TYPES, MAX_IMAGE_SIZE,
)
from cowork_agent.tools.notebook_edit import NotebookEditTool
from cowork_agent.core.scheduler import TaskScheduler, ScheduledTask
from cowork_agent.tools.scheduler_tools import (
    CreateScheduledTaskTool, ListScheduledTasksTool, UpdateScheduledTaskTool,
)
from cowork_agent.core.worktree import WorktreeManager, WorktreeInfo
from cowork_agent.tools.worktree_tool import (
    EnterWorktreeTool, ListWorktreesTool, RemoveWorktreeTool,
)
from cowork_agent.core.plugin_system import PluginSystem, PluginInfo
from cowork_agent.core.mcp_client import MCPClient, MCPServerConfig, MCPTool
from cowork_agent.tools.mcp_bridge import MCPBridgeTool, register_mcp_tools
from cowork_agent.core.context_bus import ContextBus, BusMessage, MessageType
from cowork_agent.core.agent_registry import (
    AgentRegistry, AgentConfig, AgentInstance, AgentState,
)
from cowork_agent.core.supervisor import (
    Supervisor, SupervisorConfig, ExecutionStrategy, SubTask,
)
from cowork_agent.core.delegate_tool import DelegateTaskTool, DelegateMode, DelegatedTask


# ── Helpers ──────────────────────────────────────────────────────

def _run(coro):
    """Run an async function synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_msg(role="user", content="hello", tool_calls=None,
              tool_results=None, timestamp=None):
    return Message(
        role=role, content=content, tool_calls=tool_calls,
        tool_results=tool_results, timestamp=timestamp or time.time(),
    )


def _cacheable_response(text="Hello world"):
    return AgentResponse(text=text, stop_reason="end_turn")


def _tool_response():
    return AgentResponse(
        text="Using tool",
        stop_reason="tool_use",
        tool_calls=[ToolCall(name="bash", tool_id="t1", input={"command": "ls"})],
    )


# ══════════════════════════════════════════════════════════════════
# 1. ResponseCache
# ══════════════════════════════════════════════════════════════════

class TestResponseCache(unittest.TestCase):
    """Tests for ResponseCache (LRU + TTL)."""

    def test_make_key_deterministic(self):
        cache = ResponseCache()
        msgs = [_make_msg(content="hi")]
        k1 = cache.make_key("m1", msgs, "sys")
        k2 = cache.make_key("m1", msgs, "sys")
        self.assertEqual(k1, k2)

    def test_make_key_varies_with_model(self):
        cache = ResponseCache()
        msgs = [_make_msg()]
        k1 = cache.make_key("model-a", msgs, "sys")
        k2 = cache.make_key("model-b", msgs, "sys")
        self.assertNotEqual(k1, k2)

    def test_put_and_get_hit(self):
        cache = ResponseCache()
        resp = _cacheable_response("hello")
        key = "test-key"
        cache.put(key, resp)
        self.assertEqual(cache.get(key).text, "hello")
        self.assertEqual(cache._hits, 1)

    def test_get_miss(self):
        cache = ResponseCache()
        self.assertIsNone(cache.get("nonexistent"))
        self.assertEqual(cache._misses, 1)

    def test_ttl_expiration(self):
        cache = ResponseCache(ttl=0.01)
        resp = _cacheable_response()
        cache.put("k", resp)
        time.sleep(0.02)
        self.assertIsNone(cache.get("k"))

    def test_lru_eviction(self):
        cache = ResponseCache(max_size=2)
        cache.put("a", _cacheable_response("A"))
        cache.put("b", _cacheable_response("B"))
        cache.put("c", _cacheable_response("C"))
        # 'a' should have been evicted (LRU)
        self.assertIsNone(cache.get("a"))
        self.assertIsNotNone(cache.get("b"))
        self.assertIsNotNone(cache.get("c"))

    def test_not_cacheable_tool_response(self):
        cache = ResponseCache()
        resp = _tool_response()
        self.assertFalse(cache.put("k", resp))
        self.assertIsNone(cache.get("k"))

    def test_not_cacheable_empty_text(self):
        cache = ResponseCache()
        resp = AgentResponse(text="", stop_reason="end_turn")
        self.assertFalse(cache.put("k", resp))

    def test_disabled_cache(self):
        cache = ResponseCache(enabled=False)
        cache.put("k", _cacheable_response())
        self.assertIsNone(cache.get("k"))

    def test_invalidate(self):
        cache = ResponseCache()
        cache.put("k", _cacheable_response())
        self.assertTrue(cache.invalidate("k"))
        self.assertIsNone(cache.get("k"))

    def test_clear(self):
        cache = ResponseCache()
        cache.put("a", _cacheable_response())
        cache.put("b", _cacheable_response())
        cache.clear()
        self.assertEqual(cache.size, 0)

    def test_stats(self):
        cache = ResponseCache(max_size=50, ttl=1800)
        cache.put("k", _cacheable_response())
        cache.get("k")
        cache.get("miss")
        stats = cache.stats()
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["max_size"], 50)


# ══════════════════════════════════════════════════════════════════
# 2. StreamHardener
# ══════════════════════════════════════════════════════════════════

class TestStreamHardener(unittest.TestCase):
    """Tests for StreamHardener (timeout wrapping)."""

    def test_normal_stream(self):
        """Normal chunks yield correctly."""
        async def gen():
            yield "hello "
            yield "world"

        hardener = StreamHardener()
        chunks = []
        async def collect():
            async for c in hardener.wrap(gen()):
                chunks.append(c)
        _run(collect())
        self.assertEqual("".join(chunks), "hello world")
        self.assertTrue(hardener.completed)
        self.assertFalse(hardener.timed_out)

    def test_partial_text_accumulation(self):
        async def gen():
            yield "A"
            yield "B"
            yield "C"

        hardener = StreamHardener()
        async def collect():
            async for _ in hardener.wrap(gen()):
                pass
        _run(collect())
        self.assertEqual(hardener.partial_text, "ABC")
        self.assertEqual(hardener.chunk_count, 3)

    def test_skip_empty_chunks(self):
        async def gen():
            yield ""
            yield "real"
            yield ""

        hardener = StreamHardener()
        chunks = []
        async def collect():
            async for c in hardener.wrap(gen()):
                chunks.append(c)
        _run(collect())
        self.assertEqual(chunks, ["real"])

    def test_skip_non_string_chunks(self):
        async def gen():
            yield 123  # type: ignore
            yield "text"

        hardener = StreamHardener()
        chunks = []
        async def collect():
            async for c in hardener.wrap(gen()):
                chunks.append(c)
        _run(collect())
        self.assertEqual(chunks, ["text"])

    def test_chunk_timeout(self):
        async def gen():
            yield "start"
            await asyncio.sleep(10)  # Stall
            yield "never"

        hardener = StreamHardener(chunk_timeout=0.05)
        async def collect():
            async for _ in hardener.wrap(gen()):
                pass

        with self.assertRaises(StreamTimeoutError):
            _run(collect())
        self.assertTrue(hardener.timed_out)
        self.assertEqual(hardener.partial_text, "start")

    def test_total_timeout(self):
        async def gen():
            for i in range(100):
                yield f"c{i}"
                await asyncio.sleep(0.02)

        hardener = StreamHardener(chunk_timeout=5.0, total_timeout=0.05)
        async def collect():
            async for _ in hardener.wrap(gen()):
                pass

        with self.assertRaises(StreamTimeoutError):
            _run(collect())
        self.assertTrue(hardener.timed_out)

    def test_build_partial_response_timeout(self):
        hardener = StreamHardener()
        hardener._partial_text = "partial text"
        hardener._timed_out = True
        resp = hardener.build_partial_response()
        self.assertEqual(resp.text, "partial text")
        self.assertEqual(resp.stop_reason, "max_tokens")

    def test_build_partial_response_completed(self):
        hardener = StreamHardener()
        hardener._partial_text = "full"
        hardener._completed = True
        resp = hardener.build_partial_response()
        self.assertEqual(resp.stop_reason, "end_turn")


# ══════════════════════════════════════════════════════════════════
# 3. RetryExecutor
# ══════════════════════════════════════════════════════════════════

class TestRetryExecutor(unittest.TestCase):
    """Tests for RetryExecutor (backoff + jitter)."""

    def test_success_first_try(self):
        async def ok():
            return "done"
        executor = RetryExecutor()
        result = _run(executor.execute(ok))
        self.assertTrue(result.success)
        self.assertEqual(result.result, "done")
        self.assertEqual(result.attempts, 1)

    def test_retry_on_connection_error(self):
        call_count = 0
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "recovered"

        policy = RetryPolicy(max_attempts=3, backoff_base=0.01, jitter=False)
        executor = RetryExecutor(policy)
        result = _run(executor.execute(flaky))
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.result, "recovered")

    def test_max_attempts_exhausted(self):
        async def always_fail():
            raise ConnectionError("nope")

        policy = RetryPolicy(max_attempts=2, backoff_base=0.01, jitter=False)
        executor = RetryExecutor(policy)
        result = _run(executor.execute(always_fail))
        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.errors), 2)

    def test_non_retryable_error_stops_immediately(self):
        async def bad():
            raise ValueError("bad input")

        # transient_only=False so it doesn't consult ErrorCatalog
        policy = RetryPolicy(max_attempts=3, backoff_base=0.01, transient_only=False)
        executor = RetryExecutor(policy)
        result = _run(executor.execute(bad))
        self.assertFalse(result.success)
        # ValueError is not in retryable_exceptions, so should stop immediately
        self.assertEqual(result.attempts, 1)
        self.assertIsInstance(result.last_error, ValueError)

    def test_jitter_varies_delay(self):
        policy = RetryPolicy(jitter=True, backoff_base=1.0)
        executor = RetryExecutor(policy)
        delays = [executor._calculate_delay(1) for _ in range(20)]
        # With jitter, not all delays should be identical
        self.assertTrue(len(set(delays)) > 1)

    def test_backoff_capped(self):
        policy = RetryPolicy(backoff_base=10.0, backoff_max=15.0, jitter=False)
        executor = RetryExecutor(policy)
        delay = executor._calculate_delay(10)  # Would be huge without cap
        self.assertLessEqual(delay, 15.0)

    def test_decorator(self):
        call_count = 0

        @with_retry(RetryPolicy(max_attempts=2, backoff_base=0.01, jitter=False))
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("retry me")
            return "ok"

        result = _run(flaky_func())
        self.assertEqual(result, "ok")

    def test_standalone_retry_async(self):
        call_count = 0
        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("timeout")
            return 42

        result = _run(retry_async(fn, policy=RetryPolicy(max_attempts=3, backoff_base=0.01, jitter=False)))
        self.assertEqual(result, 42)

    def test_retry_result_total_delay(self):
        async def fail():
            raise ConnectionError("fail")

        policy = RetryPolicy(max_attempts=2, backoff_base=0.01, jitter=False)
        executor = RetryExecutor(policy)
        result = _run(executor.execute(fail))
        self.assertGreater(result.total_delay, 0)

    def test_default_policy(self):
        executor = RetryExecutor()
        self.assertEqual(executor.policy.max_attempts, 3)
        self.assertTrue(executor.policy.jitter)


# ══════════════════════════════════════════════════════════════════
# 4. Multimodal
# ══════════════════════════════════════════════════════════════════

class TestMultimodal(unittest.TestCase):
    """Tests for multimodal image handling."""

    def test_image_content_to_anthropic_block(self):
        img = ImageContent(media_type="image/png", base64_data="abc123")
        block = img.to_anthropic_block()
        self.assertEqual(block["type"], "image")
        self.assertEqual(block["source"]["media_type"], "image/png")
        self.assertEqual(block["source"]["data"], "abc123")

    def test_image_content_to_openai_block(self):
        img = ImageContent(media_type="image/jpeg", base64_data="xyz")
        block = img.to_openai_block()
        self.assertEqual(block["type"], "image_url")
        self.assertIn("data:image/jpeg;base64,xyz", block["image_url"]["url"])

    def test_load_image_missing_file(self):
        result = load_image("/nonexistent/file.png")
        self.assertIsNone(result)

    def test_load_image_unsupported_type(self):
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            f.write(b"fake image")
            path = f.name
        try:
            result = load_image(path)
            self.assertIsNone(result)
        finally:
            os.unlink(path)

    def test_load_image_success(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            path = f.name
        try:
            result = load_image(path)
            self.assertIsNotNone(result)
            self.assertEqual(result.media_type, "image/png")
            self.assertTrue(len(result.base64_data) > 0)
        finally:
            os.unlink(path)

    def test_load_image_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name  # 0 bytes
        try:
            result = load_image(path)
            self.assertIsNone(result)
        finally:
            os.unlink(path)

    def test_multimodal_message_no_images(self):
        mm = MultiModalMessage(text="hello")
        self.assertFalse(mm.has_images)
        content = mm.to_anthropic_content()
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")

    def test_supported_image_types(self):
        self.assertIn(".png", SUPPORTED_IMAGE_TYPES)
        self.assertIn(".jpg", SUPPORTED_IMAGE_TYPES)
        self.assertIn(".webp", SUPPORTED_IMAGE_TYPES)


# ══════════════════════════════════════════════════════════════════
# 5. NotebookEditTool
# ══════════════════════════════════════════════════════════════════

class TestNotebookEditTool(unittest.TestCase):
    """Tests for NotebookEditTool."""

    def _make_notebook(self, cells=None):
        if cells is None:
            cells = [
                {"cell_type": "code", "source": ["print('hello')\n"], "metadata": {},
                 "outputs": [], "execution_count": 1},
                {"cell_type": "markdown", "source": ["# Title\n"], "metadata": {}},
            ]
        nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}
        f = tempfile.NamedTemporaryFile(suffix=".ipynb", mode="w", delete=False)
        json.dump(nb, f)
        f.close()
        return f.name

    def test_replace_cell(self):
        path = self._make_notebook()
        try:
            tool = NotebookEditTool()
            result = _run(tool.execute(
                notebook_path=path, new_source="print('world')",
                cell_number=0, edit_mode="replace",
            ))
            self.assertTrue(result.success)
            with open(path) as f:
                nb = json.load(f)
            self.assertIn("world", "".join(nb["cells"][0]["source"]))
        finally:
            os.unlink(path)

    def test_insert_cell(self):
        path = self._make_notebook()
        try:
            tool = NotebookEditTool()
            result = _run(tool.execute(
                notebook_path=path, new_source="# New",
                cell_number=1, cell_type="markdown", edit_mode="insert",
            ))
            self.assertTrue(result.success)
            with open(path) as f:
                nb = json.load(f)
            self.assertEqual(len(nb["cells"]), 3)
        finally:
            os.unlink(path)

    def test_delete_cell(self):
        path = self._make_notebook()
        try:
            tool = NotebookEditTool()
            result = _run(tool.execute(
                notebook_path=path, new_source="", cell_number=0, edit_mode="delete",
            ))
            self.assertTrue(result.success)
            with open(path) as f:
                nb = json.load(f)
            self.assertEqual(len(nb["cells"]), 1)
        finally:
            os.unlink(path)

    def test_out_of_range_cell(self):
        path = self._make_notebook()
        try:
            tool = NotebookEditTool()
            result = _run(tool.execute(
                notebook_path=path, new_source="x", cell_number=99,
            ))
            self.assertFalse(result.success)
        finally:
            os.unlink(path)

    def test_non_absolute_path(self):
        tool = NotebookEditTool()
        result = _run(tool.execute(notebook_path="relative.ipynb", new_source="x"))
        self.assertFalse(result.success)

    def test_non_ipynb_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a notebook")
            path = f.name
        try:
            tool = NotebookEditTool()
            result = _run(tool.execute(notebook_path=path, new_source="x"))
            self.assertFalse(result.success)
        finally:
            os.unlink(path)

    def test_insert_requires_cell_type(self):
        path = self._make_notebook()
        try:
            tool = NotebookEditTool()
            result = _run(tool.execute(
                notebook_path=path, new_source="x", edit_mode="insert",
            ))
            self.assertFalse(result.success)
            self.assertIn("cell_type", result.output or result.error or "")
        finally:
            os.unlink(path)

    def test_source_lines_formatting(self):
        lines = NotebookEditTool._to_source_lines("line1\nline2\nline3")
        self.assertEqual(lines, ["line1\n", "line2\n", "line3"])


# ══════════════════════════════════════════════════════════════════
# 6. TaskScheduler
# ══════════════════════════════════════════════════════════════════

class TestTaskScheduler(unittest.TestCase):
    """Tests for TaskScheduler."""

    def _make_scheduler(self):
        d = tempfile.mkdtemp()
        return TaskScheduler(workspace_dir=d), d

    def test_create_task(self):
        sched, d = self._make_scheduler()
        task = ScheduledTask(
            task_id="test-task", prompt="do stuff", description="Test task",
        )
        result = sched.create_task(task)
        self.assertIn("test-task", result.lower()) or self.assertTrue(len(result) > 0)

    def test_list_tasks(self):
        sched, d = self._make_scheduler()
        task = ScheduledTask(task_id="t1", prompt="p", description="d")
        sched.create_task(task)
        tasks = sched.list_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "t1")

    def test_update_task(self):
        sched, d = self._make_scheduler()
        task = ScheduledTask(task_id="t1", prompt="old", description="d")
        sched.create_task(task)
        result = sched.update_task("t1", prompt="new")
        tasks = sched.list_tasks()
        self.assertEqual(tasks[0].get("prompt", ""), "new")

    def test_update_nonexistent(self):
        sched, d = self._make_scheduler()
        result = sched.update_task("nope", prompt="x")
        self.assertIn("not found", result.lower())

    def test_create_with_cron(self):
        sched, d = self._make_scheduler()
        task = ScheduledTask(
            task_id="daily", prompt="p", description="d",
            cron_expression="0 9 * * *",
        )
        sched.create_task(task)
        tasks = sched.list_tasks()
        self.assertEqual(tasks[0].get("cron_expression"), "0 9 * * *")

    def test_persistence(self):
        d = tempfile.mkdtemp()
        sched1 = TaskScheduler(workspace_dir=d)
        task = ScheduledTask(task_id="persist", prompt="p", description="d")
        sched1.create_task(task)
        # Create new scheduler and load from same dir
        sched2 = TaskScheduler(workspace_dir=d)
        sched2.load()
        tasks = sched2.list_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "persist")

    def test_disable_task(self):
        sched, d = self._make_scheduler()
        task = ScheduledTask(task_id="t1", prompt="p", description="d")
        sched.create_task(task)
        sched.update_task("t1", enabled=False)
        tasks = sched.list_tasks()
        self.assertFalse(tasks[0].get("enabled", True))

    def test_task_to_dict(self):
        task = ScheduledTask(task_id="x", prompt="p", description="d", cron_expression="* * * * *")
        d = task.to_dict()
        self.assertEqual(d["task_id"], "x")
        self.assertEqual(d["cron_expression"], "* * * * *")

    def test_task_from_dict(self):
        data = {"task_id": "y", "prompt": "p", "description": "d", "enabled": True}
        task = ScheduledTask.from_dict(data)
        self.assertEqual(task.task_id, "y")

    def test_empty_workspace(self):
        sched = TaskScheduler()  # No workspace
        tasks = sched.list_tasks()
        self.assertEqual(len(tasks), 0)


# ══════════════════════════════════════════════════════════════════
# 7. Scheduler Tools
# ══════════════════════════════════════════════════════════════════

class TestSchedulerTools(unittest.TestCase):
    """Tests for Create/List/Update scheduled task tools."""

    def _make_scheduler(self):
        d = tempfile.mkdtemp()
        return TaskScheduler(workspace_dir=d)

    def test_create_tool(self):
        sched = self._make_scheduler()
        tool = CreateScheduledTaskTool(scheduler=sched)
        result = _run(tool.execute(
            taskId="test-task", prompt="do stuff", description="Test task",
        ))
        self.assertTrue(result.success)

    def test_create_tool_missing_id(self):
        tool = CreateScheduledTaskTool(scheduler=self._make_scheduler())
        result = _run(tool.execute(taskId="", prompt="p", description="d"))
        self.assertFalse(result.success)

    def test_list_tool(self):
        sched = self._make_scheduler()
        sched.create_task(ScheduledTask(task_id="t1", prompt="p", description="d"))
        tool = ListScheduledTasksTool(scheduler=sched)
        result = _run(tool.execute())
        self.assertTrue(result.success)
        self.assertIn("t1", result.output)

    def test_list_tool_empty(self):
        tool = ListScheduledTasksTool(scheduler=self._make_scheduler())
        result = _run(tool.execute())
        self.assertTrue(result.success)
        self.assertIn("No scheduled tasks", result.output)

    def test_update_tool(self):
        sched = self._make_scheduler()
        sched.create_task(ScheduledTask(task_id="t1", prompt="old", description="d"))
        tool = UpdateScheduledTaskTool(scheduler=sched)
        result = _run(tool.execute(taskId="t1", prompt="new"))
        self.assertTrue(result.success)

    def test_update_tool_no_fields(self):
        tool = UpdateScheduledTaskTool(scheduler=self._make_scheduler())
        result = _run(tool.execute(taskId="t1"))
        self.assertFalse(result.success)


# ══════════════════════════════════════════════════════════════════
# 8. WorktreeManager
# ══════════════════════════════════════════════════════════════════

class TestWorktreeManager(unittest.TestCase):
    """Tests for WorktreeManager."""

    def test_not_git_repo(self):
        d = tempfile.mkdtemp()
        mgr = WorktreeManager(workspace_dir=d)
        self.assertFalse(mgr.is_git_repo())

    def test_create_without_git(self):
        d = tempfile.mkdtemp()
        mgr = WorktreeManager(workspace_dir=d)
        result = mgr.create("test-wt")
        self.assertIsNone(result)

    def test_list_empty(self):
        d = tempfile.mkdtemp()
        mgr = WorktreeManager(workspace_dir=d)
        self.assertEqual(mgr.list_worktrees(), [])

    def test_remove_nonexistent(self):
        d = tempfile.mkdtemp()
        mgr = WorktreeManager(workspace_dir=d)
        result = mgr.remove("nope")
        self.assertIn("not found", result.lower())

    def test_random_name_format(self):
        name = WorktreeManager._random_name()
        parts = name.split("-")
        self.assertEqual(len(parts), 3)

    def test_worktree_info_dataclass(self):
        info = WorktreeInfo(
            name="test", path="/tmp/test", branch="worktree/test",
            created_from="main",
        )
        self.assertEqual(info.name, "test")
        self.assertFalse(info.has_changes)

    @patch("subprocess.run")
    def test_is_git_repo_true(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        mgr = WorktreeManager(workspace_dir="/tmp/fake")
        self.assertTrue(mgr.is_git_repo())

    @patch("subprocess.run")
    def test_has_changes(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="M file.py\n")
        self.assertTrue(WorktreeManager._has_changes("/tmp/fake"))


# ══════════════════════════════════════════════════════════════════
# 9. Worktree Tools
# ══════════════════════════════════════════════════════════════════

class TestWorktreeTools(unittest.TestCase):
    """Tests for Enter/List/Remove worktree tools."""

    def test_enter_not_git_repo(self):
        d = tempfile.mkdtemp()
        tool = EnterWorktreeTool(workspace_dir=d)
        result = _run(tool.execute(name="test"))
        self.assertFalse(result.success)
        self.assertIn("git", result.output.lower() or result.error.lower())

    def test_enter_nested_worktree(self):
        d = tempfile.mkdtemp()
        tool = EnterWorktreeTool(workspace_dir=d)
        tool._active_worktree = "existing"
        result = _run(tool.execute(name="new"))
        self.assertFalse(result.success)
        self.assertIn("nested", result.output.lower() or result.error.lower())

    def test_list_empty(self):
        d = tempfile.mkdtemp()
        tool = ListWorktreesTool(workspace_dir=d)
        result = _run(tool.execute())
        self.assertTrue(result.success)
        self.assertIn("No active worktrees", result.output)

    def test_remove_nonexistent(self):
        d = tempfile.mkdtemp()
        tool = RemoveWorktreeTool(workspace_dir=d)
        result = _run(tool.execute(name="nope"))
        self.assertFalse(result.success)

    def test_remove_missing_name(self):
        tool = RemoveWorktreeTool(workspace_dir=tempfile.mkdtemp())
        result = _run(tool.execute(name=""))
        self.assertFalse(result.success)

    def test_enter_set_bash_tool(self):
        tool = EnterWorktreeTool(workspace_dir=tempfile.mkdtemp())
        mock_bash = MagicMock()
        tool.set_bash_tool(mock_bash)
        self.assertEqual(tool._bash_tool, mock_bash)


# ══════════════════════════════════════════════════════════════════
# 10. PluginSystem
# ══════════════════════════════════════════════════════════════════

class TestPluginSystem(unittest.TestCase):
    """Tests for PluginSystem."""

    def test_no_plugins_dir(self):
        ps = PluginSystem(workspace_dir="/nonexistent/path")
        loaded = ps.discover_and_load()
        self.assertEqual(len(loaded), 0)

    def test_empty_plugins_dir(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, ".cowork", "plugins"), exist_ok=True)
        ps = PluginSystem(workspace_dir=d)
        loaded = ps.discover_and_load()
        self.assertEqual(len(loaded), 0)

    def test_plugin_info_dataclass(self):
        info = PluginInfo(
            name="test", description="Test plugin",
            version="1.0", location="/tmp/test",
        )
        self.assertEqual(info.name, "test")
        self.assertTrue(info.enabled)

    def test_plugin_names_empty(self):
        ps = PluginSystem()
        self.assertEqual(ps.plugin_names, [])

    def test_plugins_property(self):
        ps = PluginSystem()
        self.assertEqual(ps.plugins, {})

    def test_user_plugins_dir_default(self):
        ps = PluginSystem()
        self.assertIn(".cowork_agent/plugins", ps.user_plugins_dir)


# ══════════════════════════════════════════════════════════════════
# 11. MCPClient
# ══════════════════════════════════════════════════════════════════

class TestMCPClient(unittest.TestCase):
    """Tests for MCPClient."""

    def test_add_server(self):
        client = MCPClient()
        config = MCPServerConfig(name="test", command="echo", args=["hello"])
        client.add_server(config)
        self.assertIn("test", client._servers)

    def test_get_tools_empty(self):
        client = MCPClient()
        self.assertEqual(client.get_tools(), [])

    def test_mcp_tool_dataclass(self):
        tool = MCPTool(
            name="test_tool", description="A test",
            input_schema={"type": "object"}, server_name="srv",
        )
        self.assertEqual(tool.name, "test_tool")

    def test_mcp_server_config_defaults(self):
        cfg = MCPServerConfig(name="test")
        self.assertEqual(cfg.transport, "stdio")
        self.assertEqual(cfg.url, "")
        self.assertEqual(cfg.args, [])

    def test_bridge_tool(self):
        mcp_tool = MCPTool(
            name="my_tool", description="desc",
            input_schema={"type": "object"}, server_name="srv",
        )
        client = MCPClient()
        bridge = MCPBridgeTool(mcp_tool, client)
        self.assertEqual(bridge.name, "my_tool")
        self.assertIn("MCP:srv", bridge.description)

    def test_register_mcp_tools(self):
        client = MCPClient()
        client._tools = {
            "tool1": MCPTool("tool1", "d1", {}, "srv"),
            "tool2": MCPTool("tool2", "d2", {}, "srv"),
        }
        registry = MagicMock()
        count = register_mcp_tools(registry, client)
        self.assertEqual(count, 2)
        self.assertEqual(registry.register.call_count, 2)


# ══════════════════════════════════════════════════════════════════
# 12. ContextBus
# ══════════════════════════════════════════════════════════════════

class TestContextBus(unittest.TestCase):
    """Tests for ContextBus (pub/sub)."""

    def test_publish_and_subscribe(self):
        bus = ContextBus()
        received = []

        async def handler(msg):
            received.append(msg)

        # subscribe(sender, msg_type, callback)
        bus.subscribe("agent1", MessageType.DATA_SHARE, handler)
        msg = BusMessage(
            msg_type=MessageType.DATA_SHARE,
            sender="agent1", content="data", topic="test_topic",
        )
        _run(bus.publish(msg))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].content, "data")

    def test_shared_store(self):
        bus = ContextBus()
        _run(bus.set_shared("key1", "value1"))
        result = _run(bus.get_shared("key1"))
        self.assertEqual(result, "value1")

    def test_shared_store_default(self):
        bus = ContextBus()
        result = _run(bus.get_shared("missing"))
        self.assertIsNone(result)
        result2 = _run(bus.get_shared("missing", "default"))
        self.assertEqual(result2, "default")

    def test_message_types(self):
        self.assertEqual(MessageType.TASK_RESULT.value, "task_result")
        self.assertEqual(MessageType.ERROR.value, "error")


# ══════════════════════════════════════════════════════════════════
# 13. AgentRegistry
# ══════════════════════════════════════════════════════════════════

class TestAgentRegistry(unittest.TestCase):
    """Tests for AgentRegistry (lifecycle management)."""

    def test_empty_registry(self):
        reg = AgentRegistry()
        self.assertEqual(len(reg._agents), 0)

    def test_agent_config_dataclass(self):
        cfg = AgentConfig(name="reader", role="file_reader")
        self.assertEqual(cfg.name, "reader")
        self.assertEqual(cfg.max_iterations, 15)

    def test_agent_instance_not_running(self):
        cfg = AgentConfig(name="test", role="tester")
        agent = MagicMock()
        inst = AgentInstance(config=cfg, agent=agent)
        self.assertFalse(inst.is_running)
        self.assertFalse(inst.is_completed)

    def test_agent_instance_to_dict(self):
        cfg = AgentConfig(name="test", role="tester")
        agent = MagicMock()
        inst = AgentInstance(config=cfg, agent=agent)
        d = inst.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["state"], "idle")

    def test_agent_state_enum(self):
        self.assertEqual(AgentState.IDLE.value, "idle")
        self.assertEqual(AgentState.RUNNING.value, "running")
        self.assertEqual(AgentState.COMPLETED.value, "completed")

    def test_create_agent(self):
        reg = AgentRegistry()
        cfg = AgentConfig(name="test", role="tester")
        provider = MagicMock()
        tool_reg = MagicMock()
        tool_reg.tool_names = []
        tool_reg.get_tool.return_value = MagicMock()
        builder = MagicMock()
        inst = _run(reg.create_agent(cfg, provider, tool_reg, builder))
        self.assertIsNotNone(inst)
        self.assertEqual(inst.config.name, "test")

    def test_create_duplicate_agent(self):
        reg = AgentRegistry()
        cfg = AgentConfig(name="dup", role="tester")
        provider = MagicMock()
        tool_reg = MagicMock()
        tool_reg.tool_names = []
        tool_reg.get_tool.return_value = MagicMock()
        builder = MagicMock()
        _run(reg.create_agent(cfg, provider, tool_reg, builder))
        with self.assertRaises(ValueError):
            _run(reg.create_agent(cfg, provider, tool_reg, builder))

    def test_agent_instance_elapsed(self):
        cfg = AgentConfig(name="t", role="r")
        inst = AgentInstance(config=cfg, agent=MagicMock())
        self.assertEqual(inst.elapsed, 0.0)
        inst.started_at = time.time() - 5
        self.assertGreater(inst.elapsed, 4.0)


# ══════════════════════════════════════════════════════════════════
# 14. Supervisor
# ══════════════════════════════════════════════════════════════════

class TestSupervisor(unittest.TestCase):
    """Tests for Supervisor (multi-agent orchestration)."""

    def test_supervisor_config_defaults(self):
        cfg = SupervisorConfig()
        self.assertEqual(cfg.strategy, ExecutionStrategy.SEQUENTIAL)
        self.assertEqual(cfg.on_failure, "abort")

    def test_execution_strategy_enum(self):
        self.assertEqual(ExecutionStrategy.SEQUENTIAL.value, "sequential")
        self.assertEqual(ExecutionStrategy.PARALLEL.value, "parallel")
        self.assertEqual(ExecutionStrategy.PIPELINE.value, "pipeline")

    def test_subtask_generate_id(self):
        id1 = SubTask.generate_id()
        id2 = SubTask.generate_id()
        self.assertTrue(id1.startswith("sub_"))
        self.assertNotEqual(id1, id2)

    def test_subtask_not_completed(self):
        st = SubTask(agent_name="test", description="desc")
        self.assertFalse(st.is_completed)

    def test_subtask_completed_with_result(self):
        st = SubTask(agent_name="test", description="desc", result="done")
        self.assertTrue(st.is_completed)

    def test_supervisor_creation(self):
        reg = AgentRegistry()
        bus = ContextBus()
        sup = Supervisor(
            config=SupervisorConfig(name="test"),
            agent_registry=reg, context_bus=bus,
        )
        self.assertEqual(sup.config.name, "test")


# ══════════════════════════════════════════════════════════════════
# 15. DelegateTaskTool
# ══════════════════════════════════════════════════════════════════

class TestDelegateTaskTool(unittest.TestCase):
    """Tests for DelegateTaskTool (inter-agent delegation)."""

    def test_delegated_task_dataclass(self):
        task = DelegatedTask(
            task_id="t1", delegator="main", delegatee="helper",
            task_description="do work",
        )
        self.assertFalse(task.is_completed)
        d = task.to_dict()
        self.assertEqual(d["delegator"], "main")

    def test_delegate_mode_enum(self):
        self.assertEqual(DelegateMode.SYNC.value, "sync")
        self.assertEqual(DelegateMode.ASYNC.value, "async")

    def test_missing_agent_name(self):
        tool = DelegateTaskTool(agent_registry=MagicMock())
        result = _run(tool.execute(agent_name="", task="do stuff"))
        self.assertFalse(result.success)

    def test_missing_task(self):
        tool = DelegateTaskTool(agent_registry=MagicMock())
        result = _run(tool.execute(agent_name="helper", task=""))
        self.assertFalse(result.success)


# ══════════════════════════════════════════════════════════════════
# 16. Agent Cache Integration
# ══════════════════════════════════════════════════════════════════

class TestAgentCacheIntegration(unittest.TestCase):
    """Verify that Agent has the cache/hardener/retry attributes."""

    def test_agent_has_cache_attributes(self):
        from cowork_agent.core.agent import Agent
        provider = MagicMock()
        registry = MagicMock()
        registry.tool_names = []
        builder = MagicMock()
        agent = Agent(provider=provider, registry=registry, prompt_builder=builder)
        self.assertIsNone(agent.response_cache)
        self.assertIsNone(agent.stream_hardener)
        self.assertIsNone(agent.retry_executor)

    def test_agent_cache_can_be_set(self):
        from cowork_agent.core.agent import Agent
        provider = MagicMock()
        registry = MagicMock()
        registry.tool_names = []
        builder = MagicMock()
        agent = Agent(provider=provider, registry=registry, prompt_builder=builder)
        cache = ResponseCache()
        agent.response_cache = cache
        self.assertIs(agent.response_cache, cache)

    def test_agent_hardener_can_be_set(self):
        from cowork_agent.core.agent import Agent
        provider = MagicMock()
        registry = MagicMock()
        registry.tool_names = []
        builder = MagicMock()
        agent = Agent(provider=provider, registry=registry, prompt_builder=builder)
        hardener = StreamHardener()
        agent.stream_hardener = hardener
        self.assertIs(agent.stream_hardener, hardener)

    def test_agent_retry_can_be_set(self):
        from cowork_agent.core.agent import Agent
        provider = MagicMock()
        registry = MagicMock()
        registry.tool_names = []
        builder = MagicMock()
        agent = Agent(provider=provider, registry=registry, prompt_builder=builder)
        retry = RetryExecutor()
        agent.retry_executor = retry
        self.assertIs(agent.retry_executor, retry)


if __name__ == "__main__":
    unittest.main()
