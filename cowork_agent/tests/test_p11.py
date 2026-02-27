"""
Sprint 11 Tests — Advanced Memory System

Tests for: ConversationSummarizer, KnowledgeStore, MemoryTool,
importance-weighted pruning, sliding window summary, memory-aware
prompt building, session persistence of new fields.

Run: python -m pytest cowork_agent/tests/test_p11.py -v
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

# ── Ensure project root is on path ──────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cowork_agent.core.models import (
    Message, ToolCall, ToolResult, ToolSchema, AgentResponse,
)
from cowork_agent.core.conversation_summarizer import (
    ConversationSummarizer, SummarizationStrategy,
)
from cowork_agent.core.knowledge_store import (
    KnowledgeStore, KnowledgeEntry, VALID_CATEGORIES,
)
from cowork_agent.tools.memory_tool import MemoryTool
from cowork_agent.core.context_manager import ContextManager
from cowork_agent.core.prompt_builder import PromptBuilder
from cowork_agent.core.session_manager import SessionManager


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
    """Create a Message with sensible defaults."""
    return Message(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_results=tool_results,
        timestamp=timestamp or time.time(),
    )


def _make_tool_call(name="read", tool_id="t1", input_data=None):
    return ToolCall(name=name, tool_id=tool_id, input=input_data or {})


def _make_tool_result(tool_id="t1", success=True, output="ok", error=None):
    return ToolResult(tool_id=tool_id, success=success, output=output, error=error)


# ══════════════════════════════════════════════════════════════════
# Test Suite 1: ConversationSummarizer
# ══════════════════════════════════════════════════════════════════


class TestConversationSummarizer(unittest.TestCase):
    """Tests for rule-based conversation summarization."""

    def setUp(self):
        self.summarizer = ConversationSummarizer()

    def test_empty_messages(self):
        """Summarizing empty list returns empty string."""
        self.assertEqual(self.summarizer.summarize([]), "")

    def test_basic_summary_header(self):
        """Summary starts with message count header."""
        msgs = [_make_msg("user", "hi"), _make_msg("assistant", "hello")]
        summary = self.summarizer.summarize(msgs)
        self.assertIn("[Memory Summary", summary)
        self.assertIn("2 messages", summary)
        self.assertIn("1 user turns", summary)

    def test_extract_tools_used(self):
        """Extracts tool names from tool_calls and results."""
        msgs = [
            _make_msg("assistant", "reading file",
                      tool_calls=[_make_tool_call("read", "t1", {"file_path": "/a.py"})]),
            _make_msg("tool_result", "",
                      tool_results=[_make_tool_result("t1", True, "file content")]),
        ]
        summary = self.summarizer.summarize(msgs)
        self.assertIn("Tools:", summary)
        self.assertIn("read", summary)

    def test_extract_file_paths(self):
        """Extracts file paths from tool call inputs."""
        msgs = [
            _make_msg("assistant", "checking /foo/bar.py",
                      tool_calls=[_make_tool_call("read", "t1", {"file_path": "/foo/bar.py"})]),
        ]
        summary = self.summarizer.summarize(msgs)
        self.assertIn("Files:", summary)
        self.assertIn("/foo/bar.py", summary)

    def test_extract_decisions(self):
        """Extracts decision-related sentences from assistant messages."""
        msgs = [
            _make_msg("assistant", "I decided to use PostgreSQL for the backend."),
        ]
        summary = self.summarizer.summarize(msgs)
        self.assertIn("Decisions:", summary)
        self.assertIn("PostgreSQL", summary)

    def test_extract_errors(self):
        """Extracts error messages from failed tool results."""
        msgs = [
            _make_msg("tool_result", "",
                      tool_results=[_make_tool_result("t1", False, "",
                                                       error="FileNotFoundError: /missing.py")]),
        ]
        summary = self.summarizer.summarize(msgs)
        self.assertIn("Errors:", summary)
        self.assertIn("FileNotFoundError", summary)

    def test_extract_user_requests(self):
        """Extracts brief summaries of user messages."""
        msgs = [
            _make_msg("user", "Please create a new Python file for the authentication module"),
        ]
        summary = self.summarizer.summarize(msgs)
        self.assertIn("Requests:", summary)
        self.assertIn("authentication", summary)

    def test_sliding_summary_update_from_empty(self):
        """update_sliding_summary works from empty state."""
        msgs = [_make_msg("user", "test")]
        result = self.summarizer.update_sliding_summary("", msgs)
        self.assertIn("[Memory Summary", result)

    def test_sliding_summary_merge(self):
        """update_sliding_summary merges existing and new summaries."""
        existing = "[Memory Summary — 3 messages, 1 user turns]\n• Tools: read(1x)"
        new_msgs = [
            _make_msg("assistant", "writing file",
                      tool_calls=[_make_tool_call("write", "t2", {"file_path": "/out.py"})]),
        ]
        result = self.summarizer.update_sliding_summary(existing, new_msgs)
        # Should contain merged content
        self.assertIn("[Memory Summary", result)

    def test_no_duplicate_decisions(self):
        """Same decision sentence not repeated in output."""
        msgs = [
            _make_msg("assistant", "We decided to use React."),
            _make_msg("assistant", "We decided to use React."),
        ]
        summary = self.summarizer.summarize(msgs)
        count = summary.count("decided to use React")
        self.assertEqual(count, 1)

    def test_tool_failure_counts(self):
        """Failed tool results show failure indicator."""
        msgs = [
            _make_msg("assistant", "",
                      tool_calls=[_make_tool_call("bash", "t1")]),
            _make_msg("tool_result", "",
                      tool_results=[_make_tool_result("t1", False, "", "command failed")]),
        ]
        summary = self.summarizer.summarize(msgs)
        self.assertIn("✗", summary)

    def test_file_path_regex_filtering(self):
        """File path regex excludes short and HTTP paths."""
        msgs = [
            _make_msg("assistant", "See /a and http://example.com/path and /valid/real/path.py"),
        ]
        paths = self.summarizer._extract_file_paths(msgs)
        # /a is too short (<=5), http path excluded
        self.assertNotIn("/a", paths)
        self.assertIn("/valid/real/path.py", paths)

    def test_strategy_enum(self):
        """SummarizationStrategy enum has expected values."""
        self.assertEqual(SummarizationStrategy.RULE_BASED.value, "rule_based")
        self.assertEqual(SummarizationStrategy.LLM_BASED.value, "lm_based")

    def test_llm_mode_fallback(self):
        """LLM mode falls back to rule-based when no async context."""
        s = ConversationSummarizer(
            provider=MagicMock(),
            strategy=SummarizationStrategy.LLM_BASED,
        )
        msgs = [_make_msg("user", "hi")]
        result = s.summarize(msgs)
        self.assertIn("[Memory Summary", result)

    def test_long_decision_truncated(self):
        """Decision sentences longer than 200 chars are excluded."""
        long = "We decided to " + "x" * 250
        msgs = [_make_msg("assistant", long)]
        decisions = self.summarizer._extract_decisions(msgs)
        self.assertEqual(len(decisions), 0)


# ══════════════════════════════════════════════════════════════════
# Test Suite 2: KnowledgeStore
# ══════════════════════════════════════════════════════════════════


class TestKnowledgeStore(unittest.TestCase):
    """Tests for persistent cross-session knowledge store."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = KnowledgeStore(workspace_dir=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_remember_and_recall(self):
        """Basic remember/recall cycle."""
        self.store.remember("facts", "language", "Python")
        self.assertEqual(self.store.recall("facts", "language"), "Python")

    def test_recall_missing_key(self):
        """Recall non-existent key returns default."""
        self.assertEqual(self.store.recall("facts", "nope"), "")
        self.assertEqual(self.store.recall("facts", "nope", "fallback"), "fallback")

    def test_recall_wrong_category(self):
        """Recall with wrong category returns default."""
        self.store.remember("facts", "color", "blue")
        self.assertEqual(self.store.recall("preferences", "color"), "")

    def test_remember_updates_existing(self):
        """Remember with existing key updates the value."""
        self.store.remember("facts", "db", "MySQL")
        self.store.remember("facts", "db", "PostgreSQL")
        self.assertEqual(self.store.recall("facts", "db"), "PostgreSQL")

    def test_forget(self):
        """Forget removes entry and returns True."""
        self.store.remember("facts", "tmp", "value")
        self.assertTrue(self.store.forget("tmp"))
        self.assertEqual(self.store.recall("facts", "tmp"), "")

    def test_forget_missing(self):
        """Forget non-existent key returns False."""
        self.assertFalse(self.store.forget("nonexistent"))

    def test_search_by_key(self):
        """Search finds entries matching key."""
        self.store.remember("facts", "python_version", "3.11")
        results = self.store.search("python")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].key, "python_version")

    def test_search_by_value(self):
        """Search finds entries matching value."""
        self.store.remember("facts", "db", "PostgreSQL is great")
        results = self.store.search("PostgreSQL")
        self.assertEqual(len(results), 1)

    def test_search_empty_query(self):
        """Empty search returns empty list."""
        self.assertEqual(self.store.search(""), [])

    def test_recall_all_category(self):
        """recall_all returns entries for a specific category."""
        self.store.remember("facts", "a", "1")
        self.store.remember("facts", "b", "2")
        self.store.remember("preferences", "c", "3")
        facts = self.store.recall_all("facts")
        self.assertEqual(len(facts), 2)

    def test_persistence(self):
        """Data persists across KnowledgeStore instances."""
        self.store.remember("facts", "key1", "val1")
        # Create new store pointing to same directory
        store2 = KnowledgeStore(workspace_dir=self.tmpdir)
        self.assertEqual(store2.recall("facts", "key1"), "val1")

    def test_invalid_category(self):
        """Invalid category raises ValueError."""
        with self.assertRaises(ValueError):
            self.store.remember("invalid_cat", "key", "val")

    def test_empty_key_raises(self):
        """Empty key raises ValueError."""
        with self.assertRaises(ValueError):
            self.store.remember("facts", "", "val")

    def test_prune_keeps_recent(self):
        """Prune removes oldest entries, keeps most recent."""
        for i in range(10):
            self.store.remember("facts", f"key_{i}", f"val_{i}")
            time.sleep(0.01)  # Ensure distinct timestamps
        removed = self.store.prune(max_total=5)
        self.assertEqual(removed, 5)
        self.assertEqual(self.store.size, 5)
        # Most recent should survive
        self.assertEqual(self.store.recall("facts", "key_9"), "val_9")

    def test_stats(self):
        """Stats returns correct counts."""
        self.store.remember("facts", "a", "1")
        self.store.remember("preferences", "b", "2")
        stats = self.store.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["categories"]["facts"], 1)
        self.assertEqual(stats["categories"]["preferences"], 1)

    def test_export_import(self):
        """Export and import round-trips data."""
        self.store.remember("facts", "exp_key", "exp_val")
        data = self.store.export_data()
        # New store
        tmpdir2 = tempfile.mkdtemp()
        store2 = KnowledgeStore(workspace_dir=tmpdir2)
        imported = store2.import_data(data)
        self.assertEqual(imported, 1)
        self.assertEqual(store2.recall("facts", "exp_key"), "exp_val")
        import shutil
        shutil.rmtree(tmpdir2, ignore_errors=True)

    def test_no_workspace_dir(self):
        """Store works without workspace (no persistence)."""
        store = KnowledgeStore(workspace_dir="")
        store.remember("facts", "k", "v")
        self.assertEqual(store.recall("facts", "k"), "v")

    def test_access_count_increments(self):
        """Recall increments access_count."""
        self.store.remember("facts", "counted", "val")
        self.store.recall("facts", "counted")
        self.store.recall("facts", "counted")
        entry = self.store._entries["counted"]
        self.assertEqual(entry.access_count, 2)

    def test_search_limit(self):
        """Search respects limit parameter."""
        for i in range(20):
            self.store.remember("facts", f"item_{i}", f"searchable_{i}")
        results = self.store.search("searchable", limit=5)
        self.assertEqual(len(results), 5)


# ══════════════════════════════════════════════════════════════════
# Test Suite 3: MemoryTool
# ══════════════════════════════════════════════════════════════════


class TestMemoryTool(unittest.TestCase):
    """Tests for the agent memory tool."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = KnowledgeStore(workspace_dir=self.tmpdir)
        self.tool = MemoryTool(knowledge_store=self.store)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_tool_name(self):
        """Tool has correct name."""
        self.assertEqual(self.tool.name, "memory_store")

    def test_remember_action(self):
        """remember action stores data and returns success."""
        result = _run(self.tool.execute(
            action="remember", category="facts", key="lang", value="Python",
            tool_id="t1",
        ))
        self.assertTrue(result.success)
        self.assertIn("Remembered", result.output)

    def test_recall_action(self):
        """recall action retrieves stored data."""
        self.store.remember("facts", "db", "PostgreSQL")
        result = _run(self.tool.execute(
            action="recall", category="facts", key="db", tool_id="t1",
        ))
        self.assertTrue(result.success)
        self.assertIn("PostgreSQL", result.output)

    def test_recall_missing(self):
        """recall for missing key returns not-found message."""
        result = _run(self.tool.execute(
            action="recall", category="facts", key="missing", tool_id="t1",
        ))
        self.assertTrue(result.success)
        self.assertIn("No entry found", result.output)

    def test_search_action(self):
        """search action finds matching entries."""
        self.store.remember("facts", "python_version", "3.12")
        result = _run(self.tool.execute(
            action="search", category="facts", key="python", tool_id="t1",
        ))
        self.assertTrue(result.success)
        self.assertIn("1 result", result.output)

    def test_forget_action(self):
        """forget action removes an entry."""
        self.store.remember("facts", "tmp", "val")
        result = _run(self.tool.execute(
            action="forget", category="facts", key="tmp", tool_id="t1",
        ))
        self.assertTrue(result.success)
        self.assertIn("Forgot", result.output)

    def test_missing_action(self):
        """Missing action returns error."""
        result = _run(self.tool.execute(
            action="", category="facts", key="k", tool_id="t1",
        ))
        self.assertFalse(result.success)

    def test_remember_missing_value(self):
        """remember without value returns error."""
        result = _run(self.tool.execute(
            action="remember", category="facts", key="k", value="",
            tool_id="t1",
        ))
        self.assertFalse(result.success)

    def test_input_schema(self):
        """Tool has proper input schema."""
        schema = self.tool.input_schema
        self.assertIn("action", schema["properties"])
        self.assertIn("category", schema["properties"])
        self.assertIn("key", schema["properties"])


# ══════════════════════════════════════════════════════════════════
# Test Suite 4: Importance-Weighted Pruning
# ══════════════════════════════════════════════════════════════════


class TestImportancePruning(unittest.TestCase):
    """Tests for importance scoring and weighted pruning in ContextManager."""

    def setUp(self):
        self.summarizer = ConversationSummarizer()
        self.ctx_mgr = ContextManager(
            max_context_tokens=500,
            summarizer=self.summarizer,
        )

    def test_recency_score_newest_highest(self):
        """Newest messages get highest recency score."""
        newest = self.ctx_mgr._recency_score(9, 10)
        oldest = self.ctx_mgr._recency_score(0, 10)
        self.assertGreater(newest, oldest)

    def test_recency_score_single_message(self):
        """Single message gets score 1.0."""
        self.assertEqual(self.ctx_mgr._recency_score(0, 1), 1.0)

    def test_role_weight_order(self):
        """User > assistant > tool_result."""
        user = self.ctx_mgr._role_weight("user")
        assistant = self.ctx_mgr._role_weight("assistant")
        tool = self.ctx_mgr._role_weight("tool_result")
        self.assertGreater(user, assistant)
        self.assertGreater(assistant, tool)

    def test_content_value_decisions(self):
        """Decision phrases boost content value."""
        msg = _make_msg("assistant", "We decided to use SQLite for storage.")
        score = self.ctx_mgr._content_value(msg)
        self.assertGreater(score, 0)

    def test_content_value_errors(self):
        """Error keywords boost content value."""
        msg = _make_msg("assistant", "Got an error: FileNotFoundError")
        score = self.ctx_mgr._content_value(msg)
        self.assertGreater(score, 0)

    def test_content_value_question(self):
        """User questions get content value boost."""
        msg = _make_msg("user", "How should we handle authentication?")
        score = self.ctx_mgr._content_value(msg)
        self.assertGreater(score, 0)

    def test_content_value_short_message(self):
        """Very short messages get penalty."""
        msg = _make_msg("user", "ok")
        score = self.ctx_mgr._content_value(msg)
        self.assertLessEqual(score, 0)

    def test_tool_result_value_failure(self):
        """Failed tool results valued higher than successes."""
        fail_msg = _make_msg("tool_result", "",
                             tool_results=[_make_tool_result("t1", False, "", "err")])
        ok_msg = _make_msg("tool_result", "",
                           tool_results=[_make_tool_result("t1", True, "ok")])
        self.assertGreater(
            self.ctx_mgr._tool_result_value(fail_msg),
            self.ctx_mgr._tool_result_value(ok_msg),
        )

    def test_length_penalty_short_messages(self):
        """Short messages get no length penalty."""
        msg = _make_msg("user", "short text")
        self.assertEqual(self.ctx_mgr._length_penalty(msg), 0.0)

    def test_length_penalty_long_messages(self):
        """Very long messages get positive penalty."""
        msg = _make_msg("user", "x" * 10000)
        penalty = self.ctx_mgr._length_penalty(msg)
        self.assertGreater(penalty, 0)

    def test_score_message_range(self):
        """Score always 0.0-1.0."""
        msg = _make_msg("user", "test")
        score = self.ctx_mgr._score_message(msg, 5, 10)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_importance_pruning_drops_low_scored(self):
        """Pruning drops messages with lowest importance scores."""
        # Build messages that will exceed context
        msgs = []
        # Some routine messages (low value)
        for i in range(10):
            msgs.append(_make_msg("assistant", "x" * 500))
        # One high-value decision message early on
        msgs.insert(2, _make_msg("assistant",
                                  "We decided to use Python for the project. " * 5))
        # User messages at end (protected by MIN_RECENT_MESSAGES)
        msgs.append(_make_msg("user", "What's next?"))
        msgs.append(_make_msg("assistant", "Let me check."))

        pruned = self.ctx_mgr.prune(msgs, "system prompt here")
        self.assertLess(len(pruned), len(msgs))

    def test_pruning_adds_memory_summary(self):
        """Pruning inserts MEMORY SUMMARY notice when summarizer is present."""
        msgs = [_make_msg("assistant", "x" * 2000) for _ in range(15)]
        msgs.append(_make_msg("user", "continue"))
        pruned = self.ctx_mgr.prune(msgs, "system")
        # First message should be the summary notice
        self.assertIn("MEMORY SUMMARY", pruned[0].content)

    def test_pruning_without_summarizer(self):
        """Pruning without summarizer uses basic notice."""
        ctx_mgr = ContextManager(max_context_tokens=500, summarizer=None)
        msgs = [_make_msg("assistant", "x" * 2000) for _ in range(15)]
        msgs.append(_make_msg("user", "continue"))
        pruned = ctx_mgr.prune(msgs, "system")
        self.assertIn("pruned", pruned[0].content.lower())

    def test_no_pruning_when_under_limit(self):
        """No pruning when messages fit in context."""
        msgs = [_make_msg("user", "hi"), _make_msg("assistant", "hello")]
        ctx_mgr = ContextManager(max_context_tokens=100000)
        result = ctx_mgr.prune(msgs, "")
        self.assertEqual(len(result), 2)


# ══════════════════════════════════════════════════════════════════
# Test Suite 5: Sliding Window Summary
# ══════════════════════════════════════════════════════════════════


class TestSlidingWindowSummary(unittest.TestCase):
    """Tests for sliding window summary updates in Agent."""

    def _make_agent(self):
        """Create a minimal Agent for testing memory features."""
        from cowork_agent.core.agent import Agent
        provider = MagicMock()
        registry = MagicMock()
        registry.get_schemas.return_value = []
        config = {"agent": {"workspace_dir": "/tmp"}, "llm": {"model": "test", "provider": "test"}}
        builder = PromptBuilder(config)
        agent = Agent(
            provider=provider,
            registry=registry,
            prompt_builder=builder,
            max_iterations=5,
        )
        return agent

    def test_initial_summary_is_none(self):
        """Agent starts with no sliding summary."""
        agent = self._make_agent()
        self.assertIsNone(agent._sliding_summary)

    def test_summary_update_interval(self):
        """Summary updates after SUMMARY_UPDATE_INTERVAL user turns."""
        agent = self._make_agent()
        agent._SUMMARY_UPDATE_INTERVAL = 3
        # Add 3 user turns
        for i in range(3):
            agent._messages.append(_make_msg("user", f"msg {i}"))
            agent._messages.append(_make_msg("assistant", f"reply {i}"))
        agent._maybe_update_summary()
        self.assertIsNotNone(agent._sliding_summary)
        self.assertIn("[Memory Summary", agent._sliding_summary)

    def test_summary_not_updated_before_interval(self):
        """Summary doesn't update if interval not reached."""
        agent = self._make_agent()
        agent._SUMMARY_UPDATE_INTERVAL = 10
        agent._messages.append(_make_msg("user", "one msg"))
        agent._maybe_update_summary()
        self.assertIsNone(agent._sliding_summary)

    def test_summary_injected_into_context(self):
        """Sliding summary appears in _build_context output."""
        agent = self._make_agent()
        agent._sliding_summary = "Test summary content"
        ctx = agent._build_context()
        self.assertEqual(ctx["memory_summary"], "Test summary content")

    def test_no_summary_in_context_initially(self):
        """No memory_summary in context when summary is None."""
        agent = self._make_agent()
        ctx = agent._build_context()
        self.assertNotIn("memory_summary", ctx)

    def test_clear_history_resets_summary(self):
        """clear_history resets sliding summary."""
        agent = self._make_agent()
        agent._sliding_summary = "some summary"
        agent._summary_turn_count = 5
        agent.clear_history()
        self.assertIsNone(agent._sliding_summary)
        self.assertEqual(agent._summary_turn_count, 0)

    def test_knowledge_entries_in_context(self):
        """Knowledge entries appear in context when store has data."""
        agent = self._make_agent()
        tmpdir = tempfile.mkdtemp()
        agent.knowledge_store = KnowledgeStore(workspace_dir=tmpdir)
        agent.knowledge_store.remember("facts", "project", "test-app")
        ctx = agent._build_context()
        self.assertIn("knowledge_entries", ctx)
        self.assertEqual(len(ctx["knowledge_entries"]), 1)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_no_knowledge_entries_when_empty(self):
        """No knowledge_entries in context when store is empty."""
        agent = self._make_agent()
        tmpdir = tempfile.mkdtemp()
        agent.knowledge_store = KnowledgeStore(workspace_dir=tmpdir)
        ctx = agent._build_context()
        self.assertNotIn("knowledge_entries", ctx)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_incremental_summary_update(self):
        """Second update merges with existing summary."""
        agent = self._make_agent()
        agent._SUMMARY_UPDATE_INTERVAL = 2
        # First batch
        for i in range(2):
            agent._messages.append(_make_msg("user", f"batch1 msg {i}"))
            agent._messages.append(_make_msg("assistant", f"reply {i}"))
        agent._maybe_update_summary()
        first = agent._sliding_summary
        # Second batch
        for i in range(2):
            agent._messages.append(_make_msg("user", f"batch2 msg {i}"))
            agent._messages.append(_make_msg("assistant", f"reply {i}"))
        agent._maybe_update_summary()
        self.assertIsNotNone(agent._sliding_summary)


# ══════════════════════════════════════════════════════════════════
# Test Suite 6: Memory-Aware Prompt Building
# ══════════════════════════════════════════════════════════════════


class TestMemoryPromptBuilding(unittest.TestCase):
    """Tests for _section_memory() in PromptBuilder."""

    def setUp(self):
        self.config = {
            "agent": {"workspace_dir": "/tmp"},
            "llm": {"model": "test", "provider": "test"},
            "user": {"name": "Tester"},
        }
        self.builder = PromptBuilder(self.config)

    def test_empty_context_no_memory(self):
        """No memory section when context has no memory data."""
        result = self.builder._section_memory({})
        self.assertEqual(result, "")

    def test_summary_only(self):
        """Memory section with only summary."""
        ctx = {"memory_summary": "Test summary content here"}
        result = self.builder._section_memory(ctx)
        self.assertIn("<memory>", result)
        self.assertIn("<summary>", result)
        self.assertIn("Test summary content here", result)
        self.assertNotIn("<knowledge>", result)

    def test_knowledge_only(self):
        """Memory section with only knowledge entries."""
        entries = [
            KnowledgeEntry(key="lang", value="Python", category="facts"),
            KnowledgeEntry(key="style", value="PEP8", category="preferences"),
        ]
        ctx = {"knowledge_entries": entries}
        result = self.builder._section_memory(ctx)
        self.assertIn("<memory>", result)
        self.assertIn("<knowledge>", result)
        self.assertIn("lang: Python", result)
        self.assertIn("style: PEP8", result)
        self.assertNotIn("<summary>", result)

    def test_both_summary_and_knowledge(self):
        """Memory section with both summary and knowledge."""
        entries = [
            KnowledgeEntry(key="db", value="PostgreSQL", category="facts"),
        ]
        ctx = {
            "memory_summary": "Summary here",
            "knowledge_entries": entries,
        }
        result = self.builder._section_memory(ctx)
        self.assertIn("<memory>", result)
        self.assertIn("<summary>", result)
        self.assertIn("<knowledge>", result)
        self.assertIn("</memory>", result)

    def test_memory_injected_into_full_prompt(self):
        """Memory section appears in the full build() output."""
        ctx = {"memory_summary": "Unique memory marker 12345"}
        prompt = self.builder.build(tools=[], context=ctx)
        self.assertIn("Unique memory marker 12345", prompt)
        self.assertIn("<memory>", prompt)

    def test_no_memory_in_prompt_when_empty(self):
        """No <memory> tag in prompt when no memory data."""
        prompt = self.builder.build(tools=[], context={})
        self.assertNotIn("<memory>", prompt)

    def test_knowledge_category_shown(self):
        """Knowledge entries show their category."""
        entries = [
            KnowledgeEntry(key="theme", value="dark", category="preferences"),
        ]
        ctx = {"knowledge_entries": entries}
        result = self.builder._section_memory(ctx)
        self.assertIn("[preferences]", result)

    def test_multiple_knowledge_entries(self):
        """Multiple knowledge entries all appear."""
        entries = [
            KnowledgeEntry(key="a", value="1", category="facts"),
            KnowledgeEntry(key="b", value="2", category="facts"),
            KnowledgeEntry(key="c", value="3", category="decisions"),
        ]
        ctx = {"knowledge_entries": entries}
        result = self.builder._section_memory(ctx)
        self.assertIn("a: 1", result)
        self.assertIn("b: 2", result)
        self.assertIn("c: 3", result)


# ══════════════════════════════════════════════════════════════════
# Test Suite 7: Session Persistence of New Fields
# ══════════════════════════════════════════════════════════════════


class TestSessionPersistenceNewFields(unittest.TestCase):
    """Tests for persisting importance_score and memory_id through SessionManager."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sm = SessionManager(workspace_dir=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_importance_score_persists(self):
        """importance_score survives save/load cycle."""
        sid = self.sm.create_session()
        msg = Message(role="user", content="test", importance_score=0.85)
        self.sm.save_message(sid, msg)
        loaded = self.sm.load_messages(sid)
        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(loaded[0].importance_score, 0.85)

    def test_memory_id_persists(self):
        """memory_id survives save/load cycle."""
        sid = self.sm.create_session()
        msg = Message(role="assistant", content="hi", memory_id="mem_abc123")
        self.sm.save_message(sid, msg)
        loaded = self.sm.load_messages(sid)
        self.assertEqual(loaded[0].memory_id, "mem_abc123")

    def test_none_fields_not_in_json(self):
        """None importance_score/memory_id not written to JSON."""
        msg = Message(role="user", content="test")
        data = self.sm._message_to_dict(msg)
        self.assertNotIn("importance_score", data)
        self.assertNotIn("memory_id", data)

    def test_backward_compatibility(self):
        """Old messages without new fields load correctly."""
        sid = self.sm.create_session()
        # Write raw JSONL without new fields
        msg_path = os.path.join(self.tmpdir, ".cowork/sessions", sid, "messages.jsonl")
        with open(msg_path, "w") as f:
            f.write(json.dumps({"role": "user", "content": "old msg", "timestamp": 1.0}) + "\n")
        loaded = self.sm.load_messages(sid)
        self.assertEqual(len(loaded), 1)
        self.assertIsNone(loaded[0].importance_score)
        self.assertIsNone(loaded[0].memory_id)


# ══════════════════════════════════════════════════════════════════
# Test Suite 8: Message Model Extensions
# ══════════════════════════════════════════════════════════════════


class TestMessageModelExtensions(unittest.TestCase):
    """Tests for Sprint 11 Message model additions."""

    def test_default_importance_score_none(self):
        """importance_score defaults to None."""
        msg = Message(role="user", content="hi")
        self.assertIsNone(msg.importance_score)

    def test_default_memory_id_none(self):
        """memory_id defaults to None."""
        msg = Message(role="user", content="hi")
        self.assertIsNone(msg.memory_id)

    def test_set_importance_score(self):
        """Can set importance_score."""
        msg = Message(role="user", content="hi", importance_score=0.7)
        self.assertEqual(msg.importance_score, 0.7)

    def test_set_memory_id(self):
        """Can set memory_id."""
        msg = Message(role="user", content="hi", memory_id="mem_001")
        self.assertEqual(msg.memory_id, "mem_001")

    def test_backward_compatible_creation(self):
        """Message creation without new fields still works."""
        msg = Message(role="assistant", content="response")
        self.assertEqual(msg.role, "assistant")
        self.assertEqual(msg.content, "response")
        self.assertIsNone(msg.tool_calls)


# ══════════════════════════════════════════════════════════════════
# Test Suite 9: Integration Tests
# ══════════════════════════════════════════════════════════════════


class TestSprint11Integration(unittest.TestCase):
    """End-to-end integration tests for the memory system."""

    def test_knowledge_store_with_memory_tool(self):
        """MemoryTool writes to and reads from KnowledgeStore correctly."""
        tmpdir = tempfile.mkdtemp()
        store = KnowledgeStore(workspace_dir=tmpdir)
        tool = MemoryTool(knowledge_store=store)

        # Remember via tool
        result = _run(tool.execute(
            action="remember", category="decisions", key="framework",
            value="React", tool_id="t1",
        ))
        self.assertTrue(result.success)

        # Recall via store directly
        self.assertEqual(store.recall("decisions", "framework"), "React")

        # Search via tool
        result = _run(tool.execute(
            action="search", category="decisions", key="React", tool_id="t2",
        ))
        self.assertTrue(result.success)
        self.assertIn("1 result", result.output)

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_context_manager_with_summarizer(self):
        """ContextManager integrates ConversationSummarizer during pruning."""
        summarizer = ConversationSummarizer()
        ctx_mgr = ContextManager(max_context_tokens=300, summarizer=summarizer)

        # Create messages that exceed context
        msgs = []
        for i in range(20):
            msgs.append(_make_msg("user", f"Question {i}: " + "x" * 200))
            msgs.append(_make_msg("assistant", f"Answer {i}: " + "y" * 200))

        pruned = ctx_mgr.prune(msgs, "system prompt")
        self.assertLess(len(pruned), len(msgs))
        # First message should be memory summary
        self.assertIn("MEMORY SUMMARY", pruned[0].content)

    def test_prompt_builder_full_memory_injection(self):
        """Full prompt includes memory section when context has memory data."""
        config = {
            "agent": {"workspace_dir": "/tmp"},
            "llm": {"model": "test", "provider": "test"},
            "user": {"name": "Test"},
        }
        builder = PromptBuilder(config)
        entries = [
            KnowledgeEntry(key="db", value="PostgreSQL", category="facts"),
        ]
        ctx = {
            "memory_summary": "Tools used: read(3x), write(1x)",
            "knowledge_entries": entries,
        }
        prompt = builder.build(tools=[], context=ctx)
        self.assertIn("<memory>", prompt)
        self.assertIn("Tools used: read(3x)", prompt)
        self.assertIn("PostgreSQL", prompt)

    def test_knowledge_entry_dataclass(self):
        """KnowledgeEntry serialization round-trip."""
        entry = KnowledgeEntry(
            key="test_key", value="test_val", category="facts",
            created_at=1000.0, updated_at=2000.0, access_count=5,
        )
        d = entry.to_dict()
        restored = KnowledgeEntry.from_dict(d)
        self.assertEqual(restored.key, "test_key")
        self.assertEqual(restored.value, "test_val")
        self.assertEqual(restored.access_count, 5)

    def test_valid_categories(self):
        """VALID_CATEGORIES has expected values."""
        self.assertEqual(VALID_CATEGORIES, {"facts", "preferences", "decisions"})

    def test_summarizer_with_mixed_messages(self):
        """Summarizer handles mix of all message types."""
        msgs = [
            _make_msg("user", "Create a Python web app"),
            _make_msg("assistant", "I'll create the app. Let me start with the structure.",
                      tool_calls=[_make_tool_call("write", "t1", {"file_path": "/app/main.py"})]),
            _make_msg("tool_result", "",
                      tool_results=[_make_tool_result("t1", True, "File written")]),
            _make_msg("assistant", "I decided to use Flask for simplicity."),
            _make_msg("user", "Can you add error handling?"),
            _make_msg("assistant", "Sure, adding try/except blocks.",
                      tool_calls=[_make_tool_call("edit", "t2", {"file_path": "/app/main.py"})]),
            _make_msg("tool_result", "",
                      tool_results=[_make_tool_result("t2", False, "",
                                                       error="File locked by another process")]),
        ]
        summarizer = ConversationSummarizer()
        summary = summarizer.summarize(msgs)
        self.assertIn("[Memory Summary", summary)
        self.assertIn("7 messages", summary)
        self.assertIn("Tools:", summary)
        # Should have decisions and errors
        self.assertIn("Decisions:", summary)
        self.assertIn("Errors:", summary)

    def test_agent_accepts_memory_kwargs(self):
        """Agent constructor accepts summarizer and knowledge_store kwargs."""
        from cowork_agent.core.agent import Agent
        provider = MagicMock()
        registry = MagicMock()
        registry.get_schemas.return_value = []
        config = {"agent": {"workspace_dir": "/tmp"}, "llm": {"model": "t", "provider": "t"}}

        tmpdir = tempfile.mkdtemp()
        ks = KnowledgeStore(workspace_dir=tmpdir)
        cs = ConversationSummarizer()

        agent = Agent(
            provider=provider,
            registry=registry,
            prompt_builder=PromptBuilder(config),
            summarizer=cs,
            knowledge_store=ks,
        )
        self.assertIs(agent.summarizer, cs)
        self.assertIs(agent.knowledge_store, ks)
        # Context manager should have the summarizer
        self.assertIs(agent.context_mgr._summarizer, cs)

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_search_scoring_key_over_value(self):
        """Search ranks key matches higher than value matches."""
        tmpdir = tempfile.mkdtemp()
        store = KnowledgeStore(workspace_dir=tmpdir)
        store.remember("facts", "python_info", "A programming language")
        store.remember("facts", "java_info", "Uses python libraries sometimes")
        results = store.search("python")
        # Key match should rank first
        self.assertEqual(results[0].key, "python_info")
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
