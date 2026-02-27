"""
Sprint 7 — Persistence & State Tests

~70 tests covering:
  - TestAgentSession           (12 tests)
  - TestConversationStore      (12 tests)
  - TestTokenUsageStore        (12 tests)
  - TestHybridCache            (14 tests)
  - TestStateSnapshot          (12 tests)
  - TestSprint7Integration     (6 tests)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

# ── Path setup ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cowork_agent.core.models import AgentResponse, Message, ToolCall
from cowork_agent.core.session_manager import SessionManager, SessionMetadata
from cowork_agent.core.agent_session import AgentSessionManager, SessionConfig
from cowork_agent.core.conversation_store import (
    ConversationStats,
    ConversationStore,
    SearchResult,
)
from cowork_agent.core.token_usage_store import (
    TokenBudgetAlert,
    TokenUsageSnapshot,
    TokenUsageStore,
)
from cowork_agent.core.hybrid_cache import (
    CacheStats,
    HybridResponseCache,
)
from cowork_agent.core.state_snapshot import (
    StateSnapshot,
    StateSnapshotManager,
)


# ── Async test helper ───────────────────────────────────────────────

def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Mock helpers ────────────────────────────────────────────────────

def _make_message(role="user", content="hello", ts=None):
    return Message(role=role, content=content, timestamp=ts or time.time())


def _make_tracker(input_tokens=100, output_tokens=50, cost=0.01, call_count=1):
    """Create a mock TokenTracker."""
    tracker = MagicMock()
    tracker.total_input_tokens = input_tokens
    tracker.total_output_tokens = output_tokens
    tracker.total_tokens = input_tokens + output_tokens
    tracker.total_cache_read_tokens = 0
    tracker.total_cache_write_tokens = 0
    tracker.call_count = call_count
    tracker.estimated_cost_usd = cost
    return tracker


def _make_response(text="Hello!", stop_reason="end_turn"):
    return AgentResponse(text=text, tool_calls=[], stop_reason=stop_reason)


# ═══════════════════════════════════════════════════════════════════
#  1. AGENT SESSION — 12 tests
# ═══════════════════════════════════════════════════════════════════

class TestAgentSession(unittest.TestCase):
    """Tests for AgentSessionManager and SessionConfig."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._sm = SessionManager(workspace_dir=self._tmp)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ── SessionConfig ──────────────────────────────────────────

    def test_session_config_defaults(self):
        cfg = SessionConfig()
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.auto_create)
        self.assertIsNone(cfg.session_id)

    def test_session_config_roundtrip(self):
        cfg = SessionConfig(enabled=True, provider="ollama", model="llama3")
        d = cfg.to_dict()
        cfg2 = SessionConfig.from_dict(d)
        self.assertEqual(cfg.provider, cfg2.provider)
        self.assertEqual(cfg.model, cfg2.model)

    # ── Initialize: auto-create ────────────────────────────────

    def test_initialize_auto_creates_session(self):
        asm = AgentSessionManager(self._sm, SessionConfig(provider="test"))
        sid = asm.initialize()
        self.assertIsNotNone(sid)
        self.assertEqual(asm.session_id, sid)

    def test_initialize_disabled_returns_none(self):
        asm = AgentSessionManager(self._sm, SessionConfig(enabled=False))
        sid = asm.initialize()
        self.assertIsNone(sid)
        self.assertIsNone(asm.session_id)

    def test_initialize_auto_create_false_no_explicit_id(self):
        asm = AgentSessionManager(self._sm, SessionConfig(auto_create=False))
        sid = asm.initialize()
        self.assertIsNone(sid)

    # ── Initialize: resume ─────────────────────────────────────

    def test_initialize_resume_existing(self):
        # Create a session first
        orig_id = self._sm.create_session(title="orig")
        asm = AgentSessionManager(self._sm, SessionConfig())
        sid = asm.initialize(resume_from=orig_id)
        self.assertEqual(sid, orig_id)

    def test_initialize_resume_nonexistent_creates_new(self):
        asm = AgentSessionManager(self._sm, SessionConfig())
        sid = asm.initialize(resume_from="nonexistent_id")
        self.assertIsNotNone(sid)
        self.assertNotEqual(sid, "nonexistent_id")

    # ── save_message ───────────────────────────────────────────

    def test_save_message_persists(self):
        asm = AgentSessionManager(self._sm, SessionConfig())
        asm.initialize()
        msg = _make_message(role="user", content="test msg")
        asm.save_message(msg)
        loaded = asm.get_session_messages()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].content, "test msg")

    def test_save_message_when_disabled_noop(self):
        asm = AgentSessionManager(self._sm, SessionConfig(enabled=False))
        asm.initialize()
        asm.save_message(_make_message())
        msgs = asm.get_session_messages()
        self.assertEqual(len(msgs), 0)

    # ── Metadata helpers ───────────────────────────────────────

    def test_update_title(self):
        asm = AgentSessionManager(self._sm, SessionConfig())
        asm.initialize()
        asm.update_title("New Title")
        meta = asm.get_metadata()
        self.assertEqual(meta.title, "New Title")

    def test_list_recent(self):
        # Create 3 sessions
        self._sm.create_session(title="a")
        self._sm.create_session(title="b")
        self._sm.create_session(title="c")
        asm = AgentSessionManager(self._sm, SessionConfig())
        recent = asm.list_recent(limit=2)
        self.assertEqual(len(recent), 2)

    def test_get_metadata_no_session(self):
        asm = AgentSessionManager(self._sm, SessionConfig(enabled=False))
        meta = asm.get_metadata()
        self.assertIsNone(meta)


# ═══════════════════════════════════════════════════════════════════
#  2. CONVERSATION STORE — 12 tests
# ═══════════════════════════════════════════════════════════════════

class TestConversationStore(unittest.TestCase):
    """Tests for ConversationStore: search, export, stats, prune."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._sm = SessionManager(workspace_dir=self._tmp)
        self._store = ConversationStore(self._sm)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _create_session_with_messages(self, title="test", messages=None):
        sid = self._sm.create_session(title=title)
        for msg in (messages or []):
            self._sm.save_message(sid, msg)
        return sid

    # ── Search ─────────────────────────────────────────────────

    def test_search_by_keyword(self):
        sid = self._create_session_with_messages("chat1", [
            _make_message(content="hello world"),
            _make_message(content="goodbye world"),
        ])
        results = self._store.search(keyword="hello")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].match_count, 1)

    def test_search_case_insensitive(self):
        self._create_session_with_messages("ci", [
            _make_message(content="Hello World"),
        ])
        results = self._store.search(keyword="hello")
        self.assertEqual(len(results), 1)

    def test_search_no_match(self):
        self._create_session_with_messages("nope", [
            _make_message(content="foo bar"),
        ])
        results = self._store.search(keyword="zzzzz")
        self.assertEqual(len(results), 0)

    def test_search_date_filter(self):
        sid = self._create_session_with_messages("dated", [
            _make_message(content="recent message"),
        ])
        future = time.time() + 100
        results = self._store.search(start_date=future)
        self.assertEqual(len(results), 0)

    def test_search_no_keyword_returns_all(self):
        self._create_session_with_messages("a")
        self._create_session_with_messages("b")
        results = self._store.search()
        self.assertEqual(len(results), 2)

    # ── Export ─────────────────────────────────────────────────

    def test_export_markdown(self):
        sid = self._create_session_with_messages("export_test", [
            _make_message(role="user", content="hi"),
            _make_message(role="assistant", content="hello"),
        ])
        md = self._store.export_markdown(sid)
        self.assertIn("# Session: export_test", md)
        self.assertIn("## User", md)
        self.assertIn("## Assistant", md)
        self.assertIn("hi", md)

    def test_export_nonexistent_session(self):
        md = self._store.export_markdown("nonexistent")
        self.assertEqual(md, "")

    # ── Stats ──────────────────────────────────────────────────

    def test_get_stats(self):
        sid = self._create_session_with_messages("stats_test", [
            _make_message(content="hello world"),
            _make_message(content="foo bar baz"),
        ])
        stats = self._store.get_stats(sid)
        self.assertIsNotNone(stats)
        self.assertEqual(stats.message_count, 2)
        self.assertGreater(stats.token_count, 0)

    def test_get_stats_nonexistent(self):
        stats = self._store.get_stats("nonexistent")
        self.assertIsNone(stats)

    def test_stats_to_dict(self):
        sid = self._create_session_with_messages("dict_test", [
            _make_message(content="test"),
        ])
        stats = self._store.get_stats(sid)
        d = stats.to_dict()
        self.assertIn("message_count", d)
        self.assertIn("age_days", d)

    # ── Prune ──────────────────────────────────────────────────

    def test_prune_keeps_min_sessions(self):
        for i in range(5):
            self._sm.create_session(title=f"s{i}")
        deleted = self._store.prune_old_sessions(max_age_days=0, min_keep=5)
        self.assertEqual(len(deleted), 0)

    def test_prune_deletes_old(self):
        # Create sessions and patch their timestamps to be old
        for i in range(3):
            sid = self._sm.create_session(title=f"old{i}")
            meta = self._sm.get_metadata(sid)
            meta.updated_at = time.time() - (200 * 86400)  # 200 days old
            self._sm._save_metadata(sid, meta)

        deleted = self._store.prune_old_sessions(max_age_days=90, min_keep=0)
        self.assertEqual(len(deleted), 3)


# ═══════════════════════════════════════════════════════════════════
#  3. TOKEN USAGE STORE — 12 tests
# ═══════════════════════════════════════════════════════════════════

class TestTokenUsageStore(unittest.TestCase):
    """Tests for TokenUsageStore: recording, summaries, budget alerts."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._store = TokenUsageStore(workspace_dir=self._tmp)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ── Recording ──────────────────────────────────────────────

    def test_record_snapshot(self):
        tracker = _make_tracker(input_tokens=100, output_tokens=50)
        alerts = self._store.record_snapshot("s1", tracker)
        self.assertEqual(self._store.snapshot_count, 1)
        self.assertEqual(len(alerts), 0)

    def test_cumulative_totals(self):
        self._store.record_snapshot("s1", _make_tracker(100, 50))
        self._store.record_snapshot("s2", _make_tracker(200, 100))
        self.assertEqual(self._store.cumulative_input, 300)
        self.assertEqual(self._store.cumulative_output, 150)
        self.assertEqual(self._store.cumulative_total, 450)

    def test_cumulative_cost(self):
        self._store.record_snapshot("s1", _make_tracker(cost=0.01))
        self._store.record_snapshot("s2", _make_tracker(cost=0.02))
        self.assertAlmostEqual(self._store.cumulative_cost, 0.03, places=4)

    # ── Persistence ────────────────────────────────────────────

    def test_persistence_across_instances(self):
        self._store.record_snapshot("s1", _make_tracker(100, 50))
        # Create a new store instance pointing at same workspace
        store2 = TokenUsageStore(workspace_dir=self._tmp)
        self.assertEqual(store2.snapshot_count, 1)
        self.assertEqual(store2.cumulative_input, 100)

    # ── Summaries ──────────────────────────────────────────────

    def test_daily_summary(self):
        self._store.record_snapshot("s1", _make_tracker(100, 50, cost=0.01, call_count=5))
        summary = self._store.daily_summary()
        self.assertEqual(summary["input_tokens"], 100)
        self.assertEqual(summary["output_tokens"], 50)
        self.assertEqual(summary["calls"], 5)

    def test_weekly_summary(self):
        self._store.record_snapshot("s1", _make_tracker(100, 50))
        summary = self._store.weekly_summary()
        self.assertGreater(summary["token_count"], 0)

    def test_monthly_summary(self):
        from datetime import datetime
        self._store.record_snapshot("s1", _make_tracker(100, 50))
        now = datetime.now()
        summary = self._store.monthly_summary(now.year, now.month)
        self.assertGreater(summary["token_count"], 0)

    def test_daily_summary_no_data(self):
        from datetime import datetime, timedelta
        yesterday = datetime.now() - timedelta(days=1)
        # Record now but query yesterday — should be empty
        self._store.record_snapshot("s1", _make_tracker(100, 50))
        summary = self._store.daily_summary(date=yesterday)
        self.assertEqual(summary["token_count"], 0)

    # ── Budget alerts ──────────────────────────────────────────

    def test_budget_alert_approaching_tokens(self):
        store = TokenUsageStore(
            workspace_dir=self._tmp,
            max_session_tokens=100,
            warning_threshold_percent=80.0,
        )
        tracker = _make_tracker(input_tokens=45, output_tokens=40)  # 85 / 100 = 85%
        alerts = store.record_snapshot("s1", tracker)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].alert_type, "approaching")
        self.assertEqual(alerts[0].metric, "tokens")

    def test_budget_alert_exceeded_tokens(self):
        store = TokenUsageStore(
            workspace_dir=self._tmp,
            max_session_tokens=100,
        )
        tracker = _make_tracker(input_tokens=80, output_tokens=30)  # 110 / 100 = 110%
        alerts = store.record_snapshot("s1", tracker)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].alert_type, "exceeded")

    def test_budget_alert_cost_exceeded(self):
        store = TokenUsageStore(
            workspace_dir=self._tmp,
            max_cost_usd=0.05,
        )
        tracker = _make_tracker(cost=0.06)
        alerts = store.record_snapshot("s1", tracker)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].metric, "cost")

    def test_no_budget_no_alerts(self):
        tracker = _make_tracker(input_tokens=99999, cost=999)
        alerts = self._store.record_snapshot("s1", tracker)
        self.assertEqual(len(alerts), 0)


# ═══════════════════════════════════════════════════════════════════
#  4. HYBRID CACHE — 14 tests
# ═══════════════════════════════════════════════════════════════════

class TestHybridCache(unittest.TestCase):
    """Tests for HybridResponseCache: memory LRU, disk spill, TTL, stats."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._cache = HybridResponseCache(
            workspace_dir=self._tmp,
            max_memory_entries=3,
            max_disk_entries=10,
            ttl=60.0,
        )

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ── Basic get/put ──────────────────────────────────────────

    def test_put_and_get(self):
        resp = _make_response("Hello!")
        self._cache.put("key1", resp)
        got = self._cache.get("key1")
        self.assertIsNotNone(got)
        self.assertEqual(got.text, "Hello!")

    def test_get_miss(self):
        got = self._cache.get("nonexistent")
        self.assertIsNone(got)

    def test_put_non_cacheable_tool_calls(self):
        resp = AgentResponse(text="x", tool_calls=[ToolCall("t", "id", {})], stop_reason="tool_use")
        ok = self._cache.put("key", resp)
        self.assertFalse(ok)

    def test_put_non_cacheable_empty_text(self):
        resp = AgentResponse(text="", stop_reason="end_turn")
        ok = self._cache.put("key", resp)
        self.assertFalse(ok)

    # ── Memory LRU eviction → disk spill ───────────────────────

    def test_memory_eviction_spills_to_disk(self):
        for i in range(5):
            self._cache.put(f"k{i}", _make_response(f"resp{i}"))
        # Max memory is 3, so 2 should have been evicted to disk
        stats = self._cache.stats()
        self.assertLessEqual(stats.memory_entries, 3)
        # The evicted entries should be on disk
        got0 = self._cache.get("k0")
        self.assertIsNotNone(got0)

    def test_disk_promotion_on_access(self):
        # Fill memory, causing eviction
        for i in range(4):
            self._cache.put(f"k{i}", _make_response(f"resp{i}"))
        # k0 was evicted from memory, accessing it should promote from disk
        got = self._cache.get("k0")
        self.assertIsNotNone(got)
        # After promotion, k0 is back in memory
        stats = self._cache.stats()
        self.assertGreater(stats.disk_hits, 0)

    # ── TTL expiration ─────────────────────────────────────────

    def test_memory_ttl_expiration(self):
        cache = HybridResponseCache(
            workspace_dir=self._tmp,
            max_memory_entries=10,
            ttl=0.1,  # 100ms TTL
        )
        cache.put("fast", _make_response("expires"))
        time.sleep(0.2)
        got = cache.get("fast")
        self.assertIsNone(got)

    def test_disk_ttl_expiration(self):
        cache = HybridResponseCache(
            workspace_dir=self._tmp,
            max_memory_entries=1,
            ttl=0.1,
        )
        cache.put("a", _make_response("first"))
        cache.put("b", _make_response("second"))  # evicts 'a' to disk
        time.sleep(0.2)
        got = cache.get("a")
        self.assertIsNone(got)

    # ── Invalidate ─────────────────────────────────────────────

    def test_invalidate(self):
        self._cache.put("key", _make_response("x"))
        found = self._cache.invalidate("key")
        self.assertTrue(found)
        self.assertIsNone(self._cache.get("key"))

    def test_invalidate_nonexistent(self):
        found = self._cache.invalidate("nope")
        self.assertFalse(found)

    # ── Clear ──────────────────────────────────────────────────

    def test_clear(self):
        for i in range(5):
            self._cache.put(f"k{i}", _make_response(f"r{i}"))
        self._cache.clear()
        stats = self._cache.stats()
        self.assertEqual(stats.memory_entries, 0)

    # ── Stats ──────────────────────────────────────────────────

    def test_stats_tracking(self):
        self._cache.put("k1", _make_response("a"))
        self._cache.get("k1")  # hit
        self._cache.get("miss")  # miss
        stats = self._cache.stats()
        self.assertEqual(stats.memory_hits, 1)
        self.assertGreater(stats.total_misses, 0)

    # ── Disabled cache ─────────────────────────────────────────

    def test_disabled_cache(self):
        cache = HybridResponseCache(enabled=False)
        ok = cache.put("k", _make_response("x"))
        self.assertFalse(ok)
        self.assertIsNone(cache.get("k"))

    def test_no_workspace_memory_only(self):
        cache = HybridResponseCache(workspace_dir="", max_memory_entries=5, ttl=60)
        cache.put("k", _make_response("mem-only"))
        got = cache.get("k")
        self.assertIsNotNone(got)
        self.assertEqual(got.text, "mem-only")


# ═══════════════════════════════════════════════════════════════════
#  5. STATE SNAPSHOT — 12 tests
# ═══════════════════════════════════════════════════════════════════

class TestStateSnapshot(unittest.TestCase):
    """Tests for StateSnapshotManager: create, restore, list, delete, auto."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._mgr = StateSnapshotManager(workspace_dir=self._tmp, max_snapshots=5)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _sample_messages(self, n=3):
        return [_make_message(content=f"msg{i}") for i in range(n)]

    # ── Create ─────────────────────────────────────────────────

    def test_create_snapshot(self):
        msgs = self._sample_messages()
        snap_id = self._mgr.create_snapshot(messages=msgs, label="test")
        self.assertTrue(snap_id.startswith("snap_"))
        self.assertEqual(self._mgr.snapshot_count, 1)

    def test_create_snapshot_with_todos(self):
        todos = [{"content": "task1", "status": "pending"}]
        snap_id = self._mgr.create_snapshot(
            messages=self._sample_messages(),
            label="with todos",
            todos=todos,
        )
        self.assertIsNotNone(snap_id)

    # ── Restore ────────────────────────────────────────────────

    def test_restore_snapshot(self):
        msgs = self._sample_messages(2)
        snap_id = self._mgr.create_snapshot(messages=msgs, label="restore_me")
        snapshot = self._mgr.restore_snapshot(snap_id)
        self.assertIsNotNone(snapshot)
        self.assertEqual(len(snapshot.messages), 2)
        self.assertEqual(snapshot.label, "restore_me")

    def test_restore_nonexistent(self):
        snapshot = self._mgr.restore_snapshot("snap_doesnotexist")
        self.assertIsNone(snapshot)

    def test_restore_preserves_content(self):
        msgs = [_make_message(content="important data")]
        snap_id = self._mgr.create_snapshot(messages=msgs, label="data")
        snapshot = self._mgr.restore_snapshot(snap_id)
        self.assertEqual(snapshot.messages[0].content, "important data")

    # ── List ───────────────────────────────────────────────────

    def test_list_snapshots(self):
        for i in range(3):
            self._mgr.create_snapshot(messages=self._sample_messages(), label=f"snap{i}")
        snaps = self._mgr.list_snapshots()
        self.assertEqual(len(snaps), 3)
        # Newest first
        self.assertGreaterEqual(snaps[0].timestamp, snaps[1].timestamp)

    def test_list_snapshots_with_limit(self):
        for i in range(4):
            self._mgr.create_snapshot(messages=self._sample_messages(), label=f"s{i}")
        snaps = self._mgr.list_snapshots(limit=2)
        self.assertEqual(len(snaps), 2)

    # ── Delete ─────────────────────────────────────────────────

    def test_delete_snapshot(self):
        snap_id = self._mgr.create_snapshot(messages=self._sample_messages(), label="del")
        result = self._mgr.delete_snapshot(snap_id)
        self.assertTrue(result)
        self.assertEqual(self._mgr.snapshot_count, 0)

    def test_delete_nonexistent(self):
        result = self._mgr.delete_snapshot("snap_nope")
        self.assertFalse(result)

    # ── Auto-snapshot before risky tools ───────────────────────

    def test_auto_snapshot_risky_tool(self):
        msgs = self._sample_messages()
        snap_id = self._mgr.auto_snapshot_before_risky("bash", msgs)
        self.assertTrue(snap_id.startswith("snap_"))
        self.assertEqual(self._mgr.snapshot_count, 1)

    def test_auto_snapshot_safe_tool(self):
        msgs = self._sample_messages()
        snap_id = self._mgr.auto_snapshot_before_risky("read", msgs)
        self.assertEqual(snap_id, "")
        self.assertEqual(self._mgr.snapshot_count, 0)

    # ── Max snapshots eviction ─────────────────────────────────

    def test_max_snapshots_eviction(self):
        for i in range(8):
            self._mgr.create_snapshot(messages=self._sample_messages(), label=f"s{i}")
        # max_snapshots=5, so only 5 should remain
        self.assertEqual(self._mgr.snapshot_count, 5)
        # The oldest 3 should have been evicted
        snaps = self._mgr.list_snapshots()
        labels = [s.label for s in snaps]
        self.assertNotIn("s0", labels)
        self.assertNotIn("s1", labels)
        self.assertNotIn("s2", labels)


# ═══════════════════════════════════════════════════════════════════
#  6. SPRINT 7 INTEGRATION — 6 tests
# ═══════════════════════════════════════════════════════════════════

class TestSprint7Integration(unittest.TestCase):
    """Integration tests combining Sprint 7 features."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_session_plus_conversation_store(self):
        """AgentSessionManager + ConversationStore work together."""
        sm = SessionManager(workspace_dir=self._tmp)
        asm = AgentSessionManager(sm, SessionConfig(provider="test"))
        asm.initialize()

        # Save messages through agent session
        asm.save_message(_make_message(content="hello search me"))
        asm.save_message(_make_message(role="assistant", content="found it"))

        # Search through conversation store
        store = ConversationStore(sm)
        results = store.search(keyword="search me")
        self.assertEqual(len(results), 1)

    def test_token_store_persistence_roundtrip(self):
        """TokenUsageStore persists and reloads correctly."""
        store1 = TokenUsageStore(workspace_dir=self._tmp, max_session_tokens=10000)
        store1.record_snapshot("s1", _make_tracker(500, 200, cost=0.05))

        # New instance
        store2 = TokenUsageStore(workspace_dir=self._tmp)
        self.assertEqual(store2.cumulative_input, 500)
        self.assertEqual(store2.cumulative_output, 200)

    def test_cache_disk_roundtrip(self):
        """HybridCache writes to disk and a new instance can read it."""
        cache1 = HybridResponseCache(
            workspace_dir=self._tmp,
            max_memory_entries=5,
            ttl=3600,
        )
        cache1.put("mykey", _make_response("cached answer"))

        # New instance (memory is empty, but disk has data)
        cache2 = HybridResponseCache(
            workspace_dir=self._tmp,
            max_memory_entries=5,
            ttl=3600,
        )
        got = cache2.get("mykey")
        self.assertIsNotNone(got)
        self.assertEqual(got.text, "cached answer")

    def test_snapshot_create_and_restore_full(self):
        """StateSnapshotManager round-trips messages + todos."""
        mgr = StateSnapshotManager(workspace_dir=self._tmp)
        msgs = [_make_message(content="important"), _make_message(content="also important")]
        todos = [{"content": "task1", "status": "completed"}]

        snap_id = mgr.create_snapshot(
            messages=msgs,
            label="full test",
            todos=todos,
            token_usage_summary={"total": 1000},
        )

        snapshot = mgr.restore_snapshot(snap_id)
        self.assertEqual(len(snapshot.messages), 2)
        self.assertEqual(snapshot.todos[0]["content"], "task1")
        self.assertEqual(snapshot.token_usage_summary["total"], 1000)

    def test_session_plus_snapshot(self):
        """Session ID is captured in snapshots."""
        sm = SessionManager(workspace_dir=self._tmp)
        asm = AgentSessionManager(sm, SessionConfig(provider="test"))
        sid = asm.initialize()

        snap_mgr = StateSnapshotManager(workspace_dir=self._tmp)
        msgs = [_make_message(content="before risky op")]
        snap_id = snap_mgr.create_snapshot(
            messages=msgs,
            label="pre-bash",
            session_id=sid,
        )

        snapshot = snap_mgr.restore_snapshot(snap_id)
        self.assertEqual(snapshot.session_id, sid)

    def test_full_lifecycle(self):
        """
        Full lifecycle: create session → save messages → record tokens →
        cache response → snapshot state → verify all.
        """
        sm = SessionManager(workspace_dir=self._tmp)
        asm = AgentSessionManager(sm, SessionConfig(provider="test", model="llama3"))
        sid = asm.initialize()

        # Save messages
        asm.save_message(_make_message(content="user question"))
        asm.save_message(_make_message(role="assistant", content="answer"))

        # Record tokens
        token_store = TokenUsageStore(workspace_dir=self._tmp)
        token_store.record_snapshot(sid, _make_tracker(100, 50, cost=0.01))

        # Cache a response
        cache = HybridResponseCache(workspace_dir=self._tmp)
        cache.put("q_hash", _make_response("cached answer"))

        # Snapshot state
        snap_mgr = StateSnapshotManager(workspace_dir=self._tmp)
        snap_id = snap_mgr.create_snapshot(
            messages=asm.get_session_messages(),
            label="lifecycle test",
            session_id=sid,
            token_usage_summary={"input": 100, "output": 50},
        )

        # Verify everything
        self.assertEqual(token_store.snapshot_count, 1)
        self.assertIsNotNone(cache.get("q_hash"))
        snapshot = snap_mgr.restore_snapshot(snap_id)
        self.assertEqual(len(snapshot.messages), 2)
        self.assertEqual(snapshot.session_id, sid)


# ═══════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
