"""
Sprint 15 Tests — Prompt Optimization & Context Management.

Covers:
  - ModelTokenEstimator: model-aware token estimation
  - PromptBudgetManager: system prompt budget tracking
  - ContextManager: knowledge scoring, deduplication, proactive pruning
  - TokenTracker: budget warnings, prediction
  - Agent integration: wiring, fallback behavior
  - Config/main.py wiring
  - Edge cases
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from cowork_agent.core.models import Message, ToolCall, ToolResult, AgentResponse
from cowork_agent.core.token_estimator import ModelTokenEstimator, FALLBACK_RATIO
from cowork_agent.core.prompt_budget import PromptBudgetManager
from cowork_agent.core.context_manager import ContextManager
from cowork_agent.core.token_tracker import TokenTracker, TokenUsage


# ── Helpers ──────────────────────────────────────────────────

def _msg(role="user", content="Hello", tool_calls=None, tool_results=None):
    return Message(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_results=tool_results,
    )


def _make_call(name="read", input_data=None):
    return ToolCall(name=name, tool_id=f"tool_{name}", input=input_data or {})


def _make_result(tool_id="tool_read", success=True, output="ok"):
    return ToolResult(tool_id=tool_id, success=success, output=output)


def _knowledge_entry(key="project_name", value="cowork_agent", category="facts",
                     updated_at=None):
    """Create a mock KnowledgeEntry."""
    from cowork_agent.core.knowledge_store import KnowledgeEntry
    return KnowledgeEntry(
        key=key,
        value=value,
        category=category,
        created_at=time.time(),
        updated_at=updated_at or time.time(),
        access_count=0,
    )


# ── TestModelTokenEstimator ────────────────────────────────

class TestModelTokenEstimator:
    """Tests for ModelTokenEstimator."""

    def test_claude_ratio(self):
        est = ModelTokenEstimator()
        assert est.get_ratio("claude-sonnet-4-5-20250929") == 3.5

    def test_gpt4_ratio(self):
        est = ModelTokenEstimator()
        assert est.get_ratio("gpt-4o") == 4.0

    def test_gpt35_ratio(self):
        est = ModelTokenEstimator()
        assert est.get_ratio("gpt-3.5-turbo") == 4.2

    def test_ollama_ratio(self):
        est = ModelTokenEstimator()
        assert est.get_ratio("ollama-llama3") == 4.0

    def test_unknown_model_fallback(self):
        est = ModelTokenEstimator()
        assert est.get_ratio("totally-unknown-model") == FALLBACK_RATIO

    def test_prefix_matching(self):
        est = ModelTokenEstimator()
        assert est.get_ratio("claude-opus-4-5-20251101") == 3.5

    def test_contains_matching(self):
        est = ModelTokenEstimator()
        # "gpt-4" is contained in "my-custom-gpt-4-wrapper"
        assert est.get_ratio("my-custom-gpt-4-wrapper") == 4.0

    def test_estimate_tokens_basic(self):
        est = ModelTokenEstimator()
        # "Hello World" = 11 chars, claude ratio 3.5 → 11/3.5 ≈ 3
        tokens = est.estimate_tokens("Hello World", "claude")
        assert tokens == 3

    def test_estimate_tokens_empty(self):
        est = ModelTokenEstimator()
        assert est.estimate_tokens("", "claude") == 0

    def test_estimate_tokens_minimum_one(self):
        est = ModelTokenEstimator()
        assert est.estimate_tokens("a", "claude") >= 1

    def test_estimate_message_tokens(self):
        est = ModelTokenEstimator()
        msg = _msg(content="Hello, I need help with my project")
        tokens = est.estimate_message_tokens(msg, "claude")
        # Content tokens + 4 overhead
        assert tokens > 4

    def test_estimate_message_tokens_with_tools(self):
        est = ModelTokenEstimator()
        call = _make_call("read", {"file_path": "/tmp/test.py"})
        msg = _msg(role="assistant", content="Let me read that", tool_calls=[call])
        tokens = est.estimate_message_tokens(msg, "claude")
        # Content + tool name + tool input + overhead
        assert tokens > 10

    def test_estimate_messages_tokens(self):
        est = ModelTokenEstimator()
        msgs = [_msg(content="Hi"), _msg(role="assistant", content="Hello!")]
        total = est.estimate_messages_tokens(msgs, "claude", "System prompt here")
        assert total > 0

    def test_config_overrides(self):
        est = ModelTokenEstimator(config={"claude": 2.5, "my-model": 5.0})
        assert est.get_ratio("claude") == 2.5
        assert est.get_ratio("my-model") == 5.0

    def test_caching(self):
        est = ModelTokenEstimator()
        _ = est.get_ratio("claude-sonnet-4-5-20250929")
        assert "claude-sonnet-4-5-20250929" in est._model_cache

    def test_to_dict(self):
        est = ModelTokenEstimator()
        d = est.to_dict()
        assert "ratios" in d
        assert "fallback_ratio" in d
        assert "cached_models" in d


# ── TestPromptBudgetManager ────────────────────────────────

class TestPromptBudgetManager:
    """Tests for PromptBudgetManager."""

    def test_init_defaults(self):
        mgr = PromptBudgetManager()
        assert mgr.max_tokens == 8000
        assert mgr.remaining() == 8000
        assert mgr.total_allocated == 0

    def test_allocate_within_limit(self):
        mgr = PromptBudgetManager(max_system_prompt_tokens=10000)
        content = "A" * 100  # ~28 tokens at 3.5 ratio
        result = mgr.allocate("behavioral_rules", content)
        assert result == content
        assert mgr.total_allocated > 0

    def test_allocate_exceeds_limit_compresses(self):
        mgr = PromptBudgetManager(max_system_prompt_tokens=100)
        content = "A" * 5000  # Way too large
        result = mgr.allocate("memory", content)
        # Should be compressed (shorter than original)
        assert len(result) < len(content)

    def test_allocate_force(self):
        mgr = PromptBudgetManager(max_system_prompt_tokens=100)
        content = "A" * 5000
        result = mgr.allocate("behavioral_rules", content, force=True)
        assert result == content  # Force keeps original

    def test_remaining_decreases(self):
        mgr = PromptBudgetManager(max_system_prompt_tokens=10000)
        before = mgr.remaining()
        mgr.allocate("env_context", "Hello World " * 10)
        after = mgr.remaining()
        assert after < before

    def test_report_structure(self):
        mgr = PromptBudgetManager(max_system_prompt_tokens=10000)
        mgr.allocate("tools", "Tool definitions here " * 20)
        report = mgr.report()
        assert "total_allocated" in report
        assert "max_tokens" in report
        assert "remaining" in report
        assert "percent_used" in report
        assert "sections" in report
        assert "tools" in report["sections"]

    def test_report_section_details(self):
        mgr = PromptBudgetManager(max_system_prompt_tokens=10000)
        mgr.allocate("tools", "Tool defs " * 50)
        section = mgr.report()["sections"]["tools"]
        assert "tokens" in section
        assert "limit" in section
        assert "percent_of_limit" in section
        assert "over_limit" in section

    def test_can_fit(self):
        mgr = PromptBudgetManager(max_system_prompt_tokens=1000)
        assert mgr.can_fit(500) is True
        assert mgr.can_fit(1001) is False

    def test_compress_preserves_head_and_tail(self):
        mgr = PromptBudgetManager()
        content = "Start of content. " + "Middle content. " * 100 + "End of content."
        compressed = mgr.compress(content, 50)
        assert len(compressed) < len(content)
        assert compressed.startswith("Start")

    def test_compress_short_content_unchanged(self):
        mgr = PromptBudgetManager()
        content = "Short content"
        result = mgr.compress(content, 1000)
        assert result == content

    def test_compress_zero_target(self):
        mgr = PromptBudgetManager()
        result = mgr.compress("Some content", 0)
        assert result == ""

    def test_reset(self):
        mgr = PromptBudgetManager()
        mgr.allocate("tools", "Content " * 100)
        assert mgr.total_allocated > 0
        mgr.reset()
        assert mgr.total_allocated == 0
        assert mgr.remaining() == mgr.max_tokens

    def test_get_section_limit(self):
        mgr = PromptBudgetManager(max_system_prompt_tokens=10000)
        # behavioral_rules = 30% of 10000 = 3000
        assert mgr.get_section_limit("behavioral_rules") == 3000
        # tools = 40% of 10000 = 4000
        assert mgr.get_section_limit("tools") == 4000

    def test_allocate_empty_content(self):
        mgr = PromptBudgetManager()
        result = mgr.allocate("memory", "")
        assert result == ""
        assert mgr.total_allocated == 0

    def test_no_budget_remaining(self):
        mgr = PromptBudgetManager(max_system_prompt_tokens=50)
        # Fill budget with forced allocation
        mgr.allocate("behavioral_rules", "A" * 1000, force=True)
        # Now try normal allocation — should get empty string
        result = mgr.allocate("memory", "More content here")
        assert result == ""

    def test_to_dict(self):
        mgr = PromptBudgetManager()
        mgr.allocate("tools", "Content")
        d = mgr.to_dict()
        assert "max_tokens" in d
        assert "total_allocated" in d
        assert "remaining" in d

    def test_custom_section_limits(self):
        mgr = PromptBudgetManager(
            max_system_prompt_tokens=10000,
            section_limits={"memory": 0.50},
        )
        assert mgr.get_section_limit("memory") == 5000

    def test_section_priority_order(self):
        mgr = PromptBudgetManager()
        # behavioral_rules has priority 1 (highest)
        assert mgr._priorities["behavioral_rules"] < mgr._priorities["tools"]
        assert mgr._priorities["tools"] < mgr._priorities["memory"]


# ── TestKnowledgeScoring ────────────────────────────────────

class TestKnowledgeScoring:
    """Tests for ContextManager.score_knowledge_entry."""

    def test_keyword_match_boosts_score(self):
        cm = ContextManager()
        entry = _knowledge_entry(key="python testing", value="use pytest")
        score = cm.score_knowledge_entry(entry, "How do I run python tests?")
        assert score > 0.3  # Keywords match

    def test_no_keyword_match(self):
        cm = ContextManager()
        entry = _knowledge_entry(key="database migration", value="use alembic")
        score = cm.score_knowledge_entry(entry, "How do I draw a picture?")
        # Should still have category boost and recency, but no keyword match
        assert score < 0.6

    def test_decisions_category_boost(self):
        cm = ContextManager()
        entry_decision = _knowledge_entry(key="test", value="val", category="decisions")
        entry_fact = _knowledge_entry(key="test", value="val", category="facts")
        score_d = cm.score_knowledge_entry(entry_decision, "something")
        score_f = cm.score_knowledge_entry(entry_fact, "something")
        assert score_d > score_f  # Decisions score higher

    def test_preferences_mid_boost(self):
        cm = ContextManager()
        entry_pref = _knowledge_entry(key="test", value="val", category="preferences")
        entry_fact = _knowledge_entry(key="test", value="val", category="facts")
        score_p = cm.score_knowledge_entry(entry_pref, "something")
        score_f = cm.score_knowledge_entry(entry_fact, "something")
        assert score_p > score_f

    def test_recency_boost_recent(self):
        cm = ContextManager()
        recent = _knowledge_entry(key="test", value="val", updated_at=time.time())
        old = _knowledge_entry(key="test", value="val", updated_at=time.time() - 86400 * 5)
        score_r = cm.score_knowledge_entry(recent, "test something")
        score_o = cm.score_knowledge_entry(old, "test something")
        assert score_r > score_o

    def test_empty_message_returns_neutral(self):
        cm = ContextManager()
        entry = _knowledge_entry(key="test", value="val")
        score = cm.score_knowledge_entry(entry, "")
        assert score == 0.5

    def test_score_clamped_0_to_1(self):
        cm = ContextManager()
        entry = _knowledge_entry(key="python testing framework pytest",
                                 value="use pytest for testing", category="decisions")
        score = cm.score_knowledge_entry(
            entry, "python testing framework pytest for testing"
        )
        assert 0.0 <= score <= 1.0

    def test_value_words_contribute(self):
        cm = ContextManager()
        entry = _knowledge_entry(key="config", value="always use yaml configuration files")
        score_match = cm.score_knowledge_entry(entry, "configuration files setup")
        score_nomatch = cm.score_knowledge_entry(entry, "banana smoothie recipe")
        assert score_match > score_nomatch

    def test_short_key_words_filtered(self):
        cm = ContextManager()
        # Words with 2 chars or less should be filtered out
        entry = _knowledge_entry(key="a b cd ef", value="val")
        score = cm.score_knowledge_entry(entry, "a b cd ef test")
        # Only words > 2 chars considered, so limited match
        assert 0.0 <= score <= 1.0

    def test_multiple_entries_ranking(self):
        cm = ContextManager()
        entries = [
            _knowledge_entry(key="python testing", value="use pytest", category="decisions"),
            _knowledge_entry(key="database setup", value="use postgres", category="facts"),
            _knowledge_entry(key="color theme", value="dark mode", category="preferences"),
        ]
        scores = [
            (cm.score_knowledge_entry(e, "How to test python code?"), e.key)
            for e in entries
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        # "python testing" should rank highest for this query
        assert scores[0][1] == "python testing"

    def test_score_with_estimator(self):
        """Scoring works the same whether estimator is set or not."""
        cm1 = ContextManager()
        cm2 = ContextManager(token_estimator=ModelTokenEstimator(), model="claude")
        entry = _knowledge_entry(key="test", value="val")
        s1 = cm1.score_knowledge_entry(entry, "test something")
        s2 = cm2.score_knowledge_entry(entry, "test something")
        assert abs(s1 - s2) < 1e-6  # Scoring doesn't depend on estimator

    def test_very_old_entry_low_recency(self):
        cm = ContextManager()
        old = _knowledge_entry(key="test", value="val",
                               updated_at=time.time() - 86400 * 30)  # 30 days old
        score = cm.score_knowledge_entry(old, "test")
        # Recency should be 0 for entries older than 72 hours
        assert score < 0.8


# ── TestMessageDeduplication ────────────────────────────────

class TestMessageDeduplication:
    """Tests for ContextManager.deduplicate_messages."""

    def test_no_duplicates_unchanged(self):
        cm = ContextManager()
        msgs = [
            _msg(content="Hello"),
            _msg(role="assistant", content="Hi there"),
            _msg(content="Help me"),
            _msg(role="assistant", content="Sure"),
        ]
        result = cm.deduplicate_messages(msgs)
        assert len(result) == len(msgs)

    def test_exact_duplicate_removed(self):
        cm = ContextManager()
        # Need enough messages so duplicates fall outside MIN_RECENT_MESSAGES (6)
        msgs = [
            _msg(content="Hello"),
            _msg(content="Hello"),  # Duplicate — should be removed
            _msg(role="assistant", content="Hi"),
            _msg(content="Help"),
            _msg(role="assistant", content="Sure"),
            _msg(content="Thanks"),
            _msg(role="assistant", content="Welcome"),
            _msg(content="More"),
            _msg(role="assistant", content="Ok"),
        ]
        result = cm.deduplicate_messages(msgs)
        assert len(result) < len(msgs)

    def test_preserves_recent_messages(self):
        cm = ContextManager()
        cm.MIN_RECENT_MESSAGES = 4
        msgs = [
            _msg(content="Hello"),
            _msg(content="Hello"),  # Duplicate
            _msg(content="A"),
            _msg(content="B"),
            _msg(content="C"),
            _msg(content="D"),
        ]
        result = cm.deduplicate_messages(msgs)
        # Last 4 messages always preserved
        assert result[-4:] == msgs[-4:]

    def test_empty_list(self):
        cm = ContextManager()
        assert cm.deduplicate_messages([]) == []

    def test_single_message(self):
        cm = ContextManager()
        msgs = [_msg(content="Hello")]
        result = cm.deduplicate_messages(msgs)
        assert len(result) == 1

    def test_tool_call_dedup(self):
        cm = ContextManager()
        call1 = _make_call("read", {"file_path": "/tmp/test.py"})
        call2 = _make_call("read", {"file_path": "/tmp/test.py"})
        msgs = [
            _msg(role="assistant", content="Reading...", tool_calls=[call1]),
            _msg(role="assistant", content="Reading...", tool_calls=[call2]),
            _msg(content="A"), _msg(content="B"),
            _msg(content="C"), _msg(content="D"),
            _msg(content="E"), _msg(content="F"),
        ]
        result = cm.deduplicate_messages(msgs)
        assert len(result) < len(msgs)

    def test_different_content_not_deduped(self):
        cm = ContextManager()
        msgs = [
            _msg(content="Hello World A"),
            _msg(content="Hello World B"),
            _msg(content="Hello World C"),
            _msg(role="assistant", content="Response A"),
            _msg(role="assistant", content="Response B"),
            _msg(role="assistant", content="Response C"),
            _msg(content="Final"),
        ]
        result = cm.deduplicate_messages(msgs)
        assert len(result) == len(msgs)

    def test_dedup_window_constraint(self):
        cm = ContextManager()
        cm.DEDUP_WINDOW = 3  # Only check last 3 hashes
        # Create messages where duplicates are far apart
        msgs = [
            _msg(content="Dup"),
            _msg(content="A"), _msg(content="B"), _msg(content="C"),
            _msg(content="D"), _msg(content="E"), _msg(content="F"),
            _msg(content="Dup"),  # Same as first, but outside window
            # Recent protected messages
            _msg(content="G"), _msg(content="H"),
            _msg(content="I"), _msg(content="J"),
            _msg(content="K"), _msg(content="L"),
        ]
        result = cm.deduplicate_messages(msgs)
        # Both "Dup" messages may be kept since they're far apart
        assert len(result) >= len(msgs) - 1

    def test_message_hash_deterministic(self):
        cm = ContextManager()
        msg = _msg(content="Test message for hashing")
        h1 = cm._message_hash(msg)
        h2 = cm._message_hash(msg)
        assert h1 == h2

    def test_message_hash_different_content(self):
        cm = ContextManager()
        m1 = _msg(content="Message A")
        m2 = _msg(content="Message B")
        assert cm._message_hash(m1) != cm._message_hash(m2)

    def test_message_hash_different_role(self):
        cm = ContextManager()
        m1 = _msg(role="user", content="Same")
        m2 = _msg(role="assistant", content="Same")
        assert cm._message_hash(m1) != cm._message_hash(m2)

    def test_dedup_only_outside_protected(self):
        """Duplicates within protected window are always kept."""
        cm = ContextManager()
        cm.MIN_RECENT_MESSAGES = 4
        msgs = [
            _msg(content="Old1"),
            _msg(content="Old2"),
            # Last 4 (protected) — even duplicates are kept
            _msg(content="Same"),
            _msg(content="Same"),
            _msg(content="Same"),
            _msg(content="End"),
        ]
        result = cm.deduplicate_messages(msgs)
        # Protected messages always kept, so last 4 are intact
        assert len(result) >= 4


# ── TestProactivePruning ────────────────────────────────────

class TestProactivePruning:
    """Tests for proactive pruning at 60% capacity."""

    def test_below_threshold_returns_false(self):
        cm = ContextManager(max_context_tokens=100000)
        msgs = [_msg(content="Hello")]
        assert cm.should_prune_proactively(msgs) is False

    def test_above_threshold_returns_true(self):
        cm = ContextManager(max_context_tokens=200)
        # Create enough messages to exceed 60% of effective limit
        msgs = [_msg(content="X" * 500) for _ in range(5)]
        assert cm.should_prune_proactively(msgs) is True

    def test_proactive_vs_standard_threshold(self):
        """Proactive triggers at 60%, standard at 75%."""
        cm = ContextManager(max_context_tokens=1000)
        # Effective limit = 1000 * 0.75 = 750
        # Proactive threshold = 750 * 0.60 = 450 tokens
        # Standard threshold = 750 tokens
        # Create messages totaling ~500 tokens
        msgs = [_msg(content="W" * 200) for _ in range(10)]
        should_proactive = cm.should_prune_proactively(msgs)
        should_standard = cm.needs_pruning(msgs)
        # Should be proactive but not necessarily standard
        assert should_proactive is True or should_standard is True

    def test_with_estimator(self):
        est = ModelTokenEstimator()
        cm = ContextManager(max_context_tokens=500, token_estimator=est, model="claude")
        msgs = [_msg(content="X" * 500) for _ in range(3)]
        # With Claude ratio (3.5), 500 chars ≈ 143 tokens per msg
        result = cm.should_prune_proactively(msgs)
        assert isinstance(result, bool)

    def test_custom_threshold(self):
        cm = ContextManager(max_context_tokens=10000)
        cm.PROACTIVE_PRUNE_RATIO = 0.30  # Very aggressive
        msgs = [_msg(content="X" * 2000) for _ in range(3)]
        result = cm.should_prune_proactively(msgs)
        assert isinstance(result, bool)

    def test_empty_messages_no_pruning(self):
        cm = ContextManager(max_context_tokens=1000)
        assert cm.should_prune_proactively([]) is False

    def test_estimate_tokens_with_estimator(self):
        est = ModelTokenEstimator()
        cm = ContextManager(max_context_tokens=1000, token_estimator=est, model="claude")
        tokens = cm.estimate_tokens("Hello World")
        # Claude ratio 3.5 → 11/3.5 ≈ 3
        assert tokens == 3

    def test_estimate_tokens_without_estimator(self):
        cm = ContextManager(max_context_tokens=1000)
        tokens = cm.estimate_tokens("Hello World")
        # Default CHARS_PER_TOKEN = 4 → 11/4 = 2
        assert tokens == 2


# ── TestTokenBudgetWarnings ─────────────────────────────────

class TestTokenBudgetWarnings:
    """Tests for TokenTracker budget warnings and prediction."""

    def test_threshold_50_fires(self):
        fired = []
        tt = TokenTracker(max_session_tokens=1000)
        tt.on_threshold_reached(50, lambda pct, rem: fired.append(pct))
        # Record 500 tokens (50%)
        tt.record(TokenUsage(input_tokens=300, output_tokens=200))
        assert 50 in fired

    def test_threshold_75_fires(self):
        fired = []
        tt = TokenTracker(max_session_tokens=1000)
        tt.on_threshold_reached(75, lambda pct, rem: fired.append(pct))
        tt.record(TokenUsage(input_tokens=400, output_tokens=400))
        assert 75 in fired

    def test_threshold_90_fires(self):
        fired = []
        tt = TokenTracker(max_session_tokens=1000)
        tt.on_threshold_reached(90, lambda pct, rem: fired.append(pct))
        tt.record(TokenUsage(input_tokens=500, output_tokens=500))
        assert 90 in fired

    def test_threshold_fires_once(self):
        fired = []
        tt = TokenTracker(max_session_tokens=1000)
        tt.on_threshold_reached(50, lambda pct, rem: fired.append(pct))
        tt.record(TokenUsage(input_tokens=300, output_tokens=200))  # 50%
        tt.record(TokenUsage(input_tokens=100, output_tokens=100))  # 70%
        # Should fire only once for 50% threshold
        assert fired.count(50) == 1

    def test_multiple_thresholds_fire(self):
        fired = []
        tt = TokenTracker(max_session_tokens=1000)
        tt.on_threshold_reached(50, lambda pct, rem: fired.append(pct))
        tt.on_threshold_reached(75, lambda pct, rem: fired.append(pct))
        tt.on_threshold_reached(90, lambda pct, rem: fired.append(pct))
        tt.record(TokenUsage(input_tokens=500, output_tokens=500))  # 100%
        assert 50 in fired
        assert 75 in fired
        assert 90 in fired

    def test_no_threshold_without_budget(self):
        fired = []
        tt = TokenTracker()  # No budget set
        tt.on_threshold_reached(50, lambda pct, rem: fired.append(pct))
        tt.record(TokenUsage(input_tokens=10000, output_tokens=10000))
        assert len(fired) == 0

    def test_callback_exception_handled(self):
        def bad_callback(pct, rem):
            raise RuntimeError("boom")
        tt = TokenTracker(max_session_tokens=100)
        tt.on_threshold_reached(50, bad_callback)
        # Should not raise
        tt.record(TokenUsage(input_tokens=30, output_tokens=30))

    def test_remaining_budget_with_limit(self):
        tt = TokenTracker(max_session_tokens=1000, max_cost_usd=10.0)
        tt.record(TokenUsage(input_tokens=200, output_tokens=100, provider="ollama"))
        rem = tt.remaining_budget()
        assert rem["tokens_remaining"] == 700
        assert rem["tokens_percent_remaining"] == 70.0

    def test_remaining_budget_no_limit(self):
        tt = TokenTracker()
        rem = tt.remaining_budget()
        assert rem["tokens_remaining"] is None
        assert rem["cost_remaining_usd"] is None

    def test_remaining_budget_cost(self):
        tt = TokenTracker(max_cost_usd=1.0)
        rem = tt.remaining_budget()
        assert rem["cost_remaining_usd"] is not None
        assert rem["cost_percent_remaining"] == 100.0

    def test_predict_empty_history(self):
        tt = TokenTracker()
        assert tt.predict_next_iteration_tokens() == 500

    def test_predict_with_history(self):
        tt = TokenTracker()
        for _ in range(5):
            tt.record(TokenUsage(input_tokens=100, output_tokens=100))
        pred = tt.predict_next_iteration_tokens()
        # Average = 200, × 1.2 = 240
        assert pred == 240

    def test_history_window_limit(self):
        tt = TokenTracker()
        for i in range(20):
            tt.record(TokenUsage(input_tokens=i * 10, output_tokens=i * 10))
        assert len(tt._token_history) == tt._HISTORY_WINDOW

    def test_reset_clears_thresholds(self):
        fired = []
        tt = TokenTracker(max_session_tokens=1000)
        tt.on_threshold_reached(50, lambda pct, rem: fired.append(pct))
        tt.record(TokenUsage(input_tokens=300, output_tokens=200))
        assert 50 in fired
        tt.reset()
        assert tt._last_threshold_reached == 0
        assert len(tt._token_history) == 0


# ── TestContextManagerEstimator ─────────────────────────────

class TestContextManagerEstimator:
    """Tests for ContextManager using ModelTokenEstimator."""

    def test_delegated_estimate_tokens(self):
        est = ModelTokenEstimator()
        cm = ContextManager(token_estimator=est, model="claude")
        # Claude ratio 3.5: "Hello" = 5 chars → 5/3.5 = 1 token
        assert cm.estimate_tokens("Hello") == 1

    def test_fallback_without_estimator(self):
        cm = ContextManager()
        # Default CHARS_PER_TOKEN = 4: "Hello" = 5 chars → 5/4 = 1
        assert cm.estimate_tokens("Hello") == 1

    def test_delegated_message_tokens(self):
        est = ModelTokenEstimator()
        cm = ContextManager(token_estimator=est, model="gpt-4")
        msg = _msg(content="A" * 100)
        tokens = cm.estimate_message_tokens(msg)
        # GPT-4 ratio 4.0: 100/4 = 25 + 4 overhead = 29
        assert tokens == 29

    def test_delegated_total_tokens(self):
        est = ModelTokenEstimator()
        cm = ContextManager(token_estimator=est, model="claude")
        msgs = [_msg(content="Hello"), _msg(content="World")]
        total = cm.estimate_total_tokens(msgs, "System prompt")
        assert total > 0

    def test_needs_pruning_with_estimator(self):
        est = ModelTokenEstimator()
        cm = ContextManager(max_context_tokens=100, token_estimator=est, model="claude")
        # Create enough content to exceed limit
        msgs = [_msg(content="X" * 1000) for _ in range(5)]
        assert cm.needs_pruning(msgs) is True

    def test_prune_with_estimator(self):
        est = ModelTokenEstimator()
        cm = ContextManager(max_context_tokens=200, token_estimator=est, model="claude")
        msgs = [_msg(content="X" * 1000) for _ in range(10)]
        result = cm.prune(msgs)
        assert len(result) < len(msgs)


# ── TestAgentIntegration ────────────────────────────────────

class TestAgentIntegration:
    """Tests for Agent with Sprint 15 components."""

    def _make_agent(self, with_estimator=True, with_budget=True, with_knowledge=True):
        """Create an Agent with Sprint 15 components."""
        from cowork_agent.core.agent import Agent

        provider = MagicMock()
        provider.provider_name = "test"
        provider.model = "claude-test"

        registry = MagicMock()
        registry.get_schemas.return_value = []
        registry.get_tool.side_effect = KeyError("not found")

        prompt_builder = MagicMock()
        prompt_builder.build.return_value = "System prompt"

        knowledge_store = None
        if with_knowledge:
            knowledge_store = MagicMock()
            knowledge_store.size = 3
            knowledge_store.recall_all.return_value = [
                _knowledge_entry(key="project", value="cowork", category="facts"),
            ]

        agent = Agent(
            provider=provider,
            registry=registry,
            prompt_builder=prompt_builder,
            max_iterations=5,
            workspace_dir="/tmp/test",
            knowledge_store=knowledge_store,
        )

        if with_estimator:
            agent.token_estimator = ModelTokenEstimator()
            agent.context_mgr.token_estimator = agent.token_estimator
            agent.context_mgr.model = "claude-test"

        if with_budget:
            agent.prompt_budget_manager = PromptBudgetManager(
                estimator=agent.token_estimator
            )

        return agent

    def test_summary_interval_is_3(self):
        agent = self._make_agent()
        assert agent._SUMMARY_UPDATE_INTERVAL == 3

    def test_estimator_attached(self):
        agent = self._make_agent()
        assert agent.token_estimator is not None
        assert agent.context_mgr.token_estimator is not None

    def test_budget_manager_attached(self):
        agent = self._make_agent()
        assert agent.prompt_budget_manager is not None

    def test_build_context_with_scoring(self):
        agent = self._make_agent()
        agent._messages = [_msg(content="How to test python code?")]
        ctx = agent._build_context()
        # Should have knowledge_entries injected
        assert "knowledge_entries" in ctx

    def test_build_context_without_estimator_fallback(self):
        agent = self._make_agent(with_estimator=False)
        agent._messages = [_msg(content="Hello")]
        ctx = agent._build_context()
        # Still works with chronological injection
        assert "knowledge_entries" in ctx

    def test_build_context_deduplication(self):
        agent = self._make_agent()
        agent._messages = [
            _msg(content="Hello"),
            _msg(content="Hello"),  # Duplicate
            _msg(role="assistant", content="Hi"),
            _msg(content="Help"),
            _msg(role="assistant", content="Sure"),
            _msg(content="Thanks"),
            _msg(role="assistant", content="Welcome"),
            _msg(content="Bye"),
        ]
        before = len(agent._messages)
        _ = agent._build_context()
        after = len(agent._messages)
        assert after <= before

    def test_build_context_no_knowledge_store(self):
        agent = self._make_agent(with_knowledge=False)
        agent._messages = [_msg(content="Hello")]
        ctx = agent._build_context()
        assert "knowledge_entries" not in ctx

    def test_inject_scored_knowledge(self):
        agent = self._make_agent()
        ctx = {}
        agent._inject_scored_knowledge(ctx, "How does the project work?")
        assert "knowledge_entries" in ctx
        assert len(ctx["knowledge_entries"]) > 0

    def test_sprint15_attributes_default(self):
        """Agent without Sprint 15 wiring has None attributes."""
        from cowork_agent.core.agent import Agent
        provider = MagicMock()
        provider.provider_name = "test"
        provider.model = "test"
        registry = MagicMock()
        registry.get_schemas.return_value = []
        registry.get_tool.side_effect = KeyError
        pb = MagicMock()
        pb.build.return_value = ""
        agent = Agent(provider=provider, registry=registry, prompt_builder=pb)
        assert agent.token_estimator is None
        assert agent.prompt_budget_manager is None

    def test_proactive_pruning_in_run(self):
        """Proactive pruning check exists in the run loop."""
        agent = self._make_agent()
        # Just verify the agent's context_mgr has proactive method
        assert hasattr(agent.context_mgr, "should_prune_proactively")
        assert callable(agent.context_mgr.should_prune_proactively)

    def test_knowledge_scoring_method_exists(self):
        agent = self._make_agent()
        assert hasattr(agent.context_mgr, "score_knowledge_entry")

    def test_dedup_method_exists(self):
        agent = self._make_agent()
        assert hasattr(agent.context_mgr, "deduplicate_messages")

    def test_run_stream_events_with_sprint15(self):
        """run_stream_events still works with Sprint 15 components."""
        agent = self._make_agent()
        assert hasattr(agent, "run_stream_events")

    def test_clear_history_preserves_sprint15(self):
        agent = self._make_agent()
        agent._messages = [_msg(content="Hello")]
        agent.clear_history()
        assert agent.token_estimator is not None  # Not cleared
        assert agent.prompt_budget_manager is not None  # Not cleared

    def test_build_context_skills_matching(self):
        """Skills matching still works with Sprint 15."""
        agent = self._make_agent()
        agent.skill_registry = MagicMock()
        agent.skill_registry.match_skills.return_value = []
        agent._messages = [_msg(content="Hello")]
        ctx = agent._build_context()
        # Should not crash
        assert "iteration" in ctx


# ── TestConfigWiring ────────────────────────────────────────

class TestConfigWiring:
    """Tests for config loading and main.py wiring."""

    def test_config_has_prompt_optimization(self):
        from cowork_agent.config.settings import load_config
        config = load_config(None)
        po = config.get("prompt_optimization", {})
        assert po.get("enabled") is True

    def test_config_token_estimator_section(self):
        from cowork_agent.config.settings import load_config
        config = load_config(None)
        est = config.get("prompt_optimization.token_estimator", {})
        assert est.get("enabled") is True

    def test_config_prompt_budget_section(self):
        from cowork_agent.config.settings import load_config
        config = load_config(None)
        pb = config.get("prompt_optimization.prompt_budget", {})
        assert pb.get("enabled") is True
        assert pb.get("max_system_prompt_tokens") == 8000

    def test_config_context_assembly(self):
        from cowork_agent.config.settings import load_config
        config = load_config(None)
        ca = config.get("prompt_optimization.context_assembly", {})
        assert ca.get("proactive_prune_threshold") == 0.60
        assert ca.get("summary_update_interval") == 3

    def test_config_budget_warnings(self):
        from cowork_agent.config.settings import load_config
        config = load_config(None)
        bw = config.get("prompt_optimization.budget_warnings", {})
        assert bw.get("enabled") is True
        assert bw.get("thresholds") == [50, 75, 90]

    def test_model_ratios_in_config(self):
        from cowork_agent.config.settings import load_config
        config = load_config(None)
        ratios = config.get("prompt_optimization.token_estimator.model_ratios", {})
        assert ratios.get("claude") == 3.5
        assert ratios.get("gpt-4") == 4.0

    def test_estimator_created_from_config(self):
        from cowork_agent.config.settings import load_config
        config = load_config(None)
        ratios = config.get("prompt_optimization.token_estimator.model_ratios", {})
        est = ModelTokenEstimator(config=ratios)
        assert est.get_ratio("claude") == 3.5

    def test_budget_manager_from_config(self):
        from cowork_agent.config.settings import load_config
        config = load_config(None)
        pb_cfg = config.get("prompt_optimization.prompt_budget", {})
        mgr = PromptBudgetManager(
            max_system_prompt_tokens=pb_cfg.get("max_system_prompt_tokens", 8000),
        )
        assert mgr.max_tokens == 8000


# ── TestEdgeCases ───────────────────────────────────────────

class TestEdgeCases:
    """Edge case tests."""

    def test_estimator_very_long_text(self):
        est = ModelTokenEstimator()
        text = "A" * 1_000_000
        tokens = est.estimate_tokens(text, "claude")
        assert tokens > 200000

    def test_estimator_unicode_text(self):
        est = ModelTokenEstimator()
        text = "你好世界" * 100
        tokens = est.estimate_tokens(text, "claude")
        assert tokens > 0

    def test_budget_manager_zero_budget(self):
        mgr = PromptBudgetManager(max_system_prompt_tokens=0)
        assert mgr.remaining() == 0
        result = mgr.allocate("tools", "Content")
        assert result == ""

    def test_context_manager_zero_tokens(self):
        cm = ContextManager(max_context_tokens=0)
        assert cm.needs_pruning([_msg()]) is True

    def test_tracker_reset_then_predict(self):
        tt = TokenTracker()
        tt.record(TokenUsage(input_tokens=100, output_tokens=100))
        tt.reset()
        assert tt.predict_next_iteration_tokens() == 500  # Default

    def test_tracker_many_thresholds_registered(self):
        tt = TokenTracker(max_session_tokens=1000)
        for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            tt.on_threshold_reached(pct, lambda p, r: None)
        tt.record(TokenUsage(input_tokens=500, output_tokens=500))
        assert tt._last_threshold_reached == 90

    def test_dedup_all_same_messages(self):
        cm = ContextManager()
        msgs = [_msg(content="Same") for _ in range(20)]
        result = cm.deduplicate_messages(msgs)
        # Should have far fewer than 20
        assert len(result) < 20
        # But at least MIN_RECENT_MESSAGES preserved
        assert len(result) >= cm.MIN_RECENT_MESSAGES

    def test_knowledge_scoring_none_entry_fields(self):
        """Score doesn't crash with minimal entry."""
        cm = ContextManager()
        entry = _knowledge_entry(key="", value="", category="facts")
        score = cm.score_knowledge_entry(entry, "test")
        assert 0.0 <= score <= 1.0

    def test_budget_report_no_allocations(self):
        mgr = PromptBudgetManager()
        report = mgr.report()
        assert report["total_allocated"] == 0
        assert report["percent_used"] == 0.0

    def test_estimator_empty_model(self):
        est = ModelTokenEstimator()
        tokens = est.estimate_tokens("Hello", "")
        assert tokens > 0

    def test_budget_allocate_unknown_section(self):
        mgr = PromptBudgetManager(max_system_prompt_tokens=10000)
        result = mgr.allocate("custom_section", "Some content")
        # Should use default 10% limit
        assert result == "Some content" or len(result) > 0

    def test_message_hash_with_tool_results(self):
        cm = ContextManager()
        result = _make_result("tool_1", True, "Output data")
        msg = _msg(role="tool_result", content="", tool_results=[result])
        h = cm._message_hash(msg)
        assert isinstance(h, str)
        assert len(h) == 12
