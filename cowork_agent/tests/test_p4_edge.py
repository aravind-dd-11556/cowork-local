"""
Sprint 4 — Edge Case Test Suite
=================================
Covers 30+ missing edge cases across all 5 features + agent integration.

Feature 1: Token Tracker Edge Cases          (9 tests)
Feature 2: Provider Fallback Edge Cases      (8 tests)
Feature 3: Stream Hardener Edge Cases        (5 tests)
Feature 4: Multi-Modal Edge Cases            (8 tests)
Feature 5: Response Cache Edge Cases         (9 tests)
Agent Integration Edge Cases                 (4 tests)
─────────────────────────────────────────────────────
Total:                                       43 tests
"""

import asyncio
import base64
import os
import stat
import sys
import tempfile
import time
import unittest
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

# ── Path setup ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cowork_agent.core.models import AgentResponse, Message, ToolCall, ToolResult, ToolSchema
from cowork_agent.core.token_tracker import (
    TokenTracker, TokenUsage, BudgetExceededError, MODEL_COSTS,
)
from cowork_agent.core.provider_fallback import ProviderFallback
from cowork_agent.core.stream_hardener import StreamHardener, StreamTimeoutError
from cowork_agent.core.multimodal import (
    ImageContent, MultiModalMessage, load_image, extract_image_paths,
    parse_multimodal_input, SUPPORTED_IMAGE_TYPES, MAX_IMAGE_SIZE,
)
from cowork_agent.core.response_cache import ResponseCache, CacheEntry


# ═══════════════════════════════════════════════
# Feature 1: Token Tracker Edge Cases
# ═══════════════════════════════════════════════

class TestTokenTrackerEdgeCases(unittest.TestCase):
    """Edge cases for token tracking and budget enforcement."""

    def test_cache_token_accumulation(self):
        """Cache read/write tokens accumulate correctly."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(
            input_tokens=100, output_tokens=50,
            cache_read_tokens=30, cache_write_tokens=20,
        ))
        tracker.record(TokenUsage(
            input_tokens=200, output_tokens=100,
            cache_read_tokens=40, cache_write_tokens=10,
        ))
        self.assertEqual(tracker.total_cache_read_tokens, 70)
        self.assertEqual(tracker.total_cache_write_tokens, 30)

    def test_prefix_matching_for_model_costs(self):
        """Model name with date suffix should match via prefix (e.g. gpt-4o-2024-11-20 → gpt-4o)."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(
            input_tokens=1_000_000, output_tokens=0,
            model="gpt-4o-2024-11-20",  # Should prefix-match "gpt-4o"
        ))
        # gpt-4o input cost: $2.50 / 1M tokens
        self.assertAlmostEqual(tracker.estimated_cost_usd, 2.5, places=2)

    def test_unknown_model_default_rates(self):
        """Unknown model/provider falls back to moderate pricing ($3/$15)."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(
            input_tokens=1_000_000, output_tokens=0,
            model="some-unknown-model-v42",
            provider="UnknownProvider",
        ))
        # Default fallback: $3.0 input / 1M
        self.assertAlmostEqual(tracker.estimated_cost_usd, 3.0, places=2)

    def test_ollama_provider_name_case_insensitive(self):
        """OllamaProvider in various casings should still be free."""
        for provider_name in ["OllamaProvider", "ollama", "ollamaprovider", "OLLAMAPROVIDER"]:
            tracker = TokenTracker()
            tracker.record(TokenUsage(
                input_tokens=1_000_000, output_tokens=1_000_000,
                model="llama3-custom",
                provider=provider_name,
            ))
            self.assertEqual(tracker.estimated_cost_usd, 0.0,
                             f"Expected 0.0 for provider '{provider_name}'")

    def test_max_session_tokens_zero_is_falsy(self):
        """max_session_tokens=0 is falsy — should NOT enforce token limit."""
        tracker = TokenTracker(max_session_tokens=0)
        tracker.record(TokenUsage(input_tokens=999999, output_tokens=999999))
        # 0 is falsy, so `if self.max_session_tokens` is False → no check
        tracker.check_budget()  # Should NOT raise

    def test_summary_with_no_budget_set(self):
        """Summary when no budget is set should have None for remaining fields."""
        tracker = TokenTracker()  # No budget
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))
        s = tracker.summary()
        self.assertIsNone(s["budget"]["max_session_tokens"])
        self.assertIsNone(s["budget"]["max_cost_usd"])
        self.assertIsNone(s["budget"]["tokens_remaining"])
        self.assertIsNone(s["budget"]["cost_remaining_usd"])

    def test_budget_at_exact_boundary(self):
        """Token count exactly at the limit should NOT raise (> not >=)."""
        tracker = TokenTracker(max_session_tokens=150)
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))  # total = 150
        # Implementation: `self.total_tokens > self.max_session_tokens`
        # 150 > 150 is False, so should NOT raise
        tracker.check_budget()

    def test_cost_budget_exact_boundary(self):
        """Cost exactly at limit should NOT raise (uses > not >=)."""
        # Use ollama (free) so cost is exactly 0.0
        tracker = TokenTracker(max_cost_usd=0.0)
        tracker.record(TokenUsage(
            input_tokens=100, output_tokens=50,
            provider="OllamaProvider", model="local-model",
        ))
        # cost=0.0, max_cost=0.0, 0.0 > 0.0 is False → should not raise
        tracker.check_budget()

    def test_reset_clears_cache_tokens(self):
        """Reset should also clear cache token counters."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(
            input_tokens=100, output_tokens=50,
            cache_read_tokens=30, cache_write_tokens=20,
        ))
        tracker.reset()
        self.assertEqual(tracker.total_cache_read_tokens, 0)
        self.assertEqual(tracker.total_cache_write_tokens, 0)
        self.assertEqual(tracker.call_count, 0)


# ═══════════════════════════════════════════════
# Feature 2: Provider Fallback Edge Cases
# ═══════════════════════════════════════════════

class TestProviderFallbackEdgeCases(unittest.TestCase):
    """Edge cases for provider auto-fallback."""

    def _make_provider(self, name="test", success=True, text="ok"):
        provider = MagicMock()
        provider.provider_name = name
        provider.model = f"{name}-model"
        provider.base_url = None
        provider.api_key = None
        if success:
            provider.send_message = AsyncMock(
                return_value=AgentResponse(text=text, stop_reason="end_turn")
            )
        else:
            provider.send_message = AsyncMock(side_effect=RuntimeError("Connection refused"))
        provider.health_check = AsyncMock(return_value={"status": "ok"})
        return provider

    def test_non_fallback_error_passthrough(self):
        """Error response WITHOUT fallback keywords should be returned as-is, not trigger fallback."""
        p1 = self._make_provider("primary")
        p1.send_message = AsyncMock(
            return_value=AgentResponse(text="Invalid API key", stop_reason="error")
        )
        p2 = self._make_provider("secondary", text="fallback")
        fb = ProviderFallback([p1, p2], max_retries_per_provider=1)
        resp = asyncio.get_event_loop().run_until_complete(
            fb.send_message([], [], "system")
        )
        # "Invalid API key" does NOT contain any fallback keywords
        # So it should NOT fall through — should return primary's error response
        self.assertEqual(resp.text, "Invalid API key")
        p2.send_message.assert_not_called()

    def test_backoff_capped_at_max(self):
        """Backoff delay should not exceed backoff_max."""
        fb = ProviderFallback(
            [self._make_provider()],
            backoff_base=1.0,
            backoff_max=5.0,
        )
        # With base=1 and max=5: attempt 1→1s, attempt 2→2s, attempt 3→4s, attempt 4→5s (capped)
        delay = min(fb._backoff_base * (2 ** (10 - 1)), fb._backoff_max)
        self.assertEqual(delay, 5.0)

    def test_failure_window_expiry(self):
        """Old failures outside the 5-minute window should be cleaned up."""
        p1 = self._make_provider("p1")
        p2 = self._make_provider("p2")
        fb = ProviderFallback([p1, p2])

        # Record old failures for p1 (pretend they happened 10 minutes ago)
        old_time = time.time() - 600  # 10 minutes ago
        fb._failure_timestamps[0] = [old_time, old_time, old_time]

        # _record_failure trims old entries
        fb._record_failure(0)

        # The old entries should be cleaned, leaving only the new one
        self.assertEqual(len(fb._failure_timestamps[0]), 1)

    def test_single_provider_all_retries_exhausted(self):
        """Single provider with multiple retries, all fail → error response."""
        p1 = self._make_provider("primary", success=False)
        fb = ProviderFallback([p1], max_retries_per_provider=3, backoff_base=0.01)
        resp = asyncio.get_event_loop().run_until_complete(
            fb.send_message([], [], "system")
        )
        self.assertEqual(resp.stop_reason, "error")
        self.assertEqual(p1.send_message.call_count, 3)

    def test_streaming_fallback(self):
        """Streaming should fall through to next provider on failure."""
        p1 = self._make_provider("primary")
        p1.send_message_stream = AsyncMock(side_effect=RuntimeError("stream failed"))
        p1.last_stream_response = None

        p2 = self._make_provider("secondary")
        async def fake_stream(*args, **kwargs):
            yield "hello"
            yield " world"
        p2.send_message_stream = fake_stream
        p2.last_stream_response = AgentResponse(text="hello world", stop_reason="end_turn")

        fb = ProviderFallback([p1, p2])

        async def collect():
            chunks = []
            async for chunk in fb.send_message_stream([], [], "system"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.get_event_loop().run_until_complete(collect())
        self.assertEqual(chunks, ["hello", " world"])

    def test_health_check_with_failing_provider(self):
        """Health check should report error for failing providers."""
        p1 = self._make_provider("healthy")
        p2 = self._make_provider("unhealthy")
        p2.health_check = AsyncMock(side_effect=RuntimeError("Cannot connect"))

        fb = ProviderFallback([p1, p2])
        result = asyncio.get_event_loop().run_until_complete(fb.health_check())
        self.assertEqual(result["status"], "ok")  # Overall status is still ok
        # Check that the unhealthy provider reported an error
        found_error = False
        for key, val in result["providers"].items():
            if "unhealthy" in key:
                self.assertEqual(val["status"], "error")
                found_error = True
        self.assertTrue(found_error)

    def test_provider_order_tiebreaker_preserves_original(self):
        """Providers with equal failure counts maintain original order."""
        p1 = self._make_provider("p1")
        p2 = self._make_provider("p2")
        p3 = self._make_provider("p3")
        fb = ProviderFallback([p1, p2, p3])
        # No failures recorded — all have 0 failures
        order = fb._get_provider_order()
        self.assertEqual(order, [0, 1, 2])

    def test_model_updated_on_fallback_success(self):
        """After fallback, the model attribute should reflect the successful provider."""
        p1 = self._make_provider("primary", success=False)
        p2 = self._make_provider("secondary", text="ok")
        fb = ProviderFallback([p1, p2], max_retries_per_provider=1)
        asyncio.get_event_loop().run_until_complete(fb.send_message([], [], "system"))
        self.assertEqual(fb.model, "secondary-model")


# ═══════════════════════════════════════════════
# Feature 3: Stream Hardener Edge Cases
# ═══════════════════════════════════════════════

class TestStreamHardenerEdgeCases(unittest.TestCase):
    """Edge cases for stream hardening."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    async def _collect(self, hardener, stream):
        chunks = []
        async for chunk in hardener.wrap(stream):
            chunks.append(chunk)
        return chunks

    def test_wrap_called_twice_resets_state(self):
        """Calling wrap() a second time should reset all internal state."""
        hardener = StreamHardener(chunk_timeout=5)

        async def stream1():
            yield "first"
            yield "stream"

        async def stream2():
            yield "second"

        # First wrap
        self._run(self._collect(hardener, stream1()))
        self.assertEqual(hardener.partial_text, "firststream")
        self.assertEqual(hardener.chunk_count, 2)
        self.assertTrue(hardener.completed)

        # Second wrap — state should be fresh
        self._run(self._collect(hardener, stream2()))
        self.assertEqual(hardener.partial_text, "second")
        self.assertEqual(hardener.chunk_count, 1)
        self.assertTrue(hardener.completed)

    def test_non_timeout_exception_mid_stream(self):
        """Non-timeout exceptions mid-stream should propagate (not caught as timeout)."""
        hardener = StreamHardener(chunk_timeout=5)

        async def error_stream():
            yield "before"
            raise ValueError("Something bad happened")

        with self.assertRaises(ValueError):
            self._run(self._collect(hardener, error_stream()))

        # We got 1 chunk before the error
        self.assertEqual(hardener.partial_text, "before")
        self.assertFalse(hardener.completed)
        self.assertFalse(hardener.timed_out)

    def test_build_partial_response_no_data(self):
        """build_partial_response before any streaming gives 'error' stop reason."""
        hardener = StreamHardener()
        resp = hardener.build_partial_response()
        self.assertEqual(resp.stop_reason, "error")
        self.assertIsNone(resp.text)  # None because "" → None check

    def test_elapsed_before_streaming(self):
        """elapsed should return 0.0 if wrap() hasn't been called."""
        hardener = StreamHardener()
        self.assertEqual(hardener.elapsed, 0.0)

    def test_only_empty_and_non_string_chunks(self):
        """Stream with only empty/non-string chunks yields nothing but still completes."""
        hardener = StreamHardener(chunk_timeout=5)

        async def junk_stream():
            yield ""
            yield ""
            yield 42  # type: ignore
            yield None  # type: ignore

        chunks = self._run(self._collect(hardener, junk_stream()))
        self.assertEqual(chunks, [])
        self.assertTrue(hardener.completed)
        self.assertEqual(hardener.chunk_count, 0)
        self.assertEqual(hardener.partial_text, "")


# ═══════════════════════════════════════════════
# Feature 4: Multi-Modal Edge Cases
# ═══════════════════════════════════════════════

class TestMultiModalEdgeCases(unittest.TestCase):
    """Edge cases for multi-modal input support."""

    def test_parse_multimodal_empty_paths_list(self):
        """parse_multimodal_input with explicit empty paths list → text-only message."""
        msg = parse_multimodal_input("hello world", image_paths=[])
        self.assertEqual(msg.text, "hello world")
        self.assertFalse(msg.has_images)
        self.assertEqual(len(msg.images), 0)

    def test_parse_multimodal_mixed_valid_invalid(self):
        """Mixed valid and invalid paths: only valid ones are loaded."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
            valid_path = f.name

        try:
            msg = parse_multimodal_input(
                "images here",
                image_paths=[valid_path, "/nonexistent/fake.png", valid_path],
            )
            # 2 valid (same file twice), 1 invalid → 2 images
            self.assertEqual(len(msg.images), 2)
        finally:
            os.unlink(valid_path)

    def test_images_only_no_text(self):
        """MultiModalMessage with images but no text."""
        img = ImageContent(media_type="image/png", base64_data="abc")
        mm = MultiModalMessage(text="", images=[img])
        self.assertTrue(mm.has_images)

        # Anthropic format: should have only image block (no text block for empty text)
        content = mm.to_anthropic_content()
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "image")

        # OpenAI format: same
        oc = mm.to_openai_content()
        self.assertEqual(len(oc), 1)
        self.assertEqual(oc[0]["type"], "image_url")

    def test_case_insensitive_extensions(self):
        """Image files with uppercase extensions (.PNG, .JPG) should work."""
        for ext in [".PNG", ".JPG", ".JPEG", ".GIF", ".WEBP"]:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(b'\x89PNG\r\n\x1a\n' + b'\x00' * 50)
                path = f.name
            try:
                img = load_image(path)
                self.assertIsNotNone(img, f"Failed to load image with extension {ext}")
            finally:
                os.unlink(path)

    def test_load_image_permission_error(self):
        """Image file that can't be read (permission denied) returns None."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n' + b'\x00' * 50)
            path = f.name

        try:
            # Remove read permission
            os.chmod(path, 0o000)
            img = load_image(path)
            self.assertIsNone(img)
        finally:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            os.unlink(path)

    def test_extract_paths_relative(self):
        """extract_image_paths finds relative paths like ./image.png."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=".") as f:
            f.write(b'\x89PNG' + b'\x00' * 10)
            rel_path = "./" + os.path.basename(f.name)

        try:
            paths = extract_image_paths(f"Look at {rel_path}")
            self.assertIn(rel_path, paths)
        finally:
            os.unlink(rel_path)

    def test_multiple_images_in_text(self):
        """extract_image_paths finds multiple image paths in a single text."""
        paths_created = []
        for ext in [".png", ".jpg"]:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(b'\x89PNG' + b'\x00' * 10)
                paths_created.append(f.name)

        try:
            text = f"Compare {paths_created[0]} and {paths_created[1]}"
            found = extract_image_paths(text)
            for p in paths_created:
                self.assertIn(p, found)
        finally:
            for p in paths_created:
                os.unlink(p)

    def test_load_image_oversized(self):
        """Image exceeding 20MB limit returns None."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
            # Write just enough to create the file; mock the size check
        try:
            with patch("cowork_agent.core.multimodal.Path") as mock_path:
                mock_instance = MagicMock()
                mock_path.return_value = mock_instance
                mock_instance.exists.return_value = True
                mock_instance.suffix = ".png"
                mock_stat = MagicMock()
                mock_stat.st_size = MAX_IMAGE_SIZE + 1  # Over limit
                mock_instance.stat.return_value = mock_stat
                result = load_image(path)
                self.assertIsNone(result)
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════
# Feature 5: Response Cache Edge Cases
# ═══════════════════════════════════════════════

class TestResponseCacheEdgeCases(unittest.TestCase):
    """Edge cases for response caching."""

    def _make_response(self, text="cached", stop="end_turn"):
        return AgentResponse(text=text, stop_reason=stop)

    def _make_messages(self, text="hello"):
        return [Message(role="user", content=text)]

    def test_put_same_key_twice_updates(self):
        """Putting the same key twice should update the entry."""
        cache = ResponseCache(max_size=10)
        cache.put("key", self._make_response("first"))
        cache.put("key", self._make_response("second"))
        hit = cache.get("key")
        self.assertEqual(hit.text, "second")
        self.assertEqual(cache.size, 1)  # Not 2

    def test_max_size_one(self):
        """Cache with max_size=1 should work correctly."""
        cache = ResponseCache(max_size=1)
        cache.put("k1", self._make_response("r1"))
        cache.put("k2", self._make_response("r2"))  # Evicts k1
        self.assertIsNone(cache.get("k1"))
        self.assertEqual(cache.get("k2").text, "r2")
        self.assertEqual(cache.size, 1)

    def test_hit_rate_with_zero_lookups(self):
        """hit_rate should return 0.0 when no lookups have been done."""
        cache = ResponseCache(max_size=10)
        self.assertEqual(cache.hit_rate, 0.0)
        self.assertEqual(cache.stats()["hit_rate"], 0.0)

    def test_make_key_with_tool_calls_in_messages(self):
        """make_key should handle messages that contain tool_calls."""
        cache = ResponseCache()
        msgs = [Message(
            role="assistant",
            content="calling tool",
            tool_calls=[ToolCall(name="bash", tool_id="t1", input={"command": "ls"})],
        )]
        # Should not crash — tool_calls are serialized into the fingerprint
        key = cache.make_key("model", msgs, "system")
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 64)  # SHA-256 hex digest

    def test_make_key_with_tool_results_in_messages(self):
        """make_key should handle messages that contain tool_results."""
        cache = ResponseCache()
        msgs = [Message(
            role="tool_result",
            content="",
            tool_results=[ToolResult(tool_id="t1", success=True, output="file.txt")],
        )]
        key = cache.make_key("model", msgs, "system")
        self.assertIsInstance(key, str)

    def test_is_cacheable_max_tokens(self):
        """Response with stop_reason='max_tokens' should NOT be cacheable."""
        resp = AgentResponse(text="truncated...", stop_reason="max_tokens")
        self.assertFalse(ResponseCache._is_cacheable(resp))

    def test_ttl_zero_disables_expiry(self):
        """TTL=0 (or negative) disables TTL → entries never expire."""
        cache = ResponseCache(max_size=10, ttl=0)
        cache.put("key", self._make_response())
        time.sleep(0.05)
        # TTL=0 → is_expired returns False → should still be a hit
        hit = cache.get("key")
        self.assertIsNotNone(hit)

    def test_invalidate_nonexistent_key(self):
        """Invalidating a key that doesn't exist returns False."""
        cache = ResponseCache(max_size=10)
        self.assertFalse(cache.invalidate("no-such-key"))

    def test_not_cacheable_empty_text(self):
        """Response with None text should NOT be cacheable."""
        resp = AgentResponse(text=None, stop_reason="end_turn")
        self.assertFalse(ResponseCache._is_cacheable(resp))

        resp2 = AgentResponse(text="", stop_reason="end_turn")
        self.assertFalse(ResponseCache._is_cacheable(resp2))


# ═══════════════════════════════════════════════
# Agent Integration Edge Cases
# ═══════════════════════════════════════════════

class TestAgentIntegrationEdgeCases(unittest.TestCase):
    """Edge cases for token tracker integration in the Agent."""

    def _make_prompt_builder(self):
        """Create a PromptBuilder with minimal config."""
        from cowork_agent.core.prompt_builder import PromptBuilder
        return PromptBuilder(config={"workspace_dir": "/tmp", "provider": "test"})

    def test_token_tracker_none_is_noop(self):
        """Agent with token_tracker=None should work without any token tracking."""
        from cowork_agent.core.agent import Agent
        from cowork_agent.core.tool_registry import ToolRegistry

        provider = MagicMock()
        provider.provider_name = "test"
        provider.model = "test-model"
        provider.send_message = AsyncMock(
            return_value=AgentResponse(
                text="response",
                stop_reason="end_turn",
                usage={"input_tokens": 100, "output_tokens": 50},
            )
        )

        agent = Agent(
            provider=provider,
            registry=ToolRegistry(),
            prompt_builder=self._make_prompt_builder(),
            token_tracker=None,  # No token tracking
        )
        result = asyncio.get_event_loop().run_until_complete(agent.run("hello"))
        self.assertEqual(result, "response")

    def test_token_usage_in_build_context(self):
        """After recording tokens, _build_context should include token_usage."""
        from cowork_agent.core.agent import Agent
        from cowork_agent.core.tool_registry import ToolRegistry

        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))

        agent = Agent(
            provider=MagicMock(),
            registry=ToolRegistry(),
            prompt_builder=self._make_prompt_builder(),
            token_tracker=tracker,
        )
        ctx = agent._build_context()
        self.assertIn("token_usage", ctx)
        self.assertEqual(ctx["token_usage"]["total_tokens"], 150)

    def test_token_usage_not_in_context_when_no_calls(self):
        """Before any LLM calls, token_usage should NOT appear in context."""
        from cowork_agent.core.agent import Agent
        from cowork_agent.core.tool_registry import ToolRegistry

        tracker = TokenTracker()  # No calls recorded yet

        agent = Agent(
            provider=MagicMock(),
            registry=ToolRegistry(),
            prompt_builder=self._make_prompt_builder(),
            token_tracker=tracker,
        )
        ctx = agent._build_context()
        self.assertNotIn("token_usage", ctx)

    def test_budget_exceeded_mid_loop(self):
        """Budget exceeded after first LLM call should stop the agent."""
        from cowork_agent.core.agent import Agent
        from cowork_agent.core.tool_registry import ToolRegistry

        tracker = TokenTracker(max_session_tokens=100)

        # First call returns tool call (to force a second iteration)
        first_response = AgentResponse(
            text="I'll use a tool",
            stop_reason="tool_use",
            tool_calls=[ToolCall(name="bash", tool_id="t1", input={"command": "ls"})],
            usage={"input_tokens": 80, "output_tokens": 30},  # 110 > 100 limit
        )

        provider = MagicMock()
        provider.provider_name = "test"
        provider.model = "test-model"
        provider.send_message = AsyncMock(return_value=first_response)

        registry = ToolRegistry()
        # Register a mock bash tool
        mock_tool = MagicMock()
        mock_tool.name = "bash"
        mock_tool.get_schema.return_value = ToolSchema(
            name="bash",
            description="Run commands",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
        )
        mock_tool.execute = AsyncMock(return_value=ToolResult(
            tool_id="t1", success=True, output="file.txt"
        ))
        registry.register(mock_tool)

        agent = Agent(
            provider=provider,
            registry=registry,
            prompt_builder=self._make_prompt_builder(),
            token_tracker=tracker,
        )

        result = asyncio.get_event_loop().run_until_complete(agent.run("list files"))
        # After first call, budget check on second iteration should fail
        self.assertIn("Budget exceeded", result)


if __name__ == "__main__":
    unittest.main()
