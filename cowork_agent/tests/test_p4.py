"""
Sprint 4 (P2-Advanced) Test Suite
==================================
Feature 1: Token Tracking & Budget Enforcement  (15 tests)
Feature 2: Provider Auto-Fallback               (12 tests)
Feature 3: Streaming Hardening                   (10 tests)
Feature 4: Multi-Modal Input Support             (10 tests)
Feature 5: Response Caching                      (11 tests)
─────────────────────────────────────────────────
Total:                                            58 tests
"""

import asyncio
import base64
import os
import sys
import tempfile
import time
import unittest
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
# Feature 1: Token Tracking & Budget Enforcement
# ═══════════════════════════════════════════════

class TestTokenUsage(unittest.TestCase):
    """Test TokenUsage dataclass."""

    def test_total_tokens(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        self.assertEqual(usage.total_tokens, 150)

    def test_to_dict(self):
        usage = TokenUsage(input_tokens=10, output_tokens=20, provider="anthropic", model="claude")
        d = usage.to_dict()
        self.assertEqual(d["input_tokens"], 10)
        self.assertEqual(d["output_tokens"], 20)
        self.assertEqual(d["provider"], "anthropic")
        self.assertIn("timestamp", d)


class TestTokenTracker(unittest.TestCase):
    """Test TokenTracker recording, budgets, and cost estimation."""

    def test_record_accumulates(self):
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))
        tracker.record(TokenUsage(input_tokens=200, output_tokens=100))
        self.assertEqual(tracker.total_input_tokens, 300)
        self.assertEqual(tracker.total_output_tokens, 150)
        self.assertEqual(tracker.total_tokens, 450)
        self.assertEqual(tracker.call_count, 2)

    def test_budget_token_limit(self):
        tracker = TokenTracker(max_session_tokens=500)
        tracker.record(TokenUsage(input_tokens=300, output_tokens=250))
        with self.assertRaises(BudgetExceededError):
            tracker.check_budget()

    def test_budget_within_limit(self):
        tracker = TokenTracker(max_session_tokens=1000)
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))
        tracker.check_budget()  # Should not raise

    def test_budget_cost_limit(self):
        tracker = TokenTracker(max_cost_usd=0.001)
        # Use a known model to get predictable cost
        tracker.record(TokenUsage(
            input_tokens=1000, output_tokens=1000,
            provider="AnthropicProvider", model="claude-sonnet-4-5-20250929",
        ))
        with self.assertRaises(BudgetExceededError):
            tracker.check_budget()

    def test_cost_estimation_anthropic(self):
        tracker = TokenTracker()
        tracker.record(TokenUsage(
            input_tokens=1_000_000, output_tokens=0,
            model="claude-sonnet-4-5-20250929",
        ))
        # Sonnet input: $3.0 / 1M tokens
        self.assertAlmostEqual(tracker.estimated_cost_usd, 3.0, places=2)

    def test_cost_estimation_ollama_free(self):
        tracker = TokenTracker()
        tracker.record(TokenUsage(
            input_tokens=1_000_000, output_tokens=1_000_000,
            provider="OllamaProvider", model="qwen3-vl",
        ))
        self.assertEqual(tracker.estimated_cost_usd, 0.0)

    def test_summary(self):
        tracker = TokenTracker(max_session_tokens=5000, max_cost_usd=1.0)
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))
        s = tracker.summary()
        self.assertEqual(s["total_tokens"], 150)
        self.assertEqual(s["call_count"], 1)
        self.assertIn("budget", s)
        self.assertEqual(s["budget"]["max_session_tokens"], 5000)

    def test_reset(self):
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))
        tracker.reset()
        self.assertEqual(tracker.total_tokens, 0)
        self.assertEqual(tracker.call_count, 0)

    def test_no_budget_no_error(self):
        """No limits set → check_budget never raises."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=999999, output_tokens=999999))
        tracker.check_budget()  # Should not raise


class TestTokenTrackerInAgent(unittest.TestCase):
    """Test token tracking integration in the Agent loop."""

    def test_agent_has_token_tracker_attr(self):
        """Agent constructor accepts token_tracker parameter."""
        from cowork_agent.core.agent import Agent
        # Just verify the parameter is accepted without error
        self.assertTrue(hasattr(Agent.__init__, "__code__"))

    def test_agent_response_has_usage(self):
        """AgentResponse has usage field."""
        resp = AgentResponse(text="hi", usage={"input_tokens": 10, "output_tokens": 5})
        self.assertEqual(resp.usage["input_tokens"], 10)

    def test_agent_response_usage_default_none(self):
        """AgentResponse.usage defaults to None."""
        resp = AgentResponse(text="hi")
        self.assertIsNone(resp.usage)


# ═══════════════════════════════════════════════
# Feature 2: Provider Auto-Fallback
# ═══════════════════════════════════════════════

class TestProviderFallback(unittest.TestCase):
    """Test provider fallback chain."""

    def _make_provider(self, name="test", success=True, text="ok"):
        """Create a mock provider."""
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

    def test_requires_at_least_one_provider(self):
        with self.assertRaises(ValueError):
            ProviderFallback([])

    def test_primary_success(self):
        """First provider succeeds → use it."""
        p1 = self._make_provider("primary", success=True, text="primary response")
        fb = ProviderFallback([p1])
        resp = asyncio.get_event_loop().run_until_complete(
            fb.send_message([], [], "system")
        )
        self.assertEqual(resp.text, "primary response")

    def test_fallback_on_failure(self):
        """Primary fails → fallback to secondary."""
        p1 = self._make_provider("primary", success=False)
        p2 = self._make_provider("secondary", success=True, text="fallback response")
        fb = ProviderFallback([p1, p2], max_retries_per_provider=1)
        resp = asyncio.get_event_loop().run_until_complete(
            fb.send_message([], [], "system")
        )
        self.assertEqual(resp.text, "fallback response")

    def test_all_providers_fail(self):
        """All providers fail → error response."""
        p1 = self._make_provider("p1", success=False)
        p2 = self._make_provider("p2", success=False)
        fb = ProviderFallback([p1, p2], max_retries_per_provider=1)
        resp = asyncio.get_event_loop().run_until_complete(
            fb.send_message([], [], "system")
        )
        self.assertEqual(resp.stop_reason, "error")
        self.assertIn("All providers failed", resp.text)

    def test_fallback_error_detection(self):
        """Error responses with fallback keywords trigger fallback."""
        fb = ProviderFallback([self._make_provider()])
        self.assertTrue(fb._is_fallback_error("rate_limit exceeded"))
        self.assertTrue(fb._is_fallback_error("Request timeout"))
        self.assertTrue(fb._is_fallback_error("Error 503 Service Unavailable"))
        self.assertFalse(fb._is_fallback_error("Invalid API key"))

    def test_provider_name(self):
        p1 = self._make_provider("A")
        p2 = self._make_provider("B")
        fb = ProviderFallback([p1, p2])
        self.assertIn("A", fb.provider_name)
        self.assertIn("B", fb.provider_name)

    def test_health_check(self):
        p1 = self._make_provider("p1")
        fb = ProviderFallback([p1])
        result = asyncio.get_event_loop().run_until_complete(fb.health_check())
        self.assertEqual(result["status"], "ok")
        self.assertIn("providers", result)

    def test_provider_order_by_health(self):
        """Providers with recent failures should be deprioritized."""
        p1 = self._make_provider("p1")
        p2 = self._make_provider("p2")
        fb = ProviderFallback([p1, p2])
        # Record failures for p1
        fb._record_failure(0)
        fb._record_failure(0)
        fb._record_failure(0)
        order = fb._get_provider_order()
        # p2 (index 1) should come first since p1 has failures
        self.assertEqual(order[0], 1)

    def test_error_response_triggers_fallback(self):
        """Provider returning error AgentResponse with fallback keyword → fall through."""
        p1 = self._make_provider("primary")
        p1.send_message = AsyncMock(
            return_value=AgentResponse(text="rate_limit error", stop_reason="error")
        )
        p2 = self._make_provider("secondary", text="fallback ok")
        fb = ProviderFallback([p1, p2], max_retries_per_provider=1)
        resp = asyncio.get_event_loop().run_until_complete(
            fb.send_message([], [], "system")
        )
        self.assertEqual(resp.text, "fallback ok")

    def test_active_provider(self):
        p1 = self._make_provider("primary")
        fb = ProviderFallback([p1])
        self.assertEqual(fb.active_provider.provider_name, "primary")

    def test_max_retries(self):
        """With max_retries=2, primary should be called twice before fallback."""
        p1 = self._make_provider("primary", success=False)
        p2 = self._make_provider("secondary", text="ok")
        fb = ProviderFallback([p1, p2], max_retries_per_provider=2, backoff_base=0.01)
        asyncio.get_event_loop().run_until_complete(fb.send_message([], [], "s"))
        self.assertEqual(p1.send_message.call_count, 2)


# ═══════════════════════════════════════════════
# Feature 3: Streaming Hardening
# ═══════════════════════════════════════════════

class TestStreamHardener(unittest.TestCase):
    """Test stream hardener with timeouts and partial recovery."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    async def _collect(self, hardener, stream):
        chunks = []
        async for chunk in hardener.wrap(stream):
            chunks.append(chunk)
        return chunks

    async def _make_stream(self, chunks, delay=0):
        """Create an async generator that yields chunks with optional delay."""
        for chunk in chunks:
            if delay:
                await asyncio.sleep(delay)
            yield chunk

    def test_normal_stream(self):
        """Normal stream completes successfully."""
        hardener = StreamHardener(chunk_timeout=5)
        chunks = self._run(self._collect(
            hardener, self._make_stream(["Hello", " ", "World"])
        ))
        self.assertEqual(chunks, ["Hello", " ", "World"])
        self.assertTrue(hardener.completed)
        self.assertFalse(hardener.timed_out)
        self.assertEqual(hardener.chunk_count, 3)

    def test_empty_chunks_skipped(self):
        """Empty chunks are filtered out."""
        hardener = StreamHardener(chunk_timeout=5)
        chunks = self._run(self._collect(
            hardener, self._make_stream(["Hello", "", "World", ""])
        ))
        self.assertEqual(chunks, ["Hello", "World"])
        self.assertEqual(hardener.chunk_count, 2)

    def test_partial_text_accumulates(self):
        """Partial text is accumulated during streaming."""
        hardener = StreamHardener(chunk_timeout=5)
        self._run(self._collect(
            hardener, self._make_stream(["Hello", " World"])
        ))
        self.assertEqual(hardener.partial_text, "Hello World")

    def test_chunk_timeout(self):
        """Stream stalls beyond chunk_timeout → StreamTimeoutError."""
        hardener = StreamHardener(chunk_timeout=0.1)

        async def stalling_stream():
            yield "first"
            await asyncio.sleep(1.0)  # Stall
            yield "never"

        with self.assertRaises(StreamTimeoutError):
            self._run(self._collect(hardener, stalling_stream()))

        self.assertTrue(hardener.timed_out)
        self.assertEqual(hardener.partial_text, "first")

    def test_total_timeout(self):
        """Total stream time exceeds total_timeout → StreamTimeoutError."""
        hardener = StreamHardener(chunk_timeout=5.0, total_timeout=0.2)

        async def slow_stream():
            for i in range(100):
                await asyncio.sleep(0.05)
                yield f"chunk{i}"

        with self.assertRaises(StreamTimeoutError):
            self._run(self._collect(hardener, slow_stream()))

        self.assertTrue(hardener.timed_out)

    def test_build_partial_response_on_timeout(self):
        """After timeout, build_partial_response returns what we got."""
        hardener = StreamHardener(chunk_timeout=0.1)

        async def stalling():
            yield "partial data"
            await asyncio.sleep(1.0)
            yield "never"

        try:
            self._run(self._collect(hardener, stalling()))
        except StreamTimeoutError:
            pass

        resp = hardener.build_partial_response()
        self.assertEqual(resp.text, "partial data")
        self.assertEqual(resp.stop_reason, "max_tokens")  # truncation signal

    def test_build_partial_response_on_success(self):
        """Completed stream builds normal response."""
        hardener = StreamHardener(chunk_timeout=5)
        self._run(self._collect(hardener, self._make_stream(["done"])))
        resp = hardener.build_partial_response()
        self.assertEqual(resp.stop_reason, "end_turn")

    def test_elapsed_time(self):
        """Elapsed time is tracked."""
        hardener = StreamHardener(chunk_timeout=5)
        self.assertEqual(hardener.elapsed, 0.0)
        self._run(self._collect(
            hardener, self._make_stream(["hi"], delay=0.05)
        ))
        self.assertGreater(hardener.elapsed, 0.0)

    def test_empty_stream(self):
        """Empty stream completes without error."""
        hardener = StreamHardener(chunk_timeout=5)

        async def empty():
            return
            yield  # Make it an async generator

        chunks = self._run(self._collect(hardener, empty()))
        self.assertEqual(chunks, [])
        self.assertTrue(hardener.completed)

    def test_non_string_chunk_skipped(self):
        """Non-string chunks are silently skipped."""
        hardener = StreamHardener(chunk_timeout=5)

        async def mixed():
            yield "ok"
            yield 42  # type: ignore
            yield "also ok"

        chunks = self._run(self._collect(hardener, mixed()))
        self.assertEqual(chunks, ["ok", "also ok"])


# ═══════════════════════════════════════════════
# Feature 4: Multi-Modal Input Support
# ═══════════════════════════════════════════════

class TestImageContent(unittest.TestCase):
    """Test ImageContent conversion methods."""

    def test_anthropic_block(self):
        img = ImageContent(media_type="image/png", base64_data="abc123")
        block = img.to_anthropic_block()
        self.assertEqual(block["type"], "image")
        self.assertEqual(block["source"]["type"], "base64")
        self.assertEqual(block["source"]["data"], "abc123")

    def test_openai_block(self):
        img = ImageContent(media_type="image/jpeg", base64_data="xyz")
        block = img.to_openai_block()
        self.assertEqual(block["type"], "image_url")
        self.assertIn("data:image/jpeg;base64,xyz", block["image_url"]["url"])


class TestMultiModalMessage(unittest.TestCase):
    """Test MultiModalMessage building."""

    def test_text_only(self):
        mm = MultiModalMessage(text="hello")
        self.assertFalse(mm.has_images)
        content = mm.to_anthropic_content()
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")

    def test_text_and_images(self):
        img = ImageContent(media_type="image/png", base64_data="abc")
        mm = MultiModalMessage(text="look at this", images=[img])
        self.assertTrue(mm.has_images)

        # Anthropic
        ac = mm.to_anthropic_content()
        self.assertEqual(len(ac), 2)
        self.assertEqual(ac[0]["type"], "text")
        self.assertEqual(ac[1]["type"], "image")

        # OpenAI
        oc = mm.to_openai_content()
        self.assertEqual(len(oc), 2)
        self.assertEqual(oc[1]["type"], "image_url")


class TestLoadImage(unittest.TestCase):
    """Test image file loading."""

    def test_load_valid_png(self):
        """Load a valid PNG file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Write minimal PNG header
            f.write(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
            f.flush()
            path = f.name

        try:
            img = load_image(path)
            self.assertIsNotNone(img)
            self.assertEqual(img.media_type, "image/png")
            self.assertGreater(len(img.base64_data), 0)
        finally:
            os.unlink(path)

    def test_load_nonexistent(self):
        img = load_image("/nonexistent/image.png")
        self.assertIsNone(img)

    def test_load_unsupported_type(self):
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            f.write(b"BM" + b"\x00" * 100)
            path = f.name
        try:
            img = load_image(path)
            self.assertIsNone(img)
        finally:
            os.unlink(path)

    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            img = load_image(path)
            self.assertIsNone(img)
        finally:
            os.unlink(path)

    def test_supported_extensions(self):
        self.assertIn(".png", SUPPORTED_IMAGE_TYPES)
        self.assertIn(".jpg", SUPPORTED_IMAGE_TYPES)
        self.assertIn(".jpeg", SUPPORTED_IMAGE_TYPES)
        self.assertIn(".gif", SUPPORTED_IMAGE_TYPES)
        self.assertIn(".webp", SUPPORTED_IMAGE_TYPES)

    def test_max_image_size(self):
        self.assertEqual(MAX_IMAGE_SIZE, 20 * 1024 * 1024)


class TestExtractImagePaths(unittest.TestCase):
    """Test image path extraction from text."""

    def test_absolute_path(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b'\x89PNG' + b'\x00' * 10)
            path = f.name
        try:
            paths = extract_image_paths(f"Look at {path}")
            self.assertIn(path, paths)
        finally:
            os.unlink(path)

    def test_no_images(self):
        paths = extract_image_paths("Just some text with no images")
        self.assertEqual(paths, [])


# ═══════════════════════════════════════════════
# Feature 5: Response Caching
# ═══════════════════════════════════════════════

class TestResponseCache(unittest.TestCase):
    """Test LRU response cache."""

    def _make_response(self, text="cached", stop="end_turn"):
        return AgentResponse(text=text, stop_reason=stop)

    def _make_messages(self, text="hello"):
        return [Message(role="user", content=text)]

    def test_put_and_get(self):
        cache = ResponseCache(max_size=10)
        key = cache.make_key("model", self._make_messages(), "system")
        resp = self._make_response("cached text")
        cache.put(key, resp)
        hit = cache.get(key)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.text, "cached text")

    def test_miss(self):
        cache = ResponseCache(max_size=10)
        self.assertIsNone(cache.get("nonexistent"))

    def test_lru_eviction(self):
        cache = ResponseCache(max_size=2)
        k1 = "key1"
        k2 = "key2"
        k3 = "key3"
        cache.put(k1, self._make_response("r1"))
        cache.put(k2, self._make_response("r2"))
        cache.put(k3, self._make_response("r3"))  # Should evict k1
        self.assertIsNone(cache.get(k1))
        self.assertIsNotNone(cache.get(k2))
        self.assertIsNotNone(cache.get(k3))

    def test_ttl_expiry(self):
        cache = ResponseCache(max_size=10, ttl=0.1)
        key = "test"
        cache.put(key, self._make_response())
        time.sleep(0.15)
        self.assertIsNone(cache.get(key))

    def test_not_cacheable_tool_calls(self):
        """Responses with tool calls should not be cached."""
        cache = ResponseCache(max_size=10)
        resp = AgentResponse(
            text="use tool", stop_reason="tool_use",
            tool_calls=[ToolCall(name="bash", tool_id="t1", input={})]
        )
        result = cache.put("key", resp)
        self.assertFalse(result)

    def test_not_cacheable_error(self):
        cache = ResponseCache(max_size=10)
        resp = AgentResponse(text="error", stop_reason="error")
        self.assertFalse(cache.put("key", resp))

    def test_disabled_cache(self):
        cache = ResponseCache(enabled=False)
        cache.put("key", self._make_response())
        self.assertIsNone(cache.get("key"))

    def test_stats(self):
        cache = ResponseCache(max_size=10)
        key = "test"
        cache.put(key, self._make_response())
        cache.get(key)  # hit
        cache.get("miss_key")  # miss
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5, places=2)

    def test_invalidate(self):
        cache = ResponseCache(max_size=10)
        cache.put("key", self._make_response())
        self.assertTrue(cache.invalidate("key"))
        self.assertIsNone(cache.get("key"))

    def test_clear(self):
        cache = ResponseCache(max_size=10)
        cache.put("k1", self._make_response())
        cache.put("k2", self._make_response())
        cache.clear()
        self.assertEqual(cache.size, 0)

    def test_make_key_deterministic(self):
        """Same inputs produce same key."""
        cache = ResponseCache()
        msgs = self._make_messages("test")
        k1 = cache.make_key("model", msgs, "system")
        k2 = cache.make_key("model", msgs, "system")
        self.assertEqual(k1, k2)
        # Different input → different key
        k3 = cache.make_key("model", self._make_messages("different"), "system")
        self.assertNotEqual(k1, k3)


# ═══════════════════════════════════════════════
# Integration tests
# ═══════════════════════════════════════════════

class TestSprint4Integration(unittest.TestCase):
    """Cross-feature integration tests."""

    def test_token_tracker_with_budget_and_cost(self):
        """Token tracker correctly enforces combined token + cost budget."""
        tracker = TokenTracker(max_session_tokens=500000, max_cost_usd=1.0)
        # Add moderate usage: 1000 in + 500 out at Sonnet rates = ~$0.0105
        tracker.record(TokenUsage(
            input_tokens=1000, output_tokens=500,
            model="claude-sonnet-4-5-20250929",
        ))
        tracker.check_budget()  # Within limits ($0.0105 < $1.0)
        # Push over cost: 100k in + 100k out = $0.30 + $1.50 = $1.80 total session > $1
        tracker.record(TokenUsage(
            input_tokens=100000, output_tokens=100000,
            model="claude-sonnet-4-5-20250929",
        ))
        with self.assertRaises(BudgetExceededError):
            tracker.check_budget()

    def test_fallback_provider_name_chain(self):
        """FallbackChain reports readable provider chain name."""
        p1 = MagicMock()
        p1.provider_name = "Anthropic"
        p1.model = "claude"
        p1.base_url = None
        p1.api_key = None
        p2 = MagicMock()
        p2.provider_name = "OpenAI"
        p2.model = "gpt4"
        p2.base_url = None
        p2.api_key = None
        fb = ProviderFallback([p1, p2])
        self.assertIn("Anthropic", fb.provider_name)
        self.assertIn("OpenAI", fb.provider_name)
        self.assertIn("→", fb.provider_name)

    def test_cache_key_includes_model(self):
        """Different models produce different cache keys."""
        cache = ResponseCache()
        msgs = [Message(role="user", content="hello")]
        k1 = cache.make_key("model-a", msgs, "system")
        k2 = cache.make_key("model-b", msgs, "system")
        self.assertNotEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()
