"""
Tests for OpenRouter LLM Provider.

Covers: instantiation, env var fallback, custom headers, health check,
factory registration, mocked message sending, and mocked streaming.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.providers.base import BaseLLMProvider, ProviderFactory
from core.providers.openrouter_provider import OpenRouterProvider, _OPENROUTER_HEADERS
from core.providers.openai_provider import OpenAIProvider
from core.models import AgentResponse, Message, ToolCall, ToolSchema


def run_async(coro):
    """Helper to run async coroutines in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _ensure_openai_mock():
    """Ensure a mock 'openai' module exists in sys.modules so patch() can target it."""
    if "openai" not in sys.modules:
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock()
        sys.modules["openai"] = mock_openai
    return sys.modules["openai"]


# ───────────────────────────── Instantiation ──────────────────────────────


class TestOpenRouterInstantiation:
    """Test provider creation and default values."""

    def test_default_model(self):
        p = OpenRouterProvider(api_key="sk-or-test")
        assert p.model == "openai/gpt-4o"

    def test_default_base_url(self):
        p = OpenRouterProvider(api_key="sk-or-test")
        assert p.base_url == "https://openrouter.ai/api/v1"

    def test_custom_model(self):
        p = OpenRouterProvider(model="anthropic/claude-sonnet-4", api_key="sk-or-test")
        assert p.model == "anthropic/claude-sonnet-4"

    def test_custom_base_url(self):
        p = OpenRouterProvider(api_key="sk-or-test", base_url="https://custom.openrouter.ai/v1")
        assert p.base_url == "https://custom.openrouter.ai/v1"

    def test_is_subclass_of_openai_provider(self):
        assert issubclass(OpenRouterProvider, OpenAIProvider)

    def test_is_subclass_of_base_provider(self):
        assert issubclass(OpenRouterProvider, BaseLLMProvider)

    def test_provider_name(self):
        p = OpenRouterProvider(api_key="sk-or-test")
        assert p.provider_name == "openrouter"

    def test_temperature_passthrough(self):
        p = OpenRouterProvider(api_key="sk-or-test", temperature=0.3)
        assert p.temperature == 0.3

    def test_max_tokens_passthrough(self):
        p = OpenRouterProvider(api_key="sk-or-test", max_tokens=8192)
        assert p.max_tokens == 8192

    def test_timeout_passthrough(self):
        p = OpenRouterProvider(api_key="sk-or-test", timeout=60)
        assert p.timeout == 60


# ────────────────────────── API Key Resolution ────────────────────────────


class TestOpenRouterAPIKey:
    """Test API key resolution (explicit > env var)."""

    def test_explicit_api_key(self):
        p = OpenRouterProvider(api_key="sk-or-explicit")
        assert p.api_key == "sk-or-explicit"

    def test_env_var_fallback(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-from-env"}):
            p = OpenRouterProvider()
            assert p.api_key == "sk-or-from-env"

    def test_explicit_overrides_env(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-from-env"}):
            p = OpenRouterProvider(api_key="sk-or-explicit")
            assert p.api_key == "sk-or-explicit"

    def test_no_key_gives_none(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            p = OpenRouterProvider()
            assert p.api_key is None


# ───────────────────────── OpenRouter Headers ─────────────────────────────


class TestOpenRouterHeaders:
    """Test OpenRouter attribution headers."""

    def test_default_headers_present(self):
        p = OpenRouterProvider(api_key="sk-or-test")
        assert "HTTP-Referer" in p._extra_headers
        assert "X-Title" in p._extra_headers

    def test_default_header_values(self):
        p = OpenRouterProvider(api_key="sk-or-test")
        assert p._extra_headers["HTTP-Referer"] == "https://github.com/cowork-agent"
        assert p._extra_headers["X-Title"] == "Cowork Agent"

    def test_custom_headers_merged(self):
        p = OpenRouterProvider(api_key="sk-or-test", extra_headers={"X-Custom": "test"})
        assert p._extra_headers["X-Custom"] == "test"
        assert "HTTP-Referer" in p._extra_headers

    def test_custom_headers_can_override_defaults(self):
        p = OpenRouterProvider(
            api_key="sk-or-test",
            extra_headers={"X-Title": "My Custom App"},
        )
        assert p._extra_headers["X-Title"] == "My Custom App"

    def test_module_level_headers_constant(self):
        assert _OPENROUTER_HEADERS == {
            "HTTP-Referer": "https://github.com/cowork-agent",
            "X-Title": "Cowork Agent",
        }


# ─────────────────────── Factory Registration ────────────────────────────


class TestOpenRouterFactory:
    """Test ProviderFactory integration."""

    def test_registered_in_factory(self):
        assert "openrouter" in ProviderFactory._providers

    def test_factory_creates_openrouter(self):
        config = {
            "llm": {"provider": "openrouter", "model": "meta-llama/llama-3-70b"},
            "providers": {
                "openrouter": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key": "sk-or-factory-test",
                },
            },
        }
        provider = ProviderFactory.create(config)
        assert isinstance(provider, OpenRouterProvider)
        assert provider.model == "meta-llama/llama-3-70b"
        assert provider.api_key == "sk-or-factory-test"

    def test_factory_uses_default_base_url(self):
        config = {
            "llm": {"provider": "openrouter", "model": "openai/gpt-4o"},
            "providers": {"openrouter": {"api_key": "sk-or-test"}},
        }
        provider = ProviderFactory.create(config)
        assert provider.base_url == "https://openrouter.ai/api/v1"


# ──────────────────────── Health Check ────────────────────────────────────


class TestOpenRouterHealthCheck:
    """Test health_check method."""

    def test_health_check_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            p = OpenRouterProvider()
            result = run_async(p.health_check())
            assert result["status"] == "error"
            assert result["provider"] == "openrouter"
            assert "API key" in result["error"] or "No API key" in result["error"]

    def test_health_check_success(self):
        """Mock the models.list() call to test healthy response."""
        _ensure_openai_mock()
        p = OpenRouterProvider(api_key="sk-or-test")

        mock_model = SimpleNamespace(id="openai/gpt-4o")
        mock_models = SimpleNamespace(data=[mock_model])

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_models)

        mock_openai_mod = _ensure_openai_mock()
        mock_openai_mod.AsyncOpenAI = MagicMock(return_value=mock_client)

        result = run_async(p.health_check())
        assert result["status"] == "ok"
        assert result["provider"] == "openrouter"
        assert "openai/gpt-4o" in result["available_models"]

    def test_health_check_api_error(self):
        """Test health_check when API returns an error."""
        _ensure_openai_mock()
        p = OpenRouterProvider(api_key="sk-or-test")

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(side_effect=Exception("Connection refused"))

        mock_openai_mod = _ensure_openai_mock()
        mock_openai_mod.AsyncOpenAI = MagicMock(return_value=mock_client)

        result = run_async(p.health_check())
        assert result["status"] == "error"
        assert "Connection refused" in result["error"]


# ──────────────────────── Send Message (mocked) ──────────────────────────


class TestOpenRouterSendMessage:
    """Test send_message with mocked OpenAI client."""

    def _make_mock_response(self, content="Hello!", tool_calls=None, finish_reason="stop",
                            prompt_tokens=10, completion_tokens=5):
        """Build a mock chat completion response."""
        msg = SimpleNamespace(content=content, tool_calls=tool_calls)
        choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
        usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        return SimpleNamespace(choices=[choice], usage=usage)

    def _setup_mock_client(self, mock_response):
        """Set up mock OpenAI module and client, return the mock class."""
        mock_openai_mod = _ensure_openai_mock()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_mod.AsyncOpenAI = MagicMock(return_value=mock_client)
        return mock_openai_mod.AsyncOpenAI

    def test_text_response(self):
        p = OpenRouterProvider(api_key="sk-or-test")
        mock_resp = self._make_mock_response(content="Hi from OpenRouter!")
        self._setup_mock_client(mock_resp)

        result = run_async(p.send_message([], [], "system prompt"))
        assert isinstance(result, AgentResponse)
        assert result.text == "Hi from OpenRouter!"
        assert result.stop_reason == "end_turn"

    def test_tool_call_response(self):
        p = OpenRouterProvider(api_key="sk-or-test")

        tc = SimpleNamespace(
            id="call_123",
            function=SimpleNamespace(name="bash", arguments='{"command": "ls"}'),
        )
        mock_resp = self._make_mock_response(content=None, tool_calls=[tc])
        self._setup_mock_client(mock_resp)

        result = run_async(p.send_message([], [], "system prompt"))
        assert result.stop_reason == "tool_use"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[0].input == {"command": "ls"}

    def test_usage_tracking(self):
        p = OpenRouterProvider(api_key="sk-or-test")
        mock_resp = self._make_mock_response(prompt_tokens=42, completion_tokens=17)
        self._setup_mock_client(mock_resp)

        result = run_async(p.send_message([], [], "system prompt"))
        assert result.usage["input_tokens"] == 42
        assert result.usage["output_tokens"] == 17

    def test_max_tokens_stop_reason(self):
        p = OpenRouterProvider(api_key="sk-or-test")
        mock_resp = self._make_mock_response(finish_reason="length")
        self._setup_mock_client(mock_resp)

        result = run_async(p.send_message([], [], "system prompt"))
        assert result.stop_reason == "max_tokens"

    def test_headers_passed_to_client(self):
        p = OpenRouterProvider(api_key="sk-or-test")
        mock_resp = self._make_mock_response()
        mock_cls = self._setup_mock_client(mock_resp)

        run_async(p.send_message([], [], "system prompt"))
        # Verify AsyncOpenAI was called with default_headers
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["default_headers"]["HTTP-Referer"] == "https://github.com/cowork-agent"
        assert call_kwargs["default_headers"]["X-Title"] == "Cowork Agent"

    def test_error_handling(self):
        p = OpenRouterProvider(api_key="sk-or-test")
        mock_openai_mod = _ensure_openai_mock()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Rate limit exceeded")
        )
        mock_openai_mod.AsyncOpenAI = MagicMock(return_value=mock_client)

        result = run_async(p.send_message([], [], "system prompt"))
        assert result.stop_reason == "error"
        assert "Rate limit exceeded" in result.text

    def test_import_error_handling(self):
        p = OpenRouterProvider(api_key="sk-or-test")

        # Temporarily remove openai from sys.modules to trigger ImportError
        saved = sys.modules.pop("openai", None)
        try:
            with patch("builtins.__import__", side_effect=ImportError("No module named 'openai'")):
                result = run_async(p.send_message([], [], "system prompt"))
                assert result.stop_reason == "error"
                assert "openai" in result.text.lower() or "pip install" in result.text.lower()
        finally:
            if saved is not None:
                sys.modules["openai"] = saved

    def test_multiple_tool_calls(self):
        p = OpenRouterProvider(api_key="sk-or-test")

        tc1 = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="bash", arguments='{"command": "ls"}'),
        )
        tc2 = SimpleNamespace(
            id="call_2",
            function=SimpleNamespace(name="read", arguments='{"path": "/tmp/x"}'),
        )
        mock_resp = self._make_mock_response(content=None, tool_calls=[tc1, tc2])
        self._setup_mock_client(mock_resp)

        result = run_async(p.send_message([], [], "system prompt"))
        assert result.stop_reason == "tool_use"
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[1].name == "read"

    def test_empty_content_response(self):
        p = OpenRouterProvider(api_key="sk-or-test")
        mock_resp = self._make_mock_response(content=None, tool_calls=None)
        # Override: content=None and no tool calls → should get empty string
        msg = SimpleNamespace(content=None, tool_calls=None)
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=0)
        mock_resp = SimpleNamespace(choices=[choice], usage=usage)
        self._setup_mock_client(mock_resp)

        result = run_async(p.send_message([], [], "system prompt"))
        assert result.text == ""
        assert result.stop_reason == "end_turn"


# ───────────────────── Streaming (mocked) ─────────────────────────────────


class TestOpenRouterStreaming:
    """Test send_message_stream with mocked OpenAI client."""

    def test_text_streaming(self):
        _ensure_openai_mock()
        p = OpenRouterProvider(api_key="sk-or-test")

        chunks = []
        for text in ["Hello", " from", " OpenRouter"]:
            delta = SimpleNamespace(content=text, tool_calls=None)
            choice = SimpleNamespace(delta=delta)
            chunks.append(SimpleNamespace(choices=[choice]))

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        mock_openai_mod = _ensure_openai_mock()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_mod.AsyncOpenAI = MagicMock(return_value=mock_client)

        collected = []

        async def collect():
            async for token in p.send_message_stream([], [], "system prompt"):
                collected.append(token)

        run_async(collect())
        assert "".join(collected) == "Hello from OpenRouter"

    def test_stream_last_response(self):
        _ensure_openai_mock()
        p = OpenRouterProvider(api_key="sk-or-test")

        delta = SimpleNamespace(content="Done", tool_calls=None)
        choice = SimpleNamespace(delta=delta)
        chunk = SimpleNamespace(choices=[choice])

        async def mock_stream():
            yield chunk

        mock_openai_mod = _ensure_openai_mock()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_mod.AsyncOpenAI = MagicMock(return_value=mock_client)

        async def run():
            async for _ in p.send_message_stream([], [], "system prompt"):
                pass

        run_async(run())
        resp = p.last_stream_response
        assert resp is not None
        assert resp.text == "Done"
        assert resp.stop_reason == "end_turn"

    def test_stream_error_handling(self):
        _ensure_openai_mock()
        p = OpenRouterProvider(api_key="sk-or-test")

        mock_openai_mod = _ensure_openai_mock()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Stream broken")
        )
        mock_openai_mod.AsyncOpenAI = MagicMock(return_value=mock_client)

        collected = []

        async def collect():
            async for token in p.send_message_stream([], [], "system prompt"):
                collected.append(token)

        run_async(collect())
        assert any("Stream broken" in t or "streaming error" in t.lower() for t in collected)

    def test_stream_empty_chunks_skipped(self):
        """Chunks with no choices should be skipped."""
        _ensure_openai_mock()
        p = OpenRouterProvider(api_key="sk-or-test")

        empty_chunk = SimpleNamespace(choices=[])
        text_chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi", tool_calls=None))]
        )

        async def mock_stream():
            yield empty_chunk
            yield text_chunk
            yield empty_chunk

        mock_openai_mod = _ensure_openai_mock()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_mod.AsyncOpenAI = MagicMock(return_value=mock_client)

        collected = []

        async def collect():
            async for token in p.send_message_stream([], [], "system prompt"):
                collected.append(token)

        run_async(collect())
        assert "".join(collected) == "Hi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
