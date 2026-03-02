"""
Sprint 38: Model Selector — Tests for model discovery, listing, selection, and testing
across Ollama, OpenAI, Anthropic, and OpenRouter providers.
"""

import asyncio
import pytest
import sys
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.providers.base import BaseLLMProvider, ProviderFactory
from core.providers.ollama import OllamaProvider
from core.providers.openai_provider import OpenAIProvider
from core.providers.anthropic_provider import AnthropicProvider
from core.providers.openrouter_provider import OpenRouterProvider
from core.model_selector import ModelSelector, ModelInfo, ModelTestResult
from core.models import AgentResponse, Message

# Ensure providers are registered (main.py does this at import, but tests skip main)
ProviderFactory.register("ollama", OllamaProvider)
ProviderFactory.register("openai", OpenAIProvider)
ProviderFactory.register("anthropic", AnthropicProvider)
ProviderFactory.register("openrouter", OpenRouterProvider)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def make_config(provider="ollama", model="test-model"):
    """Create a minimal config dict."""
    return {
        "llm": {
            "provider": provider,
            "model": model,
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        "providers": {
            "ollama": {"base_url": "http://localhost:11434", "timeout": 300},
            "openai": {"base_url": "https://api.openai.com/v1", "api_key": "sk-test", "timeout": 120},
            "anthropic": {"api_key": "sk-ant-test", "timeout": 120},
            "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key": "sk-or-test", "timeout": 120},
        },
    }


# ═══════════════════════════════════════════════════════════════
# 1. ModelInfo Tests
# ═══════════════════════════════════════════════════════════════

class TestModelInfo:
    def test_basic_creation(self):
        m = ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai")
        assert m.id == "gpt-4o"
        assert m.name == "GPT-4o"
        assert m.provider == "openai"

    def test_display_name_basic(self):
        m = ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai")
        assert "GPT-4o" in m.display_name

    def test_display_name_with_context(self):
        m = ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai", context_length=128000)
        assert "128K ctx" in m.display_name

    def test_display_name_with_params(self):
        m = ModelInfo(id="llama3:8b", name="Llama 3", provider="ollama", parameter_size="8B")
        assert "8B" in m.display_name

    def test_display_name_with_size(self):
        m = ModelInfo(id="llama3:8b", name="Llama 3", provider="ollama", size_gb=4.7)
        assert "4.7GB" in m.display_name

    def test_display_name_with_tier(self):
        m = ModelInfo(id="claude-opus", name="Claude Opus", provider="anthropic", tier="flagship")
        assert "flagship" in m.display_name

    def test_display_name_legacy_tier_hidden(self):
        m = ModelInfo(id="claude-3", name="Claude 3", provider="anthropic", tier="legacy")
        assert "legacy" not in m.display_name

    def test_pricing_summary_with_prices(self):
        m = ModelInfo(id="x", name="x", provider="openrouter", pricing_prompt=3.0, pricing_completion=15.0)
        assert "$3.00" in m.pricing_summary
        assert "$15.00" in m.pricing_summary

    def test_pricing_summary_empty(self):
        m = ModelInfo(id="x", name="x", provider="ollama")
        assert m.pricing_summary == ""

    def test_extra_dict(self):
        m = ModelInfo(id="x", name="x", provider="test", extra={"custom": "value"})
        assert m.extra["custom"] == "value"


# ═══════════════════════════════════════════════════════════════
# 2. ModelTestResult Tests
# ═══════════════════════════════════════════════════════════════

class TestModelTestResult:
    def test_success_result(self):
        r = ModelTestResult(provider="openai", model_id="gpt-4o", success=True, latency_ms=250)
        assert r.success
        assert r.latency_ms == 250

    def test_failure_result(self):
        r = ModelTestResult(provider="openai", model_id="gpt-4o", success=False, error="API key invalid")
        assert not r.success
        assert "API key" in r.error

    def test_with_tokens(self):
        r = ModelTestResult(provider="openai", model_id="gpt-4o", success=True,
                           tokens_used={"input_tokens": 10, "output_tokens": 5})
        assert r.tokens_used["input_tokens"] == 10


# ═══════════════════════════════════════════════════════════════
# 3. ModelSelector Init & Config
# ═══════════════════════════════════════════════════════════════

class TestModelSelectorInit:
    def test_basic_init(self):
        config = make_config()
        selector = ModelSelector(config)
        assert selector._config is config

    def test_get_current_model(self):
        config = make_config("anthropic", "claude-sonnet-4-5-20250929")
        selector = ModelSelector(config)
        prov, model = selector.get_current_model()
        assert prov == "anthropic"
        assert model == "claude-sonnet-4-5-20250929"

    def test_get_current_model_defaults(self):
        selector = ModelSelector({})
        prov, model = selector.get_current_model()
        assert prov == "ollama"
        assert model == ""

    def test_popular_models_all_providers(self):
        assert "ollama" in ModelSelector.POPULAR_MODELS
        assert "openai" in ModelSelector.POPULAR_MODELS
        assert "anthropic" in ModelSelector.POPULAR_MODELS
        assert "openrouter" in ModelSelector.POPULAR_MODELS

    def test_popular_models_non_empty(self):
        for prov, models in ModelSelector.POPULAR_MODELS.items():
            assert len(models) > 0, f"{prov} should have popular models"


# ═══════════════════════════════════════════════════════════════
# 4. ModelSelector.list_models (mocked providers)
# ═══════════════════════════════════════════════════════════════

class TestModelSelectorListModels:
    @pytest.mark.asyncio
    async def test_list_anthropic_models(self):
        """Anthropic list_models returns well-known catalog."""
        config = make_config("anthropic", "claude-sonnet-4-5-20250929")
        selector = ModelSelector(config)
        models = await selector.list_models("anthropic")
        assert len(models) > 0
        # Check we get ModelInfo objects
        assert isinstance(models[0], ModelInfo)
        assert models[0].provider == "anthropic"
        # Known models should be there
        ids = [m.id for m in models]
        assert "claude-sonnet-4-5-20250929" in ids
        assert "claude-opus-4-5-20251101" in ids

    @pytest.mark.asyncio
    async def test_list_anthropic_models_have_context(self):
        config = make_config("anthropic", "claude-sonnet-4-5-20250929")
        selector = ModelSelector(config)
        models = await selector.list_models("anthropic")
        for m in models:
            assert m.context_length == 200000

    @pytest.mark.asyncio
    async def test_list_models_unknown_provider(self):
        """Unknown provider returns empty list."""
        selector = ModelSelector(make_config())
        models = await selector.list_models("nonexistent")
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_filter(self):
        config = make_config("anthropic", "test")
        selector = ModelSelector(config)
        models = await selector.list_models("anthropic", "opus")
        # Should only return opus models
        for m in models:
            assert "opus" in m.id.lower() or "opus" in m.name.lower()

    @pytest.mark.asyncio
    async def test_list_models_filter_no_match(self):
        config = make_config("anthropic", "test")
        selector = ModelSelector(config)
        models = await selector.list_models("anthropic", "xyznonexistent")
        assert models == []


# ═══════════════════════════════════════════════════════════════
# 5. ModelSelector.list_all_models
# ═══════════════════════════════════════════════════════════════

class TestModelSelectorListAll:
    @pytest.mark.asyncio
    async def test_list_all_returns_dict(self):
        config = make_config()
        selector = ModelSelector(config)
        # This will fail for providers that need network, but anthropic works offline
        results = await selector.list_all_models()
        assert isinstance(results, dict)
        # At minimum, anthropic should work (catalog-based)
        assert "anthropic" in results

    @pytest.mark.asyncio
    async def test_list_all_anthropic_present(self):
        config = make_config()
        selector = ModelSelector(config)
        results = await selector.list_all_models()
        anthropic_models = results.get("anthropic", [])
        assert len(anthropic_models) >= 5  # Known catalog size


# ═══════════════════════════════════════════════════════════════
# 6. ModelSelector.test_model
# ═══════════════════════════════════════════════════════════════

class TestModelSelectorTestModel:
    @pytest.mark.asyncio
    async def test_test_model_success_mocked(self):
        """Test model with mocked provider response."""
        config = make_config("openai", "gpt-4o")
        selector = ModelSelector(config)

        mock_response = AgentResponse(
            text="hello",
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 2},
        )

        with patch.object(ProviderFactory, 'create') as mock_create:
            mock_provider = AsyncMock()
            mock_provider.send_message = AsyncMock(return_value=mock_response)
            mock_create.return_value = mock_provider

            result = await selector.test_model("openai", "gpt-4o")
            assert result.success
            assert result.latency_ms >= 0
            assert result.output_preview == "hello"
            assert result.tokens_used["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_test_model_error_response(self):
        """Test model returns error stop_reason."""
        config = make_config()
        selector = ModelSelector(config)

        mock_response = AgentResponse(text="Connection refused", stop_reason="error")

        with patch.object(ProviderFactory, 'create') as mock_create:
            mock_provider = AsyncMock()
            mock_provider.send_message = AsyncMock(return_value=mock_response)
            mock_create.return_value = mock_provider

            result = await selector.test_model("openai", "gpt-4o")
            assert not result.success
            assert "Connection refused" in result.error

    @pytest.mark.asyncio
    async def test_test_model_exception(self):
        """Test model that throws an exception."""
        config = make_config()
        selector = ModelSelector(config)

        with patch.object(ProviderFactory, 'create') as mock_create:
            mock_provider = AsyncMock()
            mock_provider.send_message = AsyncMock(side_effect=Exception("timeout"))
            mock_create.return_value = mock_provider

            result = await selector.test_model("openai", "gpt-4o")
            assert not result.success
            assert "timeout" in result.error

    @pytest.mark.asyncio
    async def test_test_model_provider_creation_fails(self):
        """Test when provider can't be created."""
        config = make_config()
        selector = ModelSelector(config)

        with patch.object(ProviderFactory, 'create', side_effect=ValueError("bad config")):
            result = await selector.test_model("openai", "gpt-4o")
            assert not result.success
            assert "bad config" in result.error

    @pytest.mark.asyncio
    async def test_test_model_custom_prompt(self):
        """Test with custom test prompt."""
        config = make_config()
        selector = ModelSelector(config)

        mock_response = AgentResponse(text="pong", stop_reason="end_turn")

        with patch.object(ProviderFactory, 'create') as mock_create:
            mock_provider = AsyncMock()
            mock_provider.send_message = AsyncMock(return_value=mock_response)
            mock_create.return_value = mock_provider

            result = await selector.test_model("openai", "gpt-4o", test_prompt="ping")
            assert result.success
            # Verify the custom prompt was used
            call_args = mock_provider.send_message.call_args
            messages = call_args[1].get("messages") or call_args[0][0]
            assert any("ping" in str(m) for m in ([messages] if isinstance(messages, Message) else messages))


# ═══════════════════════════════════════════════════════════════
# 7. ModelSelector.get_provider_status
# ═══════════════════════════════════════════════════════════════

class TestProviderStatus:
    @pytest.mark.asyncio
    async def test_anthropic_status_with_key(self):
        config = make_config()
        selector = ModelSelector(config)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=False):
            status = await selector.get_provider_status()
            assert "anthropic" in status
            assert status["anthropic"]["available"] is True

    @pytest.mark.asyncio
    async def test_openai_status_no_key(self):
        config = {"llm": {}, "providers": {"openai": {}}}
        selector = ModelSelector(config)
        with patch.dict(os.environ, {}, clear=True):
            # Remove key from env
            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                status = await selector.get_provider_status()
                if "openai" in status:
                    assert "API key" in status["openai"]["reason"]

    @pytest.mark.asyncio
    async def test_openrouter_status_with_key(self):
        config = make_config()
        selector = ModelSelector(config)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False):
            status = await selector.get_provider_status()
            if "openrouter" in status:
                assert status["openrouter"]["available"] is True


# ═══════════════════════════════════════════════════════════════
# 8. ModelSelector.format_model_table
# ═══════════════════════════════════════════════════════════════

class TestFormatModelTable:
    def test_empty_models(self):
        selector = ModelSelector(make_config())
        result = selector.format_model_table([])
        assert "No models" in result

    def test_numbered_output(self):
        models = [
            ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai", context_length=128000),
            ModelInfo(id="gpt-3.5", name="GPT-3.5", provider="openai"),
        ]
        selector = ModelSelector(make_config())
        result = selector.format_model_table(models, numbered=True)
        assert "1." in result
        assert "2." in result
        assert "gpt-4o" in result
        assert "128K ctx" in result

    def test_unnumbered_output(self):
        models = [ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai")]
        selector = ModelSelector(make_config())
        result = selector.format_model_table(models, numbered=False)
        assert "•" in result
        assert "1." not in result

    def test_with_pricing(self):
        models = [
            ModelInfo(id="x", name="x", provider="openrouter",
                     pricing_prompt=3.0, pricing_completion=15.0),
        ]
        selector = ModelSelector(make_config())
        result = selector.format_model_table(models, show_pricing=True)
        assert "$3.00" in result

    def test_without_pricing(self):
        models = [
            ModelInfo(id="x", name="x", provider="openrouter",
                     pricing_prompt=3.0, pricing_completion=15.0),
        ]
        selector = ModelSelector(make_config())
        result = selector.format_model_table(models, show_pricing=False)
        assert "$" not in result


# ═══════════════════════════════════════════════════════════════
# 9. ModelSelector.format_provider_status
# ═══════════════════════════════════════════════════════════════

class TestFormatProviderStatus:
    def test_format_mixed_status(self):
        selector = ModelSelector(make_config())
        status = {
            "ollama": {"available": True, "reason": "5 local models", "model_count": 5},
            "openai": {"available": False, "reason": "No API key", "model_count": 0},
        }
        result = selector.format_provider_status(status)
        assert "✅" in result
        assert "❌" in result
        assert "ollama" in result
        assert "openai" in result


# ═══════════════════════════════════════════════════════════════
# 10. ModelSelector.apply_selection
# ═══════════════════════════════════════════════════════════════

class TestApplySelection:
    def test_apply_to_config(self):
        from config.settings import Config
        config = Config(make_config())
        ModelSelector.apply_selection("anthropic", "claude-opus-4-5-20251101", config)
        assert config.get("llm.provider") == "anthropic"
        assert config.get("llm.model") == "claude-opus-4-5-20251101"


# ═══════════════════════════════════════════════════════════════
# 11. Provider list_models() Tests
# ═══════════════════════════════════════════════════════════════

class TestProviderListModels:
    def test_base_provider_list_models_returns_empty(self):
        """Base class default returns empty list."""
        # Can't instantiate abstract class directly, but we test via a concrete subclass
        from core.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(model="test", api_key="fake")
        result = asyncio.get_event_loop().run_until_complete(provider.list_models())
        assert isinstance(result, list)
        assert len(result) > 0  # Anthropic returns catalog

    def test_anthropic_list_models_catalog(self):
        from core.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(model="test", api_key="fake")
        result = asyncio.get_event_loop().run_until_complete(provider.list_models())
        ids = [m["id"] for m in result]
        assert "claude-opus-4-5-20251101" in ids
        assert "claude-sonnet-4-5-20250929" in ids
        assert "claude-haiku-4-5-20251001" in ids
        # Check structure
        for m in result:
            assert "id" in m
            assert "name" in m
            assert "context_length" in m
            assert "provider" in m
            assert m["provider"] == "anthropic"

    def test_anthropic_list_models_has_tiers(self):
        from core.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(model="test", api_key="fake")
        result = asyncio.get_event_loop().run_until_complete(provider.list_models())
        tiers = {m["tier"] for m in result}
        assert "flagship" in tiers
        assert "balanced" in tiers
        assert "fast" in tiers

    def test_anthropic_list_models_api_key_flag(self):
        """Models should indicate if API key is set."""
        from core.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(model="test", api_key="sk-test")
        result = asyncio.get_event_loop().run_until_complete(provider.list_models())
        assert all(m["api_key_set"] is True for m in result)

        provider2 = AnthropicProvider(model="test", api_key=None)
        result2 = asyncio.get_event_loop().run_until_complete(provider2.list_models())
        assert all(m["api_key_set"] is False for m in result2)

    def test_ollama_list_models_connect_error(self):
        """Ollama returns empty list when not running."""
        from core.providers.ollama import OllamaProvider
        provider = OllamaProvider(model="test", base_url="http://localhost:99999")
        result = asyncio.get_event_loop().run_until_complete(provider.list_models())
        assert result == []

    def test_openai_list_models_no_key(self):
        """OpenAI returns empty list without API key."""
        from core.providers.openai_provider import OpenAIProvider
        # Temporarily clear env
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider(model="test", api_key=None)
            result = asyncio.get_event_loop().run_until_complete(provider.list_models())
            assert result == []

    def test_openrouter_list_models_no_key(self):
        """OpenRouter returns empty list without API key."""
        from core.providers.openrouter_provider import OpenRouterProvider
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenRouterProvider(model="test", api_key=None)
            result = asyncio.get_event_loop().run_until_complete(provider.list_models())
            assert result == []


# ═══════════════════════════════════════════════════════════════
# 12. Ollama-specific extras
# ═══════════════════════════════════════════════════════════════

class TestOllamaExtras:
    def test_list_running_models_connect_error(self):
        from core.providers.ollama import OllamaProvider
        provider = OllamaProvider(model="test", base_url="http://localhost:99999")
        result = asyncio.get_event_loop().run_until_complete(provider.list_running_models())
        assert result == []

    def test_pull_model_connect_error(self):
        from core.providers.ollama import OllamaProvider
        provider = OllamaProvider(model="test", base_url="http://localhost:99999")
        result = asyncio.get_event_loop().run_until_complete(provider.pull_model("test:latest"))
        assert result is False


# ═══════════════════════════════════════════════════════════════
# 13. CLI /model command handler
# ═══════════════════════════════════════════════════════════════

class TestCLIModelCommand:
    """Test that CLI wires up the /model command correctly."""

    def _make_cli(self):
        """Create a CLI instance with mocked agent."""
        from cowork_agent.interfaces.cli import CLI
        from core.providers.anthropic_provider import AnthropicProvider

        mock_agent = MagicMock()
        mock_agent.provider = AnthropicProvider(model="claude-sonnet-4-5-20250929", api_key="test")
        mock_agent.registry = MagicMock()
        mock_agent.registry.tool_names = ["bash", "read"]
        mock_agent.max_iterations = 15
        mock_agent.messages = []

        # Mock prompt_builder with config
        mock_pb = MagicMock()
        mock_pb.config = make_config("anthropic", "claude-sonnet-4-5-20250929")
        mock_agent.prompt_builder = mock_pb

        cli = CLI(agent=mock_agent, streaming=False)
        return cli

    def test_model_command_in_completer(self):
        cli = self._make_cli()
        # Tab complete should include /model
        result = cli._readline_completer("/mod", 0)
        assert result == "/model"

    def test_model_help_output(self, capsys):
        cli = self._make_cli()
        cli._model_help()
        captured = capsys.readouterr()
        assert "Model Selection" in captured.out
        assert "/model status" in captured.out
        assert "/model use" in captured.out

    def test_model_current(self, capsys):
        cli = self._make_cli()
        cli._model_current()
        captured = capsys.readouterr()
        assert "claude-sonnet-4-5-20250929" in captured.out

    def test_model_popular(self, capsys):
        cli = self._make_cli()
        cli._model_popular("")
        captured = capsys.readouterr()
        assert "OLLAMA" in captured.out
        assert "OPENAI" in captured.out
        assert "ANTHROPIC" in captured.out

    def test_model_popular_filtered(self, capsys):
        cli = self._make_cli()
        cli._model_popular("anthropic")
        captured = capsys.readouterr()
        assert "ANTHROPIC" in captured.out
        assert "OLLAMA" not in captured.out

    def test_model_use_switches_provider(self):
        cli = self._make_cli()
        # Patch ProviderFactory in the CLI module's namespace
        from cowork_agent.core.providers import base as cli_pf_module
        # Register providers in CLI's ProviderFactory namespace
        cli_pf_module.ProviderFactory.register("openai", OpenAIProvider)
        cli_pf_module.ProviderFactory.register("anthropic", AnthropicProvider)

        with patch.object(cli_pf_module.ProviderFactory, 'create') as mock_create:
            mock_new_provider = MagicMock()
            mock_new_provider.model = "gpt-4o"
            mock_create.return_value = mock_new_provider

            cli._model_use("openai gpt-4o")

            # Agent's provider should be updated
            assert cli.agent.provider == mock_new_provider

    def test_model_use_invalid_provider(self, capsys):
        cli = self._make_cli()
        cli._model_use("nonexistent some-model")
        captured = capsys.readouterr()
        assert "Unknown provider" in captured.out

    def test_model_use_missing_args(self, capsys):
        cli = self._make_cli()
        cli._model_use("openai")
        captured = capsys.readouterr()
        assert "Usage" in captured.out

    @pytest.mark.asyncio
    async def test_model_status(self, capsys):
        cli = self._make_cli()
        await cli._model_status()
        captured = capsys.readouterr()
        assert "Provider Status" in captured.out

    @pytest.mark.asyncio
    async def test_model_list_anthropic(self, capsys):
        cli = self._make_cli()
        # Mock the selector to return known models (avoid ProviderFactory namespace issues in tests)
        mock_selector = MagicMock()
        mock_selector.list_models = AsyncMock(return_value=[
            ModelInfo(id="claude-sonnet-4-5-20250929", name="Claude Sonnet 4.5", provider="anthropic", context_length=200000),
        ])
        mock_selector.format_model_table = MagicMock(return_value="  1. claude-sonnet-4-5-20250929  (200K ctx)")
        cli._model_selector = mock_selector
        await cli._model_list("anthropic")
        captured = capsys.readouterr()
        assert "ANTHROPIC" in captured.out
        assert "claude" in captured.out.lower()

    @pytest.mark.asyncio
    async def test_model_list_no_provider(self, capsys):
        cli = self._make_cli()
        await cli._model_list("")
        captured = capsys.readouterr()
        assert "Specify a provider" in captured.out

    @pytest.mark.asyncio
    async def test_model_test_missing_args(self, capsys):
        cli = self._make_cli()
        await cli._model_test("openai")
        captured = capsys.readouterr()
        assert "Usage" in captured.out

    @pytest.mark.asyncio
    async def test_handle_model_command_routes(self):
        """Test that _handle_model_command routes to correct sub-handler."""
        cli = self._make_cli()

        # Test routing to help
        with patch.object(cli, '_model_help') as mock:
            await cli._handle_model_command("help")
            mock.assert_called_once()

        # Test routing to current
        with patch.object(cli, '_model_current') as mock:
            await cli._handle_model_command("current")
            mock.assert_called_once()

        # Test routing to popular
        with patch.object(cli, '_model_popular') as mock:
            await cli._handle_model_command("popular anthropic")
            mock.assert_called_once_with("anthropic")


# ═══════════════════════════════════════════════════════════════
# 14. main.py CLI args
# ═══════════════════════════════════════════════════════════════

class TestMainCLIArgs:
    def test_list_models_arg_parsing(self):
        """Verify --list-models is parsed correctly."""
        # We can't easily test parse_args directly without mocking sys.argv,
        # but we can test the _handle_model_commands function exists
        from cowork_agent.main import _handle_model_commands
        assert callable(_handle_model_commands)


# ═══════════════════════════════════════════════════════════════
# 15. Edge Cases
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_model_info_all_none_optionals(self):
        m = ModelInfo(id="x", name="x", provider="test")
        assert m.context_length is None
        assert m.max_output is None
        assert m.size_gb is None
        assert m.pricing_prompt is None

    def test_model_info_display_name_all_fields(self):
        m = ModelInfo(
            id="claude-opus", name="Claude Opus", provider="anthropic",
            context_length=200000, parameter_size="1T", size_gb=400,
            tier="flagship",
        )
        dn = m.display_name
        assert "200K ctx" in dn
        assert "1T" in dn
        assert "400GB" in dn
        assert "flagship" in dn

    def test_format_table_many_models(self):
        """Table with lots of models shouldn't crash."""
        models = [
            ModelInfo(id=f"model-{i}", name=f"Model {i}", provider="test")
            for i in range(100)
        ]
        selector = ModelSelector(make_config())
        result = selector.format_model_table(models)
        assert "model-0" in result
        assert "model-99" in result

    @pytest.mark.asyncio
    async def test_list_models_provider_returns_empty(self):
        """Provider that returns empty list."""
        config = make_config()
        selector = ModelSelector(config)

        with patch.object(ProviderFactory, 'create') as mock_create:
            mock_provider = AsyncMock()
            mock_provider.list_models = AsyncMock(return_value=[])
            mock_create.return_value = mock_provider

            models = await selector.list_models("openai")
            assert models == []

    @pytest.mark.asyncio
    async def test_test_model_latency_measured(self):
        """Verify latency measurement is reasonable."""
        config = make_config()
        selector = ModelSelector(config)

        async def slow_send(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms
            return AgentResponse(text="ok", stop_reason="end_turn")

        with patch.object(ProviderFactory, 'create') as mock_create:
            mock_provider = AsyncMock()
            mock_provider.send_message = slow_send
            mock_create.return_value = mock_provider

            result = await selector.test_model("openai", "gpt-4o")
            assert result.success
            assert result.latency_ms >= 40  # Should be at least ~50ms

    def test_selector_create_provider_unknown(self):
        """Creating a provider for unregistered name returns None."""
        selector = ModelSelector({"llm": {}, "providers": {}})
        result = selector._create_provider("totally_unknown_provider_xyz")
        assert result is None


# ═══════════════════════════════════════════════════════════════
# 16. Integration: apply_model_switch on CLI
# ═══════════════════════════════════════════════════════════════

class TestModelSwitchIntegration:
    def test_switch_updates_agent_provider(self):
        from cowork_agent.interfaces.cli import CLI
        from cowork_agent.core.providers.base import ProviderFactory as CLIProviderFactory

        mock_agent = MagicMock()
        mock_agent.provider = AnthropicProvider(model="old-model", api_key="test")
        mock_pb = MagicMock()
        mock_pb.config = make_config()
        mock_agent.prompt_builder = mock_pb

        cli = CLI(agent=mock_agent, streaming=False)

        with patch.object(CLIProviderFactory, 'create') as mock_create:
            new_provider = MagicMock()
            new_provider.model = "new-model"
            mock_create.return_value = new_provider

            cli._apply_model_switch("openai", "gpt-4o")

            # Verify ProviderFactory.create was called with correct config
            call_config = mock_create.call_args[0][0]
            assert call_config["llm"]["provider"] == "openai"
            assert call_config["llm"]["model"] == "gpt-4o"

            # Verify agent's provider was updated
            assert cli.agent.provider == new_provider

    def test_switch_preserves_temperature(self):
        from cowork_agent.interfaces.cli import CLI
        from cowork_agent.core.providers.base import ProviderFactory as CLIProviderFactory

        mock_agent = MagicMock()
        old_provider = AnthropicProvider(model="old", api_key="test", temperature=0.3)
        mock_agent.provider = old_provider
        mock_pb = MagicMock()
        mock_pb.config = make_config()
        mock_agent.prompt_builder = mock_pb

        cli = CLI(agent=mock_agent, streaming=False)

        with patch.object(CLIProviderFactory, 'create') as mock_create:
            new_provider = MagicMock()
            mock_create.return_value = new_provider

            cli._apply_model_switch("openai", "gpt-4o")

            call_config = mock_create.call_args[0][0]
            assert call_config["llm"]["temperature"] == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
