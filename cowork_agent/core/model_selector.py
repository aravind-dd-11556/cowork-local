"""
Sprint 38: Interactive Model Selector — discover, list, select, and test models
from Ollama, OpenAI, Anthropic, and OpenRouter.

Usage:
    selector = ModelSelector(config)
    # List models from all configured providers
    models = await selector.list_all_models()
    # Interactive selection (CLI)
    choice = await selector.interactive_select()
    # Test a specific model
    result = await selector.test_model("anthropic", "claude-sonnet-4-5-20250929")
    # Apply selection to config
    selector.apply_selection(choice, config)
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .providers.base import BaseLLMProvider, ProviderFactory
from .models import AgentResponse, Message, ToolSchema

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Unified model information across all providers."""
    id: str
    name: str
    provider: str
    context_length: Optional[int] = None
    max_output: Optional[int] = None
    size_gb: Optional[float] = None
    parameter_size: str = ""
    quantization: str = ""
    tier: str = ""
    pricing_prompt: Optional[float] = None    # per 1M tokens
    pricing_completion: Optional[float] = None  # per 1M tokens
    modality: str = ""
    family: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Human-friendly display string."""
        parts = [self.name or self.id]
        if self.context_length:
            ctx_k = self.context_length // 1000
            parts.append(f"{ctx_k}K ctx")
        if self.parameter_size:
            parts.append(self.parameter_size)
        if self.size_gb:
            parts.append(f"{self.size_gb}GB")
        if self.tier and self.tier != "legacy":
            parts.append(f"[{self.tier}]")
        return " — ".join(parts)

    @property
    def pricing_summary(self) -> str:
        """Brief pricing string."""
        if self.pricing_prompt is not None and self.pricing_completion is not None:
            return f"${self.pricing_prompt:.2f}/${self.pricing_completion:.2f} per 1M tok"
        return ""


@dataclass
class ModelTestResult:
    """Result of testing a model connection."""
    provider: str
    model_id: str
    success: bool
    latency_ms: float = 0
    output_preview: str = ""
    error: str = ""
    tokens_used: Optional[dict] = None


class ModelSelector:
    """
    Discover, list, select, and test LLM models across providers.

    Works with the ProviderFactory to create temporary provider instances
    for listing and testing, without affecting the running agent.
    """

    # Well-known popular models per provider for quick selection
    POPULAR_MODELS = {
        "ollama": [
            "llama3.1:8b", "llama3.1:70b", "llama3.2:3b",
            "qwen3:32b", "qwen3:72b", "qwen3-vl:235b-instruct-cloud",
            "deepseek-r1:32b", "deepseek-r1:70b",
            "mistral:7b", "mixtral:8x7b",
            "codellama:34b", "phi3:14b",
            "gemma2:27b", "command-r:35b",
        ],
        "openai": [
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
            "o1", "o1-mini", "o3-mini",
        ],
        "anthropic": [
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
        ],
        "openrouter": [
            "anthropic/claude-sonnet-4", "openai/gpt-4o",
            "google/gemini-2.5-pro", "google/gemini-2.5-flash",
            "meta-llama/llama-3.1-405b-instruct",
            "deepseek/deepseek-r1", "deepseek/deepseek-chat-v3",
            "mistralai/mistral-large",
            "qwen/qwen-2.5-72b-instruct",
        ],
    }

    def __init__(self, config: dict):
        """
        Args:
            config: Raw config dict (same structure as default_config.yaml)
        """
        self._config = config
        self._providers_config = config.get("providers", {})
        self._llm_config = config.get("llm", {})

    def _create_provider(self, provider_name: str) -> Optional[BaseLLMProvider]:
        """Create a temporary provider instance for listing/testing."""
        prov_config = self._providers_config.get(provider_name, {})
        temp_config = {
            "llm": {
                "provider": provider_name,
                "model": self._llm_config.get("model", "test"),
                "temperature": self._llm_config.get("temperature", 0.7),
                "max_tokens": self._llm_config.get("max_tokens", 256),
            },
            "providers": {provider_name: prov_config},
        }
        try:
            return ProviderFactory.create(temp_config)
        except (ValueError, KeyError) as e:
            logger.warning(f"Cannot create provider {provider_name}: {e}")
            return None

    async def list_models(self, provider_name: str, filter_text: str = "") -> list[ModelInfo]:
        """
        List available models from a specific provider.

        Args:
            provider_name: One of 'ollama', 'openai', 'anthropic', 'openrouter'
            filter_text: Optional text to filter model names

        Returns:
            List of ModelInfo objects
        """
        provider = self._create_provider(provider_name)
        if not provider:
            return []

        try:
            if filter_text and hasattr(provider.list_models, '__code__') and 'filter_text' in provider.list_models.__code__.co_varnames:
                raw_models = await provider.list_models(filter_text=filter_text)
            else:
                raw_models = await provider.list_models()
        except Exception as e:
            logger.warning(f"Error listing models from {provider_name}: {e}")
            return []

        models = []
        for m in raw_models:
            mid = m.get("id", "")
            if filter_text and filter_text.lower() not in mid.lower() and filter_text.lower() not in m.get("name", "").lower():
                continue
            models.append(ModelInfo(
                id=mid,
                name=m.get("name", mid),
                provider=provider_name,
                context_length=m.get("context_length"),
                max_output=m.get("max_output"),
                size_gb=m.get("size_gb"),
                parameter_size=m.get("parameter_size", ""),
                quantization=m.get("quantization", ""),
                tier=m.get("tier", ""),
                pricing_prompt=m.get("pricing_prompt_per_1m"),
                pricing_completion=m.get("pricing_completion_per_1m"),
                modality=m.get("modality", ""),
                family=m.get("family", ""),
                extra={k: v for k, v in m.items() if k not in (
                    "id", "name", "context_length", "provider", "max_output",
                    "size_gb", "parameter_size", "quantization", "tier",
                    "pricing_prompt_per_1m", "pricing_completion_per_1m",
                    "modality", "family",
                )},
            ))
        return models

    async def list_all_models(self, filter_text: str = "") -> dict[str, list[ModelInfo]]:
        """
        List models from ALL registered providers concurrently.

        Returns:
            Dict mapping provider name → list of ModelInfo
        """
        providers = list(ProviderFactory._providers.keys())
        tasks = {
            name: asyncio.create_task(self.list_models(name, filter_text))
            for name in providers
        }

        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.warning(f"Failed to list models from {name}: {e}")
                results[name] = []

        return results

    async def test_model(
        self,
        provider_name: str,
        model_id: str,
        test_prompt: str = "Say 'hello' in one word.",
    ) -> ModelTestResult:
        """
        Test a model by sending a simple prompt and measuring latency.

        Args:
            provider_name: Provider to test
            model_id: Model identifier
            test_prompt: Simple prompt to verify the model works

        Returns:
            ModelTestResult with success status and latency
        """
        # Create a provider with the specific model
        prov_config = self._providers_config.get(provider_name, {})
        temp_config = {
            "llm": {
                "provider": provider_name,
                "model": model_id,
                "temperature": 0.1,
                "max_tokens": 50,
            },
            "providers": {provider_name: prov_config},
        }

        try:
            provider = ProviderFactory.create(temp_config)
        except Exception as e:
            return ModelTestResult(
                provider=provider_name,
                model_id=model_id,
                success=False,
                error=f"Cannot create provider: {e}",
            )

        test_messages = [Message(role="user", content=test_prompt)]

        start = time.monotonic()
        try:
            response = await provider.send_message(
                messages=test_messages,
                tools=[],
                system_prompt="You are a helpful assistant. Respond briefly.",
            )
            elapsed_ms = (time.monotonic() - start) * 1000

            if response.stop_reason == "error":
                return ModelTestResult(
                    provider=provider_name,
                    model_id=model_id,
                    success=False,
                    latency_ms=elapsed_ms,
                    error=response.text or "Unknown error",
                )

            return ModelTestResult(
                provider=provider_name,
                model_id=model_id,
                success=True,
                latency_ms=elapsed_ms,
                output_preview=(response.text or "")[:200],
                tokens_used=response.usage,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            return ModelTestResult(
                provider=provider_name,
                model_id=model_id,
                success=False,
                latency_ms=elapsed_ms,
                error=str(e),
            )

    async def get_provider_status(self) -> dict[str, dict]:
        """
        Check which providers are available/configured.

        Returns:
            Dict of provider_name → {available: bool, reason: str, model_count: int}
        """
        import os
        status = {}
        for name in ProviderFactory._providers:
            info = {"available": False, "reason": "", "model_count": 0}

            if name == "ollama":
                # Check if Ollama is running
                try:
                    import httpx
                    base = self._providers_config.get("ollama", {}).get("base_url", "http://localhost:11434")
                    async with httpx.AsyncClient(timeout=5) as client:
                        resp = await client.get(f"{base}/api/tags")
                        resp.raise_for_status()
                        models = resp.json().get("models", [])
                        info["available"] = True
                        info["model_count"] = len(models)
                        info["reason"] = f"{len(models)} local models"
                except Exception:
                    info["reason"] = "Ollama not running (start with: ollama serve)"

            elif name == "openai":
                key = self._providers_config.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
                if key:
                    info["available"] = True
                    info["reason"] = "API key configured"
                else:
                    info["reason"] = "No API key (set OPENAI_API_KEY)"

            elif name == "anthropic":
                key = self._providers_config.get("anthropic", {}).get("api_key") or os.getenv("ANTHROPIC_API_KEY")
                if key:
                    info["available"] = True
                    info["model_count"] = 8
                    info["reason"] = "API key configured"
                else:
                    info["reason"] = "No API key (set ANTHROPIC_API_KEY)"

            elif name == "openrouter":
                key = self._providers_config.get("openrouter", {}).get("api_key") or os.getenv("OPENROUTER_API_KEY")
                if key:
                    info["available"] = True
                    info["reason"] = "API key configured (200+ models)"
                else:
                    info["reason"] = "No API key (set OPENROUTER_API_KEY)"

            status[name] = info
        return status

    def get_current_model(self) -> tuple[str, str]:
        """Return (provider_name, model_id) currently configured."""
        return (
            self._llm_config.get("provider", "ollama"),
            self._llm_config.get("model", ""),
        )

    @staticmethod
    def apply_selection(provider_name: str, model_id: str, config) -> None:
        """
        Apply a model selection to a Config object.

        Args:
            provider_name: Selected provider
            model_id: Selected model ID
            config: Config object with .set() method
        """
        config.set("llm.provider", provider_name)
        config.set("llm.model", model_id)

    def format_model_table(
        self,
        models: list[ModelInfo],
        show_pricing: bool = False,
        numbered: bool = True,
    ) -> str:
        """
        Format a list of models as a displayable table string.

        Returns a formatted string ready for terminal output.
        """
        if not models:
            return "  No models available."

        lines = []
        for i, m in enumerate(models, 1):
            prefix = f"  {i:3d}. " if numbered else "  • "
            line = f"{prefix}{m.id}"

            details = []
            if m.context_length:
                details.append(f"{m.context_length // 1000}K ctx")
            if m.parameter_size:
                details.append(m.parameter_size)
            if m.size_gb:
                details.append(f"{m.size_gb}GB")
            if m.tier:
                details.append(m.tier)
            if show_pricing and m.pricing_summary:
                details.append(m.pricing_summary)

            if details:
                line += f"  ({', '.join(details)})"
            lines.append(line)

        return "\n".join(lines)

    def format_provider_status(self, status: dict[str, dict]) -> str:
        """Format provider status for terminal display."""
        lines = []
        icons = {True: "✅", False: "❌"}
        for name, info in status.items():
            icon = icons[info["available"]]
            lines.append(f"  {icon} {name:12s} — {info['reason']}")
        return "\n".join(lines)
