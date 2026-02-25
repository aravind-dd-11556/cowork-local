"""
Abstract base class for LLM providers.
All providers (Ollama, OpenAI, Anthropic) implement this interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from ..models import AgentResponse, Message, ToolSchema


class BaseLLMProvider(ABC):
    """Abstract LLM provider interface."""

    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.timeout = kwargs.get("timeout", 300)

    @abstractmethod
    async def send_message(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system_prompt: str,
    ) -> AgentResponse:
        """
        Send conversation to LLM and get response.

        Args:
            messages: Full conversation history
            tools: Available tool schemas
            system_prompt: System prompt text

        Returns:
            AgentResponse with either text or tool_calls
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict:
        """Check if the provider is available and configured."""
        pass

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__


class ProviderFactory:
    """Create LLM provider from config."""

    _providers: dict[str, type[BaseLLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[BaseLLMProvider]):
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, config: dict) -> BaseLLMProvider:
        """
        Create provider from config dict.

        Config structure:
            llm:
              provider: "ollama"
              model: "qwen3-vl:235b-instruct-cloud"
              temperature: 0.7
              max_tokens: 4096
            providers:
              ollama:
                base_url: "http://localhost:11434"
                timeout: 300
        """
        llm_config = config.get("llm", {})
        provider_name = llm_config.get("provider", "ollama")
        provider_config = config.get("providers", {}).get(provider_name, {})

        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_name]
        return provider_class(
            model=llm_config.get("model", "qwen3-vl:235b-instruct-cloud"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 4096),
            **provider_config,
        )
