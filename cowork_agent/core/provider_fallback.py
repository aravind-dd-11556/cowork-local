"""
Provider Auto-Fallback — Wraps multiple providers with automatic failover.

If the primary provider fails (network error, rate limit, timeout),
transparently retries with the next provider in a ranked fallback chain.
Includes exponential backoff and health-check based provider ranking.

Sprint 4 (P2-Advanced) Feature 2.
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import AsyncIterator, Optional

from .models import AgentResponse, Message, ToolSchema
from .providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class ProviderFallback(BaseLLMProvider):
    """
    Meta-provider that wraps a chain of BaseLLMProvider instances
    and falls through to the next on failure.

    Usage:
        primary = AnthropicProvider(...)
        fallback1 = OpenAIProvider(...)
        fallback2 = OllamaProvider(...)

        provider = ProviderFallback([primary, fallback1, fallback2])
        agent = Agent(provider=provider, ...)
    """

    # Errors that trigger fallback (recoverable / transient)
    FALLBACK_ERROR_KEYWORDS = (
        "rate limit", "rate_limit", "429",
        "timeout", "timed out",
        "connect", "connection",
        "503", "502", "500",
        "overloaded", "capacity",
        "server error",
    )

    def __init__(
        self,
        providers: list[BaseLLMProvider],
        max_retries_per_provider: int = 1,
        backoff_base: float = 1.0,
        backoff_max: float = 10.0,
    ):
        if not providers:
            raise ValueError("At least one provider is required")

        # Use the first provider's settings as defaults
        primary = providers[0]
        super().__init__(
            model=primary.model,
            base_url=primary.base_url,
            api_key=primary.api_key,
        )

        self._providers = list(providers)
        self._max_retries = max_retries_per_provider
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max

        # Track provider health for smart ordering
        self._failure_timestamps: dict[int, list[float]] = {
            i: [] for i in range(len(providers))
        }
        # Window for considering failures (seconds)
        self._failure_window = 300  # 5 minutes

    @property
    def provider_name(self) -> str:
        names = [p.provider_name for p in self._providers]
        return f"FallbackChain({' → '.join(names)})"

    @property
    def active_provider(self) -> BaseLLMProvider:
        """Return the current primary provider (first healthy one)."""
        return self._providers[0]

    async def send_message(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system_prompt: str,
    ) -> AgentResponse:
        """Try each provider in order until one succeeds."""
        last_error = None
        ordered = self._get_provider_order()

        for idx in ordered:
            provider = self._providers[idx]
            for attempt in range(1, self._max_retries + 1):
                try:
                    logger.debug(
                        f"Trying provider {provider.provider_name} "
                        f"(attempt {attempt}/{self._max_retries})"
                    )
                    response = await provider.send_message(messages, tools, system_prompt)

                    # If the provider returned an error-type response, check if fallbackable
                    if response.stop_reason == "error" and self._is_fallback_error(response.text or ""):
                        raise RuntimeError(response.text)

                    # Success — update active model info
                    self.model = provider.model
                    return response

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Provider {provider.provider_name} failed "
                        f"(attempt {attempt}): {e}"
                    )
                    self._record_failure(idx)

                    if attempt < self._max_retries:
                        delay = min(
                            self._backoff_base * (2 ** (attempt - 1)),
                            self._backoff_max,
                        )
                        logger.info(f"Backing off {delay:.1f}s before retry")
                        await asyncio.sleep(delay)

            logger.info(
                f"Provider {provider.provider_name} exhausted retries, "
                f"falling back to next provider"
            )

        # All providers failed
        return AgentResponse(
            text=f"All providers failed. Last error: {last_error}",
            stop_reason="error",
        )

    async def send_message_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """Stream with fallback — try each provider's stream in order."""
        last_error = None
        ordered = self._get_provider_order()

        for idx in ordered:
            provider = self._providers[idx]
            try:
                logger.debug(f"Streaming from {provider.provider_name}")
                chunks_yielded = False

                async for chunk in provider.send_message_stream(messages, tools, system_prompt):
                    chunks_yielded = True
                    yield chunk

                # Transfer the stream response
                self._last_stream_response = provider.last_stream_response
                self.model = provider.model

                # If we got a response (even empty), consider it success
                if chunks_yielded or self._last_stream_response:
                    return

            except Exception as e:
                last_error = e
                logger.warning(f"Stream from {provider.provider_name} failed: {e}")
                self._record_failure(idx)

        # All failed
        self._last_stream_response = AgentResponse(
            text=f"All providers failed streaming. Last error: {last_error}",
            stop_reason="error",
        )
        yield self._last_stream_response.text

    async def health_check(self) -> dict:
        """Check all providers and report status."""
        results = {}
        for i, provider in enumerate(self._providers):
            try:
                results[f"provider_{i}_{provider.provider_name}"] = await provider.health_check()
            except Exception as e:
                results[f"provider_{i}_{provider.provider_name}"] = {
                    "status": "error", "error": str(e)
                }
        return {
            "status": "ok",
            "provider": self.provider_name,
            "providers": results,
        }

    def _is_fallback_error(self, error_text: str) -> bool:
        """Check if an error should trigger fallback."""
        lower = error_text.lower()
        return any(kw in lower for kw in self.FALLBACK_ERROR_KEYWORDS)

    def _record_failure(self, provider_idx: int) -> None:
        """Record a failure timestamp for a provider."""
        self._failure_timestamps[provider_idx].append(time.time())
        # Trim old failures outside window
        cutoff = time.time() - self._failure_window
        self._failure_timestamps[provider_idx] = [
            t for t in self._failure_timestamps[provider_idx] if t > cutoff
        ]

    def _get_provider_order(self) -> list[int]:
        """
        Return provider indices ordered by recent health.
        Providers with fewer recent failures come first.
        Preserves original order as tiebreaker.
        """
        cutoff = time.time() - self._failure_window
        scored = []
        for idx in range(len(self._providers)):
            recent_failures = sum(
                1 for t in self._failure_timestamps.get(idx, []) if t > cutoff
            )
            scored.append((recent_failures, idx))

        scored.sort(key=lambda x: (x[0], x[1]))
        return [idx for _, idx in scored]
