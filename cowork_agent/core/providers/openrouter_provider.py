"""
OpenRouter LLM Provider — access 200+ models through a unified OpenAI-compatible API.

OpenRouter (https://openrouter.ai) provides a single endpoint to access models from
OpenAI, Anthropic, Google, Meta, Mistral, and many more. Since the API is fully
OpenAI-compatible, this provider subclasses OpenAIProvider and only overrides
defaults (base URL, API key env var, attribution headers).

Usage::

    export OPENROUTER_API_KEY="sk-or-..."
    python -m cowork_agent --provider openrouter --model anthropic/claude-sonnet-4
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Optional

from .base import BaseLLMProvider, ProviderFactory
from .openai_provider import OpenAIProvider
from ..models import AgentResponse, Message, ToolSchema


# Default headers for OpenRouter attribution and ranking
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/cowork-agent",
    "X-Title": "Cowork Agent",
}


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider — OpenAI-compatible API to 200+ models.

    Inherits all functionality from OpenAIProvider (tool calling, streaming,
    message conversion, token tracking) and only changes:

    - Default ``base_url`` → ``https://openrouter.ai/api/v1``
    - API key from ``OPENROUTER_API_KEY`` env var
    - Adds ``HTTP-Referer`` and ``X-Title`` headers for OpenRouter attribution
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        extra_headers: Optional[dict] = None,
        **kwargs,
    ):
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")

        # Merge user-supplied headers with OpenRouter attribution headers
        self._extra_headers = {**_OPENROUTER_HEADERS, **(extra_headers or {})}

        # Pass to OpenAIProvider — it sets self.model, self.base_url, self._api_key
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=base_url,
            **kwargs,
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    # ── Override send_message to inject extra headers ───────────────

    async def send_message(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system_prompt: str,
    ) -> AgentResponse:
        """Send message via OpenRouter with attribution headers."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                default_headers=self._extra_headers,
            )

            import json
            openai_tools = self._convert_tools(tools) if tools else None
            openai_messages = [{"role": "system", "content": system_prompt}]
            openai_messages.extend(self._convert_messages(messages))

            response = await client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                tools=openai_tools,
                tool_choice="auto" if openai_tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            choice = response.choices[0]

            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "output_tokens": getattr(response.usage, "completion_tokens", 0),
                }

            from ..models import ToolCall
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        name=tc.function.name,
                        tool_id=tc.id,
                        input=json.loads(tc.function.arguments),
                    )
                    for tc in choice.message.tool_calls
                ]
                return AgentResponse(
                    text=choice.message.content,
                    tool_calls=tool_calls,
                    stop_reason="tool_use",
                    usage=usage,
                )

            stop_reason = "end_turn"
            if choice.finish_reason == "length":
                stop_reason = "max_tokens"

            return AgentResponse(
                text=choice.message.content or "",
                stop_reason=stop_reason,
                usage=usage,
            )

        except ImportError:
            return AgentResponse(
                text="OpenAI package not installed. Run: pip install openai",
                stop_reason="error",
            )
        except Exception as e:
            return AgentResponse(
                text=f"OpenRouter error: {str(e)}",
                stop_reason="error",
            )

    async def send_message_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """Stream response tokens via OpenRouter with attribution headers."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                default_headers=self._extra_headers,
            )

            import json
            openai_tools = self._convert_tools(tools) if tools else None
            openai_messages = [{"role": "system", "content": system_prompt}]
            openai_messages.extend(self._convert_messages(messages))

            tool_call_accum = {}
            text_parts = []

            stream = await client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                tools=openai_tools,
                tool_choice="auto" if openai_tools else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text_parts.append(delta.content)
                    yield delta.content
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_accum:
                            tool_call_accum[idx] = {"id": tc_delta.id or "", "name": "", "arguments": ""}
                        if tc_delta.id:
                            tool_call_accum[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_call_accum[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_call_accum[idx]["arguments"] += tc_delta.function.arguments

            from ..models import ToolCall
            tool_calls = []
            for idx in sorted(tool_call_accum.keys()):
                tc_data = tool_call_accum[idx]
                try:
                    tc_input = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                except json.JSONDecodeError:
                    tc_input = {}
                tool_calls.append(ToolCall(name=tc_data["name"], tool_id=tc_data["id"], input=tc_input))

            text = "".join(text_parts) if text_parts else None
            stop_reason = "tool_use" if tool_calls else "end_turn"
            self._last_stream_response = AgentResponse(text=text, tool_calls=tool_calls, stop_reason=stop_reason)

        except ImportError:
            self._last_stream_response = AgentResponse(
                text="OpenAI package not installed. Run: pip install openai",
                stop_reason="error",
            )
            yield self._last_stream_response.text
        except Exception as e:
            self._last_stream_response = AgentResponse(
                text=f"OpenRouter streaming error: {str(e)}",
                stop_reason="error",
            )
            yield self._last_stream_response.text

    async def health_check(self) -> dict:
        """Check if OpenRouter API is accessible."""
        if not self.api_key:
            return {"status": "error", "provider": "openrouter", "error": "No API key set (OPENROUTER_API_KEY)"}
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=self._extra_headers,
            )
            models = await client.models.list()
            return {
                "status": "ok",
                "provider": "openrouter",
                "model": self.model,
                "available_models": [m.id for m in models.data[:10]],
            }
        except Exception as e:
            return {"status": "error", "provider": "openrouter", "error": str(e)}


ProviderFactory.register("openrouter", OpenRouterProvider)
