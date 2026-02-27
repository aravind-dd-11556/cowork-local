"""
OpenAI LLM Provider — uses native tool_use / function_calling API.
"""

from __future__ import annotations
import json
from typing import AsyncIterator, Optional

from .base import BaseLLMProvider, ProviderFactory
from ..models import AgentResponse, Message, ToolCall, ToolSchema


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider with native tool_use support."""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1", **kwargs):
        import os
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            **kwargs,
        )

    async def send_message(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system_prompt: str,
    ) -> AgentResponse:
        """Send message using OpenAI's chat completions API with tools."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

            # Convert tools to OpenAI format
            openai_tools = self._convert_tools(tools) if tools else None

            # Convert messages
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

            # Sprint 4: Extract token usage from response
            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "output_tokens": getattr(response.usage, "completion_tokens", 0),
                }

            # Check for tool calls — some models return finish_reason="stop"
            # even when tool_calls are present, so check the message directly
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

            # Map OpenAI stop reasons to our internal format
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
            return AgentResponse(text=f"OpenAI error: {str(e)}", stop_reason="error")

    async def send_message_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system_prompt: str,
    ) -> "AsyncIterator[str]":
        """Stream response tokens using OpenAI's streaming API."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

            openai_tools = self._convert_tools(tools) if tools else None
            openai_messages = [{"role": "system", "content": system_prompt}]
            openai_messages.extend(self._convert_messages(messages))

            # Accumulators for tool calls
            tool_call_accum = {}  # index -> {id, name, arguments}
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

                # Text content
                if delta.content:
                    text_parts.append(delta.content)
                    yield delta.content

                # Tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_accum:
                            tool_call_accum[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc_delta.id:
                            tool_call_accum[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_call_accum[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_call_accum[idx]["arguments"] += tc_delta.function.arguments

            # Build tool calls from accumulated data
            tool_calls = []
            for idx in sorted(tool_call_accum.keys()):
                tc_data = tool_call_accum[idx]
                try:
                    tc_input = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                except json.JSONDecodeError:
                    tc_input = {}
                tool_calls.append(ToolCall(
                    name=tc_data["name"],
                    tool_id=tc_data["id"],
                    input=tc_input,
                ))

            text = "".join(text_parts) if text_parts else None
            stop_reason = "tool_use" if tool_calls else "end_turn"

            self._last_stream_response = AgentResponse(
                text=text,
                tool_calls=tool_calls,
                stop_reason=stop_reason,
            )

        except ImportError:
            self._last_stream_response = AgentResponse(
                text="OpenAI package not installed. Run: pip install openai",
                stop_reason="error",
            )
            yield self._last_stream_response.text

        except Exception as e:
            self._last_stream_response = AgentResponse(
                text=f"OpenAI streaming error: {str(e)}",
                stop_reason="error",
            )
            yield self._last_stream_response.text

    def _convert_tools(self, tools: list[ToolSchema]) -> list[dict]:
        """Convert to OpenAI tools format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert internal messages to OpenAI format."""
        result = []
        for msg in messages:
            if msg.role == "tool_result" and msg.tool_results:
                for tr in msg.tool_results:
                    result.append({
                        "role": "tool",
                        "tool_call_id": tr.tool_id,
                        "content": tr.output or tr.error or "",
                    })
            elif msg.role == "assistant" and msg.tool_calls:
                result.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.tool_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.input),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            elif msg.role == "tool_result":
                # tool_result without tool_results data — skip empty messages
                # to avoid confusing the API with orphan user messages
                if msg.content:
                    result.append({"role": "user", "content": msg.content})
            else:
                result.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        return result

    async def health_check(self) -> dict:
        """Check if OpenAI API is accessible."""
        if not self.api_key:
            return {"status": "error", "provider": "openai", "error": "No API key set"}
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            models = await client.models.list()
            return {
                "status": "ok",
                "provider": "openai",
                "model": self.model,
                "available_models": [m.id for m in models.data[:10]],
            }
        except Exception as e:
            return {"status": "error", "provider": "openai", "error": str(e)}


ProviderFactory.register("openai", OpenAIProvider)
