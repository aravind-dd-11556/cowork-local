"""
Anthropic LLM Provider — uses native tool_use API.
"""

from __future__ import annotations
import json
from typing import AsyncIterator, Optional

from .base import BaseLLMProvider, ProviderFactory
from ..models import AgentResponse, Message, ToolCall, ToolSchema


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider with native tool_use support."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929",
                 api_key: Optional[str] = None, **kwargs):
        import os
        super().__init__(
            model=model,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            **kwargs,
        )

    async def send_message(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system_prompt: str,
    ) -> AgentResponse:
        """Send message using Anthropic's Messages API with tools."""
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=self.api_key)

            # Convert tools to Anthropic format
            anthropic_tools = self._convert_tools(tools) if tools else []

            # Convert messages
            anthropic_messages = self._convert_messages(messages)

            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=anthropic_messages,
                tools=anthropic_tools if anthropic_tools else None,
                temperature=self.temperature,
            )

            # Parse response content blocks
            text_parts = []
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        name=block.name,
                        tool_id=block.id,
                        input=block.input,
                    ))

            text = "\n".join(text_parts) if text_parts else None

            # Sprint 4: Extract token usage from response
            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "input_tokens": getattr(response.usage, "input_tokens", 0),
                    "output_tokens": getattr(response.usage, "output_tokens", 0),
                    "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
                    "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
                }

            if tool_calls:
                return AgentResponse(
                    text=text,
                    tool_calls=tool_calls,
                    stop_reason="tool_use",
                    usage=usage,
                )

            return AgentResponse(text=text or "", stop_reason="end_turn", usage=usage)

        except ImportError:
            return AgentResponse(
                text="Anthropic package not installed. Run: pip install anthropic",
                stop_reason="error",
            )
        except Exception as e:
            return AgentResponse(text=f"Anthropic error: {str(e)}", stop_reason="error")

    async def send_message_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system_prompt: str,
    ) -> "AsyncIterator[str]":
        """Stream response tokens using Anthropic's native streaming API."""
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=self.api_key)

            anthropic_tools = self._convert_tools(tools) if tools else []
            anthropic_messages = self._convert_messages(messages)

            # Accumulate tool calls as we stream
            tool_calls = []
            text_parts = []
            current_tool_block = None  # Track the tool block being built
            current_tool_json = ""     # Accumulate JSON input string

            async with client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=anthropic_messages,
                tools=anthropic_tools if anthropic_tools else None,
                temperature=self.temperature,
            ) as stream:
                async for event in stream:
                    # Text delta — yield to caller
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            text_parts.append(event.delta.text)
                            yield event.delta.text
                        elif hasattr(event.delta, "partial_json"):
                            current_tool_json += event.delta.partial_json

                    # New content block starting
                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool_block = event.content_block
                            current_tool_json = ""

                    # Content block finished
                    elif event.type == "content_block_stop":
                        if current_tool_block is not None:
                            try:
                                tool_input = json.loads(current_tool_json) if current_tool_json else {}
                            except json.JSONDecodeError:
                                tool_input = {}
                            tool_calls.append(ToolCall(
                                name=current_tool_block.name,
                                tool_id=current_tool_block.id,
                                input=tool_input,
                            ))
                            current_tool_block = None
                            current_tool_json = ""

            # Build final response
            text = "".join(text_parts) if text_parts else None
            stop_reason = "tool_use" if tool_calls else "end_turn"

            self._last_stream_response = AgentResponse(
                text=text,
                tool_calls=tool_calls,
                stop_reason=stop_reason,
            )

        except ImportError:
            self._last_stream_response = AgentResponse(
                text="Anthropic package not installed. Run: pip install anthropic",
                stop_reason="error",
            )
            yield self._last_stream_response.text

        except Exception as e:
            self._last_stream_response = AgentResponse(
                text=f"Anthropic streaming error: {str(e)}",
                stop_reason="error",
            )
            yield self._last_stream_response.text


    def _convert_tools(self, tools: list[ToolSchema]) -> list[dict]:
        """Convert to Anthropic tools format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert internal messages to Anthropic format."""
        result = []
        for msg in messages:
            if msg.role == "tool_result" and msg.tool_results:
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.tool_id,
                            "content": tr.output or tr.error or "",
                            "is_error": not tr.success,
                        }
                        for tr in msg.tool_results
                    ],
                })
            elif msg.role == "assistant" and msg.tool_calls:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.tool_id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                result.append({"role": "assistant", "content": content})
            else:
                role = "user" if msg.role in ("user", "tool_result") else "assistant"
                result.append({"role": role, "content": msg.content})
        return result

    async def health_check(self) -> dict:
        """Check if Anthropic API is accessible."""
        if not self.api_key:
            return {"status": "error", "provider": "anthropic", "error": "No API key set"}
        return {
            "status": "ok",
            "provider": "anthropic",
            "model": self.model,
            "note": "API key is set; connection verified on first call",
        }


ProviderFactory.register("anthropic", AnthropicProvider)
