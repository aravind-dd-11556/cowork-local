"""
Anthropic LLM Provider — uses native tool_use API.
"""

from __future__ import annotations
import json
from typing import Optional

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

            if tool_calls:
                return AgentResponse(
                    text=text,
                    tool_calls=tool_calls,
                    stop_reason="tool_use",
                )

            return AgentResponse(text=text or "", stop_reason="end_turn")

        except ImportError:
            return AgentResponse(
                text="Anthropic package not installed. Run: pip install anthropic",
                stop_reason="error",
            )
        except Exception as e:
            return AgentResponse(text=f"Anthropic error: {str(e)}", stop_reason="error")

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
