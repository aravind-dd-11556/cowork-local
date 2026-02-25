"""
OpenAI LLM Provider — uses native tool_use / function_calling API.
"""

from __future__ import annotations
import json
from typing import Optional

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

            # Check for tool calls
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
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
                )

            return AgentResponse(
                text=choice.message.content or "",
                stop_reason="end_turn",
            )

        except ImportError:
            return AgentResponse(
                text="OpenAI package not installed. Run: pip install openai",
                stop_reason="error",
            )
        except Exception as e:
            return AgentResponse(text=f"OpenAI error: {str(e)}", stop_reason="error")

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
            else:
                result.append({
                    "role": msg.role if msg.role != "tool_result" else "user",
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
