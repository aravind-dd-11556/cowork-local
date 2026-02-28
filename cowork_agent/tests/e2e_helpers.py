"""
E2E Test Helpers — MockLLMProvider, mock tools, test fixtures.

Provides reusable components for end-to-end integration tests that exercise
the full agent loop (prompt → LLM → tool execution → response) without
requiring a real LLM backend.

Sprint 22 (End-to-End Integration Tests) Support Module.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from cowork_agent.core.models import (
    AgentResponse, Message, ToolCall, ToolResult, ToolSchema,
)
from cowork_agent.core.providers.base import BaseLLMProvider


# ═══════════════════════════════════════════════════════════════════
#  MockLLMProvider
# ═══════════════════════════════════════════════════════════════════


class MockLLMProvider(BaseLLMProvider):
    """
    Configurable mock LLM provider for E2E testing.

    Supports:
      - Queued responses (text, tool_calls, or errors)
      - Call tracking for assertions
      - Simulated latency
      - Simulated token usage
      - Streaming simulation

    Usage::

        provider = MockLLMProvider()
        provider.enqueue_text("Hello!")
        provider.enqueue_tool_call("bash", {"command": "ls"})
        provider.enqueue_text("Done.")

        agent = make_e2e_agent(provider=provider)
        result = await agent.run("Do something")
    """

    def __init__(self, model: str = "mock-e2e", latency: float = 0.0):
        super().__init__(model=model)
        self._response_queue: List[AgentResponse] = []
        self._call_log: List[Dict[str, Any]] = []
        self._latency = latency
        self._healthy = True
        self._default_usage = {"input_tokens": 100, "output_tokens": 50}

    # ── Enqueue helpers ─────────────────────────────────────────

    def enqueue(self, response: AgentResponse) -> None:
        """Enqueue a raw AgentResponse."""
        self._response_queue.append(response)

    def enqueue_text(self, text: str, usage: Optional[dict] = None) -> None:
        """Enqueue a text-only response (end_turn)."""
        self._response_queue.append(AgentResponse(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            usage=usage or dict(self._default_usage),
        ))

    def enqueue_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
        tool_id: Optional[str] = None,
        extra_text: Optional[str] = None,
    ) -> None:
        """Enqueue a response that requests a tool call."""
        call_id = tool_id or f"tc_{len(self._response_queue) + 1}"
        self._response_queue.append(AgentResponse(
            text=extra_text or "",
            tool_calls=[ToolCall(name=tool_name, tool_id=call_id, input=tool_input)],
            stop_reason="tool_use",
            usage=dict(self._default_usage),
        ))

    def enqueue_multi_tool_call(self, calls: List[Dict[str, Any]]) -> None:
        """Enqueue a response with multiple tool calls."""
        tool_calls = []
        for i, c in enumerate(calls):
            tool_calls.append(ToolCall(
                name=c["name"],
                tool_id=c.get("tool_id", f"tc_multi_{i}"),
                input=c.get("input", {}),
            ))
        self._response_queue.append(AgentResponse(
            text="",
            tool_calls=tool_calls,
            stop_reason="tool_use",
            usage=dict(self._default_usage),
        ))

    def enqueue_error(self, error_msg: str = "Provider error") -> None:
        """Enqueue a response that raises an exception."""
        self._response_queue.append(
            _ErrorSentinel(error_msg)  # type: ignore[arg-type]
        )

    # ── BaseLLMProvider implementation ─────────────────────────

    async def send_message(
        self,
        messages: List[Message],
        tools: List[ToolSchema],
        system_prompt: str,
    ) -> AgentResponse:
        """Return next queued response, tracking the call."""
        if self._latency > 0:
            await asyncio.sleep(self._latency)

        self._call_log.append({
            "messages": messages,
            "tools": tools,
            "system_prompt": system_prompt,
            "timestamp": time.time(),
        })

        if not self._response_queue:
            return AgentResponse(
                text="[MockLLMProvider] No more queued responses.",
                stop_reason="end_turn",
                usage=dict(self._default_usage),
            )

        resp = self._response_queue.pop(0)

        if isinstance(resp, _ErrorSentinel):
            raise RuntimeError(resp.message)

        return resp

    async def send_message_stream(
        self,
        messages: List[Message],
        tools: List[ToolSchema],
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """Simulate streaming by yielding text in chunks."""
        resp = await self.send_message(messages, tools, system_prompt)
        if resp.text:
            words = resp.text.split()
            for word in words:
                yield word + " "
                if self._latency > 0:
                    await asyncio.sleep(self._latency / 10)

    async def health_check(self) -> dict:
        """Return health status."""
        if self._healthy:
            return {"status": "ok", "model": self.model}
        return {"status": "error", "error": "Mock unhealthy"}

    # ── Inspection helpers ──────────────────────────────────────

    @property
    def call_count(self) -> int:
        return len(self._call_log)

    @property
    def call_log(self) -> List[Dict[str, Any]]:
        return list(self._call_log)

    @property
    def remaining_responses(self) -> int:
        return len(self._response_queue)

    def get_last_call(self) -> Optional[Dict[str, Any]]:
        return self._call_log[-1] if self._call_log else None

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy

    def reset(self) -> None:
        """Clear all queued responses and call history."""
        self._response_queue.clear()
        self._call_log.clear()


@dataclass
class _ErrorSentinel:
    """Marker for enqueued errors (not a real AgentResponse)."""
    message: str


# ═══════════════════════════════════════════════════════════════════
#  Mock Tools
# ═══════════════════════════════════════════════════════════════════


class MockTool:
    """
    A configurable mock tool for E2E testing.

    Can be set to succeed, fail, or return custom outputs.
    """

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool for testing",
        schema: Optional[dict] = None,
        output: str = "mock output",
        success: bool = True,
        error: Optional[str] = None,
        latency: float = 0.0,
    ):
        self.name = name
        self.description = description
        self.input_schema = schema or {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }
        self._output = output
        self._success = success
        self._error = error
        self._latency = latency
        self.call_log: List[Dict[str, Any]] = []

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    async def execute(self, tool_id: str = "", **kwargs) -> ToolResult:
        if self._latency > 0:
            await asyncio.sleep(self._latency)

        self.call_log.append({"tool_id": tool_id, "kwargs": kwargs})

        if not self._success:
            return ToolResult(
                tool_id=tool_id,
                success=False,
                output="",
                error=self._error or "Mock tool error",
            )

        return ToolResult(
            tool_id=tool_id,
            success=True,
            output=self._output,
        )


class StatefulTool:
    """A tool that maintains state across calls (for multi-turn tests)."""

    def __init__(self, name: str = "stateful_tool"):
        self.name = name
        self.description = "A stateful tool that tracks calls"
        self.input_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["action"],
        }
        self.state: Dict[str, Any] = {}
        self.call_count = 0

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    async def execute(self, tool_id: str = "", **kwargs) -> ToolResult:
        self.call_count += 1
        action = kwargs.get("action", "get")
        value = kwargs.get("value", "")

        if action == "set":
            self.state["value"] = value
            return ToolResult(tool_id=tool_id, success=True, output=f"Set: {value}")
        elif action == "get":
            v = self.state.get("value", "<empty>")
            return ToolResult(tool_id=tool_id, success=True, output=f"Got: {v}")
        elif action == "count":
            return ToolResult(
                tool_id=tool_id, success=True,
                output=f"Call count: {self.call_count}",
            )
        else:
            return ToolResult(
                tool_id=tool_id, success=False, output="",
                error=f"Unknown action: {action}",
            )


# ═══════════════════════════════════════════════════════════════════
#  Mock PromptBuilder
# ═══════════════════════════════════════════════════════════════════


class MockPromptBuilder:
    """Minimal prompt builder for E2E tests."""

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self._system_prompt = system_prompt

    def build(self, tools=None, context=None):
        return self._system_prompt

    def build_system_prompt(self, tools=None, context=None):
        return self._system_prompt


# ═══════════════════════════════════════════════════════════════════
#  Agent Factory
# ═══════════════════════════════════════════════════════════════════


def make_e2e_agent(
    provider: Optional[MockLLMProvider] = None,
    tools: Optional[List[Any]] = None,
    max_iterations: int = 10,
    workspace_dir: str = "/tmp/e2e_test",
    **kwargs,
):
    """
    Create an Agent wired with mock components for E2E testing.

    Returns (agent, provider, tool_registry) for inspection.
    """
    import os
    os.makedirs(workspace_dir, exist_ok=True)

    from cowork_agent.core.agent import Agent
    from cowork_agent.core.tool_registry import ToolRegistry

    prov = provider or MockLLMProvider()
    registry = ToolRegistry()
    prompt_builder = MockPromptBuilder()

    if tools:
        for tool in tools:
            registry.register(tool)

    agent = Agent(
        provider=prov,
        registry=registry,
        prompt_builder=prompt_builder,
        max_iterations=max_iterations,
        workspace_dir=workspace_dir,
        **kwargs,
    )

    return agent, prov, registry
