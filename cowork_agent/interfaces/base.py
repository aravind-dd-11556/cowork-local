"""
BaseInterface — Abstract base class for all agent interfaces.

All interfaces (CLI, REST API, WebSocket, Telegram, Slack) follow
this contract: accept an Agent, wire callbacks, call run()/run_stream().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..core.agent import Agent
from ..core.models import ToolCall, ToolResult


class BaseInterface(ABC):
    """Abstract base for all agent interfaces.

    Subclasses must implement ``run()`` to start the interface event loop.
    Callback methods are optional overrides for tool/status events.
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        # Wire default callbacks — subclasses can override
        self.agent.on_tool_start = self._on_tool_start
        self.agent.on_tool_end = self._on_tool_end
        self.agent.on_status = self._on_status

    # ── Lifecycle ────────────────────────────────────────────────

    @abstractmethod
    async def run(self) -> None:
        """Start the interface.  May run an event loop, server, or poll."""
        ...

    # ── Event callbacks ──────────────────────────────────────────

    def _on_tool_start(self, call: ToolCall) -> None:
        """Called just before a tool begins execution."""

    def _on_tool_end(self, call: ToolCall, result: ToolResult) -> None:
        """Called when a tool finishes (success or error)."""

    def _on_status(self, message: str) -> None:
        """Called for status updates (retries, nudges, recovery)."""

    def ask_user_handler(self, question: str, options: list[str]) -> str:
        """Handle the ``ask_user`` tool.

        Override to implement interactive questioning.
        Default returns empty string (agent continues without input).
        """
        return ""
