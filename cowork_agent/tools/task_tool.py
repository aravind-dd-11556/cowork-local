"""
Task Tool — Subagent delegation for complex, multi-step tasks.

Mirrors real Cowork's Task tool:
  - Spawns a new agent instance to handle a sub-task
  - The subagent has its own conversation loop
  - Results are returned to the parent agent
  - Supports isolation (separate conversation context)
  - Recursion depth capped to prevent infinite nesting
"""

from __future__ import annotations
import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Callable

from .base import BaseTool

if TYPE_CHECKING:
    from ..core.agent import Agent

logger = logging.getLogger(__name__)

# Module-level depth tracker (thread-safe via asyncio single-thread model)
_current_depth = 0


class TaskTool(BaseTool):
    """
    Spawn a subagent to handle a complex sub-task autonomously.

    The subagent gets a fresh conversation context, executes the task,
    and returns its final response to the parent agent.

    Recursion depth is capped at MAX_DEPTH to prevent infinite nesting
    (a subagent calling task_tool to spawn another subagent, etc.).
    """
    name = "task"
    MAX_DEPTH = 3  # Maximum subagent nesting depth
    description = (
        "Launch a subagent to handle a complex, multi-step task autonomously. "
        "The subagent has access to the same tools but a fresh conversation. "
        "Use this for tasks that are independent and can be delegated, such as "
        "research, code exploration, or file generation. "
        "Returns the subagent's final response."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short (3-5 word) description of the task",
            },
            "prompt": {
                "type": "string",
                "description": "Detailed instructions for the subagent",
            },
            "max_turns": {
                "type": "integer",
                "description": "Maximum iterations for the subagent (default: 10)",
            },
        },
        "required": ["description", "prompt"],
    }

    def __init__(self, agent_factory: Callable[[], "Agent"]):
        """
        Args:
            agent_factory: A callable that creates a fresh Agent instance
                          with the same provider, tools, and config.
        """
        self._agent_factory = agent_factory

    async def execute(
        self,
        description: str = "",
        prompt: str = "",
        max_turns: int = 10,
        tool_id: str = "",
        **kwargs,
    ) -> "ToolResult":
        global _current_depth

        if not prompt:
            return self._error("Prompt is required for task delegation.", tool_id)

        # SEC-CRITICAL-2: Prevent infinite subagent recursion
        if _current_depth >= self.MAX_DEPTH:
            return self._error(
                f"Maximum subagent nesting depth ({self.MAX_DEPTH}) reached. "
                f"Cannot spawn another subagent. Complete the task directly instead.",
                tool_id,
            )

        logger.info(f"Spawning subagent for task: {description} (depth {_current_depth + 1}/{self.MAX_DEPTH})")

        try:
            # Create a fresh agent instance
            subagent = self._agent_factory()
            subagent.max_iterations = max_turns

            # Track recursion depth
            _current_depth += 1
            try:
                # Run the subagent with the prompt
                result = await subagent.run(prompt)
            finally:
                _current_depth -= 1

            logger.info(f"Subagent completed task: {description} ({len(result)} chars)")
            return self._success(
                f"[Subagent result for: {description}]\n\n{result}",
                tool_id,
            )

        except Exception as e:
            logger.error(f"Subagent error: {e}")
            return self._error(f"Subagent failed: {str(e)}", tool_id)
