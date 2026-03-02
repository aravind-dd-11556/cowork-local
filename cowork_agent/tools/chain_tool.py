"""
Chain Tool — Agent-callable tool for adaptive tool chaining.

Allows the agent to specify a goal and have it automatically planned
and executed as an optimized tool chain.

Sprint 28: Adaptive Tool Chaining.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional, TYPE_CHECKING

from .base import BaseTool
from ..core.models import ToolResult

if TYPE_CHECKING:
    from ..core.adaptive_chain import AdaptiveChainExecutor

logger = logging.getLogger(__name__)


class ChainTool(BaseTool):
    """
    Agent-facing tool for adaptive tool chaining.

    Input:
        goal: What the chain should accomplish (e.g., "fix the bug in app.py")
        context: Optional JSON string with context hints (file paths, etc.)

    The tool plans an optimal tool sequence, executes it adaptively,
    and returns a summary of the results.
    """

    name = "chain"
    description = (
        "Plan and execute an adaptive tool chain for a goal. "
        "Provide a goal description and optional context. The chain "
        "will automatically select tools, retry on failure, and adapt."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "What the chain should accomplish",
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional JSON string with context hints "
                    "(e.g., file paths, error messages)"
                ),
            },
        },
        "required": ["goal"],
    }

    def __init__(self, executor: Optional[AdaptiveChainExecutor] = None):
        self._executor = executor

    async def execute(
        self,
        *,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        goal: str = "",
        context: str = "",
        **kwargs,
    ) -> ToolResult:
        tool_id = kwargs.get("tool_id", "")

        if not self._executor:
            return self._error(
                "Adaptive tool chaining is not enabled. "
                "Set adaptive_chain.enabled=true in config.",
                tool_id,
            )

        if not goal:
            return self._error(
                "Missing required parameter: goal",
                tool_id,
            )

        # Parse context hints
        context_hints = None
        if context:
            try:
                context_hints = json.loads(context)
            except json.JSONDecodeError:
                context_hints = {"raw_context": context}

        if progress_callback:
            progress_callback(10, "Planning tool chain...")

        # Get available tools
        available_tools = None
        if self._executor._registry:
            available_tools = self._executor._registry.list_tools()

        # Plan the chain
        plan = self._executor.plan_chain(
            goal=goal,
            available_tools=available_tools,
            context_hints=context_hints,
        )

        if not plan.steps:
            return self._error(
                f"Could not plan a tool chain for goal: '{goal}'. "
                f"No matching template found or no tools available.",
                tool_id,
            )

        if progress_callback:
            progress_callback(30, f"Executing {len(plan.steps)} step chain...")

        # Execute the chain
        result = await self._executor.execute_chain(plan)

        if progress_callback:
            progress_callback(100, "Chain complete")

        # Format result
        if result.success:
            summary_lines = [
                f"Chain completed successfully ({result.steps_completed}/{result.steps_total} steps).",
                f"Template: {plan.template_used or 'custom'}",
                f"Time: {result.execution_time_ms:.0f}ms",
            ]
            if result.adaptations_made:
                summary_lines.append(f"Adaptations: {result.adaptations_made}")
            if result.step_results:
                summary_lines.append(f"\nLast result:\n{result.step_results[-1][:500]}")

            return self._success("\n".join(summary_lines), tool_id)
        else:
            return self._error(
                f"Chain failed: {result.error}\n"
                f"Completed {result.steps_completed}/{result.steps_total} steps. "
                f"Adaptations: {result.adaptations_made}",
                tool_id,
            )
