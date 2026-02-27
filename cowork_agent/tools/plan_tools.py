"""
Plan Mode Tools — EnterPlanMode and ExitPlanMode.
These tools allow the agent to switch between planning and execution modes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .base import BaseTool

if TYPE_CHECKING:
    from ..core.plan_mode import PlanManager


class EnterPlanModeTool(BaseTool):
    """Tool to enter plan mode — restricts agent to read-only exploration."""
    name = "enter_plan_mode"
    description = (
        "Enter plan mode to explore the codebase and design an implementation "
        "approach before making changes. In plan mode, only read-only tools are "
        "available (read, glob, grep, web_search, web_fetch). Use this when "
        "a task is complex and needs architectural planning first."
    )
    input_schema = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, plan_manager: "PlanManager"):
        self._plan_manager = plan_manager

    async def execute(self, tool_id: str = "", **kwargs) -> "ToolResult":
        result = self._plan_manager.enter_plan_mode()
        return self._success(result, tool_id)


class ExitPlanModeTool(BaseTool):
    """Tool to exit plan mode — returns to normal execution."""
    name = "exit_plan_mode"
    description = (
        "Exit plan mode and return to normal execution. Optionally provide "
        "your implementation plan which will be saved and presented to the "
        "user for approval."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": "The implementation plan text (markdown formatted)",
            },
        },
    }

    def __init__(self, plan_manager: "PlanManager"):
        self._plan_manager = plan_manager

    async def execute(self, plan: str = "", tool_id: str = "", **kwargs) -> "ToolResult":
        result = self._plan_manager.exit_plan_mode(plan)
        return self._success(result, tool_id)
