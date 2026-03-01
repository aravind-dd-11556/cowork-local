"""
Scheduler Tools (Extended) — Delete and run-now for scheduled tasks.

Complements the existing create/list/update tools from scheduler_tools.py.
  - delete_scheduled_task: Remove a scheduled task
  - run_scheduled_task: Immediately execute a task regardless of schedule

Sprint 25: Immutable Security Hardening + Scheduler Activation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Callable

from .base import BaseTool
from ..core.scheduler import TaskScheduler

logger = logging.getLogger(__name__)


class DeleteScheduledTaskTool(BaseTool):
    name = "delete_scheduled_task"
    description = (
        "Delete a scheduled task by its ID. This permanently removes the task "
        "and its schedule. Use list_scheduled_tasks first to see available tasks."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "taskId": {
                "type": "string",
                "description": "The exact ID of the task to delete",
            },
        },
        "required": ["taskId"],
    }

    def __init__(self, scheduler: Optional[TaskScheduler] = None):
        self._scheduler = scheduler or TaskScheduler()

    async def execute(self, taskId: str = "", tool_id: str = "", **kwargs) -> "ToolResult":
        if not taskId:
            return self._error("taskId is required", tool_id)

        result = self._scheduler.delete_task(taskId)

        if "not found" in result.lower():
            return self._error(result, tool_id)

        return self._success(result, tool_id)


class RunScheduledTaskTool(BaseTool):
    name = "run_scheduled_task"
    description = (
        "Immediately execute a scheduled task regardless of its cron schedule. "
        "The task runs once and its last_run_at is updated."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "taskId": {
                "type": "string",
                "description": "The exact ID of the task to run",
            },
        },
        "required": ["taskId"],
    }

    def __init__(
        self,
        scheduler: Optional[TaskScheduler] = None,
        agent_runner: Optional[Callable] = None,
    ):
        self._scheduler = scheduler or TaskScheduler()
        self._agent_runner = agent_runner

    def set_agent_runner(self, runner: Callable) -> None:
        """Set the agent runner callback (may be set after init)."""
        self._agent_runner = runner

    async def execute(self, taskId: str = "", tool_id: str = "", **kwargs) -> "ToolResult":
        if not taskId:
            return self._error("taskId is required", tool_id)

        if not self._agent_runner:
            return self._error(
                "Agent runner not configured. Cannot execute tasks.", tool_id
            )

        try:
            result = await self._scheduler.run_now(taskId, self._agent_runner)
            return self._success(result, tool_id)
        except Exception as e:
            return self._error(f"Task execution failed: {e}", tool_id)
