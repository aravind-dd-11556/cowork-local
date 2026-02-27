"""
Scheduler Tools — Create, list, and update scheduled tasks.

Mirrors real Cowork's scheduled task tools:
  - create_scheduled_task: Define new tasks with cron expressions
  - list_scheduled_tasks: Show all tasks with state
  - update_scheduled_task: Modify existing tasks (partial updates)
"""

from __future__ import annotations
import logging
from typing import Optional

from .base import BaseTool
from ..core.scheduler import TaskScheduler, ScheduledTask

logger = logging.getLogger(__name__)


class CreateScheduledTaskTool(BaseTool):
    name = "create_scheduled_task"
    description = (
        "Create a new scheduled task that runs automatically on a cron schedule "
        "or can be triggered manually. Provide a prompt with instructions for "
        "what the agent should do each run."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "taskId": {
                "type": "string",
                "description": "Kebab-case identifier (e.g. 'daily-standup', 'check-inbox')",
            },
            "prompt": {
                "type": "string",
                "description": "Full task instructions executed each run",
            },
            "description": {
                "type": "string",
                "description": "Short one-line description of what this task does",
            },
            "cronExpression": {
                "type": "string",
                "description": (
                    "5-field cron expression in LOCAL time (minute hour dayOfMonth month dayOfWeek). "
                    "Omit for manual-only tasks. Examples: '0 9 * * *' (daily 9am), "
                    "'0 9 * * 1-5' (weekdays 9am)"
                ),
            },
        },
        "required": ["taskId", "prompt", "description"],
    }

    def __init__(self, scheduler: Optional[TaskScheduler] = None):
        self._scheduler = scheduler or TaskScheduler()

    async def execute(self, taskId: str = "", prompt: str = "",
                      description: str = "", cronExpression: str = "",
                      tool_id: str = "", **kwargs) -> "ToolResult":
        if not taskId:
            return self._error("taskId is required", tool_id)
        if not prompt:
            return self._error("prompt is required", tool_id)
        if not description:
            return self._error("description is required", tool_id)

        # Sanitize task ID: kebab-case, strip unsafe filesystem chars
        import re as _re
        sanitized_id = taskId.lower().replace(" ", "-")
        sanitized_id = _re.sub(r'[^a-z0-9\-]', '', sanitized_id)  # Only allow a-z, 0-9, hyphens
        sanitized_id = _re.sub(r'-+', '-', sanitized_id).strip('-')  # Collapse multiple hyphens
        if not sanitized_id:
            return self._error("taskId contains no valid characters after sanitization", tool_id)

        task = ScheduledTask(
            task_id=sanitized_id,
            prompt=prompt,
            description=description,
            cron_expression=cronExpression,
        )

        result = self._scheduler.create_task(task)
        return self._success(result, tool_id)


class ListScheduledTasksTool(BaseTool):
    name = "list_scheduled_tasks"
    description = (
        "List all scheduled tasks with their current state, schedule, "
        "and last/next run times."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, scheduler: Optional[TaskScheduler] = None):
        self._scheduler = scheduler or TaskScheduler()

    async def execute(self, tool_id: str = "", **kwargs) -> "ToolResult":
        tasks = self._scheduler.list_tasks()

        if not tasks:
            return self._success("No scheduled tasks.", tool_id)

        lines = []
        for t in tasks:
            status = "enabled" if t.get("enabled", True) else "disabled"
            cron = t.get("cron_expression") or "manual only"
            next_run = t.get("next_run_at") or "N/A"
            last_run = t.get("last_run_at") or "never"
            lines.append(
                f"  {t['task_id']}: {t.get('description', '')}\n"
                f"    Schedule: {cron} | Status: {status}\n"
                f"    Next run: {next_run} | Last run: {last_run}"
            )

        return self._success(
            f"Scheduled tasks ({len(tasks)}):\n\n" + "\n\n".join(lines),
            tool_id,
        )


class UpdateScheduledTaskTool(BaseTool):
    name = "update_scheduled_task"
    description = (
        "Update an existing scheduled task. Supports partial updates — "
        "only supply the fields you want to change."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "taskId": {
                "type": "string",
                "description": "The exact ID of the task to update",
            },
            "prompt": {
                "type": "string",
                "description": "New prompt/instructions (optional)",
            },
            "description": {
                "type": "string",
                "description": "New one-line description (optional)",
            },
            "cronExpression": {
                "type": "string",
                "description": "New cron expression in LOCAL time (optional)",
            },
            "enabled": {
                "type": "boolean",
                "description": "Set false to pause, true to resume (optional)",
            },
        },
        "required": ["taskId"],
    }

    def __init__(self, scheduler: Optional[TaskScheduler] = None):
        self._scheduler = scheduler or TaskScheduler()

    async def execute(self, taskId: str = "", prompt: str = "",
                      description: str = "", cronExpression: str = "",
                      enabled: Optional[bool] = None,
                      tool_id: str = "", **kwargs) -> "ToolResult":
        if not taskId:
            return self._error("taskId is required", tool_id)

        update_kwargs = {}
        if prompt:
            update_kwargs["prompt"] = prompt
        if description:
            update_kwargs["description"] = description
        if cronExpression:
            update_kwargs["cron_expression"] = cronExpression
        if enabled is not None:
            update_kwargs["enabled"] = enabled

        if not update_kwargs:
            return self._error("No fields to update. Provide at least one field.", tool_id)

        result = self._scheduler.update_task(taskId, **update_kwargs)

        if "not found" in result.lower():
            return self._error(result, tool_id)

        return self._success(result, tool_id)
