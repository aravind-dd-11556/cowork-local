"""
Task Scheduler — Cron-based background task execution.

Mirrors real Cowork's scheduled tasks:
  - Tasks defined with cron expressions
  - Each task has a prompt that the agent executes
  - Tasks stored in workspace/.cowork/scheduled/
  - Supports create, update, list, enable/disable
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """A scheduled task definition."""
    task_id: str
    prompt: str
    description: str
    cron_expression: str = ""  # Empty = ad-hoc only (manual trigger)
    enabled: bool = True
    last_run_at: Optional[str] = None
    next_run_at: Optional[str] = None
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "description": self.description,
            "cron_expression": self.cron_expression,
            "enabled": self.enabled,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledTask":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TaskScheduler:
    """
    Manages scheduled tasks with cron-like scheduling.

    Usage:
        scheduler = TaskScheduler(workspace_dir="/path/to/workspace")
        scheduler.load()

        scheduler.create_task(ScheduledTask(
            task_id="daily-standup",
            prompt="Summarize my tasks for today",
            description="Daily standup summary",
            cron_expression="0 9 * * 1-5",
        ))

        # Start the scheduler loop (runs in background)
        await scheduler.start(agent_runner=run_func)
    """

    def __init__(self, workspace_dir: str = ""):
        self.workspace_dir = workspace_dir
        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._storage_dir = ""

        if workspace_dir:
            self._storage_dir = os.path.join(workspace_dir, ".cowork", "scheduled")
            os.makedirs(self._storage_dir, exist_ok=True)

    @property
    def tasks(self) -> dict[str, ScheduledTask]:
        return dict(self._tasks)

    def load(self) -> int:
        """Load all scheduled tasks from disk."""
        if not self._storage_dir:
            return 0

        self._tasks.clear()
        count = 0

        for filename in sorted(os.listdir(self._storage_dir)):
            if filename.endswith(".json"):
                filepath = os.path.join(self._storage_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    task = ScheduledTask.from_dict(data)
                    self._tasks[task.task_id] = task
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to load task {filename}: {e}")

        logger.info(f"Loaded {count} scheduled tasks")
        return count

    def save_task(self, task: ScheduledTask) -> None:
        """Save a task to disk."""
        if not self._storage_dir:
            return
        filepath = os.path.join(self._storage_dir, f"{task.task_id}.json")
        try:
            with open(filepath, "w") as f:
                json.dump(task.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save task {task.task_id}: {e}")

    def create_task(self, task: ScheduledTask) -> str:
        """Create a new scheduled task."""
        if not task.created_at:
            task.created_at = datetime.now().isoformat()

        if task.cron_expression:
            task.next_run_at = self._next_cron_time(task.cron_expression)

        self._tasks[task.task_id] = task
        self.save_task(task)
        logger.info(f"Created scheduled task: {task.task_id}")
        return f"Task '{task.task_id}' created. Next run: {task.next_run_at or 'manual only'}"

    def update_task(self, task_id: str, **kwargs) -> str:
        """Update an existing task."""
        task = self._tasks.get(task_id)
        if not task:
            return f"Task '{task_id}' not found."

        for key, value in kwargs.items():
            if hasattr(task, key) and value is not None:
                setattr(task, key, value)

        if "cron_expression" in kwargs and kwargs["cron_expression"]:
            task.next_run_at = self._next_cron_time(task.cron_expression)

        self.save_task(task)
        return f"Task '{task_id}' updated."

    def delete_task(self, task_id: str) -> str:
        """Delete a scheduled task."""
        if task_id not in self._tasks:
            return f"Task '{task_id}' not found."

        del self._tasks[task_id]

        if self._storage_dir:
            filepath = os.path.join(self._storage_dir, f"{task_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)

        return f"Task '{task_id}' deleted."

    def list_tasks(self) -> list[dict]:
        """Return all tasks as dicts."""
        return [t.to_dict() for t in self._tasks.values()]

    async def start(self, agent_runner: Callable[[str], asyncio.Future]) -> None:
        """
        Start the scheduler loop. Checks every 60 seconds for tasks to run.

        Args:
            agent_runner: Async function that takes a prompt and runs the agent.
        """
        self._running = True
        logger.info("Task scheduler started")

        while self._running:
            now = datetime.now()

            for task in self._tasks.values():
                if not task.enabled or not task.cron_expression:
                    continue

                if task.next_run_at:
                    try:
                        next_run = datetime.fromisoformat(task.next_run_at)
                        if now >= next_run:
                            logger.info(f"Running scheduled task: {task.task_id}")
                            try:
                                await agent_runner(task.prompt)
                            except Exception as e:
                                logger.error(f"Scheduled task '{task.task_id}' failed: {e}")

                            task.last_run_at = now.isoformat()
                            task.next_run_at = self._next_cron_time(task.cron_expression)
                            self.save_task(task)
                    except ValueError:
                        pass

            await asyncio.sleep(60)

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False

    @staticmethod
    def _next_cron_time(cron_expr: str) -> Optional[str]:
        """
        Compute the next run time from a simplified cron expression.
        Supports: minute hour dayOfMonth month dayOfWeek

        This is a simplified implementation. For production, use a
        library like croniter.
        """
        try:
            parts = cron_expr.strip().split()
            if len(parts) != 5:
                return None

            minute, hour, dom, month, dow = parts
            now = datetime.now()

            # Simple case: specific hour and minute with wildcards for rest
            if minute.isdigit() and hour.isdigit() and dom == "*" and month == "*":
                target_minute = int(minute)
                target_hour = int(hour)

                candidate = now.replace(
                    hour=target_hour, minute=target_minute, second=0, microsecond=0
                )

                if candidate <= now:
                    candidate += timedelta(days=1)

                # Handle day-of-week filter
                if dow != "*":
                    dow_values = _parse_cron_field(dow, 0, 6)
                    while candidate.weekday() not in _convert_cron_dow(dow_values):
                        candidate += timedelta(days=1)

                return candidate.isoformat()

            # Fallback: next hour
            return (now + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0
            ).isoformat()

        except Exception:
            return None


def _parse_cron_field(field: str, min_val: int, max_val: int) -> list[int]:
    """
    Parse a single cron field into a list of valid values.

    SEC-MEDIUM-3: Now validates that range bounds are within min_val..max_val
    and that start <= end in ranges.
    """
    values = set()

    for part in field.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start, end = int(start_str), int(end_str)
                # Validate range bounds
                if start > end:
                    logger.warning(f"Invalid cron range: {start}-{end} (start > end)")
                    continue
                if start < min_val or end > max_val:
                    logger.warning(
                        f"Cron range {start}-{end} out of bounds ({min_val}-{max_val}), clamping"
                    )
                    start = max(start, min_val)
                    end = min(end, max_val)
                values.update(range(start, end + 1))
            except ValueError:
                pass
        elif part.isdigit():
            val = int(part)
            if min_val <= val <= max_val:
                values.add(val)
            else:
                logger.warning(f"Cron value {val} out of bounds ({min_val}-{max_val})")

    return sorted(v for v in values if min_val <= v <= max_val)


def _convert_cron_dow(cron_values: list[int]) -> set[int]:
    """Convert cron day-of-week (0=Sun) to Python weekday (0=Mon)."""
    mapping = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    return {mapping.get(v, v) for v in cron_values}
