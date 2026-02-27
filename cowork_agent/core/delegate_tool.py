"""
Delegate Task Tool — Agent-to-agent task delegation.

Allows one agent to spawn or delegate work to another agent,
supporting both synchronous (wait for result) and asynchronous
(fire-and-forget) delegation modes.

Sprint 5 (P3-Multi-Agent Orchestration) Feature 3.
"""

from __future__ import annotations
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..tools.base import BaseTool
from .models import ToolResult, ToolSchema
from .context_bus import ContextBus, BusMessage, MessageType

logger = logging.getLogger(__name__)


class DelegateMode(Enum):
    SYNC = "sync"      # Wait for result
    ASYNC = "async"    # Fire and forget


@dataclass
class DelegatedTask:
    """A task delegated from one agent to another."""
    task_id: str
    delegator: str
    delegatee: str
    task_description: str
    mode: DelegateMode = DelegateMode.SYNC
    created_at: float = field(default_factory=time.time)
    result: Optional[str] = None
    error: Optional[str] = None
    completed_at: Optional[float] = None

    @staticmethod
    def generate_id() -> str:
        return f"task_{uuid.uuid4().hex[:12]}"

    @property
    def is_completed(self) -> bool:
        return self.result is not None or self.error is not None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "delegator": self.delegator,
            "delegatee": self.delegatee,
            "task": self.task_description[:200],
            "mode": self.mode.value,
            "is_completed": self.is_completed,
            "has_error": self.error is not None,
        }


class DelegateTaskTool(BaseTool):
    """
    Tool for delegating work to another agent.

    The supervisor agent calls this like any other tool:
        delegate_task(agent_name="analyzer", task="Analyze data.csv", wait=true)
    """

    name = "delegate_task"
    description = (
        "Delegate a task to another agent in the multi-agent team. "
        "Specify the target agent name, a task description, and whether "
        "to wait for the result (sync) or fire-and-forget (async)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Name of the agent to delegate to",
            },
            "task": {
                "type": "string",
                "description": "Task description for the target agent",
            },
            "wait": {
                "type": "boolean",
                "description": "Wait for result (true) or fire-and-forget (false)",
                "default": True,
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Max seconds to wait (sync mode only)",
                "default": 300,
            },
        },
        "required": ["agent_name", "task"],
    }

    def __init__(self, agent_registry, context_bus: Optional[ContextBus] = None):
        """
        Args:
            agent_registry: AgentRegistry instance for managing agents.
            context_bus: Optional ContextBus for publishing delegation events.
        """
        # Import here to avoid circular dependency
        self._registry = agent_registry
        self._bus = context_bus
        self._tasks: dict[str, DelegatedTask] = {}

    async def execute(self, **kwargs) -> ToolResult:
        agent_name = kwargs.get("agent_name", "")
        task = kwargs.get("task", "")
        wait = kwargs.get("wait", True)
        timeout_seconds = kwargs.get("timeout_seconds", 300)
        tool_id = kwargs.get("tool_id", "")

        if not agent_name:
            return self._error("agent_name is required", tool_id)
        if not task:
            return self._error("task description is required", tool_id)

        # Validate agent exists
        available = self._registry.list_agents()
        if agent_name not in available:
            return self._error(
                f"Agent '{agent_name}' not found. Available: {available}",
                tool_id,
            )

        # Create tracked task
        delegated = DelegatedTask(
            task_id=DelegatedTask.generate_id(),
            delegator=kwargs.get("_delegator", "supervisor"),
            delegatee=agent_name,
            task_description=task,
            mode=DelegateMode.SYNC if wait else DelegateMode.ASYNC,
        )
        self._tasks[delegated.task_id] = delegated

        # Start the agent
        try:
            await self._registry.start_agent(agent_name, task)
        except Exception as e:
            delegated.error = str(e)
            delegated.completed_at = time.time()
            return self._error(f"Failed to start agent '{agent_name}': {e}", tool_id)

        # Publish delegation event
        if self._bus:
            await self._bus.publish(BusMessage(
                msg_type=MessageType.STATUS_UPDATE,
                sender=delegated.delegator,
                content={"action": "delegated", "task_id": delegated.task_id},
                topic="delegation",
                metadata={"delegatee": agent_name},
            ))

        if wait:
            try:
                result = await self._registry.get_result(
                    agent_name, timeout=timeout_seconds
                )
                delegated.result = result
                delegated.completed_at = time.time()

                # Publish completion
                if self._bus:
                    await self._bus.publish(BusMessage(
                        msg_type=MessageType.TASK_RESULT,
                        sender=agent_name,
                        content=result[:500] if result else "",
                        topic="delegation",
                        metadata={"task_id": delegated.task_id},
                    ))

                return self._success(
                    f"[{delegated.task_id}] Agent '{agent_name}' completed:\n{result}",
                    tool_id,
                    task_id=delegated.task_id,
                )
            except asyncio.TimeoutError:
                delegated.error = f"Timeout after {timeout_seconds}s"
                delegated.completed_at = time.time()
                return self._error(
                    f"Agent '{agent_name}' timed out after {timeout_seconds}s",
                    tool_id,
                )
            except Exception as e:
                delegated.error = str(e)
                delegated.completed_at = time.time()
                return self._error(f"Delegation failed: {e}", tool_id)
        else:
            # Async mode — return immediately
            return self._success(
                f"Task delegated to '{agent_name}' (ID: {delegated.task_id}). "
                f"Use poll_task to check status.",
                tool_id,
                task_id=delegated.task_id,
            )

    def get_task(self, task_id: str) -> Optional[DelegatedTask]:
        """Get a delegated task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[DelegatedTask]:
        """List all tracked delegated tasks."""
        return list(self._tasks.values())

    @property
    def pending_count(self) -> int:
        return sum(1 for t in self._tasks.values() if not t.is_completed)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.is_completed)
