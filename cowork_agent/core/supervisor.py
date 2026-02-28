"""
Supervisor — Orchestrates a team of agents to solve complex tasks.

Supports three execution strategies:
  - SEQUENTIAL: Agents run one after another
  - PARALLEL: All agents run concurrently
  - PIPELINE: Chain agents, passing each output as the next input

Sprint 5 (P3-Multi-Agent Orchestration) Feature 4.
"""

from __future__ import annotations
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .agent_registry import AgentRegistry
from .context_bus import ContextBus, BusMessage, MessageType

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    # Sprint 21: Advanced strategies
    MAP_REDUCE = "map_reduce"
    DEBATE = "debate"
    VOTING = "voting"


@dataclass
class SupervisorConfig:
    """Configuration for a supervisor."""
    name: str = "supervisor"
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    max_retries: int = 2
    on_failure: str = "abort"          # "abort" | "skip" | "retry"
    timeout_per_agent: float = 300.0   # Per-agent timeout
    timeout_total: float = 600.0       # Total timeout


@dataclass
class SubTask:
    """A subtask assigned to a specific agent."""
    id: str = ""
    agent_name: str = ""
    description: str = ""
    parent_output: Optional[str] = None    # Pipeline: previous agent's result
    result: Optional[str] = None
    error: Optional[str] = None
    retries: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def is_completed(self) -> bool:
        return self.result is not None or self.error is not None

    @staticmethod
    def generate_id() -> str:
        return f"sub_{uuid.uuid4().hex[:8]}"


class Supervisor:
    """
    Orchestrates a team of agents to solve a complex task.

    Usage:
        supervisor = Supervisor(
            config=SupervisorConfig(
                name="code_review",
                strategy=ExecutionStrategy.PIPELINE,
            ),
            agent_registry=registry,
            context_bus=bus,
        )

        result = await supervisor.execute_task(
            task="Review the codebase for security issues",
            agents=["code_reader", "security_analyzer", "report_writer"],
        )
    """

    def __init__(
        self,
        config: SupervisorConfig,
        agent_registry: AgentRegistry,
        context_bus: Optional[ContextBus] = None,
    ):
        self.config = config
        self.registry = agent_registry
        self.bus = context_bus
        self._subtasks: list[SubTask] = []
        self._progress: dict = {}
        self._start_time: Optional[float] = None

    async def execute_task(
        self,
        task: str,
        agents: list[str],
        max_parallel: int = 5,
    ) -> str:
        """
        Execute a task using the configured strategy.

        Args:
            task: High-level task description
            agents: Ordered list of agent names to use
            max_parallel: Max concurrent agents (PARALLEL mode)

        Returns:
            Aggregated result string
        """
        if not agents:
            raise ValueError("At least one agent is required")

        self._start_time = time.time()
        self._subtasks = self._create_subtasks(task, agents)
        self._progress = {
            "total": len(agents),
            "completed": 0,
            "failed": 0,
            "strategy": self.config.strategy.value,
            "started_at": self._start_time,
        }

        # Publish start event
        if self.bus:
            await self.bus.publish(BusMessage(
                msg_type=MessageType.STATUS_UPDATE,
                sender=self.config.name,
                content={"action": "started", "agents": agents},
                topic="supervisor",
            ))

        try:
            if self.config.strategy == ExecutionStrategy.SEQUENTIAL:
                results = await self._execute_sequential()
            elif self.config.strategy == ExecutionStrategy.PARALLEL:
                results = await self._execute_parallel(max_parallel)
            elif self.config.strategy == ExecutionStrategy.PIPELINE:
                results = await self._execute_pipeline()
            else:
                raise ValueError(f"Unknown strategy: {self.config.strategy}")

            aggregated = self._aggregate_results(results)

            # Publish completion
            if self.bus:
                await self.bus.publish(BusMessage(
                    msg_type=MessageType.TASK_RESULT,
                    sender=self.config.name,
                    content={"action": "completed", "summary": aggregated[:200]},
                    topic="supervisor",
                ))

            return aggregated

        except Exception as e:
            logger.error(f"Supervisor '{self.config.name}' failed: {e}")
            if self.bus:
                await self.bus.publish(BusMessage(
                    msg_type=MessageType.ERROR,
                    sender=self.config.name,
                    content=str(e),
                    topic="supervisor",
                ))
            raise

    def _create_subtasks(self, task: str, agents: list[str]) -> list[SubTask]:
        """Create subtasks for each agent."""
        return [
            SubTask(
                id=SubTask.generate_id(),
                agent_name=agent_name,
                description=task,
            )
            for agent_name in agents
        ]

    async def _execute_sequential(self) -> list[str]:
        """Run subtasks one after another."""
        results = []
        for subtask in self._subtasks:
            result = await self._run_subtask(subtask)
            results.append(result)
            self._progress["completed"] += 1
        return results

    async def _execute_parallel(self, max_parallel: int = 5) -> list[str]:
        """Run subtasks concurrently with a semaphore cap."""
        semaphore = asyncio.Semaphore(max_parallel)

        async def _limited(subtask: SubTask) -> str:
            async with semaphore:
                return await self._run_subtask(subtask)

        tasks = [_limited(st) for st in self._subtasks]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                self._progress["failed"] += 1
                self._subtasks[i].error = str(r)
                results.append(f"[ERROR: {r}]")
            else:
                self._progress["completed"] += 1
                results.append(r)
        return results

    async def _execute_pipeline(self) -> list[str]:
        """Run subtasks in sequence, chaining output → input."""
        results = []
        previous_output = None

        for subtask in self._subtasks:
            if previous_output:
                subtask.parent_output = previous_output
                subtask.description += f"\n\nPrevious agent output:\n{previous_output}"

            result = await self._run_subtask(subtask)
            results.append(result)
            previous_output = result
            self._progress["completed"] += 1

        return results

    async def _run_subtask(self, subtask: SubTask) -> str:
        """Run a single subtask with retry logic."""
        subtask.started_at = time.time()

        for attempt in range(self.config.max_retries + 1):
            try:
                # Start the agent
                await self.registry.start_agent(
                    subtask.agent_name, subtask.description
                )
                result = await self.registry.get_result(
                    subtask.agent_name,
                    timeout=self.config.timeout_per_agent,
                )
                subtask.result = result
                subtask.completed_at = time.time()
                return result

            except Exception as e:
                subtask.retries = attempt + 1
                logger.warning(
                    f"Subtask {subtask.id} ({subtask.agent_name}) "
                    f"failed attempt {attempt + 1}: {e}"
                )

                if attempt < self.config.max_retries:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Brief backoff
                    continue

                # All retries exhausted
                subtask.error = str(e)
                subtask.completed_at = time.time()
                self._progress["failed"] += 1

                if self.config.on_failure == "abort":
                    raise
                elif self.config.on_failure == "skip":
                    return f"[SKIPPED: {e}]"
                else:
                    return f"[FAILED after {self.config.max_retries + 1} attempts: {e}]"

        return "[INTERNAL ERROR]"  # Should never reach here

    def _aggregate_results(self, results: list[str]) -> str:
        """Combine results from all agents into a single output."""
        parts = [
            f"=== Supervisor '{self.config.name}' "
            f"({self.config.strategy.value}) ===\n"
        ]

        for subtask, result in zip(self._subtasks, results):
            parts.append(f"\n--- {subtask.agent_name} ---")
            parts.append(result or "[No output]")

        elapsed = time.time() - (self._start_time or time.time())
        parts.append(
            f"\n\n[Summary: {self._progress['completed']}/{self._progress['total']} "
            f"completed, {self._progress['failed']} failed, "
            f"{elapsed:.1f}s elapsed]"
        )

        return "\n".join(parts)

    def get_progress(self) -> dict:
        """Get current execution progress."""
        return dict(self._progress)

    def get_subtasks(self) -> list[SubTask]:
        """Get all subtasks with their status."""
        return list(self._subtasks)
