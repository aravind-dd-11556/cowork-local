"""
Sprint 43 · Multi-Agent Crew Mode
====================================
CrewManager coordinates specialized sub-agents to collaborate on
complex tasks via decomposition, role assignment, and result aggregation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .crew_roles import CrewRole, RoleAssigner
from .task_decomposer import DecompositionResult, SubTask, TaskDecomposer
from .result_aggregator import AggregationStrategy, ResultAggregator


# ── Enums & Config ──────────────────────────────────────────────────

class CrewStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"


@dataclass
class CrewConfig:
    """Configuration for a crew execution."""
    name: str = "default_crew"
    strategy: CrewStrategy = CrewStrategy.SEQUENTIAL
    max_agents: int = 4
    timeout_total: float = 300.0  # seconds
    auto_review: bool = True
    aggregation_strategy: AggregationStrategy = AggregationStrategy.CONCATENATE

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "max_agents": self.max_agents,
            "timeout_total": self.timeout_total,
            "auto_review": self.auto_review,
            "aggregation_strategy": self.aggregation_strategy.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CrewConfig":
        cfg = cls()
        if "name" in d:
            cfg.name = d["name"]
        if "strategy" in d:
            cfg.strategy = CrewStrategy(d["strategy"])
        if "max_agents" in d:
            cfg.max_agents = d["max_agents"]
        if "timeout_total" in d:
            cfg.timeout_total = d["timeout_total"]
        if "auto_review" in d:
            cfg.auto_review = d["auto_review"]
        if "aggregation_strategy" in d:
            cfg.aggregation_strategy = AggregationStrategy(d["aggregation_strategy"])
        return cfg


# ── Result ──────────────────────────────────────────────────────────

@dataclass
class CrewResult:
    """Outcome of a crew task execution."""
    success: bool
    task_description: str
    sub_results: List[Dict[str, Any]] = field(default_factory=list)
    aggregated_output: str = ""
    execution_time_ms: float = 0.0
    agents_used: List[str] = field(default_factory=list)
    review_output: Optional[str] = None
    decomposition: Optional[DecompositionResult] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "task_description": self.task_description,
            "sub_results": self.sub_results,
            "aggregated_output": self.aggregated_output,
            "execution_time_ms": self.execution_time_ms,
            "agents_used": self.agents_used,
            "review_output": self.review_output,
        }

    @property
    def agent_count(self) -> int:
        return len(self.agents_used)


# ── Crew Manager ────────────────────────────────────────────────────

class CrewManager:
    """
    Orchestrates specialized sub-agents for complex tasks.

    Flow:
      1. Decompose task → sub-tasks
      2. Assign roles to sub-tasks
      3. Execute sub-tasks (sequentially or in parallel)
      4. Aggregate results
      5. Optional: run reviewer on output
    """

    def __init__(
        self,
        config: CrewConfig,
        task_decomposer: Optional[TaskDecomposer] = None,
        role_assigner: Optional[RoleAssigner] = None,
        result_aggregator: Optional[ResultAggregator] = None,
        agent_executor: Any = None,  # callable(role, task_desc) → str
    ):
        self.config = config
        self.decomposer = task_decomposer or TaskDecomposer()
        self.role_assigner = role_assigner or RoleAssigner()
        self.aggregator = result_aggregator or ResultAggregator()
        self.agent_executor = agent_executor
        self._execution_history: List[CrewResult] = []

    # ── public API ────────────────────────────────────────────────

    async def execute_crew_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> CrewResult:
        """
        Execute a complex task using a crew of specialized agents.
        """
        t0 = time.monotonic()

        # 1. Decompose
        decomposition = self.decomposer.decompose(task_description)

        # 2. Assign roles
        role_assignments: Dict[str, CrewRole] = {}
        for st in decomposition.sub_tasks:
            role = self.role_assigner.assign_role(st.description)
            role_assignments[st.id] = role

        # 3. Execute sub-tasks per strategy
        sub_results: List[Dict[str, Any]] = []
        agents_used: List[str] = []

        if self.config.strategy == CrewStrategy.SEQUENTIAL:
            sub_results, agents_used = await self._execute_sequential(
                decomposition, role_assignments, context,
            )
        elif self.config.strategy == CrewStrategy.PARALLEL:
            sub_results, agents_used = await self._execute_parallel(
                decomposition, role_assignments, context,
            )
        elif self.config.strategy == CrewStrategy.PIPELINE:
            sub_results, agents_used = await self._execute_pipeline(
                decomposition, role_assignments, context,
            )

        # 4. Aggregate
        aggregated = self.aggregator.aggregate(
            sub_results,
            strategy=self.config.aggregation_strategy,
        )

        # 5. Optional review pass
        review_output = None
        if self.config.auto_review and self.agent_executor:
            try:
                reviewer_role = self.role_assigner.get_role("reviewer")
                review_output = await self._run_review(
                    aggregated, reviewer_role, task_description,
                )
                agents_used.append("reviewer")
            except Exception:
                pass  # review is best-effort

        elapsed = (time.monotonic() - t0) * 1000

        result = CrewResult(
            success=len(sub_results) > 0,
            task_description=task_description,
            sub_results=sub_results,
            aggregated_output=aggregated,
            execution_time_ms=elapsed,
            agents_used=agents_used,
            review_output=review_output,
            decomposition=decomposition,
        )
        self._execution_history.append(result)
        return result

    # ── execution strategies ──────────────────────────────────────

    async def _execute_sequential(
        self,
        decomposition: DecompositionResult,
        role_assignments: Dict[str, CrewRole],
        context: Optional[Dict[str, Any]],
    ) -> tuple:
        """Execute sub-tasks one by one in dependency order."""
        sub_results: List[Dict[str, Any]] = []
        agents_used: List[str] = []
        prev_output = ""

        for stage in decomposition.execution_order:
            for task_id in stage:
                st = self._find_subtask(decomposition, task_id)
                if not st:
                    continue
                role = role_assignments.get(task_id)
                if not role:
                    continue

                output = await self._execute_subtask(
                    st, role, prev_output, context,
                )
                sub_results.append({
                    "task_id": task_id,
                    "role": role.name,
                    "task_description": st.description,
                    "output": output,
                    "confidence": 0.7,
                })
                agents_used.append(role.name)
                prev_output = output

        return sub_results, agents_used

    async def _execute_parallel(
        self,
        decomposition: DecompositionResult,
        role_assignments: Dict[str, CrewRole],
        context: Optional[Dict[str, Any]],
    ) -> tuple:
        """Execute all sub-tasks in parallel (ignoring dependencies)."""
        import asyncio

        sub_results: List[Dict[str, Any]] = []
        agents_used: List[str] = []

        async def _run_one(st, role):
            output = await self._execute_subtask(st, role, "", context)
            return {
                "task_id": st.id,
                "role": role.name,
                "task_description": st.description,
                "output": output,
                "confidence": 0.7,
            }

        tasks = []
        for st in decomposition.sub_tasks:
            role = role_assignments.get(st.id)
            if role:
                tasks.append(_run_one(st, role))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, dict):
                sub_results.append(r)
                agents_used.append(r["role"])

        return sub_results, agents_used

    async def _execute_pipeline(
        self,
        decomposition: DecompositionResult,
        role_assignments: Dict[str, CrewRole],
        context: Optional[Dict[str, Any]],
    ) -> tuple:
        """Execute stage by stage, passing results forward."""
        sub_results: List[Dict[str, Any]] = []
        agents_used: List[str] = []
        stage_output = ""

        for stage in decomposition.execution_order:
            stage_results = []
            for task_id in stage:
                st = self._find_subtask(decomposition, task_id)
                if not st:
                    continue
                role = role_assignments.get(task_id)
                if not role:
                    continue

                output = await self._execute_subtask(
                    st, role, stage_output, context,
                )
                result_dict = {
                    "task_id": task_id,
                    "role": role.name,
                    "task_description": st.description,
                    "output": output,
                    "confidence": 0.7,
                }
                stage_results.append(result_dict)
                sub_results.append(result_dict)
                agents_used.append(role.name)

            # Combine stage outputs for next stage
            if stage_results:
                stage_output = "\n\n".join(r["output"] for r in stage_results)

        return sub_results, agents_used

    # ── helpers ──────────────────────────────────────────────────

    async def _execute_subtask(
        self,
        sub_task: SubTask,
        role: CrewRole,
        previous_output: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Execute a single sub-task via the agent executor."""
        if self.agent_executor is None:
            return f"[{role.name}] Would execute: {sub_task.description}"

        try:
            result = self.agent_executor(role, sub_task.description)
            if hasattr(result, '__await__'):
                result = await result
            return str(result)
        except Exception as e:
            return f"[{role.name}] Error: {e}"

    async def _run_review(
        self,
        output: str,
        reviewer_role: CrewRole,
        original_task: str,
    ) -> str:
        """Run a review pass on the aggregated output."""
        review_prompt = f"Review the following output for: {original_task}\n\n{output}"
        if self.agent_executor:
            try:
                result = self.agent_executor(reviewer_role, review_prompt)
                if hasattr(result, '__await__'):
                    result = await result
                return str(result)
            except Exception:
                return ""
        return ""

    @staticmethod
    def _find_subtask(
        decomposition: DecompositionResult,
        task_id: str,
    ) -> Optional[SubTask]:
        for st in decomposition.sub_tasks:
            if st.id == task_id:
                return st
        return None

    # ── metrics ──────────────────────────────────────────────────

    @property
    def execution_history(self) -> List[CrewResult]:
        return list(self._execution_history)

    @property
    def total_executions(self) -> int:
        return len(self._execution_history)

    def stats(self) -> Dict[str, Any]:
        total = len(self._execution_history)
        successes = sum(1 for r in self._execution_history if r.success)
        return {
            "total_executions": total,
            "successful": successes,
            "failed": total - successes,
            "config": self.config.to_dict(),
        }
