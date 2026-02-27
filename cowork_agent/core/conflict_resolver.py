"""
Conflict Resolution — Detect and resolve conflicts between agent outputs.

Provides:
  - ConflictResolver: Resolves value conflicts using configurable strategies
    (voting, priority, merge, first-win, consensus)
  - ConflictDetector: Detects resource-level conflicts (e.g. two agents
    writing the same file)
  - DeadlockDetector: Cycle detection in agent wait-graphs

Sprint 5 (P3-Multi-Agent Orchestration) Feature 5.
"""

from __future__ import annotations
import asyncio
import logging
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ConflictStrategy(Enum):
    """Strategies for resolving conflicts between agent outputs."""
    VOTING = "voting"          # Majority wins
    PRIORITY = "priority"      # Highest-priority agent wins
    MERGE = "merge"            # Combine non-conflicting parts
    FIRST_WIN = "first_win"    # First value wins
    CONSENSUS = "consensus"    # All must agree (raises on disagreement)


@dataclass
class ConflictReport:
    """Record of a detected and resolved conflict."""
    conflict_id: str
    detected_at: float
    field_name: str
    values: dict[str, Any]              # {agent_name: value}
    resolution_strategy: ConflictStrategy
    resolved_value: Any = None
    resolution_reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.conflict_id,
            "field": self.field_name,
            "strategy": self.resolution_strategy.value,
            "agents_involved": list(self.values.keys()),
            "resolved_value": str(self.resolved_value)[:200],
            "reasoning": self.resolution_reasoning,
        }


class ConflictResolver:
    """
    Resolve conflicts between agent outputs using configurable strategies.

    Usage:
        resolver = ConflictResolver(strategy=ConflictStrategy.VOTING)

        if resolver.detect_conflict("answer", {"a1": "yes", "a2": "no", "a3": "yes"}):
            resolved = await resolver.resolve(
                "answer",
                {"a1": "yes", "a2": "no", "a3": "yes"}
            )
            # resolved = "yes" (majority vote)
    """

    def __init__(self, strategy: ConflictStrategy = ConflictStrategy.VOTING):
        self.strategy = strategy
        self._conflicts: list[ConflictReport] = []

    def detect_conflict(
        self,
        field_name: str,
        agent_values: dict[str, Any],
    ) -> bool:
        """Check if agents produced different values for the same field."""
        if len(agent_values) <= 1:
            return False
        unique = set(str(v) for v in agent_values.values())
        return len(unique) > 1

    async def resolve(
        self,
        field_name: str,
        agent_values: dict[str, Any],
        agent_priorities: Optional[dict[str, int]] = None,
    ) -> Any:
        """
        Resolve a conflict using the configured strategy.

        Args:
            field_name: Name of the conflicting field
            agent_values: {agent_name: value}
            agent_priorities: {agent_name: priority_int} for PRIORITY strategy

        Returns:
            The resolved value
        """
        if not agent_values:
            return None

        # No conflict → return the single value
        if not self.detect_conflict(field_name, agent_values):
            return next(iter(agent_values.values()))

        # Apply resolution strategy
        if self.strategy == ConflictStrategy.VOTING:
            resolved = self._resolve_voting(agent_values)
            reasoning = "Majority vote"
        elif self.strategy == ConflictStrategy.PRIORITY:
            resolved = self._resolve_priority(agent_values, agent_priorities or {})
            reasoning = "Highest priority agent"
        elif self.strategy == ConflictStrategy.MERGE:
            resolved = self._resolve_merge(agent_values)
            reasoning = "Merged non-conflicting parts"
        elif self.strategy == ConflictStrategy.FIRST_WIN:
            resolved = next(iter(agent_values.values()))
            reasoning = "First value wins"
        elif self.strategy == ConflictStrategy.CONSENSUS:
            resolved = self._resolve_consensus(agent_values)
            reasoning = "Consensus required"
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Record the conflict
        report = ConflictReport(
            conflict_id=f"conflict_{uuid.uuid4().hex[:8]}",
            detected_at=time.time(),
            field_name=field_name,
            values=dict(agent_values),
            resolution_strategy=self.strategy,
            resolved_value=resolved,
            resolution_reasoning=reasoning,
        )
        self._conflicts.append(report)

        logger.info(
            f"Resolved conflict on '{field_name}' via {self.strategy.value}: "
            f"{resolved}"
        )
        return resolved

    @staticmethod
    def _resolve_voting(agent_values: dict[str, Any]) -> Any:
        """Majority vote — most common value wins."""
        str_counts = Counter(str(v) for v in agent_values.values())
        winner_str = str_counts.most_common(1)[0][0]

        # Return original (non-stringified) value
        for v in agent_values.values():
            if str(v) == winner_str:
                return v
        return winner_str

    @staticmethod
    def _resolve_priority(
        agent_values: dict[str, Any],
        agent_priorities: dict[str, int],
    ) -> Any:
        """Highest-priority agent's value wins."""
        best_agent = max(
            agent_values.keys(),
            key=lambda a: agent_priorities.get(a, 0),
        )
        return agent_values[best_agent]

    @staticmethod
    def _resolve_merge(agent_values: dict[str, Any]) -> Any:
        """Merge dict values; for non-dicts, return first value."""
        values = list(agent_values.values())

        # If all are dicts, merge them
        if all(isinstance(v, dict) for v in values):
            merged: dict = {}
            for d in values:
                merged.update(d)
            return merged

        # If all are lists, concatenate
        if all(isinstance(v, list) for v in values):
            merged_list: list = []
            for lst in values:
                merged_list.extend(lst)
            return merged_list

        # If all are strings, join with newlines
        if all(isinstance(v, str) for v in values):
            return "\n".join(values)

        # Fallback: return first value
        return values[0]

    @staticmethod
    def _resolve_consensus(agent_values: dict[str, Any]) -> Any:
        """All agents must agree. Raises ValueError if not."""
        unique = set(str(v) for v in agent_values.values())
        if len(unique) > 1:
            raise ValueError(
                f"No consensus: agents produced {len(unique)} different values: "
                f"{dict((k, str(v)[:100]) for k, v in agent_values.items())}"
            )
        return next(iter(agent_values.values()))

    def get_conflict_history(self) -> list[ConflictReport]:
        """Get all recorded conflicts."""
        return list(self._conflicts)

    @property
    def conflict_count(self) -> int:
        return len(self._conflicts)

    def clear_history(self) -> None:
        self._conflicts.clear()


class ConflictDetector:
    """
    Detect resource-level conflicts between agents.

    Tracks which agents are using which resources (files, APIs, etc.)
    and detects when multiple agents try to write the same resource.
    """

    def __init__(self):
        self._resource_locks: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def check_resource_conflict(
        self,
        agent_name: str,
        operation: str,     # "read", "write", "delete"
        resource: str,      # File path, API endpoint, etc.
    ) -> Optional[str]:
        """
        Check if another agent has a conflicting lock on this resource.

        Returns:
            None if no conflict, or the name of the conflicting agent.
        """
        async with self._lock:
            locks = self._resource_locks[resource]

            for other_agent, other_op in locks:
                if other_agent == agent_name:
                    continue  # Same agent can re-acquire

                # Write/delete conflicts with everything
                if other_op in ("write", "delete") or operation in ("write", "delete"):
                    return other_agent

            # No conflict — register this lock
            locks.append((agent_name, operation))
            return None

    async def release_resource(self, agent_name: str, resource: str) -> None:
        """Release all locks held by an agent on a resource."""
        async with self._lock:
            if resource in self._resource_locks:
                self._resource_locks[resource] = [
                    (a, o) for a, o in self._resource_locks[resource]
                    if a != agent_name
                ]

    async def release_all(self, agent_name: str) -> None:
        """Release all locks held by an agent across all resources."""
        async with self._lock:
            for resource in list(self._resource_locks.keys()):
                self._resource_locks[resource] = [
                    (a, o) for a, o in self._resource_locks[resource]
                    if a != agent_name
                ]

    def get_locks(self) -> dict[str, list[tuple[str, str]]]:
        """Get all current resource locks."""
        return dict(self._resource_locks)


class DeadlockDetector:
    """
    Detect deadlock conditions using cycle detection in a wait-graph.

    If agent A waits for B, and B waits for A, that's a deadlock.
    Uses DFS to detect cycles.
    """

    def __init__(self):
        self._wait_graph: dict[str, set[str]] = defaultdict(set)

    def register_wait(self, agent_name: str, waiting_for: str) -> bool:
        """
        Register that agent_name is waiting for waiting_for.

        Returns True if this creates a deadlock (cycle).
        """
        self._wait_graph[agent_name].add(waiting_for)

        # Check for cycle using DFS
        return self._has_cycle(agent_name)

    def clear_wait(self, agent_name: str) -> None:
        """Clear all waits for an agent (it's no longer blocked)."""
        self._wait_graph.pop(agent_name, None)

    def clear_all(self) -> None:
        """Clear the entire wait graph."""
        self._wait_graph.clear()

    def _has_cycle(self, start: str) -> bool:
        """DFS cycle detection from start node."""
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self._wait_graph.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.discard(node)
            return False

        return dfs(start)

    def get_wait_graph(self) -> dict[str, set[str]]:
        """Get the current wait graph."""
        return dict(self._wait_graph)
