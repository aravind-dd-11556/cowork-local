"""
Agent Specialization — Role definitions, specialization registry, keyword matching.

Provides a system for defining agent roles and capabilities, then matching
incoming tasks to the best-suited agent based on keyword and capability analysis.

Sprint 21 (Multi-Agent Orchestration Enhancement) Module 2.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Agent Roles ───────────────────────────────────────────────────


class AgentRole(Enum):
    """Pre-defined agent specialization roles."""
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    WRITER = "writer"
    ARCHITECT = "architect"
    GENERAL = "general"


# Default keywords for each role
ROLE_KEYWORDS: Dict[AgentRole, List[str]] = {
    AgentRole.RESEARCHER: [
        "research", "find", "search", "investigate", "explore", "discover",
        "analyze", "study", "review literature", "survey", "gather",
    ],
    AgentRole.CODER: [
        "code", "implement", "program", "develop", "build", "create function",
        "write code", "fix bug", "debug", "refactor", "script",
    ],
    AgentRole.REVIEWER: [
        "review", "audit", "inspect", "check", "evaluate", "assess",
        "code review", "quality", "verify", "validate",
    ],
    AgentRole.TESTER: [
        "test", "unit test", "integration test", "qa", "quality assurance",
        "testing", "regression", "coverage", "benchmark", "verify",
    ],
    AgentRole.WRITER: [
        "write", "document", "documentation", "report", "summarize",
        "draft", "compose", "blog", "article", "readme",
    ],
    AgentRole.ARCHITECT: [
        "design", "architect", "plan", "structure", "organize", "pattern",
        "system design", "api design", "schema", "blueprint",
    ],
    AgentRole.GENERAL: [],
}


# ── Specialization Dataclass ─────────────────────────────────────


@dataclass
class AgentSpecialization:
    """Defines an agent's specialization for task routing."""
    role: AgentRole
    keywords: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.3
    capability_weights: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    priority: int = 0  # Higher = preferred when scores are equal

    def __post_init__(self):
        # Merge role-default keywords with custom keywords
        role_kw = ROLE_KEYWORDS.get(self.role, [])
        all_kw = list(set(role_kw + self.keywords))
        self.keywords = [kw.lower() for kw in all_kw]


# ── Specialization Registry ──────────────────────────────────────


class SpecializationRegistry:
    """
    Registry that maps agent names to specializations and provides
    best-fit matching for tasks.

    Usage::

        registry = SpecializationRegistry()
        registry.register_agent("code_agent", AgentSpecialization(
            role=AgentRole.CODER,
            keywords=["python", "javascript"],
        ))
        best, confidence = registry.find_best_agent("Write a Python script", agents)
    """

    def __init__(self):
        self._specializations: Dict[str, AgentSpecialization] = {}

    def register_agent(self, name: str, spec: AgentSpecialization) -> None:
        """Register an agent's specialization."""
        self._specializations[name] = spec
        logger.debug("Registered specialization for '%s': role=%s", name, spec.role.value)

    def unregister_agent(self, name: str) -> bool:
        """Remove an agent's specialization. Returns True if found."""
        return self._specializations.pop(name, None) is not None

    def get_specialization(self, name: str) -> Optional[AgentSpecialization]:
        """Get the specialization for a named agent."""
        return self._specializations.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._specializations.keys())

    def list_by_role(self, role: AgentRole) -> List[str]:
        """List agents with a specific role."""
        return [
            name for name, spec in self._specializations.items()
            if spec.role == role
        ]

    def find_best_agent(
        self,
        task: str,
        available_agents: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Find the best-suited agent for a task.

        Args:
            task: Task description to match
            available_agents: If provided, only consider these agents

        Returns:
            (agent_name, confidence) or (None, 0.0) if no match above threshold
        """
        candidates = available_agents or list(self._specializations.keys())
        task_lower = task.lower()

        best_name = None
        best_score = 0.0

        for name in candidates:
            spec = self._specializations.get(name)
            if spec is None:
                continue

            score = self._score_match(task_lower, spec)

            # Apply priority as a small tiebreaker
            score += spec.priority * 0.001

            if score > best_score and score >= spec.confidence_threshold:
                best_score = score
                best_name = name

        return best_name, best_score

    def find_top_agents(
        self,
        task: str,
        top_n: int = 3,
        available_agents: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find the top N best-suited agents for a task.

        Returns:
            List of (agent_name, confidence) sorted by confidence descending
        """
        candidates = available_agents or list(self._specializations.keys())
        task_lower = task.lower()

        scored = []
        for name in candidates:
            spec = self._specializations.get(name)
            if spec is None:
                continue
            score = self._score_match(task_lower, spec)
            score += spec.priority * 0.001
            if score >= spec.confidence_threshold:
                scored.append((name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def _score_match(self, task_lower: str, spec: AgentSpecialization) -> float:
        """Score how well a task matches an agent's specialization.

        Uses a hybrid approach: the raw ratio of matched keyword weight to total
        keyword weight is boosted so that even a single strong keyword match
        produces a meaningful score.  The formula is:

            raw = matched_weight / total_weight
            score = sqrt(raw)          # boosts small ratios toward threshold

        This ensures that matching 1-2 keywords out of 11 still yields a score
        above the default confidence_threshold of 0.3.
        """
        if not spec.keywords:
            return 0.0

        matched_weight = 0.0
        total_weight = 0.0

        for keyword in spec.keywords:
            weight = spec.capability_weights.get(keyword, 1.0)
            total_weight += weight
            if keyword in task_lower:
                matched_weight += weight

        if total_weight == 0 or matched_weight == 0:
            return 0.0

        raw = matched_weight / total_weight
        # Square-root boost so 2/11 ≈ 0.18 → 0.43 (above 0.3 threshold)
        return raw ** 0.5

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of all registered specializations."""
        return {
            "total_agents": len(self._specializations),
            "agents": {
                name: {
                    "role": spec.role.value,
                    "keywords_count": len(spec.keywords),
                    "confidence_threshold": spec.confidence_threshold,
                    "priority": spec.priority,
                }
                for name, spec in self._specializations.items()
            },
        }
