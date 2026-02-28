"""
Conversation Router — Task analysis and best-fit agent routing.

Analyzes incoming tasks, determines complexity and subtask decomposition,
then routes to the most appropriate agents using the SpecializationRegistry.

Sprint 21 (Multi-Agent Orchestration Enhancement) Module 3.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .agent_specialization import (
    AgentRole,
    AgentSpecialization,
    SpecializationRegistry,
)

logger = logging.getLogger(__name__)


# ── Task Analysis ─────────────────────────────────────────────────


@dataclass
class TaskAnalysis:
    """Result of analyzing a task."""
    keywords: List[str] = field(default_factory=list)
    complexity: str = "simple"  # simple, moderate, complex
    subtask_count: int = 1
    suggested_roles: List[AgentRole] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskAnalyzer:
    """
    Analyzes tasks to extract keywords, estimate complexity,
    and suggest decomposition.

    Usage::

        analyzer = TaskAnalyzer()
        analysis = analyzer.analyze("Write unit tests for the auth module")
        # analysis.keywords = ["write", "unit", "tests", "auth", "module"]
        # analysis.complexity = "moderate"
        # analysis.suggested_roles = [AgentRole.TESTER, AgentRole.CODER]
    """

    # Complexity indicators
    COMPLEX_INDICATORS = [
        "and then", "followed by", "after that", "first.*then",
        "multiple", "several", "comprehensive", "full",
        "end-to-end", "complete", "entire", "all",
    ]
    MODERATE_INDICATORS = [
        "and", "also", "with", "including", "plus",
    ]

    # Role keyword patterns
    ROLE_PATTERNS: Dict[AgentRole, List[str]] = {
        AgentRole.RESEARCHER: ["research", "find", "search", "investigate", "explore"],
        AgentRole.CODER: ["code", "implement", "program", "develop", "build", "script", "fix"],
        AgentRole.REVIEWER: ["review", "audit", "inspect", "check", "evaluate"],
        AgentRole.TESTER: ["test", "qa", "verify", "regression", "coverage"],
        AgentRole.WRITER: ["write", "document", "report", "summarize", "draft"],
        AgentRole.ARCHITECT: ["design", "architect", "plan", "structure", "schema"],
    }

    def analyze(self, task: str) -> TaskAnalysis:
        """Analyze a task and return structured analysis."""
        task_lower = task.lower()
        keywords = self._extract_keywords(task_lower)
        complexity = self._estimate_complexity(task_lower)
        subtask_count = self._estimate_subtasks(task_lower, complexity)
        roles = self._suggest_roles(task_lower)
        confidence = min(1.0, len(keywords) * 0.1 + len(roles) * 0.2)

        return TaskAnalysis(
            keywords=keywords,
            complexity=complexity,
            subtask_count=subtask_count,
            suggested_roles=roles,
            confidence=confidence,
            metadata={
                "word_count": len(task.split()),
                "has_steps": bool(re.search(r'\d+\.\s', task)),
            },
        )

    def _extract_keywords(self, task_lower: str) -> List[str]:
        """Extract meaningful keywords from task text."""
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "about",
            "it", "its", "this", "that", "these", "those", "i", "me", "my",
            "we", "our", "you", "your", "he", "she", "they", "them",
        }
        words = re.findall(r'\b[a-z]+\b', task_lower)
        return [w for w in words if w not in stop_words and len(w) > 2]

    def _estimate_complexity(self, task_lower: str) -> str:
        """Estimate task complexity based on indicators."""
        complex_score = sum(
            1 for pattern in self.COMPLEX_INDICATORS
            if re.search(pattern, task_lower)
        )
        moderate_score = sum(
            1 for pattern in self.MODERATE_INDICATORS
            if pattern in task_lower
        )

        word_count = len(task_lower.split())

        if complex_score >= 2 or word_count > 50:
            return "complex"
        elif complex_score >= 1 or moderate_score >= 2 or word_count > 25:
            return "moderate"
        return "simple"

    def _estimate_subtasks(self, task_lower: str, complexity: str) -> int:
        """Estimate the number of subtasks."""
        if complexity == "complex":
            return max(3, len(re.findall(r'\d+\.\s', task_lower)) or 3)
        elif complexity == "moderate":
            return 2
        return 1

    def _suggest_roles(self, task_lower: str) -> List[AgentRole]:
        """Suggest agent roles based on task content."""
        roles = []
        for role, patterns in self.ROLE_PATTERNS.items():
            if any(p in task_lower for p in patterns):
                roles.append(role)
        return roles or [AgentRole.GENERAL]


# ── Conversation Router ───────────────────────────────────────────


@dataclass
class RoutingDecision:
    """Result of routing a task to agents."""
    assignments: List[Tuple[str, str]]  # [(agent_name, subtask_description)]
    analysis: TaskAnalysis
    strategy_suggestion: str = "sequential"
    fallback_used: bool = False


class ConversationRouter:
    """
    Routes tasks to the best-suited agents using specialization matching.

    Usage::

        router = ConversationRouter(spec_registry=registry)
        decision = router.route_task("Write and test a login module", agents)
        # decision.assignments = [("coder", "Implement login"), ("tester", "Test login")]
    """

    def __init__(
        self,
        spec_registry: Optional[SpecializationRegistry] = None,
        analyzer: Optional[TaskAnalyzer] = None,
        default_strategy: str = "sequential",
    ):
        self._registry = spec_registry or SpecializationRegistry()
        self._analyzer = analyzer or TaskAnalyzer()
        self._default_strategy = default_strategy

    def route_task(
        self,
        task: str,
        available_agents: Optional[List[str]] = None,
    ) -> RoutingDecision:
        """
        Route a task to the best agents.

        Returns a RoutingDecision with agent assignments and strategy suggestion.
        """
        analysis = self._analyzer.analyze(task)
        agents = available_agents or self._registry.list_agents()

        if not agents:
            return RoutingDecision(
                assignments=[],
                analysis=analysis,
                strategy_suggestion=self._default_strategy,
                fallback_used=True,
            )

        # For simple tasks, find one best agent
        if analysis.complexity == "simple":
            best, confidence = self._registry.find_best_agent(task, agents)
            if best:
                return RoutingDecision(
                    assignments=[(best, task)],
                    analysis=analysis,
                    strategy_suggestion="sequential",
                )
            # Fallback to first available
            return RoutingDecision(
                assignments=[(agents[0], task)],
                analysis=analysis,
                strategy_suggestion="sequential",
                fallback_used=True,
            )

        # For moderate/complex tasks, try to match multiple agents
        assignments = []
        top_agents = self._registry.find_top_agents(
            task, top_n=min(analysis.subtask_count, len(agents)), available_agents=agents,
        )

        if top_agents:
            for agent_name, confidence in top_agents:
                assignments.append((agent_name, task))
        else:
            # Fallback: distribute to all available
            for agent_name in agents[:analysis.subtask_count]:
                assignments.append((agent_name, task))

        # Suggest strategy based on complexity
        if analysis.complexity == "complex" and len(assignments) >= 3:
            strategy = "pipeline"
        elif len(assignments) >= 2:
            strategy = "parallel"
        else:
            strategy = "sequential"

        return RoutingDecision(
            assignments=assignments,
            analysis=analysis,
            strategy_suggestion=strategy,
            fallback_used=len(top_agents) == 0,
        )

    def route_with_decomposition(
        self,
        task: str,
        available_agents: Optional[List[str]] = None,
    ) -> RoutingDecision:
        """
        Route task with automatic decomposition into subtasks.

        Tries to split the task into subtasks based on suggested roles,
        then assigns each subtask to the best-matching agent.
        """
        analysis = self._analyzer.analyze(task)
        agents = available_agents or self._registry.list_agents()

        if not agents:
            return RoutingDecision(
                assignments=[],
                analysis=analysis,
                strategy_suggestion=self._default_strategy,
                fallback_used=True,
            )

        # Match each suggested role to an available agent
        assignments = []
        used_agents = set()

        for role in analysis.suggested_roles:
            # Find agents with this role
            role_agents = self._registry.list_by_role(role)
            available_role_agents = [
                a for a in role_agents
                if a in agents and a not in used_agents
            ]

            if available_role_agents:
                agent_name = available_role_agents[0]
                subtask = f"[{role.value}] {task}"
                assignments.append((agent_name, subtask))
                used_agents.add(agent_name)

        # If no role-based assignments, fall back to keyword matching
        if not assignments:
            return self.route_task(task, available_agents)

        strategy = "pipeline" if len(assignments) >= 3 else "parallel" if len(assignments) >= 2 else "sequential"

        return RoutingDecision(
            assignments=assignments,
            analysis=analysis,
            strategy_suggestion=strategy,
            fallback_used=False,
        )
