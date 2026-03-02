"""
Sprint 43 · Task Decomposer
=============================
Breaks complex tasks into sub-tasks with role hints,
dependencies, and execution ordering.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Data ─────────────────────────────────────────────────────────────

@dataclass
class SubTask:
    """A single unit of work within a larger task."""
    id: str
    description: str
    role_hint: str = "coder"
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10
    estimated_complexity: str = "medium"  # simple | medium | complex

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "role_hint": self.role_hint,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "estimated_complexity": self.estimated_complexity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SubTask":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def generate_id() -> str:
        return f"sub_{uuid.uuid4().hex[:8]}"


@dataclass
class DecompositionResult:
    """Result of decomposing a task into sub-tasks."""
    original_task: str
    sub_tasks: List[SubTask] = field(default_factory=list)
    execution_order: List[List[str]] = field(default_factory=list)  # stages of parallel-safe groups
    estimated_total_time: float = 0.0  # seconds

    def to_dict(self) -> dict:
        return {
            "original_task": self.original_task,
            "sub_tasks": [s.to_dict() for s in self.sub_tasks],
            "execution_order": self.execution_order,
            "estimated_total_time": self.estimated_total_time,
        }

    @property
    def task_count(self) -> int:
        return len(self.sub_tasks)

    @property
    def stage_count(self) -> int:
        return len(self.execution_order)


# ── Templates ────────────────────────────────────────────────────────

# Each template: (pattern_keywords, list of (description, role_hint, complexity))
DECOMPOSITION_TEMPLATES: Dict[str, List[Tuple[str, str, str]]] = {
    "build": [
        ("Research existing solutions and patterns", "researcher", "simple"),
        ("Plan architecture and approach", "planner", "medium"),
        ("Implement the solution", "coder", "complex"),
        ("Write tests for the implementation", "tester", "medium"),
        ("Review code quality and correctness", "reviewer", "simple"),
    ],
    "fix_bug": [
        ("Research and read relevant code", "researcher", "simple"),
        ("Diagnose the root cause", "coder", "medium"),
        ("Implement the fix", "coder", "medium"),
        ("Write regression tests", "tester", "simple"),
    ],
    "refactor": [
        ("Analyze current code structure", "reviewer", "medium"),
        ("Plan refactoring changes", "planner", "medium"),
        ("Implement refactored code", "coder", "complex"),
        ("Run and update tests", "tester", "medium"),
        ("Review refactored code", "reviewer", "simple"),
    ],
    "test": [
        ("Analyze code to determine test coverage", "reviewer", "simple"),
        ("Plan test strategy and cases", "planner", "medium"),
        ("Write test implementations", "tester", "complex"),
        ("Run tests and verify", "tester", "simple"),
    ],
    "document": [
        ("Research the codebase and features", "researcher", "medium"),
        ("Plan documentation structure", "planner", "simple"),
        ("Write documentation", "coder", "medium"),
        ("Review documentation for accuracy", "reviewer", "simple"),
    ],
    "deploy": [
        ("Review deployment requirements", "reviewer", "simple"),
        ("Plan deployment steps", "planner", "medium"),
        ("Prepare deployment artifacts", "coder", "medium"),
        ("Verify deployment", "tester", "simple"),
    ],
}

# Keywords that map to templates
_TEMPLATE_KEYWORDS: Dict[str, List[str]] = {
    "build": ["build", "create", "implement", "add", "develop", "make", "new feature"],
    "fix_bug": ["fix", "bug", "error", "broken", "crash", "issue", "problem", "debug"],
    "refactor": ["refactor", "restructure", "reorganize", "clean up", "improve", "optimize"],
    "test": ["test", "coverage", "spec", "verify", "validate"],
    "document": ["document", "docs", "readme", "wiki", "guide", "tutorial"],
    "deploy": ["deploy", "release", "publish", "ship", "launch"],
}

# Complexity → estimated time in seconds
_COMPLEXITY_TIME = {
    "simple": 30.0,
    "medium": 60.0,
    "complex": 120.0,
}


# ── Decomposer ──────────────────────────────────────────────────────

class TaskDecomposer:
    """
    Decomposes a task description into sub-tasks with dependencies
    and execution ordering.
    """

    def __init__(self):
        self._custom_templates: Dict[str, List[Tuple[str, str, str]]] = {}

    def decompose(self, task_description: str) -> DecompositionResult:
        """
        Decompose a task into sub-tasks using template matching.
        Falls back to a generic decomposition if no template matches.
        """
        template_name = self._match_template(task_description)
        template = (
            self._custom_templates.get(template_name)
            or DECOMPOSITION_TEMPLATES.get(template_name)
        )

        if template is None:
            template = DECOMPOSITION_TEMPLATES["build"]  # sensible default

        # Build sub-tasks
        sub_tasks = []
        for i, (desc, role, complexity) in enumerate(template):
            # Contextualize description with original task
            contextualized = f"{desc} for: {task_description}"
            st = SubTask(
                id=SubTask.generate_id(),
                description=contextualized,
                role_hint=role,
                priority=10 - i,  # earlier steps = higher priority
                estimated_complexity=complexity,
            )
            sub_tasks.append(st)

        # Build dependencies (each step depends on previous)
        self._infer_dependencies(sub_tasks)

        # Build execution order (stages)
        execution_order = self._build_execution_order(sub_tasks)

        # Estimate time
        total_time = sum(
            _COMPLEXITY_TIME.get(st.estimated_complexity, 60.0)
            for st in sub_tasks
        )

        return DecompositionResult(
            original_task=task_description,
            sub_tasks=sub_tasks,
            execution_order=execution_order,
            estimated_total_time=total_time,
        )

    def register_template(
        self,
        name: str,
        steps: List[Tuple[str, str, str]],
    ) -> None:
        """Register a custom decomposition template."""
        self._custom_templates[name] = steps

    # ── helpers ──────────────────────────────────────────────────

    def _match_template(self, description: str) -> Optional[str]:
        """Find the best matching template by keyword scoring."""
        desc_lower = description.lower()
        scores: Dict[str, int] = {}

        for tmpl_name, keywords in _TEMPLATE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            if score > 0:
                scores[tmpl_name] = score

        if scores:
            return max(scores, key=scores.get)
        return None

    @staticmethod
    def _infer_dependencies(sub_tasks: List[SubTask]) -> None:
        """
        Set sequential dependencies: each step depends on the previous one.
        This is a conservative default. Parallel stages are computed from
        the dependency graph in _build_execution_order.
        """
        for i in range(1, len(sub_tasks)):
            sub_tasks[i].dependencies = [sub_tasks[i - 1].id]

    @staticmethod
    def _build_execution_order(sub_tasks: List[SubTask]) -> List[List[str]]:
        """
        Group sub-tasks into stages of parallel-safe groups.
        Tasks with no unmet dependencies go in the same stage.
        """
        if not sub_tasks:
            return []

        remaining = {st.id: st for st in sub_tasks}
        completed: set = set()
        stages: List[List[str]] = []

        while remaining:
            # Find tasks whose dependencies are all completed
            ready = [
                st_id for st_id, st in remaining.items()
                if all(dep in completed for dep in st.dependencies)
            ]
            if not ready:
                # Circular dependency or error — just dump remaining
                ready = list(remaining.keys())

            stages.append(ready)
            for st_id in ready:
                completed.add(st_id)
                del remaining[st_id]

        return stages
