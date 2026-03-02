"""
Sprint 43 · Crew Roles
========================
Predefined and custom roles for multi-agent crew mode.
Each role specifies a system prompt addon, allowed tools, and iteration limits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Data ─────────────────────────────────────────────────────────────

@dataclass
class CrewRole:
    """Definition of a specialized agent role within a crew."""
    name: str
    description: str
    system_prompt_addon: str
    allowed_tools: List[str] = field(default_factory=list)
    max_iterations: int = 10

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt_addon": self.system_prompt_addon,
            "allowed_tools": self.allowed_tools,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CrewRole":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Predefined roles ────────────────────────────────────────────────

PREDEFINED_ROLES: Dict[str, CrewRole] = {
    "researcher": CrewRole(
        name="researcher",
        description="Gathers information from the web and codebase",
        system_prompt_addon=(
            "You are a research specialist. Focus on gathering accurate, "
            "comprehensive information. Cite sources when possible. "
            "Prioritize breadth of coverage, then depth on key findings."
        ),
        allowed_tools=["web_search", "web_fetch", "read", "grep", "glob"],
        max_iterations=12,
    ),
    "coder": CrewRole(
        name="coder",
        description="Writes and modifies code",
        system_prompt_addon=(
            "You are a coding specialist. Write clean, well-documented code. "
            "Follow existing project conventions. Include error handling and "
            "type hints where appropriate."
        ),
        allowed_tools=["read", "write", "edit", "bash", "glob", "grep"],
        max_iterations=15,
    ),
    "reviewer": CrewRole(
        name="reviewer",
        description="Reviews code for quality and correctness",
        system_prompt_addon=(
            "You are a code reviewer. Examine code for bugs, security issues, "
            "performance problems, and style violations. Provide specific, "
            "actionable feedback with line references."
        ),
        allowed_tools=["read", "grep", "glob", "bash"],
        max_iterations=8,
    ),
    "tester": CrewRole(
        name="tester",
        description="Writes and runs tests",
        system_prompt_addon=(
            "You are a test engineer. Write comprehensive tests covering "
            "happy paths, edge cases, and error scenarios. Use the project's "
            "existing test framework and patterns."
        ),
        allowed_tools=["read", "write", "bash", "glob"],
        max_iterations=12,
    ),
    "planner": CrewRole(
        name="planner",
        description="Plans and decomposes complex tasks",
        system_prompt_addon=(
            "You are a planning specialist. Break down complex tasks into "
            "clear, actionable steps. Consider dependencies, risks, and "
            "alternative approaches."
        ),
        allowed_tools=["read", "glob", "grep"],
        max_iterations=6,
    ),
}


# ── Role keywords for auto-assignment ───────────────────────────────

_ROLE_KEYWORDS: Dict[str, List[str]] = {
    "researcher": ["research", "find", "search", "gather", "investigate",
                    "look up", "discover", "information", "explore", "survey"],
    "coder": ["write", "code", "implement", "build", "create", "develop",
              "add", "feature", "function", "class", "module", "fix", "bug",
              "refactor", "modify", "update"],
    "reviewer": ["review", "check", "audit", "inspect", "examine",
                  "evaluate", "assess", "quality", "critique"],
    "tester": ["test", "verify", "validate", "spec", "coverage",
               "unit test", "integration test", "assert", "expect"],
    "planner": ["plan", "design", "architect", "organize", "structure",
                "decompose", "strategy", "roadmap", "outline"],
}


# ── Role Assigner ───────────────────────────────────────────────────

class RoleAssigner:
    """Assigns roles to sub-tasks based on keyword matching."""

    def __init__(self):
        self._custom_roles: Dict[str, CrewRole] = {}

    def assign_role(self, sub_task_description: str) -> CrewRole:
        """
        Pick the best role for a sub-task description using keyword matching.
        Falls back to 'coder' if no strong match.
        """
        desc_lower = sub_task_description.lower()
        scores: Dict[str, int] = {}

        for role_name, keywords in _ROLE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            if score > 0:
                scores[role_name] = score

        if scores:
            best = max(scores, key=scores.get)
            return self.get_role(best)

        # Default to coder
        return self.get_role("coder")

    def get_role(self, name: str) -> CrewRole:
        """Retrieve a role by name (custom first, then predefined)."""
        if name in self._custom_roles:
            return self._custom_roles[name]
        if name in PREDEFINED_ROLES:
            return PREDEFINED_ROLES[name]
        raise KeyError(f"Unknown role: {name}")

    def register_custom_role(self, role: CrewRole) -> None:
        """Register a user-defined role."""
        self._custom_roles[role.name] = role

    def list_roles(self) -> List[str]:
        """List all available role names."""
        names = list(PREDEFINED_ROLES.keys())
        for n in self._custom_roles:
            if n not in names:
                names.append(n)
        return names

    @property
    def custom_roles(self) -> Dict[str, CrewRole]:
        return dict(self._custom_roles)
