"""
System Prompt Builder — composes the full system prompt from modular sections.
Mirrors Cowork's approach of structured sections with tool schemas embedded.
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import Optional

from .models import ToolSchema
from ..prompts.behavioral_rules import (
    CORE_IDENTITY,
    TOOL_USAGE_RULES,
    FILE_HANDLING_RULES,
    SAFETY_RULES,
    TODO_RULES,
)


class PromptBuilder:
    """Build system prompts from modular sections."""

    def __init__(self, config: dict):
        self.config = config
        self.workspace_dir = config.get("agent", {}).get("workspace_dir", "./workspace")

    def build(
        self,
        tools: list[ToolSchema],
        context: Optional[dict] = None,
    ) -> str:
        """
        Assemble the complete system prompt.

        Args:
            tools: Available tool schemas
            context: Runtime context (iteration count, todos, etc.)

        Returns:
            Complete system prompt string
        """
        ctx = context or {}
        sections = [
            self._section_identity(),
            self._section_tool_schemas(tools),
            self._section_behavioral_rules(),
            self._section_runtime_context(ctx),
        ]
        return "\n\n".join(sections)

    def _section_identity(self) -> str:
        """Core identity and capabilities."""
        return CORE_IDENTITY

    def _section_tool_schemas(self, tools: list[ToolSchema]) -> str:
        """Tool definitions section — tells the LLM what tools are available."""
        if not tools:
            return ""

        tool_list = []
        for t in tools:
            tool_list.append(
                f"### {t.name}\n"
                f"{t.description}\n\n"
                f"Parameters:\n```json\n{json.dumps(t.input_schema, indent=2)}\n```"
            )

        return "## Available Tools\n\n" + "\n\n---\n\n".join(tool_list)

    def _section_behavioral_rules(self) -> str:
        """All behavioral instructions."""
        return "\n\n".join([
            TOOL_USAGE_RULES,
            FILE_HANDLING_RULES,
            SAFETY_RULES,
            TODO_RULES,
        ])

    def _section_runtime_context(self, context: dict) -> str:
        """Dynamic runtime context for this session."""
        now = datetime.now()
        lines = [
            "## Current Context",
            f"- Date: {now.strftime('%A, %B %d, %Y')}",
            f"- Time: {now.strftime('%I:%M %p')}",
            f"- Workspace: {self.workspace_dir}",
        ]

        if context.get("iteration"):
            lines.append(f"- Agent iteration: {context['iteration']}")

        todos = context.get("todos", [])
        if todos:
            pending = sum(1 for t in todos if t.get("status") == "pending")
            in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
            completed = sum(1 for t in todos if t.get("status") == "completed")
            lines.append(f"- Tasks: {completed} done, {in_progress} active, {pending} pending")

        return "\n".join(lines)
