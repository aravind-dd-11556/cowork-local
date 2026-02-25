"""
System Prompt Builder — composes the full system prompt from modular XML-tagged sections.

Mirrors the real Cowork approach:
  - XML tags for structural boundaries (<env>, <tools>, <claude_behavior>, etc.)
  - Dynamic runtime context injection (date, time, workspace, todos)
  - Tool schemas with full JSON descriptions
  - Behavioral rules as nested XML sections
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import Optional

from .models import ToolSchema
from ..prompts.behavioral_rules import ALL_SECTIONS


class PromptBuilder:
    """
    Build XML-tagged system prompts matching the real Cowork structure.

    The real Cowork prompt follows this layout:
        <application_details> ... </application_details>
        <claude_behavior> ... </claude_behavior>
        <env> ... </env>
        <tools> ... (full JSON schemas) </tools>
        <file_handling_rules> ... </file_handling_rules>
        <critical_security_rules> ... </critical_security_rules>
        <todo_list_tool> ... </todo_list_tool>
        ... (more XML sections)
    """

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
            Complete XML-tagged system prompt string
        """
        ctx = context or {}

        sections = []

        # 1. All behavioral rule sections (already XML-tagged)
        for section in ALL_SECTIONS:
            sections.append(section.strip())

        # 2. User context
        sections.append(self._section_user())

        # 3. Environment / runtime context
        sections.append(self._section_env(ctx))

        # 4. Available tools with full schemas
        tool_section = self._section_tools(tools)
        if tool_section:
            sections.append(tool_section)

        # 5. Runtime reminder (active todos, iteration info)
        reminder = self._section_system_reminder(ctx)
        if reminder:
            sections.append(reminder)

        return "\n\n".join(sections)

    def _section_user(self) -> str:
        """User identity section — mirrors Cowork's <user> tag."""
        user_name = self.config.get("user", {}).get("name", "User")
        user_email = self.config.get("user", {}).get("email", "")

        lines = [f"<user>", f"Name: {user_name}"]
        if user_email:
            lines.append(f"Email: {user_email}")
        lines.append("</user>")
        return "\n".join(lines)

    def _section_env(self, context: dict) -> str:
        """
        Dynamic environment section — mirrors Cowork's <env> tag.

        This is rebuilt fresh on every prompt build so the date/time
        is always current (same approach as real Cowork).
        """
        now = datetime.now()
        model = self.config.get("llm", {}).get("model", "unknown")
        provider = self.config.get("llm", {}).get("provider", "unknown")

        lines = [
            "<env>",
            f"Today's date: {now.strftime('%A, %B %d, %Y')}",
            f"Current time: {now.strftime('%I:%M %p')}",
            f"Model: {model}",
            f"Provider: {provider}",
            f"Workspace: {self.workspace_dir}",
        ]

        if context.get("iteration"):
            lines.append(f"Agent iteration: {context['iteration']} of 15")

        lines.append("</env>")
        return "\n".join(lines)

    def _section_tools(self, tools: list[ToolSchema]) -> str:
        """
        Tool definitions section with full JSON schemas.

        Each tool gets its own block with description and parameter schema,
        similar to how Cowork's system prompt defines each tool as a
        <function> block with a full JSON schema.
        """
        if not tools:
            return ""

        tool_blocks = []
        for t in tools:
            schema_json = json.dumps(t.input_schema, indent=2)
            block = (
                f"<tool>\n"
                f"<name>{t.name}</name>\n"
                f"<description>\n{t.description}\n</description>\n"
                f"<parameters>\n{schema_json}\n</parameters>\n"
                f"</tool>"
            )
            tool_blocks.append(block)

        return (
            "<available_tools>\n"
            "You have access to the following tools. Use them when needed to complete tasks.\n\n"
            + "\n\n".join(tool_blocks)
            + "\n</available_tools>"
        )

    def _section_system_reminder(self, context: dict) -> str:
        """
        System reminder section — mirrors Cowork's <system-reminder> tag.

        Injected at the end of the prompt with current state like active
        todos. This is refreshed every iteration so the LLM always sees
        up-to-date task status.
        """
        parts = []

        # Active todos reminder
        todos = context.get("todos", [])
        if todos:
            pending = [t for t in todos if t.get("status") == "pending"]
            in_progress = [t for t in todos if t.get("status") == "in_progress"]
            completed = [t for t in todos if t.get("status") == "completed"]

            todo_lines = ["Current task status:"]
            for t in in_progress:
                todo_lines.append(f"  🔄 IN PROGRESS: {t.get('activeForm', t['content'])}")
            for t in pending:
                todo_lines.append(f"  ⬜ PENDING: {t['content']}")
            for t in completed:
                todo_lines.append(f"  ✅ DONE: {t['content']}")

            parts.append("\n".join(todo_lines))

        if not parts:
            return ""

        return (
            "<system-reminder>\n"
            + "\n\n".join(parts)
            + "\n</system-reminder>"
        )
