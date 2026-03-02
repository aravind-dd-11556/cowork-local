"""
Sprint 29: Skill Invocation Tool — lets the agent explicitly invoke a skill by name.

Mirrors real Cowork's `Skill` tool which:
  - Takes a skill name (and optional args)
  - Loads the SKILL.md content
  - Returns it so the LLM follows those instructions

This complements the auto-matching in skill_registry (which triggers on keywords).
The Skill tool gives the LLM *explicit* control: "I know I need the docx skill."
"""

from __future__ import annotations

import logging
from typing import Optional, Callable

from .base import BaseTool
from ..core.models import ToolResult
from ..core.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


class SkillTool(BaseTool):
    """
    Invoke a skill by name, loading its SKILL.md instructions.

    Usage by the LLM:
        {"skill": "docx"}                  → load Word document skill
        {"skill": "pptx", "args": ""}      → load PowerPoint skill
        {"skill": "pdf"}                   → load PDF skill
    """

    name = "skill"
    description = (
        "Invoke a skill by name to load specialized instructions for a task. "
        "Skills provide best-practice guides for creating documents (docx, pptx, "
        "xlsx, pdf), videos (remotion), scheduling tasks (schedule), or creating "
        "new skills (skill-creator). Call this tool BEFORE starting work that "
        "matches a skill. Use the skill name as the 'skill' parameter."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": (
                    "The skill name to invoke (e.g., 'docx', 'pptx', 'xlsx', "
                    "'pdf', 'remotion', 'skill-creator', 'schedule'). "
                    "Must match a discovered skill name."
                ),
            },
            "args": {
                "type": "string",
                "description": "Optional arguments for the skill.",
                "default": "",
            },
        },
        "required": ["skill"],
    }

    def __init__(self, skill_registry: Optional[SkillRegistry] = None):
        self._skill_registry = skill_registry

    @property
    def skill_registry(self) -> Optional[SkillRegistry]:
        return self._skill_registry

    @skill_registry.setter
    def skill_registry(self, value: SkillRegistry) -> None:
        self._skill_registry = value

    async def execute(
        self,
        *,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        skill: str = "",
        args: str = "",
        **kwargs,
    ) -> ToolResult:
        if not skill:
            return self._error("'skill' parameter is required. Provide the skill name.")

        if not self._skill_registry:
            return self._error(
                "No skill registry available. Skills have not been configured."
            )

        # Normalize the skill name
        skill_name = skill.strip().lower()

        # Handle fully-qualified names like "anthropic-skills:docx" or "ms-office-suite:pdf"
        if ":" in skill_name:
            skill_name = skill_name.split(":")[-1]

        # Look up the skill
        skill_obj = self._skill_registry.get_skill(skill_name)
        if not skill_obj:
            available = self._skill_registry.skill_names
            if available:
                return self._error(
                    f"Skill '{skill_name}' not found. "
                    f"Available skills: {', '.join(sorted(available))}"
                )
            return self._error(
                f"Skill '{skill_name}' not found. No skills have been discovered. "
                "Ensure SKILL.md files exist in the skills directory."
            )

        # Build the response with skill content
        lines = [
            f'The "{skill_obj.name}" skill is loaded.',
            "",
            f"<skill_instructions name=\"{skill_obj.name}\">",
            skill_obj.content,
            "</skill_instructions>",
        ]

        if args:
            lines.insert(1, f"Arguments: {args}")

        logger.info(f"Skill invoked: {skill_obj.name} (from {skill_obj.location})")

        return self._success(
            "\n".join(lines),
            skill_name=skill_obj.name,
            skill_location=skill_obj.location,
        )
