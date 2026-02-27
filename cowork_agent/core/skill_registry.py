"""
Skill Registry — Discovers, loads, and injects SKILL.md files into the system prompt.

Mirrors real Cowork's skill system:
  - Skills are directories containing a SKILL.md file
  - Each skill has a name, description, and trigger conditions
  - Skills are loaded before document creation tasks for better quality
  - Skills can be auto-detected from user intent or explicitly requested

Skill discovery locations:
  1. workspace/.skills/skills/   (project-specific skills)
  2. ~/.cowork_agent/skills/     (user global skills)
  3. Built-in default skills     (shipped with the agent)
"""

from __future__ import annotations
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A loaded skill with its metadata and instructions."""
    name: str
    description: str
    location: str  # directory path
    content: str  # raw SKILL.md content
    triggers: list[str] = field(default_factory=list)  # keywords that activate this skill

    @property
    def skill_md_path(self) -> str:
        return os.path.join(self.location, "SKILL.md")


class SkillRegistry:
    """
    Discovers and manages SKILL.md files.

    Usage:
        registry = SkillRegistry(workspace_dir="/path/to/workspace")
        registry.discover()  # scan for skills

        # Auto-detect relevant skills from user message
        relevant = registry.match_skills("create a powerpoint presentation")
        # Returns: [Skill(name="pptx", ...)]

        # Get skill content for system prompt injection
        for skill in relevant:
            prompt += skill.content
    """

    def __init__(self, workspace_dir: str = "", user_skills_dir: str = ""):
        self.workspace_dir = workspace_dir
        self.user_skills_dir = user_skills_dir or os.path.expanduser("~/.cowork_agent/skills")
        self._skills: dict[str, Skill] = {}  # name -> Skill

    @property
    def skills(self) -> dict[str, Skill]:
        return dict(self._skills)

    @property
    def skill_names(self) -> list[str]:
        return list(self._skills.keys())

    def discover(self) -> int:
        """
        Scan all skill directories and load SKILL.md files.
        Returns the number of skills discovered.
        """
        self._skills.clear()
        count = 0

        # 1. Workspace skills (highest priority)
        workspace_skills_dir = os.path.join(self.workspace_dir, ".skills", "skills")
        count += self._scan_directory(workspace_skills_dir)

        # 2. User global skills
        count += self._scan_directory(self.user_skills_dir)

        logger.info(f"Discovered {count} skills: {self.skill_names}")
        return count

    def _scan_directory(self, base_dir: str) -> int:
        """Scan a directory for skill subdirectories containing SKILL.md."""
        if not os.path.isdir(base_dir):
            return 0

        count = 0
        for entry in sorted(os.listdir(base_dir)):
            skill_dir = os.path.join(base_dir, entry)
            skill_md = os.path.join(skill_dir, "SKILL.md")

            if os.path.isdir(skill_dir) and os.path.isfile(skill_md):
                try:
                    skill = self._load_skill(entry, skill_dir, skill_md)
                    if skill:
                        # Don't overwrite if already loaded (workspace takes priority)
                        if skill.name not in self._skills:
                            self._skills[skill.name] = skill
                            count += 1
                            logger.debug(f"Loaded skill: {skill.name} from {skill_dir}")
                except Exception as e:
                    logger.warning(f"Failed to load skill from {skill_dir}: {e}")

        return count

    # SEC-HIGH-4: Max SKILL.md size to prevent oversized prompt injection
    MAX_SKILL_SIZE = 50_000  # 50KB — generous but bounded

    def _load_skill(self, name: str, directory: str, skill_md_path: str) -> Optional[Skill]:
        """Load a single SKILL.md file and extract metadata."""
        # Check file size before reading
        file_size = os.path.getsize(skill_md_path)
        if file_size > self.MAX_SKILL_SIZE:
            logger.warning(
                f"Skill '{name}' SKILL.md is too large ({file_size} bytes, "
                f"max {self.MAX_SKILL_SIZE}). Skipping."
            )
            return None

        with open(skill_md_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return None

        # Extract description from frontmatter or first paragraph
        description = self._extract_description(content)

        # Extract trigger keywords from the skill content
        triggers = self._extract_triggers(name, content)

        return Skill(
            name=name,
            description=description,
            location=directory,
            content=content,
            triggers=triggers,
        )

    @staticmethod
    def _extract_description(content: str) -> str:
        """Extract a description from SKILL.md content."""
        # Try frontmatter-style description
        match = re.search(r'description:\s*["\']?(.+?)["\']?\s*$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Fall back to first non-empty, non-heading line
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("---"):
                return line[:200]

        return "No description available."

    @staticmethod
    def _extract_triggers(name: str, content: str) -> list[str]:
        """
        Extract trigger keywords from skill content.
        Looks for explicit trigger lists or infers from name/content.
        """
        triggers = [name]

        # Look for explicit MANDATORY TRIGGERS pattern
        match = re.search(
            r'MANDATORY\s+TRIGGERS?:\s*(.+?)(?:\n|$)',
            content, re.IGNORECASE
        )
        if match:
            raw = match.group(1)
            triggers.extend(
                t.strip().lower()
                for t in re.split(r'[,;|]', raw)
                if t.strip()
            )

        # Common file extension triggers by skill name
        extension_map = {
            "docx": ["word", "document", ".docx", "report", "letter", "memo"],
            "pptx": ["powerpoint", "presentation", ".pptx", "slides", "deck", "pitch"],
            "xlsx": ["excel", "spreadsheet", ".xlsx", "data table", "budget", "chart"],
            "pdf": ["pdf", ".pdf", "form", "extract", "merge", "split"],
            "remotion": ["video", "animation", "render", "mp4"],
        }
        if name in extension_map:
            triggers.extend(extension_map[name])

        # Deduplicate
        return list(set(t.lower() for t in triggers if t))

    def match_skills(self, user_message: str) -> list[Skill]:
        """
        Find skills that match the user's message based on trigger keywords.

        Returns a list of matching skills, sorted by relevance (number of
        trigger matches).
        """
        msg_lower = user_message.lower()
        scored: list[tuple[int, Skill]] = []

        for skill in self._skills.values():
            score = sum(1 for trigger in skill.triggers if trigger in msg_lower)
            if score > 0:
                scored.append((score, skill))

        # Sort by score (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored]

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_skill_prompt_section(self, skills: list[Skill]) -> str:
        """
        Format matched skills as an XML section for system prompt injection.
        Mirrors real Cowork's <skills_instructions> + <available_skills> format.
        """
        if not skills:
            return ""

        lines = [
            "<skills_instructions>",
            "The following skills are loaded and provide specialized instructions.",
            "Follow their guidance carefully for best results.",
            "</skills_instructions>",
            "",
        ]

        for skill in skills:
            lines.append(f"<skill name=\"{skill.name}\">")
            lines.append(skill.content)
            lines.append(f"</skill>")
            lines.append("")

        return "\n".join(lines)

    def get_available_skills_section(self) -> str:
        """
        Format all discovered skills as an <available_skills> section.
        This goes in the system prompt so the LLM knows what skills exist.
        """
        if not self._skills:
            return ""

        lines = [
            "<available_skills>",
        ]

        for skill in self._skills.values():
            lines.append(f"<skill>")
            lines.append(f"<name>{skill.name}</name>")
            lines.append(f"<description>{skill.description}</description>")
            lines.append(f"<location>{skill.location}</location>")
            lines.append(f"</skill>")

        lines.append("</available_skills>")
        return "\n".join(lines)
