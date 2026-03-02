"""
Sprint 29 — Skill Tool + Skill Content + Skill-Before-Work Enforcement

Tests:
  1. SkillTool — invoke by name, error cases, qualified names
  2. SKILL.md files — all 7 exist with proper structure
  3. SkillRegistry — built-in skills discovery
  4. Agent integration — enforcement hint injected
  5. PromptBuilder — folder-selected, skill hint in env
  6. Edge cases — empty names, missing registry, large content
"""

import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch

# Helpers ----------------------------------------------------------------

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", "skills")
SKILL_NAMES = ["docx", "pptx", "xlsx", "pdf", "remotion", "skill-creator", "schedule"]


# ── 1. SkillTool Tests ──────────────────────────────────────────────────

class TestSkillToolBasic(unittest.TestCase):
    """Core SkillTool functionality."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool

        self.registry = SkillRegistry(workspace_dir="")
        # Load built-in skills
        self.registry._scan_directory(os.path.abspath(SKILLS_DIR))
        self.tool = SkillTool(skill_registry=self.registry)

    def test_invoke_docx(self):
        result = _run(self.tool.execute(skill="docx"))
        self.assertTrue(result.success)
        self.assertIn("docx", result.output)
        self.assertIn("skill_instructions", result.output)

    def test_invoke_pptx(self):
        result = _run(self.tool.execute(skill="pptx"))
        self.assertTrue(result.success)
        self.assertIn("PowerPoint", result.output)

    def test_invoke_xlsx(self):
        result = _run(self.tool.execute(skill="xlsx"))
        self.assertTrue(result.success)
        self.assertIn("Excel", result.output)

    def test_invoke_pdf(self):
        result = _run(self.tool.execute(skill="pdf"))
        self.assertTrue(result.success)
        self.assertIn("PDF", result.output)

    def test_invoke_remotion(self):
        result = _run(self.tool.execute(skill="remotion"))
        self.assertTrue(result.success)
        self.assertIn("Remotion", result.output)

    def test_invoke_skill_creator(self):
        result = _run(self.tool.execute(skill="skill-creator"))
        self.assertTrue(result.success)
        self.assertIn("Skill Creator", result.output)

    def test_invoke_schedule(self):
        result = _run(self.tool.execute(skill="schedule"))
        self.assertTrue(result.success)
        self.assertIn("Schedule", result.output)

    def test_skill_loaded_message(self):
        result = _run(self.tool.execute(skill="docx"))
        self.assertIn('The "docx" skill is loaded.', result.output)

    def test_metadata_in_result(self):
        result = _run(self.tool.execute(skill="pdf"))
        self.assertEqual(result.metadata.get("skill_name"), "pdf")
        self.assertTrue(result.metadata.get("skill_location"))


class TestSkillToolErrors(unittest.TestCase):
    """Error handling in SkillTool."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool

        self.registry = SkillRegistry(workspace_dir="")
        self.registry._scan_directory(os.path.abspath(SKILLS_DIR))
        self.tool = SkillTool(skill_registry=self.registry)

    def test_empty_skill_name(self):
        result = _run(self.tool.execute(skill=""))
        self.assertFalse(result.success)
        self.assertIn("required", result.error)

    def test_unknown_skill(self):
        result = _run(self.tool.execute(skill="nonexistent"))
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)
        # Should list available skills
        self.assertIn("Available skills", result.error)

    def test_no_registry(self):
        from cowork_agent.tools.skill_tool import SkillTool
        tool = SkillTool(skill_registry=None)
        result = _run(tool.execute(skill="docx"))
        self.assertFalse(result.success)
        self.assertIn("No skill registry", result.error)

    def test_empty_registry(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool

        empty_reg = SkillRegistry(workspace_dir="")
        tool = SkillTool(skill_registry=empty_reg)
        result = _run(tool.execute(skill="docx"))
        self.assertFalse(result.success)
        self.assertIn("No skills have been discovered", result.error)


class TestSkillToolQualifiedNames(unittest.TestCase):
    """Fully qualified skill names like 'anthropic-skills:docx'."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool

        self.registry = SkillRegistry(workspace_dir="")
        self.registry._scan_directory(os.path.abspath(SKILLS_DIR))
        self.tool = SkillTool(skill_registry=self.registry)

    def test_colon_prefix_stripped(self):
        result = _run(self.tool.execute(skill="anthropic-skills:docx"))
        self.assertTrue(result.success)
        self.assertIn("docx", result.output)

    def test_custom_prefix_stripped(self):
        result = _run(self.tool.execute(skill="ms-office-suite:xlsx"))
        self.assertTrue(result.success)
        self.assertIn("xlsx", result.output)

    def test_multiple_colons(self):
        result = _run(self.tool.execute(skill="a:b:pdf"))
        self.assertTrue(result.success)
        self.assertIn("PDF", result.output)


class TestSkillToolArgs(unittest.TestCase):
    """Skill tool with args parameter."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool

        self.registry = SkillRegistry(workspace_dir="")
        self.registry._scan_directory(os.path.abspath(SKILLS_DIR))
        self.tool = SkillTool(skill_registry=self.registry)

    def test_args_included_in_output(self):
        result = _run(self.tool.execute(skill="docx", args="-m 'Fix bug'"))
        self.assertTrue(result.success)
        self.assertIn("Arguments: -m 'Fix bug'", result.output)

    def test_empty_args_not_included(self):
        result = _run(self.tool.execute(skill="docx", args=""))
        self.assertTrue(result.success)
        self.assertNotIn("Arguments:", result.output)


class TestSkillToolSchema(unittest.TestCase):
    """Skill tool schema validation."""

    def test_name(self):
        from cowork_agent.tools.skill_tool import SkillTool
        tool = SkillTool()
        self.assertEqual(tool.name, "skill")

    def test_description_nonempty(self):
        from cowork_agent.tools.skill_tool import SkillTool
        tool = SkillTool()
        self.assertTrue(len(tool.description) > 20)

    def test_schema_has_skill_param(self):
        from cowork_agent.tools.skill_tool import SkillTool
        tool = SkillTool()
        self.assertIn("skill", tool.input_schema["properties"])
        self.assertIn("skill", tool.input_schema["required"])

    def test_schema_has_args_param(self):
        from cowork_agent.tools.skill_tool import SkillTool
        tool = SkillTool()
        self.assertIn("args", tool.input_schema["properties"])

    def test_get_schema(self):
        from cowork_agent.tools.skill_tool import SkillTool
        tool = SkillTool()
        schema = tool.get_schema()
        self.assertEqual(schema.name, "skill")


class TestSkillToolRegistryProperty(unittest.TestCase):
    """Skill registry getter/setter."""

    def test_setter(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool

        tool = SkillTool(skill_registry=None)
        self.assertIsNone(tool.skill_registry)

        reg = SkillRegistry(workspace_dir="")
        tool.skill_registry = reg
        self.assertIs(tool.skill_registry, reg)


# ── 2. SKILL.md File Tests ──────────────────────────────────────────────

class TestSkillMdFiles(unittest.TestCase):
    """Verify all 7 SKILL.md files exist and have proper structure."""

    def test_skills_directory_exists(self):
        self.assertTrue(os.path.isdir(os.path.abspath(SKILLS_DIR)))

    def test_all_7_skill_dirs_exist(self):
        for name in SKILL_NAMES:
            path = os.path.join(os.path.abspath(SKILLS_DIR), name)
            self.assertTrue(os.path.isdir(path), f"Missing skill dir: {name}")

    def test_all_7_skill_md_files_exist(self):
        for name in SKILL_NAMES:
            path = os.path.join(os.path.abspath(SKILLS_DIR), name, "SKILL.md")
            self.assertTrue(os.path.isfile(path), f"Missing SKILL.md: {name}")

    def test_all_have_frontmatter(self):
        for name in SKILL_NAMES:
            path = os.path.join(os.path.abspath(SKILLS_DIR), name, "SKILL.md")
            with open(path) as f:
                content = f.read()
            self.assertTrue(content.startswith("---"), f"{name}: missing frontmatter")
            # Must have closing ---
            second_dash = content.index("---", 3)
            self.assertGreater(second_dash, 3, f"{name}: unclosed frontmatter")

    def test_all_have_name_in_frontmatter(self):
        for name in SKILL_NAMES:
            path = os.path.join(os.path.abspath(SKILLS_DIR), name, "SKILL.md")
            with open(path) as f:
                content = f.read()
            self.assertIn("name:", content, f"{name}: missing name in frontmatter")

    def test_all_have_description_in_frontmatter(self):
        for name in SKILL_NAMES:
            path = os.path.join(os.path.abspath(SKILLS_DIR), name, "SKILL.md")
            with open(path) as f:
                content = f.read()
            self.assertIn("description:", content, f"{name}: missing description")

    def test_all_have_mandatory_triggers(self):
        for name in SKILL_NAMES:
            path = os.path.join(os.path.abspath(SKILLS_DIR), name, "SKILL.md")
            with open(path) as f:
                content = f.read()
            self.assertIn("MANDATORY TRIGGERS:", content, f"{name}: missing triggers")

    def test_all_nonempty(self):
        for name in SKILL_NAMES:
            path = os.path.join(os.path.abspath(SKILLS_DIR), name, "SKILL.md")
            size = os.path.getsize(path)
            self.assertGreater(size, 200, f"{name}: SKILL.md too small ({size} bytes)")

    def test_all_under_max_size(self):
        """Skills must be under 50KB (SkillRegistry.MAX_SKILL_SIZE)."""
        for name in SKILL_NAMES:
            path = os.path.join(os.path.abspath(SKILLS_DIR), name, "SKILL.md")
            size = os.path.getsize(path)
            self.assertLess(size, 50_000, f"{name}: SKILL.md too large ({size} bytes)")


# ── 3. SkillRegistry Built-in Discovery ─────────────────────────────────

class TestSkillRegistryBuiltins(unittest.TestCase):
    """Verify SkillRegistry can discover all built-in skills."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        self.registry = SkillRegistry(workspace_dir="")
        self.count = self.registry._scan_directory(os.path.abspath(SKILLS_DIR))

    def test_discovers_all_7(self):
        self.assertEqual(self.count, 7)

    def test_all_names_present(self):
        for name in SKILL_NAMES:
            self.assertIn(name, self.registry.skill_names)

    def test_get_skill_returns_object(self):
        skill = self.registry.get_skill("docx")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "docx")

    def test_skill_has_content(self):
        skill = self.registry.get_skill("xlsx")
        self.assertIn("Excel", skill.content)

    def test_skill_has_triggers(self):
        skill = self.registry.get_skill("pptx")
        self.assertTrue(len(skill.triggers) > 2)

    def test_skill_has_location(self):
        skill = self.registry.get_skill("pdf")
        self.assertTrue(os.path.isdir(skill.location))

    def test_skill_md_path(self):
        skill = self.registry.get_skill("schedule")
        self.assertTrue(os.path.isfile(skill.skill_md_path))


class TestSkillRegistryMatching(unittest.TestCase):
    """Verify trigger keyword matching works for built-in skills."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        self.registry = SkillRegistry(workspace_dir="")
        self.registry._scan_directory(os.path.abspath(SKILLS_DIR))

    def test_match_word_document(self):
        matched = self.registry.match_skills("create a word document")
        self.assertTrue(any(s.name == "docx" for s in matched))

    def test_match_presentation(self):
        matched = self.registry.match_skills("make a presentation about sales")
        self.assertTrue(any(s.name == "pptx" for s in matched))

    def test_match_spreadsheet(self):
        matched = self.registry.match_skills("create an excel spreadsheet")
        self.assertTrue(any(s.name == "xlsx" for s in matched))

    def test_match_pdf(self):
        matched = self.registry.match_skills("extract text from this PDF")
        self.assertTrue(any(s.name == "pdf" for s in matched))

    def test_match_video(self):
        matched = self.registry.match_skills("create a video animation")
        self.assertTrue(any(s.name == "remotion" for s in matched))

    def test_match_schedule(self):
        matched = self.registry.match_skills("create a scheduled task")
        self.assertTrue(any(s.name == "schedule" for s in matched))

    def test_no_match_unrelated(self):
        matched = self.registry.match_skills("what is the capital of France?")
        self.assertEqual(len(matched), 0)

    def test_match_returns_sorted_by_score(self):
        matched = self.registry.match_skills("create an excel spreadsheet with data table")
        # xlsx should be first (most trigger hits)
        if matched:
            self.assertEqual(matched[0].name, "xlsx")


class TestSkillRegistryPromptSections(unittest.TestCase):
    """get_available_skills_section and get_skill_prompt_section."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        self.registry = SkillRegistry(workspace_dir="")
        self.registry._scan_directory(os.path.abspath(SKILLS_DIR))

    def test_available_skills_section(self):
        section = self.registry.get_available_skills_section()
        self.assertIn("<available_skills>", section)
        self.assertIn("</available_skills>", section)
        for name in SKILL_NAMES:
            self.assertIn(name, section)

    def test_skill_prompt_section(self):
        skills = [self.registry.get_skill("docx")]
        section = self.registry.get_skill_prompt_section(skills)
        self.assertIn("<skills_instructions>", section)
        self.assertIn("docx", section)

    def test_empty_skills_returns_empty(self):
        section = self.registry.get_skill_prompt_section([])
        self.assertEqual(section, "")


# ── 4. Agent Integration ────────────────────────────────────────────────

class TestAgentSkillEnforcement(unittest.TestCase):
    """Sprint 29 skill enforcement in agent loop."""

    def _make_agent(self):
        from cowork_agent.core.agent import Agent
        from cowork_agent.core.skill_registry import SkillRegistry

        provider = MagicMock()
        registry = MagicMock()
        registry.list_tools.return_value = []
        registry.get_schemas.return_value = []
        prompt_builder = MagicMock()

        skill_registry = SkillRegistry(workspace_dir="")
        skill_registry._scan_directory(os.path.abspath(SKILLS_DIR))

        agent = Agent(
            provider=provider,
            registry=registry,
            prompt_builder=prompt_builder,
            skill_registry=skill_registry,
        )
        return agent

    def test_enforcement_enabled_by_default(self):
        agent = self._make_agent()
        self.assertTrue(agent._skill_enforcement_enabled)

    def test_enforcement_can_be_disabled(self):
        agent = self._make_agent()
        agent._skill_enforcement_enabled = False
        self.assertFalse(agent._skill_enforcement_enabled)

    def test_context_has_enforcement_hint(self):
        agent = self._make_agent()
        # Simulate a user message about a document
        from cowork_agent.core.models import Message
        agent._messages = [Message(role="user", content="create a word document")]
        ctx = agent._build_context()
        self.assertIn("skill_enforcement_hint", ctx)
        self.assertIn("docx", ctx["skill_enforcement_hint"])

    def test_no_hint_when_no_match(self):
        agent = self._make_agent()
        from cowork_agent.core.models import Message
        agent._messages = [Message(role="user", content="what is 2 + 2?")]
        ctx = agent._build_context()
        self.assertNotIn("skill_enforcement_hint", ctx)

    def test_no_hint_when_enforcement_disabled(self):
        agent = self._make_agent()
        agent._skill_enforcement_enabled = False
        from cowork_agent.core.models import Message
        agent._messages = [Message(role="user", content="create a word document")]
        ctx = agent._build_context()
        self.assertNotIn("skill_enforcement_hint", ctx)

    def test_active_skills_in_context(self):
        agent = self._make_agent()
        from cowork_agent.core.models import Message
        agent._messages = [Message(role="user", content="make a PDF")]
        ctx = agent._build_context()
        self.assertIn("active_skills", ctx)
        self.assertTrue(any(s.name == "pdf" for s in ctx["active_skills"]))


# ── 5. PromptBuilder Integration ────────────────────────────────────────

class TestPromptBuilderSprint29(unittest.TestCase):
    """Folder-selected and skill hint in env section."""

    def test_folder_selected_yes(self):
        from cowork_agent.core.prompt_builder import PromptBuilder
        pb = PromptBuilder({"llm": {"model": "test", "provider": "test"}})
        pb.workspace_dir = "/some/path"
        env = pb._section_env({})
        self.assertIn("User selected a folder: yes", env)

    def test_folder_selected_no(self):
        from cowork_agent.core.prompt_builder import PromptBuilder
        pb = PromptBuilder({"llm": {"model": "test", "provider": "test"}})
        pb.workspace_dir = ""
        env = pb._section_env({})
        self.assertIn("User selected a folder: no", env)

    def test_skill_hint_in_env(self):
        from cowork_agent.core.prompt_builder import PromptBuilder
        pb = PromptBuilder({"llm": {"model": "test", "provider": "test"}})
        ctx = {"skill_enforcement_hint": "Use docx skill first"}
        env = pb._section_env(ctx)
        self.assertIn("Skill hint: Use docx skill first", env)

    def test_no_skill_hint_when_absent(self):
        from cowork_agent.core.prompt_builder import PromptBuilder
        pb = PromptBuilder({"llm": {"model": "test", "provider": "test"}})
        env = pb._section_env({})
        self.assertNotIn("Skill hint:", env)


# ── 6. Edge Cases ───────────────────────────────────────────────────────

class TestSkillToolEdgeCases(unittest.TestCase):
    """Edge cases for the Skill tool."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool

        self.registry = SkillRegistry(workspace_dir="")
        self.registry._scan_directory(os.path.abspath(SKILLS_DIR))
        self.tool = SkillTool(skill_registry=self.registry)

    def test_whitespace_name_stripped(self):
        result = _run(self.tool.execute(skill="  docx  "))
        self.assertTrue(result.success)

    def test_uppercase_name_normalized(self):
        result = _run(self.tool.execute(skill="DOCX"))
        self.assertTrue(result.success)

    def test_mixed_case_normalized(self):
        result = _run(self.tool.execute(skill="PdF"))
        self.assertTrue(result.success)

    def test_only_whitespace(self):
        result = _run(self.tool.execute(skill="   "))
        self.assertFalse(result.success)

    def test_special_characters(self):
        result = _run(self.tool.execute(skill="d@cx!"))
        self.assertFalse(result.success)

    def test_colon_only(self):
        result = _run(self.tool.execute(skill=":"))
        self.assertFalse(result.success)

    def test_invoke_all_7_skills_sequentially(self):
        """Invoke every skill to confirm they all load."""
        for name in SKILL_NAMES:
            result = _run(self.tool.execute(skill=name))
            self.assertTrue(result.success, f"Failed to invoke: {name}")


class TestSkillRegistryEdgeCases(unittest.TestCase):
    """Edge cases for SkillRegistry."""

    def test_scan_nonexistent_directory(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        reg = SkillRegistry(workspace_dir="")
        count = reg._scan_directory("/nonexistent/path")
        self.assertEqual(count, 0)

    def test_scan_empty_directory(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillRegistry(workspace_dir="")
            count = reg._scan_directory(tmpdir)
            self.assertEqual(count, 0)

    def test_discover_with_no_dirs(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        reg = SkillRegistry(workspace_dir="/nonexistent", user_skills_dir="/also_nonexistent")
        count = reg.discover()
        self.assertEqual(count, 0)

    def test_get_skill_returns_none_for_unknown(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        reg = SkillRegistry(workspace_dir="")
        self.assertIsNone(reg.get_skill("nope"))

    def test_match_skills_empty_message(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        reg = SkillRegistry(workspace_dir="")
        reg._scan_directory(os.path.abspath(SKILLS_DIR))
        matched = reg.match_skills("")
        self.assertEqual(len(matched), 0)


class TestMainWiring(unittest.TestCase):
    """Verify Sprint 29 wiring block doesn't crash on import."""

    def test_skill_tool_importable(self):
        from cowork_agent.tools.skill_tool import SkillTool
        tool = SkillTool()
        self.assertEqual(tool.name, "skill")

    def test_skill_tool_inherits_base(self):
        from cowork_agent.tools.skill_tool import SkillTool
        from cowork_agent.tools.base import BaseTool
        self.assertTrue(issubclass(SkillTool, BaseTool))


if __name__ == "__main__":
    unittest.main()
