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


# ══════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS (added post-Sprint 29)
# ══════════════════════════════════════════════════════════════════════════


# ── 7. SkillTool Deep Edge Cases ────────────────────────────────────────

class TestSkillToolConcurrentInvocations(unittest.TestCase):
    """Multiple rapid invocations."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool
        self.registry = SkillRegistry(workspace_dir="")
        self.registry._scan_directory(os.path.abspath(SKILLS_DIR))
        self.tool = SkillTool(skill_registry=self.registry)

    def test_invoke_same_skill_twice(self):
        r1 = _run(self.tool.execute(skill="docx"))
        r2 = _run(self.tool.execute(skill="docx"))
        self.assertTrue(r1.success)
        self.assertTrue(r2.success)
        self.assertEqual(r1.output, r2.output)

    def test_invoke_different_skills_in_sequence(self):
        results = [_run(self.tool.execute(skill=n)) for n in SKILL_NAMES]
        self.assertTrue(all(r.success for r in results))
        # All outputs should be different
        outputs = [r.output for r in results]
        self.assertEqual(len(set(outputs)), 7)

    def test_invoke_after_error_still_works(self):
        _run(self.tool.execute(skill="nonexistent"))  # error
        result = _run(self.tool.execute(skill="pdf"))  # should succeed
        self.assertTrue(result.success)


class TestSkillToolNameNormalization(unittest.TestCase):
    """Exhaustive name normalization edge cases."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool
        self.registry = SkillRegistry(workspace_dir="")
        self.registry._scan_directory(os.path.abspath(SKILLS_DIR))
        self.tool = SkillTool(skill_registry=self.registry)

    def test_tab_in_name(self):
        result = _run(self.tool.execute(skill="\tdocx\t"))
        self.assertTrue(result.success)

    def test_newline_in_name(self):
        result = _run(self.tool.execute(skill="\ndocx\n"))
        self.assertTrue(result.success)

    def test_mixed_whitespace(self):
        result = _run(self.tool.execute(skill=" \t docx \n "))
        self.assertTrue(result.success)

    def test_colon_at_start(self):
        result = _run(self.tool.execute(skill=":docx"))
        self.assertTrue(result.success)

    def test_colon_at_end(self):
        """Trailing colon results in empty name after split."""
        result = _run(self.tool.execute(skill="docx:"))
        self.assertFalse(result.success)  # split(":")[-1] = ""

    def test_empty_after_colon(self):
        result = _run(self.tool.execute(skill="prefix:"))
        self.assertFalse(result.success)

    def test_only_colons(self):
        result = _run(self.tool.execute(skill=":::"))
        self.assertFalse(result.success)

    def test_unicode_name(self):
        result = _run(self.tool.execute(skill="dôcx"))
        self.assertFalse(result.success)

    def test_numeric_name(self):
        result = _run(self.tool.execute(skill="12345"))
        self.assertFalse(result.success)

    def test_hyphenated_skill_name(self):
        """skill-creator has a hyphen — must work."""
        result = _run(self.tool.execute(skill="SKILL-CREATOR"))
        self.assertTrue(result.success)

    def test_very_long_name(self):
        result = _run(self.tool.execute(skill="a" * 10000))
        self.assertFalse(result.success)


class TestSkillToolOutputFormat(unittest.TestCase):
    """Verify the output structure is correct for LLM consumption."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool
        self.registry = SkillRegistry(workspace_dir="")
        self.registry._scan_directory(os.path.abspath(SKILLS_DIR))
        self.tool = SkillTool(skill_registry=self.registry)

    def test_output_has_skill_instructions_tag(self):
        result = _run(self.tool.execute(skill="docx"))
        self.assertIn('<skill_instructions name="docx">', result.output)
        self.assertIn('</skill_instructions>', result.output)

    def test_output_has_loaded_message(self):
        result = _run(self.tool.execute(skill="xlsx"))
        self.assertIn('The "xlsx" skill is loaded.', result.output)

    def test_output_contains_skill_content(self):
        result = _run(self.tool.execute(skill="pdf"))
        # Should contain actual SKILL.md content
        self.assertIn("pdfplumber", result.output)
        self.assertIn("reportlab", result.output)

    def test_args_inserted_at_correct_position(self):
        result = _run(self.tool.execute(skill="docx", args="--format letter"))
        lines = result.output.split("\n")
        # Line 0: loaded message, Line 1: args, Line 2: empty, Line 3: tag
        self.assertEqual(lines[0], 'The "docx" skill is loaded.')
        self.assertEqual(lines[1], "Arguments: --format letter")

    def test_no_args_means_empty_line_after_loaded(self):
        result = _run(self.tool.execute(skill="docx"))
        lines = result.output.split("\n")
        self.assertEqual(lines[0], 'The "docx" skill is loaded.')
        self.assertEqual(lines[1], "")  # blank line, no args

    def test_error_result_has_no_output(self):
        result = _run(self.tool.execute(skill="nonexistent"))
        self.assertFalse(result.success)
        self.assertEqual(result.output, "")

    def test_error_result_lists_available_skills(self):
        result = _run(self.tool.execute(skill="nonexistent"))
        for name in SKILL_NAMES:
            self.assertIn(name, result.error)


class TestSkillToolProgressCallback(unittest.TestCase):
    """Progress callback parameter handling."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool
        self.registry = SkillRegistry(workspace_dir="")
        self.registry._scan_directory(os.path.abspath(SKILLS_DIR))
        self.tool = SkillTool(skill_registry=self.registry)

    def test_with_progress_callback(self):
        cb = MagicMock()
        result = _run(self.tool.execute(skill="docx", progress_callback=cb))
        self.assertTrue(result.success)

    def test_with_none_callback(self):
        result = _run(self.tool.execute(skill="docx", progress_callback=None))
        self.assertTrue(result.success)

    def test_extra_kwargs_ignored(self):
        result = _run(self.tool.execute(skill="docx", extra_param="ignored"))
        self.assertTrue(result.success)


# ── 8. SkillRegistry Deep Edge Cases ────────────────────────────────────

class TestSkillRegistryLoadEdgeCases(unittest.TestCase):
    """Edge cases in skill loading."""

    def test_empty_skill_md_file(self):
        """A SKILL.md with only whitespace should not load."""
        import tempfile
        from cowork_agent.core.skill_registry import SkillRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "empty-skill")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
                f.write("   \n  \n  ")

            reg = SkillRegistry(workspace_dir="")
            count = reg._scan_directory(tmpdir)
            self.assertEqual(count, 0)

    def test_skill_md_too_large(self):
        """A SKILL.md exceeding MAX_SKILL_SIZE should be skipped."""
        import tempfile
        from cowork_agent.core.skill_registry import SkillRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "huge-skill")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
                f.write("x" * 60_000)  # > 50KB limit

            reg = SkillRegistry(workspace_dir="")
            count = reg._scan_directory(tmpdir)
            self.assertEqual(count, 0)

    def test_skill_md_exactly_at_limit(self):
        """A SKILL.md at exactly MAX_SKILL_SIZE should load."""
        import tempfile
        from cowork_agent.core.skill_registry import SkillRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "limit-skill")
            os.makedirs(skill_dir)
            content = "---\nname: limit-skill\ndescription: test\n---\n"
            content += "x" * (50_000 - len(content.encode('utf-8')))
            with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
                f.write(content)

            reg = SkillRegistry(workspace_dir="")
            count = reg._scan_directory(tmpdir)
            self.assertEqual(count, 1)

    def test_skill_dir_without_skill_md(self):
        """A directory without SKILL.md should be skipped."""
        import tempfile
        from cowork_agent.core.skill_registry import SkillRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "no-md-skill")
            os.makedirs(skill_dir)
            # Create a different file
            with open(os.path.join(skill_dir, "README.md"), "w") as f:
                f.write("Not a skill")

            reg = SkillRegistry(workspace_dir="")
            count = reg._scan_directory(tmpdir)
            self.assertEqual(count, 0)

    def test_file_instead_of_directory(self):
        """A plain file in the skills dir should be ignored."""
        import tempfile
        from cowork_agent.core.skill_registry import SkillRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file, not a directory
            with open(os.path.join(tmpdir, "not-a-dir"), "w") as f:
                f.write("hello")

            reg = SkillRegistry(workspace_dir="")
            count = reg._scan_directory(tmpdir)
            self.assertEqual(count, 0)

    def test_duplicate_skill_workspace_takes_priority(self):
        """Workspace skills should not be overwritten by later scans."""
        import tempfile
        from cowork_agent.core.skill_registry import SkillRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "docx")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
                f.write("---\nname: docx\ndescription: custom\n---\nCustom content")

            reg = SkillRegistry(workspace_dir="")
            # Load built-in first
            reg._scan_directory(os.path.abspath(SKILLS_DIR))
            original_content = reg.get_skill("docx").content

            # Scan again with custom - should NOT overwrite
            reg._scan_directory(tmpdir)
            self.assertEqual(reg.get_skill("docx").content, original_content)

    def test_skill_with_no_frontmatter(self):
        """A SKILL.md without frontmatter should still load with fallback description."""
        import tempfile
        from cowork_agent.core.skill_registry import SkillRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "plain-skill")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
                f.write("# Plain Skill\n\nThis is a plain skill with no frontmatter.")

            reg = SkillRegistry(workspace_dir="")
            count = reg._scan_directory(tmpdir)
            self.assertEqual(count, 1)
            skill = reg.get_skill("plain-skill")
            self.assertIsNotNone(skill)
            # Should use fallback description
            self.assertIn("plain skill", skill.description.lower())

    def test_skill_with_binary_content_in_directory(self):
        """Extra non-md files in skill directory should not affect loading."""
        import tempfile
        from cowork_agent.core.skill_registry import SkillRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = os.path.join(tmpdir, "with-extras")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
                f.write("---\nname: with-extras\ndescription: test\n---\nContent")
            # Write a binary file alongside
            with open(os.path.join(skill_dir, "data.bin"), "wb") as f:
                f.write(b"\x00\x01\x02\x03")

            reg = SkillRegistry(workspace_dir="")
            count = reg._scan_directory(tmpdir)
            self.assertEqual(count, 1)


class TestSkillRegistryDescriptionExtraction(unittest.TestCase):
    """Edge cases in _extract_description."""

    def test_description_with_double_quotes(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        content = '---\ndescription: "Quoted description"\n---\nBody'
        desc = SkillRegistry._extract_description(content)
        self.assertEqual(desc, "Quoted description")

    def test_description_with_single_quotes(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        content = "---\ndescription: 'Single quoted'\n---\nBody"
        desc = SkillRegistry._extract_description(content)
        self.assertEqual(desc, "Single quoted")

    def test_description_no_quotes(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        content = "---\ndescription: No quotes here\n---\nBody"
        desc = SkillRegistry._extract_description(content)
        self.assertEqual(desc, "No quotes here")

    def test_no_description_falls_back(self):
        """Without explicit 'description:' key, regex matches 'name:' as
        description-like due to `(.+?)` pattern.  Use content without any
        `key: value` lines to exercise the true fallback path."""
        from cowork_agent.core.skill_registry import SkillRegistry
        content = "# Heading\n\nFirst non-heading line here"
        desc = SkillRegistry._extract_description(content)
        self.assertEqual(desc, "First non-heading line here")

    def test_only_headings_and_dashes(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        content = "---\n---\n# Heading\n## Sub"
        desc = SkillRegistry._extract_description(content)
        self.assertEqual(desc, "No description available.")

    def test_very_long_first_line_truncated(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        content = "x" * 500
        desc = SkillRegistry._extract_description(content)
        self.assertEqual(len(desc), 200)


class TestSkillRegistryTriggerExtraction(unittest.TestCase):
    """Edge cases in _extract_triggers."""

    def test_triggers_with_semicolons(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        content = "MANDATORY TRIGGERS: word; document; report"
        triggers = SkillRegistry._extract_triggers("test", content)
        self.assertIn("word", triggers)
        self.assertIn("document", triggers)
        self.assertIn("report", triggers)

    def test_triggers_with_pipes(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        content = "MANDATORY TRIGGERS: alpha|beta|gamma"
        triggers = SkillRegistry._extract_triggers("test", content)
        self.assertIn("alpha", triggers)
        self.assertIn("gamma", triggers)

    def test_triggers_case_insensitive_header(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        content = "mandatory triggers: abc, def"
        triggers = SkillRegistry._extract_triggers("test", content)
        self.assertIn("abc", triggers)

    def test_name_always_in_triggers(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        triggers = SkillRegistry._extract_triggers("myskill", "no triggers here")
        self.assertIn("myskill", triggers)

    def test_empty_content_returns_just_name(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        triggers = SkillRegistry._extract_triggers("test", "")
        self.assertEqual(triggers, ["test"])

    def test_extension_map_docx(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        triggers = SkillRegistry._extract_triggers("docx", "no triggers")
        self.assertIn("word", triggers)
        self.assertIn(".docx", triggers)

    def test_extension_map_not_applied_to_unknown(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        triggers = SkillRegistry._extract_triggers("custom", "no triggers")
        self.assertNotIn("word", triggers)

    def test_triggers_deduplication(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        content = "MANDATORY TRIGGERS: docx, DOCX, Docx"
        triggers = SkillRegistry._extract_triggers("docx", content)
        # All lowered, deduped
        self.assertEqual(triggers.count("docx"), 1)


class TestSkillRegistryMatchingEdgeCases(unittest.TestCase):
    """Edge cases in match_skills."""

    def setUp(self):
        from cowork_agent.core.skill_registry import SkillRegistry
        self.registry = SkillRegistry(workspace_dir="")
        self.registry._scan_directory(os.path.abspath(SKILLS_DIR))

    def test_match_case_insensitive(self):
        matched = self.registry.match_skills("CREATE A WORD DOCUMENT")
        self.assertTrue(any(s.name == "docx" for s in matched))

    def test_match_partial_trigger(self):
        """'spreadsheet' is a trigger, 'spreadsheets' should also match."""
        matched = self.registry.match_skills("make spreadsheets")
        self.assertTrue(any(s.name == "xlsx" for s in matched))

    def test_match_multiple_skills(self):
        """A message mentioning both PDF and merge should match pdf."""
        matched = self.registry.match_skills("merge these PDF files")
        self.assertTrue(any(s.name == "pdf" for s in matched))

    def test_no_match_for_gibberish(self):
        matched = self.registry.match_skills("xyzzy plugh plover")
        self.assertEqual(len(matched), 0)

    def test_match_with_only_extension(self):
        """Just '.xlsx' should trigger xlsx."""
        matched = self.registry.match_skills("open the .xlsx file")
        self.assertTrue(any(s.name == "xlsx" for s in matched))

    def test_match_preserves_order_by_score(self):
        """Multiple trigger hits = higher ranking."""
        matched = self.registry.match_skills("create excel spreadsheet xlsx budget data table chart")
        if len(matched) > 1:
            # xlsx should be first with the most hits
            self.assertEqual(matched[0].name, "xlsx")

    def test_skills_property_returns_copy(self):
        skills = self.registry.skills
        skills["fake"] = None
        self.assertNotIn("fake", self.registry.skills)


# ── 9. Skill Dataclass Edge Cases ───────────────────────────────────────

class TestSkillDataclass(unittest.TestCase):
    """Edge cases for the Skill dataclass."""

    def test_skill_md_path_construction(self):
        from cowork_agent.core.skill_registry import Skill
        s = Skill(name="test", description="d", location="/foo/bar", content="c")
        self.assertEqual(s.skill_md_path, "/foo/bar/SKILL.md")

    def test_skill_default_triggers(self):
        from cowork_agent.core.skill_registry import Skill
        s = Skill(name="test", description="d", location="/foo", content="c")
        self.assertEqual(s.triggers, [])

    def test_skill_with_triggers(self):
        from cowork_agent.core.skill_registry import Skill
        s = Skill(name="t", description="d", location="/", content="c",
                  triggers=["a", "b"])
        self.assertEqual(s.triggers, ["a", "b"])

    def test_skill_md_path_with_spaces(self):
        from cowork_agent.core.skill_registry import Skill
        s = Skill(name="test", description="d", location="/path with spaces/skill",
                  content="c")
        self.assertEqual(s.skill_md_path, "/path with spaces/skill/SKILL.md")


# ── 10. PromptBuilder Edge Cases ────────────────────────────────────────

class TestPromptBuilderEdgeCases(unittest.TestCase):
    """Edge cases for Sprint 29 prompt builder changes."""

    def test_env_section_always_has_folder_status(self):
        from cowork_agent.core.prompt_builder import PromptBuilder
        pb = PromptBuilder({"llm": {"model": "m", "provider": "p"}})
        pb.workspace_dir = "/some/dir"
        env = pb._section_env({})
        self.assertIn("User selected a folder:", env)

    def test_env_section_with_all_context(self):
        from cowork_agent.core.prompt_builder import PromptBuilder
        pb = PromptBuilder({"llm": {"model": "m", "provider": "p"}})
        pb.workspace_dir = "/dir"
        ctx = {
            "iteration": 5,
            "skill_enforcement_hint": "Use pdf skill",
        }
        env = pb._section_env(ctx)
        self.assertIn("Agent iteration: 5 of 15", env)
        self.assertIn("Skill hint: Use pdf skill", env)
        self.assertIn("User selected a folder: yes", env)

    def test_env_section_empty_workspace_is_no(self):
        from cowork_agent.core.prompt_builder import PromptBuilder
        pb = PromptBuilder({"llm": {"model": "m", "provider": "p"}})
        pb.workspace_dir = ""
        env = pb._section_env({})
        self.assertIn("User selected a folder: no", env)

    def test_env_still_has_xml_tags(self):
        from cowork_agent.core.prompt_builder import PromptBuilder
        pb = PromptBuilder({"llm": {"model": "m", "provider": "p"}})
        env = pb._section_env({})
        self.assertTrue(env.startswith("<env>"))
        self.assertTrue(env.endswith("</env>"))

    def test_skill_hint_not_present_without_context(self):
        from cowork_agent.core.prompt_builder import PromptBuilder
        pb = PromptBuilder({"llm": {"model": "m", "provider": "p"}})
        env = pb._section_env({})
        self.assertNotIn("Skill hint:", env)

    def test_skill_hint_with_empty_string(self):
        from cowork_agent.core.prompt_builder import PromptBuilder
        pb = PromptBuilder({"llm": {"model": "m", "provider": "p"}})
        env = pb._section_env({"skill_enforcement_hint": ""})
        self.assertNotIn("Skill hint:", env)

    def test_available_skills_in_full_build(self):
        """Full build includes available_skills when registry has skills."""
        from cowork_agent.core.prompt_builder import PromptBuilder
        from cowork_agent.core.skill_registry import SkillRegistry

        reg = SkillRegistry(workspace_dir="")
        reg._scan_directory(os.path.abspath(SKILLS_DIR))
        pb = PromptBuilder({"llm": {"model": "m", "provider": "p"}},
                           skill_registry=reg)
        full_prompt = pb.build(tools=[])
        self.assertIn("<available_skills>", full_prompt)

    def test_active_skills_injected_in_full_build(self):
        """Full build includes active skill content when matched."""
        from cowork_agent.core.prompt_builder import PromptBuilder
        from cowork_agent.core.skill_registry import SkillRegistry

        reg = SkillRegistry(workspace_dir="")
        reg._scan_directory(os.path.abspath(SKILLS_DIR))
        skills = [reg.get_skill("docx")]
        pb = PromptBuilder({"llm": {"model": "m", "provider": "p"}},
                           skill_registry=reg)
        full_prompt = pb.build(tools=[], context={"active_skills": skills})
        self.assertIn("<skills_instructions>", full_prompt)
        self.assertIn("docx", full_prompt)


# ── 11. Agent Enforcement Deep Edge Cases ───────────────────────────────

class TestAgentEnforcementDeepEdges(unittest.TestCase):
    """Deep edge cases for skill enforcement in agent context."""

    def _make_agent(self, **kwargs):
        from cowork_agent.core.agent import Agent
        from cowork_agent.core.skill_registry import SkillRegistry

        provider = MagicMock()
        registry = MagicMock()
        registry.list_tools.return_value = []
        registry.get_schemas.return_value = []
        prompt_builder = MagicMock()

        skill_registry = SkillRegistry(workspace_dir="")
        skill_registry._scan_directory(os.path.abspath(SKILLS_DIR))

        return Agent(
            provider=provider,
            registry=registry,
            prompt_builder=prompt_builder,
            skill_registry=skill_registry,
            **kwargs,
        )

    def test_hint_references_first_matched_skill(self):
        from cowork_agent.core.models import Message
        agent = self._make_agent()
        agent._messages = [Message(role="user", content="create a word document")]
        ctx = agent._build_context()
        self.assertIn("skill='docx'", ctx["skill_enforcement_hint"])

    def test_hint_lists_all_matched_skills(self):
        from cowork_agent.core.models import Message
        agent = self._make_agent()
        # "merge" and "pdf" and "split" all trigger pdf skill
        agent._messages = [Message(role="user", content="merge and split this pdf")]
        ctx = agent._build_context()
        self.assertIn("pdf", ctx.get("skill_enforcement_hint", ""))

    def test_no_messages_no_crash(self):
        agent = self._make_agent()
        agent._messages = []
        ctx = agent._build_context()
        self.assertNotIn("skill_enforcement_hint", ctx)

    def test_assistant_message_only_no_match(self):
        from cowork_agent.core.models import Message
        agent = self._make_agent()
        agent._messages = [Message(role="assistant", content="create a word document")]
        ctx = agent._build_context()
        # Should not match because we look for last user message
        self.assertNotIn("skill_enforcement_hint", ctx)

    def test_multiple_user_messages_uses_last(self):
        from cowork_agent.core.models import Message
        agent = self._make_agent()
        agent._messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
            Message(role="user", content="create a pdf"),
        ]
        ctx = agent._build_context()
        self.assertIn("pdf", ctx.get("skill_enforcement_hint", ""))

    def test_enforcement_flag_persists_across_contexts(self):
        agent = self._make_agent()
        agent._skill_enforcement_enabled = False
        from cowork_agent.core.models import Message
        agent._messages = [Message(role="user", content="create a word document")]
        ctx1 = agent._build_context()
        ctx2 = agent._build_context()
        self.assertNotIn("skill_enforcement_hint", ctx1)
        self.assertNotIn("skill_enforcement_hint", ctx2)

    def test_agent_without_skill_registry(self):
        from cowork_agent.core.agent import Agent
        provider = MagicMock()
        registry = MagicMock()
        registry.list_tools.return_value = []
        registry.get_schemas.return_value = []
        prompt_builder = MagicMock()
        agent = Agent(provider=provider, registry=registry,
                      prompt_builder=prompt_builder, skill_registry=None)
        from cowork_agent.core.models import Message
        agent._messages = [Message(role="user", content="create a word document")]
        ctx = agent._build_context()
        self.assertNotIn("active_skills", ctx)
        self.assertNotIn("skill_enforcement_hint", ctx)


# ── 12. SKILL.md Content Validation ─────────────────────────────────────

class TestSkillMdContentValidation(unittest.TestCase):
    """Validate that SKILL.md content is meaningful and complete."""

    def _read_skill(self, name):
        path = os.path.join(os.path.abspath(SKILLS_DIR), name, "SKILL.md")
        with open(path) as f:
            return f.read()

    def test_docx_has_python_docx(self):
        self.assertIn("python-docx", self._read_skill("docx"))

    def test_pptx_has_python_pptx(self):
        self.assertIn("python-pptx", self._read_skill("pptx"))

    def test_xlsx_has_openpyxl(self):
        self.assertIn("openpyxl", self._read_skill("xlsx"))

    def test_xlsx_has_formula_rule(self):
        content = self._read_skill("xlsx")
        self.assertIn("ALWAYS use Excel formulas", content)

    def test_pdf_has_pdfplumber(self):
        self.assertIn("pdfplumber", self._read_skill("pdf"))

    def test_pdf_has_reportlab(self):
        self.assertIn("reportlab", self._read_skill("pdf"))

    def test_remotion_has_react(self):
        self.assertIn("React", self._read_skill("remotion"))

    def test_schedule_has_cron(self):
        self.assertIn("cron", self._read_skill("schedule").lower())

    def test_skill_creator_has_eval(self):
        content = self._read_skill("skill-creator")
        self.assertIn("eval", content.lower())

    def test_all_skills_have_install_section(self):
        for name in ["docx", "pptx", "xlsx", "pdf"]:
            content = self._read_skill(name)
            self.assertIn("install", content.lower(),
                          f"{name}: missing install section")

    def test_all_skills_have_code_examples(self):
        for name in ["docx", "pptx", "xlsx", "pdf", "remotion"]:
            content = self._read_skill(name)
            self.assertIn("```", content,
                          f"{name}: missing code examples")

    def test_docx_has_heading_example(self):
        self.assertIn("add_heading", self._read_skill("docx"))

    def test_pptx_has_slide_layout(self):
        self.assertIn("slide_layouts", self._read_skill("pptx"))

    def test_xlsx_has_chart_example(self):
        self.assertIn("BarChart", self._read_skill("xlsx"))

    def test_schedule_has_local_timezone_warning(self):
        content = self._read_skill("schedule")
        self.assertIn("local timezone", content.lower())


# ── 13. Cross-Component Integration Edge Cases ─────────────────────────

class TestCrossComponentEdgeCases(unittest.TestCase):
    """Edge cases spanning multiple Sprint 29 components."""

    def test_registry_to_tool_roundtrip(self):
        """Discover skills, then invoke each via SkillTool."""
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool

        reg = SkillRegistry(workspace_dir="")
        reg._scan_directory(os.path.abspath(SKILLS_DIR))

        tool = SkillTool(skill_registry=reg)
        for name in reg.skill_names:
            result = _run(tool.execute(skill=name))
            self.assertTrue(result.success, f"Roundtrip failed for: {name}")
            self.assertIn(name, result.metadata.get("skill_name", ""))

    def test_matched_skills_content_matches_tool_output(self):
        """Content from registry.match_skills matches what SkillTool returns."""
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool

        reg = SkillRegistry(workspace_dir="")
        reg._scan_directory(os.path.abspath(SKILLS_DIR))
        tool = SkillTool(skill_registry=reg)

        matched = reg.match_skills("create an excel spreadsheet")
        self.assertTrue(any(s.name == "xlsx" for s in matched))

        result = _run(tool.execute(skill="xlsx"))
        xlsx_skill = reg.get_skill("xlsx")
        self.assertIn(xlsx_skill.content, result.output)

    def test_prompt_builder_with_agent_context(self):
        """PromptBuilder uses the enforcement hint from agent context."""
        from cowork_agent.core.prompt_builder import PromptBuilder
        from cowork_agent.core.skill_registry import SkillRegistry

        reg = SkillRegistry(workspace_dir="")
        reg._scan_directory(os.path.abspath(SKILLS_DIR))
        pb = PromptBuilder({"llm": {"model": "m", "provider": "p"}},
                           skill_registry=reg)
        pb.workspace_dir = "/workspace"

        ctx = {
            "skill_enforcement_hint": "Use docx skill before creating documents.",
            "active_skills": [reg.get_skill("docx")],
        }
        prompt = pb.build(tools=[], context=ctx)
        self.assertIn("Skill hint:", prompt)
        self.assertIn("<skills_instructions>", prompt)
        self.assertIn("User selected a folder: yes", prompt)

    def test_skill_registry_discover_then_tool(self):
        """Full flow: discover() → SkillTool.execute()."""
        import tempfile
        from cowork_agent.core.skill_registry import SkillRegistry
        from cowork_agent.tools.skill_tool import SkillTool

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a custom skill
            skill_dir = os.path.join(tmpdir, ".skills", "skills", "custom")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
                f.write("---\nname: custom\ndescription: A custom skill\n---\n"
                        "# Custom Skill\nMANDATORY TRIGGERS: custom, special\n"
                        "Do custom things.")

            reg = SkillRegistry(workspace_dir=tmpdir)
            count = reg.discover()
            self.assertGreaterEqual(count, 1)
            self.assertIn("custom", reg.skill_names)

            tool = SkillTool(skill_registry=reg)
            result = _run(tool.execute(skill="custom"))
            self.assertTrue(result.success)
            self.assertIn("Custom Skill", result.output)


if __name__ == "__main__":
    unittest.main()
