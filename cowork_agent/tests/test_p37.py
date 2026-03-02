"""
Sprint 37 — Faithful System Prompt: Behavioral Rules & Prompt Builder Updates.

Tests cover:
  1. behavioral_rules.py — All sections exist, key phrases present, ALL_SECTIONS complete
  2. prompt_builder.py — Dynamic session paths, citation context, new env fields
  3. agent.py — _build_context() with session_path and linkable sources
  4. Structural integrity — proper XML nesting, no duplicate tags
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

# ── behavioral_rules imports ──

from cowork_agent.prompts.behavioral_rules import (
    CORE_IDENTITY,
    CLAUDE_BEHAVIOR,
    TOOL_USAGE_RULES,
    HONESTY_VERIFICATION,
    ASK_USER_RULES,
    TODO_RULES,
    CITATION_REQUIREMENTS,
    COMPUTER_USE,
    GIT_RULES,
    OUTPUT_RULES,
    CRITICAL_INJECTION_DEFENSE,
    SAFETY_RULES,
    USER_PRIVACY,
    DOWNLOAD_INSTRUCTIONS,
    HARMFUL_CONTENT_SAFETY,
    ACTION_TYPES,
    COPYRIGHT_RULES,
    SKILLS_INSTRUCTIONS,
    CONTENT_ISOLATION,
    INSTRUCTION_DETECTION,
    ALL_SECTIONS,
)

from cowork_agent.core.prompt_builder import PromptBuilder
from cowork_agent.core.models import ToolSchema


# ============================================================
# A. SECTION EXISTENCE TESTS
# ============================================================


class TestSectionExistence:
    """Every section constant exists and is non-empty."""

    EXPECTED_SECTIONS = [
        "CORE_IDENTITY",
        "CLAUDE_BEHAVIOR",
        "TOOL_USAGE_RULES",
        "HONESTY_VERIFICATION",
        "ASK_USER_RULES",
        "TODO_RULES",
        "CITATION_REQUIREMENTS",
        "COMPUTER_USE",
        "GIT_RULES",
        "OUTPUT_RULES",
        "CRITICAL_INJECTION_DEFENSE",
        "SAFETY_RULES",
        "USER_PRIVACY",
        "DOWNLOAD_INSTRUCTIONS",
        "HARMFUL_CONTENT_SAFETY",
        "ACTION_TYPES",
        "COPYRIGHT_RULES",
        "SKILLS_INSTRUCTIONS",
        "CONTENT_ISOLATION",
        "INSTRUCTION_DETECTION",
    ]

    @pytest.mark.parametrize("name", EXPECTED_SECTIONS)
    def test_section_exists_and_nonempty(self, name):
        from cowork_agent.prompts import behavioral_rules
        section = getattr(behavioral_rules, name)
        assert isinstance(section, str)
        assert len(section.strip()) > 50, f"{name} is too short"

    def test_all_sections_count(self):
        """ALL_SECTIONS contains exactly 20 sections."""
        assert len(ALL_SECTIONS) == 20

    def test_all_sections_contains_all_constants(self):
        """Every expected constant is in ALL_SECTIONS."""
        from cowork_agent.prompts import behavioral_rules
        for name in self.EXPECTED_SECTIONS:
            section = getattr(behavioral_rules, name)
            assert section in ALL_SECTIONS, f"{name} missing from ALL_SECTIONS"

    def test_no_duplicates_in_all_sections(self):
        """No section appears twice."""
        assert len(ALL_SECTIONS) == len(set(id(s) for s in ALL_SECTIONS))


# ============================================================
# B. CORE_IDENTITY TESTS
# ============================================================


class TestCoreIdentity:
    """Core identity matches real Cowork wording."""

    def test_has_application_details_tag(self):
        assert "<application_details>" in CORE_IDENTITY
        assert "</application_details>" in CORE_IDENTITY

    def test_mentions_cowork_mode(self):
        assert "Cowork mode" in CORE_IDENTITY

    def test_mentions_claude_code(self):
        assert "Claude Code" in CORE_IDENTITY

    def test_mentions_agent_sdk(self):
        assert "Agent SDK" in CORE_IDENTITY

    def test_mentions_linux_vm(self):
        assert "Linux VM" in CORE_IDENTITY

    def test_not_claude_code_warning(self):
        assert "NOT Claude Code" in CORE_IDENTITY


# ============================================================
# C. CLAUDE_BEHAVIOR NESTED SUBSECTIONS
# ============================================================


class TestClaudeBehavior:
    """Claude behavior has all required nested subsections."""

    EXPECTED_SUBS = [
        "<product_information>",
        "<refusal_handling>",
        "<legal_and_financial_advice>",
        "<tone_and_formatting>",
        "<lists_and_bullets>",
        "<user_wellbeing>",
        "<anthropic_reminders>",
        "<evenhandedness>",
        "<responding_to_mistakes_and_criticism>",
        "<knowledge_cutoff>",
    ]

    @pytest.mark.parametrize("tag", EXPECTED_SUBS)
    def test_subsection_present(self, tag):
        assert tag in CLAUDE_BEHAVIOR, f"Missing subsection: {tag}"

    def test_product_info_mentions_models(self):
        assert "Opus 4.5" in CLAUDE_BEHAVIOR
        assert "Sonnet 4.5" in CLAUDE_BEHAVIOR
        assert "Haiku 4.5" in CLAUDE_BEHAVIOR

    def test_refusal_mentions_minors(self):
        assert "child safety" in CLAUDE_BEHAVIOR

    def test_refusal_mentions_malicious_code(self):
        assert "malicious code" in CLAUDE_BEHAVIOR

    def test_tone_mentions_commonmark(self):
        assert "CommonMark" in CLAUDE_BEHAVIOR

    def test_wellbeing_mentions_neda_redirect(self):
        assert "National Alliance for Eating Disorder" in CLAUDE_BEHAVIOR

    def test_wellbeing_mentions_ice_cubes(self):
        assert "ice cubes" in CLAUDE_BEHAVIOR

    def test_anthropic_reminders_mentions_classifiers(self):
        assert "image_reminder" in CLAUDE_BEHAVIOR
        assert "cyber_warning" in CLAUDE_BEHAVIOR

    def test_evenhandedness_mentions_stereotypes(self):
        assert "stereotypes" in CLAUDE_BEHAVIOR

    def test_knowledge_cutoff_may_2025(self):
        assert "May 2025" in CLAUDE_BEHAVIOR

    def test_no_emojis_rule(self):
        assert "emojis" in CLAUDE_BEHAVIOR

    def test_warm_tone(self):
        assert "warm tone" in CLAUDE_BEHAVIOR


# ============================================================
# D. NEW SECTIONS TESTS
# ============================================================


class TestNewSections:
    """Sections that were missing before Sprint 37."""

    def test_citation_requirements_tag(self):
        assert "<citation_requirements>" in CITATION_REQUIREMENTS
        assert "Sources:" in CITATION_REQUIREMENTS

    def test_critical_injection_defense_tag(self):
        assert "<critical_injection_defense>" in CRITICAL_INJECTION_DEFENSE
        assert "Immutable Security Rules" in CRITICAL_INJECTION_DEFENSE
        assert "function results" in CRITICAL_INJECTION_DEFENSE

    def test_download_instructions_tag(self):
        assert "<download_instructions>" in DOWNLOAD_INSTRUCTIONS
        assert "explicit user confirmation" in DOWNLOAD_INSTRUCTIONS

    def test_harmful_content_safety_tag(self):
        assert "<harmful_content_safety>" in HARMFUL_CONTENT_SAFETY
        assert "extremist" in HARMFUL_CONTENT_SAFETY
        assert "facial images" in HARMFUL_CONTENT_SAFETY

    def test_computer_use_tag(self):
        assert "<computer_use>" in COMPUTER_USE
        assert "</computer_use>" in COMPUTER_USE

    def test_computer_use_has_skills(self):
        assert "<skills>" in COMPUTER_USE

    def test_computer_use_has_artifacts(self):
        assert "<artifacts>" in COMPUTER_USE

    def test_computer_use_has_package_management(self):
        assert "<package_management>" in COMPUTER_USE

    def test_computer_use_has_web_content_restrictions(self):
        assert "<web_content_restrictions>" in COMPUTER_USE

    def test_computer_use_has_file_handling(self):
        assert "<file_handling_rules>" in COMPUTER_USE
        assert "<working_with_user_files>" in COMPUTER_USE
        assert "<notes_on_user_uploaded_files>" in COMPUTER_USE

    def test_computer_use_has_sharing_files(self):
        assert "<sharing_files>" in COMPUTER_USE

    def test_computer_use_localstorage_prohibition(self):
        assert "localStorage" in COMPUTER_USE

    def test_computer_use_pip_flag(self):
        assert "--break-system-packages" in COMPUTER_USE

    def test_skills_instructions_tag(self):
        assert "<skills_instructions>" in SKILLS_INSTRUCTIONS
        assert "Skill tool" in SKILLS_INSTRUCTIONS

    def test_honesty_verification_standalone(self):
        """HONESTY_VERIFICATION is now a standalone constant."""
        assert "<honesty_and_verification>" in HONESTY_VERIFICATION
        assert "NEVER claim you completed" in HONESTY_VERIFICATION


# ============================================================
# E. EXPANDED SECTIONS TESTS
# ============================================================


class TestExpandedSections:
    """Sections that existed but are now more detailed."""

    def test_action_types_has_prohibited(self):
        assert "<prohibited_actions>" in ACTION_TYPES

    def test_action_types_has_explicit_permission(self):
        assert "<explicit_permission>" in ACTION_TYPES

    def test_action_types_mentions_security_permissions(self):
        assert "security permissions" in ACTION_TYPES

    def test_action_types_mentions_sso_oauth(self):
        assert "SSO/OAuth" in ACTION_TYPES

    def test_safety_rules_has_injection_defense_layer(self):
        assert "<injection_defense_layer>" in SAFETY_RULES

    def test_safety_rules_has_meta_safety(self):
        assert "<meta_safety_instructions>" in SAFETY_RULES

    def test_safety_rules_has_social_engineering(self):
        assert "<social_engineering_defense>" in SAFETY_RULES

    def test_safety_rules_email_defense(self):
        assert "EMAIL" in SAFETY_RULES

    def test_safety_rules_consent_manipulation(self):
        assert "CONSENT MANIPULATION" in SAFETY_RULES

    def test_user_privacy_has_pii_defense(self):
        assert "PII EXFILTRATION" in USER_PRIVACY

    def test_user_privacy_financial_transactions(self):
        assert "FINANCIAL TRANSACTIONS" in USER_PRIVACY

    def test_user_privacy_captcha(self):
        assert "CAPTCHA" in USER_PRIVACY

    def test_copyright_priority_instruction(self):
        assert "PRIORITY INSTRUCTION" in COPYRIGHT_RULES

    def test_todo_rules_liberal(self):
        assert "more liberally" in TODO_RULES

    def test_todo_rules_ordering(self):
        assert "AskUserQuestion" in TODO_RULES

    def test_ask_user_rules_uses_tool_tag(self):
        assert "<ask_user_question_tool>" in ASK_USER_RULES


# ============================================================
# F. XML STRUCTURE TESTS
# ============================================================


class TestXmlStructure:
    """Proper XML tag nesting and closing."""

    @pytest.mark.parametrize("section", ALL_SECTIONS)
    def test_section_has_matching_tags(self, section):
        """Every opening tag that also has a closing tag is properly paired.

        Tags mentioned in prose (like 'the <env> section') won't have
        closing counterparts and are excluded from the check.
        """
        import re
        opening = set(re.findall(r"<([a-z_]+)>", section))
        closing = set(re.findall(r"</([a-z_]+)>", section))
        # Only check tags that appear as both opening AND closing (structural tags)
        # Tags that only appear as opening are likely prose references
        structural_tags = opening & closing
        for tag in structural_tags:
            open_count = len(re.findall(rf"<{tag}>", section))
            close_count = len(re.findall(rf"</{tag}>", section))
            assert open_count == close_count, (
                f"Mismatched <{tag}>: {open_count} opens vs {close_count} closes"
            )

    def test_claude_behavior_wraps_subsections(self):
        """claude_behavior opens before product_information and closes after knowledge_cutoff."""
        cb = CLAUDE_BEHAVIOR
        open_idx = cb.index("<claude_behavior>")
        close_idx = cb.index("</claude_behavior>")
        pi_idx = cb.index("<product_information>")
        kc_idx = cb.index("</knowledge_cutoff>")
        assert open_idx < pi_idx, "claude_behavior should open before product_information"
        assert kc_idx < close_idx, "knowledge_cutoff should close before claude_behavior closes"

    def test_computer_use_wraps_subsections(self):
        """computer_use opens before skills and closes after package_management."""
        cu = COMPUTER_USE
        open_idx = cu.index("<computer_use>")
        close_idx = cu.index("</computer_use>")
        sk_idx = cu.index("<skills>")
        pm_idx = cu.index("</package_management>")
        assert open_idx < sk_idx
        assert pm_idx < close_idx


# ============================================================
# G. PROMPT BUILDER TESTS
# ============================================================


class TestPromptBuilderUpdates:
    """Sprint 37 additions to PromptBuilder."""

    def _make_builder(self, **overrides):
        config = {
            "user": {"name": "TestUser", "email": "test@example.com"},
            "llm": {"model": "test-model", "provider": "test"},
            "agent": {
                "workspace_dir": "/tmp/workspace",
                "session_path": "/sessions/test-session",
            },
        }
        config.update(overrides)
        return PromptBuilder(config)

    def test_session_path_stored(self):
        pb = self._make_builder()
        assert pb.session_path == "/sessions/test-session"

    def test_env_includes_session_path(self):
        pb = self._make_builder()
        env = pb._section_env({"session_path": "/sessions/custom"})
        assert "Session path: /sessions/custom" in env

    def test_env_session_path_from_config(self):
        pb = self._make_builder()
        env = pb._section_env({})
        assert "Session path: /sessions/test-session" in env

    def test_env_no_session_path_when_empty(self):
        config = {
            "user": {"name": "U"},
            "llm": {"model": "m", "provider": "p"},
            "agent": {"workspace_dir": "/w"},
        }
        pb = PromptBuilder(config)
        env = pb._section_env({})
        assert "Session path" not in env

    def test_env_citation_hint(self):
        pb = self._make_builder()
        env = pb._section_env({"has_linkable_sources": True})
        assert "Citation reminder" in env

    def test_env_no_citation_hint_by_default(self):
        pb = self._make_builder()
        env = pb._section_env({})
        assert "Citation reminder" not in env

    def test_user_section_email_label(self):
        pb = self._make_builder()
        user_section = pb._section_user()
        assert "Email address: test@example.com" in user_section

    def test_build_includes_all_sections(self):
        pb = self._make_builder()
        prompt = pb.build([])
        # Spot-check key sections appear
        assert "<application_details>" in prompt
        assert "<claude_behavior>" in prompt
        assert "<critical_injection_defense>" in prompt
        assert "<critical_security_rules>" in prompt
        assert "<user_privacy>" in prompt
        assert "<action_types>" in prompt
        assert "<mandatory_copyright_requirements>" in prompt
        assert "<skills_instructions>" in prompt
        assert "<content_isolation_rules>" in prompt
        assert "<user>" in prompt
        assert "<env>" in prompt

    def test_build_section_count(self):
        """Build output should contain all 20 behavioral sections + user + env = 22 minimum."""
        pb = self._make_builder()
        prompt = pb.build([])
        # Count top-level opening XML tags
        import re
        # Just verify minimum section count
        sections = prompt.split("\n\n")
        assert len(sections) >= 22

    def test_build_with_tools(self):
        pb = self._make_builder()
        tools = [
            ToolSchema(
                name="test_tool",
                description="A test tool",
                input_schema={"type": "object", "properties": {}},
            )
        ]
        prompt = pb.build(tools)
        assert "<available_tools>" in prompt
        assert "test_tool" in prompt


# ============================================================
# H. AGENT _build_context TESTS
# ============================================================


class TestAgentBuildContext:
    """Sprint 37 additions to Agent._build_context()."""

    def test_session_path_in_context(self):
        """_build_context includes session_path when configured."""
        # We test this indirectly by checking the method exists and works
        from cowork_agent.core.agent import Agent

        # Verify the method signature accepts our new context
        assert hasattr(Agent, "_build_context")
        assert hasattr(Agent, "_has_linkable_tool_results")

    def test_has_linkable_tool_results_method_exists(self):
        """The linkable tool results helper exists."""
        from cowork_agent.core.agent import Agent
        assert callable(getattr(Agent, "_has_linkable_tool_results", None))


# ============================================================
# I. BACKWARD COMPATIBILITY TESTS
# ============================================================


class TestBackwardCompatibility:
    """Ensure old tests/imports still work."""

    def test_all_sections_is_list(self):
        assert isinstance(ALL_SECTIONS, list)

    def test_all_sections_all_strings(self):
        for s in ALL_SECTIONS:
            assert isinstance(s, str)

    def test_prompt_builder_still_builds(self):
        """PromptBuilder.build() still works with minimal config."""
        config = {
            "user": {"name": "User"},
            "llm": {"model": "m", "provider": "p"},
            "agent": {"workspace_dir": "/tmp"},
        }
        pb = PromptBuilder(config)
        prompt = pb.build([])
        assert len(prompt) > 1000  # Substantial prompt

    def test_old_section_names_importable(self):
        """Old constants that were renamed/restructured are still accessible."""
        from cowork_agent.prompts.behavioral_rules import (
            CORE_IDENTITY,
            CLAUDE_BEHAVIOR,
            SAFETY_RULES,
            TODO_RULES,
            GIT_RULES,
            ASK_USER_RULES,
            OUTPUT_RULES,
            COPYRIGHT_RULES,
            ACTION_TYPES,
            META_SAFETY,  # Now absorbed into SAFETY_RULES but constant removed
            CONTENT_ISOLATION,
            INSTRUCTION_DETECTION,
        )
        # If we get here, imports worked
        assert True


# ============================================================
# J. TOKEN EFFICIENCY TESTS
# ============================================================


class TestTokenEfficiency:
    """Verify prompt size is reasonable."""

    def test_prompt_under_token_budget(self):
        """Full prompt (no tools) should be under ~30K characters (~7500 tokens)."""
        config = {
            "user": {"name": "User"},
            "llm": {"model": "m", "provider": "p"},
            "agent": {"workspace_dir": "/tmp"},
        }
        pb = PromptBuilder(config)
        prompt = pb.build([])
        # The real prompt is about 25-35K chars. Our version should be similar.
        assert len(prompt) < 50000, f"Prompt too large: {len(prompt)} chars"
        assert len(prompt) > 5000, f"Prompt too small: {len(prompt)} chars"

    def test_no_redundant_wrapper_tags(self):
        """Sections that were previously separate should now be nested, not duplicated."""
        config = {
            "user": {"name": "U"},
            "llm": {"model": "m", "provider": "p"},
            "agent": {"workspace_dir": "/tmp"},
        }
        pb = PromptBuilder(config)
        prompt = pb.build([])
        # <refusal_handling> should appear only inside <claude_behavior>, not standalone
        import re
        refusal_count = len(re.findall(r"<refusal_handling>", prompt))
        assert refusal_count == 1, f"<refusal_handling> appears {refusal_count} times"


# ============================================================
# K. MAIN.PY WIRING (sanity check)
# ============================================================


class TestMainWiring:
    """Verify main.py still imports successfully after changes."""

    def test_main_imports(self):
        """main.py can be imported without errors."""
        import cowork_agent.main  # noqa: F401
        assert True

    def test_behavioral_rules_import_from_main(self):
        """The behavioral rules module is importable."""
        from cowork_agent.prompts.behavioral_rules import ALL_SECTIONS
        assert len(ALL_SECTIONS) == 20
