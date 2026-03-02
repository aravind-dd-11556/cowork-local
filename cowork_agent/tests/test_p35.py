"""
Sprint 35 Tests — Browser Automation Extras.

Tests for:
  - GifCreatorTool
  - ShortcutsListTool
  - ShortcutsExecuteTool
  - SwitchBrowserTool
  - Main.py wiring
  - Edge cases

~65 tests total.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

from cowork_agent.core.browser_session import BrowserSession
from cowork_agent.tools.browser_tools_extra import (
    GifCreatorTool,
    ShortcutsListTool,
    ShortcutsExecuteTool,
    SwitchBrowserTool,
)


def run(coro):
    """Run async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_session_with_tab():
    """Create a browser session with one tab, return (session, tab_id)."""
    session = BrowserSession()
    group = session.get_or_create_group()
    tab_id = list(group.tabs.keys())[0]
    return session, tab_id


# ═══════════════════════════════════════════════════════════════════════
# GifCreatorTool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestGifCreatorTool(unittest.TestCase):
    """Tests for GifCreatorTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.session, self.tab_id = _make_session_with_tab()
        self.tool = GifCreatorTool(browser_session=self.session)

    def test_no_session(self):
        """Test error when no session."""
        tool = GifCreatorTool()
        result = run(tool.execute(action="start_recording", tabId=1))
        self.assertFalse(result.success)

    def test_no_action(self):
        """Test error when no action."""
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertFalse(result.success)

    def test_invalid_action(self):
        """Test error for invalid action."""
        result = run(self.tool.execute(action="invalid", tabId=self.tab_id))
        self.assertFalse(result.success)
        self.assertIn("Invalid action", result.error)

    def test_no_tab_id(self):
        """Test error when no tabId."""
        result = run(self.tool.execute(action="start_recording"))
        self.assertFalse(result.success)

    def test_invalid_tab(self):
        """Test error for invalid tab."""
        result = run(self.tool.execute(action="start_recording", tabId=99999))
        self.assertFalse(result.success)

    def test_start_recording(self):
        """Test starting a recording."""
        result = run(self.tool.execute(action="start_recording", tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertIn("Recording started", result.output)
        self.assertEqual(result.metadata["status"], "recording")

    def test_stop_recording(self):
        """Test stopping a recording."""
        run(self.tool.execute(action="start_recording", tabId=self.tab_id))
        result = run(self.tool.execute(action="stop_recording", tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertIn("stopped", result.output)
        self.assertEqual(result.metadata["status"], "stopped")

    def test_stop_without_start(self):
        """Test stopping when no recording active."""
        result = run(self.tool.execute(action="stop_recording", tabId=self.tab_id))
        self.assertFalse(result.success)
        self.assertIn("No active recording", result.error)

    def test_export(self):
        """Test exporting a recording."""
        run(self.tool.execute(action="start_recording", tabId=self.tab_id))
        run(self.tool.execute(action="stop_recording", tabId=self.tab_id))
        result = run(self.tool.execute(action="export", tabId=self.tab_id, download=True))
        self.assertTrue(result.success)
        self.assertIn("exported", result.output)
        self.assertTrue(result.metadata["download"])

    def test_export_custom_filename(self):
        """Test exporting with custom filename."""
        run(self.tool.execute(action="start_recording", tabId=self.tab_id))
        run(self.tool.execute(action="stop_recording", tabId=self.tab_id))
        result = run(self.tool.execute(
            action="export", tabId=self.tab_id, filename="my-recording.gif",
        ))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["filename"], "my-recording.gif")

    def test_export_with_options(self):
        """Test exporting with enhancement options."""
        run(self.tool.execute(action="start_recording", tabId=self.tab_id))
        run(self.tool.execute(action="stop_recording", tabId=self.tab_id))
        opts = {"showClickIndicators": True, "quality": 5}
        result = run(self.tool.execute(
            action="export", tabId=self.tab_id, options=opts,
        ))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["options_applied"], opts)

    def test_export_no_recording(self):
        """Test export when no recording exists."""
        result = run(self.tool.execute(action="export", tabId=self.tab_id))
        self.assertFalse(result.success)
        self.assertIn("No recording data", result.error)

    def test_export_while_recording(self):
        """Test export while still recording."""
        run(self.tool.execute(action="start_recording", tabId=self.tab_id))
        result = run(self.tool.execute(action="export", tabId=self.tab_id))
        self.assertFalse(result.success)
        self.assertIn("Stop recording", result.error)

    def test_clear(self):
        """Test clearing recording data."""
        run(self.tool.execute(action="start_recording", tabId=self.tab_id))
        run(self.tool.execute(action="stop_recording", tabId=self.tab_id))
        result = run(self.tool.execute(action="clear", tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertIn("cleared", result.output)

    def test_clear_no_data(self):
        """Test clearing when nothing recorded — should still succeed."""
        result = run(self.tool.execute(action="clear", tabId=self.tab_id))
        self.assertTrue(result.success)

    def test_real_handler(self):
        """Test with real GIF handler callback."""
        handler = MagicMock(return_value={"status": "ok", "url": "blob://gif"})
        tool = GifCreatorTool(browser_session=self.session, gif_handler=handler)
        result = run(tool.execute(action="start_recording", tabId=self.tab_id))
        self.assertTrue(result.success)
        handler.assert_called_once()

    def test_real_handler_error(self):
        """Test real handler raising an error."""
        handler = MagicMock(side_effect=RuntimeError("GIF error"))
        tool = GifCreatorTool(browser_session=self.session, gif_handler=handler)
        result = run(tool.execute(action="start_recording", tabId=self.tab_id))
        self.assertFalse(result.success)
        self.assertIn("GIF handler error", result.error)

    def test_tool_name(self):
        """Test tool name is correct."""
        self.assertEqual(self.tool.name, "gif_creator")


# ═══════════════════════════════════════════════════════════════════════
# ShortcutsListTool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestShortcutsListTool(unittest.TestCase):
    """Tests for ShortcutsListTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.session, self.tab_id = _make_session_with_tab()
        self.tool = ShortcutsListTool(browser_session=self.session)

    def test_no_session(self):
        """Test error when no session."""
        tool = ShortcutsListTool()
        result = run(tool.execute(tabId=1))
        self.assertFalse(result.success)

    def test_no_tab_id(self):
        """Test error when no tabId."""
        result = run(self.tool.execute())
        self.assertFalse(result.success)

    def test_invalid_tab(self):
        """Test error for invalid tab."""
        result = run(self.tool.execute(tabId=99999))
        self.assertFalse(result.success)

    def test_list_shortcuts(self):
        """Test listing simulated shortcuts."""
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["shortcut_count"], 3)
        self.assertIn("debug", result.output)
        self.assertIn("summarize", result.output)
        self.assertIn("run-tests", result.output)

    def test_shortcuts_include_workflows(self):
        """Test that workflows are indicated."""
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertIn("workflow", result.output)
        self.assertIn("shortcut", result.output)

    def test_shortcuts_metadata(self):
        """Test shortcuts are in metadata."""
        result = run(self.tool.execute(tabId=self.tab_id))
        shortcuts = result.metadata["shortcuts"]
        self.assertEqual(len(shortcuts), 3)
        ids = {sc["shortcutId"] for sc in shortcuts}
        self.assertIn("sc_debug", ids)

    def test_real_provider(self):
        """Test with real shortcuts provider."""
        provider = MagicMock(return_value=[
            {"command": "custom", "description": "Custom shortcut"},
        ])
        tool = ShortcutsListTool(
            browser_session=self.session, shortcuts_provider=provider,
        )
        result = run(tool.execute(tabId=self.tab_id))
        self.assertTrue(result.success)
        provider.assert_called_once_with(self.tab_id)

    def test_real_provider_error(self):
        """Test real provider error."""
        provider = MagicMock(side_effect=RuntimeError("Provider error"))
        tool = ShortcutsListTool(
            browser_session=self.session, shortcuts_provider=provider,
        )
        result = run(tool.execute(tabId=self.tab_id))
        self.assertFalse(result.success)
        self.assertIn("Shortcuts provider error", result.error)

    def test_tool_name(self):
        """Test tool name is correct."""
        self.assertEqual(self.tool.name, "shortcuts_list")


# ═══════════════════════════════════════════════════════════════════════
# ShortcutsExecuteTool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestShortcutsExecuteTool(unittest.TestCase):
    """Tests for ShortcutsExecuteTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.session, self.tab_id = _make_session_with_tab()
        self.tool = ShortcutsExecuteTool(browser_session=self.session)

    def test_no_session(self):
        """Test error when no session."""
        tool = ShortcutsExecuteTool()
        result = run(tool.execute(tabId=1, command="debug"))
        self.assertFalse(result.success)

    def test_no_tab_id(self):
        """Test error when no tabId."""
        result = run(self.tool.execute(command="debug"))
        self.assertFalse(result.success)

    def test_no_command_or_id(self):
        """Test error when neither command nor shortcutId provided."""
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertFalse(result.success)
        self.assertIn("Either", result.error)

    def test_invalid_tab(self):
        """Test error for invalid tab."""
        result = run(self.tool.execute(tabId=99999, command="debug"))
        self.assertFalse(result.success)

    def test_execute_by_command(self):
        """Test executing by command name."""
        result = run(self.tool.execute(tabId=self.tab_id, command="debug"))
        self.assertTrue(result.success)
        self.assertIn("debug", result.output)
        self.assertEqual(result.metadata["status"], "started")
        self.assertIsNotNone(result.metadata["execution_id"])

    def test_execute_by_shortcut_id(self):
        """Test executing by shortcut ID."""
        result = run(self.tool.execute(tabId=self.tab_id, shortcutId="sc_debug"))
        self.assertTrue(result.success)
        self.assertIn("sc_debug", result.output)

    def test_execute_both_command_and_id(self):
        """Test with both command and shortcutId — should use command."""
        result = run(self.tool.execute(
            tabId=self.tab_id, command="debug", shortcutId="sc_debug",
        ))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["command"], "debug")
        self.assertEqual(result.metadata["shortcut_id"], "sc_debug")

    def test_real_executor(self):
        """Test with real shortcut executor."""
        executor = MagicMock(return_value={"status": "running"})
        tool = ShortcutsExecuteTool(
            browser_session=self.session, shortcut_executor=executor,
        )
        result = run(tool.execute(tabId=self.tab_id, command="debug"))
        self.assertTrue(result.success)
        executor.assert_called_once()

    def test_real_executor_error(self):
        """Test real executor error."""
        executor = MagicMock(side_effect=RuntimeError("Exec error"))
        tool = ShortcutsExecuteTool(
            browser_session=self.session, shortcut_executor=executor,
        )
        result = run(tool.execute(tabId=self.tab_id, command="debug"))
        self.assertFalse(result.success)
        self.assertIn("Shortcut execution error", result.error)

    def test_tool_name(self):
        """Test tool name is correct."""
        self.assertEqual(self.tool.name, "shortcuts_execute")


# ═══════════════════════════════════════════════════════════════════════
# SwitchBrowserTool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSwitchBrowserTool(unittest.TestCase):
    """Tests for SwitchBrowserTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.session, self.tab_id = _make_session_with_tab()
        self.tool = SwitchBrowserTool(browser_session=self.session)

    def test_switch_simulated(self):
        """Test simulated browser switch."""
        result = run(self.tool.execute())
        self.assertTrue(result.success)
        self.assertIn("Connection request", result.output)
        self.assertEqual(result.metadata["status"], "awaiting_connection")
        self.assertIn("request_id", result.metadata)

    def test_switch_no_session(self):
        """Test switch works without session (it's optional for switch)."""
        tool = SwitchBrowserTool()
        result = run(tool.execute())
        self.assertTrue(result.success)

    def test_real_handler(self):
        """Test with real switch handler."""
        handler = MagicMock(return_value="Connected to Chrome 2")
        tool = SwitchBrowserTool(browser_session=self.session, switch_handler=handler)
        result = run(tool.execute())
        self.assertTrue(result.success)
        handler.assert_called_once()

    def test_real_handler_error(self):
        """Test real handler error."""
        handler = MagicMock(side_effect=RuntimeError("Connection failed"))
        tool = SwitchBrowserTool(browser_session=self.session, switch_handler=handler)
        result = run(tool.execute())
        self.assertFalse(result.success)
        self.assertIn("Browser switch error", result.error)

    def test_tool_name(self):
        """Test tool name is correct."""
        self.assertEqual(self.tool.name, "switch_browser")

    def test_empty_schema(self):
        """Test input schema has no required fields."""
        self.assertEqual(self.tool.input_schema.get("required", []), [])


# ═══════════════════════════════════════════════════════════════════════
# Tool Schemas
# ═══════════════════════════════════════════════════════════════════════


class TestToolSchemas(unittest.TestCase):
    """Tests for tool schema validation."""

    def test_all_tools_have_schemas(self):
        """All extra tools have valid input_schema."""
        session = BrowserSession()
        session.get_or_create_group()
        tools = [
            GifCreatorTool(browser_session=session),
            ShortcutsListTool(browser_session=session),
            ShortcutsExecuteTool(browser_session=session),
            SwitchBrowserTool(browser_session=session),
        ]
        for tool in tools:
            self.assertIsInstance(tool.input_schema, dict, f"{tool.name} missing schema")
            self.assertEqual(tool.input_schema["type"], "object")

    def test_tool_names_unique(self):
        """All tool names are unique."""
        session = BrowserSession()
        session.get_or_create_group()
        tools = [
            GifCreatorTool(browser_session=session),
            ShortcutsListTool(browser_session=session),
            ShortcutsExecuteTool(browser_session=session),
            SwitchBrowserTool(browser_session=session),
        ]
        names = [t.name for t in tools]
        self.assertEqual(len(names), len(set(names)))

    def test_expected_tool_names(self):
        """Verify the expected set of tool names."""
        session = BrowserSession()
        session.get_or_create_group()
        tools = [
            GifCreatorTool(browser_session=session),
            ShortcutsListTool(browser_session=session),
            ShortcutsExecuteTool(browser_session=session),
            SwitchBrowserTool(browser_session=session),
        ]
        names = {t.name for t in tools}
        expected = {
            "gif_creator", "shortcuts_list",
            "shortcuts_execute", "switch_browser",
        }
        self.assertEqual(names, expected)


# ═══════════════════════════════════════════════════════════════════════
# Main Wiring
# ═══════════════════════════════════════════════════════════════════════


class TestMainWiring(unittest.TestCase):
    """Tests for main.py wiring of Sprint 35 tools."""

    def test_extras_wiring(self):
        """Verify Sprint 35 tools are wired in main.py."""
        import inspect
        from cowork_agent import main as main_mod

        source = inspect.getsource(main_mod)
        self.assertIn("GifCreatorTool", source)
        self.assertIn("ShortcutsListTool", source)
        self.assertIn("ShortcutsExecuteTool", source)
        self.assertIn("SwitchBrowserTool", source)
        self.assertIn("browser_tools_extra", source)


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases(unittest.TestCase):
    """Edge case tests for Sprint 35 tools."""

    def test_gif_full_lifecycle(self):
        """Test complete GIF lifecycle: start -> stop -> export -> clear."""
        session, tab_id = _make_session_with_tab()
        tool = GifCreatorTool(browser_session=session)

        # Start
        r1 = run(tool.execute(action="start_recording", tabId=tab_id))
        self.assertTrue(r1.success)

        # Stop
        r2 = run(tool.execute(action="stop_recording", tabId=tab_id))
        self.assertTrue(r2.success)

        # Export
        r3 = run(tool.execute(action="export", tabId=tab_id, download=True))
        self.assertTrue(r3.success)

        # Clear
        r4 = run(tool.execute(action="clear", tabId=tab_id))
        self.assertTrue(r4.success)

        # Export after clear should fail
        r5 = run(tool.execute(action="export", tabId=tab_id))
        self.assertFalse(r5.success)

    def test_gif_restart_recording(self):
        """Test restarting a recording overwrites the previous one."""
        session, tab_id = _make_session_with_tab()
        tool = GifCreatorTool(browser_session=session)

        run(tool.execute(action="start_recording", tabId=tab_id))
        run(tool.execute(action="stop_recording", tabId=tab_id))

        # Start again — should overwrite
        r = run(tool.execute(action="start_recording", tabId=tab_id))
        self.assertTrue(r.success)
        self.assertEqual(r.metadata["status"], "recording")

    def test_shortcut_execution_id_unique(self):
        """Test each execution gets a unique ID."""
        session, tab_id = _make_session_with_tab()
        tool = ShortcutsExecuteTool(browser_session=session)

        r1 = run(tool.execute(tabId=tab_id, command="debug"))
        r2 = run(tool.execute(tabId=tab_id, command="debug"))
        self.assertNotEqual(
            r1.metadata["execution_id"],
            r2.metadata["execution_id"],
        )

    def test_switch_browser_request_id_unique(self):
        """Test each switch request gets a unique ID."""
        tool = SwitchBrowserTool()
        r1 = run(tool.execute())
        r2 = run(tool.execute())
        self.assertNotEqual(
            r1.metadata["request_id"],
            r2.metadata["request_id"],
        )

    def test_shortcuts_list_output_format(self):
        """Test shortcuts list has consistent format."""
        session, tab_id = _make_session_with_tab()
        tool = ShortcutsListTool(browser_session=session)
        result = run(tool.execute(tabId=tab_id))
        self.assertIn("/debug", result.output)
        self.assertIn("/summarize", result.output)
        self.assertIn("/run-tests", result.output)

    def test_gif_default_filename(self):
        """Test export generates a default filename."""
        session, tab_id = _make_session_with_tab()
        tool = GifCreatorTool(browser_session=session)
        run(tool.execute(action="start_recording", tabId=tab_id))
        run(tool.execute(action="stop_recording", tabId=tab_id))
        result = run(tool.execute(action="export", tabId=tab_id))
        self.assertTrue(result.metadata["filename"].startswith("recording-"))
        self.assertTrue(result.metadata["filename"].endswith(".gif"))


if __name__ == "__main__":
    unittest.main()
