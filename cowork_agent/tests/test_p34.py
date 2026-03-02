"""
Sprint 34 Tests — Browser Automation Extended.

Tests for:
  - JavaScriptTool
  - GetPageTextTool
  - ReadConsoleMessagesTool
  - ReadNetworkRequestsTool
  - UploadImageTool
  - ResizeWindowTool
  - Main.py wiring
  - Edge cases

~90 tests total.
"""

import asyncio
import unittest
from unittest.mock import MagicMock

from cowork_agent.core.browser_session import (
    AccessibilityNode,
    BrowserSession,
)
from cowork_agent.tools.browser_tools_ext import (
    GetPageTextTool,
    JavaScriptTool,
    ReadConsoleMessagesTool,
    ReadNetworkRequestsTool,
    ResizeWindowTool,
    UploadImageTool,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_session_with_tab():
    """Create a browser session with one tab, return (session, tab_id)."""
    session = BrowserSession()
    group = session.get_or_create_group()
    tab_id = list(group.tabs.keys())[0]
    return session, tab_id


def _make_text_tree():
    """Create a tree with text content for extraction."""
    return AccessibilityNode(
        ref_id="ref_0", role="document", name="",
        children=[
            AccessibilityNode(
                ref_id="ref_1", role="heading", name="Article Title",
            ),
            AccessibilityNode(
                ref_id="ref_2", role="paragraph", name="First paragraph of content.",
            ),
            AccessibilityNode(
                ref_id="ref_3", role="paragraph", name="Second paragraph here.",
            ),
            AccessibilityNode(
                ref_id="ref_4", role="link", name="Read more",
                interactive=True,
            ),
            AccessibilityNode(
                ref_id="ref_5", role="div", name="Hidden",
                visible=False,
                children=[
                    AccessibilityNode(
                        ref_id="ref_6", role="paragraph", name="Hidden text",
                        visible=False,
                    ),
                ],
            ),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════
# JavaScriptTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestJavaScriptTool(unittest.TestCase):

    def setUp(self):
        self.session, self.tab_id = _make_session_with_tab()
        self.tool = JavaScriptTool(browser_session=self.session)

    def test_no_session(self):
        tool = JavaScriptTool()
        result = run(tool.execute(action="javascript_exec", text="1+1", tabId=1))
        self.assertFalse(result.success)

    def test_no_text(self):
        result = run(self.tool.execute(action="javascript_exec", text="", tabId=self.tab_id))
        self.assertFalse(result.success)

    def test_no_tab_id(self):
        result = run(self.tool.execute(action="javascript_exec", text="1+1"))
        self.assertFalse(result.success)

    def test_invalid_tab(self):
        result = run(self.tool.execute(action="javascript_exec", text="1+1", tabId=9999))
        self.assertFalse(result.success)

    def test_simple_expression(self):
        result = run(self.tool.execute(
            action="javascript_exec", text="2+3", tabId=self.tab_id,
        ))
        self.assertTrue(result.success)
        self.assertIn("5", result.output)

    def test_document_title(self):
        result = run(self.tool.execute(
            action="javascript_exec", text="document.title", tabId=self.tab_id,
        ))
        self.assertTrue(result.success)

    def test_window_location(self):
        result = run(self.tool.execute(
            action="javascript_exec", text="window.location.href", tabId=self.tab_id,
        ))
        self.assertTrue(result.success)

    def test_complex_code_simulated(self):
        result = run(self.tool.execute(
            action="javascript_exec",
            text="Array.from(document.querySelectorAll('a')).length",
            tabId=self.tab_id,
        ))
        self.assertTrue(result.success)

    def test_code_size_limit(self):
        big_code = "x" * 100_001
        result = run(self.tool.execute(
            action="javascript_exec", text=big_code, tabId=self.tab_id,
        ))
        self.assertFalse(result.success)
        self.assertIn("maximum size", result.error)

    def test_real_executor(self):
        executor = MagicMock(return_value=42)
        tool = JavaScriptTool(browser_session=self.session, js_executor=executor)
        result = run(tool.execute(
            action="javascript_exec", text="getAnswer()", tabId=self.tab_id,
        ))
        self.assertTrue(result.success)
        self.assertIn("42", result.output)
        executor.assert_called_once_with(self.tab_id, "getAnswer()")

    def test_real_executor_error(self):
        executor = MagicMock(side_effect=Exception("ReferenceError"))
        tool = JavaScriptTool(browser_session=self.session, js_executor=executor)
        result = run(tool.execute(
            action="javascript_exec", text="badCode()", tabId=self.tab_id,
        ))
        self.assertFalse(result.success)
        self.assertIn("ReferenceError", result.error)

    def test_real_executor_returns_none(self):
        executor = MagicMock(return_value=None)
        tool = JavaScriptTool(browser_session=self.session, js_executor=executor)
        result = run(tool.execute(
            action="javascript_exec", text="void 0", tabId=self.tab_id,
        ))
        self.assertTrue(result.success)
        self.assertIn("undefined", result.output)

    def test_tool_name(self):
        self.assertEqual(self.tool.name, "javascript_tool")


# ═══════════════════════════════════════════════════════════════════════
# GetPageTextTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGetPageTextTool(unittest.TestCase):

    def setUp(self):
        self.session, self.tab_id = _make_session_with_tab()
        self.tool = GetPageTextTool(browser_session=self.session)

    def test_no_session(self):
        tool = GetPageTextTool()
        result = run(tool.execute(tabId=1))
        self.assertFalse(result.success)

    def test_no_tab_id(self):
        result = run(self.tool.execute())
        self.assertFalse(result.success)

    def test_blank_page(self):
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertIn("empty", result.output.lower())

    def test_page_with_tree(self):
        self.session.set_accessibility_tree(self.tab_id, _make_text_tree())
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertIn("Article Title", result.output)
        self.assertIn("First paragraph", result.output)

    def test_hidden_text_excluded(self):
        self.session.set_accessibility_tree(self.tab_id, _make_text_tree())
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertNotIn("Hidden text", result.output)

    def test_navigated_page_no_tree(self):
        self.session.navigate(self.tab_id, "https://example.com")
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertIn("example.com", result.output)

    def test_real_extractor(self):
        extractor = MagicMock(return_value="Extracted content here")
        tool = GetPageTextTool(browser_session=self.session, text_extractor=extractor)
        result = run(tool.execute(tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertEqual(result.output, "Extracted content here")

    def test_real_extractor_error(self):
        extractor = MagicMock(side_effect=Exception("DOM error"))
        tool = GetPageTextTool(browser_session=self.session, text_extractor=extractor)
        result = run(tool.execute(tabId=self.tab_id))
        self.assertFalse(result.success)

    def test_tool_name(self):
        self.assertEqual(self.tool.name, "get_page_text")


# ═══════════════════════════════════════════════════════════════════════
# ReadConsoleMessagesTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestReadConsoleMessagesTool(unittest.TestCase):

    def setUp(self):
        self.session, self.tab_id = _make_session_with_tab()
        self.tool = ReadConsoleMessagesTool(browser_session=self.session)
        # Populate console messages
        tab = self.session.get_tab(self.tab_id)
        tab.console_messages = [
            {"level": "log", "text": "App started", "timestamp": "12:00:01"},
            {"level": "warn", "text": "Deprecation warning: use v2 API", "timestamp": "12:00:02"},
            {"level": "error", "text": "Failed to fetch /api/data", "timestamp": "12:00:03"},
            {"level": "log", "text": "User clicked button", "timestamp": "12:00:04"},
            {"level": "error", "text": "TypeError: cannot read property", "timestamp": "12:00:05"},
            {"level": "exception", "text": "Uncaught ReferenceError", "timestamp": "12:00:06"},
        ]

    def test_no_session(self):
        tool = ReadConsoleMessagesTool()
        result = run(tool.execute(tabId=1))
        self.assertFalse(result.success)

    def test_no_tab_id(self):
        result = run(self.tool.execute())
        self.assertFalse(result.success)

    def test_read_all(self):
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["message_count"], 6)

    def test_errors_only(self):
        result = run(self.tool.execute(tabId=self.tab_id, onlyErrors=True))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["message_count"], 3)  # 2 errors + 1 exception

    def test_pattern_filter(self):
        result = run(self.tool.execute(tabId=self.tab_id, pattern="TypeError"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["message_count"], 1)

    def test_pattern_regex(self):
        result = run(self.tool.execute(tabId=self.tab_id, pattern="error|warning"))
        self.assertTrue(result.success)
        # Matches: "Deprecation warning", "Failed to fetch", "TypeError", "ReferenceError"
        self.assertGreater(result.metadata["message_count"], 0)

    def test_invalid_pattern(self):
        result = run(self.tool.execute(tabId=self.tab_id, pattern="[invalid"))
        self.assertFalse(result.success)
        self.assertIn("Invalid regex", result.error)

    def test_limit(self):
        result = run(self.tool.execute(tabId=self.tab_id, limit=2))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["message_count"], 2)

    def test_clear(self):
        result = run(self.tool.execute(tabId=self.tab_id, clear=True))
        self.assertTrue(result.success)
        tab = self.session.get_tab(self.tab_id)
        self.assertEqual(len(tab.console_messages), 0)

    def test_empty_messages(self):
        tab = self.session.get_tab(self.tab_id)
        tab.console_messages = []
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["message_count"], 0)

    def test_tool_name(self):
        self.assertEqual(self.tool.name, "read_console_messages")


# ═══════════════════════════════════════════════════════════════════════
# ReadNetworkRequestsTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestReadNetworkRequestsTool(unittest.TestCase):

    def setUp(self):
        self.session, self.tab_id = _make_session_with_tab()
        self.tool = ReadNetworkRequestsTool(browser_session=self.session)
        # Populate network requests
        tab = self.session.get_tab(self.tab_id)
        tab.network_requests = [
            {"method": "GET", "url": "https://example.com/", "status": 200, "type": "document", "size": 5000},
            {"method": "GET", "url": "https://cdn.example.com/style.css", "status": 200, "type": "stylesheet", "size": 1200},
            {"method": "GET", "url": "https://example.com/api/data", "status": 200, "type": "xhr", "size": 3400},
            {"method": "POST", "url": "https://example.com/api/submit", "status": 201, "type": "xhr", "size": 100},
            {"method": "GET", "url": "https://cdn.example.com/image.png", "status": 200, "type": "image", "size": 50000},
        ]

    def test_no_session(self):
        tool = ReadNetworkRequestsTool()
        result = run(tool.execute(tabId=1))
        self.assertFalse(result.success)

    def test_no_tab_id(self):
        result = run(self.tool.execute())
        self.assertFalse(result.success)

    def test_read_all(self):
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["request_count"], 5)

    def test_url_pattern(self):
        result = run(self.tool.execute(tabId=self.tab_id, urlPattern="/api/"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["request_count"], 2)  # data + submit

    def test_url_pattern_case_insensitive(self):
        result = run(self.tool.execute(tabId=self.tab_id, urlPattern="/API/"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["request_count"], 2)

    def test_limit(self):
        result = run(self.tool.execute(tabId=self.tab_id, limit=3))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["request_count"], 3)

    def test_clear(self):
        result = run(self.tool.execute(tabId=self.tab_id, clear=True))
        self.assertTrue(result.success)
        tab = self.session.get_tab(self.tab_id)
        self.assertEqual(len(tab.network_requests), 0)

    def test_empty_requests(self):
        tab = self.session.get_tab(self.tab_id)
        tab.network_requests = []
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["request_count"], 0)

    def test_output_format(self):
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertIn("GET", result.output)
        self.assertIn("POST", result.output)
        self.assertIn("/api/", result.output)

    def test_tool_name(self):
        self.assertEqual(self.tool.name, "read_network_requests")


# ═══════════════════════════════════════════════════════════════════════
# UploadImageTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUploadImageTool(unittest.TestCase):

    def setUp(self):
        self.session, self.tab_id = _make_session_with_tab()
        self.tool = UploadImageTool(browser_session=self.session)

    def test_no_session(self):
        tool = UploadImageTool()
        result = run(tool.execute(imageId="img_1", tabId=1, ref="ref_1"))
        self.assertFalse(result.success)

    def test_no_image_id(self):
        result = run(self.tool.execute(imageId="", tabId=self.tab_id, ref="ref_1"))
        self.assertFalse(result.success)

    def test_no_tab_id(self):
        result = run(self.tool.execute(imageId="img_1", ref="ref_1"))
        self.assertFalse(result.success)

    def test_no_target(self):
        result = run(self.tool.execute(imageId="img_1", tabId=self.tab_id))
        self.assertFalse(result.success)
        self.assertIn("ref", result.error)

    def test_both_targets(self):
        result = run(self.tool.execute(
            imageId="img_1", tabId=self.tab_id,
            ref="ref_1", coordinate=[100, 200],
        ))
        self.assertFalse(result.success)
        self.assertIn("not both", result.error)

    def test_upload_via_ref(self):
        result = run(self.tool.execute(
            imageId="screenshot_abc123", tabId=self.tab_id, ref="ref_1",
        ))
        self.assertTrue(result.success)
        self.assertIn("ref_1", result.output)
        self.assertIn("screenshot_abc123", result.output)

    def test_upload_via_coordinate(self):
        result = run(self.tool.execute(
            imageId="img_42", tabId=self.tab_id, coordinate=[300, 400],
        ))
        self.assertTrue(result.success)
        self.assertIn("drag & drop", result.output)

    def test_custom_filename(self):
        result = run(self.tool.execute(
            imageId="img_1", tabId=self.tab_id, ref="ref_1",
            filename="my_photo.jpg",
        ))
        self.assertTrue(result.success)
        self.assertIn("my_photo.jpg", result.output)

    def test_real_upload_handler(self):
        handler = MagicMock(return_value="Upload successful")
        tool = UploadImageTool(browser_session=self.session, upload_handler=handler)
        result = run(tool.execute(
            imageId="img_1", tabId=self.tab_id, ref="ref_input",
        ))
        self.assertTrue(result.success)
        handler.assert_called_once()

    def test_real_upload_handler_error(self):
        handler = MagicMock(side_effect=Exception("File too large"))
        tool = UploadImageTool(browser_session=self.session, upload_handler=handler)
        result = run(tool.execute(
            imageId="img_1", tabId=self.tab_id, ref="ref_1",
        ))
        self.assertFalse(result.success)

    def test_invalid_tab(self):
        result = run(self.tool.execute(
            imageId="img_1", tabId=9999, ref="ref_1",
        ))
        self.assertFalse(result.success)

    def test_tool_name(self):
        self.assertEqual(self.tool.name, "upload_image")


# ═══════════════════════════════════════════════════════════════════════
# ResizeWindowTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestResizeWindowTool(unittest.TestCase):

    def setUp(self):
        self.session, self.tab_id = _make_session_with_tab()
        self.tool = ResizeWindowTool(browser_session=self.session)

    def test_no_session(self):
        tool = ResizeWindowTool()
        result = run(tool.execute(tabId=1, width=1024, height=768))
        self.assertFalse(result.success)

    def test_no_tab_id(self):
        result = run(self.tool.execute(width=1024, height=768))
        self.assertFalse(result.success)

    def test_no_dimensions(self):
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertFalse(result.success)

    def test_resize_success(self):
        result = run(self.tool.execute(tabId=self.tab_id, width=1024, height=768))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["width"], 1024)
        self.assertEqual(result.metadata["height"], 768)

    def test_resize_mobile(self):
        result = run(self.tool.execute(tabId=self.tab_id, width=375, height=812))
        self.assertTrue(result.success)

    def test_resize_too_small(self):
        result = run(self.tool.execute(tabId=self.tab_id, width=50, height=50))
        self.assertFalse(result.success)

    def test_resize_too_large(self):
        result = run(self.tool.execute(tabId=self.tab_id, width=10000, height=10000))
        self.assertFalse(result.success)

    def test_invalid_tab(self):
        result = run(self.tool.execute(tabId=9999, width=1024, height=768))
        self.assertFalse(result.success)

    def test_tool_name(self):
        self.assertEqual(self.tool.name, "resize_window")


# ═══════════════════════════════════════════════════════════════════════
# Tool Schema & Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestToolSchemas(unittest.TestCase):

    def test_all_tools_have_schemas(self):
        session = BrowserSession()
        tools = [
            JavaScriptTool(session),
            GetPageTextTool(session),
            ReadConsoleMessagesTool(session),
            ReadNetworkRequestsTool(session),
            UploadImageTool(session),
            ResizeWindowTool(session),
        ]
        for tool in tools:
            schema = tool.get_schema()
            self.assertTrue(schema.name, f"Missing name for {tool.__class__.__name__}")
            self.assertTrue(schema.description)
            self.assertIsInstance(schema.input_schema, dict)

    def test_tool_names_unique(self):
        session = BrowserSession()
        tools = [
            JavaScriptTool(session),
            GetPageTextTool(session),
            ReadConsoleMessagesTool(session),
            ReadNetworkRequestsTool(session),
            UploadImageTool(session),
            ResizeWindowTool(session),
        ]
        names = [t.name for t in tools]
        self.assertEqual(len(names), len(set(names)))

    def test_expected_tool_names(self):
        expected = {
            "javascript_tool", "get_page_text", "read_console_messages",
            "read_network_requests", "upload_image", "resize_window",
        }
        session = BrowserSession()
        tools = [
            JavaScriptTool(session),
            GetPageTextTool(session),
            ReadConsoleMessagesTool(session),
            ReadNetworkRequestsTool(session),
            UploadImageTool(session),
            ResizeWindowTool(session),
        ]
        actual = {t.name for t in tools}
        self.assertEqual(actual, expected)


# ═══════════════════════════════════════════════════════════════════════
# Main.py Wiring Test
# ═══════════════════════════════════════════════════════════════════════

class TestMainWiring(unittest.TestCase):

    def test_extended_tools_wiring(self):
        import inspect
        import cowork_agent.main as main_mod
        source = inspect.getsource(main_mod)
        self.assertIn("Sprint 34", source)
        self.assertIn("JavaScriptTool", source)
        self.assertIn("GetPageTextTool", source)
        self.assertIn("ReadConsoleMessagesTool", source)
        self.assertIn("ReadNetworkRequestsTool", source)
        self.assertIn("UploadImageTool", source)
        self.assertIn("ResizeWindowTool", source)


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def test_js_arithmetic(self):
        session, tab_id = _make_session_with_tab()
        tool = JavaScriptTool(browser_session=session)
        result = run(tool.execute(
            action="javascript_exec", text="10 * 5", tabId=tab_id,
        ))
        self.assertTrue(result.success)
        self.assertIn("50", result.output)

    def test_js_querySelector(self):
        session, tab_id = _make_session_with_tab()
        tool = JavaScriptTool(browser_session=session)
        result = run(tool.execute(
            action="javascript_exec",
            text="document.querySelector('.main')",
            tabId=tab_id,
        ))
        self.assertTrue(result.success)
        self.assertIn("HTMLElement", result.output)

    def test_console_combined_filters(self):
        """Test onlyErrors + pattern combined."""
        session, tab_id = _make_session_with_tab()
        tab = session.get_tab(tab_id)
        tab.console_messages = [
            {"level": "error", "text": "Network error on /api/users"},
            {"level": "error", "text": "Timeout on /api/posts"},
            {"level": "log", "text": "API call to /api/users successful"},
            {"level": "warn", "text": "Slow response from /api/users"},
        ]
        tool = ReadConsoleMessagesTool(browser_session=session)
        result = run(tool.execute(tabId=tab_id, onlyErrors=True, pattern="users"))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["message_count"], 1)  # Only the error about users

    def test_network_output_includes_method_and_status(self):
        session, tab_id = _make_session_with_tab()
        tab = session.get_tab(tab_id)
        tab.network_requests = [
            {"method": "POST", "url": "https://api.test.com/data", "status": 404, "type": "xhr"},
        ]
        tool = ReadNetworkRequestsTool(browser_session=session)
        result = run(tool.execute(tabId=tab_id))
        self.assertIn("POST", result.output)
        self.assertIn("404", result.output)
        self.assertIn("xhr", result.output)

    def test_page_text_no_text_nodes(self):
        """Tree with only non-text roles."""
        session, tab_id = _make_session_with_tab()
        tree = AccessibilityNode(
            ref_id="ref_0", role="document", name="",
            children=[
                AccessibilityNode(ref_id="ref_1", role="button", name="Click"),
                AccessibilityNode(ref_id="ref_2", role="img", name="Photo"),
            ],
        )
        session.set_accessibility_tree(tab_id, tree)
        tool = GetPageTextTool(browser_session=session)
        result = run(tool.execute(tabId=tab_id))
        self.assertTrue(result.success)
        # Button and img are not text-extracting roles
        self.assertNotIn("Click", result.output)

    def test_resize_verifies_actual_change(self):
        session, tab_id = _make_session_with_tab()
        tool = ResizeWindowTool(browser_session=session)
        run(tool.execute(tabId=tab_id, width=800, height=600))
        tab = session.get_tab(tab_id)
        self.assertEqual(tab.viewport_width, 800)
        self.assertEqual(tab.viewport_height, 600)

    def test_upload_image_metadata(self):
        session, tab_id = _make_session_with_tab()
        tool = UploadImageTool(browser_session=session)
        result = run(tool.execute(
            imageId="screenshot_abc", tabId=tab_id, ref="ref_file_input",
        ))
        self.assertEqual(result.metadata["imageId"], "screenshot_abc")
        self.assertEqual(result.metadata["ref"], "ref_file_input")

    def test_console_clear_then_read(self):
        """Clear messages then read should return 0."""
        session, tab_id = _make_session_with_tab()
        tab = session.get_tab(tab_id)
        tab.console_messages = [{"level": "log", "text": "msg1"}]
        tool = ReadConsoleMessagesTool(browser_session=session)
        # Clear
        run(tool.execute(tabId=tab_id, clear=True))
        # Read again
        result = run(tool.execute(tabId=tab_id))
        self.assertEqual(result.metadata["message_count"], 0)


if __name__ == "__main__":
    unittest.main()
