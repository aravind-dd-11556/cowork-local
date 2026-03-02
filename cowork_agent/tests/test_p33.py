"""
Sprint 33 Tests — Browser Automation Core.

Tests for:
  - BrowserSession (tab groups, tabs, navigation, accessibility tree,
    element search, form input, actions, resize, context)
  - 7 Browser Tools (tabs_context, tabs_create, navigate, read_page,
    find, form_input, computer)
  - Main.py wiring
  - Edge cases and error handling

~95 tests total.
"""

import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch

from cowork_agent.core.browser_session import (
    AccessibilityNode,
    BrowserSession,
    INTERACTIVE_ROLES,
    PageLoadState,
    TabGroup,
    TabInfo,
)
from cowork_agent.tools.browser_tools import (
    ComputerTool,
    FindElementTool,
    FormInputTool,
    NavigateTool,
    ReadPageTool,
    TabsContextTool,
    TabsCreateTool,
)


def run(coro):
    """Helper to run async coroutines in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_sample_tree() -> AccessibilityNode:
    """Create a sample accessibility tree for testing."""
    return AccessibilityNode(
        ref_id="ref_0", role="document", name="Test Page",
        children=[
            AccessibilityNode(
                ref_id="ref_1", role="heading", name="Welcome",
                visible=True,
            ),
            AccessibilityNode(
                ref_id="ref_2", role="textbox", name="Search",
                interactive=True, visible=True, value="",
            ),
            AccessibilityNode(
                ref_id="ref_3", role="button", name="Submit",
                interactive=True, visible=True,
            ),
            AccessibilityNode(
                ref_id="ref_4", role="link", name="About Us",
                interactive=True, visible=True,
            ),
            AccessibilityNode(
                ref_id="ref_5", role="div", name="Hidden Section",
                visible=False,
                children=[
                    AccessibilityNode(
                        ref_id="ref_6", role="paragraph", name="Hidden text",
                        visible=False,
                    ),
                ],
            ),
            AccessibilityNode(
                ref_id="ref_7", role="checkbox", name="Agree to terms",
                interactive=True, visible=True, value="false",
            ),
            AccessibilityNode(
                ref_id="ref_8", role="combobox", name="Country",
                interactive=True, visible=True, value="US",
            ),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════
# AccessibilityNode Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAccessibilityNode(unittest.TestCase):

    def test_creation(self):
        node = AccessibilityNode(ref_id="ref_1", role="button", name="Click Me")
        self.assertEqual(node.ref_id, "ref_1")
        self.assertEqual(node.role, "button")
        self.assertEqual(node.name, "Click Me")
        self.assertFalse(node.interactive)
        self.assertTrue(node.visible)

    def test_to_dict_minimal(self):
        node = AccessibilityNode(ref_id="ref_1", role="div", name="container")
        d = node.to_dict()
        self.assertEqual(d["ref_id"], "ref_1")
        self.assertEqual(d["role"], "div")
        self.assertNotIn("interactive", d)
        self.assertNotIn("value", d)

    def test_to_dict_full(self):
        node = AccessibilityNode(
            ref_id="ref_2", role="textbox", name="email",
            value="a@b.com", interactive=True, visible=False,
            bounds=(10, 20, 200, 30),
        )
        d = node.to_dict()
        self.assertTrue(d["interactive"])
        self.assertFalse(d["visible"])
        self.assertEqual(d["value"], "a@b.com")
        self.assertEqual(d["bounds"], [10, 20, 200, 30])

    def test_to_dict_with_children(self):
        parent = AccessibilityNode(
            ref_id="ref_0", role="div", name="parent",
            children=[
                AccessibilityNode(ref_id="ref_1", role="span", name="child"),
            ],
        )
        d = parent.to_dict()
        self.assertEqual(len(d["children"]), 1)
        self.assertEqual(d["children"][0]["ref_id"], "ref_1")

    def test_flatten(self):
        tree = _make_sample_tree()
        flat = tree.flatten()
        # Should include all nodes (including hidden)
        self.assertEqual(len(flat), 9)  # root + 7 children + 1 nested child

    def test_flatten_exclude_hidden(self):
        tree = _make_sample_tree()
        flat = tree.flatten(include_hidden=False)
        # Should exclude hidden section and its child
        visible_count = sum(1 for n in tree.flatten() if n.visible)
        self.assertEqual(len(flat), visible_count)


# ═══════════════════════════════════════════════════════════════════════
# BrowserSession — Tab Group Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBrowserSessionTabGroups(unittest.TestCase):

    def test_create_group_with_tab(self):
        session = BrowserSession()
        group = session.get_or_create_group(create_if_empty=True)
        self.assertIsNotNone(group)
        self.assertEqual(group.tab_count, 1)

    def test_get_existing_group(self):
        session = BrowserSession()
        g1 = session.get_or_create_group(create_if_empty=True)
        g2 = session.get_or_create_group(create_if_empty=True)
        self.assertEqual(g1.group_id, g2.group_id)  # Same group

    def test_no_create_returns_none(self):
        session = BrowserSession()
        group = session.get_or_create_group(create_if_empty=False)
        self.assertIsNone(group)

    def test_active_group_property(self):
        session = BrowserSession()
        self.assertIsNone(session.active_group)
        session.get_or_create_group()
        self.assertIsNotNone(session.active_group)

    def test_get_context_no_group(self):
        session = BrowserSession()
        ctx = session.get_context(create_if_empty=False)
        self.assertFalse(ctx["has_group"])

    def test_get_context_with_group(self):
        session = BrowserSession()
        session.get_or_create_group()
        ctx = session.get_context()
        self.assertTrue(ctx["has_group"])
        self.assertEqual(ctx["tab_count"], 1)
        self.assertIn("tabs", ctx)


# ═══════════════════════════════════════════════════════════════════════
# BrowserSession — Tab Management Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBrowserSessionTabs(unittest.TestCase):

    def setUp(self):
        self.session = BrowserSession()
        self.session.get_or_create_group()

    def test_create_tab(self):
        tab = self.session.create_tab()
        self.assertIsNotNone(tab)
        self.assertEqual(tab.url, "about:blank")
        self.assertEqual(tab.state, PageLoadState.INITIAL)

    def test_get_tab(self):
        tab = self.session.create_tab()
        retrieved = self.session.get_tab(tab.tab_id)
        self.assertEqual(retrieved.tab_id, tab.tab_id)

    def test_validate_tab_valid(self):
        tab = self.session.create_tab()
        valid, msg = self.session.validate_tab(tab.tab_id)
        self.assertTrue(valid)

    def test_validate_tab_invalid(self):
        valid, msg = self.session.validate_tab(9999)
        self.assertFalse(valid)
        self.assertIn("not found", msg)

    def test_close_tab(self):
        tab = self.session.create_tab()
        self.assertTrue(self.session.close_tab(tab.tab_id))
        self.assertIsNone(self.session.get_tab(tab.tab_id))

    def test_close_nonexistent_tab(self):
        self.assertFalse(self.session.close_tab(9999))

    def test_close_all(self):
        self.session.create_tab()
        self.session.close_all()
        self.assertIsNone(self.session.active_group)
        self.assertEqual(len(self.session), 0)

    def test_tab_default_viewport(self):
        session = BrowserSession(default_viewport=(1920, 1080))
        session.get_or_create_group()
        tab = session.create_tab()
        self.assertEqual(tab.viewport_width, 1920)
        self.assertEqual(tab.viewport_height, 1080)

    def test_tab_to_dict(self):
        tab = self.session.create_tab()
        d = tab.to_dict()
        self.assertIn("tab_id", d)
        self.assertIn("url", d)
        self.assertIn("viewport", d)

    def test_len(self):
        self.session.create_tab()
        self.session.create_tab()
        # 1 initial + 2 created = 3
        self.assertEqual(len(self.session), 3)


# ═══════════════════════════════════════════════════════════════════════
# BrowserSession — Navigation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBrowserSessionNavigation(unittest.TestCase):

    def setUp(self):
        self.session = BrowserSession()
        group = self.session.get_or_create_group()
        self.tab_id = list(group.tabs.keys())[0]

    def test_navigate_to_url(self):
        ok, msg = self.session.navigate(self.tab_id, "https://example.com")
        self.assertTrue(ok)
        tab = self.session.get_tab(self.tab_id)
        self.assertEqual(tab.url, "https://example.com")
        self.assertEqual(tab.state, PageLoadState.LOADED)

    def test_navigate_adds_https(self):
        ok, _ = self.session.navigate(self.tab_id, "example.com")
        self.assertTrue(ok)
        tab = self.session.get_tab(self.tab_id)
        self.assertEqual(tab.url, "https://example.com")

    def test_navigate_preserves_http(self):
        ok, _ = self.session.navigate(self.tab_id, "http://insecure.com")
        self.assertTrue(ok)
        tab = self.session.get_tab(self.tab_id)
        self.assertEqual(tab.url, "http://insecure.com")

    def test_navigate_back(self):
        ok, msg = self.session.navigate(self.tab_id, "back")
        self.assertTrue(ok)
        self.assertIn("back", msg.lower())

    def test_navigate_forward(self):
        ok, msg = self.session.navigate(self.tab_id, "forward")
        self.assertTrue(ok)
        self.assertIn("forward", msg.lower())

    def test_navigate_clears_tree(self):
        tab = self.session.get_tab(self.tab_id)
        tab.accessibility_tree = _make_sample_tree()
        self.session.navigate(self.tab_id, "https://new-page.com")
        self.assertIsNone(tab.accessibility_tree)

    def test_navigate_resets_scroll(self):
        tab = self.session.get_tab(self.tab_id)
        tab.scroll_x = 100
        tab.scroll_y = 200
        self.session.navigate(self.tab_id, "https://new-page.com")
        self.assertEqual(tab.scroll_x, 0)
        self.assertEqual(tab.scroll_y, 0)

    def test_navigate_invalid_tab(self):
        ok, msg = self.session.navigate(9999, "https://example.com")
        self.assertFalse(ok)

    def test_navigate_empty_url(self):
        ok, _ = self.session.navigate(self.tab_id, "")
        self.assertTrue(ok)
        tab = self.session.get_tab(self.tab_id)
        self.assertEqual(tab.url, "about:blank")

    def test_navigate_callback(self):
        callback = MagicMock()
        session = BrowserSession(on_navigate=callback)
        session.get_or_create_group()
        tab_id = list(session.active_group.tabs.keys())[0]
        session.navigate(tab_id, "https://example.com")
        callback.assert_called_once_with(tab_id, "https://example.com")

    def test_navigate_callback_failure(self):
        callback = MagicMock(side_effect=Exception("Connection refused"))
        session = BrowserSession(on_navigate=callback)
        session.get_or_create_group()
        tab_id = list(session.active_group.tabs.keys())[0]
        ok, msg = session.navigate(tab_id, "https://fail.com")
        self.assertFalse(ok)
        self.assertIn("Connection refused", msg)


# ═══════════════════════════════════════════════════════════════════════
# BrowserSession — Accessibility Tree Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBrowserSessionAccessibility(unittest.TestCase):

    def setUp(self):
        self.session = BrowserSession()
        group = self.session.get_or_create_group()
        self.tab_id = list(group.tabs.keys())[0]
        self.tree = _make_sample_tree()
        self.session.set_accessibility_tree(self.tab_id, self.tree)

    def test_set_and_get_tree(self):
        tree = self.session.get_accessibility_tree(self.tab_id)
        self.assertIsNotNone(tree)
        self.assertEqual(tree.ref_id, "ref_0")

    def test_filter_interactive(self):
        tree = self.session.get_accessibility_tree(
            self.tab_id, filter_interactive=True,
        )
        self.assertIsNotNone(tree)
        flat = tree.flatten()
        for node in flat:
            # Every leaf should be interactive (or an ancestor of interactive)
            if not node.children:
                self.assertTrue(node.interactive)

    def test_depth_limit(self):
        tree = self.session.get_accessibility_tree(self.tab_id, max_depth=1)
        self.assertIsNotNone(tree)
        # Children should exist but their children should not
        for child in tree.children:
            self.assertEqual(len(child.children), 0)

    def test_ref_id_lookup(self):
        tree = self.session.get_accessibility_tree(self.tab_id, ref_id="ref_3")
        self.assertIsNotNone(tree)
        self.assertEqual(tree.role, "button")
        self.assertEqual(tree.name, "Submit")

    def test_ref_id_not_found(self):
        tree = self.session.get_accessibility_tree(self.tab_id, ref_id="ref_999")
        self.assertIsNone(tree)

    def test_no_tree_returns_none(self):
        tab = self.session.create_tab()
        tree = self.session.get_accessibility_tree(tab.tab_id)
        self.assertIsNone(tree)


# ═══════════════════════════════════════════════════════════════════════
# BrowserSession — Element Find Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBrowserSessionFind(unittest.TestCase):

    def setUp(self):
        self.session = BrowserSession()
        group = self.session.get_or_create_group()
        self.tab_id = list(group.tabs.keys())[0]
        self.session.set_accessibility_tree(self.tab_id, _make_sample_tree())

    def test_find_by_name(self):
        results = self.session.find_elements(self.tab_id, "Submit")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].name, "Submit")

    def test_find_by_role(self):
        results = self.session.find_elements(self.tab_id, "button")
        self.assertTrue(len(results) > 0)

    def test_find_by_role_alias(self):
        results = self.session.find_elements(self.tab_id, "search bar")
        self.assertTrue(len(results) > 0)

    def test_find_no_results(self):
        results = self.session.find_elements(self.tab_id, "nonexistent xyz")
        self.assertEqual(len(results), 0)

    def test_find_max_results(self):
        results = self.session.find_elements(self.tab_id, "ref", max_results=2)
        self.assertLessEqual(len(results), 2)

    def test_find_no_tree(self):
        tab = self.session.create_tab()
        results = self.session.find_elements(tab.tab_id, "anything")
        self.assertEqual(len(results), 0)


# ═══════════════════════════════════════════════════════════════════════
# BrowserSession — Form Input Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBrowserSessionFormInput(unittest.TestCase):

    def setUp(self):
        self.session = BrowserSession()
        group = self.session.get_or_create_group()
        self.tab_id = list(group.tabs.keys())[0]
        self.session.set_accessibility_tree(self.tab_id, _make_sample_tree())

    def test_set_textbox_value(self):
        ok, msg = self.session.set_form_value(self.tab_id, "ref_2", "hello")
        self.assertTrue(ok)
        tree = self.session.get_accessibility_tree(self.tab_id)
        node = self.session._find_node(tree, "ref_2")
        self.assertEqual(node.value, "hello")

    def test_set_checkbox_value(self):
        ok, msg = self.session.set_form_value(self.tab_id, "ref_7", True)
        self.assertTrue(ok)
        tree = self.session.get_accessibility_tree(self.tab_id)
        node = self.session._find_node(tree, "ref_7")
        self.assertEqual(node.value, "True")

    def test_set_combobox_value(self):
        ok, msg = self.session.set_form_value(self.tab_id, "ref_8", "UK")
        self.assertTrue(ok)

    def test_set_non_interactive_fails(self):
        ok, msg = self.session.set_form_value(self.tab_id, "ref_1", "value")
        self.assertFalse(ok)
        self.assertIn("not interactive", msg)

    def test_set_non_input_role_fails(self):
        # ref_3 is button — not an input role
        ok, msg = self.session.set_form_value(self.tab_id, "ref_3", "value")
        self.assertFalse(ok)
        self.assertIn("not a form input", msg)

    def test_set_unknown_ref_fails(self):
        ok, msg = self.session.set_form_value(self.tab_id, "ref_999", "value")
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_set_no_tree_fails(self):
        tab = self.session.create_tab()
        ok, msg = self.session.set_form_value(tab.tab_id, "ref_1", "value")
        self.assertFalse(ok)
        self.assertIn("accessibility tree", msg.lower())


# ═══════════════════════════════════════════════════════════════════════
# BrowserSession — Actions Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBrowserSessionActions(unittest.TestCase):

    def setUp(self):
        self.session = BrowserSession()
        group = self.session.get_or_create_group()
        self.tab_id = list(group.tabs.keys())[0]

    def test_screenshot(self):
        ok, msg = self.session.perform_action(self.tab_id, "screenshot")
        self.assertTrue(ok)
        tab = self.session.get_tab(self.tab_id)
        self.assertIsNotNone(tab.last_screenshot_id)

    def test_wait(self):
        ok, msg = self.session.perform_action(self.tab_id, "wait", duration=1.0)
        self.assertTrue(ok)

    def test_wait_too_long(self):
        ok, msg = self.session.perform_action(self.tab_id, "wait", duration=31.0)
        self.assertFalse(ok)

    def test_left_click_with_coordinate(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "left_click", coordinate=(100, 200),
        )
        self.assertTrue(ok)
        self.assertIn("left_click", msg)

    def test_click_with_ref(self):
        self.session.set_accessibility_tree(self.tab_id, _make_sample_tree())
        ok, msg = self.session.perform_action(
            self.tab_id, "left_click", ref="ref_3",
        )
        self.assertTrue(ok)
        self.assertIn("ref_3", msg)

    def test_click_no_target(self):
        ok, msg = self.session.perform_action(self.tab_id, "left_click")
        self.assertFalse(ok)

    def test_type_text(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "type", text="Hello World",
        )
        self.assertTrue(ok)

    def test_type_no_text(self):
        ok, msg = self.session.perform_action(self.tab_id, "type")
        self.assertFalse(ok)

    def test_key_press(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "key", text="Enter",
        )
        self.assertTrue(ok)

    def test_scroll_down(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "scroll",
            scroll_direction="down", scroll_amount=5,
        )
        self.assertTrue(ok)
        tab = self.session.get_tab(self.tab_id)
        self.assertGreater(tab.scroll_y, 0)

    def test_scroll_up(self):
        tab = self.session.get_tab(self.tab_id)
        tab.scroll_y = 500
        ok, msg = self.session.perform_action(
            self.tab_id, "scroll",
            scroll_direction="up", scroll_amount=3,
        )
        self.assertTrue(ok)
        self.assertEqual(tab.scroll_y, 200)

    def test_scroll_no_direction(self):
        ok, msg = self.session.perform_action(self.tab_id, "scroll")
        self.assertFalse(ok)

    def test_scroll_invalid_direction(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "scroll", scroll_direction="diagonal",
        )
        self.assertFalse(ok)

    def test_scroll_to(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "scroll_to", ref="ref_3",
        )
        self.assertTrue(ok)

    def test_scroll_to_no_ref(self):
        ok, msg = self.session.perform_action(self.tab_id, "scroll_to")
        self.assertFalse(ok)

    def test_hover(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "hover", coordinate=(100, 200),
        )
        self.assertTrue(ok)

    def test_zoom(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "zoom", region=(0, 0, 200, 200),
        )
        self.assertTrue(ok)

    def test_zoom_no_region(self):
        ok, msg = self.session.perform_action(self.tab_id, "zoom")
        self.assertFalse(ok)

    def test_drag(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "left_click_drag", coordinate=(300, 400),
        )
        self.assertTrue(ok)

    def test_drag_no_coordinate(self):
        ok, msg = self.session.perform_action(self.tab_id, "left_click_drag")
        self.assertFalse(ok)

    def test_unknown_action(self):
        ok, msg = self.session.perform_action(self.tab_id, "fly")
        self.assertFalse(ok)

    def test_invalid_tab(self):
        ok, msg = self.session.perform_action(9999, "screenshot")
        self.assertFalse(ok)

    def test_right_click(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "right_click", coordinate=(50, 50),
        )
        self.assertTrue(ok)

    def test_double_click(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "double_click", coordinate=(50, 50),
        )
        self.assertTrue(ok)

    def test_triple_click(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "triple_click", coordinate=(50, 50),
        )
        self.assertTrue(ok)

    def test_click_with_modifiers(self):
        ok, msg = self.session.perform_action(
            self.tab_id, "left_click",
            coordinate=(50, 50), modifiers="ctrl+shift",
        )
        self.assertTrue(ok)
        self.assertIn("ctrl+shift", msg)


# ═══════════════════════════════════════════════════════════════════════
# BrowserSession — Resize Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBrowserSessionResize(unittest.TestCase):

    def setUp(self):
        self.session = BrowserSession()
        group = self.session.get_or_create_group()
        self.tab_id = list(group.tabs.keys())[0]

    def test_resize(self):
        ok, msg = self.session.resize_window(self.tab_id, 1920, 1080)
        self.assertTrue(ok)
        tab = self.session.get_tab(self.tab_id)
        self.assertEqual(tab.viewport_width, 1920)
        self.assertEqual(tab.viewport_height, 1080)

    def test_resize_too_small(self):
        ok, msg = self.session.resize_window(self.tab_id, 100, 100)
        self.assertFalse(ok)

    def test_resize_too_large(self):
        ok, msg = self.session.resize_window(self.tab_id, 10000, 10000)
        self.assertFalse(ok)

    def test_resize_invalid_tab(self):
        ok, msg = self.session.resize_window(9999, 1280, 900)
        self.assertFalse(ok)


# ═══════════════════════════════════════════════════════════════════════
# TabsContextTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTabsContextTool(unittest.TestCase):

    def test_no_session(self):
        tool = TabsContextTool()
        result = run(tool.execute())
        self.assertFalse(result.success)

    def test_no_group(self):
        session = BrowserSession()
        tool = TabsContextTool(browser_session=session)
        result = run(tool.execute(createIfEmpty=False))
        self.assertTrue(result.success)
        self.assertIn("No active tab group", result.output)

    def test_create_if_empty(self):
        session = BrowserSession()
        tool = TabsContextTool(browser_session=session)
        result = run(tool.execute(createIfEmpty=True))
        self.assertTrue(result.success)
        self.assertIn("tab_ids", result.metadata)
        self.assertEqual(result.metadata["tab_count"], 1)

    def test_existing_group(self):
        session = BrowserSession()
        session.get_or_create_group()
        tool = TabsContextTool(browser_session=session)
        result = run(tool.execute())
        self.assertTrue(result.success)
        self.assertIn("group_id", result.metadata)

    def test_tool_name(self):
        tool = TabsContextTool()
        self.assertEqual(tool.name, "tabs_context_mcp")


# ═══════════════════════════════════════════════════════════════════════
# TabsCreateTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTabsCreateTool(unittest.TestCase):

    def test_no_session(self):
        tool = TabsCreateTool()
        result = run(tool.execute())
        self.assertFalse(result.success)

    def test_create_tab_no_group(self):
        session = BrowserSession()
        tool = TabsCreateTool(browser_session=session)
        # Should auto-create group
        result = run(tool.execute())
        self.assertTrue(result.success)
        self.assertIn("tab_id", result.metadata)

    def test_create_tab_in_group(self):
        session = BrowserSession()
        session.get_or_create_group()
        tool = TabsCreateTool(browser_session=session)
        result = run(tool.execute())
        self.assertTrue(result.success)
        self.assertEqual(session.active_group.tab_count, 2)

    def test_tool_name(self):
        tool = TabsCreateTool()
        self.assertEqual(tool.name, "tabs_create_mcp")


# ═══════════════════════════════════════════════════════════════════════
# NavigateTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestNavigateTool(unittest.TestCase):

    def setUp(self):
        self.session = BrowserSession()
        group = self.session.get_or_create_group()
        self.tab_id = list(group.tabs.keys())[0]
        self.tool = NavigateTool(browser_session=self.session)

    def test_no_session(self):
        tool = NavigateTool()
        result = run(tool.execute(url="https://example.com", tabId=1))
        self.assertFalse(result.success)

    def test_no_url(self):
        result = run(self.tool.execute(url="", tabId=self.tab_id))
        self.assertFalse(result.success)

    def test_no_tab_id(self):
        result = run(self.tool.execute(url="https://example.com"))
        self.assertFalse(result.success)

    def test_navigate_success(self):
        result = run(self.tool.execute(url="https://example.com", tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["url"], "https://example.com")

    def test_navigate_invalid_tab(self):
        result = run(self.tool.execute(url="https://example.com", tabId=9999))
        self.assertFalse(result.success)

    def test_tool_name(self):
        self.assertEqual(self.tool.name, "navigate")


# ═══════════════════════════════════════════════════════════════════════
# ReadPageTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestReadPageTool(unittest.TestCase):

    def setUp(self):
        self.session = BrowserSession()
        group = self.session.get_or_create_group()
        self.tab_id = list(group.tabs.keys())[0]
        self.session.set_accessibility_tree(self.tab_id, _make_sample_tree())
        self.tool = ReadPageTool(browser_session=self.session)

    def test_no_session(self):
        tool = ReadPageTool()
        result = run(tool.execute(tabId=1))
        self.assertFalse(result.success)

    def test_no_tab_id(self):
        result = run(self.tool.execute())
        self.assertFalse(result.success)

    def test_read_all(self):
        result = run(self.tool.execute(tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertIn("[ref_0]", result.output)
        self.assertIn("Submit", result.output)
        self.assertGreater(result.metadata["element_count"], 0)

    def test_read_interactive_only(self):
        result = run(self.tool.execute(tabId=self.tab_id, filter="interactive"))
        self.assertTrue(result.success)
        self.assertIn("interactive_count", result.metadata)

    def test_read_blank_page(self):
        tab = self.session.create_tab()
        result = run(self.tool.execute(tabId=tab.tab_id))
        self.assertTrue(result.success)
        self.assertIn("blank", result.output.lower())

    def test_tool_name(self):
        self.assertEqual(self.tool.name, "read_page")


# ═══════════════════════════════════════════════════════════════════════
# FindElementTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFindElementTool(unittest.TestCase):

    def setUp(self):
        self.session = BrowserSession()
        group = self.session.get_or_create_group()
        self.tab_id = list(group.tabs.keys())[0]
        self.session.set_accessibility_tree(self.tab_id, _make_sample_tree())
        self.tool = FindElementTool(browser_session=self.session)

    def test_no_session(self):
        tool = FindElementTool()
        result = run(tool.execute(query="button", tabId=1))
        self.assertFalse(result.success)

    def test_no_query(self):
        result = run(self.tool.execute(query="", tabId=self.tab_id))
        self.assertFalse(result.success)

    def test_find_button(self):
        result = run(self.tool.execute(query="Submit button", tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertGreater(result.metadata["match_count"], 0)
        self.assertIn("refs", result.metadata)

    def test_find_no_results(self):
        result = run(self.tool.execute(query="nonexistent xyz", tabId=self.tab_id))
        self.assertTrue(result.success)  # Success with 0 results
        self.assertEqual(result.metadata["match_count"], 0)

    def test_tool_name(self):
        self.assertEqual(self.tool.name, "find")


# ═══════════════════════════════════════════════════════════════════════
# FormInputTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFormInputTool(unittest.TestCase):

    def setUp(self):
        self.session = BrowserSession()
        group = self.session.get_or_create_group()
        self.tab_id = list(group.tabs.keys())[0]
        self.session.set_accessibility_tree(self.tab_id, _make_sample_tree())
        self.tool = FormInputTool(browser_session=self.session)

    def test_no_session(self):
        tool = FormInputTool()
        result = run(tool.execute(ref="ref_2", tabId=1, value="test"))
        self.assertFalse(result.success)

    def test_no_ref(self):
        result = run(self.tool.execute(ref="", tabId=self.tab_id, value="test"))
        self.assertFalse(result.success)

    def test_set_value(self):
        result = run(self.tool.execute(ref="ref_2", tabId=self.tab_id, value="test query"))
        self.assertTrue(result.success)

    def test_set_non_interactive(self):
        result = run(self.tool.execute(ref="ref_1", tabId=self.tab_id, value="test"))
        self.assertFalse(result.success)

    def test_tool_name(self):
        self.assertEqual(self.tool.name, "form_input")


# ═══════════════════════════════════════════════════════════════════════
# ComputerTool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestComputerTool(unittest.TestCase):

    def setUp(self):
        self.session = BrowserSession()
        group = self.session.get_or_create_group()
        self.tab_id = list(group.tabs.keys())[0]
        self.tool = ComputerTool(browser_session=self.session)

    def test_no_session(self):
        tool = ComputerTool()
        result = run(tool.execute(action="screenshot", tabId=1))
        self.assertFalse(result.success)

    def test_no_action(self):
        result = run(self.tool.execute(action="", tabId=self.tab_id))
        self.assertFalse(result.success)

    def test_no_tab_id(self):
        result = run(self.tool.execute(action="screenshot"))
        self.assertFalse(result.success)

    def test_screenshot(self):
        result = run(self.tool.execute(action="screenshot", tabId=self.tab_id))
        self.assertTrue(result.success)
        self.assertIn("Screenshot", result.output)

    def test_click(self):
        result = run(self.tool.execute(
            action="left_click", tabId=self.tab_id, coordinate=[100, 200],
        ))
        self.assertTrue(result.success)

    def test_type(self):
        result = run(self.tool.execute(
            action="type", tabId=self.tab_id, text="Hello",
        ))
        self.assertTrue(result.success)

    def test_key_with_repeat(self):
        result = run(self.tool.execute(
            action="key", tabId=self.tab_id, text="ArrowDown", repeat=5,
        ))
        self.assertTrue(result.success)
        self.assertIn("5 times", result.output)

    def test_scroll(self):
        result = run(self.tool.execute(
            action="scroll", tabId=self.tab_id,
            scroll_direction="down", scroll_amount=3,
        ))
        self.assertTrue(result.success)

    def test_wait(self):
        result = run(self.tool.execute(
            action="wait", tabId=self.tab_id, duration=0.5,
        ))
        self.assertTrue(result.success)

    def test_invalid_tab(self):
        result = run(self.tool.execute(
            action="screenshot", tabId=9999,
        ))
        self.assertFalse(result.success)

    def test_tool_name(self):
        self.assertEqual(self.tool.name, "computer")


# ═══════════════════════════════════════════════════════════════════════
# Tool Schema & Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestToolSchemas(unittest.TestCase):

    def test_all_tools_have_schemas(self):
        session = BrowserSession()
        tools = [
            TabsContextTool(session),
            TabsCreateTool(session),
            NavigateTool(session),
            ReadPageTool(session),
            FindElementTool(session),
            FormInputTool(session),
            ComputerTool(session),
        ]
        for tool in tools:
            schema = tool.get_schema()
            self.assertTrue(schema.name)
            self.assertTrue(schema.description)
            self.assertIsInstance(schema.input_schema, dict)

    def test_tool_names_unique(self):
        session = BrowserSession()
        tools = [
            TabsContextTool(session),
            TabsCreateTool(session),
            NavigateTool(session),
            ReadPageTool(session),
            FindElementTool(session),
            FormInputTool(session),
            ComputerTool(session),
        ]
        names = [t.name for t in tools]
        self.assertEqual(len(names), len(set(names)))

    def test_expected_tool_names(self):
        expected = {
            "tabs_context_mcp", "tabs_create_mcp", "navigate",
            "read_page", "find", "form_input", "computer",
        }
        session = BrowserSession()
        tools = [
            TabsContextTool(session),
            TabsCreateTool(session),
            NavigateTool(session),
            ReadPageTool(session),
            FindElementTool(session),
            FormInputTool(session),
            ComputerTool(session),
        ]
        actual = {t.name for t in tools}
        self.assertEqual(actual, expected)


# ═══════════════════════════════════════════════════════════════════════
# Main.py Wiring Test
# ═══════════════════════════════════════════════════════════════════════

class TestMainWiring(unittest.TestCase):

    def test_browser_tools_wiring_in_main(self):
        """Verify Sprint 33 wiring block exists in main.py."""
        import inspect
        import cowork_agent.main as main_mod
        source = inspect.getsource(main_mod)
        self.assertIn("Sprint 33", source)
        self.assertIn("BrowserSession", source)
        self.assertIn("TabsContextTool", source)
        self.assertIn("ComputerTool", source)
        self.assertIn("browser_automation", source)


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def test_interactive_roles_constant(self):
        self.assertIn("button", INTERACTIVE_ROLES)
        self.assertIn("textbox", INTERACTIVE_ROLES)
        self.assertIn("link", INTERACTIVE_ROLES)
        self.assertNotIn("div", INTERACTIVE_ROLES)

    def test_page_load_state_enum(self):
        self.assertEqual(PageLoadState.INITIAL.value, "initial")
        self.assertEqual(PageLoadState.LOADED.value, "loaded")
        self.assertEqual(PageLoadState.FAILED.value, "failed")

    def test_tab_group_properties(self):
        group = TabGroup(group_id="test-group")
        self.assertEqual(group.tab_ids, [])
        self.assertEqual(group.tab_count, 0)

    def test_navigate_about_blank(self):
        session = BrowserSession()
        group = session.get_or_create_group()
        tab_id = list(group.tabs.keys())[0]
        ok, _ = session.navigate(tab_id, "about:blank")
        self.assertTrue(ok)

    def test_navigate_data_uri(self):
        session = BrowserSession()
        group = session.get_or_create_group()
        tab_id = list(group.tabs.keys())[0]
        ok, _ = session.navigate(tab_id, "data:text/html,<h1>Test</h1>")
        self.assertTrue(ok)

    def test_scroll_doesnt_go_negative(self):
        session = BrowserSession()
        group = session.get_or_create_group()
        tab_id = list(group.tabs.keys())[0]
        session.perform_action(tab_id, "scroll", scroll_direction="up", scroll_amount=10)
        tab = session.get_tab(tab_id)
        self.assertGreaterEqual(tab.scroll_y, 0)
        self.assertGreaterEqual(tab.scroll_x, 0)

    def test_screenshot_callback(self):
        callback = MagicMock()
        session = BrowserSession(on_screenshot=callback)
        group = session.get_or_create_group()
        tab_id = list(group.tabs.keys())[0]
        session.perform_action(tab_id, "screenshot")
        self.assertTrue(callback.called)

    def test_read_page_max_chars_truncation(self):
        """Build a big tree and verify truncation works."""
        # Create a tree with many nodes
        children = []
        for i in range(100):
            children.append(AccessibilityNode(
                ref_id=f"ref_{i}", role="paragraph",
                name=f"This is paragraph number {i} with some extra text to take up space",
            ))
        big_tree = AccessibilityNode(
            ref_id="ref_root", role="document", name="Big Page",
            children=children,
        )
        session = BrowserSession()
        group = session.get_or_create_group()
        tab_id = list(group.tabs.keys())[0]
        session.set_accessibility_tree(tab_id, big_tree)

        tool = ReadPageTool(browser_session=session)
        result = run(tool.execute(tabId=tab_id, max_chars=500))
        self.assertTrue(result.success)
        self.assertLessEqual(len(result.output), 600)  # Allow small overflow from last line

    def test_computer_tool_hover_no_target(self):
        session = BrowserSession()
        group = session.get_or_create_group()
        tab_id = list(group.tabs.keys())[0]
        tool = ComputerTool(browser_session=session)
        result = run(tool.execute(action="hover", tabId=tab_id))
        self.assertFalse(result.success)

    def test_navigate_sets_last_navigated_at(self):
        session = BrowserSession()
        group = session.get_or_create_group()
        tab_id = list(group.tabs.keys())[0]
        before = time.time()
        session.navigate(tab_id, "https://example.com")
        tab = session.get_tab(tab_id)
        self.assertGreaterEqual(tab.last_navigated_at, before)

    def test_multiple_tabs_in_group(self):
        session = BrowserSession()
        session.get_or_create_group()
        t1 = session.create_tab()
        t2 = session.create_tab()
        t3 = session.create_tab()
        # Navigate each to different URLs
        session.navigate(t1.tab_id, "https://a.com")
        session.navigate(t2.tab_id, "https://b.com")
        session.navigate(t3.tab_id, "https://c.com")
        self.assertEqual(session.get_tab(t1.tab_id).url, "https://a.com")
        self.assertEqual(session.get_tab(t2.tab_id).url, "https://b.com")
        self.assertEqual(session.get_tab(t3.tab_id).url, "https://c.com")


if __name__ == "__main__":
    unittest.main()
