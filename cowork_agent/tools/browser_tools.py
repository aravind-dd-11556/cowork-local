"""
Browser Automation Tools — Sprint 33: Browser Automation Core.

Mirrors real Cowork's Claude-in-Chrome MCP tools (core set):
  1. tabs_context_mcp  — Get tab group context, create if empty
  2. tabs_create_mcp   — Create a new tab in the MCP tab group
  3. navigate          — Navigate to URL or go forward/back
  4. read_page         — Get accessibility tree of the page
  5. find              — Find elements by natural language query
  6. form_input        — Set values in form elements
  7. computer          — Mouse/keyboard actions + screenshots

These tools delegate to BrowserSession for state management.
"""

from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from .base import BaseTool

if TYPE_CHECKING:
    from ..core.browser_session import BrowserSession

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Tool 1: tabs_context_mcp
# ═══════════════════════════════════════════════════════════════════════

class TabsContextTool(BaseTool):
    """
    Get context about the current MCP tab group.

    Returns all tab IDs and their info. MUST be called before using
    other browser tools so you know what tabs exist.
    """
    name = "tabs_context_mcp"
    description = (
        "Get context information about the current MCP tab group. "
        "Returns all tab IDs inside the group if it exists. "
        "CRITICAL: You must get the context at least once before using "
        "other browser automation tools so you know what tabs exist. "
        "Each new conversation should create its own new tab rather "
        "than reusing existing tabs."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "createIfEmpty": {
                "type": "boolean",
                "description": (
                    "Creates a new MCP tab group if none exists, with an empty tab. "
                    "If a tab group already exists, this parameter has no effect."
                ),
            },
        },
        "required": [],
    }

    def __init__(self, browser_session: Optional["BrowserSession"] = None):
        self._session = browser_session

    async def execute(
        self, *, progress_callback=None, createIfEmpty=False, **kwargs
    ) -> "ToolResult":
        if self._session is None:
            return self._error("No browser session available.")

        context = self._session.get_context(create_if_empty=bool(createIfEmpty))

        if not context.get("has_group"):
            return self._success(
                "No active tab group. Set createIfEmpty=true to create one.",
                has_group=False,
            )

        tabs_info = context["tabs"]
        lines = [f"Tab group: {context['group_id']} ({context['tab_count']} tab(s))\n"]
        for tab in tabs_info:
            lines.append(
                f"  Tab {tab['tab_id']}: {tab['title']} ({tab['url']}) "
                f"[{tab['state']}] {tab['viewport']}"
            )

        return self._success(
            "\n".join(lines),
            group_id=context["group_id"],
            tab_ids=[t["tab_id"] for t in tabs_info],
            tab_count=context["tab_count"],
        )


# ═══════════════════════════════════════════════════════════════════════
# Tool 2: tabs_create_mcp
# ═══════════════════════════════════════════════════════════════════════

class TabsCreateTool(BaseTool):
    """
    Create a new empty tab in the MCP tab group.

    CRITICAL: Must get context via tabs_context_mcp at least once first.
    """
    name = "tabs_create_mcp"
    description = (
        "Creates a new empty tab in the MCP tab group. "
        "CRITICAL: You must get the context using tabs_context_mcp "
        "at least once before using other browser automation tools."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, browser_session: Optional["BrowserSession"] = None):
        self._session = browser_session

    async def execute(self, *, progress_callback=None, **kwargs) -> "ToolResult":
        if self._session is None:
            return self._error("No browser session available.")

        tab = self._session.create_tab()
        if not tab:
            return self._error(
                "Failed to create tab. Ensure a tab group exists "
                "(use tabs_context_mcp with createIfEmpty=true first)."
            )

        return self._success(
            f"Created new tab (ID: {tab.tab_id}) in group {tab.group_id}.",
            tab_id=tab.tab_id,
            group_id=tab.group_id,
        )


# ═══════════════════════════════════════════════════════════════════════
# Tool 3: navigate
# ═══════════════════════════════════════════════════════════════════════

class NavigateTool(BaseTool):
    """
    Navigate to a URL, or go forward/back in browser history.
    """
    name = "navigate"
    description = (
        "Navigate to a URL, or go forward/back in browser history. "
        "If you don't have a valid tab ID, use tabs_context_mcp first. "
        "URLs default to https:// if no protocol is specified. "
        'Use "forward" or "back" to navigate history.'
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    'The URL to navigate to. Use "forward" to go forward '
                    'or "back" to go back in history.'
                ),
            },
            "tabId": {
                "type": "number",
                "description": "Tab ID to navigate. Must be in the current group.",
            },
        },
        "required": ["url", "tabId"],
    }

    def __init__(self, browser_session: Optional["BrowserSession"] = None):
        self._session = browser_session

    async def execute(
        self, *, progress_callback=None, url="", tabId=None, **kwargs
    ) -> "ToolResult":
        if self._session is None:
            return self._error("No browser session available.")
        if not url:
            return self._error("'url' parameter is required.")
        if tabId is None:
            return self._error("'tabId' parameter is required.")

        tab_id = int(tabId)
        valid, msg = self._session.validate_tab(tab_id)
        if not valid:
            return self._error(msg)

        success, message = self._session.navigate(tab_id, url)
        if not success:
            return self._error(message)

        tab = self._session.get_tab(tab_id)
        return self._success(
            message,
            tab_id=tab_id,
            url=tab.url if tab else url,
            title=tab.title if tab else "",
        )


# ═══════════════════════════════════════════════════════════════════════
# Tool 4: read_page
# ═══════════════════════════════════════════════════════════════════════

class ReadPageTool(BaseTool):
    """
    Get an accessibility tree representation of elements on the page.
    """
    name = "read_page"
    description = (
        "Get an accessibility tree representation of elements on the page. "
        "By default returns all elements including non-visible ones. "
        "Optionally filter for only interactive elements. "
        "If output is too large, specify a smaller depth or focus on "
        "a specific element using ref_id."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tabId": {
                "type": "number",
                "description": "Tab ID to read from. Must be in the current group.",
            },
            "filter": {
                "type": "string",
                "enum": ["interactive", "all"],
                "description": (
                    'Filter: "interactive" for buttons/links/inputs only, '
                    '"all" for all elements (default: all).'
                ),
            },
            "depth": {
                "type": "number",
                "description": "Maximum depth of tree to traverse (default: 15).",
            },
            "ref_id": {
                "type": "string",
                "description": (
                    "Reference ID of a parent element to read. "
                    "Returns that element and all its children."
                ),
            },
            "max_chars": {
                "type": "number",
                "description": "Maximum characters for output (default: 50000).",
            },
        },
        "required": ["tabId"],
    }

    def __init__(self, browser_session: Optional["BrowserSession"] = None):
        self._session = browser_session

    async def execute(
        self, *, progress_callback=None,
        tabId=None, filter="all", depth=15, ref_id=None, max_chars=50000,
        **kwargs
    ) -> "ToolResult":
        if self._session is None:
            return self._error("No browser session available.")
        if tabId is None:
            return self._error("'tabId' parameter is required.")

        tab_id = int(tabId)
        valid, msg = self._session.validate_tab(tab_id)
        if not valid:
            return self._error(msg)

        filter_interactive = (filter == "interactive")
        max_depth = int(depth) if depth else 15

        tree = self._session.get_accessibility_tree(
            tab_id,
            filter_interactive=filter_interactive,
            max_depth=max_depth,
            ref_id=ref_id,
        )

        if not tree:
            tab = self._session.get_tab(tab_id)
            if tab and tab.url == "about:blank":
                return self._success(
                    "Page is blank (about:blank). Navigate to a URL first.",
                    element_count=0,
                )
            return self._success(
                "No accessibility tree available for this page. "
                "The page may not have loaded yet or has no accessible elements.",
                element_count=0,
            )

        # Serialize tree to text
        output = self._render_tree(tree, max_chars=int(max_chars))
        all_nodes = tree.flatten()
        interactive_count = sum(1 for n in all_nodes if n.interactive)

        return self._success(
            output,
            element_count=len(all_nodes),
            interactive_count=interactive_count,
        )

    def _render_tree(
        self, node, indent: int = 0, max_chars: int = 50000,
    ) -> str:
        """Render accessibility tree as indented text."""
        lines = []
        self._render_node(node, lines, indent, max_chars)
        output = "\n".join(lines)
        if len(output) > max_chars:
            output = output[:max_chars] + "\n... (output truncated)"
        return output

    def _render_node(
        self, node, lines: list, indent: int, max_chars: int,
    ) -> None:
        prefix = "  " * indent
        # Build line: [ref_id] role "name" (value)
        parts = [f"{prefix}[{node.ref_id}]", node.role]
        if node.name:
            parts.append(f'"{node.name}"')
        if node.value:
            parts.append(f"value={node.value}")
        if node.interactive:
            parts.append("(interactive)")
        if not node.visible:
            parts.append("(hidden)")

        lines.append(" ".join(parts))

        # Check approximate size
        total_len = sum(len(l) for l in lines)
        if total_len > max_chars:
            return

        for child in node.children:
            self._render_node(child, lines, indent + 1, max_chars)


# ═══════════════════════════════════════════════════════════════════════
# Tool 5: find
# ═══════════════════════════════════════════════════════════════════════

class FindElementTool(BaseTool):
    """
    Find elements on the page using natural language.
    """
    name = "find"
    description = (
        "Find elements on the page using natural language. "
        'Can search by purpose (e.g., "search bar", "login button") '
        'or by text content (e.g., "organic mango product"). '
        "Returns up to 20 matching elements with references."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language description of what to find "
                    '(e.g., "search bar", "add to cart button").'
                ),
            },
            "tabId": {
                "type": "number",
                "description": "Tab ID to search in. Must be in the current group.",
            },
        },
        "required": ["query", "tabId"],
    }

    def __init__(self, browser_session: Optional["BrowserSession"] = None):
        self._session = browser_session

    async def execute(
        self, *, progress_callback=None, query="", tabId=None, **kwargs
    ) -> "ToolResult":
        if self._session is None:
            return self._error("No browser session available.")
        if not query:
            return self._error("'query' parameter is required.")
        if tabId is None:
            return self._error("'tabId' parameter is required.")

        tab_id = int(tabId)
        valid, msg = self._session.validate_tab(tab_id)
        if not valid:
            return self._error(msg)

        matches = self._session.find_elements(tab_id, query)

        if not matches:
            return self._success(
                f"No elements found matching: '{query}'. "
                "Try a different query or use read_page to see all elements.",
                match_count=0,
            )

        lines = [f"Found {len(matches)} element(s) matching '{query}':\n"]
        for node in matches:
            parts = [f"  [{node.ref_id}]", node.role]
            if node.name:
                parts.append(f'"{node.name}"')
            if node.value:
                parts.append(f"value={node.value}")
            if node.interactive:
                parts.append("(interactive)")
            lines.append(" ".join(parts))

        if len(matches) >= 20:
            lines.append(
                "\n(More than 20 matches. Use a more specific query.)"
            )

        return self._success(
            "\n".join(lines),
            match_count=len(matches),
            refs=[n.ref_id for n in matches],
        )


# ═══════════════════════════════════════════════════════════════════════
# Tool 6: form_input
# ═══════════════════════════════════════════════════════════════════════

class FormInputTool(BaseTool):
    """
    Set values in form elements using element reference ID.
    """
    name = "form_input"
    description = (
        "Set values in form elements using element reference ID from "
        "the read_page tool. For checkboxes use boolean, for selects "
        "use option value or text, for other inputs use string/number."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": 'Element reference ID (e.g., "ref_1", "ref_2").',
            },
            "tabId": {
                "type": "number",
                "description": "Tab ID to set form value in.",
            },
            "value": {
                "description": (
                    "The value to set. For checkboxes use boolean, "
                    "for selects use option value, for other inputs use string."
                ),
            },
        },
        "required": ["ref", "tabId"],
    }

    def __init__(self, browser_session: Optional["BrowserSession"] = None):
        self._session = browser_session

    async def execute(
        self, *, progress_callback=None, ref="", tabId=None, value=None, **kwargs
    ) -> "ToolResult":
        if self._session is None:
            return self._error("No browser session available.")
        if not ref:
            return self._error("'ref' parameter is required.")
        if tabId is None:
            return self._error("'tabId' parameter is required.")

        tab_id = int(tabId)
        valid, msg = self._session.validate_tab(tab_id)
        if not valid:
            return self._error(msg)

        success, message = self._session.set_form_value(tab_id, ref, value)
        if not success:
            return self._error(message)

        return self._success(message, ref=ref, value=str(value))


# ═══════════════════════════════════════════════════════════════════════
# Tool 7: computer
# ═══════════════════════════════════════════════════════════════════════

class ComputerTool(BaseTool):
    """
    Use mouse and keyboard to interact with a web browser, and take screenshots.
    """
    name = "computer"
    description = (
        "Use a mouse and keyboard to interact with a web browser, and "
        "take screenshots. Actions: left_click, right_click, double_click, "
        "triple_click, type, screenshot, wait, scroll, key, left_click_drag, "
        "zoom, scroll_to, hover."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "left_click", "right_click", "double_click", "triple_click",
                    "type", "screenshot", "wait", "scroll", "key",
                    "left_click_drag", "zoom", "scroll_to", "hover",
                ],
                "description": "The action to perform.",
            },
            "tabId": {
                "type": "number",
                "description": "Tab ID to execute the action on.",
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "number"},
                "description": "(x, y) coordinates for click/scroll actions.",
            },
            "ref": {
                "type": "string",
                "description": (
                    "Element reference ID. Required for scroll_to, "
                    "alternative to coordinate for clicks."
                ),
            },
            "text": {
                "type": "string",
                "description": "Text to type or key(s) to press.",
            },
            "scroll_direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "Scroll direction. Required for scroll.",
            },
            "scroll_amount": {
                "type": "number",
                "description": "Number of scroll ticks (default: 3, max: 10).",
            },
            "modifiers": {
                "type": "string",
                "description": 'Modifier keys (e.g., "ctrl", "shift", "ctrl+shift").',
            },
            "region": {
                "type": "array",
                "items": {"type": "number"},
                "description": "(x0, y0, x1, y1) region for zoom action.",
            },
            "duration": {
                "type": "number",
                "description": "Wait duration in seconds (max 30). For wait action.",
            },
            "start_coordinate": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Start coordinates for left_click_drag.",
            },
            "repeat": {
                "type": "number",
                "description": "Repeat count for key action (1-100, default: 1).",
            },
        },
        "required": ["action", "tabId"],
    }

    def __init__(self, browser_session: Optional["BrowserSession"] = None):
        self._session = browser_session

    async def execute(
        self, *, progress_callback=None,
        action="", tabId=None,
        coordinate=None, ref=None, text=None,
        scroll_direction=None, scroll_amount=3,
        modifiers=None, region=None, duration=None,
        start_coordinate=None, repeat=1,
        **kwargs
    ) -> "ToolResult":
        if self._session is None:
            return self._error("No browser session available.")
        if not action:
            return self._error("'action' parameter is required.")
        if tabId is None:
            return self._error("'tabId' parameter is required.")

        tab_id = int(tabId)
        valid, msg = self._session.validate_tab(tab_id)
        if not valid:
            return self._error(msg)

        # Convert coordinate list to tuple
        coord_tuple = tuple(coordinate) if coordinate else None
        region_tuple = tuple(region) if region else None

        # Handle repeat for key action
        if action == "key" and text and repeat and repeat > 1:
            repeat = min(int(repeat), 100)
            results = []
            for _ in range(repeat):
                ok, msg = self._session.perform_action(
                    tab_id, action, text=text, modifiers=modifiers,
                )
                if not ok:
                    return self._error(msg)
                results.append(msg)
            return self._success(
                f"Pressed key(s) '{text}' {repeat} times.",
                action=action, repeat=repeat,
            )

        success, message = self._session.perform_action(
            tab_id, action,
            coordinate=coord_tuple,
            ref=ref,
            text=text,
            scroll_direction=scroll_direction,
            scroll_amount=int(scroll_amount) if scroll_amount else 3,
            modifiers=modifiers,
            region=region_tuple,
            duration=float(duration) if duration else None,
        )

        if not success:
            return self._error(message)

        return self._success(message, action=action, tab_id=tab_id)
