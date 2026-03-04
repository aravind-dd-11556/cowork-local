"""
Browser Session Manager — Sprint 33: Browser Automation Core.

Manages browser tab groups, navigation state, and accessibility trees.
Mirrors real Cowork's Claude-in-Chrome MCP architecture where:
  - A tab group isolates tabs per conversation
  - Each tab tracks URL, title, scroll position, and page state
  - The session provides tab lifecycle (create, navigate, close)
  - Accessibility tree snapshots are cached per tab
"""

from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PageLoadState(Enum):
    """State of a browser tab's page load."""
    INITIAL = "initial"        # Tab just created, no navigation yet
    LOADING = "loading"        # Navigation in progress
    LOADED = "loaded"          # DOM ready, page interactive
    FAILED = "failed"          # Navigation failed (timeout, network error, etc.)


@dataclass
class AccessibilityNode:
    """
    Single node in the accessibility tree.

    Maps to DOM elements with role, name, and optional interactivity.
    """
    ref_id: str                       # e.g. "ref_1", "ref_2"
    role: str                         # e.g. "button", "link", "textbox", "heading"
    name: str                         # Accessible name / text content
    value: str = ""                   # Current value (for inputs)
    children: List["AccessibilityNode"] = field(default_factory=list)
    interactive: bool = False         # True for buttons, inputs, links, etc.
    visible: bool = True
    bounds: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)

    def to_dict(self) -> dict:
        d = {
            "ref_id": self.ref_id,
            "role": self.role,
            "name": self.name,
        }
        if self.value:
            d["value"] = self.value
        if self.interactive:
            d["interactive"] = True
        if not self.visible:
            d["visible"] = False
        if self.bounds:
            d["bounds"] = list(self.bounds)
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    def flatten(self, include_hidden: bool = True) -> List["AccessibilityNode"]:
        """Flatten tree into list (pre-order traversal)."""
        result = []
        if include_hidden or self.visible:
            result.append(self)
        for child in self.children:
            result.extend(child.flatten(include_hidden=include_hidden))
        return result


INTERACTIVE_ROLES = frozenset({
    "button", "link", "textbox", "textarea", "checkbox", "radio",
    "combobox", "listbox", "menuitem", "option", "searchbox",
    "slider", "spinbutton", "switch", "tab", "menuitemcheckbox",
    "menuitemradio", "treeitem",
})


@dataclass
class TabInfo:
    """State of a single browser tab."""
    tab_id: int
    group_id: str
    url: str = "about:blank"
    title: str = "New Tab"
    state: PageLoadState = PageLoadState.INITIAL
    scroll_x: int = 0
    scroll_y: int = 0
    viewport_width: int = 1280
    viewport_height: int = 900
    created_at: float = field(default_factory=time.time)
    last_navigated_at: float = 0.0
    # Cached accessibility tree (populated on read_page)
    accessibility_tree: Optional[AccessibilityNode] = None
    # Console messages and network requests buffers
    console_messages: List[Dict[str, Any]] = field(default_factory=list)
    network_requests: List[Dict[str, Any]] = field(default_factory=list)
    # Screenshot tracking
    last_screenshot_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "tab_id": self.tab_id,
            "group_id": self.group_id,
            "url": self.url,
            "title": self.title,
            "state": self.state.value,
            "viewport": f"{self.viewport_width}x{self.viewport_height}",
        }


@dataclass
class TabGroup:
    """Group of tabs for a single conversation/session."""
    group_id: str
    tabs: Dict[int, TabInfo] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def tab_ids(self) -> List[int]:
        return sorted(self.tabs.keys())

    @property
    def tab_count(self) -> int:
        return len(self.tabs)


class BrowserSession:
    """
    Manages browser tabs and page state for automation.

    In the real Cowork, this communicates with a Chrome extension via MCP.
    Here we simulate the browser state to enable tool development and testing.
    """

    def __init__(
        self,
        default_viewport: Tuple[int, int] = (1280, 900),
        on_navigate: Optional[Callable] = None,
        on_screenshot: Optional[Callable] = None,
        # Sprint 46: Extended callbacks for Chrome bridge integration
        on_get_tree: Optional[Callable] = None,
        on_find: Optional[Callable] = None,
        on_form_input: Optional[Callable] = None,
        on_perform_action: Optional[Callable] = None,
        on_js_execute: Optional[Callable] = None,
        on_get_text: Optional[Callable] = None,
        on_read_console: Optional[Callable] = None,
        on_read_network: Optional[Callable] = None,
        on_resize: Optional[Callable] = None,
    ):
        self._groups: Dict[str, TabGroup] = {}
        self._next_tab_id = 1
        self._default_viewport = default_viewport
        self._on_navigate = on_navigate      # Callback for real browser integration
        self._on_screenshot = on_screenshot  # Callback for real screenshot capture
        # Sprint 46: Extended callbacks for Chrome extension bridge
        self._on_get_tree = on_get_tree          # Get real accessibility tree
        self._on_find = on_find                  # Find elements in real DOM
        self._on_form_input = on_form_input      # Set form values in real DOM
        self._on_perform_action = on_perform_action  # Real mouse/keyboard actions
        self._on_js_execute = on_js_execute      # Execute JS in real browser
        self._on_get_text = on_get_text          # Get real page text
        self._on_read_console = on_read_console  # Read real console messages
        self._on_read_network = on_read_network  # Read real network requests
        self._on_resize = on_resize              # Resize real browser window
        # Current active group for MCP context
        self._active_group_id: Optional[str] = None

    # ── Tab Group Management ──────────────────────────────────────────

    def get_or_create_group(self, create_if_empty: bool = True) -> Optional[TabGroup]:
        """Get the active tab group, optionally creating one with a blank tab."""
        if self._active_group_id and self._active_group_id in self._groups:
            return self._groups[self._active_group_id]

        if not create_if_empty:
            return None

        # Create new group with one blank tab
        group_id = f"group_{uuid.uuid4().hex[:8]}"
        group = TabGroup(group_id=group_id)
        self._groups[group_id] = group
        self._active_group_id = group_id

        # Auto-create first tab
        self._create_tab_in_group(group)
        logger.info("Created new tab group %s with initial tab", group_id)
        return group

    def get_group(self, group_id: str) -> Optional[TabGroup]:
        return self._groups.get(group_id)

    @property
    def active_group(self) -> Optional[TabGroup]:
        if self._active_group_id:
            return self._groups.get(self._active_group_id)
        return None

    # ── Tab Management ────────────────────────────────────────────────

    def create_tab(self) -> Optional[TabInfo]:
        """Create a new tab in the active group."""
        group = self.get_or_create_group()
        if not group:
            return None
        return self._create_tab_in_group(group)

    def _create_tab_in_group(self, group: TabGroup) -> TabInfo:
        tab_id = self._next_tab_id
        self._next_tab_id += 1
        tab = TabInfo(
            tab_id=tab_id,
            group_id=group.group_id,
            viewport_width=self._default_viewport[0],
            viewport_height=self._default_viewport[1],
        )
        group.tabs[tab_id] = tab
        return tab

    def get_tab(self, tab_id: int) -> Optional[TabInfo]:
        """Get a tab by ID from the active group."""
        group = self.active_group
        if not group:
            return None
        return group.tabs.get(tab_id)

    def validate_tab(self, tab_id: int) -> Tuple[bool, str]:
        """Check if a tab ID is valid and in the active group."""
        group = self.active_group
        if not group:
            return False, "No active tab group. Use tabs_context_mcp first."
        if tab_id not in group.tabs:
            valid_ids = group.tab_ids
            return False, (
                f"Tab {tab_id} not found in the current group. "
                f"Valid tab IDs: {valid_ids}"
            )
        return True, ""

    # ── Navigation ────────────────────────────────────────────────────

    def navigate(self, tab_id: int, url: str) -> Tuple[bool, str]:
        """
        Navigate a tab to a URL. Handles forward/back and URL normalization.

        Returns (success, message).
        """
        tab = self.get_tab(tab_id)
        if not tab:
            return False, f"Tab {tab_id} not found."

        # Handle special navigation commands
        if url.lower() == "back":
            tab.state = PageLoadState.LOADED
            return True, f"Navigated back in tab {tab_id}."
        elif url.lower() == "forward":
            tab.state = PageLoadState.LOADED
            return True, f"Navigated forward in tab {tab_id}."

        # Normalize URL
        normalized = self._normalize_url(url)

        # Update tab state
        tab.url = normalized
        tab.state = PageLoadState.LOADING
        tab.last_navigated_at = time.time()
        tab.scroll_x = 0
        tab.scroll_y = 0
        # Clear cached tree on navigation
        tab.accessibility_tree = None
        tab.console_messages = []
        tab.network_requests = []

        # Simulate page load (real impl would call browser)
        if self._on_navigate:
            try:
                self._on_navigate(tab_id, normalized)
            except Exception as e:
                tab.state = PageLoadState.FAILED
                return False, f"Navigation failed: {e}"

        tab.state = PageLoadState.LOADED
        tab.title = self._derive_title(normalized)
        return True, f"Navigated tab {tab_id} to {normalized}."

    def _normalize_url(self, url: str) -> str:
        """Add https:// if no protocol specified."""
        url = url.strip()
        if not url:
            return "about:blank"
        if url.startswith(("http://", "https://", "about:", "data:", "file:")):
            return url
        # Default to https
        return f"https://{url}"

    def _derive_title(self, url: str) -> str:
        """Derive a page title from URL (placeholder for real DOM title)."""
        if url == "about:blank":
            return "New Tab"
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc or url
        except Exception:
            return url

    # ── Accessibility Tree ────────────────────────────────────────────

    def set_accessibility_tree(self, tab_id: int, tree: AccessibilityNode) -> bool:
        """Set the accessibility tree for a tab (after read_page)."""
        tab = self.get_tab(tab_id)
        if not tab:
            return False
        tab.accessibility_tree = tree
        return True

    def get_accessibility_tree(
        self,
        tab_id: int,
        *,
        filter_interactive: bool = False,
        max_depth: int = 15,
        ref_id: Optional[str] = None,
    ) -> Optional[AccessibilityNode]:
        """Get the cached accessibility tree for a tab."""
        tab = self.get_tab(tab_id)
        if not tab or not tab.accessibility_tree:
            return None

        tree = tab.accessibility_tree

        # If ref_id specified, find that subtree
        if ref_id:
            tree = self._find_node(tree, ref_id)
            if not tree:
                return None

        # Apply depth limit
        if max_depth > 0:
            tree = self._limit_depth(tree, max_depth)

        # Apply interactive filter
        if filter_interactive:
            tree = self._filter_interactive(tree)

        return tree

    def _find_node(self, tree: AccessibilityNode, ref_id: str) -> Optional[AccessibilityNode]:
        """Find a node by ref_id in the tree."""
        if tree.ref_id == ref_id:
            return tree
        for child in tree.children:
            found = self._find_node(child, ref_id)
            if found:
                return found
        return None

    def _limit_depth(self, node: AccessibilityNode, max_depth: int) -> AccessibilityNode:
        """Create a copy of the tree limited to max_depth levels."""
        if max_depth <= 0:
            return AccessibilityNode(
                ref_id=node.ref_id, role=node.role, name=node.name,
                value=node.value, interactive=node.interactive,
                visible=node.visible, bounds=node.bounds,
            )
        return AccessibilityNode(
            ref_id=node.ref_id, role=node.role, name=node.name,
            value=node.value, interactive=node.interactive,
            visible=node.visible, bounds=node.bounds,
            children=[self._limit_depth(c, max_depth - 1) for c in node.children],
        )

    def _filter_interactive(self, node: AccessibilityNode) -> Optional[AccessibilityNode]:
        """Filter tree to only include interactive elements and their ancestors."""
        filtered_children = []
        for child in node.children:
            filtered = self._filter_interactive(child)
            if filtered:
                filtered_children.append(filtered)

        if node.interactive or filtered_children:
            return AccessibilityNode(
                ref_id=node.ref_id, role=node.role, name=node.name,
                value=node.value, interactive=node.interactive,
                visible=node.visible, bounds=node.bounds,
                children=filtered_children,
            )
        return None

    # ── Element Lookup ────────────────────────────────────────────────

    def find_elements(
        self, tab_id: int, query: str, max_results: int = 20,
    ) -> List[AccessibilityNode]:
        """
        Find elements matching a natural language query.

        Searches by role keywords, text content, and accessible name.
        """
        tab = self.get_tab(tab_id)
        if not tab or not tab.accessibility_tree:
            return []

        query_lower = query.lower().strip()
        all_nodes = tab.accessibility_tree.flatten()

        scored: List[Tuple[int, AccessibilityNode]] = []
        for node in all_nodes:
            if not node.visible:
                continue
            score = self._score_element_match(node, query_lower)
            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:max_results]]

    def _score_element_match(self, node: AccessibilityNode, query: str) -> int:
        """Score how well an element matches a search query."""
        score = 0
        name_lower = node.name.lower()
        role_lower = node.role.lower()

        # Exact name match
        if query == name_lower:
            score += 10
        # Name contains query
        elif query in name_lower:
            score += 5
        # Query contains name
        elif name_lower and name_lower in query:
            score += 3

        # Role match (e.g. "button" in query matches button role)
        if role_lower in query:
            score += 4
        # Role keyword aliases
        role_aliases = {
            "button": ["btn", "click", "press", "submit"],
            "textbox": ["input", "text", "field", "search bar", "search"],
            "link": ["anchor", "href", "navigate", "go to"],
            "heading": ["title", "header", "h1", "h2", "h3"],
            "checkbox": ["check", "toggle", "tick"],
            "combobox": ["dropdown", "select", "picker"],
            "img": ["image", "picture", "photo", "icon"],
        }
        for role_name, aliases in role_aliases.items():
            if role_lower == role_name:
                for alias in aliases:
                    if alias in query:
                        score += 3
                        break

        # Value match
        if node.value and query in node.value.lower():
            score += 2

        return score

    # ── Form Input ────────────────────────────────────────────────────

    def set_form_value(
        self, tab_id: int, ref_id: str, value: Any,
    ) -> Tuple[bool, str]:
        """Set a form element's value by ref_id."""
        tab = self.get_tab(tab_id)
        if not tab or not tab.accessibility_tree:
            return False, "No accessibility tree available. Call read_page first."

        node = self._find_node(tab.accessibility_tree, ref_id)
        if not node:
            return False, f"Element with ref_id '{ref_id}' not found."

        if not node.interactive:
            return False, f"Element '{ref_id}' ({node.role}) is not interactive."

        # Validate role accepts values
        input_roles = {
            "textbox", "textarea", "searchbox", "combobox", "listbox",
            "spinbutton", "slider", "checkbox", "radio", "switch", "option",
        }
        if node.role not in input_roles:
            return False, (
                f"Element '{ref_id}' is a {node.role}, not a form input. "
                f"Expected one of: {', '.join(sorted(input_roles))}"
            )

        # Handle boolean for checkboxes
        if node.role in ("checkbox", "radio", "switch"):
            node.value = str(bool(value))
        else:
            node.value = str(value)

        return True, f"Set {node.role} '{node.name}' to '{node.value}'."

    # ── Mouse/Keyboard Actions ────────────────────────────────────────

    def perform_action(
        self,
        tab_id: int,
        action: str,
        *,
        coordinate: Optional[Tuple[int, int]] = None,
        ref: Optional[str] = None,
        text: Optional[str] = None,
        scroll_direction: Optional[str] = None,
        scroll_amount: int = 3,
        modifiers: Optional[str] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
        duration: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Perform a browser action (click, type, scroll, screenshot, etc.).

        Returns (success, description/result).
        """
        tab = self.get_tab(tab_id)
        if not tab:
            return False, f"Tab {tab_id} not found."

        action = action.lower()

        if action == "screenshot":
            screenshot_id = f"screenshot_{uuid.uuid4().hex[:8]}"
            tab.last_screenshot_id = screenshot_id
            if self._on_screenshot:
                self._on_screenshot(tab_id, screenshot_id)
            return True, f"Screenshot captured (ID: {screenshot_id})."

        if action == "wait":
            wait_time = duration or 1.0
            if wait_time > 30:
                return False, "Maximum wait time is 30 seconds."
            return True, f"Waited {wait_time} seconds."

        if action in ("left_click", "right_click", "double_click", "triple_click"):
            target = self._resolve_target(tab, coordinate, ref)
            if not target:
                return False, f"No target specified for {action}. Provide coordinate or ref."
            mod_str = f" with {modifiers}" if modifiers else ""
            return True, f"Performed {action} at {target}{mod_str}."

        if action == "hover":
            target = self._resolve_target(tab, coordinate, ref)
            if not target:
                return False, "No target specified for hover."
            return True, f"Hovered at {target}."

        if action == "type":
            if not text:
                return False, "No text specified for type action."
            return True, f"Typed '{text[:50]}{'...' if len(text) > 50 else ''}'."

        if action == "key":
            if not text:
                return False, "No key specified for key action."
            return True, f"Pressed key(s): {text}."

        if action == "scroll":
            if not scroll_direction:
                return False, "No scroll_direction specified."
            if scroll_direction not in ("up", "down", "left", "right"):
                return False, f"Invalid scroll direction: {scroll_direction}"
            dx, dy = 0, 0
            if scroll_direction == "up":
                dy = -scroll_amount * 100
            elif scroll_direction == "down":
                dy = scroll_amount * 100
            elif scroll_direction == "left":
                dx = -scroll_amount * 100
            elif scroll_direction == "right":
                dx = scroll_amount * 100
            tab.scroll_x = max(0, tab.scroll_x + dx)
            tab.scroll_y = max(0, tab.scroll_y + dy)
            return True, f"Scrolled {scroll_direction} by {scroll_amount} ticks."

        if action == "scroll_to":
            if not ref:
                return False, "scroll_to requires a ref element ID."
            return True, f"Scrolled element {ref} into view."

        if action == "zoom":
            if not region:
                return False, "zoom requires a region (x0, y0, x1, y1)."
            return True, f"Zoomed into region {region}."

        if action == "left_click_drag":
            if not coordinate:
                return False, "left_click_drag requires coordinate (end position)."
            return True, f"Dragged to {coordinate}."

        return False, f"Unknown action: {action}"

    def _resolve_target(
        self, tab: TabInfo,
        coordinate: Optional[Tuple[int, int]],
        ref: Optional[str],
    ) -> Optional[str]:
        """Resolve a click target from coordinate or ref."""
        if ref:
            if tab.accessibility_tree:
                node = self._find_node(tab.accessibility_tree, ref)
                if node:
                    return f"element '{ref}' ({node.role}: {node.name})"
            return f"element '{ref}'"
        if coordinate:
            return f"({coordinate[0]}, {coordinate[1]})"
        return None

    # ── Window Resize ─────────────────────────────────────────────────

    def resize_window(
        self, tab_id: int, width: int, height: int,
    ) -> Tuple[bool, str]:
        """Resize the browser window for a tab."""
        tab = self.get_tab(tab_id)
        if not tab:
            return False, f"Tab {tab_id} not found."
        if width < 200 or height < 200:
            return False, "Minimum window size is 200x200."
        if width > 7680 or height > 4320:
            return False, "Maximum window size is 7680x4320."
        tab.viewport_width = width
        tab.viewport_height = height
        return True, f"Resized tab {tab_id} window to {width}x{height}."

    # ── Context Info ──────────────────────────────────────────────────

    def get_context(self, create_if_empty: bool = False) -> dict:
        """Get context info about the current tab group."""
        group = self.get_or_create_group(create_if_empty=create_if_empty)
        if not group:
            return {
                "has_group": False,
                "message": "No active tab group. Set createIfEmpty=true to create one.",
            }
        return {
            "has_group": True,
            "group_id": group.group_id,
            "tabs": [tab.to_dict() for tab in group.tabs.values()],
            "tab_count": group.tab_count,
        }

    # ── Cleanup ───────────────────────────────────────────────────────

    def close_tab(self, tab_id: int) -> bool:
        group = self.active_group
        if group and tab_id in group.tabs:
            del group.tabs[tab_id]
            return True
        return False

    def close_all(self) -> None:
        self._groups.clear()
        self._active_group_id = None

    def __len__(self) -> int:
        """Total tabs across all groups."""
        return sum(g.tab_count for g in self._groups.values())
