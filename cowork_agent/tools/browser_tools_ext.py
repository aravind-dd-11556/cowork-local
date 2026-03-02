"""
Browser Automation Tools (Extended) — Sprint 34.

Mirrors real Cowork's Claude-in-Chrome MCP tools (extended set):
  1. javascript_tool   — Execute JS in page context
  2. get_page_text     — Extract raw text from page
  3. read_console_messages — Read browser console logs
  4. read_network_requests — Read HTTP network requests
  5. upload_image      — Upload image to file input or drag & drop
  6. resize_window     — Resize browser window

These tools delegate to BrowserSession for state management.
"""

from __future__ import annotations
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any, Optional

from .base import BaseTool

if TYPE_CHECKING:
    from ..core.browser_session import BrowserSession

logger = logging.getLogger(__name__)


# ── Safe math evaluator (no eval()) ──────────────────────────────────

import ast
import operator as _op

_SAFE_OPS = {
    ast.Add: _op.add, ast.Sub: _op.sub,
    ast.Mult: _op.mul, ast.Div: _op.truediv,
    ast.Mod: _op.mod, ast.Pow: _op.pow,
    ast.USub: _op.neg, ast.UAdd: _op.pos,
}


def _safe_math_eval(node):
    """Safely compute an AST numeric expression via tree-walking."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](
            _safe_math_eval(node.left), _safe_math_eval(node.right),
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_math_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


# ═══════════════════════════════════════════════════════════════════════
# Tool 1: javascript_tool
# ═══════════════════════════════════════════════════════════════════════

class JavaScriptTool(BaseTool):
    """
    Execute JavaScript code in the context of the current page.
    """
    name = "javascript_tool"
    description = (
        "Execute JavaScript code in the context of the current page. "
        "The code runs in the page's context and can interact with the DOM, "
        "window object, and page variables. Returns the result of the last "
        "expression or any thrown errors. Use tabs_context_mcp first if "
        "you don't have a valid tab ID."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Must be set to 'javascript_exec'.",
            },
            "tabId": {
                "type": "number",
                "description": "Tab ID to execute the code in.",
            },
            "text": {
                "type": "string",
                "description": (
                    "The JavaScript code to execute. The result of the last "
                    "expression will be returned. Do NOT use 'return' statements."
                ),
            },
        },
        "required": ["action", "text", "tabId"],
    }

    def __init__(
        self,
        browser_session: Optional["BrowserSession"] = None,
        js_executor: Optional[Any] = None,
    ):
        self._session = browser_session
        self._js_executor = js_executor  # Callback for real JS execution

    async def execute(
        self, *, progress_callback=None,
        action="", text="", tabId=None, **kwargs
    ) -> "ToolResult":
        if self._session is None:
            return self._error("No browser session available.")
        if not text:
            return self._error("'text' parameter (JavaScript code) is required.")
        if tabId is None:
            return self._error("'tabId' parameter is required.")

        tab_id = int(tabId)
        valid, msg = self._session.validate_tab(tab_id)
        if not valid:
            return self._error(msg)

        # Safety check: limit code size
        if len(text) > 100_000:
            return self._error("JavaScript code exceeds maximum size (100KB).")

        # If real executor is available, use it
        if self._js_executor:
            try:
                result = self._js_executor(tab_id, text)
                return self._success(
                    str(result) if result is not None else "undefined",
                    tab_id=tab_id,
                )
            except Exception as e:
                return self._error(f"JavaScript execution error: {e}")

        # Simulated execution — parse simple expressions
        result = self._simulate_js(text)
        return self._success(result, tab_id=tab_id)

    def _simulate_js(self, code: str) -> str:
        """Simulate basic JS execution for testing."""
        code = code.strip()
        # Simple expression evaluation
        if code.startswith("document.title"):
            return "Page Title"
        if code.startswith("document.querySelector"):
            return "[HTMLElement]"
        if code.startswith("window.location"):
            return "https://example.com"
        if code.isdigit():
            return code
        # Try basic arithmetic using safe AST evaluator (no eval)
        try:
            if all(c in "0123456789+-*/().% " for c in code):
                result = _safe_math_eval(ast.parse(code, mode="eval").body)
                return str(result)
        except Exception:
            pass
        return f"(simulated result for: {code[:80]})"


# ═══════════════════════════════════════════════════════════════════════
# Tool 2: get_page_text
# ═══════════════════════════════════════════════════════════════════════

class GetPageTextTool(BaseTool):
    """
    Extract raw text content from the page.
    """
    name = "get_page_text"
    description = (
        "Extract raw text content from the page, prioritizing article content. "
        "Ideal for reading articles, blog posts, or other text-heavy pages. "
        "Returns plain text without HTML formatting."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tabId": {
                "type": "number",
                "description": "Tab ID to extract text from.",
            },
        },
        "required": ["tabId"],
    }

    def __init__(
        self,
        browser_session: Optional["BrowserSession"] = None,
        text_extractor: Optional[Any] = None,
    ):
        self._session = browser_session
        self._text_extractor = text_extractor  # Callback for real text extraction

    async def execute(
        self, *, progress_callback=None, tabId=None, **kwargs
    ) -> "ToolResult":
        if self._session is None:
            return self._error("No browser session available.")
        if tabId is None:
            return self._error("'tabId' parameter is required.")

        tab_id = int(tabId)
        valid, msg = self._session.validate_tab(tab_id)
        if not valid:
            return self._error(msg)

        # If real extractor is available, use it
        if self._text_extractor:
            try:
                text = self._text_extractor(tab_id)
                return self._success(text, tab_id=tab_id, char_count=len(text))
            except Exception as e:
                return self._error(f"Text extraction error: {e}")

        # Simulated extraction from accessibility tree
        tab = self._session.get_tab(tab_id)
        if not tab:
            return self._error(f"Tab {tab_id} not found.")

        if tab.accessibility_tree:
            text = self._extract_text_from_tree(tab.accessibility_tree)
            return self._success(
                text or "(no text content found)",
                tab_id=tab_id,
                char_count=len(text),
            )

        if tab.url == "about:blank":
            return self._success(
                "(empty page)", tab_id=tab_id, char_count=0,
            )

        return self._success(
            f"(page text from {tab.url} — use read_page for structured content)",
            tab_id=tab_id,
            char_count=0,
        )

    def _extract_text_from_tree(self, node) -> str:
        """Extract text from accessibility tree nodes."""
        parts = []
        # Skip hidden nodes
        if not node.visible:
            return ""
        if node.name and node.role in (
            "heading", "paragraph", "text", "statictext",
            "listitem", "link", "label", "cell",
        ):
            parts.append(node.name)
        for child in node.children:
            child_text = self._extract_text_from_tree(child)
            if child_text:
                parts.append(child_text)
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Tool 3: read_console_messages
# ═══════════════════════════════════════════════════════════════════════

class ReadConsoleMessagesTool(BaseTool):
    """
    Read browser console messages (console.log, console.error, etc.).
    """
    name = "read_console_messages"
    description = (
        "Read browser console messages (console.log, console.error, "
        "console.warn, etc.) from a specific tab. Useful for debugging "
        "JavaScript errors and viewing application logs. "
        "IMPORTANT: Always provide a pattern to filter messages."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tabId": {
                "type": "number",
                "description": "Tab ID to read console messages from.",
            },
            "pattern": {
                "type": "string",
                "description": (
                    "Regex pattern to filter messages (e.g., 'error|warning')."
                ),
            },
            "onlyErrors": {
                "type": "boolean",
                "description": "If true, only return error/exception messages.",
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of messages to return (default: 100).",
            },
            "clear": {
                "type": "boolean",
                "description": "If true, clear messages after reading.",
            },
        },
        "required": ["tabId"],
    }

    def __init__(self, browser_session: Optional["BrowserSession"] = None):
        self._session = browser_session

    async def execute(
        self, *, progress_callback=None,
        tabId=None, pattern=None, onlyErrors=False, limit=100, clear=False,
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

        tab = self._session.get_tab(tab_id)
        messages = list(tab.console_messages)

        # Filter by errors only
        if onlyErrors:
            messages = [
                m for m in messages
                if m.get("level") in ("error", "exception")
            ]

        # Filter by pattern
        if pattern:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                messages = [
                    m for m in messages
                    if regex.search(m.get("text", ""))
                ]
            except re.error as e:
                return self._error(f"Invalid regex pattern: {e}")

        # Apply limit
        max_limit = min(int(limit), 1000)
        messages = messages[:max_limit]

        # Clear if requested
        if clear:
            tab.console_messages = []

        if not messages:
            return self._success(
                "No console messages found matching the criteria.",
                message_count=0,
            )

        lines = [f"Console messages ({len(messages)}):\n"]
        for m in messages:
            level = m.get("level", "log")
            text = m.get("text", "")
            ts = m.get("timestamp", "")
            lines.append(f"  [{level}] {text}" + (f" ({ts})" if ts else ""))

        return self._success(
            "\n".join(lines),
            message_count=len(messages),
        )


# ═══════════════════════════════════════════════════════════════════════
# Tool 4: read_network_requests
# ═══════════════════════════════════════════════════════════════════════

class ReadNetworkRequestsTool(BaseTool):
    """
    Read HTTP network requests from a specific tab.
    """
    name = "read_network_requests"
    description = (
        "Read HTTP network requests (XHR, Fetch, documents, images, etc.) "
        "from a specific tab. Useful for debugging API calls and monitoring "
        "network activity. Returns all network requests made by the page."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tabId": {
                "type": "number",
                "description": "Tab ID to read network requests from.",
            },
            "urlPattern": {
                "type": "string",
                "description": (
                    "URL pattern to filter requests (e.g., '/api/', 'example.com')."
                ),
            },
            "limit": {
                "type": "number",
                "description": "Maximum requests to return (default: 100).",
            },
            "clear": {
                "type": "boolean",
                "description": "If true, clear requests after reading.",
            },
        },
        "required": ["tabId"],
    }

    def __init__(self, browser_session: Optional["BrowserSession"] = None):
        self._session = browser_session

    async def execute(
        self, *, progress_callback=None,
        tabId=None, urlPattern=None, limit=100, clear=False,
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

        tab = self._session.get_tab(tab_id)
        requests = list(tab.network_requests)

        # Filter by URL pattern
        if urlPattern:
            pattern_lower = urlPattern.lower()
            requests = [
                r for r in requests
                if pattern_lower in r.get("url", "").lower()
            ]

        # Apply limit
        max_limit = min(int(limit), 1000)
        requests = requests[:max_limit]

        # Clear if requested
        if clear:
            tab.network_requests = []

        if not requests:
            return self._success(
                "No network requests found matching the criteria.",
                request_count=0,
            )

        lines = [f"Network requests ({len(requests)}):\n"]
        for r in requests:
            method = r.get("method", "GET")
            url = r.get("url", "")
            status = r.get("status", "")
            rtype = r.get("type", "")
            size = r.get("size", "")
            parts = [f"  {method} {url}"]
            if status:
                parts.append(f"[{status}]")
            if rtype:
                parts.append(f"({rtype})")
            if size:
                parts.append(f"{size}B")
            lines.append(" ".join(parts))

        return self._success(
            "\n".join(lines),
            request_count=len(requests),
        )


# ═══════════════════════════════════════════════════════════════════════
# Tool 5: upload_image
# ═══════════════════════════════════════════════════════════════════════

class UploadImageTool(BaseTool):
    """
    Upload a screenshot or user-uploaded image to a file input or drag & drop.
    """
    name = "upload_image"
    description = (
        "Upload a previously captured screenshot or user-uploaded image "
        "to a file input or drag & drop target. Supports two approaches: "
        "(1) ref - for targeting specific elements, especially hidden file "
        "inputs, (2) coordinate - for drag & drop to visible locations."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "imageId": {
                "type": "string",
                "description": "ID of a previously captured screenshot or uploaded image.",
            },
            "tabId": {
                "type": "number",
                "description": "Tab ID where the target element is located.",
            },
            "ref": {
                "type": "string",
                "description": (
                    "Element reference ID for file inputs. "
                    "Provide either ref or coordinate, not both."
                ),
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "number"},
                "description": (
                    "Viewport coordinates [x, y] for drag & drop. "
                    "Provide either ref or coordinate, not both."
                ),
            },
            "filename": {
                "type": "string",
                "description": 'Optional filename (default: "image.png").',
            },
        },
        "required": ["imageId", "tabId"],
    }

    def __init__(
        self,
        browser_session: Optional["BrowserSession"] = None,
        upload_handler: Optional[Any] = None,
    ):
        self._session = browser_session
        self._upload_handler = upload_handler

    async def execute(
        self, *, progress_callback=None,
        imageId="", tabId=None, ref=None, coordinate=None,
        filename="image.png", **kwargs
    ) -> "ToolResult":
        if self._session is None:
            return self._error("No browser session available.")
        if not imageId:
            return self._error("'imageId' parameter is required.")
        if tabId is None:
            return self._error("'tabId' parameter is required.")

        tab_id = int(tabId)
        valid, msg = self._session.validate_tab(tab_id)
        if not valid:
            return self._error(msg)

        # Validate mutually exclusive targeting
        if ref and coordinate:
            return self._error("Provide either 'ref' or 'coordinate', not both.")
        if not ref and not coordinate:
            return self._error("Provide either 'ref' or 'coordinate' for the upload target.")

        # If real upload handler available, use it
        if self._upload_handler:
            try:
                target = ref or coordinate
                result = self._upload_handler(tab_id, imageId, target, filename)
                return self._success(str(result), tab_id=tab_id, imageId=imageId)
            except Exception as e:
                return self._error(f"Upload error: {e}")

        # Simulated upload
        if ref:
            return self._success(
                f"Uploaded image '{filename}' (ID: {imageId}) to element {ref}.",
                tab_id=tab_id, imageId=imageId, ref=ref,
            )
        else:
            coord = tuple(coordinate)
            return self._success(
                f"Uploaded image '{filename}' (ID: {imageId}) via drag & drop to {coord}.",
                tab_id=tab_id, imageId=imageId, coordinate=list(coord),
            )


# ═══════════════════════════════════════════════════════════════════════
# Tool 6: resize_window
# ═══════════════════════════════════════════════════════════════════════

class ResizeWindowTool(BaseTool):
    """
    Resize the current browser window to specified dimensions.
    """
    name = "resize_window"
    description = (
        "Resize the current browser window to specified dimensions. "
        "Useful for testing responsive designs or setting up specific "
        "screen sizes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tabId": {
                "type": "number",
                "description": "Tab ID to get the window for.",
            },
            "width": {
                "type": "number",
                "description": "Target window width in pixels.",
            },
            "height": {
                "type": "number",
                "description": "Target window height in pixels.",
            },
        },
        "required": ["width", "height", "tabId"],
    }

    def __init__(self, browser_session: Optional["BrowserSession"] = None):
        self._session = browser_session

    async def execute(
        self, *, progress_callback=None,
        tabId=None, width=None, height=None, **kwargs
    ) -> "ToolResult":
        if self._session is None:
            return self._error("No browser session available.")
        if tabId is None:
            return self._error("'tabId' parameter is required.")
        if width is None or height is None:
            return self._error("Both 'width' and 'height' are required.")

        tab_id = int(tabId)
        valid, msg = self._session.validate_tab(tab_id)
        if not valid:
            return self._error(msg)

        success, message = self._session.resize_window(
            tab_id, int(width), int(height),
        )
        if not success:
            return self._error(message)

        return self._success(
            message,
            tab_id=tab_id,
            width=int(width),
            height=int(height),
        )
