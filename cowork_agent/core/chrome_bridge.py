"""
Chrome Bridge — Sprint 46: Chrome Extension Bridge.

Adapter that connects BrowserSession callbacks to a real Chrome browser
via the Chrome extension WebSocket MCP server.

When attached to a BrowserSession, this bridge replaces simulated behavior
with real Chrome operations (navigate, screenshot, click, type, etc.).
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cowork_agent.core.browser_session import BrowserSession

from cowork_agent.core.browser_session import AccessibilityNode
from cowork_agent.core.chrome_ws_client import ChromeWSClient, ChromeWSConfig

logger = logging.getLogger(__name__)


@dataclass
class ChromeBridgeConfig:
    """Configuration for Chrome extension bridge."""
    enabled: bool = False
    ws_url: str = "ws://localhost:9222"
    connect_timeout: float = 10.0
    request_timeout: float = 30.0
    auto_reconnect: bool = True


class ChromeBridge:
    """
    Bridges BrowserSession callbacks to a real Chrome browser via WebSocket.

    Usage:
        bridge = ChromeBridge(ChromeBridgeConfig(enabled=True))
        await bridge.connect()
        bridge.attach_to_session(browser_session)
        # Now all browser tools use real Chrome
    """

    def __init__(self, config: Optional[ChromeBridgeConfig] = None):
        self._config = config or ChromeBridgeConfig()
        self._ws_config = ChromeWSConfig(
            url=self._config.ws_url,
            connect_timeout=self._config.connect_timeout,
            request_timeout=self._config.request_timeout,
            auto_reconnect=self._config.auto_reconnect,
        )
        self._client = ChromeWSClient(self._ws_config)
        self._session: Optional["BrowserSession"] = None

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket connection to Chrome extension is active."""
        return self._client.is_connected

    async def connect(self) -> bool:
        """Connect to Chrome extension WebSocket server."""
        if not self._config.enabled:
            logger.info("Chrome bridge is disabled")
            return False
        return await self._client.connect()

    async def disconnect(self) -> None:
        """Disconnect from Chrome extension."""
        await self._client.disconnect()

    def attach_to_session(self, session: "BrowserSession") -> None:
        """
        Attach bridge callbacks to a BrowserSession.

        Replaces simulation with real Chrome calls for:
        navigate, screenshot, accessibility tree, find elements,
        form input, perform action, JS execution, page text,
        console messages, network requests, resize.
        """
        self._session = session

        # Wire the two existing callbacks
        session._on_navigate = self._handle_navigate
        session._on_screenshot = self._handle_screenshot

        # Wire extended callbacks (added in Sprint 46)
        session._on_get_tree = self._handle_get_tree
        session._on_find = self._handle_find
        session._on_form_input = self._handle_form_input
        session._on_perform_action = self._handle_perform_action
        session._on_js_execute = self._handle_js_execute
        session._on_get_text = self._handle_get_text
        session._on_read_console = self._handle_read_console
        session._on_read_network = self._handle_read_network
        session._on_resize = self._handle_resize

        logger.info("Chrome bridge attached to BrowserSession")

    def detach(self) -> None:
        """Remove bridge callbacks from session (revert to simulation)."""
        if self._session is not None:
            self._session._on_navigate = None
            self._session._on_screenshot = None
            self._session._on_get_tree = None
            self._session._on_find = None
            self._session._on_form_input = None
            self._session._on_perform_action = None
            self._session._on_js_execute = None
            self._session._on_get_text = None
            self._session._on_read_console = None
            self._session._on_read_network = None
            self._session._on_resize = None
            self._session = None
            logger.info("Chrome bridge detached from BrowserSession")

    # ── Callback Handlers ──────────────────────────────────────────

    def _handle_navigate(self, tab_id: int, url: str) -> None:
        """Forward navigation to Chrome extension."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self._client.call("browser/navigate", {"tabId": tab_id, "url": url})
                )
            else:
                loop.run_until_complete(
                    self._client.call("browser/navigate", {"tabId": tab_id, "url": url})
                )
        except Exception as e:
            logger.error("Navigate via bridge failed: %s", e)
            raise

    def _handle_screenshot(self, tab_id: int, screenshot_id: str) -> None:
        """Forward screenshot capture to Chrome extension."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self._client.call("browser/screenshot", {
                        "tabId": tab_id, "screenshotId": screenshot_id
                    })
                )
            else:
                loop.run_until_complete(
                    self._client.call("browser/screenshot", {
                        "tabId": tab_id, "screenshotId": screenshot_id
                    })
                )
        except Exception as e:
            logger.error("Screenshot via bridge failed: %s", e)
            raise

    async def get_accessibility_tree(self, tab_id: int) -> Optional[AccessibilityNode]:
        """Request accessibility tree from Chrome extension."""
        try:
            result = await self._client.call("browser/getAccessibilityTree", {
                "tabId": tab_id,
            })
            tree_data = result.get("tree")
            if tree_data:
                return self._parse_accessibility_node(tree_data)
            return None
        except Exception as e:
            logger.error("Get accessibility tree via bridge failed: %s", e)
            return None

    async def find_elements(self, tab_id: int, query: str) -> list[dict]:
        """Request element search from Chrome extension."""
        try:
            result = await self._client.call("browser/findElements", {
                "tabId": tab_id, "query": query,
            })
            return result.get("elements", [])
        except Exception as e:
            logger.error("Find elements via bridge failed: %s", e)
            return []

    async def set_form_value(self, tab_id: int, ref_id: str, value: Any) -> dict:
        """Forward form input to Chrome extension."""
        try:
            return await self._client.call("browser/formInput", {
                "tabId": tab_id, "ref": ref_id, "value": value,
            })
        except Exception as e:
            logger.error("Form input via bridge failed: %s", e)
            return {"success": False, "error": str(e)}

    async def perform_action(self, tab_id: int, action: str, params: dict) -> dict:
        """Forward mouse/keyboard action to Chrome extension."""
        try:
            return await self._client.call("browser/performAction", {
                "tabId": tab_id, "action": action, **params,
            })
        except Exception as e:
            logger.error("Perform action via bridge failed: %s", e)
            return {"success": False, "error": str(e)}

    async def execute_js(self, tab_id: int, code: str) -> dict:
        """Execute JavaScript in Chrome via extension."""
        try:
            return await self._client.call("browser/executeScript", {
                "tabId": tab_id, "code": code,
            })
        except Exception as e:
            logger.error("JS execution via bridge failed: %s", e)
            return {"success": False, "error": str(e)}

    async def get_page_text(self, tab_id: int) -> str:
        """Get page text content from Chrome extension."""
        try:
            result = await self._client.call("browser/getPageText", {
                "tabId": tab_id,
            })
            return result.get("text", "")
        except Exception as e:
            logger.error("Get page text via bridge failed: %s", e)
            return ""

    async def read_console(self, tab_id: int) -> list[dict]:
        """Read console messages from Chrome extension."""
        try:
            result = await self._client.call("browser/readConsole", {
                "tabId": tab_id,
            })
            return result.get("messages", [])
        except Exception as e:
            logger.error("Read console via bridge failed: %s", e)
            return []

    async def read_network(self, tab_id: int) -> list[dict]:
        """Read network requests from Chrome extension."""
        try:
            result = await self._client.call("browser/readNetwork", {
                "tabId": tab_id,
            })
            return result.get("requests", [])
        except Exception as e:
            logger.error("Read network via bridge failed: %s", e)
            return []

    async def resize_window(self, tab_id: int, width: int, height: int) -> dict:
        """Resize browser window via Chrome extension."""
        try:
            return await self._client.call("browser/resize", {
                "tabId": tab_id, "width": width, "height": height,
            })
        except Exception as e:
            logger.error("Resize via bridge failed: %s", e)
            return {"success": False, "error": str(e)}

    # ── Internal helpers ───────────────────────────────────────────

    def _handle_get_tree(self, tab_id: int) -> Optional[AccessibilityNode]:
        """Sync wrapper for async get_accessibility_tree (used as callback)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't await in sync callback, return None (tool will handle async)
                return None
            return loop.run_until_complete(self.get_accessibility_tree(tab_id))
        except Exception as e:
            logger.error("Get tree callback failed: %s", e)
            return None

    def _handle_find(self, tab_id: int, query: str) -> list[dict]:
        """Sync wrapper for async find_elements."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return []
            return loop.run_until_complete(self.find_elements(tab_id, query))
        except Exception:
            return []

    def _handle_form_input(self, tab_id: int, ref_id: str, value: Any) -> dict:
        """Sync wrapper for async set_form_value."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return {"success": False, "error": "Cannot call in running loop"}
            return loop.run_until_complete(self.set_form_value(tab_id, ref_id, value))
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _handle_perform_action(self, tab_id: int, action: str, **params) -> dict:
        """Sync wrapper for async perform_action."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return {"success": False, "error": "Cannot call in running loop"}
            return loop.run_until_complete(self.perform_action(tab_id, action, params))
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _handle_js_execute(self, tab_id: int, code: str) -> dict:
        """Sync wrapper for async execute_js."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return {"success": False, "error": "Cannot call in running loop"}
            return loop.run_until_complete(self.execute_js(tab_id, code))
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _handle_get_text(self, tab_id: int) -> str:
        """Sync wrapper for async get_page_text."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return ""
            return loop.run_until_complete(self.get_page_text(tab_id))
        except Exception:
            return ""

    def _handle_read_console(self, tab_id: int) -> list[dict]:
        """Sync wrapper for async read_console."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return []
            return loop.run_until_complete(self.read_console(tab_id))
        except Exception:
            return []

    def _handle_read_network(self, tab_id: int) -> list[dict]:
        """Sync wrapper for async read_network."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return []
            return loop.run_until_complete(self.read_network(tab_id))
        except Exception:
            return []

    def _handle_resize(self, tab_id: int, width: int, height: int) -> dict:
        """Sync wrapper for async resize_window."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return {"success": False, "error": "Cannot call in running loop"}
            return loop.run_until_complete(self.resize_window(tab_id, width, height))
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _parse_accessibility_node(data: dict) -> AccessibilityNode:
        """Parse JSON accessibility tree data into AccessibilityNode."""
        children = [
            ChromeBridge._parse_accessibility_node(child)
            for child in data.get("children", [])
        ]
        bounds = None
        if "bounds" in data:
            b = data["bounds"]
            bounds = (b.get("x", 0), b.get("y", 0), b.get("width", 0), b.get("height", 0))

        return AccessibilityNode(
            ref_id=data.get("ref_id", "ref_0"),
            role=data.get("role", "generic"),
            name=data.get("name", ""),
            value=data.get("value", ""),
            children=children,
            interactive=data.get("interactive", False),
            visible=data.get("visible", True),
            bounds=bounds,
        )
