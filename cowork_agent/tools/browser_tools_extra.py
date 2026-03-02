"""
Browser Automation Tools (Extras) — Sprint 35.

Mirrors real Cowork's Claude-in-Chrome MCP tools (extras):
  1. gif_creator       — Record and export GIF of browser actions
  2. shortcuts_list    — List available shortcuts/workflows
  3. shortcuts_execute — Execute a shortcut/workflow
  4. switch_browser    — Switch to a different Chrome browser

These tools delegate to BrowserSession for state management.
"""

from __future__ import annotations
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

from .base import BaseTool

if TYPE_CHECKING:
    from ..core.browser_session import BrowserSession

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Tool 1: gif_creator
# ═══════════════════════════════════════════════════════════════════════

class GifCreatorTool(BaseTool):
    """
    Manage GIF recording and export for browser automation sessions.

    Supports start_recording, stop_recording, export, and clear actions.
    """
    name = "gif_creator"
    description = (
        "Manage GIF recording and export for browser automation sessions. "
        "Control when to start/stop recording browser actions (clicks, scrolls, "
        "navigation), then export as an animated GIF with visual overlays."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start_recording", "stop_recording", "export", "clear"],
                "description": (
                    "Action to perform: 'start_recording', 'stop_recording', "
                    "'export', or 'clear'."
                ),
            },
            "tabId": {
                "type": "number",
                "description": "Tab ID to scope the recording to.",
            },
            "download": {
                "type": "boolean",
                "description": "Set true for 'export' to download the GIF.",
            },
            "filename": {
                "type": "string",
                "description": "Optional filename for exported GIF.",
            },
            "options": {
                "type": "object",
                "description": (
                    "GIF enhancement options for export: "
                    "showClickIndicators, showDragPaths, showActionLabels, "
                    "showProgressBar, showWatermark, quality."
                ),
            },
        },
        "required": ["action", "tabId"],
    }

    _VALID_ACTIONS = frozenset({
        "start_recording", "stop_recording", "export", "clear",
    })

    def __init__(
        self,
        browser_session: Optional["BrowserSession"] = None,
        gif_handler: Optional[Any] = None,
    ):
        self._session = browser_session
        self._gif_handler = gif_handler  # Real GIF recording backend
        # Internal recording state per tab group
        self._recordings: dict[int, dict] = {}

    async def execute(
        self, *, progress_callback=None, action=None, tabId=None,
        download=None, filename=None, options=None, **kwargs,
    ) -> "ToolResult":
        """Execute a GIF recording action."""
        if self._session is None:
            return self._error("No browser session available.")
        if action is None:
            return self._error("'action' parameter is required.")
        if action not in self._VALID_ACTIONS:
            return self._error(
                f"Invalid action '{action}'. Must be one of: "
                f"{', '.join(sorted(self._VALID_ACTIONS))}."
            )
        if tabId is None:
            return self._error("'tabId' parameter is required.")

        tab_id = int(tabId)
        valid, msg = self._session.validate_tab(tab_id)
        if not valid:
            return self._error(msg)

        # Delegate to real handler if available
        if self._gif_handler:
            try:
                result = self._gif_handler(
                    action=action, tab_id=tab_id,
                    download=download, filename=filename, options=options,
                )
                return self._success(
                    str(result), tab_id=tab_id, action=action,
                )
            except Exception as e:
                return self._error(f"GIF handler error: {e}")

        # Simulated recording state
        if action == "start_recording":
            self._recordings[tab_id] = {
                "status": "recording",
                "frames": [],
                "started_at": time.time(),
            }
            return self._success(
                "Recording started.",
                tab_id=tab_id, action=action, status="recording",
            )

        if action == "stop_recording":
            rec = self._recordings.get(tab_id)
            if not rec or rec["status"] != "recording":
                return self._error("No active recording for this tab.")
            rec["status"] = "stopped"
            rec["stopped_at"] = time.time()
            frame_count = len(rec.get("frames", []))
            return self._success(
                f"Recording stopped. {frame_count} frames captured.",
                tab_id=tab_id, action=action,
                status="stopped", frame_count=frame_count,
            )

        if action == "export":
            rec = self._recordings.get(tab_id)
            if not rec:
                return self._error("No recording data for this tab.")
            if rec["status"] == "recording":
                return self._error("Stop recording before exporting.")

            fname = filename or f"recording-{int(time.time())}.gif"
            export_options = options or {}

            return self._success(
                f"GIF exported as '{fname}'.",
                tab_id=tab_id, action=action,
                filename=fname,
                download=bool(download),
                options_applied=export_options,
            )

        if action == "clear":
            if tab_id in self._recordings:
                del self._recordings[tab_id]
            return self._success(
                "Recording data cleared.",
                tab_id=tab_id, action=action,
            )

        return self._error(f"Unhandled action: {action}")


# ═══════════════════════════════════════════════════════════════════════
# Tool 2: shortcuts_list
# ═══════════════════════════════════════════════════════════════════════

class ShortcutsListTool(BaseTool):
    """
    List all available shortcuts and workflows.
    """
    name = "shortcuts_list"
    description = (
        "List all available shortcuts and workflows. Returns shortcuts "
        "with their commands, descriptions, and whether they are workflows."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tabId": {
                "type": "number",
                "description": "Tab ID to list shortcuts from.",
            },
        },
        "required": ["tabId"],
    }

    def __init__(
        self,
        browser_session: Optional["BrowserSession"] = None,
        shortcuts_provider: Optional[Any] = None,
    ):
        self._session = browser_session
        self._shortcuts_provider = shortcuts_provider

    async def execute(
        self, *, progress_callback=None, tabId=None, **kwargs,
    ) -> "ToolResult":
        """List available shortcuts."""
        if self._session is None:
            return self._error("No browser session available.")
        if tabId is None:
            return self._error("'tabId' parameter is required.")

        tab_id = int(tabId)
        valid, msg = self._session.validate_tab(tab_id)
        if not valid:
            return self._error(msg)

        # Real provider
        if self._shortcuts_provider:
            try:
                shortcuts = self._shortcuts_provider(tab_id)
                return self._success(
                    str(shortcuts), tab_id=tab_id,
                    shortcut_count=len(shortcuts) if isinstance(shortcuts, list) else 0,
                )
            except Exception as e:
                return self._error(f"Shortcuts provider error: {e}")

        # Simulated shortcuts
        shortcuts = [
            {
                "shortcutId": "sc_debug",
                "command": "debug",
                "description": "Debug the current page",
                "isWorkflow": False,
            },
            {
                "shortcutId": "sc_summarize",
                "command": "summarize",
                "description": "Summarize page content",
                "isWorkflow": False,
            },
            {
                "shortcutId": "wf_test",
                "command": "run-tests",
                "description": "Run test suite workflow",
                "isWorkflow": True,
            },
        ]

        lines = ["Available shortcuts/workflows:\n"]
        for sc in shortcuts:
            kind = "workflow" if sc["isWorkflow"] else "shortcut"
            lines.append(
                f"  /{sc['command']} ({kind}) — {sc['description']} "
                f"[id: {sc['shortcutId']}]"
            )

        return self._success(
            "\n".join(lines),
            tab_id=tab_id,
            shortcut_count=len(shortcuts),
            shortcuts=shortcuts,
        )


# ═══════════════════════════════════════════════════════════════════════
# Tool 3: shortcuts_execute
# ═══════════════════════════════════════════════════════════════════════

class ShortcutsExecuteTool(BaseTool):
    """
    Execute a shortcut or workflow by command name or ID.
    """
    name = "shortcuts_execute"
    description = (
        "Execute a shortcut or workflow by running it in a new sidepanel "
        "window using the current tab. Use shortcuts_list first to see "
        "available shortcuts."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tabId": {
                "type": "number",
                "description": "Tab ID to execute the shortcut on.",
            },
            "command": {
                "type": "string",
                "description": "The command name of the shortcut (e.g., 'debug').",
            },
            "shortcutId": {
                "type": "string",
                "description": "The ID of the shortcut to execute.",
            },
        },
        "required": ["tabId"],
    }

    def __init__(
        self,
        browser_session: Optional["BrowserSession"] = None,
        shortcut_executor: Optional[Any] = None,
    ):
        self._session = browser_session
        self._shortcut_executor = shortcut_executor

    async def execute(
        self, *, progress_callback=None, tabId=None,
        command=None, shortcutId=None, **kwargs,
    ) -> "ToolResult":
        """Execute a shortcut or workflow."""
        if self._session is None:
            return self._error("No browser session available.")
        if tabId is None:
            return self._error("'tabId' parameter is required.")
        if command is None and shortcutId is None:
            return self._error(
                "Either 'command' or 'shortcutId' must be provided."
            )

        tab_id = int(tabId)
        valid, msg = self._session.validate_tab(tab_id)
        if not valid:
            return self._error(msg)

        # Real executor
        if self._shortcut_executor:
            try:
                result = self._shortcut_executor(
                    tab_id=tab_id, command=command, shortcut_id=shortcutId,
                )
                return self._success(
                    str(result), tab_id=tab_id,
                    command=command, shortcut_id=shortcutId,
                )
            except Exception as e:
                return self._error(f"Shortcut execution error: {e}")

        # Simulated execution
        identifier = command or shortcutId
        execution_id = str(uuid.uuid4())[:8]

        return self._success(
            f"Shortcut '{identifier}' started (execution: {execution_id}). "
            f"Running in sidepanel — does not wait for completion.",
            tab_id=tab_id,
            command=command,
            shortcut_id=shortcutId,
            execution_id=execution_id,
            status="started",
        )


# ═══════════════════════════════════════════════════════════════════════
# Tool 4: switch_browser
# ═══════════════════════════════════════════════════════════════════════

class SwitchBrowserTool(BaseTool):
    """
    Switch which Chrome browser is used for browser automation.
    """
    name = "switch_browser"
    description = (
        "Switch which Chrome browser is used for browser automation. "
        "Broadcasts a connection request to all Chrome browsers with the "
        "extension installed — the user clicks 'Connect' in the desired browser."
    )
    input_schema = {
        "type": "object",
        "properties": {},
    }

    def __init__(
        self,
        browser_session: Optional["BrowserSession"] = None,
        switch_handler: Optional[Any] = None,
    ):
        self._session = browser_session
        self._switch_handler = switch_handler

    async def execute(
        self, *, progress_callback=None, **kwargs,
    ) -> "ToolResult":
        """Broadcast connection request to switch browser."""
        # Real handler
        if self._switch_handler:
            try:
                result = self._switch_handler()
                return self._success(str(result), status="switching")
            except Exception as e:
                return self._error(f"Browser switch error: {e}")

        # Simulated — broadcast request
        request_id = str(uuid.uuid4())[:8]

        return self._success(
            "Connection request broadcast. The user should click 'Connect' "
            "in the desired Chrome browser.",
            request_id=request_id,
            status="awaiting_connection",
        )
