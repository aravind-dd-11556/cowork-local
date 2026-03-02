"""
MCP Registry & Plugin Marketplace Tools — Sprint 31.

Mirrors real Cowork's discovery layer:
  - search_mcp_registry: Search for available MCP connectors by keywords
  - suggest_connectors: Present connector suggestions to the user
  - search_plugins: Search for installable plugins
  - suggest_plugin_install: Display plugin install suggestion to user

These tools enable the agent to discover and recommend external
integrations when users ask about services the agent doesn't yet
have access to.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Optional

from .base import BaseTool

if TYPE_CHECKING:
    from ..core.connector_registry import ConnectorRegistry, PluginRegistry

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Tool 1: search_mcp_registry
# ═══════════════════════════════════════════════════════════════════════

class SearchMCPRegistryTool(BaseTool):
    """
    Search for available MCP connectors by keywords.

    Call this when the user asks about external apps and you don't have
    a matching connector already available. Returns results with
    connected status.
    """
    name = "search_mcp_registry"
    description = (
        "Search for available connectors. Call this when the user asks about "
        "external apps and you don't have a matching connector already available.\n\n"
        "Examples:\n"
        '- "check my Asana tasks" → search ["asana", "tasks", "todo"]\n'
        '- "find issues in Jira" → search ["jira", "issues"]\n\n'
        "Returns results with connected status. Call suggest_connectors "
        "to show unconnected ones to the user."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Search keywords in English extracted from user's request "
                    '(e.g., ["asana", "tasks", "todo"] for task-related requests)'
                ),
            },
        },
        "required": ["keywords"],
    }

    def __init__(self, connector_registry: Optional["ConnectorRegistry"] = None):
        self._registry = connector_registry

    async def execute(
        self, *, progress_callback=None, keywords=None, **kwargs
    ) -> "ToolResult":
        if not keywords:
            return self._error("'keywords' parameter is required and must be a non-empty list.")

        if not isinstance(keywords, list):
            return self._error("'keywords' must be a list of strings.")

        if not self._registry:
            return self._error("No connector registry available.")

        results = self._registry.search(keywords, include_connected=True)

        if not results:
            return self._success(
                f"No connectors found matching keywords: {', '.join(keywords)}. "
                "No matching integrations are available.",
                match_count=0,
            )

        lines = [f"Found {len(results)} connector(s) matching: {', '.join(keywords)}\n"]
        unconnected_uuids = []

        for conn in results:
            status = "✅ Connected" if conn.connected else "⬜ Not connected"
            lines.append(
                f"- **{conn.name}** ({conn.uuid}): {conn.description} [{status}]"
            )
            if not conn.connected:
                unconnected_uuids.append(conn.uuid)

        if unconnected_uuids:
            lines.append(
                f"\nTo suggest unconnected connectors to the user, call "
                f"suggest_connectors with UUIDs: {unconnected_uuids}"
            )

        return self._success(
            "\n".join(lines),
            match_count=len(results),
            unconnected_uuids=unconnected_uuids,
        )


# ═══════════════════════════════════════════════════════════════════════
# Tool 2: suggest_connectors
# ═══════════════════════════════════════════════════════════════════════

class SuggestConnectorsTool(BaseTool):
    """
    Display connector suggestions to the user with Connect buttons.

    Call this after search_mcp_registry when results include unconnected
    connectors that would help with the user's task.
    """
    name = "suggest_connectors"
    description = (
        "Display connector suggestions to the user with Connect buttons. Call this:\n"
        "- After search_mcp_registry when search returned connectors that are NOT "
        "already connected AND would help with the user's task\n"
        "- When a tool call fails with an authentication or credential error\n\n"
        "Do NOT call this if the connector is already connected and working."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "uuids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "UUIDs of connectors to suggest. From search results' directoryUuid, "
                    "or extract the server UUID from a failed tool name "
                    "(format: mcp__{uuid}__{toolName})."
                ),
            },
        },
        "required": ["uuids"],
    }

    def __init__(self, connector_registry: Optional["ConnectorRegistry"] = None):
        self._registry = connector_registry

    async def execute(
        self, *, progress_callback=None, uuids=None, **kwargs
    ) -> "ToolResult":
        if not uuids:
            return self._error("'uuids' parameter is required and must be a non-empty list.")

        if not isinstance(uuids, list):
            return self._error("'uuids' must be a list of strings.")

        if not self._registry:
            return self._error("No connector registry available.")

        suggestions = []
        not_found = []

        for uuid in uuids:
            conn = self._registry.get(uuid)
            if conn:
                suggestions.append(conn)
            else:
                not_found.append(uuid)

        if not suggestions:
            return self._error(
                f"No connectors found for UUIDs: {', '.join(uuids)}. "
                "Verify the UUIDs from search results."
            )

        lines = ["The following connectors are available for setup:\n"]
        for conn in suggestions:
            status = "Already connected" if conn.connected else "Not yet connected"
            lines.append(
                f"📦 **{conn.name}** — {conn.description}\n"
                f"   Status: {status} | UUID: {conn.uuid}"
            )

        if not_found:
            lines.append(f"\n⚠️ UUIDs not found in registry: {', '.join(not_found)}")

        lines.append(
            "\nTo connect a service, configure its MCP server in your settings."
        )

        return self._success(
            "\n".join(lines),
            suggested_count=len(suggestions),
            suggested_uuids=[c.uuid for c in suggestions],
        )


# ═══════════════════════════════════════════════════════════════════════
# Tool 3: search_plugins
# ═══════════════════════════════════════════════════════════════════════

class SearchPluginsTool(BaseTool):
    """
    Search for available plugins that can be installed.

    Call this silently when the user's request involves a specific
    professional workflow that might benefit from a specialized plugin.
    """
    name = "search_plugins"
    description = (
        "Search for available plugins that can be installed. "
        "Call this when the user's request involves a specific professional "
        "workflow that might benefit from a specialized plugin.\n\n"
        "Examples:\n"
        '- "help me prep for a sales call" → search ["sales", "call", "prep"]\n'
        '- "review this contract" → search ["legal", "contract", "review"]\n\n'
        "Returns matching uninstalled plugins. If relevant, call "
        "suggest_plugin_install."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Search keywords extracted from user's request "
                    '(e.g., ["sales", "email", "follow-up"])'
                ),
            },
        },
        "required": ["keywords"],
    }

    def __init__(self, plugin_registry: Optional["PluginRegistry"] = None):
        self._registry = plugin_registry

    async def execute(
        self, *, progress_callback=None, keywords=None, **kwargs
    ) -> "ToolResult":
        if not keywords:
            return self._error("'keywords' parameter is required and must be a non-empty list.")

        if not isinstance(keywords, list):
            return self._error("'keywords' must be a list of strings.")

        if not self._registry:
            return self._error("No plugin registry available.")

        results = self._registry.search(keywords, include_installed=False)

        if not results:
            return self._success(
                f"No plugins found matching keywords: {', '.join(keywords)}. "
                "Proceed with the user's request using available tools.",
                match_count=0,
            )

        lines = [f"Found {len(results)} plugin(s) matching: {', '.join(keywords)}\n"]
        for plugin in results:
            lines.append(
                f"- **{plugin.name}** (ID: {plugin.plugin_id}): {plugin.description}"
            )

        return self._success(
            "\n".join(lines),
            match_count=len(results),
            plugin_ids=[p.plugin_id for p in results],
        )


# ═══════════════════════════════════════════════════════════════════════
# Tool 4: suggest_plugin_install
# ═══════════════════════════════════════════════════════════════════════

class SuggestPluginInstallTool(BaseTool):
    """
    Display a plugin installation suggestion banner to the user.

    Call this silently after search_plugins returns a relevant match.
    """
    name = "suggest_plugin_install"
    description = (
        "Display a plugin installation suggestion banner to the user. "
        "Call this silently after search_plugins returns a relevant match.\n\n"
        "Do NOT call this if:\n"
        "- The suggestion is not relevant to what the user asked about\n"
        "- You are unsure whether the plugin would actually help"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pluginId": {
                "type": "string",
                "description": "The plugin ID from search_plugins results",
            },
            "pluginName": {
                "type": "string",
                "description": "The name of the plugin to suggest (e.g., 'Sales')",
            },
        },
        "required": ["pluginName", "pluginId"],
    }

    def __init__(self, plugin_registry: Optional["PluginRegistry"] = None):
        self._registry = plugin_registry

    async def execute(
        self, *, progress_callback=None, pluginId="", pluginName="", **kwargs
    ) -> "ToolResult":
        if not pluginId:
            return self._error("'pluginId' parameter is required.")
        if not pluginName:
            return self._error("'pluginName' parameter is required.")

        if not self._registry:
            return self._error("No plugin registry available.")

        plugin = self._registry.get(pluginId)
        if not plugin:
            return self._error(
                f"Plugin '{pluginId}' not found in the registry."
            )

        if plugin.installed:
            return self._success(
                f"Plugin '{plugin.name}' is already installed.",
                already_installed=True,
                plugin_id=pluginId,
            )

        return self._success(
            f"💡 **{plugin.name}** plugin is available!\n"
            f"{plugin.description}\n\n"
            f"To install, add the '{plugin.name}' plugin to your configuration.",
            plugin_id=pluginId,
            plugin_name=plugin.name,
            suggested=True,
        )
