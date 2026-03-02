"""
Connector Registry — Discovery layer for MCP connectors and plugins.

Sprint 31: Provides a searchable catalog of available connectors and plugins,
enabling marketplace-style discovery, search, and suggestion.

This sits above the MCP client and plugin system, providing:
  - Keyword-based search across connectors
  - Connection status tracking
  - Suggestion generation for unconnected services
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ConnectorInfo:
    """Metadata for a discoverable connector (MCP server or plugin)."""
    uuid: str
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    connected: bool = False
    connector_type: str = "mcp"  # "mcp" or "plugin"
    tool_names: List[str] = field(default_factory=list)
    icon_url: str = ""
    homepage_url: str = ""
    version: str = ""


@dataclass
class PluginInfo:
    """Metadata for a discoverable plugin."""
    plugin_id: str
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    installed: bool = False
    tool_names: List[str] = field(default_factory=list)
    version: str = ""


class ConnectorRegistry:
    """
    Searchable catalog of available MCP connectors.

    Connectors can be:
      - Pre-registered (built-in catalog of known services)
      - Dynamically discovered from running MCP servers
      - Manually added via configuration

    Search uses keyword matching against name, description, and keywords.
    """

    def __init__(self):
        self._connectors: Dict[str, ConnectorInfo] = {}

    def register(self, connector: ConnectorInfo) -> None:
        """Add or update a connector in the registry."""
        self._connectors[connector.uuid] = connector

    def unregister(self, uuid: str) -> bool:
        """Remove a connector. Returns True if found and removed."""
        if uuid in self._connectors:
            del self._connectors[uuid]
            return True
        return False

    def get(self, uuid: str) -> Optional[ConnectorInfo]:
        """Look up a connector by UUID."""
        return self._connectors.get(uuid)

    def mark_connected(self, uuid: str, tool_names: Optional[List[str]] = None) -> bool:
        """Mark a connector as connected (active)."""
        conn = self._connectors.get(uuid)
        if conn:
            conn.connected = True
            if tool_names:
                conn.tool_names = tool_names
            return True
        return False

    def mark_disconnected(self, uuid: str) -> bool:
        """Mark a connector as disconnected."""
        conn = self._connectors.get(uuid)
        if conn:
            conn.connected = False
            conn.tool_names = []
            return True
        return False

    def search(self, keywords: List[str], include_connected: bool = True) -> List[ConnectorInfo]:
        """
        Search connectors by keywords.

        Matches against name, description, and keyword tags.
        Returns results sorted by relevance (number of keyword matches).
        """
        if not keywords:
            return []

        normalized_keywords = [kw.lower().strip() for kw in keywords if kw.strip()]
        if not normalized_keywords:
            return []

        scored: List[tuple] = []
        for conn in self._connectors.values():
            if not include_connected and conn.connected:
                continue
            score = self._score_match(conn, normalized_keywords)
            if score > 0:
                scored.append((score, conn))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [conn for _, conn in scored]

    def _score_match(self, conn: ConnectorInfo, keywords: List[str]) -> int:
        """Score a connector against search keywords."""
        score = 0
        name_lower = conn.name.lower()
        desc_lower = conn.description.lower()
        conn_keywords = {kw.lower() for kw in conn.keywords}

        for kw in keywords:
            # Exact name match = highest score
            if kw == name_lower:
                score += 10
            # Name contains keyword
            elif kw in name_lower:
                score += 5
            # Keyword tag match
            elif kw in conn_keywords:
                score += 3
            # Description contains keyword
            elif kw in desc_lower:
                score += 1

        return score

    @property
    def all_connectors(self) -> List[ConnectorInfo]:
        return list(self._connectors.values())

    @property
    def connected_connectors(self) -> List[ConnectorInfo]:
        return [c for c in self._connectors.values() if c.connected]

    @property
    def available_connectors(self) -> List[ConnectorInfo]:
        return [c for c in self._connectors.values() if not c.connected]

    def __len__(self) -> int:
        return len(self._connectors)


class PluginRegistry:
    """
    Searchable catalog of available plugins.

    Similar to ConnectorRegistry but for plugins — installable bundles
    of tools, skills, and MCP configurations.
    """

    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}

    def register(self, plugin: PluginInfo) -> None:
        """Add or update a plugin in the registry."""
        self._plugins[plugin.plugin_id] = plugin

    def unregister(self, plugin_id: str) -> bool:
        if plugin_id in self._plugins:
            del self._plugins[plugin_id]
            return True
        return False

    def get(self, plugin_id: str) -> Optional[PluginInfo]:
        return self._plugins.get(plugin_id)

    def mark_installed(self, plugin_id: str) -> bool:
        p = self._plugins.get(plugin_id)
        if p:
            p.installed = True
            return True
        return False

    def search(self, keywords: List[str], include_installed: bool = False) -> List[PluginInfo]:
        """
        Search plugins by keywords.
        By default, only returns uninstalled plugins (marketplace behavior).
        """
        if not keywords:
            return []

        normalized = [kw.lower().strip() for kw in keywords if kw.strip()]
        if not normalized:
            return []

        scored: List[tuple] = []
        for plugin in self._plugins.values():
            if not include_installed and plugin.installed:
                continue
            score = self._score_match(plugin, normalized)
            if score > 0:
                scored.append((score, plugin))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]

    def _score_match(self, plugin: PluginInfo, keywords: List[str]) -> int:
        score = 0
        name_lower = plugin.name.lower()
        desc_lower = plugin.description.lower()
        plugin_keywords = {kw.lower() for kw in plugin.keywords}

        for kw in keywords:
            if kw == name_lower:
                score += 10
            elif kw in name_lower:
                score += 5
            elif kw in plugin_keywords:
                score += 3
            elif kw in desc_lower:
                score += 1

        return score

    @property
    def all_plugins(self) -> List[PluginInfo]:
        return list(self._plugins.values())

    @property
    def installed_plugins(self) -> List[PluginInfo]:
        return [p for p in self._plugins.values() if p.installed]

    def __len__(self) -> int:
        return len(self._plugins)


# ── Built-in Connector Catalog ──────────────────────────────────────────

def create_default_connector_catalog() -> ConnectorRegistry:
    """
    Create a registry pre-populated with well-known connectors.
    These represent services the agent *could* connect to if configured.
    """
    registry = ConnectorRegistry()

    # Common enterprise connectors
    _BUILTIN_CONNECTORS = [
        ConnectorInfo(
            uuid="zoho-crm-001",
            name="Zoho CRM",
            description="Customer relationship management — manage leads, contacts, deals, accounts",
            keywords=["zoho", "crm", "sales", "leads", "contacts", "deals", "accounts", "pipeline"],
        ),
        ConnectorInfo(
            uuid="google-drive-001",
            name="Google Drive",
            description="Cloud file storage — create, share, and collaborate on documents",
            keywords=["google", "drive", "docs", "sheets", "files", "storage", "cloud"],
        ),
        ConnectorInfo(
            uuid="gmail-001",
            name="Gmail",
            description="Email service — read, send, and manage email messages",
            keywords=["gmail", "email", "mail", "inbox", "messages", "send"],
        ),
        ConnectorInfo(
            uuid="slack-001",
            name="Slack",
            description="Team messaging — read and send messages in channels and DMs",
            keywords=["slack", "messaging", "chat", "channels", "team", "communication"],
        ),
        ConnectorInfo(
            uuid="asana-001",
            name="Asana",
            description="Project management — manage tasks, projects, and workflows",
            keywords=["asana", "tasks", "projects", "todo", "project management", "workflow"],
        ),
        ConnectorInfo(
            uuid="jira-001",
            name="Jira",
            description="Issue tracking — manage bugs, stories, and sprint boards",
            keywords=["jira", "issues", "bugs", "sprint", "agile", "tickets", "stories"],
        ),
        ConnectorInfo(
            uuid="github-001",
            name="GitHub",
            description="Code hosting — manage repositories, pull requests, and issues",
            keywords=["github", "git", "repos", "pull requests", "issues", "code", "repository"],
        ),
        ConnectorInfo(
            uuid="notion-001",
            name="Notion",
            description="Knowledge management — create and organize docs, databases, wikis",
            keywords=["notion", "docs", "wiki", "database", "knowledge", "notes", "pages"],
        ),
        ConnectorInfo(
            uuid="dropbox-001",
            name="Dropbox",
            description="Cloud file storage and sharing service",
            keywords=["dropbox", "files", "storage", "cloud", "sharing", "sync"],
        ),
        ConnectorInfo(
            uuid="trello-001",
            name="Trello",
            description="Kanban board for visual project management",
            keywords=["trello", "kanban", "boards", "cards", "project management", "tasks"],
        ),
        ConnectorInfo(
            uuid="salesforce-001",
            name="Salesforce",
            description="CRM platform for sales, service, and marketing",
            keywords=["salesforce", "crm", "sales", "leads", "opportunities", "accounts"],
        ),
        ConnectorInfo(
            uuid="hubspot-001",
            name="HubSpot",
            description="CRM and marketing automation platform",
            keywords=["hubspot", "crm", "marketing", "contacts", "deals", "email"],
        ),
        ConnectorInfo(
            uuid="canva-001",
            name="Canva",
            description="Graphic design platform for creating visual content",
            keywords=["canva", "design", "graphic", "images", "templates", "visual"],
        ),
        ConnectorInfo(
            uuid="linear-001",
            name="Linear",
            description="Issue tracking tool for software teams",
            keywords=["linear", "issues", "bugs", "engineering", "sprint", "backlog"],
        ),
        ConnectorInfo(
            uuid="confluence-001",
            name="Confluence",
            description="Team wiki and documentation platform",
            keywords=["confluence", "wiki", "docs", "documentation", "knowledge base"],
        ),
    ]

    for conn in _BUILTIN_CONNECTORS:
        registry.register(conn)

    return registry


def create_default_plugin_catalog() -> PluginRegistry:
    """
    Create a plugin registry pre-populated with available plugins.
    """
    registry = PluginRegistry()

    _BUILTIN_PLUGINS = [
        PluginInfo(
            plugin_id="sales-plugin",
            name="Sales",
            description="Sales call preparation, follow-up emails, CRM integration",
            keywords=["sales", "call", "prep", "follow-up", "email", "crm", "pipeline"],
        ),
        PluginInfo(
            plugin_id="legal-plugin",
            name="Legal",
            description="Contract review, legal document analysis, compliance checking",
            keywords=["legal", "contract", "review", "compliance", "law", "agreement"],
        ),
        PluginInfo(
            plugin_id="data-analysis-plugin",
            name="Data Analysis",
            description="Advanced data analysis, visualization, and statistical modeling",
            keywords=["data", "analysis", "statistics", "visualization", "charts", "modeling"],
        ),
        PluginInfo(
            plugin_id="project-mgmt-plugin",
            name="Project Management",
            description="Task tracking, sprint planning, and team coordination",
            keywords=["project", "management", "tasks", "sprint", "planning", "todo"],
        ),
        PluginInfo(
            plugin_id="writing-plugin",
            name="Writing Assistant",
            description="Advanced writing tools — grammar, style, tone, and content optimization",
            keywords=["writing", "grammar", "style", "editing", "content", "copywriting"],
        ),
        PluginInfo(
            plugin_id="devops-plugin",
            name="DevOps",
            description="CI/CD pipeline management, deployment automation, monitoring",
            keywords=["devops", "ci", "cd", "deploy", "pipeline", "docker", "kubernetes"],
        ),
    ]

    for plugin in _BUILTIN_PLUGINS:
        registry.register(plugin)

    return registry
