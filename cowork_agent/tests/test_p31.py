"""
Sprint 31 Tests — MCP Registry + Plugin Marketplace.

Tests:
  - ConnectorRegistry: register, unregister, search, scoring, status tracking
  - PluginRegistry: register, search, install status
  - SearchMCPRegistryTool: keyword search, result formatting, error handling
  - SuggestConnectorsTool: UUID lookup, suggestion output
  - SearchPluginsTool: keyword search, result formatting
  - SuggestPluginInstallTool: suggestion banner, already-installed check
  - Default catalogs: built-in connectors and plugins
  - Main.py wiring
"""

import asyncio
import unittest
from unittest.mock import MagicMock

from cowork_agent.core.connector_registry import (
    ConnectorInfo,
    ConnectorRegistry,
    PluginInfo,
    PluginRegistry,
    create_default_connector_catalog,
    create_default_plugin_catalog,
)
from cowork_agent.tools.mcp_registry_tools import (
    SearchMCPRegistryTool,
    SuggestConnectorsTool,
    SearchPluginsTool,
    SuggestPluginInstallTool,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════
# TEST: ConnectorRegistry
# ═══════════════════════════════════════════════════════════════════════

class TestConnectorRegistry(unittest.TestCase):

    def setUp(self):
        self.reg = ConnectorRegistry()
        self.conn1 = ConnectorInfo(
            uuid="slack-001", name="Slack",
            description="Team messaging platform",
            keywords=["slack", "messaging", "chat"],
        )
        self.conn2 = ConnectorInfo(
            uuid="jira-001", name="Jira",
            description="Issue tracking for software teams",
            keywords=["jira", "issues", "bugs", "sprint"],
        )

    def test_register_and_get(self):
        self.reg.register(self.conn1)
        self.assertIsNotNone(self.reg.get("slack-001"))
        self.assertEqual(self.reg.get("slack-001").name, "Slack")

    def test_get_nonexistent(self):
        self.assertIsNone(self.reg.get("nope"))

    def test_unregister(self):
        self.reg.register(self.conn1)
        self.assertTrue(self.reg.unregister("slack-001"))
        self.assertIsNone(self.reg.get("slack-001"))

    def test_unregister_nonexistent(self):
        self.assertFalse(self.reg.unregister("nope"))

    def test_len(self):
        self.assertEqual(len(self.reg), 0)
        self.reg.register(self.conn1)
        self.assertEqual(len(self.reg), 1)

    def test_mark_connected(self):
        self.reg.register(self.conn1)
        self.assertTrue(self.reg.mark_connected("slack-001", ["slack_send", "slack_read"]))
        conn = self.reg.get("slack-001")
        self.assertTrue(conn.connected)
        self.assertEqual(conn.tool_names, ["slack_send", "slack_read"])

    def test_mark_connected_nonexistent(self):
        self.assertFalse(self.reg.mark_connected("nope"))

    def test_mark_disconnected(self):
        self.reg.register(self.conn1)
        self.reg.mark_connected("slack-001", ["tool1"])
        self.assertTrue(self.reg.mark_disconnected("slack-001"))
        conn = self.reg.get("slack-001")
        self.assertFalse(conn.connected)
        self.assertEqual(conn.tool_names, [])

    def test_search_by_name(self):
        self.reg.register(self.conn1)
        self.reg.register(self.conn2)
        results = self.reg.search(["slack"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Slack")

    def test_search_by_keyword(self):
        self.reg.register(self.conn1)
        self.reg.register(self.conn2)
        results = self.reg.search(["issues"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Jira")

    def test_search_by_description(self):
        self.reg.register(self.conn1)
        results = self.reg.search(["messaging"])
        self.assertEqual(len(results), 1)

    def test_search_multiple_keywords(self):
        self.reg.register(self.conn1)
        self.reg.register(self.conn2)
        results = self.reg.search(["team", "messaging"])
        self.assertGreater(len(results), 0)
        # Slack should rank higher (matches both)
        self.assertEqual(results[0].name, "Slack")

    def test_search_empty_keywords(self):
        self.reg.register(self.conn1)
        self.assertEqual(self.reg.search([]), [])

    def test_search_no_match(self):
        self.reg.register(self.conn1)
        self.assertEqual(self.reg.search(["quantum"]), [])

    def test_search_exclude_connected(self):
        self.reg.register(self.conn1)
        self.reg.register(self.conn2)
        self.reg.mark_connected("slack-001")
        results = self.reg.search(["slack", "jira"], include_connected=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Jira")

    def test_all_connectors(self):
        self.reg.register(self.conn1)
        self.reg.register(self.conn2)
        self.assertEqual(len(self.reg.all_connectors), 2)

    def test_connected_connectors(self):
        self.reg.register(self.conn1)
        self.reg.register(self.conn2)
        self.reg.mark_connected("slack-001")
        self.assertEqual(len(self.reg.connected_connectors), 1)

    def test_available_connectors(self):
        self.reg.register(self.conn1)
        self.reg.register(self.conn2)
        self.reg.mark_connected("slack-001")
        self.assertEqual(len(self.reg.available_connectors), 1)

    def test_scoring_exact_name_highest(self):
        self.reg.register(self.conn1)
        self.reg.register(ConnectorInfo(
            uuid="slack-alt",
            name="Slackbot Helper",
            description="A helper for slack",
            keywords=["helper"],
        ))
        results = self.reg.search(["slack"])
        # Exact name match "slack" == "slack" scores 10; partial scores 5
        self.assertEqual(results[0].uuid, "slack-001")

    def test_search_case_insensitive(self):
        self.reg.register(self.conn1)
        results = self.reg.search(["SLACK"])
        self.assertEqual(len(results), 1)

    def test_search_whitespace_keywords(self):
        self.reg.register(self.conn1)
        results = self.reg.search(["  slack  ", ""])
        self.assertEqual(len(results), 1)


# ═══════════════════════════════════════════════════════════════════════
# TEST: PluginRegistry
# ═══════════════════════════════════════════════════════════════════════

class TestPluginRegistry(unittest.TestCase):

    def setUp(self):
        self.reg = PluginRegistry()
        self.p1 = PluginInfo(
            plugin_id="sales-plugin", name="Sales",
            description="Sales call preparation and follow-up",
            keywords=["sales", "call", "crm"],
        )
        self.p2 = PluginInfo(
            plugin_id="legal-plugin", name="Legal",
            description="Contract review and compliance",
            keywords=["legal", "contract", "compliance"],
        )

    def test_register_and_get(self):
        self.reg.register(self.p1)
        self.assertIsNotNone(self.reg.get("sales-plugin"))

    def test_get_nonexistent(self):
        self.assertIsNone(self.reg.get("nope"))

    def test_unregister(self):
        self.reg.register(self.p1)
        self.assertTrue(self.reg.unregister("sales-plugin"))
        self.assertIsNone(self.reg.get("sales-plugin"))

    def test_mark_installed(self):
        self.reg.register(self.p1)
        self.assertTrue(self.reg.mark_installed("sales-plugin"))
        self.assertTrue(self.reg.get("sales-plugin").installed)

    def test_search_excludes_installed_by_default(self):
        self.reg.register(self.p1)
        self.reg.register(self.p2)
        self.reg.mark_installed("sales-plugin")
        results = self.reg.search(["sales", "legal"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Legal")

    def test_search_includes_installed_when_asked(self):
        self.reg.register(self.p1)
        self.reg.mark_installed("sales-plugin")
        results = self.reg.search(["sales"], include_installed=True)
        self.assertEqual(len(results), 1)

    def test_search_no_match(self):
        self.reg.register(self.p1)
        self.assertEqual(self.reg.search(["quantum"]), [])

    def test_all_plugins(self):
        self.reg.register(self.p1)
        self.reg.register(self.p2)
        self.assertEqual(len(self.reg.all_plugins), 2)

    def test_installed_plugins(self):
        self.reg.register(self.p1)
        self.reg.register(self.p2)
        self.reg.mark_installed("sales-plugin")
        self.assertEqual(len(self.reg.installed_plugins), 1)

    def test_len(self):
        self.assertEqual(len(self.reg), 0)
        self.reg.register(self.p1)
        self.assertEqual(len(self.reg), 1)


# ═══════════════════════════════════════════════════════════════════════
# TEST: Default Catalogs
# ═══════════════════════════════════════════════════════════════════════

class TestDefaultCatalogs(unittest.TestCase):

    def test_connector_catalog_has_entries(self):
        cat = create_default_connector_catalog()
        self.assertGreater(len(cat), 0)

    def test_connector_catalog_has_common_services(self):
        cat = create_default_connector_catalog()
        names = {c.name for c in cat.all_connectors}
        self.assertIn("Slack", names)
        self.assertIn("Jira", names)
        self.assertIn("GitHub", names)
        self.assertIn("Gmail", names)
        self.assertIn("Google Drive", names)

    def test_connector_catalog_all_disconnected(self):
        cat = create_default_connector_catalog()
        for conn in cat.all_connectors:
            self.assertFalse(conn.connected)

    def test_connector_catalog_all_have_keywords(self):
        cat = create_default_connector_catalog()
        for conn in cat.all_connectors:
            self.assertGreater(len(conn.keywords), 0, f"{conn.name} has no keywords")

    def test_plugin_catalog_has_entries(self):
        cat = create_default_plugin_catalog()
        self.assertGreater(len(cat), 0)

    def test_plugin_catalog_has_common_plugins(self):
        cat = create_default_plugin_catalog()
        names = {p.name for p in cat.all_plugins}
        self.assertIn("Sales", names)
        self.assertIn("Legal", names)

    def test_plugin_catalog_all_uninstalled(self):
        cat = create_default_plugin_catalog()
        for p in cat.all_plugins:
            self.assertFalse(p.installed)

    def test_connector_searchable_by_common_queries(self):
        cat = create_default_connector_catalog()
        # User says "check my email"
        results = cat.search(["email"])
        names = {r.name for r in results}
        self.assertIn("Gmail", names)

    def test_connector_search_tasks(self):
        cat = create_default_connector_catalog()
        results = cat.search(["tasks", "project management"])
        names = {r.name for r in results}
        self.assertTrue(names & {"Asana", "Trello"})

    def test_connector_search_crm(self):
        cat = create_default_connector_catalog()
        results = cat.search(["crm", "sales"])
        names = {r.name for r in results}
        self.assertTrue(names & {"Zoho CRM", "Salesforce", "HubSpot"})


# ═══════════════════════════════════════════════════════════════════════
# TEST: SearchMCPRegistryTool
# ═══════════════════════════════════════════════════════════════════════

class TestSearchMCPRegistryTool(unittest.TestCase):

    def setUp(self):
        self.cat = create_default_connector_catalog()
        self.tool = SearchMCPRegistryTool(connector_registry=self.cat)

    def test_search_finds_slack(self):
        result = run(self.tool.execute(keywords=["slack", "messaging"]))
        self.assertTrue(result.success)
        self.assertIn("Slack", result.output)
        self.assertGreater(result.metadata["match_count"], 0)

    def test_search_no_match(self):
        result = run(self.tool.execute(keywords=["quantum_nonexistent"]))
        self.assertTrue(result.success)
        self.assertIn("No connectors found", result.output)
        self.assertEqual(result.metadata["match_count"], 0)

    def test_search_empty_keywords_error(self):
        result = run(self.tool.execute(keywords=[]))
        self.assertFalse(result.success)

    def test_search_none_keywords_error(self):
        result = run(self.tool.execute(keywords=None))
        self.assertFalse(result.success)

    def test_search_invalid_type_error(self):
        result = run(self.tool.execute(keywords="not a list"))
        self.assertFalse(result.success)

    def test_search_no_registry_error(self):
        tool = SearchMCPRegistryTool(connector_registry=None)
        result = run(tool.execute(keywords=["slack"]))
        self.assertFalse(result.success)

    def test_search_shows_connection_status(self):
        self.cat.mark_connected("slack-001")
        result = run(self.tool.execute(keywords=["slack"]))
        self.assertIn("Connected", result.output)

    def test_search_returns_unconnected_uuids(self):
        result = run(self.tool.execute(keywords=["slack"]))
        self.assertIn("slack-001", result.metadata["unconnected_uuids"])

    def test_schema_has_keywords(self):
        self.assertIn("keywords", self.tool.input_schema["properties"])
        self.assertEqual(self.tool.input_schema["required"], ["keywords"])


# ═══════════════════════════════════════════════════════════════════════
# TEST: SuggestConnectorsTool
# ═══════════════════════════════════════════════════════════════════════

class TestSuggestConnectorsTool(unittest.TestCase):

    def setUp(self):
        self.cat = create_default_connector_catalog()
        self.tool = SuggestConnectorsTool(connector_registry=self.cat)

    def test_suggest_valid_uuid(self):
        result = run(self.tool.execute(uuids=["slack-001"]))
        self.assertTrue(result.success)
        self.assertIn("Slack", result.output)
        self.assertEqual(result.metadata["suggested_count"], 1)

    def test_suggest_multiple_uuids(self):
        result = run(self.tool.execute(uuids=["slack-001", "jira-001"]))
        self.assertTrue(result.success)
        self.assertIn("Slack", result.output)
        self.assertIn("Jira", result.output)
        self.assertEqual(result.metadata["suggested_count"], 2)

    def test_suggest_unknown_uuid_error(self):
        result = run(self.tool.execute(uuids=["nonexistent-uuid"]))
        self.assertFalse(result.success)

    def test_suggest_mixed_known_unknown(self):
        result = run(self.tool.execute(uuids=["slack-001", "nonexistent"]))
        self.assertTrue(result.success)
        self.assertIn("Slack", result.output)
        self.assertIn("nonexistent", result.output)  # Mentioned in warning

    def test_suggest_empty_uuids_error(self):
        result = run(self.tool.execute(uuids=[]))
        self.assertFalse(result.success)

    def test_suggest_none_uuids_error(self):
        result = run(self.tool.execute(uuids=None))
        self.assertFalse(result.success)

    def test_suggest_no_registry_error(self):
        tool = SuggestConnectorsTool(connector_registry=None)
        result = run(tool.execute(uuids=["slack-001"]))
        self.assertFalse(result.success)

    def test_suggest_shows_connected_status(self):
        self.cat.mark_connected("slack-001")
        result = run(self.tool.execute(uuids=["slack-001"]))
        self.assertIn("Already connected", result.output)

    def test_schema_has_uuids(self):
        self.assertIn("uuids", self.tool.input_schema["properties"])


# ═══════════════════════════════════════════════════════════════════════
# TEST: SearchPluginsTool
# ═══════════════════════════════════════════════════════════════════════

class TestSearchPluginsTool(unittest.TestCase):

    def setUp(self):
        self.cat = create_default_plugin_catalog()
        self.tool = SearchPluginsTool(plugin_registry=self.cat)

    def test_search_finds_sales(self):
        result = run(self.tool.execute(keywords=["sales", "call"]))
        self.assertTrue(result.success)
        self.assertIn("Sales", result.output)
        self.assertGreater(result.metadata["match_count"], 0)

    def test_search_finds_legal(self):
        result = run(self.tool.execute(keywords=["contract", "review"]))
        self.assertTrue(result.success)
        self.assertIn("Legal", result.output)

    def test_search_no_match(self):
        result = run(self.tool.execute(keywords=["quantum_nonexistent"]))
        self.assertTrue(result.success)
        self.assertIn("No plugins found", result.output)

    def test_search_empty_error(self):
        result = run(self.tool.execute(keywords=[]))
        self.assertFalse(result.success)

    def test_search_no_registry_error(self):
        tool = SearchPluginsTool(plugin_registry=None)
        result = run(tool.execute(keywords=["sales"]))
        self.assertFalse(result.success)

    def test_search_excludes_installed(self):
        self.cat.mark_installed("sales-plugin")
        result = run(self.tool.execute(keywords=["sales"]))
        # Sales should not appear since it's installed
        self.assertNotIn("sales-plugin", result.metadata.get("plugin_ids", []))

    def test_schema_has_keywords(self):
        self.assertIn("keywords", self.tool.input_schema["properties"])


# ═══════════════════════════════════════════════════════════════════════
# TEST: SuggestPluginInstallTool
# ═══════════════════════════════════════════════════════════════════════

class TestSuggestPluginInstallTool(unittest.TestCase):

    def setUp(self):
        self.cat = create_default_plugin_catalog()
        self.tool = SuggestPluginInstallTool(plugin_registry=self.cat)

    def test_suggest_valid_plugin(self):
        result = run(self.tool.execute(pluginId="sales-plugin", pluginName="Sales"))
        self.assertTrue(result.success)
        self.assertIn("Sales", result.output)
        self.assertTrue(result.metadata.get("suggested"))

    def test_suggest_already_installed(self):
        self.cat.mark_installed("sales-plugin")
        result = run(self.tool.execute(pluginId="sales-plugin", pluginName="Sales"))
        self.assertTrue(result.success)
        self.assertIn("already installed", result.output)
        self.assertTrue(result.metadata.get("already_installed"))

    def test_suggest_unknown_plugin(self):
        result = run(self.tool.execute(pluginId="nonexistent", pluginName="X"))
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)

    def test_suggest_missing_id_error(self):
        result = run(self.tool.execute(pluginId="", pluginName="Sales"))
        self.assertFalse(result.success)

    def test_suggest_missing_name_error(self):
        result = run(self.tool.execute(pluginId="sales-plugin", pluginName=""))
        self.assertFalse(result.success)

    def test_suggest_no_registry_error(self):
        tool = SuggestPluginInstallTool(plugin_registry=None)
        result = run(tool.execute(pluginId="sales-plugin", pluginName="Sales"))
        self.assertFalse(result.success)

    def test_schema_has_required_fields(self):
        self.assertIn("pluginId", self.tool.input_schema["properties"])
        self.assertIn("pluginName", self.tool.input_schema["properties"])
        self.assertIn("pluginName", self.tool.input_schema["required"])
        self.assertIn("pluginId", self.tool.input_schema["required"])


# ═══════════════════════════════════════════════════════════════════════
# TEST: Tool Names and Descriptions
# ═══════════════════════════════════════════════════════════════════════

class TestToolNamesAndDescriptions(unittest.TestCase):

    def test_search_mcp_registry_name(self):
        tool = SearchMCPRegistryTool()
        self.assertEqual(tool.name, "search_mcp_registry")

    def test_suggest_connectors_name(self):
        tool = SuggestConnectorsTool()
        self.assertEqual(tool.name, "suggest_connectors")

    def test_search_plugins_name(self):
        tool = SearchPluginsTool()
        self.assertEqual(tool.name, "search_plugins")

    def test_suggest_plugin_install_name(self):
        tool = SuggestPluginInstallTool()
        self.assertEqual(tool.name, "suggest_plugin_install")

    def test_all_have_descriptions(self):
        for cls in [SearchMCPRegistryTool, SuggestConnectorsTool,
                     SearchPluginsTool, SuggestPluginInstallTool]:
            tool = cls()
            self.assertGreater(len(tool.description), 20, f"{cls.__name__} has short description")


# ═══════════════════════════════════════════════════════════════════════
# TEST: ConnectorInfo & PluginInfo Dataclasses
# ═══════════════════════════════════════════════════════════════════════

class TestDataclasses(unittest.TestCase):

    def test_connector_info_defaults(self):
        c = ConnectorInfo(uuid="x", name="X", description="desc")
        self.assertFalse(c.connected)
        self.assertEqual(c.connector_type, "mcp")
        self.assertEqual(c.tool_names, [])
        self.assertEqual(c.keywords, [])

    def test_plugin_info_defaults(self):
        p = PluginInfo(plugin_id="x", name="X", description="desc")
        self.assertFalse(p.installed)
        self.assertEqual(p.keywords, [])

    def test_connector_info_with_all_fields(self):
        c = ConnectorInfo(
            uuid="u1", name="N", description="D",
            keywords=["k1"], connected=True, connector_type="plugin",
            tool_names=["t1"], icon_url="http://icon", homepage_url="http://home",
            version="1.0",
        )
        self.assertTrue(c.connected)
        self.assertEqual(c.version, "1.0")


# ═══════════════════════════════════════════════════════════════════════
# TEST: Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def test_register_overwrites_existing(self):
        reg = ConnectorRegistry()
        reg.register(ConnectorInfo(uuid="x", name="Old", description="old"))
        reg.register(ConnectorInfo(uuid="x", name="New", description="new"))
        self.assertEqual(reg.get("x").name, "New")

    def test_search_with_only_whitespace_keywords(self):
        reg = ConnectorRegistry()
        reg.register(ConnectorInfo(uuid="x", name="Slack", description="chat", keywords=["chat"]))
        self.assertEqual(reg.search(["   ", ""]), [])

    def test_plugin_search_empty_registry(self):
        reg = PluginRegistry()
        self.assertEqual(reg.search(["anything"]), [])

    def test_connector_search_empty_registry(self):
        reg = ConnectorRegistry()
        self.assertEqual(reg.search(["anything"]), [])

    def test_search_returns_sorted_by_relevance(self):
        reg = ConnectorRegistry()
        reg.register(ConnectorInfo(
            uuid="low", name="Misc Tool", description="generic",
            keywords=["generic"],
        ))
        reg.register(ConnectorInfo(
            uuid="high", name="Slack", description="Slack messaging",
            keywords=["slack", "messaging", "chat"],
        ))
        results = reg.search(["slack", "messaging"])
        self.assertEqual(results[0].uuid, "high")

    def test_mark_connected_with_tools_then_disconnect(self):
        reg = ConnectorRegistry()
        reg.register(ConnectorInfo(uuid="x", name="X", description="d"))
        reg.mark_connected("x", ["tool1", "tool2"])
        self.assertEqual(len(reg.get("x").tool_names), 2)
        reg.mark_disconnected("x")
        self.assertEqual(len(reg.get("x").tool_names), 0)

    def test_multiple_connectors_same_keyword(self):
        reg = ConnectorRegistry()
        reg.register(ConnectorInfo(
            uuid="a", name="A", description="d", keywords=["tasks"],
        ))
        reg.register(ConnectorInfo(
            uuid="b", name="B", description="d", keywords=["tasks"],
        ))
        results = reg.search(["tasks"])
        self.assertEqual(len(results), 2)

    def test_suggest_connectors_returns_uuids_in_metadata(self):
        cat = create_default_connector_catalog()
        tool = SuggestConnectorsTool(connector_registry=cat)
        result = run(tool.execute(uuids=["slack-001", "jira-001"]))
        self.assertIn("slack-001", result.metadata["suggested_uuids"])
        self.assertIn("jira-001", result.metadata["suggested_uuids"])

    def test_search_plugins_returns_plugin_ids(self):
        cat = create_default_plugin_catalog()
        tool = SearchPluginsTool(plugin_registry=cat)
        result = run(tool.execute(keywords=["sales"]))
        self.assertIn("sales-plugin", result.metadata.get("plugin_ids", []))

    def test_end_to_end_search_then_suggest(self):
        """Full workflow: search → get UUIDs → suggest."""
        cat = create_default_connector_catalog()
        search_tool = SearchMCPRegistryTool(connector_registry=cat)
        suggest_tool = SuggestConnectorsTool(connector_registry=cat)

        search_result = run(search_tool.execute(keywords=["email"]))
        self.assertTrue(search_result.success)
        uuids = search_result.metadata.get("unconnected_uuids", [])
        self.assertGreater(len(uuids), 0)

        suggest_result = run(suggest_tool.execute(uuids=uuids))
        self.assertTrue(suggest_result.success)
        self.assertIn("Gmail", suggest_result.output)

    def test_end_to_end_plugin_search_then_suggest(self):
        """Full workflow: search plugins → suggest install."""
        cat = create_default_plugin_catalog()
        search_tool = SearchPluginsTool(plugin_registry=cat)
        suggest_tool = SuggestPluginInstallTool(plugin_registry=cat)

        search_result = run(search_tool.execute(keywords=["contract", "legal"]))
        self.assertTrue(search_result.success)
        plugin_ids = search_result.metadata.get("plugin_ids", [])
        self.assertGreater(len(plugin_ids), 0)

        suggest_result = run(suggest_tool.execute(
            pluginId=plugin_ids[0], pluginName="Legal"
        ))
        self.assertTrue(suggest_result.success)
        self.assertTrue(suggest_result.metadata.get("suggested"))


if __name__ == "__main__":
    unittest.main()
