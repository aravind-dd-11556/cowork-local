"""
Plugin System — Discover and load user-installable tool plugins.

Plugins are Python packages in a plugins directory with a standard structure:
  plugins/
    my_plugin/
      __init__.py        # Must export: TOOLS (list of tool classes)
      plugin.json        # Plugin metadata (name, description, version)
      tools.py           # Tool implementations

Plugin discovery locations:
  1. workspace/.cowork/plugins/
  2. ~/.cowork_agent/plugins/

Each plugin's __init__.py must export:
  TOOLS: list — A list of BaseTool subclass instances to register
"""

from __future__ import annotations
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Metadata about a loaded plugin."""
    name: str
    description: str
    version: str
    location: str
    tool_names: list[str] = field(default_factory=list)
    enabled: bool = True
    error: str = ""


class PluginSystem:
    """
    Discovers and loads tool plugins from filesystem.

    Usage:
        plugins = PluginSystem(workspace_dir="/path/to/workspace")
        loaded = plugins.discover_and_load()
        # loaded is a list of (PluginInfo, [tool_instances])
    """

    def __init__(self, workspace_dir: str = "", user_plugins_dir: str = ""):
        self.workspace_dir = workspace_dir
        self.user_plugins_dir = user_plugins_dir or os.path.expanduser("~/.cowork_agent/plugins")
        self._plugins: dict[str, PluginInfo] = {}

    @property
    def plugins(self) -> dict[str, PluginInfo]:
        return dict(self._plugins)

    @property
    def plugin_names(self) -> list[str]:
        return list(self._plugins.keys())

    def discover_and_load(self) -> list[tuple[PluginInfo, list]]:
        """
        Scan all plugin directories, load plugins, and return their tools.

        Returns:
            List of (PluginInfo, [tool_instances]) tuples
        """
        self._plugins.clear()
        results = []

        # Scan plugin directories
        dirs_to_scan = []
        if self.workspace_dir:
            dirs_to_scan.append(os.path.join(self.workspace_dir, ".cowork", "plugins"))
        dirs_to_scan.append(self.user_plugins_dir)

        for plugins_dir in dirs_to_scan:
            if not os.path.isdir(plugins_dir):
                continue

            for entry in sorted(os.listdir(plugins_dir)):
                plugin_dir = os.path.join(plugins_dir, entry)
                init_file = os.path.join(plugin_dir, "__init__.py")

                if os.path.isdir(plugin_dir) and os.path.isfile(init_file):
                    # Don't overwrite workspace plugins with user plugins
                    if entry in self._plugins:
                        continue

                    info, tools = self._load_plugin(entry, plugin_dir)
                    self._plugins[entry] = info
                    results.append((info, tools))

        logger.info(f"Loaded {len(results)} plugins: {self.plugin_names}")
        return results

    def _load_plugin(self, name: str, plugin_dir: str) -> tuple[PluginInfo, list]:
        """Load a single plugin and return its info and tool instances."""

        # Load plugin.json metadata (optional)
        metadata = self._load_metadata(plugin_dir)

        info = PluginInfo(
            name=metadata.get("name", name),
            description=metadata.get("description", ""),
            version=metadata.get("version", "0.0.0"),
            location=plugin_dir,
        )

        # Try to import the plugin module
        try:
            # Add plugin parent dir to sys.path temporarily for import resolution
            parent_dir = os.path.dirname(plugin_dir)
            added_to_path = False
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
                added_to_path = True

            spec = importlib.util.spec_from_file_location(
                f"cowork_plugin_{name}",
                os.path.join(plugin_dir, "__init__.py"),
            )
            if spec is None or spec.loader is None:
                info.error = "Failed to create module spec"
                info.enabled = False
                return info, []

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get TOOLS export
            tools = getattr(module, "TOOLS", [])
            if not isinstance(tools, (list, tuple)):
                info.error = f"TOOLS export is not a list: {type(tools)}"
                info.enabled = False
                return info, []

            info.tool_names = [
                getattr(t, "name", str(i)) for i, t in enumerate(tools)
            ]
            logger.info(f"Plugin '{name}' loaded: {info.tool_names}")
            return info, list(tools)

        except Exception as e:
            info.error = str(e)
            info.enabled = False
            logger.warning(f"Failed to load plugin '{name}': {e}")
            return info, []

        finally:
            # Clean up sys.path to avoid pollution
            if added_to_path and parent_dir in sys.path:
                sys.path.remove(parent_dir)

    @staticmethod
    def _load_metadata(plugin_dir: str) -> dict:
        """Load plugin.json metadata file."""
        meta_file = os.path.join(plugin_dir, "plugin.json")
        if not os.path.isfile(meta_file):
            return {}
        try:
            with open(meta_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def get_plugins_summary(self) -> str:
        """Get a summary of all loaded plugins for display."""
        if not self._plugins:
            return "No plugins loaded."

        lines = []
        for info in self._plugins.values():
            status = "✅" if info.enabled else "❌"
            tools = ", ".join(info.tool_names) if info.tool_names else "no tools"
            lines.append(f"  {status} {info.name} v{info.version} — {tools}")
            if info.error:
                lines.append(f"     ⚠️ {info.error}")

        return "\n".join(lines)
