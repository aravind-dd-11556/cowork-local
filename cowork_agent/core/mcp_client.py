"""
MCP Client — Model Context Protocol client for connecting to external services.

Implements the MCP protocol (JSON-RPC over stdio/SSE) to:
  1. Discover tools from MCP servers
  2. Execute tool calls via MCP servers
  3. Manage server lifecycle (start/stop)

MCP servers are configured in the config file:
  mcp_servers:
    - name: "github"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_TOKEN: "..."
    - name: "slack"
      transport: "sse"
      url: "http://localhost:3001/sse"
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str = ""  # For stdio transport
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # "stdio" or "sse"
    url: str = ""  # For SSE transport


@dataclass
class MCPTool:
    """A tool discovered from an MCP server."""
    name: str
    description: str
    input_schema: dict
    server_name: str  # Which MCP server provides this tool


class MCPClient:
    """
    Manages MCP server connections and tool discovery.

    Usage:
        client = MCPClient()
        client.add_server(MCPServerConfig(name="github", command="npx", args=[...]))
        await client.start_all()

        tools = client.get_tools()  # Returns list of MCPTool
        result = await client.call_tool("github__list_repos", {"owner": "..."})

        await client.stop_all()
    """

    def __init__(self):
        self._servers: dict[str, MCPServerConfig] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._tools: dict[str, MCPTool] = {}  # qualified_name -> MCPTool
        self._request_id = 0

    def add_server(self, config: MCPServerConfig) -> None:
        """Register an MCP server configuration."""
        self._servers[config.name] = config
        logger.info(f"Added MCP server: {config.name}")

    async def start_all(self) -> dict[str, bool]:
        """Start all configured MCP servers and discover tools."""
        results = {}
        for name, config in self._servers.items():
            try:
                if config.transport == "stdio":
                    success = await self._start_stdio_server(name, config)
                elif config.transport == "sse":
                    success = await self._start_sse_server(name, config)
                else:
                    logger.warning(f"Unknown transport: {config.transport}")
                    success = False
                results[name] = success
            except Exception as e:
                logger.error(f"Failed to start MCP server '{name}': {e}")
                results[name] = False
        return results

    async def stop_all(self) -> None:
        """Stop all running MCP servers."""
        for name, proc in self._processes.items():
            try:
                proc.terminate()
                proc.wait(timeout=5)
                logger.info(f"Stopped MCP server: {name}")
            except Exception as e:
                logger.warning(f"Error stopping MCP server '{name}': {e}")
                try:
                    proc.kill()
                except Exception:
                    pass
        self._processes.clear()

    # Allowed MCP server commands — only well-known package runners/runtimes
    ALLOWED_MCP_COMMANDS = {"npx", "node", "python", "python3", "uvx", "deno"}

    async def _start_stdio_server(self, name: str, config: MCPServerConfig) -> bool:
        """Start a stdio-based MCP server and discover its tools."""
        # SEC-CRITICAL-3: Validate command to prevent arbitrary command injection.
        # Only allow well-known runtimes. Block shell metacharacters in args.
        command_base = os.path.basename(config.command)
        if command_base not in self.ALLOWED_MCP_COMMANDS:
            logger.error(
                f"MCP server '{name}' uses disallowed command: {config.command}. "
                f"Allowed: {self.ALLOWED_MCP_COMMANDS}"
            )
            return False

        # Validate args don't contain shell metacharacters
        shell_metachar = set(';|&`$\n\r')
        for arg in config.args:
            if any(ch in arg for ch in shell_metachar):
                logger.error(
                    f"MCP server '{name}' has suspicious arg with shell metacharacter: "
                    f"{repr(arg[:50])}"
                )
                return False

        env = os.environ.copy()
        env.update(config.env)

        try:
            proc = subprocess.Popen(
                [config.command] + config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            self._processes[name] = proc

            # Send initialize request
            init_response = await self._send_request(name, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "cowork-agent", "version": "0.1.0"},
            })

            if not init_response:
                logger.warning(f"MCP server '{name}' did not respond to initialize")
                return False

            # Send initialized notification
            await self._send_notification(name, "notifications/initialized", {})

            # Discover tools
            tools_response = await self._send_request(name, "tools/list", {})
            if tools_response and "tools" in tools_response:
                for tool_def in tools_response["tools"]:
                    qualified_name = f"mcp__{name}__{tool_def['name']}"
                    self._tools[qualified_name] = MCPTool(
                        name=qualified_name,
                        description=tool_def.get("description", ""),
                        input_schema=tool_def.get("inputSchema", {}),
                        server_name=name,
                    )
                logger.info(f"Discovered {len(tools_response['tools'])} tools from '{name}'")

            return True

        except FileNotFoundError:
            logger.error(f"MCP server command not found: {config.command}")
            return False
        except Exception as e:
            logger.error(f"Failed to start MCP server '{name}': {e}")
            return False

    async def _start_sse_server(self, name: str, config: MCPServerConfig) -> bool:
        """Connect to an SSE-based MCP server (placeholder — needs httpx-sse)."""
        logger.info(f"SSE transport for '{name}' at {config.url} — not yet implemented")
        return False

    async def _send_request(self, server_name: str, method: str, params: dict) -> Optional[dict]:
        """Send a JSON-RPC request to an MCP server via stdio."""
        proc = self._processes.get(server_name)
        if not proc or not proc.stdin or not proc.stdout:
            return None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        try:
            request_bytes = (json.dumps(request) + "\n").encode()
            proc.stdin.write(request_bytes)
            proc.stdin.flush()

            # Read response (with timeout)
            response_line = await asyncio.wait_for(
                asyncio.to_thread(proc.stdout.readline),
                timeout=30,
            )

            if response_line:
                response = json.loads(response_line.decode())
                if "error" in response:
                    logger.warning(f"MCP error from '{server_name}': {response['error']}")
                    return None
                return response.get("result", {})

        except asyncio.TimeoutError:
            logger.warning(f"MCP request to '{server_name}' timed out: {method}")
        except Exception as e:
            logger.warning(f"MCP request error: {e}")

        return None

    async def _send_notification(self, server_name: str, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        proc = self._processes.get(server_name)
        if not proc or not proc.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            notification_bytes = (json.dumps(notification) + "\n").encode()
            proc.stdin.write(notification_bytes)
            proc.stdin.flush()
        except Exception as e:
            logger.warning(f"MCP notification error: {e}")

    async def call_tool(self, qualified_name: str, arguments: dict) -> dict:
        """
        Call a tool on its MCP server.

        Args:
            qualified_name: Tool name in format "mcp__{server}__{tool}"
            arguments: Tool input arguments

        Returns:
            Tool result dict with "content" field
        """
        tool = self._tools.get(qualified_name)
        if not tool:
            return {"error": f"Unknown MCP tool: {qualified_name}"}

        # Extract original tool name (strip the mcp__{server}__ prefix)
        parts = qualified_name.split("__", 2)
        original_name = parts[2] if len(parts) > 2 else qualified_name

        result = await self._send_request(tool.server_name, "tools/call", {
            "name": original_name,
            "arguments": arguments,
        })

        if result is None:
            return {"error": f"MCP tool call failed: {qualified_name}"}

        return result

    def get_tools(self) -> list[MCPTool]:
        """Return all discovered MCP tools."""
        return list(self._tools.values())

    def get_tool_schemas(self) -> list[dict]:
        """Return tool schemas in a format compatible with ToolSchema."""
        return [
            {
                "name": tool.name,
                "description": f"[MCP:{tool.server_name}] {tool.description}",
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]
