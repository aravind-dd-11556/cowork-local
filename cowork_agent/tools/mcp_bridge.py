"""
MCP Bridge — Wraps MCP-discovered tools as agent BaseTool instances.
Each MCP tool gets a dynamically generated wrapper that calls the MCP server.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from .base import BaseTool

if TYPE_CHECKING:
    from ..core.mcp_client import MCPClient, MCPTool

logger = logging.getLogger(__name__)


class MCPBridgeTool(BaseTool):
    """
    Dynamic wrapper that bridges an MCP tool to the agent's tool system.
    One instance is created per MCP tool discovered.
    """

    def __init__(self, mcp_tool: "MCPTool", mcp_client: "MCPClient"):
        self.name = mcp_tool.name
        self.description = f"[MCP:{mcp_tool.server_name}] {mcp_tool.description}"
        self.input_schema = mcp_tool.input_schema or {
            "type": "object",
            "properties": {},
        }
        self._mcp_client = mcp_client
        self._qualified_name = mcp_tool.name

    async def execute(self, tool_id: str = "", **kwargs) -> "ToolResult":
        try:
            result = await self._mcp_client.call_tool(self._qualified_name, kwargs)

            if "error" in result:
                return self._error(str(result["error"]), tool_id)

            # Extract content from MCP result format
            content = result.get("content", [])
            if isinstance(content, list):
                # MCP returns content as list of {type, text} objects
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)
                output = "\n".join(text_parts)
            elif isinstance(content, str):
                output = content
            else:
                output = str(result)

            return self._success(output, tool_id)

        except Exception as e:
            logger.error(f"MCP bridge error for {self._qualified_name}: {e}")
            return self._error(f"MCP tool error: {str(e)}", tool_id)


def register_mcp_tools(registry, mcp_client: "MCPClient") -> int:
    """
    Register all MCP-discovered tools with the agent's tool registry.
    Returns the number of tools registered.
    """
    count = 0
    for mcp_tool in mcp_client.get_tools():
        bridge = MCPBridgeTool(mcp_tool, mcp_client)
        registry.register(bridge)
        count += 1
        logger.info(f"Registered MCP tool: {mcp_tool.name}")
    return count
