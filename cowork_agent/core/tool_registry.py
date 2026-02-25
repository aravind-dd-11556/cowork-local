"""
Tool Registry — central registry for all available tools.
Handles registration, schema retrieval, and parallel execution.
"""

from __future__ import annotations
import asyncio
from typing import Optional

from .models import ToolCall, ToolResult, ToolSchema


class ToolRegistry:
    """Central registry for all agent tools."""

    def __init__(self):
        self._tools: dict = {}  # name -> BaseTool instance
        self._read_files: set[str] = set()  # Track files read (for Edit tool)

    def register(self, tool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str):
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def get_schemas(self) -> list[ToolSchema]:
        """Return all tool schemas for system prompt / LLM."""
        return [tool.get_schema() for tool in self._tools.values()]

    def list_tools(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    @property
    def tool_names(self) -> list[str]:
        """Property alias for list_tools()."""
        return self.list_tools()

    async def execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        try:
            tool = self.get_tool(call.name)
            result = await tool.execute(tool_id=call.tool_id, **call.input)
            # Track file reads for Edit tool validation
            if call.name == "read" and result.success:
                self._read_files.add(call.input.get("file_path", ""))
            return result
        except KeyError as e:
            return ToolResult(
                tool_id=call.tool_id,
                success=False,
                output="",
                error=str(e),
            )
        except Exception as e:
            return ToolResult(
                tool_id=call.tool_id,
                success=False,
                output="",
                error=f"Tool execution error: {str(e)}",
            )

    async def execute_parallel(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls in parallel using asyncio.gather."""
        tasks = [self.execute_tool(call) for call in calls]
        return await asyncio.gather(*tasks)

    def has_been_read(self, file_path: str) -> bool:
        """Check if a file has been read in this session (for Edit validation)."""
        return file_path in self._read_files

    def mark_as_read(self, file_path: str) -> None:
        """Mark a file as read."""
        self._read_files.add(file_path)
