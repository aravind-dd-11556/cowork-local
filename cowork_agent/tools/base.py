"""
Base tool class — all tools inherit from this.
Defines the standard interface: name, description, schema, execute().
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from ..core.models import ToolResult, ToolSchema


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    name: str = ""
    description: str = ""
    input_schema: dict = {}

    @abstractmethod
    async def execute(self, *, progress_callback: Optional[Callable[[int, str], None]] = None, **kwargs) -> ToolResult:
        """
        Execute the tool with the given parameters.
        Must return a ToolResult with success status and output.

        Args:
            progress_callback: Optional callback for reporting progress.
                Signature: (percent: int, message: str) -> None
                Use percent 0–100 or -1 for indeterminate.
            **kwargs: Tool-specific parameters.
        """
        pass

    def get_schema(self) -> ToolSchema:
        """Return the tool's schema for LLM consumption."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    def _success(self, output: str, tool_id: str = "", **metadata) -> ToolResult:
        """Helper to create a successful result."""
        return ToolResult(
            tool_id=tool_id,
            success=True,
            output=output,
            metadata=metadata,
        )

    def _error(self, error: str, tool_id: str = "") -> ToolResult:
        """Helper to create an error result."""
        return ToolResult(
            tool_id=tool_id,
            success=False,
            output="",
            error=error,
        )
