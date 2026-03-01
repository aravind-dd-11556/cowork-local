"""
Generate Tool — Agent-callable tool for creating dynamic tools at runtime.

Allows the agent to define new tools by providing a name, description,
input schema, and Python code. The code is validated for safety before
being registered.

Sprint 27: Tier 2 Differentiating Feature 3.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional, TYPE_CHECKING

from .base import BaseTool
from ..core.models import ToolResult

if TYPE_CHECKING:
    from ..core.tool_generator import ToolGenerator

logger = logging.getLogger(__name__)


class GenerateToolTool(BaseTool):
    """
    Agent-facing tool for dynamically creating new tools.

    Input:
        name: Tool name (alphanumeric + underscores)
        description: Human-readable description
        parameters: JSON Schema string for input parameters
        python_code: Python code defining a `run(**kwargs)` function

    Example agent usage:
        "Create a tool called 'word_count' that counts words in text"
        → agent calls generate_tool with appropriate code
    """

    name = "generate_tool"
    description = (
        "Create a new tool at runtime. Provide a name, description, "
        "JSON schema for parameters, and Python code defining a "
        "run(**kwargs) function. The code is validated for safety."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Tool name (alphanumeric, underscores, hyphens)",
            },
            "description": {
                "type": "string",
                "description": "Human-readable description of what the tool does",
            },
            "parameters": {
                "type": "string",
                "description": "JSON Schema string defining tool input parameters",
            },
            "python_code": {
                "type": "string",
                "description": (
                    "Python code that defines a run(**kwargs) function. "
                    "The function receives tool input as kwargs and must "
                    "return a string result."
                ),
            },
        },
        "required": ["name", "description", "parameters", "python_code"],
    }

    def __init__(self, tool_generator: Optional[ToolGenerator] = None):
        self._generator = tool_generator

    async def execute(
        self,
        *,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        name: str = "",
        description: str = "",
        parameters: str = "",
        python_code: str = "",
        **kwargs,
    ) -> ToolResult:
        tool_id = kwargs.get("tool_id", "")

        if not self._generator:
            return self._error(
                "Dynamic tool generation is not enabled. "
                "Set dynamic_tools.enabled=true in config.",
                tool_id,
            )

        if not name or not description or not python_code:
            return self._error(
                "Missing required parameters: name, description, and python_code "
                "are all required.",
                tool_id,
            )

        # Parse parameters schema
        try:
            if parameters:
                input_schema = json.loads(parameters)
            else:
                input_schema = {
                    "type": "object",
                    "properties": {},
                }
        except json.JSONDecodeError as e:
            return self._error(
                f"Invalid JSON in parameters schema: {e}",
                tool_id,
            )

        if progress_callback:
            progress_callback(30, "Validating code safety...")

        # Generate the tool
        try:
            tool = self._generator.generate_tool(
                name=name,
                description=description,
                input_schema=input_schema,
                python_code=python_code,
            )

            if progress_callback:
                progress_callback(100, "Tool created successfully")

            return self._success(
                f"Tool '{name}' created successfully.\n"
                f"Description: {tool.description}\n"
                f"You can now use it by calling the '{name}' tool.",
                tool_id,
            )

        except ValueError as e:
            return self._error(str(e), tool_id)
        except Exception as e:
            logger.error(f"Failed to generate tool '{name}': {e}")
            return self._error(
                f"Failed to generate tool: {e}",
                tool_id,
            )
