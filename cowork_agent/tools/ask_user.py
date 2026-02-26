"""
AskUser Tool — Allows the agent to ask the user clarifying questions mid-task.

This tool pauses the agent loop and waits for user input through the CLI.
It supports free-form questions as well as multiple-choice options.
"""

from __future__ import annotations
import asyncio
from typing import Optional, Callable

from .base import BaseTool


class AskUserTool(BaseTool):
    name = "ask_user"
    description = (
        "Ask the user a clarifying question when their request is ambiguous "
        "or you need more information to proceed. Supports free-text questions "
        "and optional multiple-choice options. Use this BEFORE doing major work "
        "to ensure you're building the right thing."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user. Be specific and concise.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of choices for the user. "
                    "User can always type a custom answer instead."
                ),
            },
        },
        "required": ["question"],
    }

    def __init__(self):
        self._input_callback: Optional[Callable] = None

    def set_input_callback(self, callback: Callable) -> None:
        """
        Set the callback that will be used to get user input.
        The callback should accept (question: str, options: list[str]) -> str
        and return the user's response.
        """
        self._input_callback = callback

    async def execute(self, question: str, options: list[str] | None = None,
                      tool_id: str = "", **kwargs) -> "ToolResult":
        if not question:
            return self._error("Question cannot be empty", tool_id)

        if self._input_callback is None:
            return self._error(
                "No input callback configured. Cannot ask user questions.",
                tool_id,
            )

        try:
            # Call the input callback (the CLI will handle display and input)
            response = await asyncio.to_thread(
                self._input_callback, question, options or []
            )

            if not response:
                return self._success("User provided no response (empty input).", tool_id)

            return self._success(f"User response: {response}", tool_id)

        except (EOFError, KeyboardInterrupt):
            return self._success("User cancelled the question.", tool_id)
        except Exception as e:
            return self._error(f"Error getting user input: {str(e)}", tool_id)
