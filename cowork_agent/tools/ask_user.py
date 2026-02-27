"""
AskUser Tool — Allows the agent to ask the user clarifying questions mid-task.

Mirrors real Cowork's AskUserQuestion:
  - Supports structured questions with header, options (label+description+markdown)
  - Supports multiSelect for non-mutually-exclusive choices
  - Supports 1-4 questions per call
  - User can always provide custom "Other" input
  - Falls back to simple text input if no callback configured
"""

from __future__ import annotations
import asyncio
import json
from typing import Optional, Callable

from .base import BaseTool


class AskUserTool(BaseTool):
    name = "ask_user"
    description = (
        "Ask the user clarifying questions when their request is ambiguous "
        "or you need more information to proceed. Supports structured multiple-choice "
        "questions with headers, labeled options with descriptions, multiSelect, "
        "and markdown previews. Use this BEFORE doing major work to ensure alignment."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "description": "1-4 structured questions to ask the user",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The complete question to ask. Should be clear, specific, and end with ?",
                        },
                        "header": {
                            "type": "string",
                            "description": "Short label for the question (max 12 chars), e.g. 'Auth method', 'Library'",
                        },
                        "options": {
                            "type": "array",
                            "description": "2-4 available choices. An 'Other' option is always added automatically.",
                            "minItems": 2,
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "description": "Display text (1-5 words)",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Explanation of this option or its implications",
                                    },
                                    "markdown": {
                                        "type": "string",
                                        "description": "Optional preview content (code snippets, ASCII mockups) shown when focused",
                                    },
                                },
                                "required": ["label", "description"],
                            },
                        },
                        "multiSelect": {
                            "type": "boolean",
                            "description": "Allow selecting multiple options (default: false)",
                            "default": False,
                        },
                    },
                    "required": ["question", "header", "options", "multiSelect"],
                },
            },
        },
        "required": ["questions"],
    }

    def __init__(self):
        self._input_callback: Optional[Callable] = None

    def set_input_callback(self, callback: Callable) -> None:
        """
        Set the callback that will be used to get user input.

        The callback should accept:
            (question: str, options: list[dict], multi_select: bool) -> str | list[str]
        and return the user's response(s).

        For backward compatibility, it also accepts the legacy signature:
            (question: str, options: list[str]) -> str
        """
        self._input_callback = callback

    async def execute(self, questions: list[dict] | None = None,
                      # Legacy simple format support
                      question: str = "", options: list | None = None,
                      tool_id: str = "", **kwargs) -> "ToolResult":
        """Execute the ask_user tool with structured or simple questions."""

        # Handle legacy simple format: question + flat string options
        if question and not questions:
            return await self._execute_simple(question, options or [], tool_id)

        if not questions:
            return self._error("Either 'questions' or 'question' is required.", tool_id)

        if self._input_callback is None:
            return self._error(
                "No input callback configured. Cannot ask user questions.",
                tool_id,
            )

        # Process each structured question
        all_answers = {}
        try:
            for q in questions:
                q_text = q.get("question", "")
                header = q.get("header", "")
                q_options = q.get("options", [])
                multi_select = q.get("multiSelect", False)

                if not q_text:
                    continue

                # Format the question for display
                display_text = f"[{header}] {q_text}" if header else q_text

                # Try structured callback first, fall back to simple
                try:
                    response = await asyncio.to_thread(
                        self._input_callback, display_text, q_options, multi_select
                    )
                except TypeError:
                    # Legacy callback: (question, flat_options) -> str
                    flat_opts = [o.get("label", str(o)) if isinstance(o, dict) else str(o) for o in q_options]
                    response = await asyncio.to_thread(
                        self._input_callback, display_text, flat_opts
                    )

                if isinstance(response, list):
                    all_answers[q_text] = ", ".join(str(r) for r in response)
                else:
                    all_answers[q_text] = str(response) if response else "(no response)"

            # Format all answers
            if len(all_answers) == 1:
                answer_text = list(all_answers.values())[0]
                return self._success(f"User response: {answer_text}", tool_id)
            else:
                lines = []
                for q_text, answer in all_answers.items():
                    lines.append(f"Q: {q_text}\nA: {answer}")
                return self._success("User responses:\n\n" + "\n\n".join(lines), tool_id)

        except (EOFError, KeyboardInterrupt):
            return self._success("User cancelled the question.", tool_id)
        except Exception as e:
            return self._error(f"Error getting user input: {str(e)}", tool_id)

    async def _execute_simple(self, question: str, options: list, tool_id: str) -> "ToolResult":
        """Handle legacy simple question format."""
        if self._input_callback is None:
            return self._error(
                "No input callback configured. Cannot ask user questions.",
                tool_id,
            )

        try:
            response = await asyncio.to_thread(
                self._input_callback, question, options
            )

            if not response:
                return self._success("User provided no response (empty input).", tool_id)

            return self._success(f"User response: {response}", tool_id)

        except (EOFError, KeyboardInterrupt):
            return self._success("User cancelled the question.", tool_id)
        except Exception as e:
            return self._error(f"Error getting user input: {str(e)}", tool_id)
