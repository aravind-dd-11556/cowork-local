"""
Agent Loop — The core orchestrator.
Receives user messages, calls the LLM, executes tool calls, loops until done.
"""

from __future__ import annotations
import asyncio
import logging
from typing import AsyncIterator, Callable, Optional

from .models import Message, ToolCall, ToolResult, AgentResponse
from .providers.base import BaseLLMProvider
from .tool_registry import ToolRegistry
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class Agent:
    """
    Main agent loop.

    Flow:
      user message → add to memory → build system prompt → call LLM
      → if tool_calls: execute all → add results to memory → loop
      → if no tool_calls: return final text to user
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        registry: ToolRegistry,
        prompt_builder: PromptBuilder,
        max_iterations: int = 15,
        on_tool_start: Optional[Callable] = None,
        on_tool_end: Optional[Callable] = None,
        on_status: Optional[Callable] = None,
    ):
        self.provider = provider
        self.registry = registry
        self.prompt_builder = prompt_builder
        self.max_iterations = max_iterations

        # Callbacks for UI updates
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_status = on_status  # Called with (message: str) for status updates

        # Conversation memory
        self._messages: list[Message] = []
        self._iteration = 0

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def clear_history(self) -> None:
        """Reset conversation memory."""
        self._messages.clear()
        self._iteration = 0

    # Maximum retries for recovery handlers before giving up
    MAX_TRUNCATION_RETRIES = 2
    MAX_INTENT_NUDGES = 2

    async def run(self, user_input: str) -> str:
        """
        Process a user message and return the final assistant text.
        Runs the full tool-use loop until the LLM produces a text response
        or max iterations are reached.
        """
        # Add user message to memory
        self._messages.append(
            Message(role="user", content=user_input)
        )

        final_text = ""
        self._iteration = 0
        truncation_retries = 0
        intent_nudges = 0

        while self._iteration < self.max_iterations:
            self._iteration += 1
            logger.debug(f"Agent iteration {self._iteration}")

            # Build system prompt with current context
            context = self._build_context()
            system_prompt = self.prompt_builder.build(
                tools=self.registry.get_schemas(),
                context=context,
            )

            # Call the LLM
            try:
                response = await self.provider.send_message(
                    messages=self._messages,
                    tools=self.registry.get_schemas(),
                    system_prompt=system_prompt,
                )
            except Exception as e:
                logger.error(f"LLM error: {e}")
                error_text = f"Error communicating with LLM: {str(e)}"
                self._messages.append(
                    Message(role="assistant", content=error_text)
                )
                return error_text

            # If the LLM returned text, accumulate it
            if response.text:
                final_text += response.text

            # ── Handle truncated response (hit token limit mid-tool-call) ──
            if response.stop_reason == "max_tokens" and not response.tool_calls:
                truncation_retries += 1
                if truncation_retries <= self.MAX_TRUNCATION_RETRIES:
                    logger.warning(
                        f"Response truncated mid-tool-call, retry {truncation_retries}/{self.MAX_TRUNCATION_RETRIES}"
                    )
                    if self.on_status:
                        self.on_status(
                            f"Response was truncated (token limit). "
                            f"Retrying with shorter content ({truncation_retries}/{self.MAX_TRUNCATION_RETRIES})..."
                        )
                    self._messages.append(
                        Message(role="assistant", content=response.text or "")
                    )
                    self._messages.append(
                        Message(
                            role="user",
                            content=(
                                "Your previous response was cut off before the tool call "
                                "could be completed. Please try again — this time, call the "
                                "tool directly with MUCH shorter content (a brief summary, "
                                "not a full article). Do NOT repeat your explanation, just "
                                "output the tool_calls JSON block immediately."
                            ),
                        )
                    )
                    final_text = ""
                    continue
                else:
                    # Exhausted retries — return what we have with an explanation
                    logger.error("Truncation retries exhausted, returning partial response")
                    fallback = (
                        "\n\n[The content was too long to fit in a single tool call. "
                        "Try asking for a shorter piece of content, or break the request "
                        "into smaller parts.]"
                    )
                    final_text = (response.text or "") + fallback
                    self._messages.append(
                        Message(role="assistant", content=final_text)
                    )
                    return final_text

            # If no tool calls, check for "intent without action"
            if not response.tool_calls:
                if self._looks_like_unfulfilled_intent(final_text):
                    intent_nudges += 1
                    if intent_nudges <= self.MAX_INTENT_NUDGES:
                        logger.info(
                            f"Detected tool intent without action, nudge {intent_nudges}/{self.MAX_INTENT_NUDGES}"
                        )
                        if self.on_status:
                            self.on_status(
                                f"LLM described intent but didn't execute. "
                                f"Nudging to call the tool ({intent_nudges}/{self.MAX_INTENT_NUDGES})..."
                            )
                        self._messages.append(
                            Message(role="assistant", content=final_text)
                        )
                        self._messages.append(
                            Message(
                                role="user",
                                content=(
                                    "You said you would use a tool but didn't include a "
                                    "tool_calls JSON block. Please proceed with the actual "
                                    "tool call now. Output ONLY the ```json {\"tool_calls\": "
                                    "[...]} ``` block — no other text."
                                ),
                            )
                        )
                        final_text = ""
                        continue
                    else:
                        # Exhausted nudges — return text as-is
                        logger.warning("Intent nudge retries exhausted, returning text response")
                        self._messages.append(
                            Message(role="assistant", content=final_text)
                        )
                        return final_text

                # Genuinely done — return final text
                if final_text:
                    self._messages.append(
                        Message(role="assistant", content=final_text)
                    )
                return final_text

            # ── Tool calls found — reset recovery counters ──
            truncation_retries = 0
            intent_nudges = 0

            # Add assistant message with tool calls to memory
            self._messages.append(
                Message(
                    role="assistant",
                    content=response.text or "",
                    tool_calls=response.tool_calls,
                )
            )

            # Execute all tool calls in parallel
            results = await self._execute_tools(response.tool_calls)

            # Add tool results to memory
            self._messages.append(
                Message(
                    role="tool_result",
                    content="",
                    tool_results=results,
                )
            )

            # Reset final_text for next iteration (LLM will produce new text)
            final_text = ""

        # Max iterations reached
        timeout_msg = (
            f"[Agent reached maximum iterations ({self.max_iterations}). "
            "Stopping to prevent infinite loops.]"
        )
        if final_text:
            final_text += "\n\n" + timeout_msg
        else:
            final_text = timeout_msg

        self._messages.append(
            Message(role="assistant", content=final_text)
        )
        return final_text

    async def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls in parallel and return results."""
        # Notify UI of each tool start
        for call in tool_calls:
            if self.on_tool_start:
                self.on_tool_start(call)

        # Execute all tools in parallel
        results = await self.registry.execute_parallel(tool_calls)

        # Notify UI of each tool end
        for call, result in zip(tool_calls, results):
            if self.on_tool_end:
                self.on_tool_end(call, result)

        return results

    @staticmethod
    def _looks_like_unfulfilled_intent(text: str) -> bool:
        """
        Detect if the LLM said it would use a tool but didn't actually call one.

        Common patterns from local LLMs:
        - "I'll create the file now..."
        - "Let me write this blog post..."
        - "Now I'll use the write tool to..."
        """
        if not text:
            return False

        text_lower = text.lower()
        intent_phrases = [
            "i'll create", "i will create",
            "let me create", "let me write",
            "i'll write", "i will write",
            "now i'll", "now i will",
            "let me save", "i'll save",
            "i'll use the write", "i'll use the bash",
            "let me use the", "i'll call the",
            "here's the file", "here is the file",
            "creating the file", "writing the file",
        ]

        # Only trigger if one of these intent phrases is near the end of the text
        # (indicating the LLM was about to act but stopped)
        last_200 = text_lower[-200:]
        return any(phrase in last_200 for phrase in intent_phrases)

    def _build_context(self) -> dict:
        """Build runtime context for the prompt builder."""
        ctx = {
            "iteration": self._iteration,
        }

        # Get todos from the todo tool if available (safe — won't crash if missing)
        try:
            todo_tool = self.registry.get_tool("todo_write")
            if hasattr(todo_tool, "get_context"):
                ctx["todos"] = todo_tool.get_context()
        except KeyError:
            pass

        return ctx
