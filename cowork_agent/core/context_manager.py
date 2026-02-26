"""
Context Manager — Prevents context window overflow.

Tracks approximate token usage and prunes conversation history when
approaching the model's context limit. Keeps the most recent messages
and summarizes/drops old tool results to free space.
"""

from __future__ import annotations
import logging
from typing import Optional

from .models import Message

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages conversation history to prevent context window overflow.

    Strategy:
      1. Estimate token count for each message (~4 chars per token)
      2. When total exceeds threshold, prune old messages:
         - Always keep: system prompt, first user message, last N messages
         - Drop: old tool_result messages (largest token consumers)
         - Truncate: old assistant messages to summaries
    """

    # Rough chars-per-token estimate (conservative for English text)
    CHARS_PER_TOKEN = 4

    # How much of the context window to keep free for LLM response
    RESPONSE_RESERVE_RATIO = 0.25  # Reserve 25% for response

    # Minimum recent messages to always keep
    MIN_RECENT_MESSAGES = 6  # Keep at least the last 3 turns (user+assistant pairs)

    def __init__(self, max_context_tokens: int = 32000):
        """
        Args:
            max_context_tokens: Model's total context window size in tokens.
        """
        self.max_context_tokens = max_context_tokens
        self._effective_limit = int(
            max_context_tokens * (1 - self.RESPONSE_RESERVE_RATIO)
        )

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimate."""
        if not text:
            return 0
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def estimate_message_tokens(self, msg: Message) -> int:
        """Estimate tokens for a single message including tool data."""
        tokens = self.estimate_tokens(msg.content)

        # Add tool call tokens
        if msg.tool_calls:
            for call in msg.tool_calls:
                tokens += self.estimate_tokens(call.name)
                tokens += self.estimate_tokens(str(call.input))

        # Add tool result tokens
        if msg.tool_results:
            for result in msg.tool_results:
                tokens += self.estimate_tokens(result.output)
                if result.error:
                    tokens += self.estimate_tokens(result.error)

        # Message overhead (role, formatting)
        tokens += 4

        return tokens

    def estimate_total_tokens(self, messages: list[Message], system_prompt: str = "") -> int:
        """Estimate total token count for all messages + system prompt."""
        total = self.estimate_tokens(system_prompt)
        for msg in messages:
            total += self.estimate_message_tokens(msg)
        return total

    def needs_pruning(self, messages: list[Message], system_prompt: str = "") -> bool:
        """Check if messages exceed the effective context limit."""
        total = self.estimate_total_tokens(messages, system_prompt)
        return total > self._effective_limit

    def prune(self, messages: list[Message], system_prompt: str = "") -> list[Message]:
        """
        Prune conversation history to fit within context limits.

        Returns a new list of messages (does not modify the original).
        """
        total_tokens = self.estimate_total_tokens(messages, system_prompt)

        if total_tokens <= self._effective_limit:
            return list(messages)  # No pruning needed

        logger.info(
            f"Context pruning triggered: {total_tokens} tokens "
            f"exceeds limit of {self._effective_limit}"
        )

        result = list(messages)
        target = self._effective_limit

        # Phase 1: Truncate old tool_result messages (biggest token hogs)
        result = self._truncate_old_tool_results(result, target, system_prompt)
        if self.estimate_total_tokens(result, system_prompt) <= target:
            return result

        # Phase 2: Truncate old assistant messages
        result = self._truncate_old_assistant_messages(result, target, system_prompt)
        if self.estimate_total_tokens(result, system_prompt) <= target:
            return result

        # Phase 3: Drop oldest messages entirely (keep recent ones)
        result = self._drop_oldest_messages(result, target, system_prompt)

        final_tokens = self.estimate_total_tokens(result, system_prompt)
        logger.info(
            f"After pruning: {final_tokens} tokens, "
            f"{len(result)} messages (was {len(messages)})"
        )

        return result

    def _truncate_old_tool_results(
        self, messages: list[Message], target: int, system_prompt: str
    ) -> list[Message]:
        """Truncate tool result outputs in older messages."""
        result = list(messages)
        # Process from oldest to newest, skip the last MIN_RECENT_MESSAGES
        cutoff = max(0, len(result) - self.MIN_RECENT_MESSAGES)

        for i in range(cutoff):
            if self.estimate_total_tokens(result, system_prompt) <= target:
                break

            msg = result[i]
            if msg.role == "tool_result" and msg.tool_results:
                truncated_results = []
                for tr in msg.tool_results:
                    if len(tr.output) > 500:
                        # Create a truncated copy
                        from copy import copy
                        new_tr = copy(tr)
                        new_tr.output = (
                            tr.output[:200]
                            + f"\n... [truncated from {len(tr.output)} chars] ..."
                            + tr.output[-100:]
                        )
                        truncated_results.append(new_tr)
                    else:
                        truncated_results.append(tr)

                result[i] = Message(
                    role=msg.role,
                    content=msg.content,
                    tool_calls=msg.tool_calls,
                    tool_results=truncated_results,
                    timestamp=msg.timestamp,
                )

        return result

    def _truncate_old_assistant_messages(
        self, messages: list[Message], target: int, system_prompt: str
    ) -> list[Message]:
        """Truncate old assistant messages to summaries."""
        result = list(messages)
        cutoff = max(0, len(result) - self.MIN_RECENT_MESSAGES)

        for i in range(cutoff):
            if self.estimate_total_tokens(result, system_prompt) <= target:
                break

            msg = result[i]
            if msg.role == "assistant" and len(msg.content) > 500:
                result[i] = Message(
                    role=msg.role,
                    content=(
                        msg.content[:200]
                        + f"\n... [truncated from {len(msg.content)} chars]"
                    ),
                    tool_calls=msg.tool_calls,
                    tool_results=msg.tool_results,
                    timestamp=msg.timestamp,
                )

        return result

    def _drop_oldest_messages(
        self, messages: list[Message], target: int, system_prompt: str
    ) -> list[Message]:
        """Drop oldest messages, keeping at least MIN_RECENT_MESSAGES."""
        result = list(messages)

        while (
            len(result) > self.MIN_RECENT_MESSAGES
            and self.estimate_total_tokens(result, system_prompt) > target
        ):
            # Remove the oldest message
            dropped = result.pop(0)
            logger.debug(
                f"Dropped oldest message: role={dropped.role}, "
                f"chars={len(dropped.content)}"
            )

        # Insert a context notice if we dropped messages
        if len(result) < len(messages):
            dropped_count = len(messages) - len(result)
            notice = Message(
                role="user",
                content=(
                    f"[System: {dropped_count} earlier messages were pruned "
                    f"to fit the context window. Continue the conversation "
                    f"based on the remaining context.]"
                ),
            )
            result.insert(0, notice)

        return result
