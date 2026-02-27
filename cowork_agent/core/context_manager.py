"""
Context Manager — Prevents context window overflow.

Tracks approximate token usage and prunes conversation history when
approaching the model's context limit.

Sprint 11: Importance-weighted pruning — messages are scored by recency,
role, content value, and tool result quality.  Dropped messages are
summarized (via ConversationSummarizer) instead of silently lost.
"""

from __future__ import annotations
import logging
import math
import re
from typing import Optional, TYPE_CHECKING

from .models import Message

if TYPE_CHECKING:
    from .conversation_summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages conversation history to prevent context window overflow.

    Strategy:
      1. Estimate token count for each message (~4 chars per token)
      2. When total exceeds threshold, prune old messages:
         Phase 1: Truncate old tool_result outputs
         Phase 2: Truncate old assistant messages
         Phase 3: Score remaining messages by importance, summarize
                  and drop lowest-scored ones first
    """

    # Rough chars-per-token estimate (conservative for English text)
    CHARS_PER_TOKEN = 4

    # How much of the context window to keep free for LLM response
    RESPONSE_RESERVE_RATIO = 0.25  # Reserve 25% for response

    # Minimum recent messages to always keep
    MIN_RECENT_MESSAGES = 6  # Keep at least the last 3 turns (user+assistant pairs)

    # ── Importance scoring weights ───────────────────────────
    W_RECENCY = 0.30
    W_ROLE = 0.25
    W_CONTENT = 0.20
    W_TOOL = 0.15
    W_LENGTH_PENALTY = 0.10

    # Decision indicator phrases (mirrors ConversationSummarizer)
    DECISION_PHRASES = [
        "decided", "agreed", "confirmed", "chose", "resolved",
        "will use", "going with", "settled on", "picked",
        "the plan is", "approach is", "strategy is",
    ]

    def __init__(
        self,
        max_context_tokens: int = 32000,
        summarizer: Optional[ConversationSummarizer] = None,
    ):
        """
        Args:
            max_context_tokens: Model's total context window size in tokens.
            summarizer: Optional ConversationSummarizer for summarizing
                        dropped messages.
        """
        self.max_context_tokens = max_context_tokens
        self._effective_limit = int(
            max_context_tokens * (1 - self.RESPONSE_RESERVE_RATIO)
        )
        self._summarizer = summarizer

    # ── Token estimation ─────────────────────────────────────

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

    # ── Main prune pipeline ──────────────────────────────────

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

        # Phase 3: Score and drop least-important messages (with summarization)
        result = self._drop_lowest_scored_messages(result, target, system_prompt)

        final_tokens = self.estimate_total_tokens(result, system_prompt)
        logger.info(
            f"After pruning: {final_tokens} tokens, "
            f"{len(result)} messages (was {len(messages)})"
        )

        return result

    # ── Phase 1: Truncate tool results ───────────────────────

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

    # ── Phase 2: Truncate assistant messages ─────────────────

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

    # ── Phase 3: Importance-weighted drop ────────────────────

    def _drop_lowest_scored_messages(
        self, messages: list[Message], target: int, system_prompt: str
    ) -> list[Message]:
        """Drop least-important messages, summarizing what's lost."""
        result = list(messages)
        total = len(result)

        # Score every message outside the protected recent window
        protected = self.MIN_RECENT_MESSAGES
        scorable_count = max(0, total - protected)

        if scorable_count == 0:
            return result

        # Build (index, score) pairs for scorable messages
        scored: list[tuple[int, float]] = []
        for i in range(scorable_count):
            score = self._score_message(result[i], position=i, total=total)
            # Store the score on the message itself for debugging
            result[i].importance_score = score
            scored.append((i, score))

        # Sort by score ascending (lowest first = drop candidates)
        scored.sort(key=lambda x: x[1])

        # Collect messages to drop
        dropped_messages: list[Message] = []
        drop_indices: set[int] = set()

        for idx, _score in scored:
            if self.estimate_total_tokens(result, system_prompt) <= target:
                break
            if len(result) - len(drop_indices) <= protected:
                break
            drop_indices.add(idx)
            dropped_messages.append(result[idx])

        if not drop_indices:
            return result

        # Summarize dropped messages
        summary_text = self._summarize_dropped(dropped_messages, len(messages))

        # Remove dropped messages (iterate in reverse to preserve indices)
        kept = [msg for i, msg in enumerate(result) if i not in drop_indices]

        logger.debug(
            f"Importance pruning: dropped {len(drop_indices)} messages "
            f"(scores: {[f'{s:.2f}' for _, s in scored[:len(drop_indices)]]})"
        )

        # Insert summary notice at the top
        notice = Message(
            role="user",
            content=summary_text,
        )
        kept.insert(0, notice)

        return kept

    # ── Importance scoring ───────────────────────────────────

    def _score_message(self, msg: Message, position: int, total: int) -> float:
        """
        Score a message 0.0–1.0 for importance.

        Higher scores = more important = kept longer.

        Factors:
          - Recency (30%): exponential decay from newest to oldest
          - Role weight (25%): user > assistant > tool_result
          - Content value (20%): decisions/errors rate higher
          - Tool result value (15%): failed tools rate higher
          - Length penalty (-10%): very long messages penalized
        """
        score = (
            self.W_RECENCY * self._recency_score(position, total)
            + self.W_ROLE * self._role_weight(msg.role)
            + self.W_CONTENT * self._content_value(msg)
            + self.W_TOOL * self._tool_result_value(msg)
            - self.W_LENGTH_PENALTY * self._length_penalty(msg)
        )
        return max(0.0, min(1.0, score))

    @staticmethod
    def _recency_score(position: int, total: int) -> float:
        """Exponential decay: newest=1.0, oldest→0.0."""
        if total <= 1:
            return 1.0
        # Normalize position to 0..1 range (0 = oldest, 1 = newest)
        normalized = position / (total - 1)
        # Exponential curve so recent messages are strongly favored
        return math.pow(normalized, 1.5)

    @staticmethod
    def _role_weight(role: str) -> float:
        """User messages are most important, tool results least."""
        weights = {
            "user": 1.0,
            "assistant": 0.8,
            "tool_result": 0.6,
        }
        return weights.get(role, 0.5)

    def _content_value(self, msg: Message) -> float:
        """Score based on content significance."""
        if not msg.content:
            return 0.0

        lower = msg.content.lower()
        score = 0.0

        # Decisions are highly valuable
        if any(phrase in lower for phrase in self.DECISION_PHRASES):
            score += 0.3

        # Error mentions are valuable context
        if any(word in lower for word in ("error", "exception", "traceback", "failed", "bug")):
            score += 0.4

        # Questions from the user are valuable
        if msg.role == "user" and "?" in msg.content:
            score += 0.2

        # Very short messages (like "ok", "yes") are less valuable
        if len(msg.content) < 20:
            score -= 0.1

        return max(0.0, min(1.0, score))

    @staticmethod
    def _tool_result_value(msg: Message) -> float:
        """Failed tool results are more valuable to keep (learn from mistakes)."""
        if not msg.tool_results:
            return 0.0

        has_failure = any(not tr.success for tr in msg.tool_results)
        has_success = any(tr.success for tr in msg.tool_results)

        if has_failure:
            return 0.4  # Failures are informative
        if has_success:
            return 0.2  # Successes are routine
        return 0.0

    def _length_penalty(self, msg: Message) -> float:
        """Penalize very long messages (they consume disproportionate space)."""
        tokens = self.estimate_message_tokens(msg)
        if tokens < 200:
            return 0.0
        if tokens < 500:
            return 0.2
        if tokens < 1000:
            return 0.4
        return 0.6  # Very long messages get a high penalty

    # ── Summarization helper ─────────────────────────────────

    def _summarize_dropped(self, dropped: list[Message], original_count: int) -> str:
        """Create a summary notice for dropped messages."""
        if self._summarizer and dropped:
            summary = self._summarizer.summarize(dropped)
            return (
                f"[MEMORY SUMMARY — {len(dropped)} of {original_count} "
                f"messages summarized]\n{summary}"
            )
        else:
            # Fallback: basic notice without summarizer
            return (
                f"[System: {len(dropped)} earlier messages were pruned "
                f"to fit the context window. Continue the conversation "
                f"based on the remaining context.]"
            )
