"""
Context Manager — Prevents context window overflow.

Tracks approximate token usage and prunes conversation history when
approaching the model's context limit.

Sprint 11: Importance-weighted pruning — messages are scored by recency,
role, content value, and tool result quality.  Dropped messages are
summarized (via ConversationSummarizer) instead of silently lost.

Sprint 15: Model-aware token estimation, knowledge relevance scoring,
message deduplication, and proactive pruning at 60% capacity.

Sprint 27: ContextPriorityScorer with 8 scoring signals for smarter
context assembly based on tool usage, knowledge alignment, conversation
flow, error context, and user intent.
"""

from __future__ import annotations
import hashlib
import logging
import math
import re
import time
from typing import Optional, TYPE_CHECKING

from .models import Message

if TYPE_CHECKING:
    from .conversation_summarizer import ConversationSummarizer
    from .token_estimator import ModelTokenEstimator
    from .knowledge_store import KnowledgeEntry, KnowledgeStore
    from .metrics_collector import MetricsCollector

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

    # Sprint 15: Proactive pruning and deduplication
    PROACTIVE_PRUNE_RATIO = 0.60  # Start proactive pruning at 60% capacity
    DEDUP_WINDOW = 10  # Check last N messages for duplicates

    def __init__(
        self,
        max_context_tokens: int = 32000,
        summarizer: Optional[ConversationSummarizer] = None,
        token_estimator: Optional[ModelTokenEstimator] = None,
        model: str = "",
        metrics_collector: Optional[MetricsCollector] = None,
        knowledge_store: Optional[KnowledgeStore] = None,
    ):
        """
        Args:
            max_context_tokens: Model's total context window size in tokens.
            summarizer: Optional ConversationSummarizer for summarizing
                        dropped messages.
            token_estimator: Optional ModelTokenEstimator for model-aware
                             token estimation (Sprint 15).
            model: Model name for token estimation ratio selection.
            metrics_collector: Optional MetricsCollector for tool-aware
                               scoring (Sprint 27).
            knowledge_store: Optional KnowledgeStore for knowledge-aligned
                             scoring (Sprint 27).
        """
        self.max_context_tokens = max_context_tokens
        self._effective_limit = int(
            max_context_tokens * (1 - self.RESPONSE_RESERVE_RATIO)
        )
        self._summarizer = summarizer

        # Sprint 15: Model-aware token estimation
        self.token_estimator: Optional[ModelTokenEstimator] = token_estimator
        self.model = model

        # Sprint 27: Smart context priority scoring
        self._priority_scorer: Optional[ContextPriorityScorer] = None
        if metrics_collector or knowledge_store:
            self._priority_scorer = ContextPriorityScorer(
                metrics_collector=metrics_collector,
                knowledge_store=knowledge_store,
            )

    # ── Token estimation ─────────────────────────────────────

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (delegates to ModelTokenEstimator if available)."""
        if not text:
            return 0
        if self.token_estimator:
            return self.token_estimator.estimate_tokens(text, self.model)
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def estimate_message_tokens(self, msg: Message) -> int:
        """Estimate tokens for a single message including tool data."""
        if self.token_estimator:
            return self.token_estimator.estimate_message_tokens(msg, self.model)

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
        if self.token_estimator:
            return self.token_estimator.estimate_messages_tokens(
                messages, self.model, system_prompt
            )
        total = self.estimate_tokens(system_prompt)
        for msg in messages:
            total += self.estimate_message_tokens(msg)
        return total

    def needs_pruning(self, messages: list[Message], system_prompt: str = "") -> bool:
        """Check if messages exceed the effective context limit."""
        total = self.estimate_total_tokens(messages, system_prompt)
        return total > self._effective_limit

    def should_prune_proactively(self, messages: list[Message], system_prompt: str = "") -> bool:
        """
        Check if we're approaching capacity and should start proactive pruning.

        Sprint 15: Returns True at 60% of effective limit (earlier than the
        standard 75% threshold used by needs_pruning).
        """
        total = self.estimate_total_tokens(messages, system_prompt)
        threshold = int(self._effective_limit * self.PROACTIVE_PRUNE_RATIO)
        return total > threshold

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

        Sprint 27: Delegates to ContextPriorityScorer (8 signals) if available,
        otherwise uses the original 5-signal scoring.

        Factors (original):
          - Recency (30%): exponential decay from newest to oldest
          - Role weight (25%): user > assistant > tool_result
          - Content value (20%): decisions/errors rate higher
          - Tool result value (15%): failed tools rate higher
          - Length penalty (-10%): very long messages penalized
        """
        # Sprint 27: Use priority scorer if available
        if self._priority_scorer:
            return self._priority_scorer.score_message(
                msg, position, total
            )

        # Original scoring (fallback)
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

    # ── Sprint 15: Knowledge relevance scoring ─────────────

    def score_knowledge_entry(
        self,
        entry: KnowledgeEntry,
        recent_user_message: str,
    ) -> float:
        """
        Score a knowledge entry for relevance to the current user message.

        Returns a float 0.0–1.0 combining keyword match, category boost,
        and recency.

        Args:
            entry: The KnowledgeEntry to score.
            recent_user_message: The latest user message for relevance matching.
        """
        if not recent_user_message:
            return 0.5  # Neutral if no context

        score = 0.0
        lower_msg = recent_user_message.lower()

        # Keyword matching: check if entry key words appear in user message
        key_words = [w for w in entry.key.lower().split() if len(w) > 2]
        if key_words:
            matches = sum(1 for w in key_words if w in lower_msg)
            score += 0.4 * (matches / len(key_words))

        # Value keyword matching (lighter weight)
        value_words = [w for w in entry.value.lower().split()[:10] if len(w) > 3]
        if value_words:
            val_matches = sum(1 for w in value_words if w in lower_msg)
            score += 0.1 * (val_matches / len(value_words))

        # Category boost: decisions are most valuable for context
        category_boosts = {"decisions": 0.3, "preferences": 0.2, "facts": 0.1}
        score += category_boosts.get(entry.category, 0.1)

        # Recency bonus: decays over 72 hours
        age_hours = (time.time() - entry.updated_at) / 3600
        recency = max(0.0, 1.0 - age_hours / 72)
        score += recency * 0.2

        return min(1.0, max(0.0, score))

    # ── Sprint 15: Message deduplication ────────────────────

    @staticmethod
    def _message_hash(msg: Message) -> str:
        """
        Create a short hash of a message for deduplication.

        Hashes the role, tool call signatures, and first 200 chars of content.
        """
        tool_sig = ""
        if msg.tool_calls:
            tool_sig = "|".join(
                f"{tc.name}:{sorted(tc.input.keys()) if isinstance(tc.input, dict) else ''}"
                for tc in msg.tool_calls
            )
        content_sig = msg.content[:200] if msg.content else ""
        combined = f"{msg.role}|{tool_sig}|{content_sig}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def deduplicate_messages(self, messages: list[Message]) -> list[Message]:
        """
        Remove duplicate messages from the conversation history.

        Only checks within a sliding window of DEDUP_WINDOW messages.
        Always preserves the last MIN_RECENT_MESSAGES messages.

        Returns a new list (does not modify the original).
        """
        if len(messages) <= self.MIN_RECENT_MESSAGES:
            return list(messages)

        # Protected recent messages (always kept)
        protected_count = self.MIN_RECENT_MESSAGES
        protected_start = max(0, len(messages) - protected_count)

        result = []
        seen_hashes: list[str] = []

        for i, msg in enumerate(messages):
            # Always keep protected recent messages
            if i >= protected_start:
                result.append(msg)
                continue

            msg_hash = self._message_hash(msg)

            # Check if duplicate within the window
            if msg_hash in seen_hashes[-self.DEDUP_WINDOW:]:
                logger.debug(
                    f"Deduplicating message at index {i}: "
                    f"role={msg.role}, hash={msg_hash}"
                )
                continue

            result.append(msg)
            seen_hashes.append(msg_hash)

        if len(result) < len(messages):
            logger.info(
                f"Deduplicated {len(messages) - len(result)} messages "
                f"({len(messages)} → {len(result)})"
            )

        return result


# ── Sprint 27: Context Priority Scorer ───────────────────────────

class ContextPriorityScorer:
    """
    Advanced 8-signal message importance scorer for smart context assembly.

    Replaces the basic 5-weight scoring in ContextManager with a more
    nuanced system that considers tool usage patterns, knowledge alignment,
    conversation flow, error context, and user intent.

    Scoring signals and weights:
      1. Recency (0.20)      — exponential decay favoring recent messages
      2. Role (0.15)         — user > assistant > tool_result
      3. Content value (0.15) — decisions, questions, summaries score higher
      4. Tool success (0.15)  — messages about actively-used tools preserved
      5. Knowledge (0.10)     — messages related to stored knowledge
      6. Flow (0.10)          — unbroken chains of thought
      7. Error context (0.10) — error messages and recovery preserved
      8. User intent (0.05)   — messages with questions/requests
    """

    # Weights for 8 scoring signals
    W_RECENCY = 0.20
    W_ROLE = 0.15
    W_CONTENT = 0.15
    W_TOOL_SUCCESS = 0.15
    W_KNOWLEDGE = 0.10
    W_FLOW = 0.10
    W_ERROR_CONTEXT = 0.10
    W_USER_INTENT = 0.05

    # Decision indicator phrases
    DECISION_PHRASES = [
        "decided", "agreed", "confirmed", "chose", "resolved",
        "will use", "going with", "settled on", "picked",
        "the plan is", "approach is", "strategy is",
    ]

    # Error indicator words
    ERROR_WORDS = [
        "error", "exception", "traceback", "failed", "bug",
        "crash", "fault", "broken", "fix", "debug", "issue",
    ]

    # Recovery indicator words
    RECOVERY_WORDS = [
        "fixed", "resolved", "workaround", "fallback", "retry",
        "recovered", "solved", "patched",
    ]

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        knowledge_store: Optional[KnowledgeStore] = None,
    ):
        self._metrics = metrics_collector
        self._knowledge = knowledge_store
        self._recent_tools: list[str] = []

    # ── Properties ───────────────────────────────────────────

    @property
    def recent_tools(self) -> list[str]:
        return list(self._recent_tools)

    def set_recent_tools(self, tools: list[str]) -> None:
        """Update the list of recently used tools for scoring context."""
        self._recent_tools = list(tools)

    # ── Main scoring ─────────────────────────────────────────

    def score_message(
        self,
        msg: Message,
        position: int,
        total: int,
        recent_query: str = "",
    ) -> float:
        """
        Score a message 0.0–1.0 using 8 signals.

        Args:
            msg: The message to score.
            position: Index of the message (0 = oldest).
            total: Total number of messages.
            recent_query: Most recent user query for relevance matching.

        Returns:
            Float 0.0–1.0 where higher = more important.
        """
        score = (
            self.W_RECENCY * self._recency_score(position, total)
            + self.W_ROLE * self._role_score(msg.role)
            + self.W_CONTENT * self._content_value_score(msg)
            + self.W_TOOL_SUCCESS * self._tool_success_score(msg)
            + self.W_KNOWLEDGE * self._knowledge_alignment_score(msg)
            + self.W_FLOW * self._conversation_flow_score(msg, position, total)
            + self.W_ERROR_CONTEXT * self._error_context_score(msg)
            + self.W_USER_INTENT * self._user_intent_score(msg)
        )
        return max(0.0, min(1.0, score))

    # ── Individual scoring signals ───────────────────────────

    @staticmethod
    def _recency_score(position: int, total: int) -> float:
        """Signal 1: Exponential decay favoring recent messages."""
        if total <= 1:
            return 1.0
        normalized = position / (total - 1)
        return math.pow(normalized, 1.5)

    @staticmethod
    def _role_score(role: str) -> float:
        """Signal 2: User messages most important."""
        weights = {
            "user": 1.0,
            "assistant": 0.8,
            "tool_result": 0.5,
        }
        return weights.get(role, 0.5)

    def _content_value_score(self, msg: Message) -> float:
        """Signal 3: Decisions, summaries, and substantive content score higher."""
        if not msg.content:
            return 0.0

        lower = msg.content.lower()
        score = 0.0

        # Decisions are highly valuable
        if any(phrase in lower for phrase in self.DECISION_PHRASES):
            score += 0.4

        # Summaries contain condensed information
        if "summary" in lower or "overview" in lower:
            score += 0.3

        # Code snippets are often valuable
        if "```" in msg.content or "def " in msg.content:
            score += 0.2

        # Very short messages are less valuable
        if len(msg.content) < 20:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _tool_success_score(self, msg: Message) -> float:
        """Signal 4: Messages about actively-used tools are preserved."""
        if not msg.tool_results and not msg.tool_calls:
            return 0.3  # Neutral for non-tool messages

        score = 0.0

        # Check if message involves recently-used tools
        tool_names = set()
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_names.add(tc.name)
        if msg.tool_results:
            for tr in msg.tool_results:
                if hasattr(tr, "tool_name"):
                    tool_names.add(tr.tool_name)

        # Boost messages involving recently-used tools
        if self._recent_tools and tool_names:
            overlap = tool_names & set(self._recent_tools)
            if overlap:
                score += 0.5

        # Failed tool results are more informative
        if msg.tool_results:
            has_failure = any(not tr.success for tr in msg.tool_results)
            if has_failure:
                score += 0.4
            else:
                score += 0.2

        return max(0.0, min(1.0, score))

    def _knowledge_alignment_score(self, msg: Message) -> float:
        """Signal 5: Messages related to stored knowledge score higher."""
        if not self._knowledge or not msg.content:
            return 0.3  # Neutral when no knowledge store

        # Quick keyword search against knowledge store
        try:
            results = self._knowledge.search(msg.content[:100], limit=3)
            if results:
                return min(1.0, 0.3 + 0.2 * len(results))
        except Exception:
            pass

        return 0.2

    @staticmethod
    def _conversation_flow_score(msg: Message, position: int, total: int) -> float:
        """Signal 6: Preserve unbroken chains of thought."""
        # Messages in the middle of a sequence get a bonus
        # (they're part of a chain, removing them breaks flow)
        if total <= 2:
            return 0.5

        # Newer messages in a chain are more important
        relative_pos = position / (total - 1) if total > 1 else 0.5

        # Slight boost for messages that are part of a continuous run
        if 0.3 < relative_pos < 0.8:
            return 0.6  # Mid-conversation messages get slight boost

        return 0.4

    def _error_context_score(self, msg: Message) -> float:
        """Signal 7: Error messages and recovery attempts preserved aggressively."""
        if not msg.content:
            return 0.0

        lower = msg.content.lower()
        score = 0.0

        # Error messages
        if any(word in lower for word in self.ERROR_WORDS):
            score += 0.6

        # Recovery/fix messages (keep to remember what worked)
        if any(word in lower for word in self.RECOVERY_WORDS):
            score += 0.5

        # Tool failures
        if msg.tool_results:
            has_failure = any(not tr.success for tr in msg.tool_results)
            if has_failure:
                score += 0.4

        return max(0.0, min(1.0, score))

    @staticmethod
    def _user_intent_score(msg: Message) -> float:
        """Signal 8: Messages with questions or requests preserved."""
        if msg.role != "user" or not msg.content:
            return 0.0

        # Questions
        if "?" in msg.content:
            return 0.8

        # Imperative requests (common openers)
        lower = msg.content.lower().strip()
        request_starters = [
            "please", "can you", "could you", "help me", "show me",
            "create", "fix", "update", "add", "remove", "explain",
        ]
        if any(lower.startswith(s) for s in request_starters):
            return 0.7

        return 0.3
