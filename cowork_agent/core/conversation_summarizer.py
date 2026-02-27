"""
Conversation Summarizer — Compresses old messages into compact summaries.

Two strategies:
  - Rule-based (default): Extract tool usage, file paths, decisions, errors.
    Zero LLM calls — fast and free.
  - LLM-based (optional): Ask the provider to summarize. More nuanced but
    costs tokens.

Used by ContextManager during pruning (so dropped messages are summarized,
not just lost) and by Agent for sliding window summaries.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from enum import Enum
from typing import Optional

from .models import Message

logger = logging.getLogger(__name__)


class SummarizationStrategy(Enum):
    RULE_BASED = "rule_based"
    LLM_BASED = "lm_based"


class ConversationSummarizer:
    """Compresses conversation messages into compact summaries."""

    # Decision indicator phrases (case-insensitive)
    DECISION_PHRASES = [
        "decided", "agreed", "confirmed", "chose", "resolved",
        "will use", "going with", "settled on", "picked",
        "the plan is", "approach is", "strategy is",
    ]

    # File path pattern
    FILE_PATH_RE = re.compile(
        r'(?:/[\w.\-]+)+(?:\.\w+)?'  # Unix-style paths like /foo/bar.py
    )

    def __init__(self, provider=None, strategy: SummarizationStrategy = SummarizationStrategy.RULE_BASED):
        self._provider = provider  # Optional LLM provider for LLM-based mode
        self._strategy = strategy

    def summarize(self, messages: list[Message]) -> str:
        """Summarize a list of messages into a compact text block.

        Returns a ~100-200 token summary capturing tools used, files
        accessed, decisions made, and errors encountered.
        """
        if not messages:
            return ""

        if self._strategy == SummarizationStrategy.LLM_BASED and self._provider:
            return self._summarize_llm(messages)

        return self._summarize_rule_based(messages)

    def update_sliding_summary(self, existing: str, new_messages: list[Message]) -> str:
        """Incrementally update a running summary with new messages.

        Merges the existing summary with information from new messages.
        """
        new_summary = self.summarize(new_messages)
        if not existing:
            return new_summary
        if not new_summary:
            return existing

        # Merge: keep existing summary header, append new info
        # Parse sections from both summaries and merge them
        return self._merge_summaries(existing, new_summary)

    # ── Rule-based summarization ─────────────────────────────

    def _summarize_rule_based(self, messages: list[Message]) -> str:
        """Extract key patterns from messages without LLM calls."""
        tools = self._extract_tools_used(messages)
        files = self._extract_file_paths(messages)
        decisions = self._extract_decisions(messages)
        errors = self._extract_errors(messages)
        user_requests = self._extract_user_requests(messages)

        # Count turns
        user_turns = sum(1 for m in messages if m.role == "user")
        assistant_turns = sum(1 for m in messages if m.role == "assistant")

        parts = []

        # Turn count
        total = len(messages)
        parts.append(f"[Memory Summary — {total} messages, {user_turns} user turns]")

        # Tools
        if tools:
            tool_strs = []
            for name, counts in sorted(tools.items(), key=lambda x: sum(x[1].values()), reverse=True):
                s, f = counts.get("success", 0), counts.get("failure", 0)
                if f:
                    tool_strs.append(f"{name}({s}✓ {f}✗)")
                else:
                    tool_strs.append(f"{name}({s}x)")
            parts.append(f"• Tools: {', '.join(tool_strs)}")

        # Files
        if files:
            # Show up to 8 most relevant files
            parts.append(f"• Files: {', '.join(files[:8])}")
            if len(files) > 8:
                parts.append(f"  + {len(files) - 8} more files")

        # User requests (brief)
        if user_requests:
            parts.append(f"• Requests: {'; '.join(user_requests[:4])}")

        # Decisions
        if decisions:
            parts.append(f"• Decisions: {'; '.join(decisions[:4])}")

        # Errors
        if errors:
            parts.append(f"• Errors: {'; '.join(errors[:4])}")

        return "\n".join(parts)

    def _extract_tools_used(self, messages: list[Message]) -> dict[str, Counter]:
        """Extract tool names and success/failure counts."""
        tools: dict[str, Counter] = {}

        for msg in messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.name not in tools:
                        tools[tc.name] = Counter()

            if msg.tool_results:
                for tr in msg.tool_results:
                    # Find the tool name from tool_calls in previous messages
                    tool_name = self._find_tool_name(messages, tr.tool_id)
                    if tool_name:
                        if tool_name not in tools:
                            tools[tool_name] = Counter()
                        if tr.success:
                            tools[tool_name]["success"] += 1
                        else:
                            tools[tool_name]["failure"] += 1

        return tools

    def _find_tool_name(self, messages: list[Message], tool_id: str) -> Optional[str]:
        """Find the tool name for a given tool_id by searching tool_calls."""
        for msg in messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.tool_id == tool_id:
                        return tc.name
        return None

    def _extract_file_paths(self, messages: list[Message]) -> list[str]:
        """Extract file paths from tool call inputs and message content."""
        paths = set()

        for msg in messages:
            # From tool call inputs
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.name in ("read", "write", "edit", "glob", "grep"):
                        fp = tc.input.get("file_path") or tc.input.get("path") or ""
                        if fp:
                            paths.add(fp)
                        pattern = tc.input.get("pattern", "")
                        if pattern and "/" in pattern:
                            paths.add(pattern)

            # From message content (regex)
            if msg.content:
                found = self.FILE_PATH_RE.findall(msg.content)
                for p in found:
                    # Filter out common false positives
                    if len(p) > 5 and not p.startswith("/api/") and "http" not in p:
                        paths.add(p)

        # Sort by frequency of occurrence (most mentioned first)
        return sorted(paths, key=lambda p: len(p))[:15]

    def _extract_decisions(self, messages: list[Message]) -> list[str]:
        """Extract decision-related sentences from messages."""
        decisions = []

        for msg in messages:
            if msg.role != "assistant" or not msg.content:
                continue

            # Split into sentences and check each
            sentences = re.split(r'[.!?\n]', msg.content)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10 or len(sentence) > 200:
                    continue
                lower = sentence.lower()
                if any(phrase in lower for phrase in self.DECISION_PHRASES):
                    # Clean up and truncate
                    clean = sentence[:150].strip()
                    if clean and clean not in decisions:
                        decisions.append(clean)

        return decisions[:6]  # Max 6 decisions

    def _extract_errors(self, messages: list[Message]) -> list[str]:
        """Extract error messages from failed tool results."""
        errors = []

        for msg in messages:
            if msg.tool_results:
                for tr in msg.tool_results:
                    if not tr.success and tr.error:
                        # Truncate long error messages
                        err = tr.error[:120].strip()
                        if err and err not in errors:
                            errors.append(err)

        return errors[:6]

    def _extract_user_requests(self, messages: list[Message]) -> list[str]:
        """Extract brief summaries of user requests."""
        requests = []

        for msg in messages:
            if msg.role == "user" and msg.content:
                # Take first 80 chars of each user message as a request summary
                text = msg.content.strip()
                if text.startswith("[") and "]" in text:
                    continue  # Skip system messages like [MEMORY SUMMARY]
                brief = text[:80]
                if len(text) > 80:
                    brief += "..."
                if brief:
                    requests.append(brief)

        return requests[:5]

    # ── LLM-based summarization ──────────────────────────────

    def _summarize_llm(self, messages: list[Message]) -> str:
        """Use the LLM provider to generate a summary (costs tokens)."""
        # Build a compact representation of the conversation
        lines = []
        for msg in messages:
            if msg.role == "user":
                lines.append(f"User: {msg.content[:200]}")
            elif msg.role == "assistant":
                lines.append(f"Agent: {msg.content[:200]}")
            elif msg.role == "tool_result" and msg.tool_results:
                for tr in msg.tool_results:
                    status = "OK" if tr.success else "FAIL"
                    lines.append(f"Tool[{tr.tool_id}]: {status} — {(tr.output or tr.error or '')[:100]}")

        conversation_text = "\n".join(lines)

        # We can't call async from sync context, so return rule-based as fallback
        # LLM summarization would need to be async — but the interface is sync
        # for compatibility with the pruning pipeline.
        logger.warning("LLM-based summarization requires async context; falling back to rule-based")
        return self._summarize_rule_based(messages)

    # ── Summary merging ──────────────────────────────────────

    def _merge_summaries(self, existing: str, new: str) -> str:
        """Merge two summaries, keeping the most relevant info from both."""
        # Parse sections from new summary
        existing_lines = existing.strip().split("\n")
        new_lines = new.strip().split("\n")

        # Keep the header from the new summary (it has the latest turn count)
        merged = []
        header = new_lines[0] if new_lines else existing_lines[0]
        merged.append(header)

        # Merge bullet points by type
        sections: dict[str, list[str]] = {}
        for line in existing_lines[1:] + new_lines[1:]:
            line = line.strip()
            if line.startswith("• "):
                # Extract section name (e.g., "Tools", "Files", "Decisions")
                colon_idx = line.find(":")
                if colon_idx > 0:
                    section_name = line[2:colon_idx].strip()
                    section_content = line[colon_idx + 1:].strip()
                    if section_name not in sections:
                        sections[section_name] = []
                    sections[section_name].append(section_content)
            elif line.startswith("  +"):
                continue  # Skip "X more files" lines

        # Rebuild merged sections
        for name, contents in sections.items():
            # Combine and deduplicate
            combined = "; ".join(contents)
            # Truncate if too long
            if len(combined) > 300:
                combined = combined[:297] + "..."
            merged.append(f"• {name}: {combined}")

        return "\n".join(merged)
