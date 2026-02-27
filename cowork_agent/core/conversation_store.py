"""
Conversation Store — high-level API for searching, exporting, and pruning
conversations stored via SessionManager.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .session_manager import SessionManager, SessionMetadata

logger = logging.getLogger(__name__)


@dataclass
class ConversationStats:
    """Statistics about a single conversation."""
    session_id: str
    message_count: int
    token_count: int           # estimated ~4 chars per token
    duration_seconds: float
    created_at: float
    updated_at: float

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "message_count": self.message_count,
            "token_count": self.token_count,
            "duration_seconds": round(self.duration_seconds, 2),
            "age_days": round(self.age_days, 1),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class SearchResult:
    """A conversation matching search criteria."""
    session_id: str
    title: str
    excerpt: str          # first N chars of first message
    match_count: int      # number of messages containing the keyword
    created_at: float


class ConversationStore:
    """
    High-level abstraction over ``SessionManager``.

    Provides search, export, statistics, and pruning capabilities
    without introducing any new storage format — it delegates
    all I/O to the underlying SessionManager.
    """

    def __init__(self, session_manager: SessionManager):
        self._sm = session_manager

    # ── Search ─────────────────────────────────────────────────

    def search(
        self,
        keyword: Optional[str] = None,
        start_date: Optional[float] = None,
        end_date: Optional[float] = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        """
        Search conversations by keyword and/or date range.

        Parameters
        ----------
        keyword : str, optional
            Case-insensitive substring to search in message content.
        start_date : float, optional
            Unix timestamp — only sessions created *after* this.
        end_date : float, optional
            Unix timestamp — only sessions updated *before* this.
        limit : int
            Maximum number of results.
        """
        results: list[SearchResult] = []

        for meta in self._sm.list_sessions(limit=10000):
            # Date filters
            if start_date and meta.created_at < start_date:
                continue
            if end_date and meta.updated_at > end_date:
                continue

            match_count = 0
            if keyword:
                messages = self._sm.load_messages(meta.session_id)
                kw = keyword.lower()
                match_count = sum(1 for m in messages if kw in m.content.lower())
                if match_count == 0:
                    continue

            results.append(SearchResult(
                session_id=meta.session_id,
                title=meta.title,
                excerpt=self._get_excerpt(meta.session_id),
                match_count=match_count,
                created_at=meta.created_at,
            ))

        # Sort: most matches first, then most recent
        results.sort(key=lambda r: (-r.match_count, -r.created_at))
        return results[:limit]

    # ── Export ──────────────────────────────────────────────────

    def export_markdown(self, session_id: str) -> str:
        """Export a conversation to Markdown."""
        meta = self._sm.get_metadata(session_id)
        messages = self._sm.load_messages(session_id)

        if not meta:
            return ""

        lines = [
            f"# Session: {meta.title}",
            f"\n*Created: {self._fmt(meta.created_at)}, "
            f"Updated: {self._fmt(meta.updated_at)}*\n",
        ]

        for msg in messages:
            lines.append(f"## {msg.role.title()}")
            lines.append(msg.content)

            if msg.tool_calls:
                lines.append("\n**Tool Calls:**")
                for tc in msg.tool_calls:
                    lines.append(f"  - {tc.name} (id={tc.tool_id})")
                    lines.append(f"    Input: {json.dumps(tc.input, indent=2)}")

            if msg.tool_results:
                lines.append("\n**Tool Results:**")
                for tr in msg.tool_results:
                    status = "✓" if tr.success else "✗"
                    lines.append(f"  {status} {tr.tool_id}: {tr.output[:100]}")

            lines.append("")

        return "\n".join(lines)

    # ── Statistics ──────────────────────────────────────────────

    def get_stats(self, session_id: str) -> Optional[ConversationStats]:
        """Compute statistics for a conversation."""
        meta = self._sm.get_metadata(session_id)
        if not meta:
            return None

        messages = self._sm.load_messages(session_id)
        total_chars = sum(len(m.content) for m in messages)

        return ConversationStats(
            session_id=session_id,
            message_count=len(messages),
            token_count=total_chars // 4,          # rough estimate
            duration_seconds=meta.updated_at - meta.created_at,
            created_at=meta.created_at,
            updated_at=meta.updated_at,
        )

    # ── Pruning ────────────────────────────────────────────────

    def prune_old_sessions(
        self,
        max_age_days: int = 90,
        min_keep: int = 10,
    ) -> list[str]:
        """
        Delete sessions older than *max_age_days*.

        Always keeps at least *min_keep* most-recent sessions.
        Returns list of deleted session IDs.
        """
        cutoff = time.time() - (max_age_days * 86400)

        all_sessions = self._sm.list_sessions(limit=10000)
        # Sort newest first
        all_sessions.sort(key=lambda s: s.updated_at, reverse=True)

        # Never delete the most recent min_keep
        candidates = all_sessions[min_keep:]

        deleted: list[str] = []
        for session in candidates:
            if session.updated_at < cutoff:
                result = self._sm.delete_session(session.session_id)
                if result and "deleted" in str(result).lower():
                    deleted.append(session.session_id)

        return deleted

    # ── Helpers ─────────────────────────────────────────────────

    def _get_excerpt(self, session_id: str, length: int = 100) -> str:
        messages = self._sm.load_messages(session_id)
        if messages:
            return messages[0].content[:length]
        return ""

    @staticmethod
    def _fmt(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
