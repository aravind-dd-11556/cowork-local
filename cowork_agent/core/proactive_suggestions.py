"""
Sprint 42 · Proactive Suggestions
===================================
Generates user-facing suggestions from file events + analysis.
"""

from __future__ import annotations

import uuid
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set

from .file_watcher import FileEvent
from .workspace_analyzer import FileIssue, WorkspaceAnalyzer


# ── Data ────────────────────────────────────────────────────────────

@dataclass
class Suggestion:
    """A proactive suggestion for the user."""
    suggestion_id: str
    file_path: str
    issue: FileIssue
    proposed_action: str
    auto_applicable: bool = False
    created_at: float = 0.0
    status: str = "pending"  # pending | accepted | dismissed

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "suggestion_id": self.suggestion_id,
            "file_path": self.file_path,
            "issue": self.issue.to_dict(),
            "proposed_action": self.proposed_action,
            "auto_applicable": self.auto_applicable,
            "created_at": self.created_at,
            "status": self.status,
        }

    @staticmethod
    def generate_id() -> str:
        return f"sug_{uuid.uuid4().hex[:10]}"


# ── Action mapping ──────────────────────────────────────────────────

_ACTION_MAP = {
    "invalid_json": "Fix the JSON syntax error",
    "broken_yaml": "Fix the YAML syntax error",
    "syntax_error": "Fix the Python syntax error",
    "merge_conflict": "Resolve the merge conflict markers",
    "large_file": "Consider splitting or compressing this file",
}


# ── Engine ──────────────────────────────────────────────────────────

class SuggestionEngine:
    """
    Processes file events through WorkspaceAnalyzer and produces
    user-facing Suggestions.
    """

    def __init__(
        self,
        workspace_analyzer: WorkspaceAnalyzer,
        max_pending: int = 10,
    ):
        self.analyzer = workspace_analyzer
        self.max_pending = max_pending
        self._queue: Deque[Suggestion] = deque(maxlen=max_pending)
        self._dismissed: Set[str] = set()  # file paths we won't re-suggest
        self._all_suggestions: List[Suggestion] = []

    # ── process events ──────────────────────────────────────────

    def process_events(self, events: List[FileEvent]) -> List[Suggestion]:
        """
        Analyze the changed files and generate suggestions.
        Returns newly created suggestions.
        """
        new_suggestions: List[Suggestion] = []
        # Only analyze created/modified files
        paths = [
            ev.path for ev in events
            if ev.event_type in ("created", "modified")
            and ev.path not in self._dismissed
        ]
        if not paths:
            return new_suggestions

        issues = self.analyzer.analyze_batch(paths)
        for issue in issues:
            if issue.file_path in self._dismissed:
                continue
            sug = Suggestion(
                suggestion_id=Suggestion.generate_id(),
                file_path=issue.file_path,
                issue=issue,
                proposed_action=_ACTION_MAP.get(issue.issue_type, "Review this file"),
                auto_applicable=issue.auto_fixable,
            )
            self._queue.append(sug)
            self._all_suggestions.append(sug)
            new_suggestions.append(sug)

        return new_suggestions

    # ── management ──────────────────────────────────────────────

    def get_pending(self) -> List[Suggestion]:
        return [s for s in self._queue if s.status == "pending"]

    def accept(self, suggestion_id: str) -> bool:
        for s in self._queue:
            if s.suggestion_id == suggestion_id:
                s.status = "accepted"
                return True
        return False

    def dismiss(self, suggestion_id: str) -> bool:
        for s in self._queue:
            if s.suggestion_id == suggestion_id:
                s.status = "dismissed"
                self._dismissed.add(s.file_path)
                return True
        return False

    def clear(self) -> None:
        self._queue.clear()
        self._dismissed.clear()

    @property
    def dismissed_paths(self) -> Set[str]:
        return set(self._dismissed)

    @property
    def total_generated(self) -> int:
        return len(self._all_suggestions)

    # ── formatting ──────────────────────────────────────────────

    @staticmethod
    def format_for_user(suggestion: Suggestion) -> str:
        """Produce a human-readable string for the user."""
        parts = [
            f"💡 {suggestion.file_path}: {suggestion.issue.description}",
        ]
        if suggestion.issue.line_number:
            parts.append(f"   Line {suggestion.issue.line_number}")
        parts.append(f"   Suggested: {suggestion.proposed_action}")
        return "\n".join(parts)
