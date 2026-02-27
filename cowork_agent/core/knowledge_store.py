"""
Knowledge Store — Persistent cross-session memory.

Stores facts, preferences, and decisions as key-value entries in a JSON file
under the workspace directory.  Survives across agent sessions, giving the
agent continuity it would otherwise lack.
"""

from __future__ import annotations

import json
import logging
import os
import time
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """A single piece of remembered knowledge."""
    key: str
    value: str
    category: str  # "facts", "preferences", "decisions"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeEntry:
        return cls(
            key=data["key"],
            value=data["value"],
            category=data.get("category", "facts"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            access_count=data.get("access_count", 0),
        )


VALID_CATEGORIES = {"facts", "preferences", "decisions"}


class KnowledgeStore:
    """Persistent key-value store for cross-session memory.

    Storage layout:
        {workspace}/.cowork/knowledge.json

    Categories:
        facts        — project info, file locations, architecture notes
        preferences  — user preferences, coding style, tool choices
        decisions    — key decisions made during conversations
    """

    def __init__(self, workspace_dir: str = "", max_entries: int = 500):
        self._entries: dict[str, KnowledgeEntry] = {}  # key → entry
        self._max_entries = max_entries
        self._dirty = False

        # Set up persistence path
        if workspace_dir:
            store_dir = os.path.join(workspace_dir, ".cowork")
            os.makedirs(store_dir, exist_ok=True)
            self._path = os.path.join(store_dir, "knowledge.json")
        else:
            self._path = ""

        # Load existing knowledge
        self.load()

    # ── CRUD ───────────────────────────────────────────────

    def remember(self, category: str, key: str, value: str) -> None:
        """Store or update a knowledge entry."""
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Must be one of {VALID_CATEGORIES}")
        if not key or not key.strip():
            raise ValueError("Key cannot be empty")

        key = key.strip()
        now = time.time()

        if key in self._entries:
            entry = self._entries[key]
            entry.value = value
            entry.category = category
            entry.updated_at = now
            logger.debug(f"Knowledge updated: [{category}] {key}")
        else:
            self._entries[key] = KnowledgeEntry(
                key=key,
                value=value,
                category=category,
                created_at=now,
                updated_at=now,
            )
            logger.debug(f"Knowledge stored: [{category}] {key}")

        self._dirty = True
        self._auto_prune()
        self.save()

    def recall(self, category: str, key: str, default: str = "") -> str:
        """Retrieve a knowledge entry by key. Returns default if not found."""
        entry = self._entries.get(key.strip())
        if entry is None:
            return default
        if category and entry.category != category:
            return default
        entry.access_count += 1
        entry.updated_at = time.time()
        self._dirty = True
        return entry.value

    def recall_all(self, category: str) -> list[KnowledgeEntry]:
        """Retrieve all entries in a category, sorted by most recently updated."""
        entries = [e for e in self._entries.values() if e.category == category]
        entries.sort(key=lambda e: e.updated_at, reverse=True)
        return entries

    def forget(self, key: str) -> bool:
        """Remove a knowledge entry. Returns True if it existed."""
        key = key.strip()
        if key in self._entries:
            del self._entries[key]
            self._dirty = True
            self.save()
            logger.debug(f"Knowledge forgotten: {key}")
            return True
        return False

    # ── Search ─────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> list[KnowledgeEntry]:
        """Search entries by substring match in key or value.

        Results scored by: match in key (2x) + match in value (1x) + recency.
        """
        if not query or not query.strip():
            return []

        query_lower = query.strip().lower()
        scored: list[tuple[float, KnowledgeEntry]] = []

        for entry in self._entries.values():
            score = 0.0
            if query_lower in entry.key.lower():
                score += 2.0
            if query_lower in entry.value.lower():
                score += 1.0
            if score > 0:
                # Boost by recency (entries accessed recently score higher)
                age_days = (time.time() - entry.updated_at) / 86400
                recency_bonus = max(0, 1.0 - age_days / 30)  # Decays over 30 days
                score += recency_bonus * 0.5
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    # ── Persistence ────────────────────────────────────────

    def save(self) -> None:
        """Write knowledge to disk (atomic write)."""
        if not self._path:
            return

        data = {
            "version": 1,
            "entries": [e.to_dict() for e in self._entries.values()],
        }

        try:
            # Atomic write: write to temp file, then rename
            dir_name = os.path.dirname(self._path)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, self._path)
            except Exception:
                # Clean up temp file on failure
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

            self._dirty = False
            logger.debug(f"Knowledge saved: {len(self._entries)} entries")
        except Exception as e:
            logger.warning(f"Failed to save knowledge store: {e}")

    def load(self) -> None:
        """Load knowledge from disk."""
        if not self._path or not os.path.exists(self._path):
            return

        try:
            with open(self._path, "r") as f:
                data = json.load(f)

            for entry_data in data.get("entries", []):
                entry = KnowledgeEntry.from_dict(entry_data)
                self._entries[entry.key] = entry

            logger.debug(f"Knowledge loaded: {len(self._entries)} entries")
        except Exception as e:
            logger.warning(f"Failed to load knowledge store: {e}")

    # ── Maintenance ────────────────────────────────────────

    def _auto_prune(self) -> None:
        """Remove oldest entries if over the limit."""
        if len(self._entries) <= self._max_entries:
            return
        self.prune(self._max_entries)

    def prune(self, max_total: int = 500) -> int:
        """Remove oldest entries to stay within limit. Returns count removed."""
        if len(self._entries) <= max_total:
            return 0

        # Sort by updated_at ascending (oldest first)
        sorted_keys = sorted(
            self._entries.keys(),
            key=lambda k: self._entries[k].updated_at,
        )

        to_remove = len(self._entries) - max_total
        removed = 0
        for key in sorted_keys[:to_remove]:
            del self._entries[key]
            removed += 1

        if removed:
            self._dirty = True
            logger.debug(f"Knowledge pruned: {removed} entries removed")

        return removed

    def stats(self) -> dict:
        """Return counts per category and total."""
        counts: dict[str, int] = {}
        for entry in self._entries.values():
            counts[entry.category] = counts.get(entry.category, 0) + 1
        return {
            "total": len(self._entries),
            "categories": counts,
            "max_entries": self._max_entries,
            "path": self._path,
        }

    def export_data(self) -> dict:
        """Export all knowledge as a dict (for backup/transfer)."""
        return {
            "version": 1,
            "entries": [e.to_dict() for e in self._entries.values()],
        }

    def import_data(self, data: dict) -> int:
        """Import knowledge entries from a dict. Returns count imported."""
        imported = 0
        for entry_data in data.get("entries", []):
            try:
                entry = KnowledgeEntry.from_dict(entry_data)
                self._entries[entry.key] = entry
                imported += 1
            except (KeyError, TypeError):
                continue
        if imported:
            self._dirty = True
            self.save()
        return imported

    @property
    def size(self) -> int:
        return len(self._entries)
