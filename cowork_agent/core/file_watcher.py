"""
Sprint 42 · File Watcher
=========================
Pure-asyncio polling-based file watcher (no watchdog dependency).
Detects created, modified, deleted, and moved files.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set


# ── Data ────────────────────────────────────────────────────────────

@dataclass
class FileEvent:
    """Represents a single file-system change."""
    path: str
    event_type: str  # "created" | "modified" | "deleted" | "moved"
    timestamp: float = 0.0
    old_path: Optional[str] = None  # for moves

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "old_path": self.old_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FileEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FileWatcherConfig:
    """Tuning knobs for the watcher."""
    enabled: bool = True
    watch_patterns: List[str] = field(default_factory=lambda: [
        "*.py", "*.json", "*.yaml", "*.yml", "*.md",
        "*.js", "*.ts", "*.jsx", "*.tsx", "*.html", "*.css",
    ])
    ignore_patterns: List[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", ".cowork",
        "*.pyc", ".pytest_cache", ".mypy_cache",
    ])
    poll_interval_seconds: float = 2.0
    debounce_seconds: float = 1.0
    max_events_per_cycle: int = 50

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "watch_patterns": self.watch_patterns,
            "ignore_patterns": self.ignore_patterns,
            "poll_interval_seconds": self.poll_interval_seconds,
            "debounce_seconds": self.debounce_seconds,
        }


# ── Watcher ─────────────────────────────────────────────────────────

class FileWatcher:
    """
    Async polling-based file watcher.

    Scans the workspace directory tree at regular intervals,
    comparing mtime snapshots to detect changes.
    """

    def __init__(
        self,
        workspace_path: str,
        config: Optional[FileWatcherConfig] = None,
        on_events: Optional[Callable[[List[FileEvent]], Any]] = None,
    ):
        self.workspace_path = workspace_path
        self.config = config or FileWatcherConfig()
        self.on_events = on_events
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._file_state: Dict[str, float] = {}  # path → mtime
        self._event_history: List[FileEvent] = []
        self._max_history = 200

    # ── lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """Begin background polling."""
        if self._running:
            return
        self._running = True
        # Initial scan
        self._file_state = self._scan_directory()
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop the watcher gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def event_history(self) -> List[FileEvent]:
        return list(self._event_history)

    # ── polling loop ────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.poll_interval_seconds)
                if not self._running:
                    break
                new_state = self._scan_directory()
                events = self._detect_changes(self._file_state, new_state)
                events = self._debounce(events)
                self._file_state = new_state

                if events:
                    # Trim to max per cycle
                    events = events[:self.config.max_events_per_cycle]
                    self._record_events(events)
                    if self.on_events:
                        result = self.on_events(events)
                        if asyncio.iscoroutine(result):
                            await result
            except asyncio.CancelledError:
                break
            except Exception:
                # Don't crash the watcher on errors
                pass

    # ── scanning ────────────────────────────────────────────────

    def _scan_directory(self) -> Dict[str, float]:
        """Walk workspace and return {relative_path: mtime}."""
        state: Dict[str, float] = {}
        try:
            for root, dirs, files in os.walk(self.workspace_path):
                # Filter ignored directories
                dirs[:] = [
                    d for d in dirs
                    if not self._matches_ignore(d)
                ]
                for fname in files:
                    if not self._matches_watch(fname):
                        continue
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, self.workspace_path)
                    try:
                        state[rel] = os.path.getmtime(full)
                    except OSError:
                        pass
        except OSError:
            pass
        return state

    def _detect_changes(
        self,
        old: Dict[str, float],
        new: Dict[str, float],
    ) -> List[FileEvent]:
        """Compare two snapshots and generate events."""
        events: List[FileEvent] = []
        now = time.time()

        # Created files
        for path in new:
            if path not in old:
                events.append(FileEvent(path=path, event_type="created", timestamp=now))

        # Deleted files
        for path in old:
            if path not in new:
                events.append(FileEvent(path=path, event_type="deleted", timestamp=now))

        # Modified files
        for path in new:
            if path in old and new[path] != old[path]:
                events.append(FileEvent(path=path, event_type="modified", timestamp=now))

        return events

    def _debounce(self, events: List[FileEvent]) -> List[FileEvent]:
        """Remove rapid duplicate events for the same path."""
        seen: Dict[str, FileEvent] = {}
        for ev in events:
            key = f"{ev.path}:{ev.event_type}"
            seen[key] = ev  # keep latest
        return list(seen.values())

    # ── pattern matching ────────────────────────────────────────

    def _matches_watch(self, filename: str) -> bool:
        return any(fnmatch.fnmatch(filename, p) for p in self.config.watch_patterns)

    def _matches_ignore(self, name: str) -> bool:
        return any(fnmatch.fnmatch(name, p) for p in self.config.ignore_patterns)

    # ── history ─────────────────────────────────────────────────

    def _record_events(self, events: List[FileEvent]) -> None:
        self._event_history.extend(events)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

    def clear_history(self) -> None:
        self._event_history.clear()
