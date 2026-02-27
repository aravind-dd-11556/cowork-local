"""
State Snapshot Manager — checkpoint and restore full agent state.

Stores snapshots in ``{workspace}/.cowork/snapshots/{snapshot_id}/snapshot.json``.
Supports manual and auto-snapshot before risky tool executions, and
rollback to any previous checkpoint.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import Message

logger = logging.getLogger(__name__)


# ── Dataclass ───────────────────────────────────────────────────────

@dataclass
class StateSnapshot:
    """A point-in-time snapshot of agent state."""
    snapshot_id: str
    timestamp: float
    label: str

    # State bundle
    messages: List[Message] = field(default_factory=list)
    todos: List[dict] = field(default_factory=list)
    token_usage_summary: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "label": self.label,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                }
                for m in self.messages
            ],
            "todos": self.todos,
            "token_usage_summary": self.token_usage_summary,
            "session_id": self.session_id,
            "config_snapshot": self.config_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StateSnapshot":
        messages = [
            Message(
                role=m["role"],
                content=m["content"],
                timestamp=m.get("timestamp", 0.0),
            )
            for m in data.get("messages", [])
        ]
        return cls(
            snapshot_id=data["snapshot_id"],
            timestamp=data["timestamp"],
            label=data.get("label", ""),
            messages=messages,
            todos=data.get("todos", []),
            token_usage_summary=data.get("token_usage_summary", {}),
            session_id=data.get("session_id"),
            config_snapshot=data.get("config_snapshot", {}),
        )


# ── StateSnapshotManager ───────────────────────────────────────────

class StateSnapshotManager:
    """
    Capture and restore full agent state.

    Storage layout::

        {workspace}/.cowork/snapshots/
            {snapshot_id}/
                snapshot.json
    """

    # Tools considered "risky" — auto-snapshot before execution
    RISKY_TOOLS = {"bash", "write", "edit", "delete_file"}

    def __init__(self, workspace_dir: str = "", max_snapshots: int = 20):
        self._workspace_dir = workspace_dir
        self._max_snapshots = max_snapshots

        self._snapshots_dir = ""
        if workspace_dir:
            self._snapshots_dir = os.path.join(workspace_dir, ".cowork", "snapshots")
            os.makedirs(self._snapshots_dir, exist_ok=True)

        self._index: List[StateSnapshot] = []  # lightweight index (no messages loaded)
        self._load_index()

    # ── Create ─────────────────────────────────────────────────

    def create_snapshot(
        self,
        messages: List[Message],
        label: str = "",
        todos: Optional[List[dict]] = None,
        token_usage_summary: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Capture current state as a snapshot.

        Returns the snapshot_id.
        """
        snap_id = f"snap_{uuid.uuid4().hex[:12]}"

        snapshot = StateSnapshot(
            snapshot_id=snap_id,
            timestamp=time.time(),
            label=label or f"Snapshot {snap_id[:12]}",
            messages=list(messages),
            todos=todos or [],
            token_usage_summary=token_usage_summary or {},
            session_id=session_id,
        )

        self._save_to_disk(snapshot)
        self._index.append(snapshot)

        # Evict oldest if over limit
        while len(self._index) > self._max_snapshots:
            oldest = self._index.pop(0)
            self._delete_from_disk(oldest.snapshot_id)

        logger.info("Created snapshot: %s (%s)", snap_id, label)
        return snap_id

    # ── Restore ────────────────────────────────────────────────

    def restore_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """
        Load a full snapshot from disk.

        Returns the StateSnapshot or None if not found.
        The caller is responsible for applying the state to the Agent.
        """
        snapshot = self._load_from_disk(snapshot_id)
        if snapshot:
            logger.info("Restored snapshot: %s", snapshot_id)
        return snapshot

    # ── List / Delete ──────────────────────────────────────────

    def list_snapshots(self, limit: int = 20) -> List[StateSnapshot]:
        """List recent snapshots (newest first). Messages not loaded."""
        sorted_snaps = sorted(self._index, key=lambda s: s.timestamp, reverse=True)
        return sorted_snaps[:limit]

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        self._index = [s for s in self._index if s.snapshot_id != snapshot_id]
        return self._delete_from_disk(snapshot_id)

    # ── Auto-snapshot before risky tools ───────────────────────

    def auto_snapshot_before_risky(
        self,
        tool_name: str,
        messages: List[Message],
        session_id: Optional[str] = None,
    ) -> str:
        """
        Auto-snapshot before executing a risky tool.

        Returns snapshot_id if created, or empty string if the tool isn't risky.
        """
        if tool_name not in self.RISKY_TOOLS:
            return ""
        return self.create_snapshot(
            messages=messages,
            label=f"Before {tool_name}",
            session_id=session_id,
        )

    # ── Properties ─────────────────────────────────────────────

    @property
    def snapshot_count(self) -> int:
        return len(self._index)

    # ── Internal: disk I/O ─────────────────────────────────────

    def _save_to_disk(self, snapshot: StateSnapshot) -> None:
        if not self._snapshots_dir:
            return
        snap_dir = os.path.join(self._snapshots_dir, snapshot.snapshot_id)
        os.makedirs(snap_dir, exist_ok=True)
        try:
            path = os.path.join(snap_dir, "snapshot.json")
            with open(path, "w") as f:
                json.dump(snapshot.to_dict(), f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save snapshot %s: %s", snapshot.snapshot_id, exc)

    def _load_from_disk(self, snapshot_id: str) -> Optional[StateSnapshot]:
        if not self._snapshots_dir:
            return None
        path = os.path.join(self._snapshots_dir, snapshot_id, "snapshot.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return StateSnapshot.from_dict(data)
        except Exception as exc:
            logger.warning("Failed to load snapshot %s: %s", snapshot_id, exc)
            return None

    def _delete_from_disk(self, snapshot_id: str) -> bool:
        if not self._snapshots_dir:
            return False
        snap_dir = os.path.join(self._snapshots_dir, snapshot_id)
        if not os.path.exists(snap_dir):
            return False
        try:
            shutil.rmtree(snap_dir)
            return True
        except Exception as exc:
            logger.warning("Failed to delete snapshot %s: %s", snapshot_id, exc)
            return False

    def _load_index(self) -> None:
        """Scan snapshot directories and build the in-memory index."""
        if not self._snapshots_dir or not os.path.isdir(self._snapshots_dir):
            return
        for entry in os.listdir(self._snapshots_dir):
            snap_path = os.path.join(self._snapshots_dir, entry, "snapshot.json")
            if os.path.isfile(snap_path):
                try:
                    with open(snap_path, "r") as f:
                        data = json.load(f)
                    # Load lightweight index (messages loaded on restore only)
                    self._index.append(StateSnapshot(
                        snapshot_id=data["snapshot_id"],
                        timestamp=data["timestamp"],
                        label=data.get("label", ""),
                    ))
                except Exception:
                    pass
