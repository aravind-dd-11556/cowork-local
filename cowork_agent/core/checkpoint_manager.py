"""
Sprint 41 · Checkpoint Manager
================================
Enhanced checkpointing that captures execution progress —
step index, partial tool results, pending operations.

Storage: {workspace}/.cowork/checkpoints/{checkpoint_id}.json
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Data ────────────────────────────────────────────────────────────

@dataclass
class ExecutionCheckpoint:
    """Snapshot of execution progress at a specific step."""
    checkpoint_id: str
    task_id: str
    step_index: int
    total_steps: int
    messages_count: int = 0
    tool_results_so_far: List[Dict[str, Any]] = field(default_factory=list)
    pending_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "task_id": self.task_id,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "messages_count": self.messages_count,
            "tool_results_so_far": self.tool_results_so_far,
            "pending_tool_calls": self.pending_tool_calls,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionCheckpoint":
        return cls(
            checkpoint_id=d["checkpoint_id"],
            task_id=d.get("task_id", ""),
            step_index=d.get("step_index", 0),
            total_steps=d.get("total_steps", 0),
            messages_count=d.get("messages_count", 0),
            tool_results_so_far=d.get("tool_results_so_far", []),
            pending_tool_calls=d.get("pending_tool_calls", []),
            metadata=d.get("metadata", {}),
            timestamp=d.get("timestamp", 0.0),
        )

    @staticmethod
    def generate_id() -> str:
        return f"ckpt_{uuid.uuid4().hex[:12]}"

    @property
    def progress_pct(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return min((self.step_index / self.total_steps) * 100, 100.0)


# ── Manager ─────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Manages execution checkpoints for task continuity.

    Stores checkpoints in ``{workspace}/.cowork/checkpoints/{id}.json``
    and supports per-task eviction to keep storage bounded.
    """

    STORAGE_DIR = ".cowork/checkpoints"
    MAX_CHECKPOINTS_PER_TASK = 10

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self._storage_dir = os.path.join(workspace_path, self.STORAGE_DIR)
        os.makedirs(self._storage_dir, exist_ok=True)

    # ── persistence ─────────────────────────────────────────────

    def _checkpoint_path(self, checkpoint_id: str) -> str:
        return os.path.join(self._storage_dir, f"{checkpoint_id}.json")

    def create_checkpoint(
        self,
        task_id: str,
        step_index: int,
        total_steps: int,
        messages_count: int = 0,
        tool_results_so_far: Optional[List[dict]] = None,
        pending_tool_calls: Optional[List[dict]] = None,
        **metadata,
    ) -> str:
        """Create and persist a checkpoint.  Returns checkpoint_id."""
        cp = ExecutionCheckpoint(
            checkpoint_id=ExecutionCheckpoint.generate_id(),
            task_id=task_id,
            step_index=step_index,
            total_steps=total_steps,
            messages_count=messages_count,
            tool_results_so_far=tool_results_so_far or [],
            pending_tool_calls=pending_tool_calls or [],
            metadata=metadata,
        )
        path = self._checkpoint_path(cp.checkpoint_id)
        with open(path, "w") as f:
            json.dump(cp.to_dict(), f, indent=2)

        # Evict old checkpoints for this task
        self._evict_oldest(task_id)
        return cp.checkpoint_id

    def restore_checkpoint(self, checkpoint_id: str) -> Optional[ExecutionCheckpoint]:
        """Load a checkpoint from disk."""
        path = self._checkpoint_path(checkpoint_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return ExecutionCheckpoint.from_dict(json.load(f))

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        path = self._checkpoint_path(checkpoint_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    # ── queries ─────────────────────────────────────────────────

    def list_checkpoints(self, task_id: Optional[str] = None) -> List[ExecutionCheckpoint]:
        """List checkpoints, optionally filtered by task_id. Sorted by timestamp."""
        results = []
        if not os.path.isdir(self._storage_dir):
            return results
        for fname in os.listdir(self._storage_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self._storage_dir, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                cp = ExecutionCheckpoint.from_dict(data)
                if task_id is None or cp.task_id == task_id:
                    results.append(cp)
            except (json.JSONDecodeError, KeyError):
                continue
        results.sort(key=lambda c: c.timestamp)
        return results

    def get_latest(self, task_id: str) -> Optional[ExecutionCheckpoint]:
        """Get the most recent checkpoint for a task."""
        cps = self.list_checkpoints(task_id)
        return cps[-1] if cps else None

    def count_checkpoints(self, task_id: Optional[str] = None) -> int:
        return len(self.list_checkpoints(task_id))

    # ── eviction ────────────────────────────────────────────────

    def _evict_oldest(self, task_id: str) -> int:
        """Keep only MAX_CHECKPOINTS_PER_TASK for *task_id*."""
        cps = self.list_checkpoints(task_id)
        removed = 0
        while len(cps) > self.MAX_CHECKPOINTS_PER_TASK:
            oldest = cps.pop(0)
            self.delete_checkpoint(oldest.checkpoint_id)
            removed += 1
        return removed

    def cleanup_all(self) -> int:
        """Remove all checkpoints.  Returns count removed."""
        removed = 0
        for cp in self.list_checkpoints():
            self.delete_checkpoint(cp.checkpoint_id)
            removed += 1
        return removed
