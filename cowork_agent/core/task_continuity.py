"""
Sprint 41 · Cross-Session Task Continuity
==========================================
Allows users to pause a complex task, leave, and resume later
from exactly where they left off.

Storage layout:
    {workspace}/.cowork/continuity/{task_id}.json
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enums & Data ────────────────────────────────────────────────────

class TaskState(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    RESUMABLE = "resumable"


@dataclass
class ContinuableTask:
    """A task that can be paused and resumed across sessions."""
    task_id: str
    description: str
    state: TaskState = TaskState.PENDING
    created_at: float = 0.0
    paused_at: Optional[float] = None
    resumed_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress_pct: float = 0.0
    partial_results: List[Dict[str, Any]] = field(default_factory=list)
    pending_operations: List[Dict[str, Any]] = field(default_factory=list)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    checkpoint_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id,
            "description": self.description,
            "state": self.state.value,
            "created_at": self.created_at,
            "paused_at": self.paused_at,
            "resumed_at": self.resumed_at,
            "completed_at": self.completed_at,
            "progress_pct": self.progress_pct,
            "partial_results": self.partial_results,
            "pending_operations": self.pending_operations,
            "context_snapshot": self.context_snapshot,
            "checkpoint_id": self.checkpoint_id,
            "metadata": self.metadata,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ContinuableTask":
        return cls(
            task_id=d["task_id"],
            description=d.get("description", ""),
            state=TaskState(d.get("state", "pending")),
            created_at=d.get("created_at", 0.0),
            paused_at=d.get("paused_at"),
            resumed_at=d.get("resumed_at"),
            completed_at=d.get("completed_at"),
            progress_pct=d.get("progress_pct", 0.0),
            partial_results=d.get("partial_results", []),
            pending_operations=d.get("pending_operations", []),
            context_snapshot=d.get("context_snapshot", {}),
            checkpoint_id=d.get("checkpoint_id"),
            metadata=d.get("metadata", {}),
        )

    @staticmethod
    def generate_id() -> str:
        return f"task_{uuid.uuid4().hex[:12]}"


# ── Manager ─────────────────────────────────────────────────────────

class TaskContinuityManager:
    """
    Manages pausing and resuming tasks across sessions.

    Persists task state as JSON files under ``{workspace}/.cowork/continuity/``.
    """

    STORAGE_DIR = ".cowork/continuity"
    MAX_TASKS = 50

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self._storage_dir = os.path.join(workspace_path, self.STORAGE_DIR)
        os.makedirs(self._storage_dir, exist_ok=True)

    # ── persistence ─────────────────────────────────────────────

    def _task_path(self, task_id: str) -> str:
        return os.path.join(self._storage_dir, f"{task_id}.json")

    def save_task(self, task: ContinuableTask) -> str:
        """Persist task to disk.  Returns task_id."""
        path = self._task_path(task.task_id)
        with open(path, "w") as f:
            json.dump(task.to_dict(), f, indent=2)
        return task.task_id

    def load_task(self, task_id: str) -> Optional[ContinuableTask]:
        """Load a task from disk.  Returns None if not found."""
        path = self._task_path(task_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return ContinuableTask.from_dict(json.load(f))

    def delete_task(self, task_id: str) -> bool:
        path = self._task_path(task_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    # ── task lifecycle ──────────────────────────────────────────

    def create_task(self, description: str, **metadata) -> ContinuableTask:
        """Create and persist a new task."""
        task = ContinuableTask(
            task_id=ContinuableTask.generate_id(),
            description=description,
            state=TaskState.PENDING,
            metadata=metadata,
        )
        self.save_task(task)
        return task

    def pause_task(
        self,
        task_id: str,
        partial_results: Optional[List[dict]] = None,
        pending_operations: Optional[List[dict]] = None,
        context_snapshot: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
        progress_pct: float = 0.0,
    ) -> Optional[ContinuableTask]:
        """Pause a running task, saving its current state."""
        task = self.load_task(task_id)
        if task is None:
            return None
        task.state = TaskState.PAUSED
        task.paused_at = time.time()
        task.progress_pct = progress_pct
        if partial_results is not None:
            task.partial_results = partial_results
        if pending_operations is not None:
            task.pending_operations = pending_operations
        if context_snapshot is not None:
            task.context_snapshot = context_snapshot
        if checkpoint_id is not None:
            task.checkpoint_id = checkpoint_id
        self.save_task(task)
        return task

    def resume_task(self, task_id: str) -> Optional[ContinuableTask]:
        """Mark a paused task as resumable and return it."""
        task = self.load_task(task_id)
        if task is None:
            return None
        if task.state not in (TaskState.PAUSED, TaskState.RESUMABLE):
            return None
        task.state = TaskState.IN_PROGRESS
        task.resumed_at = time.time()
        self.save_task(task)
        return task

    def mark_completed(self, task_id: str) -> Optional[ContinuableTask]:
        task = self.load_task(task_id)
        if task is None:
            return None
        task.state = TaskState.COMPLETED
        task.completed_at = time.time()
        task.progress_pct = 100.0
        self.save_task(task)
        return task

    def mark_failed(self, task_id: str, error: str = "") -> Optional[ContinuableTask]:
        task = self.load_task(task_id)
        if task is None:
            return None
        task.state = TaskState.FAILED
        task.metadata["error"] = error
        self.save_task(task)
        return task

    def update_progress(
        self,
        task_id: str,
        progress_pct: float,
        partial_result: Optional[dict] = None,
    ) -> Optional[ContinuableTask]:
        """Incrementally update progress."""
        task = self.load_task(task_id)
        if task is None:
            return None
        task.progress_pct = min(progress_pct, 100.0)
        if partial_result is not None:
            task.partial_results.append(partial_result)
        self.save_task(task)
        return task

    # ── queries ─────────────────────────────────────────────────

    def list_tasks(self, state: Optional[TaskState] = None) -> List[ContinuableTask]:
        """List all tasks, optionally filtered by state."""
        tasks = []
        if not os.path.isdir(self._storage_dir):
            return tasks
        for fname in sorted(os.listdir(self._storage_dir)):
            if not fname.endswith(".json"):
                continue
            tid = fname[:-5]
            task = self.load_task(tid)
            if task is None:
                continue
            if state is not None and task.state != state:
                continue
            tasks.append(task)
        return tasks

    def list_resumable(self) -> List[ContinuableTask]:
        """Convenience: tasks that can be resumed."""
        return [
            t for t in self.list_tasks()
            if t.state in (TaskState.PAUSED, TaskState.RESUMABLE)
        ]

    # ── cleanup ─────────────────────────────────────────────────

    def cleanup_old_tasks(self, max_age_days: int = 30) -> int:
        """Remove tasks older than *max_age_days*.  Returns count deleted."""
        cutoff = time.time() - (max_age_days * 86400)
        removed = 0
        for task in self.list_tasks():
            if task.created_at < cutoff:
                self.delete_task(task.task_id)
                removed += 1
        return removed

    def build_resume_context(self, task: ContinuableTask) -> str:
        """Build a text summary for injecting into agent messages on resume."""
        lines = [
            f"Resuming task: {task.description}",
            f"Progress: {task.progress_pct:.0f}%",
        ]
        if task.partial_results:
            lines.append(f"Partial results so far: {len(task.partial_results)} item(s)")
            for i, r in enumerate(task.partial_results[-3:], 1):
                desc = r.get("description", r.get("output", str(r))[:100])
                lines.append(f"  {i}. {desc}")
        if task.pending_operations:
            lines.append(f"Pending operations: {len(task.pending_operations)}")
            for op in task.pending_operations[:3]:
                lines.append(f"  - {op.get('description', str(op)[:80])}")
        return "\n".join(lines)
