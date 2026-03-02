"""
Sprint 41 · Persistent Task Queue
===================================
Priority-based task queue with dependency tracking.
Persists to JSON so it survives session restarts.

Storage: {workspace}/.cowork/task_queue/queue.json
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


# ── Priority ────────────────────────────────────────────────────────

class TaskPriority(IntEnum):
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


# ── Data ────────────────────────────────────────────────────────────

@dataclass
class QueuedTask:
    """A single task in the persistent queue."""
    task_id: str
    description: str
    priority: int = TaskPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    created_at: float = 0.0
    status: str = "pending"  # pending | running | completed | failed
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "created_at": self.created_at,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QueuedTask":
        return cls(
            task_id=d["task_id"],
            description=d.get("description", ""),
            priority=d.get("priority", TaskPriority.NORMAL),
            dependencies=d.get("dependencies", []),
            created_at=d.get("created_at", 0.0),
            status=d.get("status", "pending"),
            result=d.get("result"),
            error=d.get("error"),
            metadata=d.get("metadata", {}),
        )

    @staticmethod
    def generate_id() -> str:
        return f"qtask_{uuid.uuid4().hex[:10]}"

    @property
    def is_done(self) -> bool:
        return self.status in ("completed", "failed")


# ── Queue ───────────────────────────────────────────────────────────

class PersistentTaskQueue:
    """
    A priority queue with dependency resolution, backed by a JSON file.

    Tasks are dequeued in *priority* order (highest first), but only
    if all their dependencies are satisfied (completed).
    """

    STORAGE_DIR = ".cowork/task_queue"
    FILENAME = "queue.json"
    MAX_QUEUE_SIZE = 200

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self._storage_dir = os.path.join(workspace_path, self.STORAGE_DIR)
        os.makedirs(self._storage_dir, exist_ok=True)
        self._queue_path = os.path.join(self._storage_dir, self.FILENAME)
        self._tasks: Dict[str, QueuedTask] = {}
        self.load()

    # ── persistence ─────────────────────────────────────────────

    def save(self) -> None:
        data = [t.to_dict() for t in self._tasks.values()]
        with open(self._queue_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        self._tasks.clear()
        if os.path.exists(self._queue_path):
            try:
                with open(self._queue_path) as f:
                    data = json.load(f)
                for item in data:
                    t = QueuedTask.from_dict(item)
                    self._tasks[t.task_id] = t
            except (json.JSONDecodeError, KeyError):
                pass  # corrupted file → start empty

    # ── enqueue / dequeue ───────────────────────────────────────

    def enqueue(
        self,
        description: str,
        priority: int = TaskPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        **metadata,
    ) -> str:
        """Add a task to the queue.  Returns task_id."""
        if len(self._tasks) >= self.MAX_QUEUE_SIZE:
            raise ValueError(f"Queue full ({self.MAX_QUEUE_SIZE} tasks)")
        task = QueuedTask(
            task_id=QueuedTask.generate_id(),
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            metadata=metadata,
        )
        self._tasks[task.task_id] = task
        self.save()
        return task.task_id

    def dequeue(self) -> Optional[QueuedTask]:
        """
        Return the highest-priority *ready* task and mark it ``running``.
        A task is ready when all its dependencies are ``completed``.
        Returns ``None`` if nothing is ready.
        """
        ready = self.get_ready_tasks()
        if not ready:
            return None
        # Sort by priority descending, then created_at ascending (FIFO tie-break)
        ready.sort(key=lambda t: (-t.priority, t.created_at))
        task = ready[0]
        task.status = "running"
        self.save()
        return task

    def peek(self) -> Optional[QueuedTask]:
        """Return the next ready task without changing its status."""
        ready = self.get_ready_tasks()
        if not ready:
            return None
        ready.sort(key=lambda t: (-t.priority, t.created_at))
        return ready[0]

    # ── task state transitions ──────────────────────────────────

    def complete_task(self, task_id: str, result: str = "") -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.status = "completed"
        task.result = result
        self.save()
        return True

    def fail_task(self, task_id: str, error: str = "") -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.status = "failed"
        task.error = error
        self.save()
        return True

    # ── queries ─────────────────────────────────────────────────

    def get_task(self, task_id: str) -> Optional[QueuedTask]:
        return self._tasks.get(task_id)

    def get_ready_tasks(self) -> List[QueuedTask]:
        """Return pending tasks whose dependencies are all completed."""
        completed_ids = {
            tid for tid, t in self._tasks.items() if t.status == "completed"
        }
        ready = []
        for t in self._tasks.values():
            if t.status != "pending":
                continue
            deps_satisfied = all(d in completed_ids for d in t.dependencies)
            if deps_satisfied:
                ready.append(t)
        return ready

    def get_all(self) -> List[QueuedTask]:
        return list(self._tasks.values())

    def get_by_status(self, status: str) -> List[QueuedTask]:
        return [t for t in self._tasks.values() if t.status == status]

    @property
    def size(self) -> int:
        return len(self._tasks)

    @property
    def pending_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == "pending")

    # ── removal ─────────────────────────────────────────────────

    def remove_task(self, task_id: str) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            self.save()
            return True
        return False

    def clear_completed(self) -> int:
        """Remove all completed tasks.  Returns count removed."""
        to_remove = [tid for tid, t in self._tasks.items() if t.status == "completed"]
        for tid in to_remove:
            del self._tasks[tid]
        if to_remove:
            self.save()
        return len(to_remove)
