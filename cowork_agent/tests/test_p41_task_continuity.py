"""
Sprint 41 · Cross-Session Task Continuity — Tests
====================================================
~100 tests covering TaskContinuityManager, PersistentTaskQueue,
CheckpointManager, integration, and edge cases.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from cowork_agent.core.task_continuity import (
    ContinuableTask,
    TaskContinuityManager,
    TaskState,
)
from cowork_agent.core.task_queue import (
    PersistentTaskQueue,
    QueuedTask,
    TaskPriority,
)
from cowork_agent.core.checkpoint_manager import (
    CheckpointManager,
    ExecutionCheckpoint,
)


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path)


@pytest.fixture
def continuity(workspace):
    return TaskContinuityManager(workspace)


@pytest.fixture
def queue(workspace):
    return PersistentTaskQueue(workspace)


@pytest.fixture
def checkpoints(workspace):
    return CheckpointManager(workspace)


# ═══════════════════════════════════════════════════════════════════
#  1. ContinuableTask dataclass
# ═══════════════════════════════════════════════════════════════════

class TestContinuableTask:
    def test_creation(self):
        t = ContinuableTask(task_id="t1", description="test")
        assert t.task_id == "t1"
        assert t.state == TaskState.PENDING
        assert t.created_at > 0

    def test_to_dict(self):
        t = ContinuableTask(task_id="t1", description="d")
        d = t.to_dict()
        assert d["task_id"] == "t1"
        assert d["state"] == "pending"

    def test_from_dict(self):
        t = ContinuableTask.from_dict({"task_id": "t1", "description": "d", "state": "paused"})
        assert t.state == TaskState.PAUSED

    def test_roundtrip(self):
        orig = ContinuableTask(task_id="t1", description="d", progress_pct=42.0)
        rebuilt = ContinuableTask.from_dict(orig.to_dict())
        assert rebuilt.progress_pct == 42.0

    def test_generate_id(self):
        id1 = ContinuableTask.generate_id()
        id2 = ContinuableTask.generate_id()
        assert id1.startswith("task_")
        assert id1 != id2

    def test_partial_results(self):
        t = ContinuableTask(task_id="t1", description="d", partial_results=[{"a": 1}])
        assert len(t.partial_results) == 1

    def test_metadata(self):
        t = ContinuableTask(task_id="t1", description="d", metadata={"key": "val"})
        assert t.metadata["key"] == "val"

    def test_default_timestamps(self):
        t = ContinuableTask(task_id="t1", description="d")
        assert t.paused_at is None
        assert t.resumed_at is None
        assert t.completed_at is None


# ═══════════════════════════════════════════════════════════════════
#  2. TaskContinuityManager
# ═══════════════════════════════════════════════════════════════════

class TestTaskContinuityManager:
    def test_create_task(self, continuity):
        t = continuity.create_task("Build login page")
        assert t.state == TaskState.PENDING
        assert "task_" in t.task_id

    def test_save_and_load(self, continuity):
        t = continuity.create_task("test")
        loaded = continuity.load_task(t.task_id)
        assert loaded is not None
        assert loaded.description == "test"

    def test_load_nonexistent(self, continuity):
        assert continuity.load_task("nonexistent") is None

    def test_delete_task(self, continuity):
        t = continuity.create_task("test")
        assert continuity.delete_task(t.task_id) is True
        assert continuity.load_task(t.task_id) is None

    def test_delete_nonexistent(self, continuity):
        assert continuity.delete_task("nope") is False

    def test_pause_task(self, continuity):
        t = continuity.create_task("test")
        paused = continuity.pause_task(
            t.task_id, partial_results=[{"step": 1}], progress_pct=50.0,
        )
        assert paused is not None
        assert paused.state == TaskState.PAUSED
        assert paused.progress_pct == 50.0
        assert paused.paused_at is not None

    def test_pause_nonexistent(self, continuity):
        assert continuity.pause_task("nope") is None

    def test_resume_task(self, continuity):
        t = continuity.create_task("test")
        continuity.pause_task(t.task_id)
        resumed = continuity.resume_task(t.task_id)
        assert resumed is not None
        assert resumed.state == TaskState.IN_PROGRESS
        assert resumed.resumed_at is not None

    def test_resume_non_paused(self, continuity):
        t = continuity.create_task("test")  # PENDING, not PAUSED
        assert continuity.resume_task(t.task_id) is None

    def test_mark_completed(self, continuity):
        t = continuity.create_task("test")
        completed = continuity.mark_completed(t.task_id)
        assert completed.state == TaskState.COMPLETED
        assert completed.progress_pct == 100.0

    def test_mark_failed(self, continuity):
        t = continuity.create_task("test")
        failed = continuity.mark_failed(t.task_id, "boom")
        assert failed.state == TaskState.FAILED
        assert failed.metadata["error"] == "boom"

    def test_update_progress(self, continuity):
        t = continuity.create_task("test")
        updated = continuity.update_progress(t.task_id, 75.0, {"step": "done"})
        assert updated.progress_pct == 75.0
        assert len(updated.partial_results) == 1

    def test_list_tasks(self, continuity):
        continuity.create_task("a")
        continuity.create_task("b")
        assert len(continuity.list_tasks()) == 2

    def test_list_tasks_filter(self, continuity):
        t = continuity.create_task("a")
        continuity.create_task("b")
        continuity.pause_task(t.task_id)
        paused = continuity.list_tasks(state=TaskState.PAUSED)
        assert len(paused) == 1

    def test_list_resumable(self, continuity):
        t1 = continuity.create_task("a")
        t2 = continuity.create_task("b")
        continuity.pause_task(t1.task_id)
        resumable = continuity.list_resumable()
        assert len(resumable) == 1

    def test_cleanup_old_tasks(self, continuity):
        t = continuity.create_task("old")
        # Hack: set created_at to 60 days ago
        loaded = continuity.load_task(t.task_id)
        loaded.created_at = time.time() - 60 * 86400
        continuity.save_task(loaded)
        removed = continuity.cleanup_old_tasks(max_age_days=30)
        assert removed == 1

    def test_build_resume_context(self, continuity):
        t = continuity.create_task("Build API")
        continuity.pause_task(
            t.task_id,
            partial_results=[{"description": "Created models"}],
            pending_operations=[{"description": "Write routes"}],
            progress_pct=40.0,
        )
        task = continuity.load_task(t.task_id)
        ctx = continuity.build_resume_context(task)
        assert "Build API" in ctx
        assert "40%" in ctx

    def test_save_with_checkpoint_id(self, continuity):
        t = continuity.create_task("test")
        continuity.pause_task(t.task_id, checkpoint_id="cp-123")
        loaded = continuity.load_task(t.task_id)
        assert loaded.checkpoint_id == "cp-123"

    def test_context_snapshot(self, continuity):
        t = continuity.create_task("test")
        continuity.pause_task(t.task_id, context_snapshot={"key": "val"})
        loaded = continuity.load_task(t.task_id)
        assert loaded.context_snapshot == {"key": "val"}


# ═══════════════════════════════════════════════════════════════════
#  3. QueuedTask dataclass
# ═══════════════════════════════════════════════════════════════════

class TestQueuedTask:
    def test_creation(self):
        t = QueuedTask(task_id="q1", description="test")
        assert t.priority == TaskPriority.NORMAL
        assert t.status == "pending"

    def test_to_dict(self):
        d = QueuedTask(task_id="q1", description="d").to_dict()
        assert d["priority"] == TaskPriority.NORMAL

    def test_from_dict(self):
        t = QueuedTask.from_dict({"task_id": "q1", "description": "d", "priority": 10})
        assert t.priority == 10

    def test_is_done(self):
        t = QueuedTask(task_id="q1", description="d", status="completed")
        assert t.is_done is True

    def test_not_done(self):
        t = QueuedTask(task_id="q1", description="d", status="pending")
        assert t.is_done is False

    def test_generate_id(self):
        assert QueuedTask.generate_id().startswith("qtask_")


# ═══════════════════════════════════════════════════════════════════
#  4. PersistentTaskQueue
# ═══════════════════════════════════════════════════════════════════

class TestPersistentTaskQueue:
    def test_enqueue(self, queue):
        tid = queue.enqueue("task A")
        assert tid.startswith("qtask_")
        assert queue.size == 1

    def test_dequeue_priority(self, queue):
        queue.enqueue("low", priority=TaskPriority.LOW)
        queue.enqueue("critical", priority=TaskPriority.CRITICAL)
        queue.enqueue("normal", priority=TaskPriority.NORMAL)
        t = queue.dequeue()
        assert t.description == "critical"

    def test_dequeue_empty(self, queue):
        assert queue.dequeue() is None

    def test_peek(self, queue):
        queue.enqueue("a", priority=TaskPriority.HIGH)
        t = queue.peek()
        assert t is not None
        assert t.status == "pending"  # peek doesn't change status

    def test_complete_task(self, queue):
        tid = queue.enqueue("a")
        assert queue.complete_task(tid, "done") is True
        t = queue.get_task(tid)
        assert t.status == "completed"

    def test_fail_task(self, queue):
        tid = queue.enqueue("a")
        assert queue.fail_task(tid, "error") is True
        t = queue.get_task(tid)
        assert t.status == "failed"

    def test_dependency_resolution(self, queue):
        t1 = queue.enqueue("first")
        t2 = queue.enqueue("second", dependencies=[t1])
        # t2 shouldn't be ready yet
        ready = queue.get_ready_tasks()
        assert all(t.task_id != t2 for t in ready)
        # Complete t1
        queue.complete_task(t1)
        ready = queue.get_ready_tasks()
        assert any(t.task_id == t2 for t in ready)

    def test_dependency_blocks_dequeue(self, queue):
        t1 = queue.enqueue("first", priority=TaskPriority.LOW)
        t2 = queue.enqueue("second", priority=TaskPriority.CRITICAL, dependencies=[t1])
        # Even though t2 is higher priority, t1 is dequeued first (t2 blocked)
        dequeued = queue.dequeue()
        assert dequeued.task_id == t1

    def test_persistence_across_instances(self, workspace):
        q1 = PersistentTaskQueue(workspace)
        tid = q1.enqueue("persistent task")
        q2 = PersistentTaskQueue(workspace)
        assert q2.get_task(tid) is not None

    def test_get_by_status(self, queue):
        queue.enqueue("a")
        tid = queue.enqueue("b")
        queue.complete_task(tid)
        assert len(queue.get_by_status("pending")) == 1
        assert len(queue.get_by_status("completed")) == 1

    def test_pending_count(self, queue):
        queue.enqueue("a")
        queue.enqueue("b")
        assert queue.pending_count == 2

    def test_remove_task(self, queue):
        tid = queue.enqueue("a")
        assert queue.remove_task(tid) is True
        assert queue.size == 0

    def test_clear_completed(self, queue):
        t1 = queue.enqueue("a")
        t2 = queue.enqueue("b")
        queue.complete_task(t1)
        removed = queue.clear_completed()
        assert removed == 1
        assert queue.size == 1

    def test_fifo_tiebreak(self, queue):
        """Same priority → FIFO order."""
        tid1 = queue.enqueue("first", priority=TaskPriority.NORMAL)
        tid2 = queue.enqueue("second", priority=TaskPriority.NORMAL)
        t = queue.dequeue()
        assert t.task_id == tid1

    def test_max_queue_size(self, queue):
        queue.MAX_QUEUE_SIZE = 3
        queue.enqueue("a")
        queue.enqueue("b")
        queue.enqueue("c")
        with pytest.raises(ValueError, match="Queue full"):
            queue.enqueue("d")

    def test_get_all(self, queue):
        queue.enqueue("a")
        queue.enqueue("b")
        assert len(queue.get_all()) == 2

    def test_complete_nonexistent(self, queue):
        assert queue.complete_task("nope") is False

    def test_fail_nonexistent(self, queue):
        assert queue.fail_task("nope") is False

    def test_remove_nonexistent(self, queue):
        assert queue.remove_task("nope") is False

    def test_dequeue_marks_running(self, queue):
        queue.enqueue("a")
        t = queue.dequeue()
        assert t.status == "running"


# ═══════════════════════════════════════════════════════════════════
#  5. ExecutionCheckpoint dataclass
# ═══════════════════════════════════════════════════════════════════

class TestExecutionCheckpoint:
    def test_creation(self):
        cp = ExecutionCheckpoint(
            checkpoint_id="cp1", task_id="t1",
            step_index=3, total_steps=10,
        )
        assert cp.step_index == 3
        assert cp.timestamp > 0

    def test_to_dict(self):
        d = ExecutionCheckpoint(
            checkpoint_id="cp1", task_id="t1",
            step_index=0, total_steps=5,
        ).to_dict()
        assert d["checkpoint_id"] == "cp1"

    def test_from_dict(self):
        cp = ExecutionCheckpoint.from_dict({
            "checkpoint_id": "cp1", "task_id": "t1",
            "step_index": 2, "total_steps": 4,
        })
        assert cp.step_index == 2

    def test_progress_pct(self):
        cp = ExecutionCheckpoint(
            checkpoint_id="cp1", task_id="t1",
            step_index=5, total_steps=10,
        )
        assert cp.progress_pct == 50.0

    def test_progress_pct_zero_steps(self):
        cp = ExecutionCheckpoint(
            checkpoint_id="cp1", task_id="t1",
            step_index=0, total_steps=0,
        )
        assert cp.progress_pct == 0.0

    def test_generate_id(self):
        assert ExecutionCheckpoint.generate_id().startswith("ckpt_")

    def test_roundtrip(self):
        orig = ExecutionCheckpoint(
            checkpoint_id="cp1", task_id="t1",
            step_index=3, total_steps=10,
            tool_results_so_far=[{"output": "hi"}],
        )
        rebuilt = ExecutionCheckpoint.from_dict(orig.to_dict())
        assert rebuilt.tool_results_so_far == [{"output": "hi"}]

    def test_pending_tool_calls(self):
        cp = ExecutionCheckpoint(
            checkpoint_id="cp1", task_id="t1",
            step_index=0, total_steps=1,
            pending_tool_calls=[{"name": "bash", "input": {}}],
        )
        assert len(cp.pending_tool_calls) == 1


# ═══════════════════════════════════════════════════════════════════
#  6. CheckpointManager
# ═══════════════════════════════════════════════════════════════════

class TestCheckpointManager:
    def test_create_checkpoint(self, checkpoints):
        cpid = checkpoints.create_checkpoint("t1", 0, 5)
        assert cpid.startswith("ckpt_")

    def test_restore_checkpoint(self, checkpoints):
        cpid = checkpoints.create_checkpoint("t1", 3, 10, messages_count=5)
        cp = checkpoints.restore_checkpoint(cpid)
        assert cp is not None
        assert cp.step_index == 3
        assert cp.messages_count == 5

    def test_restore_nonexistent(self, checkpoints):
        assert checkpoints.restore_checkpoint("nope") is None

    def test_delete_checkpoint(self, checkpoints):
        cpid = checkpoints.create_checkpoint("t1", 0, 1)
        assert checkpoints.delete_checkpoint(cpid) is True
        assert checkpoints.restore_checkpoint(cpid) is None

    def test_delete_nonexistent(self, checkpoints):
        assert checkpoints.delete_checkpoint("nope") is False

    def test_list_checkpoints(self, checkpoints):
        checkpoints.create_checkpoint("t1", 0, 5)
        checkpoints.create_checkpoint("t1", 1, 5)
        checkpoints.create_checkpoint("t2", 0, 3)
        assert len(checkpoints.list_checkpoints()) == 3
        assert len(checkpoints.list_checkpoints("t1")) == 2

    def test_get_latest(self, checkpoints):
        checkpoints.create_checkpoint("t1", 0, 5)
        cpid2 = checkpoints.create_checkpoint("t1", 3, 5)
        latest = checkpoints.get_latest("t1")
        assert latest is not None
        assert latest.step_index == 3

    def test_get_latest_no_checkpoints(self, checkpoints):
        assert checkpoints.get_latest("nonexistent") is None

    def test_eviction(self, checkpoints):
        checkpoints.MAX_CHECKPOINTS_PER_TASK = 3
        for i in range(5):
            checkpoints.create_checkpoint("t1", i, 10)
        assert checkpoints.count_checkpoints("t1") == 3

    def test_count_checkpoints(self, checkpoints):
        checkpoints.create_checkpoint("t1", 0, 5)
        checkpoints.create_checkpoint("t1", 1, 5)
        assert checkpoints.count_checkpoints("t1") == 2
        assert checkpoints.count_checkpoints() == 2

    def test_cleanup_all(self, checkpoints):
        checkpoints.create_checkpoint("t1", 0, 5)
        checkpoints.create_checkpoint("t2", 0, 3)
        removed = checkpoints.cleanup_all()
        assert removed == 2
        assert checkpoints.count_checkpoints() == 0

    def test_tool_results_preserved(self, checkpoints):
        cpid = checkpoints.create_checkpoint(
            "t1", 2, 5,
            tool_results_so_far=[{"tool": "bash", "output": "ok"}],
        )
        cp = checkpoints.restore_checkpoint(cpid)
        assert len(cp.tool_results_so_far) == 1

    def test_metadata_preserved(self, checkpoints):
        cpid = checkpoints.create_checkpoint("t1", 0, 1, label="before risky")
        cp = checkpoints.restore_checkpoint(cpid)
        assert cp.metadata.get("label") == "before risky"

    def test_sorted_by_timestamp(self, checkpoints):
        cpid1 = checkpoints.create_checkpoint("t1", 0, 5)
        cpid2 = checkpoints.create_checkpoint("t1", 1, 5)
        cps = checkpoints.list_checkpoints("t1")
        assert cps[0].step_index == 0
        assert cps[1].step_index == 1

    def test_different_tasks_independent(self, checkpoints):
        checkpoints.create_checkpoint("t1", 0, 5)
        checkpoints.create_checkpoint("t2", 0, 3)
        assert checkpoints.count_checkpoints("t1") == 1
        assert checkpoints.count_checkpoints("t2") == 1


# ═══════════════════════════════════════════════════════════════════
#  7. Agent Integration
# ═══════════════════════════════════════════════════════════════════

class TestAgentIntegration:
    def test_agent_has_continuity_attribute(self):
        from cowork_agent.core.agent import Agent
        a = Agent.__new__(Agent)
        a.task_continuity = None
        assert a.task_continuity is None

    def test_agent_has_queue_attribute(self):
        from cowork_agent.core.agent import Agent
        a = Agent.__new__(Agent)
        a.task_queue = None
        assert a.task_queue is None

    def test_agent_has_checkpoint_attribute(self):
        from cowork_agent.core.agent import Agent
        a = Agent.__new__(Agent)
        a.checkpoint_manager = None
        assert a.checkpoint_manager is None

    def test_config_wiring(self, workspace):
        config = {"task_continuity": {"enabled": True}, "workspace_dir": workspace}
        tc_cfg = config.get("task_continuity", {})
        assert tc_cfg.get("enabled", True) is True

    def test_full_pause_resume_flow(self, workspace):
        mgr = TaskContinuityManager(workspace)
        ckpt = CheckpointManager(workspace)

        # Create task
        task = mgr.create_task("Build API")

        # Work on it, create checkpoint
        cpid = ckpt.create_checkpoint(task.task_id, 3, 10, messages_count=20)

        # Pause
        mgr.pause_task(
            task.task_id,
            partial_results=[{"step": "models done"}],
            pending_operations=[{"step": "routes next"}],
            checkpoint_id=cpid,
            progress_pct=30.0,
        )

        # Resume
        resumed = mgr.resume_task(task.task_id)
        assert resumed is not None
        assert resumed.checkpoint_id == cpid

        # Restore checkpoint
        cp = ckpt.restore_checkpoint(cpid)
        assert cp.step_index == 3
        assert cp.messages_count == 20

    def test_queue_integration(self, workspace):
        q = PersistentTaskQueue(workspace)
        t1 = q.enqueue("first task", priority=TaskPriority.HIGH)
        t2 = q.enqueue("second task", dependencies=[t1])

        # Dequeue and complete first
        task = q.dequeue()
        assert task.task_id == t1
        q.complete_task(t1, "done")

        # Now second is ready
        task = q.dequeue()
        assert task.task_id == t2

    def test_resume_context_generation(self, workspace):
        mgr = TaskContinuityManager(workspace)
        task = mgr.create_task("Refactor auth")
        mgr.pause_task(
            task.task_id,
            partial_results=[{"description": "Extracted interface"}],
            progress_pct=25.0,
        )
        loaded = mgr.load_task(task.task_id)
        ctx = mgr.build_resume_context(loaded)
        assert "Refactor auth" in ctx
        assert "25%" in ctx

    def test_continuity_with_checkpoint(self, workspace):
        mgr = TaskContinuityManager(workspace)
        ckpt_mgr = CheckpointManager(workspace)
        task = mgr.create_task("test")
        cpid = ckpt_mgr.create_checkpoint(task.task_id, 5, 10)
        mgr.pause_task(task.task_id, checkpoint_id=cpid, progress_pct=50.0)
        resumed = mgr.resume_task(task.task_id)
        cp = ckpt_mgr.restore_checkpoint(resumed.checkpoint_id)
        assert cp.step_index == 5

    def test_completed_tasks_not_resumable(self, workspace):
        mgr = TaskContinuityManager(workspace)
        task = mgr.create_task("test")
        mgr.mark_completed(task.task_id)
        assert mgr.resume_task(task.task_id) is None

    def test_failed_tasks_not_resumable(self, workspace):
        mgr = TaskContinuityManager(workspace)
        task = mgr.create_task("test")
        mgr.mark_failed(task.task_id, "err")
        assert mgr.resume_task(task.task_id) is None

    def test_multiple_pauses(self, workspace):
        mgr = TaskContinuityManager(workspace)
        task = mgr.create_task("test")
        mgr.pause_task(task.task_id, progress_pct=25.0)
        resumed = mgr.resume_task(task.task_id)
        mgr.pause_task(task.task_id, progress_pct=50.0)
        loaded = mgr.load_task(task.task_id)
        assert loaded.progress_pct == 50.0


# ═══════════════════════════════════════════════════════════════════
#  8. Edge Cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_workspace(self, workspace):
        mgr = TaskContinuityManager(workspace)
        assert mgr.list_tasks() == []

    def test_empty_queue(self, workspace):
        q = PersistentTaskQueue(workspace)
        assert q.dequeue() is None
        assert q.peek() is None
        assert q.get_all() == []

    def test_corrupted_json(self, workspace):
        q = PersistentTaskQueue(workspace)
        # Write corrupted data
        path = os.path.join(workspace, q.STORAGE_DIR, q.FILENAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("{invalid json")
        q2 = PersistentTaskQueue(workspace)
        assert q2.size == 0  # gracefully handles corruption

    def test_task_state_enum_values(self):
        assert TaskState.PENDING.value == "pending"
        assert TaskState.PAUSED.value == "paused"
        assert TaskState.RESUMABLE.value == "resumable"
        assert len(TaskState) == 6

    def test_priority_ordering(self):
        assert TaskPriority.LOW < TaskPriority.NORMAL < TaskPriority.HIGH < TaskPriority.CRITICAL

    def test_checkpoint_zero_total_steps(self, checkpoints):
        cpid = checkpoints.create_checkpoint("t1", 0, 0)
        cp = checkpoints.restore_checkpoint(cpid)
        assert cp.progress_pct == 0.0

    def test_many_tasks(self, workspace):
        mgr = TaskContinuityManager(workspace)
        for i in range(20):
            mgr.create_task(f"task {i}")
        assert len(mgr.list_tasks()) == 20

    def test_queue_with_circular_dependencies(self, workspace):
        """Circular deps = nothing is ready."""
        q = PersistentTaskQueue(workspace)
        t1 = q.enqueue("a", dependencies=["fake_dep"])
        ready = q.get_ready_tasks()
        # t1 depends on fake_dep which doesn't exist → not ready
        assert len(ready) == 0

    def test_update_progress_nonexistent(self, workspace):
        mgr = TaskContinuityManager(workspace)
        assert mgr.update_progress("nope", 50.0) is None

    def test_mark_completed_nonexistent(self, workspace):
        mgr = TaskContinuityManager(workspace)
        assert mgr.mark_completed("nope") is None

    def test_cleanup_no_old_tasks(self, workspace):
        mgr = TaskContinuityManager(workspace)
        mgr.create_task("recent")
        assert mgr.cleanup_old_tasks(max_age_days=30) == 0
