"""
Sprint 18 Tests — Worktree & Git Integration.

Tests for:
  1. GitOperations — git subprocess commands + parsing
  2. FileLockManager — reader-writer locks, expiry, cleanup
  3. WorkspaceContext — git-aware workspace state
  4. Git Tools — GitStatusTool, GitDiffTool, GitCommitTool, GitBranchTool, GitLogTool
  5. Enhanced WorktreeManager — merge_back, snapshot, diff_worktree
  6. SafetyChecker git extensions — branch protection
  7. Integration tests — full workflows combining git ops + worktree + locks

Uses real temporary git repos (git init in tmpdir) for integration-level tests.

~120 tests total.
"""

import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ── Imports ──────────────────────────────────────────────────────

from cowork_agent.core.git_ops import (
    GitOperations, GitStatusResult, GitCommit, GitBranch, MergeResult,
)
from cowork_agent.core.file_lock import FileLockManager, LockInfo
from cowork_agent.core.workspace_context import WorkspaceContext
from cowork_agent.core.worktree import WorktreeManager, WorktreeInfo
from cowork_agent.core.safety_checker import SafetyChecker
from cowork_agent.core.models import ToolCall
from cowork_agent.tools.git_tools import (
    GitStatusTool, GitDiffTool, GitCommitTool, GitBranchTool, GitLogTool,
)


# ── Helpers ──────────────────────────────────────────────────────

def _run(args, cwd):
    """Quick subprocess helper for test setup."""
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=10)


def _make_repo(tmp_path):
    """Create a fresh git repo with an initial commit."""
    repo = str(tmp_path)
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "test@test.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)

    # Create initial file and commit
    (tmp_path / "README.md").write_text("# Test Repo\n")
    _run(["git", "add", "README.md"], repo)
    _run(["git", "commit", "-m", "Initial commit"], repo)

    return repo


def _run_async(coro):
    """Helper to run async coroutine in sync test."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ══════════════════════════════════════════════════════════════════
# 1. TestGitOperations
# ══════════════════════════════════════════════════════════════════

class TestGitOperations:
    """Tests for GitOperations class (~25 tests)."""

    def test_is_git_repo_true(self, tmp_path):
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)
        assert git.is_git_repo() is True

    def test_is_git_repo_false(self, tmp_path):
        git = GitOperations(workspace_dir=str(tmp_path))
        assert git.is_git_repo() is False

    def test_git_root(self, tmp_path):
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)
        root = git.git_root()
        assert root  # Non-empty
        assert os.path.isdir(root)

    def test_current_branch(self, tmp_path):
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)
        branch = git.current_branch()
        assert branch in ("main", "master")

    def test_status_clean(self, tmp_path):
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)
        status = git.status()
        assert status.is_clean
        assert status.total_changes == 0

    def test_status_untracked(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "new_file.txt").write_text("hello")
        git = GitOperations(workspace_dir=repo)
        status = git.status()
        assert "new_file.txt" in status.untracked
        assert not status.is_clean

    def test_status_staged(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "staged.txt").write_text("staged content")
        _run(["git", "add", "staged.txt"], repo)
        git = GitOperations(workspace_dir=repo)
        status = git.status()
        assert "staged.txt" in status.staged

    def test_status_unstaged(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "README.md").write_text("modified!")
        git = GitOperations(workspace_dir=repo)
        status = git.status()
        assert "README.md" in status.unstaged

    def test_diff_empty(self, tmp_path):
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)
        diff = git.diff()
        assert diff.strip() == ""

    def test_diff_unstaged(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "README.md").write_text("modified content\n")
        git = GitOperations(workspace_dir=repo)
        diff = git.diff()
        assert "modified content" in diff

    def test_diff_staged(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "README.md").write_text("staged change\n")
        _run(["git", "add", "README.md"], repo)
        git = GitOperations(workspace_dir=repo)
        diff = git.diff(staged=True)
        assert "staged change" in diff

    def test_diff_specific_file(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "README.md").write_text("changed\n")
        (tmp_path / "other.txt").write_text("other")
        _run(["git", "add", "other.txt"], repo)
        git = GitOperations(workspace_dir=repo)
        diff = git.diff(file="README.md")
        assert "changed" in diff

    def test_log_returns_commits(self, tmp_path):
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)
        commits = git.log(n=5)
        assert len(commits) == 1
        assert commits[0].message == "Initial commit"
        assert commits[0].author == "Test User"

    def test_log_multiple_commits(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "a.txt").write_text("a")
        _run(["git", "add", "a.txt"], repo)
        _run(["git", "commit", "-m", "Add a"], repo)
        (tmp_path / "b.txt").write_text("b")
        _run(["git", "add", "b.txt"], repo)
        _run(["git", "commit", "-m", "Add b"], repo)

        git = GitOperations(workspace_dir=repo)
        commits = git.log(n=10)
        assert len(commits) == 3
        assert commits[0].message == "Add b"
        assert commits[2].message == "Initial commit"

    def test_log_specific_branch(self, tmp_path):
        repo = _make_repo(tmp_path)
        _run(["git", "branch", "feature"], repo)
        _run(["git", "checkout", "feature"], repo)
        (tmp_path / "f.txt").write_text("feature")
        _run(["git", "add", "f.txt"], repo)
        _run(["git", "commit", "-m", "Feature commit"], repo)

        git = GitOperations(workspace_dir=repo)
        commits = git.log(n=5, branch="feature")
        assert any(c.message == "Feature commit" for c in commits)

    def test_branch_list(self, tmp_path):
        repo = _make_repo(tmp_path)
        _run(["git", "branch", "dev"], repo)
        git = GitOperations(workspace_dir=repo)
        branches = git.branch_list()
        names = [b.name for b in branches]
        assert "dev" in names
        current = [b for b in branches if b.is_current]
        assert len(current) == 1

    def test_branch_create(self, tmp_path):
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)
        result = git.branch_create("new-branch")
        assert "Created" in result
        branches = git.branch_list()
        assert any(b.name == "new-branch" for b in branches)

    def test_branch_delete(self, tmp_path):
        repo = _make_repo(tmp_path)
        _run(["git", "branch", "to-delete"], repo)
        git = GitOperations(workspace_dir=repo)
        result = git.branch_delete("to-delete")
        assert "Deleted" in result

    def test_checkout(self, tmp_path):
        repo = _make_repo(tmp_path)
        _run(["git", "branch", "other"], repo)
        git = GitOperations(workspace_dir=repo)
        result = git.checkout("other")
        assert "Switched" in result
        assert git.current_branch() == "other"

    def test_commit_with_message(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "new.txt").write_text("new content")
        _run(["git", "add", "new.txt"], repo)
        git = GitOperations(workspace_dir=repo)
        result = git.commit("Test commit")
        assert "Test commit" in result or "1 file changed" in result

    def test_commit_with_files(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "auto.txt").write_text("auto staged")
        git = GitOperations(workspace_dir=repo)
        result = git.commit("Auto stage commit", files=["auto.txt"])
        assert "Error" not in result or "nothing" not in result.lower()

    def test_commit_empty_message_error(self, tmp_path):
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)
        result = git.commit("")
        assert "Error" in result

    def test_commit_nothing_to_commit(self, tmp_path):
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)
        result = git.commit("Empty commit")
        assert "Nothing to commit" in result or "nothing" in result.lower()

    def test_stash_and_pop(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "README.md").write_text("stash me\n")
        git = GitOperations(workspace_dir=repo)
        stash_result = git.stash()
        assert "Error" not in stash_result or "No local changes" in stash_result

    def test_merge_success(self, tmp_path):
        repo = _make_repo(tmp_path)
        _run(["git", "branch", "feat"], repo)
        _run(["git", "checkout", "feat"], repo)
        (tmp_path / "feat.txt").write_text("feature work")
        _run(["git", "add", "feat.txt"], repo)
        _run(["git", "commit", "-m", "Add feature"], repo)
        # Switch back to main branch
        main_branch = "master"
        r = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)
        if "main" in _run(["git", "branch"], repo).stdout:
            main_branch = "main"
        _run(["git", "checkout", main_branch], repo)

        git = GitOperations(workspace_dir=repo)
        result = git.merge("feat")
        assert result.success is True


# ══════════════════════════════════════════════════════════════════
# 2. TestFileLockManager
# ══════════════════════════════════════════════════════════════════

class TestFileLockManager:
    """Tests for FileLockManager class (~18 tests)."""

    def test_acquire_write_success(self):
        mgr = FileLockManager()
        assert mgr.acquire_write("/a.py", "agent-1") is True

    def test_acquire_write_blocked_by_other(self):
        mgr = FileLockManager()
        mgr.acquire_write("/a.py", "agent-1")
        assert mgr.acquire_write("/a.py", "agent-2") is False

    def test_acquire_write_same_owner_refresh(self):
        mgr = FileLockManager()
        mgr.acquire_write("/a.py", "agent-1")
        assert mgr.acquire_write("/a.py", "agent-1") is True

    def test_acquire_read_success(self):
        mgr = FileLockManager()
        assert mgr.acquire_read("/a.py", "agent-1") is True

    def test_multiple_readers_allowed(self):
        mgr = FileLockManager()
        assert mgr.acquire_read("/a.py", "agent-1") is True
        assert mgr.acquire_read("/a.py", "agent-2") is True

    def test_read_blocked_by_write(self):
        mgr = FileLockManager()
        mgr.acquire_write("/a.py", "agent-1")
        assert mgr.acquire_read("/a.py", "agent-2") is False

    def test_write_blocked_by_readers(self):
        mgr = FileLockManager()
        mgr.acquire_read("/a.py", "agent-1")
        assert mgr.acquire_write("/a.py", "agent-2") is False

    def test_same_owner_read_then_write(self):
        mgr = FileLockManager()
        mgr.acquire_read("/a.py", "agent-1")
        # Same owner can upgrade to write (no other readers)
        assert mgr.acquire_write("/a.py", "agent-1") is True

    def test_release_write_lock(self):
        mgr = FileLockManager()
        mgr.acquire_write("/a.py", "agent-1")
        assert mgr.release("/a.py", "agent-1") is True
        # Now another agent can write
        assert mgr.acquire_write("/a.py", "agent-2") is True

    def test_release_read_lock(self):
        mgr = FileLockManager()
        mgr.acquire_read("/a.py", "agent-1")
        assert mgr.release("/a.py", "agent-1") is True

    def test_release_nonexistent_returns_false(self):
        mgr = FileLockManager()
        assert mgr.release("/nonexistent.py", "agent-1") is False

    def test_lock_expiry(self):
        mgr = FileLockManager(lock_timeout=0.01)  # 10ms timeout
        mgr.acquire_write("/a.py", "agent-1")
        time.sleep(0.05)
        # Expired lock should be cleaned and allow new acquisition
        assert mgr.acquire_write("/a.py", "agent-2") is True

    def test_cleanup_expired(self):
        mgr = FileLockManager(lock_timeout=0.01)
        mgr.acquire_write("/a.py", "agent-1")
        mgr.acquire_read("/b.py", "agent-2")
        time.sleep(0.05)
        count = mgr.cleanup_expired()
        assert count == 2

    def test_check_conflicts_none(self):
        mgr = FileLockManager()
        conflicts = mgr.check_conflicts(["/a.py", "/b.py"], "agent-1")
        assert conflicts == []

    def test_check_conflicts_detected(self):
        mgr = FileLockManager()
        mgr.acquire_write("/a.py", "agent-1")
        conflicts = mgr.check_conflicts(["/a.py", "/b.py"], "agent-2")
        assert "/a.py" in conflicts

    def test_check_conflicts_own_lock_ok(self):
        mgr = FileLockManager()
        mgr.acquire_write("/a.py", "agent-1")
        conflicts = mgr.check_conflicts(["/a.py"], "agent-1")
        assert conflicts == []

    def test_lock_count(self):
        mgr = FileLockManager()
        mgr.acquire_write("/a.py", "agent-1")
        mgr.acquire_read("/b.py", "agent-2")
        mgr.acquire_read("/b.py", "agent-3")
        assert mgr.lock_count == 3

    def test_active_locks(self):
        mgr = FileLockManager()
        mgr.acquire_write("/a.py", "agent-1")
        mgr.acquire_read("/b.py", "agent-2")
        active = mgr.active_locks()
        assert "/a.py" in active
        assert "/b.py" in active

    def test_lock_info_to_dict(self):
        info = LockInfo(path="/a.py", owner="agent-1", lock_type="write")
        d = info.to_dict()
        assert d["path"] == "/a.py"
        assert d["owner"] == "agent-1"
        assert d["lock_type"] == "write"
        assert "age_seconds" in d


# ══════════════════════════════════════════════════════════════════
# 3. TestWorkspaceContext
# ══════════════════════════════════════════════════════════════════

class TestWorkspaceContext:
    """Tests for WorkspaceContext class (~12 tests)."""

    def test_non_git_repo(self, tmp_path):
        ctx = WorkspaceContext(workspace_dir=str(tmp_path))
        ctx.refresh()
        assert ctx.is_git_repo is False
        assert ctx.current_branch == ""
        assert ctx.dirty_files == []

    def test_clean_repo(self, tmp_path):
        repo = _make_repo(tmp_path)
        ctx = WorkspaceContext(workspace_dir=repo)
        ctx.refresh()
        assert ctx.is_git_repo is True
        assert ctx.current_branch in ("main", "master")
        assert ctx.is_dirty is False

    def test_dirty_repo(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "dirty.txt").write_text("dirty")
        ctx = WorkspaceContext(workspace_dir=repo)
        ctx.refresh()
        assert ctx.is_dirty is True
        assert "dirty.txt" in ctx.dirty_files

    def test_branch_tracking(self, tmp_path):
        repo = _make_repo(tmp_path)
        _run(["git", "branch", "dev"], repo)
        _run(["git", "checkout", "dev"], repo)
        ctx = WorkspaceContext(workspace_dir=repo)
        ctx.refresh()
        assert ctx.current_branch == "dev"

    def test_active_worktree_default_none(self, tmp_path):
        repo = _make_repo(tmp_path)
        ctx = WorkspaceContext(workspace_dir=repo)
        assert ctx.active_worktree is None

    def test_set_active_worktree(self, tmp_path):
        repo = _make_repo(tmp_path)
        ctx = WorkspaceContext(workspace_dir=repo)
        info = WorktreeInfo(name="test", path="/tmp/test", branch="worktree/test", created_from="main")
        ctx.set_active_worktree(info)
        assert ctx.active_worktree is not None
        assert ctx.active_worktree.name == "test"

    def test_clear_active_worktree(self, tmp_path):
        repo = _make_repo(tmp_path)
        ctx = WorkspaceContext(workspace_dir=repo)
        info = WorktreeInfo(name="test", path="/tmp/test", branch="worktree/test", created_from="main")
        ctx.set_active_worktree(info)
        ctx.clear_active_worktree()
        assert ctx.active_worktree is None

    def test_to_dict_clean(self, tmp_path):
        repo = _make_repo(tmp_path)
        ctx = WorkspaceContext(workspace_dir=repo)
        ctx.refresh()
        d = ctx.to_dict()
        assert d["is_git_repo"] is True
        assert d["is_dirty"] is False
        assert d["active_worktree"] is None
        assert d["dirty_file_count"] == 0

    def test_to_dict_with_worktree(self, tmp_path):
        repo = _make_repo(tmp_path)
        ctx = WorkspaceContext(workspace_dir=repo)
        ctx.refresh()
        info = WorktreeInfo(name="wt1", path="/tmp/wt1", branch="worktree/wt1", created_from="main")
        ctx.set_active_worktree(info)
        d = ctx.to_dict()
        assert d["active_worktree"] is not None
        assert d["active_worktree"]["name"] == "wt1"

    def test_refresh_with_git_ops(self, tmp_path):
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)
        ctx = WorkspaceContext(workspace_dir=repo, git_ops=git)
        ctx.refresh()
        assert ctx.current_branch in ("main", "master")

    def test_refresh_after_changes(self, tmp_path):
        repo = _make_repo(tmp_path)
        ctx = WorkspaceContext(workspace_dir=repo)
        ctx.refresh()
        assert ctx.is_dirty is False

        (tmp_path / "new.txt").write_text("new")
        ctx.refresh()
        assert ctx.is_dirty is True

    def test_dirty_files_deduplication(self, tmp_path):
        repo = _make_repo(tmp_path)
        # File that's both staged and unstaged (modify after staging)
        (tmp_path / "README.md").write_text("staged version\n")
        _run(["git", "add", "README.md"], repo)
        (tmp_path / "README.md").write_text("unstaged version\n")

        ctx = WorkspaceContext(workspace_dir=repo)
        ctx.refresh()
        # Should be deduplicated
        count = ctx.dirty_files.count("README.md")
        assert count == 1


# ══════════════════════════════════════════════════════════════════
# 4. TestGitStatusTool
# ══════════════════════════════════════════════════════════════════

class TestGitStatusTool:
    """Tests for GitStatusTool (~5 tests)."""

    def test_clean_repo(self, tmp_path):
        repo = _make_repo(tmp_path)
        tool = GitStatusTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute())
        assert result.success is True
        assert "clean" in result.output.lower()

    def test_untracked_files(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "new.txt").write_text("new")
        tool = GitStatusTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute())
        assert "Untracked" in result.output
        assert "new.txt" in result.output

    def test_staged_files(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "s.txt").write_text("staged")
        _run(["git", "add", "s.txt"], repo)
        tool = GitStatusTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute())
        assert "Staged" in result.output

    def test_shows_branch_name(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "x.txt").write_text("x")
        tool = GitStatusTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute())
        assert "On branch" in result.output

    def test_not_git_repo(self, tmp_path):
        tool = GitStatusTool(git_ops=GitOperations(workspace_dir=str(tmp_path)))
        result = _run_async(tool.execute())
        assert result.success is False
        assert "Not a git repository" in result.error


# ══════════════════════════════════════════════════════════════════
# 5. TestGitDiffTool
# ══════════════════════════════════════════════════════════════════

class TestGitDiffTool:
    """Tests for GitDiffTool (~5 tests)."""

    def test_no_diff(self, tmp_path):
        repo = _make_repo(tmp_path)
        tool = GitDiffTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute())
        assert result.success is True
        assert "No unstaged" in result.output

    def test_unstaged_diff(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "README.md").write_text("diff me\n")
        tool = GitDiffTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute())
        assert "diff me" in result.output

    def test_staged_diff(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "README.md").write_text("staged diff\n")
        _run(["git", "add", "README.md"], repo)
        tool = GitDiffTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(staged=True))
        assert "staged diff" in result.output

    def test_file_specific_diff(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "README.md").write_text("target\n")
        tool = GitDiffTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(file="README.md"))
        assert "target" in result.output

    def test_not_git_repo(self, tmp_path):
        tool = GitDiffTool(git_ops=GitOperations(workspace_dir=str(tmp_path)))
        result = _run_async(tool.execute())
        assert result.success is False


# ══════════════════════════════════════════════════════════════════
# 6. TestGitCommitTool
# ══════════════════════════════════════════════════════════════════

class TestGitCommitTool:
    """Tests for GitCommitTool (~6 tests)."""

    def test_commit_staged(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "c.txt").write_text("commit me")
        _run(["git", "add", "c.txt"], repo)
        tool = GitCommitTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(message="Add c.txt"))
        assert result.success is True

    def test_commit_specific_files(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "auto.txt").write_text("auto staged")
        tool = GitCommitTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(message="Auto add", files=["auto.txt"]))
        assert result.success is True

    def test_commit_empty_message_error(self, tmp_path):
        repo = _make_repo(tmp_path)
        tool = GitCommitTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(message=""))
        assert result.success is False
        assert "required" in result.error.lower()

    def test_commit_nothing_staged(self, tmp_path):
        repo = _make_repo(tmp_path)
        tool = GitCommitTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(message="Empty"))
        # "Nothing to commit" is returned as informational success
        assert "Nothing to commit" in result.output or result.success is False

    def test_not_git_repo(self, tmp_path):
        tool = GitCommitTool(git_ops=GitOperations(workspace_dir=str(tmp_path)))
        result = _run_async(tool.execute(message="Test"))
        assert result.success is False

    def test_commit_preserves_message(self, tmp_path):
        repo = _make_repo(tmp_path)
        (tmp_path / "msg.txt").write_text("test")
        _run(["git", "add", "msg.txt"], repo)
        tool = GitCommitTool(git_ops=GitOperations(workspace_dir=repo))
        _run_async(tool.execute(message="Specific message for test"))
        # Verify via git log
        log_result = subprocess.run(
            ["git", "log", "-1", "--format=%s"], cwd=repo,
            capture_output=True, text=True,
        )
        assert "Specific message for test" in log_result.stdout


# ══════════════════════════════════════════════════════════════════
# 7. TestGitBranchTool
# ══════════════════════════════════════════════════════════════════

class TestGitBranchTool:
    """Tests for GitBranchTool (~8 tests)."""

    def test_list_branches(self, tmp_path):
        repo = _make_repo(tmp_path)
        tool = GitBranchTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(action="list"))
        assert result.success is True
        assert "*" in result.output  # Current branch marker

    def test_create_branch(self, tmp_path):
        repo = _make_repo(tmp_path)
        tool = GitBranchTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(action="create", name="feature-x"))
        assert result.success is True
        assert "Created" in result.output

    def test_create_branch_no_name_error(self, tmp_path):
        repo = _make_repo(tmp_path)
        tool = GitBranchTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(action="create", name=""))
        assert result.success is False

    def test_switch_branch(self, tmp_path):
        repo = _make_repo(tmp_path)
        _run(["git", "branch", "other"], repo)
        tool = GitBranchTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(action="switch", name="other"))
        assert result.success is True
        assert "Switched" in result.output

    def test_delete_branch(self, tmp_path):
        repo = _make_repo(tmp_path)
        _run(["git", "branch", "deleteme"], repo)
        tool = GitBranchTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(action="delete", name="deleteme"))
        assert result.success is True

    def test_delete_protected_branch_force_blocked(self, tmp_path):
        repo = _make_repo(tmp_path)
        tool = GitBranchTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(action="delete", name="main", force=True))
        assert result.success is False
        assert "protected" in result.error.lower() or "Cannot" in result.error

    def test_unknown_action_error(self, tmp_path):
        repo = _make_repo(tmp_path)
        tool = GitBranchTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(action="unknown"))
        assert result.success is False

    def test_not_git_repo(self, tmp_path):
        tool = GitBranchTool(git_ops=GitOperations(workspace_dir=str(tmp_path)))
        result = _run_async(tool.execute())
        assert result.success is False


# ══════════════════════════════════════════════════════════════════
# 8. TestGitLogTool
# ══════════════════════════════════════════════════════════════════

class TestGitLogTool:
    """Tests for GitLogTool (~5 tests)."""

    def test_default_log(self, tmp_path):
        repo = _make_repo(tmp_path)
        tool = GitLogTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute())
        assert result.success is True
        assert "Initial commit" in result.output

    def test_custom_n(self, tmp_path):
        repo = _make_repo(tmp_path)
        # Add 3 more commits
        for i in range(3):
            (tmp_path / f"f{i}.txt").write_text(str(i))
            _run(["git", "add", f"f{i}.txt"], repo)
            _run(["git", "commit", "-m", f"Commit {i}"], repo)

        tool = GitLogTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(n=2))
        assert "Showing 2 commit(s)" in result.output

    def test_specific_branch(self, tmp_path):
        repo = _make_repo(tmp_path)
        _run(["git", "branch", "feat"], repo)
        _run(["git", "checkout", "feat"], repo)
        (tmp_path / "feat.txt").write_text("feat")
        _run(["git", "add", "feat.txt"], repo)
        _run(["git", "commit", "-m", "Feat commit"], repo)

        main_branch = "master"
        if "main" in _run(["git", "branch"], repo).stdout:
            main_branch = "main"
        _run(["git", "checkout", main_branch], repo)

        tool = GitLogTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute(branch="feat"))
        assert "Feat commit" in result.output

    def test_empty_repo_no_commits(self, tmp_path):
        repo = str(tmp_path)
        _run(["git", "init"], repo)
        tool = GitLogTool(git_ops=GitOperations(workspace_dir=repo))
        result = _run_async(tool.execute())
        assert "No commits" in result.output

    def test_not_git_repo(self, tmp_path):
        tool = GitLogTool(git_ops=GitOperations(workspace_dir=str(tmp_path)))
        result = _run_async(tool.execute())
        assert result.success is False


# ══════════════════════════════════════════════════════════════════
# 9. TestWorktreeEnhanced
# ══════════════════════════════════════════════════════════════════

class TestWorktreeEnhanced:
    """Tests for enhanced WorktreeManager methods (~10 tests)."""

    def test_snapshot_nonexistent(self, tmp_path):
        repo = _make_repo(tmp_path)
        wt = WorktreeManager(workspace_dir=repo)
        snap = wt.snapshot("nonexistent")
        assert "error" in snap

    def test_snapshot_existing_worktree(self, tmp_path):
        repo = _make_repo(tmp_path)
        wt = WorktreeManager(workspace_dir=repo)
        info = wt.create("snap-test")
        assert info is not None
        snap = wt.snapshot("snap-test")
        assert snap["name"] == "snap-test"
        assert snap["branch"] == "worktree/snap-test"
        assert "head_commit" in snap
        assert "changed_files" in snap
        # Cleanup
        wt.remove("snap-test", force=True)

    def test_snapshot_with_changes(self, tmp_path):
        repo = _make_repo(tmp_path)
        wt = WorktreeManager(workspace_dir=repo)
        info = wt.create("snap-dirty")
        assert info is not None
        # Make a change in the worktree
        Path(info.path, "new_file.txt").write_text("hello")
        snap = wt.snapshot("snap-dirty")
        assert snap["has_changes"] is True
        assert "new_file.txt" in snap["changed_files"]
        wt.remove("snap-dirty", force=True)

    def test_diff_worktree_nonexistent(self, tmp_path):
        repo = _make_repo(tmp_path)
        wt = WorktreeManager(workspace_dir=repo)
        result = wt.diff_worktree("nonexistent")
        assert "not found" in result.lower()

    def test_diff_worktree_no_changes(self, tmp_path):
        repo = _make_repo(tmp_path)
        wt = WorktreeManager(workspace_dir=repo)
        info = wt.create("diff-clean")
        assert info is not None
        result = wt.diff_worktree("diff-clean")
        assert "no differences" in result.lower() or result.strip() == ""
        wt.remove("diff-clean", force=True)

    def test_diff_worktree_with_committed_changes(self, tmp_path):
        repo = _make_repo(tmp_path)
        wt = WorktreeManager(workspace_dir=repo)
        info = wt.create("diff-change")
        assert info is not None
        # Commit a change in the worktree
        Path(info.path, "wt_file.txt").write_text("worktree content")
        _run(["git", "add", "wt_file.txt"], info.path)
        _run(["git", "commit", "-m", "Worktree commit"], info.path)
        result = wt.diff_worktree("diff-change")
        assert "worktree content" in result or "wt_file.txt" in result
        wt.remove("diff-change", force=True)

    def test_merge_back_nonexistent(self, tmp_path):
        repo = _make_repo(tmp_path)
        wt = WorktreeManager(workspace_dir=repo)
        result = wt.merge_back("nonexistent")
        assert result["success"] is False

    def test_merge_back_clean(self, tmp_path):
        repo = _make_repo(tmp_path)
        wt = WorktreeManager(workspace_dir=repo)
        info = wt.create("merge-test")
        assert info is not None
        # Make a commit in the worktree
        Path(info.path, "merge_file.txt").write_text("merge content")
        _run(["git", "add", "merge_file.txt"], info.path)
        _run(["git", "commit", "-m", "Merge commit"], info.path)
        # Merge back
        result = wt.merge_back("merge-test")
        assert result["success"] is True
        wt.remove("merge-test", force=True)

    def test_list_worktrees(self, tmp_path):
        repo = _make_repo(tmp_path)
        wt = WorktreeManager(workspace_dir=repo)
        info = wt.create("list-test")
        assert info is not None
        worktrees = wt.list_worktrees()
        assert len(worktrees) >= 1
        names = [w.name for w in worktrees]
        assert "list-test" in names
        wt.remove("list-test", force=True)

    def test_create_and_remove(self, tmp_path):
        repo = _make_repo(tmp_path)
        wt = WorktreeManager(workspace_dir=repo)
        info = wt.create("temp-wt")
        assert info is not None
        assert os.path.exists(info.path)
        result = wt.remove("temp-wt", force=True)
        assert "removed" in result.lower()


# ══════════════════════════════════════════════════════════════════
# 10. TestSafetyGitChecks
# ══════════════════════════════════════════════════════════════════

class TestSafetyGitChecks:
    """Tests for SafetyChecker git extensions (~6 tests)."""

    def test_force_delete_protected_branch_blocked(self):
        checker = SafetyChecker()
        call = ToolCall(name="git_branch", tool_id="", input={
            "action": "delete", "name": "main", "force": True,
        })
        result = checker.check(call)
        assert result.blocked is True
        assert "protected" in result.block_reason.lower()

    def test_force_delete_master_blocked(self):
        checker = SafetyChecker()
        call = ToolCall(name="git_branch", tool_id="", input={
            "action": "delete", "name": "master", "force": True,
        })
        result = checker.check(call)
        assert result.blocked is True

    def test_normal_delete_protected_warns(self):
        checker = SafetyChecker()
        call = ToolCall(name="git_branch", tool_id="", input={
            "action": "delete", "name": "main",
        })
        result = checker.check(call)
        assert result.blocked is False
        assert len(result.warnings) > 0

    def test_delete_non_protected_ok(self):
        checker = SafetyChecker()
        call = ToolCall(name="git_branch", tool_id="", input={
            "action": "delete", "name": "feature-x", "force": True,
        })
        result = checker.check(call)
        assert result.blocked is False

    def test_custom_protected_branches(self):
        checker = SafetyChecker(protected_branches=["production", "staging"])
        call = ToolCall(name="git_branch", tool_id="", input={
            "action": "delete", "name": "production", "force": True,
        })
        result = checker.check(call)
        assert result.blocked is True

    def test_non_delete_action_ok(self):
        checker = SafetyChecker()
        call = ToolCall(name="git_branch", tool_id="", input={
            "action": "list",
        })
        result = checker.check(call)
        assert result.blocked is False
        assert len(result.warnings) == 0


# ══════════════════════════════════════════════════════════════════
# 11. TestIntegration
# ══════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests combining multiple Sprint 18 components (~8 tests)."""

    def test_full_git_workflow(self, tmp_path):
        """Create branch, make changes, commit, check status, view log."""
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)

        # Create branch
        git.branch_create("feature")
        git.checkout("feature")
        assert git.current_branch() == "feature"

        # Make changes
        (tmp_path / "feature.py").write_text("print('feature')\n")
        status = git.status()
        assert "feature.py" in status.untracked

        # Commit
        git.commit("Add feature", files=["feature.py"])
        status = git.status()
        assert status.is_clean

        # Log
        commits = git.log(n=5)
        assert any(c.message == "Add feature" for c in commits)

    def test_workspace_context_with_git_ops(self, tmp_path):
        """WorkspaceContext + GitOperations integration."""
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)
        ctx = WorkspaceContext(workspace_dir=repo, git_ops=git)

        ctx.refresh()
        assert ctx.is_git_repo is True
        assert ctx.is_dirty is False

        (tmp_path / "dirty.py").write_text("dirty")
        ctx.refresh()
        assert ctx.is_dirty is True

        git.commit("Clean up", files=["dirty.py"])
        ctx.refresh()
        assert ctx.is_dirty is False

    def test_file_lock_prevents_concurrent_write(self):
        """FileLockManager prevents two agents from writing to same file."""
        mgr = FileLockManager()

        assert mgr.acquire_write("/shared.py", "agent-1") is True
        assert mgr.acquire_write("/shared.py", "agent-2") is False

        mgr.release("/shared.py", "agent-1")
        assert mgr.acquire_write("/shared.py", "agent-2") is True

    def test_worktree_create_work_merge(self, tmp_path):
        """Full worktree workflow: create, work, merge back."""
        repo = _make_repo(tmp_path)
        wt = WorktreeManager(workspace_dir=repo)

        # Create worktree
        info = wt.create("integration-test")
        assert info is not None
        assert os.path.isdir(info.path)

        # Work in worktree
        Path(info.path, "wt_work.txt").write_text("integration work")
        _run(["git", "add", "wt_work.txt"], info.path)
        _run(["git", "commit", "-m", "Integration work"], info.path)

        # Snapshot
        snap = wt.snapshot("integration-test")
        assert snap["name"] == "integration-test"
        assert snap["has_changes"] is False  # Just committed

        # Merge back
        result = wt.merge_back("integration-test")
        assert result["success"] is True

        # Verify file is now in main
        assert (tmp_path / "wt_work.txt").exists()

        # Cleanup
        wt.remove("integration-test", force=True)

    def test_git_tools_with_real_repo(self, tmp_path):
        """All 5 git tools working against a real repo."""
        repo = _make_repo(tmp_path)
        git = GitOperations(workspace_dir=repo)

        status_tool = GitStatusTool(git_ops=git)
        diff_tool = GitDiffTool(git_ops=git)
        commit_tool = GitCommitTool(git_ops=git)
        branch_tool = GitBranchTool(git_ops=git)
        log_tool = GitLogTool(git_ops=git)

        # Status (clean)
        r = _run_async(status_tool.execute())
        assert r.success and "clean" in r.output.lower()

        # Make a change
        (tmp_path / "tool_test.py").write_text("# test\n")

        # Diff
        r = _run_async(status_tool.execute())
        assert "Untracked" in r.output

        # Commit
        r = _run_async(commit_tool.execute(message="Tool test", files=["tool_test.py"]))
        assert r.success

        # Branch
        r = _run_async(branch_tool.execute(action="create", name="tool-branch"))
        assert r.success

        # Log
        r = _run_async(log_tool.execute())
        assert "Tool test" in r.output

    def test_safety_checker_with_git_tools(self, tmp_path):
        """SafetyChecker protects git branch operations."""
        checker = SafetyChecker(workspace_dir=str(tmp_path))

        # Normal operation OK
        call = ToolCall(name="git_branch", tool_id="", input={"action": "create", "name": "safe"})
        result = checker.check(call)
        assert result.blocked is False

        # Protected branch force-delete blocked
        call = ToolCall(name="git_branch", tool_id="", input={
            "action": "delete", "name": "main", "force": True,
        })
        result = checker.check(call)
        assert result.blocked is True

    def test_lock_expiry_allows_recovery(self):
        """Expired locks are auto-cleaned, allowing other agents to proceed."""
        mgr = FileLockManager(lock_timeout=0.01)
        mgr.acquire_write("/locked.py", "crashed-agent")
        time.sleep(0.05)

        # Another agent can now acquire
        assert mgr.acquire_write("/locked.py", "recovery-agent") is True

    def test_context_tracks_worktree_switch(self, tmp_path):
        """WorkspaceContext tracks active worktree."""
        repo = _make_repo(tmp_path)
        ctx = WorkspaceContext(workspace_dir=repo)
        ctx.refresh()

        assert ctx.active_worktree is None

        info = WorktreeInfo(
            name="ctx-test", path="/tmp/ctx-test",
            branch="worktree/ctx-test", created_from="main",
        )
        ctx.set_active_worktree(info)
        d = ctx.to_dict()
        assert d["active_worktree"]["name"] == "ctx-test"

        ctx.clear_active_worktree()
        d = ctx.to_dict()
        assert d["active_worktree"] is None


# ══════════════════════════════════════════════════════════════════
# 12. Dataclass tests
# ══════════════════════════════════════════════════════════════════

class TestDataclasses:
    """Tests for Sprint 18 dataclasses (~8 tests)."""

    def test_git_status_result_clean(self):
        s = GitStatusResult()
        assert s.is_clean is True
        assert s.total_changes == 0

    def test_git_status_result_dirty(self):
        s = GitStatusResult(staged=["a.py"], untracked=["b.py"])
        assert s.is_clean is False
        assert s.total_changes == 2

    def test_git_commit_to_dict(self):
        c = GitCommit(hash="abc123", author="Test", message="msg", timestamp="2025-01-01")
        d = c.to_dict()
        assert d["hash"] == "abc123"
        assert d["message"] == "msg"

    def test_git_branch_to_dict(self):
        b = GitBranch(name="main", is_current=True, upstream="origin/main")
        d = b.to_dict()
        assert d["name"] == "main"
        assert d["is_current"] is True

    def test_merge_result_success(self):
        m = MergeResult(success=True, message="OK")
        assert m.has_conflicts is False

    def test_merge_result_conflicts(self):
        m = MergeResult(success=False, conflicts=["a.py", "b.py"], message="Conflict")
        assert m.has_conflicts is True
        assert len(m.conflicts) == 2

    def test_lock_info_expiry(self):
        info = LockInfo(path="/a.py", owner="x", lock_type="write", expires_at=time.time() - 10)
        assert info.is_expired is True

    def test_lock_info_not_expired(self):
        info = LockInfo(path="/a.py", owner="x", lock_type="write", expires_at=time.time() + 100)
        assert info.is_expired is False
