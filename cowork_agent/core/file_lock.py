"""
File Lock Manager — Reader-writer locks for safe concurrent file editing.

Provides per-file locking to prevent two agents or worktrees from
writing to the same file simultaneously:
  - Exclusive write locks (one writer at a time)
  - Shared read locks (multiple readers allowed)
  - Timeout-based expiry for orphan detection
  - Owner tracking for debugging and release

Sprint 18 (Worktree & Git Integration) Feature 2.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LockInfo:
    """Metadata about a file lock."""
    path: str
    owner: str
    lock_type: str  # "read" or "write"
    acquired_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    @property
    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            return False
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        return time.time() - self.acquired_at

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "owner": self.owner,
            "lock_type": self.lock_type,
            "acquired_at": self.acquired_at,
            "expires_at": self.expires_at,
            "is_expired": self.is_expired,
            "age_seconds": round(self.age_seconds, 1),
        }


class FileLockManager:
    """
    Per-file reader-writer lock manager.

    Rules:
      - Multiple readers can hold shared read locks simultaneously
      - Only one writer can hold an exclusive write lock
      - Write lock blocks all readers and other writers
      - Locks expire after timeout (prevents deadlock from crashed agents)

    Usage:
        mgr = FileLockManager(lock_timeout=300)

        if mgr.acquire_write("/path/to/file.py", owner="agent-main"):
            # ... write to file ...
            mgr.release("/path/to/file.py", owner="agent-main")

        # Check for conflicts before writing
        conflicts = mgr.check_conflicts(["/a.py", "/b.py"], owner="agent-1")
        if conflicts:
            print(f"Cannot write: {conflicts} locked by other agents")
    """

    def __init__(self, lock_timeout: float = 300.0):
        self.lock_timeout = lock_timeout
        # Write locks: path -> LockInfo (exclusive)
        self._write_locks: dict[str, LockInfo] = {}
        # Read locks: path -> [LockInfo] (shared)
        self._read_locks: dict[str, list[LockInfo]] = {}

    def acquire_write(self, path: str, owner: str) -> bool:
        """
        Acquire an exclusive write lock on a file.

        Returns True if acquired, False if blocked by another lock.
        Auto-cleans expired locks before checking.
        """
        path = os.path.normpath(path)
        self._cleanup_expired_for(path)

        # Check for existing write lock by another owner
        existing = self._write_locks.get(path)
        if existing and existing.owner != owner:
            logger.debug(f"Write lock blocked: {path} held by {existing.owner}")
            return False

        # Check for existing read locks by other owners
        readers = self._read_locks.get(path, [])
        other_readers = [r for r in readers if r.owner != owner]
        if other_readers:
            logger.debug(f"Write lock blocked: {path} has {len(other_readers)} reader(s)")
            return False

        # Acquire or refresh lock
        now = time.time()
        self._write_locks[path] = LockInfo(
            path=path,
            owner=owner,
            lock_type="write",
            acquired_at=now,
            expires_at=now + self.lock_timeout,
        )
        logger.debug(f"Write lock acquired: {path} by {owner}")
        return True

    def acquire_read(self, path: str, owner: str) -> bool:
        """
        Acquire a shared read lock on a file.

        Returns True if acquired, False if blocked by a write lock.
        Multiple readers are allowed simultaneously.
        """
        path = os.path.normpath(path)
        self._cleanup_expired_for(path)

        # Check for existing write lock by another owner
        existing_write = self._write_locks.get(path)
        if existing_write and existing_write.owner != owner:
            logger.debug(f"Read lock blocked: {path} write-locked by {existing_write.owner}")
            return False

        # Add read lock
        now = time.time()
        if path not in self._read_locks:
            self._read_locks[path] = []

        # Don't duplicate — refresh existing lock for same owner
        self._read_locks[path] = [r for r in self._read_locks[path] if r.owner != owner]
        self._read_locks[path].append(LockInfo(
            path=path,
            owner=owner,
            lock_type="read",
            acquired_at=now,
            expires_at=now + self.lock_timeout,
        ))
        return True

    def release(self, path: str, owner: str) -> bool:
        """
        Release any lock held by owner on path.

        Returns True if a lock was released, False if no matching lock found.
        """
        path = os.path.normpath(path)
        released = False

        # Release write lock
        existing = self._write_locks.get(path)
        if existing and existing.owner == owner:
            del self._write_locks[path]
            released = True
            logger.debug(f"Write lock released: {path} by {owner}")

        # Release read lock
        if path in self._read_locks:
            before = len(self._read_locks[path])
            self._read_locks[path] = [r for r in self._read_locks[path] if r.owner != owner]
            if len(self._read_locks[path]) < before:
                released = True
            if not self._read_locks[path]:
                del self._read_locks[path]

        return released

    def check_conflicts(self, paths: list[str], owner: str) -> list[str]:
        """
        Check if any paths are locked by a different owner.

        Returns list of paths that have conflicting locks.
        Non-blocking — just reports conflicts.
        """
        self.cleanup_expired()
        conflicts = []
        for path in paths:
            path = os.path.normpath(path)

            # Check write locks
            wl = self._write_locks.get(path)
            if wl and wl.owner != owner and not wl.is_expired:
                conflicts.append(path)
                continue

            # Check read locks (only conflict if we need write access)
            readers = self._read_locks.get(path, [])
            other_readers = [r for r in readers if r.owner != owner and not r.is_expired]
            if other_readers:
                conflicts.append(path)

        return conflicts

    def get_lock_info(self, path: str) -> Optional[LockInfo]:
        """Get lock info for a specific file. Prefers write lock if both exist."""
        path = os.path.normpath(path)
        wl = self._write_locks.get(path)
        if wl and not wl.is_expired:
            return wl
        readers = self._read_locks.get(path, [])
        active = [r for r in readers if not r.is_expired]
        if active:
            return active[0]
        return None

    def cleanup_expired(self) -> int:
        """Remove all expired locks. Returns count of removed locks."""
        count = 0

        # Clean write locks
        expired_writes = [p for p, l in self._write_locks.items() if l.is_expired]
        for p in expired_writes:
            del self._write_locks[p]
            count += 1
            logger.debug(f"Expired write lock cleaned: {p}")

        # Clean read locks
        for path in list(self._read_locks.keys()):
            before = len(self._read_locks[path])
            self._read_locks[path] = [r for r in self._read_locks[path] if not r.is_expired]
            removed = before - len(self._read_locks[path])
            count += removed
            if not self._read_locks[path]:
                del self._read_locks[path]

        if count > 0:
            logger.info(f"Cleaned up {count} expired lock(s)")
        return count

    def active_locks(self) -> dict[str, LockInfo]:
        """Return all active (non-expired) locks as {path: LockInfo}."""
        self.cleanup_expired()
        result = {}
        for path, lock in self._write_locks.items():
            result[path] = lock
        for path, readers in self._read_locks.items():
            if path not in result and readers:
                result[path] = readers[0]
        return result

    @property
    def lock_count(self) -> int:
        """Total number of active locks."""
        write_count = len(self._write_locks)
        read_count = sum(len(readers) for readers in self._read_locks.values())
        return write_count + read_count

    def _cleanup_expired_for(self, path: str) -> None:
        """Clean expired locks for a specific path only."""
        wl = self._write_locks.get(path)
        if wl and wl.is_expired:
            del self._write_locks[path]

        if path in self._read_locks:
            self._read_locks[path] = [r for r in self._read_locks[path] if not r.is_expired]
            if not self._read_locks[path]:
                del self._read_locks[path]
