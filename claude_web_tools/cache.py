"""
Self-cleaning in-memory cache with TTL expiry.
Mirrors the 15-minute cache behavior in Claude's WebFetch tool.
"""

import time
import threading
from typing import Any, Optional


class TTLCache:
    """Thread-safe in-memory cache with automatic expiry."""

    def __init__(self, ttl_seconds: int = 900):
        self._store: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        # Start background cleaner
        self._cleaner = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleaner.start()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value if it exists and hasn't expired."""
        with self._lock:
            if key in self._store:
                timestamp, value = self._store[key]
                if time.time() - timestamp < self._ttl:
                    return value
                else:
                    del self._store[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Store a value with the current timestamp."""
        with self._lock:
            self._store[key] = (time.time(), value)

    def invalidate(self, key: str) -> None:
        """Remove a specific entry."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._store.clear()

    def _cleanup_loop(self) -> None:
        """Background thread that removes expired entries every 60 seconds."""
        while True:
            time.sleep(60)
            now = time.time()
            with self._lock:
                expired = [
                    k for k, (ts, _) in self._store.items()
                    if now - ts >= self._ttl
                ]
                for k in expired:
                    del self._store[k]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            now = time.time()
            active = sum(1 for ts, _ in self._store.values() if now - ts < self._ttl)
            return {
                "total_entries": len(self._store),
                "active_entries": active,
                "ttl_seconds": self._ttl,
            }
