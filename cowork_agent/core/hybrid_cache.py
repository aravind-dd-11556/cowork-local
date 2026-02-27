"""
Hybrid Response Cache — two-tier memory LRU + disk JSON cache.

Memory tier: fast in-process OrderedDict LRU, capped at ``max_memory_entries``.
Disk tier:   secondary JSON files in ``{workspace}/.cowork/cache/``, TTL-evicted.

Lookup order: memory → disk → miss.
Evicted memory entries spill to disk; disk hits are promoted to memory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .models import AgentResponse

logger = logging.getLogger(__name__)


# ── Cache entry (reused concept from response_cache.py) ─────────────

@dataclass
class _CacheEntry:
    """In-memory cache entry."""
    response: AgentResponse
    created_at: float
    hit_count: int = 0
    cache_key: str = ""

    def is_expired(self, ttl: float) -> bool:
        if ttl <= 0:
            return False
        return (time.time() - self.created_at) > ttl


# ── Stats ───────────────────────────────────────────────────────────

@dataclass
class CacheStats:
    """Statistics about hybrid cache usage."""
    memory_entries: int
    disk_entries: int
    memory_hits: int
    memory_misses: int
    disk_hits: int
    disk_misses: int
    total_hits: int
    total_misses: int

    @property
    def memory_hit_rate(self) -> float:
        total = self.memory_hits + self.memory_misses
        return self.memory_hits / total if total > 0 else 0.0

    @property
    def disk_hit_rate(self) -> float:
        total = self.disk_hits + self.disk_misses
        return self.disk_hits / total if total > 0 else 0.0


# ── HybridResponseCache ────────────────────────────────────────────

class HybridResponseCache:
    """Two-tier cache: memory LRU + disk JSON files."""

    def __init__(
        self,
        workspace_dir: str = "",
        max_memory_entries: int = 100,
        max_disk_entries: int = 1000,
        ttl: float = 3600.0,
        enabled: bool = True,
    ):
        self._workspace_dir = workspace_dir
        self._max_memory = max_memory_entries
        self._max_disk = max_disk_entries
        self._ttl = ttl
        self._enabled = enabled

        # Memory tier
        self._mem: OrderedDict[str, _CacheEntry] = OrderedDict()

        # Disk tier
        self._disk_dir = ""
        if workspace_dir:
            self._disk_dir = os.path.join(workspace_dir, ".cowork", "cache")
            os.makedirs(self._disk_dir, exist_ok=True)

        # Counters
        self._mem_hits = 0
        self._mem_misses = 0
        self._disk_hits = 0
        self._disk_misses = 0

    # ── Public API ─────────────────────────────────────────────

    def get(self, key: str) -> Optional[AgentResponse]:
        """Look up a cached response: memory first, then disk."""
        if not self._enabled:
            self._mem_misses += 1
            self._disk_misses += 1
            return None

        # Memory tier
        if key in self._mem:
            entry = self._mem[key]
            if not entry.is_expired(self._ttl):
                self._mem.move_to_end(key)
                entry.hit_count += 1
                self._mem_hits += 1
                return entry.response
            else:
                del self._mem[key]

        self._mem_misses += 1

        # Disk tier
        if self._disk_dir:
            response = self._load_from_disk(key)
            if response is not None:
                self._disk_hits += 1
                # Promote to memory
                self._mem[key] = _CacheEntry(
                    response=response,
                    created_at=time.time(),
                    cache_key=key,
                )
                self._evict_memory()
                return response

        self._disk_misses += 1
        return None

    def put(self, key: str, response: AgentResponse) -> bool:
        """Cache a response (memory + optional disk spill)."""
        if not self._enabled:
            return False
        if not self._is_cacheable(response):
            return False

        self._mem[key] = _CacheEntry(
            response=response,
            created_at=time.time(),
            cache_key=key,
        )
        self._evict_memory()

        # Also write to disk for persistence across restarts
        if self._disk_dir:
            self._save_to_disk(key, response)
            self._evict_disk()

        return True

    def invalidate(self, key: str) -> bool:
        """Remove entry from both memory and disk."""
        found = False

        if key in self._mem:
            del self._mem[key]
            found = True

        if self._disk_dir:
            path = self._disk_path(key)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    found = True
                except OSError:
                    pass

        return found

    def clear(self) -> None:
        """Clear all entries from memory and disk."""
        self._mem.clear()
        if self._disk_dir and os.path.exists(self._disk_dir):
            try:
                shutil.rmtree(self._disk_dir)
                os.makedirs(self._disk_dir, exist_ok=True)
            except OSError:
                pass

    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        disk_count = len(self._disk_keys()) if self._disk_dir else 0
        return CacheStats(
            memory_entries=len(self._mem),
            disk_entries=disk_count,
            memory_hits=self._mem_hits,
            memory_misses=self._mem_misses,
            disk_hits=self._disk_hits,
            disk_misses=self._disk_misses,
            total_hits=self._mem_hits + self._disk_hits,
            total_misses=self._mem_misses + self._disk_misses,
        )

    # ── Internal: memory eviction ──────────────────────────────

    def _evict_memory(self) -> None:
        while len(self._mem) > self._max_memory:
            evicted_key, evicted = self._mem.popitem(last=False)
            # Spill to disk
            if self._disk_dir:
                self._save_to_disk(evicted_key, evicted.response)

    # ── Internal: disk I/O ─────────────────────────────────────

    def _disk_path(self, key: str) -> str:
        return os.path.join(self._disk_dir, f"{key}.json")

    def _disk_keys(self) -> list[str]:
        if not self._disk_dir or not os.path.isdir(self._disk_dir):
            return []
        return [f[:-5] for f in os.listdir(self._disk_dir) if f.endswith(".json")]

    def _save_to_disk(self, key: str, response: AgentResponse) -> None:
        try:
            data = {
                "key": key,
                "created_at": time.time(),
                "response": {
                    "text": response.text,
                    "stop_reason": response.stop_reason,
                    "usage": response.usage,
                },
            }
            with open(self._disk_path(key), "w") as f:
                json.dump(data, f)
        except Exception as exc:
            logger.warning("Failed to write cache to disk: %s", exc)

    def _load_from_disk(self, key: str) -> Optional[AgentResponse]:
        path = self._disk_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            # TTL check
            if self._ttl > 0 and (time.time() - data.get("created_at", 0)) > self._ttl:
                os.remove(path)
                return None
            rd = data.get("response", {})
            return AgentResponse(
                text=rd.get("text"),
                tool_calls=[],
                stop_reason=rd.get("stop_reason", "end_turn"),
                usage=rd.get("usage"),
            )
        except Exception as exc:
            logger.warning("Failed to load cache from disk: %s", exc)
            return None

    def _evict_disk(self) -> None:
        """Remove oldest disk entries if over limit."""
        keys = self._disk_keys()
        if len(keys) <= self._max_disk:
            return
        # Sort by modification time, remove oldest
        paths = [(k, os.path.getmtime(self._disk_path(k))) for k in keys]
        paths.sort(key=lambda p: p[1])
        to_remove = len(keys) - self._max_disk
        for k, _ in paths[:to_remove]:
            try:
                os.remove(self._disk_path(k))
            except OSError:
                pass

    # ── Internal: cacheability check ───────────────────────────

    @staticmethod
    def _is_cacheable(response: AgentResponse) -> bool:
        if response.tool_calls:
            return False
        if response.stop_reason != "end_turn":
            return False
        if not response.text:
            return False
        return True
