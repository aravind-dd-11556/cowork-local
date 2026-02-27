"""
Response Cache — LRU cache for LLM responses.

Caches deterministic tool-free LLM calls to avoid redundant API hits
for identical prompts. Uses an in-memory LRU with optional TTL.

Cache key = hash(model + system_prompt + messages_fingerprint)

Only caches responses that:
  - Have no tool calls (pure text responses)
  - Completed normally (stop_reason == "end_turn")

Sprint 4 (P2-Advanced) Feature 5.
"""

from __future__ import annotations
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from .models import AgentResponse, Message, ToolSchema

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached LLM response."""
    response: AgentResponse
    created_at: float
    hit_count: int = 0
    cache_key: str = ""

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def is_expired(self, ttl: float) -> bool:
        """Check if entry has exceeded its time-to-live."""
        if ttl <= 0:
            return False  # TTL disabled
        return self.age_seconds > ttl


class ResponseCache:
    """
    In-memory LRU cache for LLM text responses.

    Usage:
        cache = ResponseCache(max_size=100, ttl=3600)

        key = cache.make_key(model, messages, system_prompt)
        cached = cache.get(key)
        if cached:
            return cached  # Cache hit

        response = await provider.send_message(...)
        cache.put(key, response)
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl: float = 3600.0,  # 1 hour default
        enabled: bool = True,
    ):
        self.max_size = max(1, max_size)
        self.ttl = ttl
        self.enabled = enabled

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def make_key(
        self,
        model: str,
        messages: list[Message],
        system_prompt: str,
        tools: Optional[list[ToolSchema]] = None,
    ) -> str:
        """
        Generate a deterministic cache key from the request parameters.

        Uses SHA-256 of the serialized inputs to produce a stable key.
        """
        # Build a fingerprint of the messages
        msg_parts = []
        for msg in messages:
            part = f"{msg.role}:{msg.content}"
            if msg.tool_calls:
                tc_str = json.dumps(
                    [{"n": tc.name, "i": tc.input} for tc in msg.tool_calls],
                    sort_keys=True, default=str,
                )
                part += f"|tc:{tc_str}"
            if msg.tool_results:
                tr_str = json.dumps(
                    [{"id": tr.tool_id, "ok": tr.success, "o": tr.output[:200]}
                     for tr in msg.tool_results],
                    sort_keys=True, default=str,
                )
                part += f"|tr:{tr_str}"
            msg_parts.append(part)

        fingerprint = "\n".join([
            f"model:{model}",
            f"system:{system_prompt[:500]}",  # First 500 chars of system prompt
            *msg_parts,
        ])

        return hashlib.sha256(fingerprint.encode()).hexdigest()

    def get(self, key: str) -> Optional[AgentResponse]:
        """
        Look up a cached response.

        Returns None on miss or expired entry.
        """
        if not self.enabled:
            self._misses += 1
            return None

        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        # Check TTL
        if entry.is_expired(self.ttl):
            logger.debug(f"Cache entry expired (age={entry.age_seconds:.0f}s)")
            del self._cache[key]
            self._misses += 1
            return None

        # Hit — move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hit_count += 1
        self._hits += 1

        logger.debug(f"Cache HIT (hits={entry.hit_count}, key={key[:12]}...)")
        return entry.response

    def put(self, key: str, response: AgentResponse) -> bool:
        """
        Cache a response if it's cacheable.

        Returns True if cached, False if rejected.
        Only caches:
          - Pure text responses (no tool calls)
          - Normal completions (stop_reason == "end_turn")
        """
        if not self.enabled:
            return False

        # Only cache clean text responses
        if not self._is_cacheable(response):
            return False

        # If key already exists, update it
        if key in self._cache:
            self._cache[key] = CacheEntry(
                response=response,
                created_at=time.time(),
                cache_key=key,
            )
            self._cache.move_to_end(key)
            return True

        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(f"Cache evicted LRU entry: {evicted_key[:12]}...")

        self._cache[key] = CacheEntry(
            response=response,
            created_at=time.time(),
            cache_key=key,
        )
        return True

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry from the cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        logger.info("Response cache cleared")

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "ttl": self.ttl,
            "enabled": self.enabled,
        }

    @staticmethod
    def _is_cacheable(response: AgentResponse) -> bool:
        """Check if a response is suitable for caching."""
        # Don't cache tool calls — they're side-effecting
        if response.tool_calls:
            return False
        # Don't cache errors or truncated responses
        if response.stop_reason != "end_turn":
            return False
        # Don't cache empty responses
        if not response.text:
            return False
        return True
