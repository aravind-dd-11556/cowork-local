"""
Agent Pool — Pooled agent management with acquire/release and auto-scaling.

Provides AgentPool for managing reusable agent instances, and AutoScaler
for dynamically adjusting pool size based on utilization.

Sprint 21 (Multi-Agent Orchestration Enhancement) Module 4.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for an agent pool."""
    name: str = "default"
    min_size: int = 1
    max_size: int = 10
    initial_size: int = 2
    idle_timeout_seconds: float = 300.0
    acquire_timeout_seconds: float = 30.0


@dataclass
class PooledAgent:
    """Wrapper tracking a pooled agent's state."""
    agent: Any  # Agent instance
    pool_name: str = ""
    in_use: bool = False
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0
    id: str = ""

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used_at


class AgentPool:
    """
    Pool of reusable agent instances with acquire/release semantics.

    Usage::

        pool = AgentPool(
            config=PoolConfig(name="workers", min_size=2, max_size=8),
            agent_factory=lambda: create_agent(),
        )
        await pool.initialize()

        agent = await pool.acquire()
        try:
            result = await agent.agent.run("do something")
        finally:
            await pool.release(agent)
    """

    def __init__(
        self,
        config: Optional[PoolConfig] = None,
        agent_factory: Optional[Callable] = None,
    ):
        self.config = config or PoolConfig()
        self._factory = agent_factory
        self._available: deque[PooledAgent] = deque()
        self._in_use: Dict[str, PooledAgent] = {}
        self._all: Dict[str, PooledAgent] = {}
        self._lock = asyncio.Lock()
        self._available_event = asyncio.Event()
        self._next_id = 0
        self._initialized = False

    async def initialize(self) -> None:
        """Pre-populate the pool with initial_size agents."""
        if self._factory is None:
            raise RuntimeError("No agent_factory configured")

        async with self._lock:
            for _ in range(self.config.initial_size):
                agent = self._create_pooled_agent()
                self._available.append(agent)
            self._initialized = True
            if self._available:
                self._available_event.set()

        logger.info(
            "Pool '%s' initialized with %d agents",
            self.config.name, len(self._available),
        )

    def _create_pooled_agent(self) -> PooledAgent:
        """Create a new PooledAgent with a unique ID."""
        self._next_id += 1
        agent_id = f"{self.config.name}_{self._next_id}"
        raw_agent = self._factory()
        pa = PooledAgent(
            agent=raw_agent,
            pool_name=self.config.name,
            id=agent_id,
        )
        self._all[agent_id] = pa
        return pa

    async def acquire(self, timeout: Optional[float] = None) -> PooledAgent:
        """
        Acquire an agent from the pool.

        If no agents are available and pool isn't at max, creates a new one.
        If at max, waits up to timeout for one to become available.

        Args:
            timeout: Max seconds to wait. Defaults to config.acquire_timeout_seconds.

        Returns:
            PooledAgent ready for use.

        Raises:
            TimeoutError: If no agent available within timeout.
        """
        timeout = timeout or self.config.acquire_timeout_seconds

        async with self._lock:
            if self._available:
                return self._checkout(self._available.popleft())

            # Can we grow?
            if self.size < self.config.max_size:
                agent = self._create_pooled_agent()
                return self._checkout(agent)

        # Pool is at max — wait for release
        try:
            await asyncio.wait_for(self._wait_for_available(), timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Pool '{self.config.name}' acquire timeout after {timeout}s"
            )

        async with self._lock:
            if self._available:
                return self._checkout(self._available.popleft())

        raise TimeoutError(f"Pool '{self.config.name}' no agent available")

    async def _wait_for_available(self) -> None:
        """Wait until an agent becomes available."""
        self._available_event.clear()
        await self._available_event.wait()

    def _checkout(self, pa: PooledAgent) -> PooledAgent:
        """Mark an agent as in-use."""
        pa.in_use = True
        pa.last_used_at = time.time()
        pa.use_count += 1
        self._in_use[pa.id] = pa
        return pa

    async def release(self, pa: PooledAgent) -> None:
        """Return an agent to the pool."""
        async with self._lock:
            pa.in_use = False
            pa.last_used_at = time.time()
            self._in_use.pop(pa.id, None)
            self._available.append(pa)
            self._available_event.set()

    async def scale(self, target_size: int) -> int:
        """
        Scale the pool to target_size. Returns the number of agents added/removed.

        Clamps to [min_size, max_size].
        """
        target = max(self.config.min_size, min(target_size, self.config.max_size))
        delta = 0

        async with self._lock:
            current = self.size
            if target > current:
                # Scale up
                to_add = target - current
                for _ in range(to_add):
                    if self._factory:
                        agent = self._create_pooled_agent()
                        self._available.append(agent)
                        delta += 1
                if delta > 0:
                    self._available_event.set()
            elif target < current:
                # Scale down — only remove idle agents
                to_remove = current - target
                removed = 0
                while removed < to_remove and self._available:
                    pa = self._available.pop()
                    del self._all[pa.id]
                    removed += 1
                    delta -= 1

        logger.info(
            "Pool '%s' scaled: %d -> %d (delta=%d)",
            self.config.name, current, self.size, delta,
        )
        return abs(delta)

    @property
    def size(self) -> int:
        """Total number of agents in the pool."""
        return len(self._all)

    @property
    def available_count(self) -> int:
        """Number of idle agents."""
        return len(self._available)

    @property
    def in_use_count(self) -> int:
        """Number of agents currently in use."""
        return len(self._in_use)

    @property
    def utilization(self) -> float:
        """Current utilization ratio (0.0–1.0)."""
        total = self.size
        if total == 0:
            return 0.0
        return self.in_use_count / total

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "name": self.config.name,
            "size": self.size,
            "available": self.available_count,
            "in_use": self.in_use_count,
            "utilization": round(self.utilization, 3),
            "min_size": self.config.min_size,
            "max_size": self.config.max_size,
            "initialized": self._initialized,
        }

    async def cleanup_idle(self) -> int:
        """Remove agents that have been idle beyond idle_timeout_seconds."""
        removed = 0
        async with self._lock:
            new_available = deque()
            for pa in self._available:
                if (
                    pa.idle_seconds > self.config.idle_timeout_seconds
                    and self.size - removed > self.config.min_size
                ):
                    del self._all[pa.id]
                    removed += 1
                else:
                    new_available.append(pa)
            self._available = new_available
        return removed

    async def shutdown(self) -> None:
        """Shut down the pool and release all resources."""
        async with self._lock:
            self._available.clear()
            self._in_use.clear()
            self._all.clear()
            self._initialized = False
        logger.info("Pool '%s' shut down", self.config.name)


# ── Auto-Scaler ──────────────────────────────────────────────────


@dataclass
class AutoScalerConfig:
    """Configuration for the auto-scaler."""
    scale_up_threshold: float = 0.8    # Scale up when utilization > this
    scale_down_threshold: float = 0.2  # Scale down when utilization < this
    scale_up_step: int = 2             # Add this many agents per scale-up
    scale_down_step: int = 1           # Remove this many per scale-down
    cooldown_seconds: float = 30.0     # Min time between scaling actions
    check_interval_seconds: float = 10.0


class AutoScaler:
    """
    Monitors pool utilization and automatically scales up/down.

    Usage::

        scaler = AutoScaler(pool=pool, config=AutoScalerConfig())
        await scaler.check_and_scale()  # Call periodically or run as task
    """

    def __init__(
        self,
        pool: AgentPool,
        config: Optional[AutoScalerConfig] = None,
    ):
        self.pool = pool
        self.config = config or AutoScalerConfig()
        self._last_scale_time: float = 0.0
        self._scale_history: List[Dict[str, Any]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def check_and_scale(self) -> Optional[Dict[str, Any]]:
        """
        Check utilization and scale if needed.

        Returns scaling action taken (or None if no action).
        """
        now = time.time()
        if now - self._last_scale_time < self.config.cooldown_seconds:
            return None

        utilization = self.pool.utilization
        action = None

        if utilization > self.config.scale_up_threshold:
            target = min(
                self.pool.size + self.config.scale_up_step,
                self.pool.config.max_size,
            )
            if target > self.pool.size:
                changed = await self.pool.scale(target)
                action = {
                    "action": "scale_up",
                    "utilization": utilization,
                    "old_size": self.pool.size - changed,
                    "new_size": self.pool.size,
                    "timestamp": now,
                }

        elif utilization < self.config.scale_down_threshold:
            target = max(
                self.pool.size - self.config.scale_down_step,
                self.pool.config.min_size,
            )
            if target < self.pool.size:
                changed = await self.pool.scale(target)
                action = {
                    "action": "scale_down",
                    "utilization": utilization,
                    "old_size": self.pool.size + changed,
                    "new_size": self.pool.size,
                    "timestamp": now,
                }

        if action:
            self._last_scale_time = now
            self._scale_history.append(action)
            logger.info(
                "AutoScaler: %s pool '%s' (utilization=%.2f, size=%d)",
                action["action"], self.pool.config.name,
                utilization, self.pool.size,
            )

        return action

    async def start(self) -> None:
        """Start the auto-scaler as a background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the auto-scaler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self) -> None:
        """Background loop that periodically checks and scales."""
        while self._running:
            try:
                await self.check_and_scale()
            except Exception as e:
                logger.error("AutoScaler error: %s", e)
            await asyncio.sleep(self.config.check_interval_seconds)

    @property
    def scale_history(self) -> List[Dict[str, Any]]:
        """Get the history of scaling actions."""
        return list(self._scale_history)

    @property
    def is_running(self) -> bool:
        return self._running
