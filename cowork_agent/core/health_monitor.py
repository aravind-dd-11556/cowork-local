"""
Unified Health Monitor — liveness / readiness checks for every component.

Register async health-check functions for each component (providers, scheduler,
agent registry, …).  The monitor aggregates individual results into a
``HealthReport`` with overall status:

- HEALTHY   — all components healthy.
- DEGRADED  — some components failed.
- UNHEALTHY — critical components (e.g. all providers) down.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Enums & dataclasses ─────────────────────────────────────────────

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Result of a single component's health check."""
    name: str
    status: HealthStatus
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: float = 0.0
    response_time_ms: float = 0.0


@dataclass
class HealthReport:
    """Aggregate health report across all registered components."""
    status: HealthStatus
    components: List[ComponentHealth] = field(default_factory=list)
    timestamp: float = 0.0
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "response_time_ms": round(c.response_time_ms, 2),
                    "details": c.details,
                }
                for c in self.components
            ],
        }


# ── HealthMonitor ───────────────────────────────────────────────────

class HealthMonitor:
    """
    Central registry of health-check functions.

    Each registered component provides an async callable that returns a dict:

    .. code-block:: python

        async def my_check() -> dict:
            return {"status": "ok", "latency_ms": 42}

    The dict **must** contain a ``"status"`` key whose value is one of
    ``"ok"``, ``"healthy"``, ``"degraded"``, or ``"unhealthy"`` /
    ``"error"`` / ``"failed"``.
    """

    def __init__(self) -> None:
        self._components: Dict[str, Callable[[], Awaitable[dict]]] = {}
        self._start_time: float = time.time()
        self._last_report: Optional[HealthReport] = None
        self._shutdown_flag: bool = False

    # ── Registration ───────────────────────────────────────────

    def register_component(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[dict]],
    ) -> None:
        """Register a named health-check function."""
        self._components[name] = check_fn

    def unregister_component(self, name: str) -> None:
        """Remove a component by name (no-op if missing)."""
        self._components.pop(name, None)

    @property
    def component_names(self) -> List[str]:
        return list(self._components.keys())

    # ── Shutdown awareness ─────────────────────────────────────

    def set_shutting_down(self, value: bool = True) -> None:
        """Mark the system as shutting down (affects readiness)."""
        self._shutdown_flag = value

    # ── Health checks ──────────────────────────────────────────

    async def check_health(self) -> HealthReport:
        """Run all registered health checks and produce a report."""
        now = time.time()
        components: List[ComponentHealth] = []

        for name, check_fn in self._components.items():
            comp = await self._check_one(name, check_fn)
            components.append(comp)

        overall = self._aggregate_status(components)

        report = HealthReport(
            status=overall,
            components=components,
            timestamp=now,
            uptime_seconds=now - self._start_time,
        )
        self._last_report = report
        return report

    async def check_liveness(self) -> bool:
        """
        Liveness probe — is the process alive and the event loop responding?

        Always True when this code executes.
        """
        return True

    async def check_readiness(self) -> bool:
        """
        Readiness probe — can the system accept work?

        False if shutting down or if the most recent health report is UNHEALTHY.
        """
        if self._shutdown_flag:
            return False

        # If we haven't run a health check yet, run one now
        report = self._last_report
        if report is None:
            report = await self.check_health()

        return report.status != HealthStatus.UNHEALTHY

    def get_last_report(self) -> Optional[HealthReport]:
        """Return the most recently computed health report (or None)."""
        return self._last_report

    # ── Internal helpers ───────────────────────────────────────

    async def _check_one(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[dict]],
    ) -> ComponentHealth:
        """Run a single component's health check with timing."""
        start = time.time()
        try:
            result = await asyncio.wait_for(check_fn(), timeout=10.0)
            elapsed_ms = (time.time() - start) * 1000

            status = self._parse_status(result.get("status", "ok"))
            return ComponentHealth(
                name=name,
                status=status,
                details=result,
                checked_at=start,
                response_time_ms=elapsed_ms,
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.time() - start) * 1000
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                details={"error": "Health check timed out"},
                checked_at=start,
                response_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.time() - start) * 1000
            logger.warning("Health check failed for %s: %s", name, exc)
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                details={"error": str(exc)},
                checked_at=start,
                response_time_ms=elapsed_ms,
            )

    @staticmethod
    def _parse_status(raw: str) -> HealthStatus:
        """Normalise a free-form status string into HealthStatus."""
        raw_lower = raw.lower()
        if raw_lower in ("ok", "healthy", "up"):
            return HealthStatus.HEALTHY
        elif raw_lower == "degraded":
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY

    @staticmethod
    def _aggregate_status(components: List[ComponentHealth]) -> HealthStatus:
        """Derive an overall status from individual component statuses."""
        if not components:
            return HealthStatus.HEALTHY

        statuses = {c.status for c in components}
        if statuses == {HealthStatus.HEALTHY}:
            return HealthStatus.HEALTHY
        elif HealthStatus.UNHEALTHY in statuses:
            # If all are unhealthy → UNHEALTHY; mix → DEGRADED
            if statuses == {HealthStatus.UNHEALTHY}:
                return HealthStatus.UNHEALTHY
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.DEGRADED
