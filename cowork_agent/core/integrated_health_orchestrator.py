"""
Integrated Health Check Orchestrator — unified health checking with correlation IDs.

Extends HealthMonitor with trend tracking, component metrics, failure
prediction, and event bus integration.

Sprint 16 (Testing & Observability Hardening) Module 6.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Dict, List, Optional

from .health_monitor import HealthMonitor, HealthReport, HealthStatus, ComponentHealth

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class HealthTrend:
    """Historical health trend for a component."""
    component_name: str
    reports: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 100

    def add(self, status: str, response_time_ms: float, timestamp: float) -> None:
        self.reports.append({
            "status": status,
            "response_time_ms": round(response_time_ms, 2),
            "timestamp": timestamp,
        })
        if len(self.reports) > self.max_history:
            self.reports = self.reports[-self.max_history:]

    @property
    def failure_rate(self) -> float:
        if not self.reports:
            return 0.0
        failures = sum(
            1 for r in self.reports if r["status"] in ("unhealthy", "degraded")
        )
        return failures / len(self.reports)

    @property
    def avg_response_time_ms(self) -> float:
        if not self.reports:
            return 0.0
        return sum(r["response_time_ms"] for r in self.reports) / len(self.reports)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_name": self.component_name,
            "total_checks": len(self.reports),
            "failure_rate": round(self.failure_rate, 4),
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "recent": self.reports[-5:] if self.reports else [],
        }


@dataclass
class FailurePrediction:
    """Prediction of component failure likelihood."""
    component_name: str
    failure_probability: float  # 0.0–1.0
    trend_direction: str  # "improving", "stable", "degrading"
    confidence: float  # 0.0–1.0
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_name": self.component_name,
            "failure_probability": round(self.failure_probability, 4),
            "trend_direction": self.trend_direction,
            "confidence": round(self.confidence, 4),
            "recommendation": self.recommendation,
        }


# ── IntegratedHealthOrchestrator ────────────────────────────────

class IntegratedHealthOrchestrator(HealthMonitor):
    """
    Extended health monitoring with trends, predictions, and event integration.

    Usage::

        orch = IntegratedHealthOrchestrator()
        orch.register_component("provider", my_check_fn)
        report = await orch.run_full_check(correlation_id="abc123")
        trends = orch.get_trends("provider")
    """

    def __init__(
        self,
        max_trend_history: int = 100,
        event_bus: Optional[Any] = None,
        correlation_manager: Optional[Any] = None,
    ):
        super().__init__()
        self._max_trend_history = max_trend_history
        self._trends: Dict[str, HealthTrend] = {}
        self._check_history: List[HealthReport] = []
        self._max_check_history = 50
        self._event_bus = event_bus
        self._correlation_manager = correlation_manager
        self._events_enabled: bool = event_bus is not None

    # ── Extended health checks ────────────────────────────────

    async def run_full_check(
        self,
        correlation_id: str = "",
    ) -> HealthReport:
        """
        Run all health checks with correlation ID and trend tracking.

        Emits HEALTH_CHECK_RUN event if event bus is configured.
        """
        report = await self.check_health()

        # Track trends for each component
        for comp in report.components:
            trend = self._trends.setdefault(
                comp.name,
                HealthTrend(component_name=comp.name, max_history=self._max_trend_history),
            )
            trend.add(
                status=comp.status.value,
                response_time_ms=comp.response_time_ms,
                timestamp=comp.checked_at,
            )

        # Store in check history
        self._check_history.append(report)
        if len(self._check_history) > self._max_check_history:
            self._check_history = self._check_history[-self._max_check_history:]

        # Emit event
        if self._events_enabled and self._event_bus is not None:
            try:
                from .observability_event_bus import ObservabilityEvent, EventType
                event = ObservabilityEvent(
                    event_type=EventType.HEALTH_CHECK_RUN,
                    component="health_orchestrator",
                    trace_id=correlation_id,
                    severity="info" if report.status == HealthStatus.HEALTHY else "warning",
                    metadata={
                        "status": report.status.value,
                        "component_count": len(report.components),
                        "unhealthy_components": [
                            c.name for c in report.components
                            if c.status == HealthStatus.UNHEALTHY
                        ],
                    },
                )
                self._event_bus.emit(event)
            except Exception as exc:
                logger.debug("Failed to emit health check event: %s", exc)

        return report

    async def run_component_check(
        self,
        component_name: str,
        correlation_id: str = "",
    ) -> Optional[ComponentHealth]:
        """Run a health check for a single component."""
        check_fn = self._components.get(component_name)
        if check_fn is None:
            return None

        result = await self._check_one(component_name, check_fn)

        # Track trend
        trend = self._trends.setdefault(
            component_name,
            HealthTrend(component_name=component_name, max_history=self._max_trend_history),
        )
        trend.add(
            status=result.status.value,
            response_time_ms=result.response_time_ms,
            timestamp=result.checked_at,
        )

        return result

    # ── Trends ────────────────────────────────────────────────

    def get_trends(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Get health trends for a component or all components."""
        if component_name:
            trend = self._trends.get(component_name)
            if trend:
                return {component_name: trend.to_dict()}
            return {}
        return {name: trend.to_dict() for name, trend in self._trends.items()}

    def get_check_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent full health check reports."""
        return [r.to_dict() for r in self._check_history[-limit:]]

    # ── Failure prediction ────────────────────────────────────

    def predict_failure(self, component_name: str) -> Optional[FailurePrediction]:
        """
        Predict failure likelihood based on trends.

        Returns None if insufficient data.
        """
        trend = self._trends.get(component_name)
        if trend is None or len(trend.reports) < 5:
            return None

        # Analyze recent vs historical failure rates
        total = len(trend.reports)
        mid = total // 2
        recent = trend.reports[mid:]
        historical = trend.reports[:mid]

        recent_failures = sum(
            1 for r in recent if r["status"] in ("unhealthy", "degraded")
        )
        hist_failures = sum(
            1 for r in historical if r["status"] in ("unhealthy", "degraded")
        )

        recent_rate = recent_failures / len(recent) if recent else 0.0
        hist_rate = hist_failures / len(historical) if historical else 0.0

        # Determine trend direction
        if recent_rate > hist_rate + 0.1:
            direction = "degrading"
        elif recent_rate < hist_rate - 0.1:
            direction = "improving"
        else:
            direction = "stable"

        # Failure probability (weighted recent more heavily)
        prob = recent_rate * 0.7 + hist_rate * 0.3

        # Confidence based on data size
        confidence = min(1.0, total / 20.0)

        # Recommendation
        if prob > 0.5:
            rec = f"High failure risk for {component_name}; investigate immediately"
        elif prob > 0.2:
            rec = f"Moderate failure risk for {component_name}; monitor closely"
        elif direction == "degrading":
            rec = f"{component_name} is degrading; watch for further deterioration"
        else:
            rec = f"{component_name} is healthy; no action needed"

        return FailurePrediction(
            component_name=component_name,
            failure_probability=prob,
            trend_direction=direction,
            confidence=confidence,
            recommendation=rec,
        )

    # ── Configuration ─────────────────────────────────────────

    def enable_events(self, enabled: bool) -> None:
        """Enable or disable event emission."""
        self._events_enabled = enabled

    def set_event_bus(self, event_bus: Any) -> None:
        """Set or update the event bus reference."""
        self._event_bus = event_bus
        self._events_enabled = event_bus is not None

    def set_correlation_manager(self, manager: Any) -> None:
        """Set or update the correlation manager reference."""
        self._correlation_manager = manager

    # ── Extended stats ────────────────────────────────────────

    def orchestrator_stats(self) -> Dict[str, Any]:
        """Return detailed orchestrator statistics."""
        return {
            "registered_components": list(self._components.keys()),
            "total_checks_run": len(self._check_history),
            "trends_tracked": list(self._trends.keys()),
            "events_enabled": self._events_enabled,
            "last_check_status": (
                self._last_report.status.value if self._last_report else None
            ),
        }

    def reset_trends(self) -> None:
        """Clear all trend data."""
        self._trends.clear()
        self._check_history.clear()
