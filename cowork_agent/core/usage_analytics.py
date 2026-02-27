"""
Usage Analytics — Session-level analytics aggregating costs, metrics, and routing decisions.

Combines data from CostTracker, MetricsCollector, and ProviderHealthTracker
into comprehensive session reports with efficiency scoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .model_router import ModelTier


@dataclass
class RoutingDecision:
    """Log entry for a model routing decision."""
    timestamp: float
    tier: str                  # ModelTier.value
    provider: str
    model: str
    reason: str
    escalated: bool = False

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "tier": self.tier,
            "provider": self.provider,
            "model": self.model,
            "reason": self.reason,
            "escalated": self.escalated,
        }


class UsageAnalytics:
    """
    Aggregates session-level analytics from multiple sources.

    Combines:
        - CostTracker: spending, per-model costs
        - MetricsCollector: tool latencies, call counts
        - ProviderHealthTracker: provider health scores
        - Routing decisions: which model was chosen and why
    """

    def __init__(
        self,
        cost_tracker=None,           # CostTracker instance
        metrics_collector=None,      # MetricsCollector instance
        health_tracker=None,         # ProviderHealthTracker instance
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._cost_tracker = cost_tracker
        self._metrics_collector = metrics_collector
        self._health_tracker = health_tracker
        self._routing_decisions: list[RoutingDecision] = []
        self._session_start = time.time()

    # ── Routing log ───────────────────────────────────────────────

    def record_routing_decision(
        self,
        tier: ModelTier,
        provider: str,
        model: str,
        reason: str,
        escalated: bool = False,
    ) -> None:
        """Log a model routing decision."""
        if not self._enabled:
            return
        self._routing_decisions.append(RoutingDecision(
            timestamp=time.time(),
            tier=tier.value,
            provider=provider,
            model=model,
            reason=reason,
            escalated=escalated,
        ))

    @property
    def routing_decisions(self) -> list[RoutingDecision]:
        return list(self._routing_decisions)

    # ── Session report ────────────────────────────────────────────

    def session_report(self) -> dict:
        """
        Generate a comprehensive session report.

        Returns:
            dict with sections: session, cost, routing, tools, providers, recommendations
        """
        elapsed = time.time() - self._session_start

        report: dict = {
            "session": {
                "duration_seconds": round(elapsed, 1),
                "start_time": self._session_start,
            },
        }

        # Cost section
        if self._cost_tracker:
            report["cost"] = self._cost_tracker.summary()
        else:
            report["cost"] = {"total_cost": 0, "note": "Cost tracking not enabled"}

        # Routing section
        tier_counts: dict[str, int] = {}
        escalation_count = 0
        for d in self._routing_decisions:
            tier_counts[d.tier] = tier_counts.get(d.tier, 0) + 1
            if d.escalated:
                escalation_count += 1

        report["routing"] = {
            "total_decisions": len(self._routing_decisions),
            "tier_distribution": tier_counts,
            "escalation_count": escalation_count,
            "recent_decisions": [
                d.to_dict() for d in self._routing_decisions[-5:]
            ],
        }

        # Tool metrics section
        if self._metrics_collector:
            report["tools"] = self._metrics_collector.summary()
        else:
            report["tools"] = {"note": "Metrics collector not enabled"}

        # Provider health section
        if self._health_tracker:
            report["providers"] = self._health_tracker.to_dict()
        else:
            report["providers"] = {"note": "Health tracking not enabled"}

        # Recommendations
        report["recommendations"] = self._generate_recommendations(report)

        return report

    # ── Efficiency score ──────────────────────────────────────────

    def efficiency_score(self) -> float:
        """
        Calculate a 0–100 efficiency score for the session.

        Factors:
            - Cost efficiency (low cost = higher score)
            - Routing efficiency (more FAST tier usage = higher score)
            - Error rate (fewer errors = higher score)
            - Cache utilization (more cache hits = higher score)

        Returns 50.0 if insufficient data.
        """
        if not self._routing_decisions:
            return 50.0

        scores: list[float] = []

        # 1. Routing efficiency (0–100)
        tier_counts: dict[str, int] = {}
        for d in self._routing_decisions:
            tier_counts[d.tier] = tier_counts.get(d.tier, 0) + 1
        total_decisions = len(self._routing_decisions)
        fast_pct = tier_counts.get("fast", 0) / total_decisions if total_decisions else 0
        balanced_pct = tier_counts.get("balanced", 0) / total_decisions if total_decisions else 0
        # More FAST usage → more efficient, weighted scoring
        routing_score = (fast_pct * 100) + (balanced_pct * 60) + ((1 - fast_pct - balanced_pct) * 30)
        scores.append(min(routing_score, 100))

        # 2. Escalation penalty (fewer escalations = better)
        escalation_pct = sum(1 for d in self._routing_decisions if d.escalated) / total_decisions
        escalation_score = (1 - escalation_pct) * 100
        scores.append(escalation_score)

        # 3. Cost efficiency (based on cache savings ratio)
        if self._cost_tracker:
            cost_summary = self._cost_tracker.summary()
            total_cost = cost_summary.get("total_cost", 0)
            cache_savings = cost_summary.get("total_cache_savings", 0)
            if total_cost > 0:
                savings_ratio = cache_savings / (total_cost + cache_savings)
                scores.append(savings_ratio * 100 + 50)  # base 50 + savings bonus
            else:
                scores.append(80)  # zero cost is great

        # 4. Tool error rate
        if self._metrics_collector:
            metrics = self._metrics_collector.summary()
            total_calls = metrics.get("total_tool_calls", 0)
            total_errors = metrics.get("total_errors", 0)
            if total_calls > 0:
                tool_success_rate = 1 - (total_errors / total_calls)
                scores.append(tool_success_rate * 100)

        return round(sum(scores) / len(scores), 1) if scores else 50.0

    # ── Recommendations ───────────────────────────────────────────

    def _generate_recommendations(self, report: dict) -> list[str]:
        """Generate actionable recommendations from the session data."""
        recs: list[str] = []

        # Check routing distribution
        routing = report.get("routing", {})
        tier_dist = routing.get("tier_distribution", {})
        total_decisions = routing.get("total_decisions", 0)

        if total_decisions > 5:
            powerful_pct = tier_dist.get("powerful", 0) / total_decisions
            if powerful_pct > 0.7:
                recs.append(
                    f"{powerful_pct:.0%} of calls used POWERFUL tier — "
                    "consider upgrading default model for this workload"
                )
            fast_pct = tier_dist.get("fast", 0) / total_decisions
            if fast_pct > 0.8:
                recs.append(
                    f"{fast_pct:.0%} of calls used FAST tier — "
                    "workload is simple, downgrading default model could save costs"
                )

        # Check escalation rate
        escalation_count = routing.get("escalation_count", 0)
        if total_decisions > 3 and escalation_count / total_decisions > 0.3:
            recs.append(
                f"High escalation rate ({escalation_count}/{total_decisions}) — "
                "consider starting with a higher tier to avoid retries"
            )

        # Check provider health
        providers = report.get("providers", {})
        provider_data = providers.get("providers", {})
        for name, data in provider_data.items():
            if isinstance(data, dict):
                error_rate = data.get("error_rate", 0)
                if error_rate > 0.2:
                    recs.append(
                        f"Provider '{name}' has {error_rate:.0%} error rate — "
                        "check API key and connectivity"
                    )

        # Check cost
        cost = report.get("cost", {})
        if cost.get("is_over_budget"):
            recs.append(
                "Session exceeded budget — consider lowering default tier "
                "or reducing max iterations"
            )

        if not recs:
            recs.append("Session looks healthy — no optimization recommendations")

        return recs

    def reset(self) -> None:
        """Clear routing decisions and reset session timer."""
        self._routing_decisions.clear()
        self._session_start = time.time()
