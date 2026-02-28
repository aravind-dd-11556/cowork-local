"""
Performance Benchmark — utilities for latency analysis and regression detection.

Records benchmark runs, computes statistical summaries, compares runs,
detects trends, and exports reports in JSON/Markdown.

Sprint 16 (Testing & Observability Hardening) Module 4.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class BenchmarkRun:
    """A single benchmark measurement."""
    name: str
    duration_ms: float
    component: str = ""  # tool, provider, module
    success: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "component": self.component,
            "success": self.success,
            "tags": self.tags,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkStats:
    """Statistical summary for a set of benchmark runs."""
    name: str
    count: int
    min_ms: float
    max_ms: float
    avg_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float
    success_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "avg_ms": round(self.avg_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
            "success_rate": round(self.success_rate, 4),
        }


@dataclass
class ComparisonReport:
    """Comparison between two benchmark sets."""
    name_a: str
    name_b: str
    avg_diff_ms: float
    avg_diff_percent: float
    faster: str  # Which one is faster
    statistically_significant: bool
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "avg_diff_ms": round(self.avg_diff_ms, 2),
            "avg_diff_percent": round(self.avg_diff_percent, 2),
            "faster": self.faster,
            "statistically_significant": self.statistically_significant,
            "recommendation": self.recommendation,
        }


# ── PerformanceBenchmark ────────────────────────────────────────

class PerformanceBenchmark:
    """
    Performance benchmarking and analysis.

    Usage::

        bench = PerformanceBenchmark()
        bench.record("tool_bash", 42.5, component="tool")
        bench.record("tool_bash", 38.1, component="tool")
        stats = bench.get_stats("tool_bash")
        report = bench.export_report("markdown")
    """

    def __init__(self, max_runs: int = 1000):
        self._runs: Dict[str, List[BenchmarkRun]] = {}
        self._max_runs = max_runs

    # ── Recording ─────────────────────────────────────────────

    def record(
        self,
        name: str,
        duration_ms: float,
        component: str = "",
        success: bool = True,
        tags: Optional[List[str]] = None,
        **metadata: Any,
    ) -> BenchmarkRun:
        """Record a single benchmark run."""
        run = BenchmarkRun(
            name=name,
            duration_ms=duration_ms,
            component=component,
            success=success,
            tags=tags or [],
            metadata=metadata,
        )
        runs = self._runs.setdefault(name, [])
        runs.append(run)
        if len(runs) > self._max_runs:
            self._runs[name] = runs[-self._max_runs:]
        return run

    # ── Statistics ────────────────────────────────────────────

    def get_stats(self, name: str) -> Optional[BenchmarkStats]:
        """Get statistical summary for a benchmark name."""
        runs = self._runs.get(name)
        if not runs:
            return None
        return self._compute_stats(name, runs)

    def get_all_stats(self) -> Dict[str, BenchmarkStats]:
        """Get statistics for all benchmarks."""
        return {
            name: self._compute_stats(name, runs)
            for name, runs in self._runs.items()
            if runs
        }

    def _compute_stats(self, name: str, runs: List[BenchmarkRun]) -> BenchmarkStats:
        durations = [r.duration_ms for r in runs]
        sorted_d = sorted(durations)
        n = len(sorted_d)
        avg = sum(sorted_d) / n
        variance = sum((d - avg) ** 2 for d in sorted_d) / n if n > 1 else 0.0
        std_dev = math.sqrt(variance)
        successes = sum(1 for r in runs if r.success)

        return BenchmarkStats(
            name=name,
            count=n,
            min_ms=sorted_d[0],
            max_ms=sorted_d[-1],
            avg_ms=avg,
            median_ms=self._percentile(sorted_d, 50),
            p95_ms=self._percentile(sorted_d, 95),
            p99_ms=self._percentile(sorted_d, 99),
            std_dev_ms=std_dev,
            success_rate=successes / n if n > 0 else 0.0,
        )

    # ── Comparison ────────────────────────────────────────────

    def compare(self, name_a: str, name_b: str) -> Optional[ComparisonReport]:
        """
        Compare two benchmarks statistically.

        Returns None if either benchmark has insufficient data.
        """
        stats_a = self.get_stats(name_a)
        stats_b = self.get_stats(name_b)
        if stats_a is None or stats_b is None:
            return None
        if stats_a.count < 2 or stats_b.count < 2:
            return None

        diff = stats_a.avg_ms - stats_b.avg_ms
        pct = (diff / stats_a.avg_ms * 100) if stats_a.avg_ms > 0 else 0.0
        faster = name_b if diff > 0 else name_a

        # Simple significance check: diff > combined std_dev
        combined_std = math.sqrt(stats_a.std_dev_ms ** 2 + stats_b.std_dev_ms ** 2)
        significant = abs(diff) > combined_std if combined_std > 0 else False

        if significant:
            rec = f"{faster} is significantly faster by {abs(diff):.1f}ms ({abs(pct):.1f}%)"
        else:
            rec = "No significant difference between the two benchmarks"

        return ComparisonReport(
            name_a=name_a,
            name_b=name_b,
            avg_diff_ms=diff,
            avg_diff_percent=pct,
            faster=faster,
            statistically_significant=significant,
            recommendation=rec,
        )

    # ── Ranking ───────────────────────────────────────────────

    def get_slowest(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return the slowest benchmarks by average latency."""
        all_stats = self.get_all_stats()
        ranked = sorted(all_stats.values(), key=lambda s: s.avg_ms, reverse=True)
        return [s.to_dict() for s in ranked[:top_n]]

    def get_by_tag(self, tag: str) -> Dict[str, BenchmarkStats]:
        """Get statistics for all benchmarks matching a tag."""
        result: Dict[str, BenchmarkStats] = {}
        for name, runs in self._runs.items():
            tagged = [r for r in runs if tag in r.tags]
            if tagged:
                result[name] = self._compute_stats(name, tagged)
        return result

    def get_by_component(self, component: str) -> Dict[str, BenchmarkStats]:
        """Get statistics for all benchmarks of a component type."""
        result: Dict[str, BenchmarkStats] = {}
        for name, runs in self._runs.items():
            matching = [r for r in runs if r.component == component]
            if matching:
                result[name] = self._compute_stats(name, matching)
        return result

    # ── Trend detection ───────────────────────────────────────

    def detect_regression(
        self,
        name: str,
        window_size: int = 10,
        threshold_percent: float = 20.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect performance regression by comparing recent runs to historical.

        Returns a dict with regression details, or None if no regression.
        """
        runs = self._runs.get(name, [])
        if len(runs) < window_size * 2:
            return None

        historical = runs[-(window_size * 2):-window_size]
        recent = runs[-window_size:]

        hist_avg = sum(r.duration_ms for r in historical) / len(historical)
        recent_avg = sum(r.duration_ms for r in recent) / len(recent)

        if hist_avg <= 0:
            return None

        change_pct = ((recent_avg - hist_avg) / hist_avg) * 100

        if change_pct > threshold_percent:
            return {
                "name": name,
                "regression": True,
                "historical_avg_ms": round(hist_avg, 2),
                "recent_avg_ms": round(recent_avg, 2),
                "change_percent": round(change_pct, 2),
                "window_size": window_size,
            }
        return None

    # ── Export ─────────────────────────────────────────────────

    def export_report(self, format: str = "json") -> str:
        """Export all benchmark data.  Formats: json, markdown."""
        if format == "markdown":
            return self._export_markdown()
        return self._export_json()

    def _export_json(self) -> str:
        all_stats = self.get_all_stats()
        data = {
            "benchmarks": {
                name: stats.to_dict() for name, stats in all_stats.items()
            },
            "total_benchmarks": len(all_stats),
            "total_runs": sum(s.count for s in all_stats.values()),
        }
        return json.dumps(data, indent=2)

    def _export_markdown(self) -> str:
        all_stats = self.get_all_stats()
        if not all_stats:
            return "# Performance Benchmark Report\n\nNo benchmark data recorded."

        lines = [
            "# Performance Benchmark Report",
            "",
            f"**Total benchmarks:** {len(all_stats)}",
            f"**Total runs:** {sum(s.count for s in all_stats.values())}",
            "",
            "| Benchmark | Count | Avg (ms) | P95 (ms) | P99 (ms) | Std Dev | Success % |",
            "|-----------|-------|----------|----------|----------|---------|-----------|",
        ]

        for stats in sorted(all_stats.values(), key=lambda s: s.avg_ms, reverse=True):
            lines.append(
                f"| {stats.name} | {stats.count} | {stats.avg_ms:.1f} "
                f"| {stats.p95_ms:.1f} | {stats.p99_ms:.1f} "
                f"| {stats.std_dev_ms:.1f} | {stats.success_rate*100:.1f}% |"
            )

        return "\n".join(lines)

    # ── Reset & history ───────────────────────────────────────

    def get_runs(self, name: Optional[str] = None) -> List[BenchmarkRun]:
        """Get raw run data."""
        if name:
            return list(self._runs.get(name, []))
        all_runs: List[BenchmarkRun] = []
        for runs in self._runs.values():
            all_runs.extend(runs)
        return all_runs

    def reset(self) -> None:
        """Clear all benchmark data."""
        self._runs.clear()

    # ── Internal ──────────────────────────────────────────────

    @staticmethod
    def _percentile(sorted_data: List[float], p: int) -> float:
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
