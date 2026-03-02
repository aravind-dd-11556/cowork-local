"""
Sprint 43 · Result Aggregator
===============================
Aggregates sub-results from multiple agents into a single output.
Supports different strategies: concatenate, merge, best-of, consensus.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Strategy ─────────────────────────────────────────────────────────

class AggregationStrategy(Enum):
    CONCATENATE = "concatenate"
    MERGE = "merge"
    BEST_OF = "best_of"
    CONSENSUS = "consensus"


# ── Aggregator ───────────────────────────────────────────────────────

class ResultAggregator:
    """
    Aggregates sub-results from crew agents into a unified output.
    """

    def __init__(self, default_strategy: AggregationStrategy = AggregationStrategy.CONCATENATE):
        self.default_strategy = default_strategy

    def aggregate(
        self,
        sub_results: List[Dict[str, Any]],
        strategy: Optional[AggregationStrategy] = None,
    ) -> str:
        """
        Aggregate a list of sub-results into a single string.

        Each sub-result dict should have:
          - "output": str (the main output text)
          - "role": str (optional, the role that produced it)
          - "confidence": float (optional, 0.0-1.0)
          - "task_description": str (optional)
        """
        strat = strategy or self.default_strategy

        if not sub_results:
            return ""

        if strat == AggregationStrategy.CONCATENATE:
            return self._concatenate(sub_results)
        elif strat == AggregationStrategy.MERGE:
            return self._merge(sub_results)
        elif strat == AggregationStrategy.BEST_OF:
            return self._best_of(sub_results)
        elif strat == AggregationStrategy.CONSENSUS:
            return self._consensus(sub_results)
        else:
            return self._concatenate(sub_results)

    # ── strategies ───────────────────────────────────────────────

    @staticmethod
    def _concatenate(results: List[Dict[str, Any]]) -> str:
        """Join results with section headers."""
        parts = []
        for i, r in enumerate(results, 1):
            role = r.get("role", f"Agent {i}")
            task = r.get("task_description", "")
            output = r.get("output", "")

            header = f"## {role}"
            if task:
                header += f": {task}"
            parts.append(f"{header}\n\n{output}")

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _merge(results: List[Dict[str, Any]]) -> str:
        """Deduplicate and merge outputs by removing duplicate lines."""
        seen_lines: set = set()
        merged_parts: List[str] = []

        for r in results:
            output = r.get("output", "")
            for line in output.split("\n"):
                stripped = line.strip()
                if stripped and stripped not in seen_lines:
                    seen_lines.add(stripped)
                    merged_parts.append(line)

        return "\n".join(merged_parts)

    @staticmethod
    def _best_of(results: List[Dict[str, Any]]) -> str:
        """Pick the result with highest confidence."""
        if not results:
            return ""

        best = max(results, key=lambda r: r.get("confidence", 0.0))
        role = best.get("role", "Agent")
        output = best.get("output", "")
        confidence = best.get("confidence", 0.0)

        return f"[Best result from {role} (confidence: {confidence:.2f})]\n\n{output}"

    @staticmethod
    def _consensus(results: List[Dict[str, Any]]) -> str:
        """Find common elements across all results."""
        if not results:
            return ""

        if len(results) == 1:
            return results[0].get("output", "")

        # Extract lines from each result
        result_line_sets = []
        for r in results:
            output = r.get("output", "")
            lines = {line.strip() for line in output.split("\n") if line.strip()}
            result_line_sets.append(lines)

        # Find lines present in at least half the results
        threshold = len(results) / 2
        all_lines: Dict[str, int] = {}
        for line_set in result_line_sets:
            for line in line_set:
                all_lines[line] = all_lines.get(line, 0) + 1

        consensus_lines = [
            line for line, count in all_lines.items()
            if count >= threshold
        ]

        if consensus_lines:
            return "[Consensus from multiple agents]\n\n" + "\n".join(
                sorted(consensus_lines, key=lambda l: -all_lines[l])
            )

        # No consensus — fall back to concatenation
        return ResultAggregator._concatenate(results)

    # ── utility ──────────────────────────────────────────────────

    @staticmethod
    def summarize_results(sub_results: List[Dict[str, Any]]) -> str:
        """Quick one-line summary of aggregation inputs."""
        count = len(sub_results)
        roles = [r.get("role", "unknown") for r in sub_results]
        return f"{count} sub-result(s) from: {', '.join(roles)}"
