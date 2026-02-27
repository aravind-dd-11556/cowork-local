"""
Token Usage Store — persist token tracking across sessions with budget alerts.

Stores snapshots in ``{workspace}/.cowork/token_usage.json`` and provides
daily/weekly/monthly summaries plus configurable budget alerting.
"""

from __future__ import annotations

import json
import logging
import os
import time
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class TokenUsageSnapshot:
    """A single recorded snapshot of token usage."""
    session_id: str
    timestamp: float
    session_input: int
    session_output: int
    session_cache_read: int = 0
    session_cache_write: int = 0
    call_count: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def session_total(self) -> int:
        return self.session_input + self.session_output

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "session_input": self.session_input,
            "session_output": self.session_output,
            "session_cache_read": self.session_cache_read,
            "session_cache_write": self.session_cache_write,
            "call_count": self.call_count,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenUsageSnapshot":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TokenBudgetAlert:
    """Alert emitted when a budget threshold is crossed."""
    alert_type: str        # "approaching" | "exceeded"
    metric: str            # "tokens" | "cost"
    current_value: float
    limit: float
    percent_used: float
    message: str


# ── TokenUsageStore ─────────────────────────────────────────────────

class TokenUsageStore:
    """
    Persistent token usage tracking with budget alerting.

    Storage format (``token_usage.json``)::

        {
          "snapshots": [...],
          "budget_config": { "max_session_tokens": ..., ... }
        }
    """

    def __init__(
        self,
        workspace_dir: str,
        max_session_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
        warning_threshold_percent: float = 80.0,
    ):
        self.workspace_dir = workspace_dir
        self.max_session_tokens = max_session_tokens
        self.max_cost_usd = max_cost_usd
        self.warning_threshold_percent = warning_threshold_percent

        self._store_path = os.path.join(workspace_dir, ".cowork", "token_usage.json")
        self._snapshots: List[TokenUsageSnapshot] = []
        self._load_from_disk()

    # ── Recording ──────────────────────────────────────────────

    def record_snapshot(
        self,
        session_id: str,
        tracker: Any,          # TokenTracker — kept as Any to avoid circular import
    ) -> List[TokenBudgetAlert]:
        """
        Record a token-usage snapshot from a TokenTracker instance.

        Returns a list of budget alerts (empty if no thresholds crossed).
        """
        snapshot = TokenUsageSnapshot(
            session_id=session_id,
            timestamp=time.time(),
            session_input=tracker.total_input_tokens,
            session_output=tracker.total_output_tokens,
            session_cache_read=getattr(tracker, "total_cache_read_tokens", 0),
            session_cache_write=getattr(tracker, "total_cache_write_tokens", 0),
            call_count=tracker.call_count,
            estimated_cost_usd=tracker.estimated_cost_usd,
        )
        self._snapshots.append(snapshot)
        self._save_to_disk()

        return self._check_budget(tracker)

    # ── Cumulative properties ──────────────────────────────────

    @property
    def cumulative_input(self) -> int:
        return sum(s.session_input for s in self._snapshots)

    @property
    def cumulative_output(self) -> int:
        return sum(s.session_output for s in self._snapshots)

    @property
    def cumulative_total(self) -> int:
        return self.cumulative_input + self.cumulative_output

    @property
    def cumulative_cost(self) -> float:
        return sum(s.estimated_cost_usd for s in self._snapshots)

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    # ── Summaries ──────────────────────────────────────────────

    def daily_summary(self, date: Optional[datetime] = None) -> dict:
        """Token usage summary for a specific day."""
        if date is None:
            date = datetime.now()
        start = date.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        end = start + 86400
        return self._summarise_range(start, end, date.strftime("%Y-%m-%d"))

    def weekly_summary(self, week_start: Optional[datetime] = None) -> dict:
        """Token usage summary for a 7-day period."""
        if week_start is None:
            today = datetime.now()
            week_start = today - timedelta(days=today.weekday())
        start = week_start.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        end = start + 7 * 86400
        return self._summarise_range(start, end, f"week of {week_start.strftime('%Y-%m-%d')}")

    def monthly_summary(self, year: int, month: int) -> dict:
        """Token usage summary for a calendar month."""
        start = datetime(year, month, 1).timestamp()
        _, days = monthrange(year, month)
        end = start + days * 86400
        return self._summarise_range(start, end, f"{year}-{month:02d}")

    # ── Internal helpers ───────────────────────────────────────

    def _summarise_range(self, start: float, end: float, label: str) -> dict:
        snaps = [s for s in self._snapshots if start <= s.timestamp < end]
        return {
            "period": label,
            "token_count": sum(s.session_total for s in snaps),
            "input_tokens": sum(s.session_input for s in snaps),
            "output_tokens": sum(s.session_output for s in snaps),
            "cost_usd": round(sum(s.estimated_cost_usd for s in snaps), 6),
            "calls": sum(s.call_count for s in snaps),
            "sessions": len({s.session_id for s in snaps}),
        }

    def _check_budget(self, tracker: Any) -> List[TokenBudgetAlert]:
        alerts: List[TokenBudgetAlert] = []

        if self.max_session_tokens:
            pct = (tracker.total_tokens / self.max_session_tokens) * 100
            if pct >= 100:
                alerts.append(TokenBudgetAlert(
                    alert_type="exceeded", metric="tokens",
                    current_value=tracker.total_tokens,
                    limit=self.max_session_tokens, percent_used=pct,
                    message=f"Token budget EXCEEDED: {tracker.total_tokens}/{self.max_session_tokens}",
                ))
            elif pct >= self.warning_threshold_percent:
                alerts.append(TokenBudgetAlert(
                    alert_type="approaching", metric="tokens",
                    current_value=tracker.total_tokens,
                    limit=self.max_session_tokens, percent_used=pct,
                    message=f"Warning: {pct:.0f}% of token budget used",
                ))

        if self.max_cost_usd is not None:
            cost = tracker.estimated_cost_usd
            pct = (cost / self.max_cost_usd) * 100 if self.max_cost_usd > 0 else 0
            if pct >= 100:
                alerts.append(TokenBudgetAlert(
                    alert_type="exceeded", metric="cost",
                    current_value=cost, limit=self.max_cost_usd,
                    percent_used=pct,
                    message=f"Cost budget EXCEEDED: ${cost:.4f}/${self.max_cost_usd}",
                ))
            elif pct >= self.warning_threshold_percent:
                alerts.append(TokenBudgetAlert(
                    alert_type="approaching", metric="cost",
                    current_value=cost, limit=self.max_cost_usd,
                    percent_used=pct,
                    message=f"Warning: {pct:.0f}% of cost budget used",
                ))

        return alerts

    def _load_from_disk(self) -> None:
        if not os.path.exists(self._store_path):
            return
        try:
            with open(self._store_path, "r") as f:
                data = json.load(f)
            for snap_data in data.get("snapshots", []):
                self._snapshots.append(TokenUsageSnapshot.from_dict(snap_data))
        except Exception as exc:
            logger.warning("Failed to load token usage store: %s", exc)

    def _save_to_disk(self) -> None:
        os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
        try:
            data = {
                "snapshots": [s.to_dict() for s in self._snapshots],
                "budget_config": {
                    "max_session_tokens": self.max_session_tokens,
                    "max_cost_usd": self.max_cost_usd,
                    "warning_threshold_percent": self.warning_threshold_percent,
                },
            }
            with open(self._store_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save token usage store: %s", exc)
