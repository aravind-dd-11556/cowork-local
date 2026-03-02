"""
Sprint 40 · Recovery Strategy Engine
======================================
Decision-tree that maps a FailureDiagnosis to the best HealingAction,
and provides concrete parameter modifications / tool alternatives.

Decision rules (in priority order):
  1. transient + has param fix    → RETRY_MODIFIED_PARAMS
  2. has alternative tool         → USE_ALTERNATIVE_TOOL
  3. high severity + rollback ok  → ROLLBACK_AND_RETRY
  4. degradation recommended      → DEGRADE
  5. else                         → ESCALATE
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .failure_diagnosis import (
    PARAM_FIX_RULES,
    TOOL_ALTERNATIVES,
    FailureDiagnosis,
)
from .self_healing import HealingAction
from .models import ToolCall


# ── Strategy engine ─────────────────────────────────────────────────

class RecoveryStrategyEngine:
    """
    Pure-logic engine that decides *what* recovery action to take and
    provides the concrete parameters for that action.
    """

    def __init__(self, custom_rules: Optional[Dict[str, Any]] = None):
        self._custom_rules = custom_rules or {}
        # track which actions were selected for metrics
        self._action_counts: Dict[str, int] = {}

    # ── primary decision ────────────────────────────────────────

    def suggest_recovery(self, diagnosis: FailureDiagnosis) -> HealingAction:
        """
        Return the best *HealingAction* for the given *diagnosis*.

        Priority:
          1. RETRY_MODIFIED_PARAMS — if transient and we have param fixes
          2. USE_ALTERNATIVE_TOOL  — if alternatives exist
          3. ROLLBACK_AND_RETRY    — if high-severity + rollback recommended
          4. DEGRADE               — if degradation recommended
          5. ESCALATE              — last resort
        """
        action = self._decide(diagnosis)
        self._action_counts[action.value] = (
            self._action_counts.get(action.value, 0) + 1
        )
        return action

    def _decide(self, d: FailureDiagnosis) -> HealingAction:
        # 1. Transient with param fix available
        if d.is_transient and d.suggested_param_changes:
            return HealingAction.RETRY_MODIFIED_PARAMS

        # 2. Alternative tool exists
        if d.suggested_alternative_tools:
            return HealingAction.USE_ALTERNATIVE_TOOL

        # 3. High severity with rollback
        if d.severity in ("high", "critical") and d.rollback_recommended:
            return HealingAction.ROLLBACK_AND_RETRY

        # 4. Degradation recommended
        if d.degradation_recommended:
            return HealingAction.DEGRADE

        # 5. Escalate
        return HealingAction.ESCALATE

    # ── param helpers ───────────────────────────────────────────

    def get_modified_params(
        self,
        tool_call: ToolCall,
        diagnosis: FailureDiagnosis,
    ) -> Dict[str, Any]:
        """
        Return a dict of parameter overrides to apply on retry.
        Merges static PARAM_FIX_RULES with any diagnosis-level suggestions.
        """
        params: Dict[str, Any] = {}
        # Static rules
        if diagnosis.error_code in PARAM_FIX_RULES:
            params.update(PARAM_FIX_RULES[diagnosis.error_code])
        # Diagnosis-level
        if diagnosis.suggested_param_changes:
            params.update(diagnosis.suggested_param_changes)
        return params

    def get_alternative_tool(
        self,
        tool_name: str,
        diagnosis: FailureDiagnosis,
    ) -> Optional[str]:
        """
        Return the best alternative tool name, or *None*.
        Prefers alternatives from the diagnosis, then from TOOL_ALTERNATIVES.
        """
        if diagnosis.suggested_alternative_tools:
            return diagnosis.suggested_alternative_tools[0]
        alts = TOOL_ALTERNATIVES.get(tool_name, [])
        return alts[0] if alts else None

    def get_degraded_params(
        self,
        tool_call: ToolCall,
        diagnosis: FailureDiagnosis,
    ) -> Dict[str, Any]:
        """
        Return parameter overrides that simplify the request
        (smaller batch, lower timeout, streaming enabled, etc.).
        """
        params: Dict[str, Any] = {}

        if diagnosis.is_transient:
            # General degradation: reduce complexity
            current_timeout = tool_call.input.get("timeout", 30)
            params["timeout"] = max(current_timeout // 2, 5)

        if diagnosis.severity in ("high", "critical"):
            params["max_results"] = min(
                tool_call.input.get("max_results", 100), 10,
            )
            params["streaming"] = True

        # Ensure at least one change
        if not params:
            params["_degraded"] = True

        return params

    # ── metrics ─────────────────────────────────────────────────

    @property
    def action_counts(self) -> Dict[str, int]:
        return dict(self._action_counts)

    def reset_counts(self) -> None:
        self._action_counts.clear()
