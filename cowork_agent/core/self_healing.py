"""
Sprint 40 · Self-Healing Pipelines
===================================
SelfHealingEngine wraps tool execution with automatic diagnosis → recovery.

Flow:
    execute tool → failure? → diagnose → pick strategy → execute recovery
    → repeat up to max_recovery_attempts → escalate to user

Integrates with:
  - FailureDiagnostic  (error analysis)
  - RecoveryStrategyEngine  (action selection)
  - RollbackJournal    (state rollback)
  - StateSnapshotManager (state capture)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .models import ToolCall, ToolResult


# ── Enums & Config ──────────────────────────────────────────────────

class HealingAction(Enum):
    """Recovery action the self-healing engine can take."""
    RETRY_MODIFIED_PARAMS = "retry_modified_params"
    USE_ALTERNATIVE_TOOL = "use_alternative_tool"
    ROLLBACK_AND_RETRY = "rollback_and_retry"
    DEGRADE = "degrade"
    ESCALATE = "escalate"


@dataclass
class SelfHealingConfig:
    """Tuning knobs for self-healing behaviour."""
    enabled: bool = True
    max_recovery_attempts: int = 2
    auto_rollback_on_failure: bool = True
    recovery_timeout_seconds: float = 60.0
    escalation_strategy: str = "user_confirm"  # user_confirm | immediate

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "max_recovery_attempts": self.max_recovery_attempts,
            "auto_rollback_on_failure": self.auto_rollback_on_failure,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
            "escalation_strategy": self.escalation_strategy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SelfHealingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Recovery history ────────────────────────────────────────────────

@dataclass
class RecoveryAttempt:
    """A single recovery attempt record."""
    action: HealingAction
    success: bool
    detail: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "success": self.success,
            "detail": self.detail,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class RecoveryHistory:
    """Full record of recovery attempts for one failed tool call."""
    original_tool_call: ToolCall
    original_error: str
    attempts: List[RecoveryAttempt] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    final_action: Optional[HealingAction] = None
    recovered: bool = False

    def to_dict(self) -> dict:
        return {
            "original_tool": self.original_tool_call.name,
            "original_error": self.original_error,
            "attempts": [a.to_dict() for a in self.attempts],
            "total_elapsed_ms": self.total_elapsed_ms,
            "final_action": self.final_action.value if self.final_action else None,
            "recovered": self.recovered,
        }

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    def summary(self) -> str:
        """One-line summary for appending to tool result."""
        if self.recovered:
            return (
                f"[SELF-HEALED] Recovered after {self.attempt_count} attempt(s) "
                f"via {self.final_action.value if self.final_action else 'unknown'} "
                f"({self.total_elapsed_ms:.0f}ms)"
            )
        return (
            f"[ESCALATED] Could not recover after {self.attempt_count} attempt(s) "
            f"({self.total_elapsed_ms:.0f}ms)"
        )


# ── Engine ──────────────────────────────────────────────────────────

class SelfHealingEngine:
    """
    Wraps tool execution with automatic diagnosis and recovery.

    Usage::

        engine = SelfHealingEngine(config, diagnostic, strategy_engine)
        result, history = await engine.execute_with_recovery(
            call, executor_fn, context
        )
    """

    def __init__(
        self,
        config: SelfHealingConfig,
        diagnostic: Any,           # FailureDiagnostic
        strategy_engine: Any,      # RecoveryStrategyEngine
        state_manager: Any = None,       # StateSnapshotManager (optional)
        rollback_journal: Any = None,    # RollbackJournal (optional)
    ):
        self.config = config
        self.diagnostic = diagnostic
        self.strategy_engine = strategy_engine
        self.state_manager = state_manager
        self.rollback_journal = rollback_journal
        # Metrics
        self._total_recoveries = 0
        self._total_escalations = 0

    # ── public API ──────────────────────────────────────────────

    async def execute_with_recovery(
        self,
        tool_call: ToolCall,
        executor_fn: Callable[[ToolCall], Awaitable[ToolResult]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ToolResult, Optional[RecoveryHistory]]:
        """
        Execute *tool_call* via *executor_fn*.  On failure, diagnose and
        attempt recovery up to ``max_recovery_attempts`` times.

        Returns ``(result, history)`` — history is ``None`` when the tool
        succeeds on the first try.
        """
        if not self.config.enabled:
            result = await executor_fn(tool_call)
            return (result, None)

        ctx = context or {}
        t0 = time.monotonic()

        # ── first attempt ───────────────────────────────────────
        result = await executor_fn(tool_call)
        if result.success:
            return (result, None)

        # ── failure → enter recovery loop ───────────────────────
        history = RecoveryHistory(
            original_tool_call=tool_call,
            original_error=result.error or "unknown error",
        )

        last_error = result.error or "unknown error"

        for _attempt_idx in range(self.config.max_recovery_attempts):
            attempt_t0 = time.monotonic()

            # 1. diagnose
            diagnosis = self.diagnostic.diagnose(
                tool_name=tool_call.name,
                error_msg=last_error,
                tool_input=tool_call.input,
                context=ctx,
            )

            # 2. pick recovery action
            action = self.strategy_engine.suggest_recovery(diagnosis)

            # 3. execute recovery
            rec_result = await self._execute_recovery(
                action, tool_call, diagnosis, executor_fn, ctx,
            )

            elapsed = (time.monotonic() - attempt_t0) * 1000
            attempt_rec = RecoveryAttempt(
                action=action,
                success=rec_result.success if rec_result else False,
                detail={"diagnosis_severity": diagnosis.severity},
                elapsed_ms=elapsed,
            )
            history.attempts.append(attempt_rec)

            if rec_result and rec_result.success:
                history.recovered = True
                history.final_action = action
                history.total_elapsed_ms = (time.monotonic() - t0) * 1000
                self._total_recoveries += 1
                # Annotate the result with recovery summary
                rec_result = ToolResult(
                    tool_id=rec_result.tool_id,
                    success=True,
                    output=rec_result.output + f"\n{history.summary()}",
                    metadata={**rec_result.metadata, "recovery_history": history.to_dict()},
                )
                return (rec_result, history)

            # update last_error for next diagnosis round
            if rec_result:
                last_error = rec_result.error or last_error

        # ── exhausted retries → escalate ────────────────────────
        history.final_action = HealingAction.ESCALATE
        history.total_elapsed_ms = (time.monotonic() - t0) * 1000
        self._total_escalations += 1

        escalated_result = ToolResult(
            tool_id=result.tool_id,
            success=False,
            output="",
            error=f"{result.error}\n{history.summary()}",
            metadata={**result.metadata, "recovery_history": history.to_dict()},
        )
        return (escalated_result, history)

    # ── internal helpers ────────────────────────────────────────

    async def _execute_recovery(
        self,
        action: HealingAction,
        original_call: ToolCall,
        diagnosis: Any,  # FailureDiagnosis
        executor_fn: Callable[[ToolCall], Awaitable[ToolResult]],
        context: Dict[str, Any],
    ) -> Optional[ToolResult]:
        """Dispatch to the correct recovery handler."""

        if action == HealingAction.RETRY_MODIFIED_PARAMS:
            modified_params = self.strategy_engine.get_modified_params(
                original_call, diagnosis,
            )
            modified_call = ToolCall(
                name=original_call.name,
                tool_id=ToolCall.generate_id(),
                input={**original_call.input, **modified_params},
            )
            return await executor_fn(modified_call)

        if action == HealingAction.USE_ALTERNATIVE_TOOL:
            alt_tool = self.strategy_engine.get_alternative_tool(
                original_call.name, diagnosis,
            )
            if alt_tool:
                alt_call = ToolCall(
                    name=alt_tool,
                    tool_id=ToolCall.generate_id(),
                    input=original_call.input,
                )
                return await executor_fn(alt_call)
            return None  # no alternative available

        if action == HealingAction.ROLLBACK_AND_RETRY:
            if self.rollback_journal and self.config.auto_rollback_on_failure:
                # rollback to last checkpoint, then retry original
                try:
                    checkpoints = self.rollback_journal.list_checkpoints()
                    if checkpoints:
                        self.rollback_journal.rollback(checkpoints[-1].checkpoint_id)
                except Exception:
                    pass  # rollback best-effort
            return await executor_fn(original_call)

        if action == HealingAction.DEGRADE:
            degraded_params = self.strategy_engine.get_degraded_params(
                original_call, diagnosis,
            )
            degraded_call = ToolCall(
                name=original_call.name,
                tool_id=ToolCall.generate_id(),
                input={**original_call.input, **degraded_params},
            )
            return await executor_fn(degraded_call)

        # ESCALATE or unknown → do nothing
        return None

    # ── metrics ─────────────────────────────────────────────────

    @property
    def total_recoveries(self) -> int:
        return self._total_recoveries

    @property
    def total_escalations(self) -> int:
        return self._total_escalations

    def stats(self) -> Dict[str, Any]:
        return {
            "total_recoveries": self._total_recoveries,
            "total_escalations": self._total_escalations,
            "config": self.config.to_dict(),
        }
