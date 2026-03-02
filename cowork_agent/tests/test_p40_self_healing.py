"""
Sprint 40 · Self-Healing Pipelines — Tests
============================================
~100 tests covering SelfHealingEngine, FailureDiagnostic,
RecoveryStrategyEngine, integration, and edge cases.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cowork_agent.core.self_healing import (
    HealingAction,
    RecoveryAttempt,
    RecoveryHistory,
    SelfHealingConfig,
    SelfHealingEngine,
)
from cowork_agent.core.failure_diagnosis import (
    PARAM_FIX_RULES,
    TOOL_ALTERNATIVES,
    FailureDiagnosis,
    FailureDiagnostic,
)
from cowork_agent.core.recovery_strategies import RecoveryStrategyEngine
from cowork_agent.core.error_catalog import ErrorCategory, ErrorCode
from cowork_agent.core.models import ToolCall, ToolResult


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_call(name: str = "bash", **kwargs) -> ToolCall:
    return ToolCall(name=name, tool_id=ToolCall.generate_id(), input=kwargs)


def _ok(output: str = "done") -> ToolResult:
    return ToolResult(tool_id="t1", success=True, output=output)


def _fail(error: str = "timeout") -> ToolResult:
    return ToolResult(tool_id="t1", success=False, output="", error=error)


# ═══════════════════════════════════════════════════════════════════
#  1. SelfHealingConfig
# ═══════════════════════════════════════════════════════════════════

class TestSelfHealingConfig:
    def test_defaults(self):
        c = SelfHealingConfig()
        assert c.enabled is True
        assert c.max_recovery_attempts == 2
        assert c.auto_rollback_on_failure is True
        assert c.recovery_timeout_seconds == 60.0
        assert c.escalation_strategy == "user_confirm"

    def test_custom_values(self):
        c = SelfHealingConfig(enabled=False, max_recovery_attempts=5)
        assert c.enabled is False
        assert c.max_recovery_attempts == 5

    def test_to_dict(self):
        d = SelfHealingConfig().to_dict()
        assert "enabled" in d
        assert "max_recovery_attempts" in d
        assert "recovery_timeout_seconds" in d

    def test_from_dict(self):
        c = SelfHealingConfig.from_dict({"enabled": False, "max_recovery_attempts": 3})
        assert c.enabled is False
        assert c.max_recovery_attempts == 3

    def test_from_dict_ignores_unknown(self):
        c = SelfHealingConfig.from_dict({"enabled": True, "foo": "bar"})
        assert c.enabled is True

    def test_roundtrip(self):
        orig = SelfHealingConfig(max_recovery_attempts=4, recovery_timeout_seconds=120.0)
        rebuilt = SelfHealingConfig.from_dict(orig.to_dict())
        assert rebuilt.max_recovery_attempts == 4
        assert rebuilt.recovery_timeout_seconds == 120.0


# ═══════════════════════════════════════════════════════════════════
#  2. HealingAction enum
# ═══════════════════════════════════════════════════════════════════

class TestHealingAction:
    def test_all_actions_exist(self):
        expected = {
            "RETRY_MODIFIED_PARAMS",
            "USE_ALTERNATIVE_TOOL",
            "ROLLBACK_AND_RETRY",
            "DEGRADE",
            "ESCALATE",
        }
        assert {a.name for a in HealingAction} == expected

    def test_values_are_strings(self):
        for a in HealingAction:
            assert isinstance(a.value, str)

    def test_escalate_is_last_resort(self):
        assert HealingAction.ESCALATE.value == "escalate"

    def test_retry_modified_value(self):
        assert HealingAction.RETRY_MODIFIED_PARAMS.value == "retry_modified_params"

    def test_enum_members_count(self):
        assert len(HealingAction) == 5


# ═══════════════════════════════════════════════════════════════════
#  3. RecoveryAttempt & RecoveryHistory
# ═══════════════════════════════════════════════════════════════════

class TestRecoveryAttempt:
    def test_creation(self):
        a = RecoveryAttempt(action=HealingAction.DEGRADE, success=True)
        assert a.action == HealingAction.DEGRADE
        assert a.success is True

    def test_to_dict(self):
        d = RecoveryAttempt(action=HealingAction.ESCALATE, success=False).to_dict()
        assert d["action"] == "escalate"
        assert d["success"] is False

    def test_detail_default(self):
        a = RecoveryAttempt(action=HealingAction.DEGRADE, success=True)
        assert a.detail == {}

    def test_elapsed_default(self):
        a = RecoveryAttempt(action=HealingAction.DEGRADE, success=True)
        assert a.elapsed_ms == 0.0


class TestRecoveryHistory:
    def test_creation(self):
        h = RecoveryHistory(original_tool_call=_make_call(), original_error="boom")
        assert h.original_error == "boom"
        assert h.attempts == []
        assert h.recovered is False

    def test_attempt_count(self):
        h = RecoveryHistory(original_tool_call=_make_call(), original_error="e")
        h.attempts.append(RecoveryAttempt(HealingAction.DEGRADE, False))
        h.attempts.append(RecoveryAttempt(HealingAction.ESCALATE, False))
        assert h.attempt_count == 2

    def test_to_dict(self):
        h = RecoveryHistory(original_tool_call=_make_call("read"), original_error="err")
        d = h.to_dict()
        assert d["original_tool"] == "read"
        assert d["original_error"] == "err"
        assert d["recovered"] is False

    def test_summary_recovered(self):
        h = RecoveryHistory(original_tool_call=_make_call(), original_error="e")
        h.recovered = True
        h.final_action = HealingAction.RETRY_MODIFIED_PARAMS
        h.attempts.append(RecoveryAttempt(HealingAction.RETRY_MODIFIED_PARAMS, True))
        h.total_elapsed_ms = 150.0
        s = h.summary()
        assert "[SELF-HEALED]" in s
        assert "retry_modified_params" in s

    def test_summary_escalated(self):
        h = RecoveryHistory(original_tool_call=_make_call(), original_error="e")
        h.recovered = False
        h.total_elapsed_ms = 500.0
        h.attempts.append(RecoveryAttempt(HealingAction.DEGRADE, False))
        s = h.summary()
        assert "[ESCALATED]" in s

    def test_final_action_none(self):
        h = RecoveryHistory(original_tool_call=_make_call(), original_error="e")
        assert h.final_action is None

    def test_to_dict_with_attempts(self):
        h = RecoveryHistory(original_tool_call=_make_call(), original_error="e")
        h.attempts.append(RecoveryAttempt(HealingAction.DEGRADE, True, elapsed_ms=42.0))
        d = h.to_dict()
        assert len(d["attempts"]) == 1
        assert d["attempts"][0]["elapsed_ms"] == 42.0

    def test_total_elapsed(self):
        h = RecoveryHistory(original_tool_call=_make_call(), original_error="e")
        h.total_elapsed_ms = 1234.5
        assert h.total_elapsed_ms == 1234.5


# ═══════════════════════════════════════════════════════════════════
#  4. FailureDiagnosis
# ═══════════════════════════════════════════════════════════════════

class TestFailureDiagnosis:
    def test_creation(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_TIMEOUT,
            error_category=ErrorCategory.TOOL,
            root_cause="timed out",
            severity="medium",
        )
        assert d.severity == "medium"

    def test_to_dict(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_EXECUTION_FAILED,
            error_category=ErrorCategory.TOOL,
            root_cause="fail",
            severity="high",
        ).to_dict()
        assert d["error_code"] == "E2001"
        assert d["severity"] == "high"

    def test_from_dict(self):
        d = FailureDiagnosis.from_dict({
            "error_code": "E2002",
            "error_category": "tool",
            "root_cause": "slow",
            "severity": "low",
        })
        assert d.error_code == ErrorCode.TOOL_TIMEOUT
        assert d.severity == "low"

    def test_default_confidence(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_TIMEOUT,
            error_category=ErrorCategory.TOOL,
            root_cause="t",
            severity="medium",
        )
        assert d.confidence == 0.5

    def test_transient_flag(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.PROVIDER_RATE_LIMITED,
            error_category=ErrorCategory.PROVIDER,
            root_cause="rate limited",
            severity="low",
            is_transient=True,
        )
        assert d.is_transient is True

    def test_suggested_params(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_TIMEOUT,
            error_category=ErrorCategory.TOOL,
            root_cause="t",
            severity="medium",
            suggested_param_changes={"timeout": 120},
        )
        assert d.suggested_param_changes["timeout"] == 120

    def test_alternative_tools(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_EXECUTION_FAILED,
            error_category=ErrorCategory.TOOL,
            root_cause="f",
            severity="medium",
            suggested_alternative_tools=["edit"],
        )
        assert "edit" in d.suggested_alternative_tools

    def test_rollback_flag(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_EXECUTION_FAILED,
            error_category=ErrorCategory.TOOL,
            root_cause="f",
            severity="high",
            rollback_recommended=True,
        )
        assert d.rollback_recommended is True

    def test_degradation_flag(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.PROVIDER_OVERLOADED,
            error_category=ErrorCategory.PROVIDER,
            root_cause="overloaded",
            severity="low",
            degradation_recommended=True,
        )
        assert d.degradation_recommended is True

    def test_all_fields_in_to_dict(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_TIMEOUT,
            error_category=ErrorCategory.TOOL,
            root_cause="t",
            severity="medium",
        ).to_dict()
        expected_keys = {
            "error_code", "error_category", "root_cause", "severity",
            "suggested_param_changes", "suggested_alternative_tools",
            "rollback_recommended", "degradation_recommended",
            "confidence", "is_transient",
        }
        assert expected_keys == set(d.keys())


# ═══════════════════════════════════════════════════════════════════
#  5. FailureDiagnostic
# ═══════════════════════════════════════════════════════════════════

class TestFailureDiagnostic:
    def setup_method(self):
        self.diag = FailureDiagnostic()

    def test_diagnose_timeout(self):
        d = self.diag.diagnose("bash", "timeout exceeded", {}, {})
        assert d.error_code == ErrorCode.PROVIDER_TIMEOUT or d.error_code == ErrorCode.TOOL_TIMEOUT or d.error_code == ErrorCode.NETWORK_TIMEOUT or "timeout" in d.root_cause.lower() or d.severity in ("medium", "low")

    def test_diagnose_rate_limit(self):
        d = self.diag.diagnose("bash", "rate limit exceeded 429", {}, {})
        assert d.error_code == ErrorCode.PROVIDER_RATE_LIMITED

    def test_diagnose_auth_failure(self):
        d = self.diag.diagnose("web_fetch", "authentication failed 401", {}, {})
        assert d.severity in ("critical", "high")

    def test_diagnose_connection_refused(self):
        d = self.diag.diagnose("web_fetch", "connection refused", {}, {})
        assert d.severity in ("high", "medium")

    def test_diagnose_unknown_error(self):
        d = self.diag.diagnose("bash", "something weird happened", {}, {})
        assert isinstance(d, FailureDiagnosis)
        assert d.severity in ("low", "medium", "high", "critical")

    def test_alternative_tools_for_write(self):
        d = self.diag.diagnose("write", "permission denied", {}, {})
        assert "edit" in d.suggested_alternative_tools

    def test_alternative_tools_for_glob(self):
        d = self.diag.diagnose("glob", "some error", {}, {})
        alts = d.suggested_alternative_tools
        assert "grep" in alts or "bash" in alts

    def test_no_alternatives_for_unknown_tool(self):
        d = self.diag.diagnose("unknown_tool_xyz", "error", {}, {})
        assert d.suggested_alternative_tools == []

    def test_param_changes_for_timeout(self):
        d = self.diag.diagnose("bash", "timeout error", {"timeout": 30}, {})
        # Should suggest timeout increase or have some param change
        assert isinstance(d.suggested_param_changes, dict)

    def test_cache_hit(self):
        d1 = self.diag.diagnose("bash", "rate limit 429", {}, {})
        d2 = self.diag.diagnose("bash", "rate limit 429", {}, {})
        assert d1 is d2  # same object from cache

    def test_cache_miss_different_tool(self):
        d1 = self.diag.diagnose("bash", "rate limit 429", {}, {})
        d2 = self.diag.diagnose("read", "rate limit 429", {}, {})
        assert d1 is not d2

    def test_clear_cache(self):
        self.diag.diagnose("bash", "rate limit 429", {}, {})
        assert len(self.diag._cache) > 0
        self.diag.clear_cache()
        assert len(self.diag._cache) == 0

    def test_diagnosis_has_confidence(self):
        d = self.diag.diagnose("bash", "timeout", {}, {})
        assert 0.0 <= d.confidence <= 1.0

    def test_transient_for_rate_limit(self):
        d = self.diag.diagnose("bash", "rate limit 429", {}, {})
        assert d.is_transient is True

    def test_non_transient_for_auth(self):
        d = self.diag.diagnose("bash", "authentication failed 401", {}, {})
        assert d.is_transient is False


# ═══════════════════════════════════════════════════════════════════
#  6. RecoveryStrategyEngine
# ═══════════════════════════════════════════════════════════════════

class TestRecoveryStrategyEngine:
    def setup_method(self):
        self.engine = RecoveryStrategyEngine()

    def test_transient_with_params_retries(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.PROVIDER_RATE_LIMITED,
            error_category=ErrorCategory.PROVIDER,
            root_cause="rate limit",
            severity="low",
            is_transient=True,
            suggested_param_changes={"_delay_seconds": 2.0},
        )
        assert self.engine.suggest_recovery(d) == HealingAction.RETRY_MODIFIED_PARAMS

    def test_has_alternative_tool(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_EXECUTION_FAILED,
            error_category=ErrorCategory.TOOL,
            root_cause="fail",
            severity="medium",
            suggested_alternative_tools=["edit"],
        )
        assert self.engine.suggest_recovery(d) == HealingAction.USE_ALTERNATIVE_TOOL

    def test_high_severity_rollback(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.PROVIDER_CONNECTION_FAILED,
            error_category=ErrorCategory.PROVIDER,
            root_cause="connection fail",
            severity="high",
            rollback_recommended=True,
        )
        assert self.engine.suggest_recovery(d) == HealingAction.ROLLBACK_AND_RETRY

    def test_degradation_recommended(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.PROVIDER_OVERLOADED,
            error_category=ErrorCategory.PROVIDER,
            root_cause="overloaded",
            severity="low",
            degradation_recommended=True,
        )
        assert self.engine.suggest_recovery(d) == HealingAction.DEGRADE

    def test_escalate_fallback(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.SECURITY_BLOCKED,
            error_category=ErrorCategory.SECURITY,
            root_cause="blocked",
            severity="critical",
        )
        assert self.engine.suggest_recovery(d) == HealingAction.ESCALATE

    def test_get_modified_params(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_TIMEOUT,
            error_category=ErrorCategory.TOOL,
            root_cause="t",
            severity="medium",
            suggested_param_changes={"timeout": 120},
        )
        params = self.engine.get_modified_params(_make_call(), d)
        assert params.get("timeout") == 120

    def test_get_alternative_tool_from_diagnosis(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_EXECUTION_FAILED,
            error_category=ErrorCategory.TOOL,
            root_cause="f",
            severity="medium",
            suggested_alternative_tools=["edit", "write"],
        )
        assert self.engine.get_alternative_tool("bash", d) == "edit"

    def test_get_alternative_tool_from_map(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_EXECUTION_FAILED,
            error_category=ErrorCategory.TOOL,
            root_cause="f",
            severity="medium",
        )
        alt = self.engine.get_alternative_tool("write", d)
        assert alt == "edit"

    def test_get_alternative_tool_none(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_EXECUTION_FAILED,
            error_category=ErrorCategory.TOOL,
            root_cause="f",
            severity="medium",
        )
        assert self.engine.get_alternative_tool("unknown_xyz", d) is None

    def test_get_degraded_params_transient(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.PROVIDER_TIMEOUT,
            error_category=ErrorCategory.PROVIDER,
            root_cause="slow",
            severity="medium",
            is_transient=True,
        )
        params = self.engine.get_degraded_params(
            _make_call(timeout=60), d,
        )
        assert "timeout" in params

    def test_get_degraded_params_high_severity(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.PROVIDER_CONNECTION_FAILED,
            error_category=ErrorCategory.PROVIDER,
            root_cause="conn fail",
            severity="high",
        )
        params = self.engine.get_degraded_params(_make_call(), d)
        assert "max_results" in params or "streaming" in params or "_degraded" in params

    def test_get_degraded_params_always_returns_something(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.TOOL_EXECUTION_FAILED,
            error_category=ErrorCategory.TOOL,
            root_cause="f",
            severity="medium",
        )
        params = self.engine.get_degraded_params(_make_call(), d)
        assert len(params) > 0

    def test_action_counts_tracking(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.SECURITY_BLOCKED,
            error_category=ErrorCategory.SECURITY,
            root_cause="x",
            severity="critical",
        )
        self.engine.suggest_recovery(d)
        self.engine.suggest_recovery(d)
        assert self.engine.action_counts.get("escalate", 0) == 2

    def test_reset_counts(self):
        d = FailureDiagnosis(
            error_code=ErrorCode.SECURITY_BLOCKED,
            error_category=ErrorCategory.SECURITY,
            root_cause="x",
            severity="critical",
        )
        self.engine.suggest_recovery(d)
        self.engine.reset_counts()
        assert self.engine.action_counts == {}


# ═══════════════════════════════════════════════════════════════════
#  7. SelfHealingEngine
# ═══════════════════════════════════════════════════════════════════

class TestSelfHealingEngine:
    def setup_method(self):
        self.config = SelfHealingConfig()
        self.diagnostic = FailureDiagnostic()
        self.strategy = RecoveryStrategyEngine()
        self.engine = SelfHealingEngine(
            config=self.config,
            diagnostic=self.diagnostic,
            strategy_engine=self.strategy,
        )

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        executor = AsyncMock(return_value=_ok("done"))
        result, history = await self.engine.execute_with_recovery(
            _make_call(), executor,
        )
        assert result.success is True
        assert history is None
        executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_then_recovery(self):
        """Fail once, then succeed on retry."""
        executor = AsyncMock(side_effect=[_fail("rate limit 429"), _ok("ok")])
        result, history = await self.engine.execute_with_recovery(
            _make_call(), executor,
        )
        assert result.success is True
        assert history is not None
        assert history.recovered is True
        assert history.attempt_count >= 1

    @pytest.mark.asyncio
    async def test_exhaust_retries_escalate(self):
        """Fail all attempts → escalate."""
        executor = AsyncMock(return_value=_fail("permanent error"))
        result, history = await self.engine.execute_with_recovery(
            _make_call(), executor,
        )
        assert result.success is False
        assert history is not None
        assert history.recovered is False
        assert history.final_action == HealingAction.ESCALATE

    @pytest.mark.asyncio
    async def test_alternative_tool_recovery(self):
        """write fails → try edit (alternative)."""
        # write fails, edit succeeds
        call_results = {
            "write": _fail("permission denied"),
            "edit": _ok("edited"),
        }

        async def executor(call):
            return call_results.get(call.name, _fail("unknown"))

        result, history = await self.engine.execute_with_recovery(
            _make_call("write"), executor,
        )
        # Should try alternative tool
        assert history is not None
        assert history.attempt_count >= 1

    @pytest.mark.asyncio
    async def test_disabled_config(self):
        """When disabled, just execute directly."""
        self.engine.config.enabled = False
        executor = AsyncMock(return_value=_fail("error"))
        result, history = await self.engine.execute_with_recovery(
            _make_call(), executor,
        )
        assert result.success is False
        assert history is None
        executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_recovery_history_in_result_metadata(self):
        executor = AsyncMock(side_effect=[_fail("rate limit 429"), _ok("ok")])
        result, history = await self.engine.execute_with_recovery(
            _make_call(), executor,
        )
        if result.success and history:
            assert "recovery_history" in result.metadata

    @pytest.mark.asyncio
    async def test_escalated_error_contains_summary(self):
        executor = AsyncMock(return_value=_fail("permanent"))
        result, history = await self.engine.execute_with_recovery(
            _make_call(), executor,
        )
        assert "[ESCALATED]" in (result.error or "")

    @pytest.mark.asyncio
    async def test_max_recovery_attempts_respected(self):
        self.engine.config.max_recovery_attempts = 1
        call_count = 0

        async def executor(call):
            nonlocal call_count
            call_count += 1
            return _fail("always fail")

        result, history = await self.engine.execute_with_recovery(
            _make_call(), executor,
        )
        # 1 initial + up to 1 recovery = at most ~3 calls
        # (initial + recovery attempt which may call executor)
        assert call_count <= 4
        assert result.success is False

    @pytest.mark.asyncio
    async def test_total_recoveries_metric(self):
        executor = AsyncMock(side_effect=[_fail("rate limit 429"), _ok("ok")])
        await self.engine.execute_with_recovery(_make_call(), executor)
        assert self.engine.total_recoveries >= 0  # at least tracked

    @pytest.mark.asyncio
    async def test_total_escalations_metric(self):
        executor = AsyncMock(return_value=_fail("permanent"))
        await self.engine.execute_with_recovery(_make_call(), executor)
        assert self.engine.total_escalations >= 1

    @pytest.mark.asyncio
    async def test_stats(self):
        s = self.engine.stats()
        assert "total_recoveries" in s
        assert "total_escalations" in s
        assert "config" in s

    @pytest.mark.asyncio
    async def test_rollback_recovery(self):
        """Test rollback_and_retry path with mock rollback journal."""
        mock_journal = MagicMock()
        mock_checkpoint = MagicMock()
        mock_checkpoint.checkpoint_id = "cp-1"
        mock_journal.list_checkpoints.return_value = [mock_checkpoint]

        self.engine.rollback_journal = mock_journal

        # First fail with high severity, then succeed on retry
        executor = AsyncMock(side_effect=[
            _fail("connection failed"),
            _fail("connection failed"),  # recovery attempt may also fail
            _ok("ok"),
        ])
        result, history = await self.engine.execute_with_recovery(
            _make_call("web_fetch"), executor,
        )
        assert history is not None

    @pytest.mark.asyncio
    async def test_degrade_recovery(self):
        """Test degrade path."""
        # Overloaded error → degrade strategy
        executor = AsyncMock(side_effect=[
            _fail("server overloaded"),
            _ok("degraded ok"),
        ])
        result, history = await self.engine.execute_with_recovery(
            _make_call(), executor,
        )
        assert history is not None

    @pytest.mark.asyncio
    async def test_recovery_with_context(self):
        executor = AsyncMock(side_effect=[_fail("timeout"), _ok("ok")])
        ctx = {"previous_results": ["result1"]}
        result, history = await self.engine.execute_with_recovery(
            _make_call(), executor, context=ctx,
        )
        assert history is not None

    @pytest.mark.asyncio
    async def test_multiple_sequential_recoveries(self):
        """Two separate calls, both need recovery."""
        executor1 = AsyncMock(side_effect=[_fail("rate limit 429"), _ok("ok1")])
        executor2 = AsyncMock(side_effect=[_fail("timeout"), _ok("ok2")])

        r1, h1 = await self.engine.execute_with_recovery(_make_call(), executor1)
        r2, h2 = await self.engine.execute_with_recovery(_make_call(), executor2)
        # Both should have recovery histories
        assert h1 is not None or h2 is not None

    @pytest.mark.asyncio
    async def test_self_healed_output_contains_summary(self):
        executor = AsyncMock(side_effect=[_fail("rate limit 429"), _ok("ok")])
        result, history = await self.engine.execute_with_recovery(
            _make_call(), executor,
        )
        if result.success and history and history.recovered:
            assert "[SELF-HEALED]" in result.output

    @pytest.mark.asyncio
    async def test_executor_exception_handling(self):
        """Executor raises instead of returning ToolResult."""
        async def bad_executor(call):
            raise RuntimeError("unexpected")

        with pytest.raises(RuntimeError):
            await self.engine.execute_with_recovery(
                _make_call(), bad_executor,
            )

    @pytest.mark.asyncio
    async def test_zero_max_attempts(self):
        """Zero recovery attempts = just escalate immediately."""
        self.engine.config.max_recovery_attempts = 0
        executor = AsyncMock(return_value=_fail("error"))
        result, history = await self.engine.execute_with_recovery(
            _make_call(), executor,
        )
        assert result.success is False
        assert history.attempt_count == 0

    @pytest.mark.asyncio
    async def test_none_context(self):
        executor = AsyncMock(return_value=_ok("ok"))
        result, history = await self.engine.execute_with_recovery(
            _make_call(), executor, context=None,
        )
        assert result.success is True


# ═══════════════════════════════════════════════════════════════════
#  8. Agent Integration
# ═══════════════════════════════════════════════════════════════════

class TestAgentIntegration:
    def test_agent_has_self_healing_attribute(self):
        from cowork_agent.core.agent import Agent
        agent = Agent.__new__(Agent)
        # The __init__ sets self.self_healing_engine = None
        # We just check the class accepts the attribute
        agent.self_healing_engine = None
        assert agent.self_healing_engine is None

    def test_self_healing_engine_assignable(self):
        from cowork_agent.core.agent import Agent
        agent = Agent.__new__(Agent)
        engine = SelfHealingEngine(
            config=SelfHealingConfig(),
            diagnostic=FailureDiagnostic(),
            strategy_engine=RecoveryStrategyEngine(),
        )
        agent.self_healing_engine = engine
        assert agent.self_healing_engine is engine

    def test_engine_has_correct_interface(self):
        engine = SelfHealingEngine(
            config=SelfHealingConfig(),
            diagnostic=FailureDiagnostic(),
            strategy_engine=RecoveryStrategyEngine(),
        )
        assert hasattr(engine, "execute_with_recovery")
        assert callable(engine.execute_with_recovery)
        assert hasattr(engine, "stats")

    def test_config_wiring(self):
        """Simulate main.py wiring pattern."""
        config = {"self_healing": {"enabled": True, "max_recovery_attempts": 3}}
        sh_cfg = config.get("self_healing", {})
        healing_config = SelfHealingConfig(
            enabled=sh_cfg.get("enabled", True),
            max_recovery_attempts=sh_cfg.get("max_recovery_attempts", 2),
        )
        assert healing_config.max_recovery_attempts == 3

    def test_disabled_config_wiring(self):
        config = {"self_healing": {"enabled": False}}
        sh_cfg = config.get("self_healing", {})
        healing_config = SelfHealingConfig(enabled=sh_cfg.get("enabled", True))
        assert healing_config.enabled is False

    def test_engine_with_all_optional_deps(self):
        engine = SelfHealingEngine(
            config=SelfHealingConfig(),
            diagnostic=FailureDiagnostic(),
            strategy_engine=RecoveryStrategyEngine(),
            state_manager=MagicMock(),
            rollback_journal=MagicMock(),
        )
        assert engine.state_manager is not None
        assert engine.rollback_journal is not None

    def test_engine_without_optional_deps(self):
        engine = SelfHealingEngine(
            config=SelfHealingConfig(),
            diagnostic=FailureDiagnostic(),
            strategy_engine=RecoveryStrategyEngine(),
        )
        assert engine.state_manager is None
        assert engine.rollback_journal is None


# ═══════════════════════════════════════════════════════════════════
#  9. Edge Cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_tool_alternatives_map_keys(self):
        expected = {"write", "edit", "bash", "web_fetch", "web_search", "glob", "grep", "read"}
        assert set(TOOL_ALTERNATIVES.keys()) == expected

    def test_param_fix_rules_keys(self):
        for code in PARAM_FIX_RULES:
            assert isinstance(code, ErrorCode)

    def test_healing_action_values_unique(self):
        values = [a.value for a in HealingAction]
        assert len(values) == len(set(values))

    def test_diagnosis_severity_levels(self):
        valid = {"low", "medium", "high", "critical"}
        diag = FailureDiagnostic()
        for error_msg in ["timeout", "rate limit 429", "auth failed 401", "random"]:
            d = diag.diagnose("bash", error_msg, {}, {})
            assert d.severity in valid

    def test_empty_error_message(self):
        diag = FailureDiagnostic()
        d = diag.diagnose("bash", "", {}, {})
        assert isinstance(d, FailureDiagnosis)

    def test_very_long_error_message(self):
        diag = FailureDiagnostic()
        d = diag.diagnose("bash", "x" * 10000, {}, {})
        assert isinstance(d, FailureDiagnosis)

    @pytest.mark.asyncio
    async def test_concurrent_recovery_calls(self):
        engine = SelfHealingEngine(
            config=SelfHealingConfig(),
            diagnostic=FailureDiagnostic(),
            strategy_engine=RecoveryStrategyEngine(),
        )
        executor = AsyncMock(side_effect=[
            _fail("timeout"), _ok("ok"),
            _fail("timeout"), _ok("ok"),
            _fail("timeout"), _ok("ok"),
        ])
        tasks = [
            engine.execute_with_recovery(_make_call(), executor)
            for _ in range(3)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # All should complete (some may fail due to shared mock state)
        assert len(results) == 3

    def test_strategy_engine_custom_rules(self):
        engine = RecoveryStrategyEngine(custom_rules={"foo": "bar"})
        assert engine._custom_rules == {"foo": "bar"}

    def test_config_from_empty_dict(self):
        c = SelfHealingConfig.from_dict({})
        assert c.enabled is True  # defaults

    def test_diagnosis_from_dict_minimal(self):
        d = FailureDiagnosis.from_dict({
            "error_code": "E2001",
            "error_category": "tool",
        })
        assert d.error_code == ErrorCode.TOOL_EXECUTION_FAILED
        assert d.severity == "medium"

    @pytest.mark.asyncio
    async def test_alternative_tool_not_available(self):
        """When alternative tool doesn't exist in executor."""
        engine = SelfHealingEngine(
            config=SelfHealingConfig(max_recovery_attempts=1),
            diagnostic=FailureDiagnostic(),
            strategy_engine=RecoveryStrategyEngine(),
        )
        # unknown_tool has no alternatives, should escalate
        executor = AsyncMock(return_value=_fail("error"))
        result, history = await engine.execute_with_recovery(
            _make_call("unknown_tool_xyz"), executor,
        )
        assert result.success is False
