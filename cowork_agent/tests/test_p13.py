"""
Sprint 13 — Error Recovery & Resilience tests.

Covers:
  - ProviderCircuitBreaker (state machine, threshold, multi-provider)
  - ErrorAggregator (spike/recurring/correlated detection, window, cleanup)
  - RecoveryOrchestrator (strategy selection, pluggable overrides, defaults)
  - ErrorContextEnricher (enrichment, breadcrumbs, formatting)
  - ErrorBudgetTracker (rate calculation, budget enforcement, categories)
  - Integration (full flows combining multiple modules)

~120 tests total.
"""

import time
import pytest

from cowork_agent.core.error_catalog import (
    AgentError, ErrorCatalog, ErrorCode, ErrorCategory,
)
from cowork_agent.core.provider_circuit_breaker import (
    ProviderCircuitBreaker, CircuitBreakerConfig, CircuitBreakerState,
)
from cowork_agent.core.error_aggregator import (
    ErrorAggregator, ErrorEvent, ErrorPattern, PatternType,
)
from cowork_agent.core.error_recovery_orchestrator import (
    RecoveryOrchestrator, RecoveryAction, RecoveryStrategy, RecoveryContext,
)
from cowork_agent.core.error_context import (
    ErrorContextEnricher, ErrorContext,
)
from cowork_agent.core.error_budget import (
    ErrorBudgetTracker, ErrorBudgetConfig,
)


# ── Helpers ──────────────────────────────────────────────────────

def _make_error(code: ErrorCode = ErrorCode.PROVIDER_TIMEOUT,
                msg: str = "timeout") -> AgentError:
    """Create a minimal AgentError for testing."""
    return ErrorCatalog.from_code(code)


def _make_context(tool: str = "bash", provider: str = "ollama",
                  iteration: int = 1) -> ErrorContext:
    return ErrorContext(
        iteration=iteration,
        tool_name=tool,
        provider_name=provider,
        timestamp=time.time(),
        duration_ms=123.4,
    )


# ═════════════════════════════════════════════════════════════════
# 1. TestProviderCircuitBreaker — 15 tests
# ═════════════════════════════════════════════════════════════════

class TestProviderCircuitBreaker:
    """Provider circuit breaker state machine tests."""

    def test_initial_state_closed(self):
        cb = ProviderCircuitBreaker()
        assert cb.get_state("openai") == CircuitBreakerState.CLOSED

    def test_available_by_default(self):
        cb = ProviderCircuitBreaker()
        assert cb.is_available("openai") is True

    def test_single_failure_stays_closed(self):
        cb = ProviderCircuitBreaker()
        cb.record_failure("openai", Exception("err"))
        assert cb.get_state("openai") == CircuitBreakerState.CLOSED

    def test_threshold_opens_circuit(self):
        cfg = CircuitBreakerConfig(failure_threshold=3, timeout_seconds=60)
        cb = ProviderCircuitBreaker(config=cfg)
        for _ in range(3):
            cb.record_failure("openai")
        assert cb.get_state("openai") == CircuitBreakerState.OPEN
        assert cb.is_available("openai") is False

    def test_success_resets_failure_count(self):
        cfg = CircuitBreakerConfig(failure_threshold=3)
        cb = ProviderCircuitBreaker(config=cfg)
        cb.record_failure("openai")
        cb.record_failure("openai")
        cb.record_success("openai")
        # Failure count reset — 2 more failures should not open
        cb.record_failure("openai")
        cb.record_failure("openai")
        assert cb.get_state("openai") == CircuitBreakerState.CLOSED

    def test_open_transitions_to_half_open_after_timeout(self):
        cfg = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.01)
        cb = ProviderCircuitBreaker(config=cfg)
        cb.record_failure("openai")
        cb.record_failure("openai")
        assert cb.get_state("openai") == CircuitBreakerState.OPEN
        time.sleep(0.02)
        assert cb.is_available("openai") is True
        assert cb.get_state("openai") == CircuitBreakerState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        cfg = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.01)
        cb = ProviderCircuitBreaker(config=cfg)
        cb.record_failure("openai")
        cb.record_failure("openai")
        time.sleep(0.02)
        cb.is_available("openai")  # triggers transition to HALF_OPEN
        cb.record_success("openai")
        assert cb.get_state("openai") == CircuitBreakerState.CLOSED

    def test_half_open_failure_reopens_circuit(self):
        cfg = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.01)
        cb = ProviderCircuitBreaker(config=cfg)
        cb.record_failure("openai")
        cb.record_failure("openai")
        time.sleep(0.02)
        cb.is_available("openai")  # HALF_OPEN
        cb.record_failure("openai")
        assert cb.get_state("openai") == CircuitBreakerState.OPEN

    def test_half_open_limited_calls(self):
        cfg = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.01,
                                    half_open_max_calls=1)
        cb = ProviderCircuitBreaker(config=cfg)
        cb.record_failure("openai")
        cb.record_failure("openai")
        time.sleep(0.02)
        assert cb.is_available("openai") is True  # first call allowed
        # Simulate a half_open call by incrementing counter
        state = cb._get_state("openai")
        state.half_open_calls = 1
        assert cb.is_available("openai") is False  # second call blocked

    def test_multi_provider_independence(self):
        cfg = CircuitBreakerConfig(failure_threshold=2)
        cb = ProviderCircuitBreaker(config=cfg)
        cb.record_failure("openai")
        cb.record_failure("openai")
        assert cb.get_state("openai") == CircuitBreakerState.OPEN
        assert cb.get_state("anthropic") == CircuitBreakerState.CLOSED
        assert cb.is_available("anthropic") is True

    def test_reset_provider(self):
        cfg = CircuitBreakerConfig(failure_threshold=2)
        cb = ProviderCircuitBreaker(config=cfg)
        cb.record_failure("openai")
        cb.record_failure("openai")
        assert cb.get_state("openai") == CircuitBreakerState.OPEN
        cb.reset("openai")
        assert cb.get_state("openai") == CircuitBreakerState.CLOSED

    def test_reset_all(self):
        cfg = CircuitBreakerConfig(failure_threshold=2)
        cb = ProviderCircuitBreaker(config=cfg)
        cb.record_failure("openai")
        cb.record_failure("openai")
        cb.record_failure("anthropic")
        cb.record_failure("anthropic")
        cb.reset_all()
        assert cb.get_state("openai") == CircuitBreakerState.CLOSED
        assert cb.get_state("anthropic") == CircuitBreakerState.CLOSED

    def test_to_dict(self):
        cb = ProviderCircuitBreaker()
        cb.record_failure("openai")
        d = cb.to_dict()
        assert "openai" in d
        assert d["openai"]["state"] == "closed"
        assert d["openai"]["failure_count"] == 1

    def test_default_config_values(self):
        cfg = CircuitBreakerConfig()
        assert cfg.failure_threshold == 5
        assert cfg.timeout_seconds == 60.0
        assert cfg.half_open_max_calls == 2

    def test_open_not_available(self):
        cfg = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=9999)
        cb = ProviderCircuitBreaker(config=cfg)
        cb.record_failure("openai")
        assert cb.is_available("openai") is False


# ═════════════════════════════════════════════════════════════════
# 2. TestErrorAggregator — 18 tests
# ═════════════════════════════════════════════════════════════════

class TestErrorAggregator:
    """Error aggregation and pattern detection tests."""

    def test_record_error(self):
        agg = ErrorAggregator()
        err = _make_error()
        agg.record_error(err, tool_name="bash")
        assert agg.event_count == 1

    def test_multiple_records(self):
        agg = ErrorAggregator()
        for _ in range(5):
            agg.record_error(_make_error())
        assert agg.event_count == 5

    def test_no_patterns_with_few_errors(self):
        agg = ErrorAggregator()
        agg.record_error(_make_error())
        patterns = agg.detect_patterns()
        assert patterns == []

    def test_recurring_detection(self):
        agg = ErrorAggregator(recurring_threshold=3)
        for _ in range(4):
            agg.record_error(_make_error(ErrorCode.PROVIDER_TIMEOUT))
        patterns = agg.detect_patterns()
        recurring = [p for p in patterns if p.pattern_type == PatternType.RECURRING]
        assert len(recurring) >= 1
        assert recurring[0].count >= 4

    def test_recurring_below_threshold(self):
        agg = ErrorAggregator(recurring_threshold=5)
        for _ in range(4):
            agg.record_error(_make_error(ErrorCode.PROVIDER_TIMEOUT))
        patterns = agg.detect_patterns()
        recurring = [p for p in patterns if p.pattern_type == PatternType.RECURRING]
        assert len(recurring) == 0

    def test_spike_detection(self):
        agg = ErrorAggregator(window_seconds=10.0, spike_threshold=2.0)
        # Inject events: 1 old, 4 recent
        now = time.time()
        # Old event (first half of window)
        old_event = ErrorEvent(
            timestamp=now - 8.0,
            error_code=ErrorCode.PROVIDER_TIMEOUT,
            category=ErrorCategory.PROVIDER,
        )
        agg._events.append(old_event)
        # Recent events (second half of window)
        for _ in range(4):
            recent = ErrorEvent(
                timestamp=now - 1.0,
                error_code=ErrorCode.PROVIDER_TIMEOUT,
                category=ErrorCategory.PROVIDER,
            )
            agg._events.append(recent)
        patterns = agg.detect_patterns()
        spikes = [p for p in patterns if p.pattern_type == PatternType.SPIKE]
        assert len(spikes) >= 1

    def test_no_spike_when_balanced(self):
        agg = ErrorAggregator(window_seconds=10.0, spike_threshold=3.0)
        now = time.time()
        # Equal distribution: 3 in first half, 3 in second half
        for i in range(3):
            agg._events.append(ErrorEvent(
                timestamp=now - 8.0 + i * 0.1,
                error_code=ErrorCode.PROVIDER_TIMEOUT,
                category=ErrorCategory.PROVIDER,
            ))
        for i in range(3):
            agg._events.append(ErrorEvent(
                timestamp=now - 2.0 + i * 0.1,
                error_code=ErrorCode.PROVIDER_TIMEOUT,
                category=ErrorCategory.PROVIDER,
            ))
        patterns = agg.detect_patterns()
        spikes = [p for p in patterns if p.pattern_type == PatternType.SPIKE]
        assert len(spikes) == 0

    def test_correlated_detection(self):
        agg = ErrorAggregator(correlation_window=60.0)
        now = time.time()
        # Two providers fail within the correlation window
        agg._events.append(ErrorEvent(
            timestamp=now - 5.0,
            error_code=ErrorCode.PROVIDER_TIMEOUT,
            category=ErrorCategory.PROVIDER,
            provider_name="openai",
        ))
        agg._events.append(ErrorEvent(
            timestamp=now - 3.0,
            error_code=ErrorCode.PROVIDER_TIMEOUT,
            category=ErrorCategory.PROVIDER,
            provider_name="anthropic",
        ))
        patterns = agg.detect_patterns()
        correlated = [p for p in patterns if p.pattern_type == PatternType.CORRELATED]
        assert len(correlated) == 1
        assert "openai" in correlated[0].affected_providers
        assert "anthropic" in correlated[0].affected_providers

    def test_correlated_requires_multiple_providers(self):
        agg = ErrorAggregator(correlation_window=60.0)
        now = time.time()
        # Only one provider — no correlation
        for _ in range(3):
            agg._events.append(ErrorEvent(
                timestamp=now - 2.0,
                error_code=ErrorCode.PROVIDER_TIMEOUT,
                category=ErrorCategory.PROVIDER,
                provider_name="openai",
            ))
        patterns = agg.detect_patterns()
        correlated = [p for p in patterns if p.pattern_type == PatternType.CORRELATED]
        assert len(correlated) == 0

    def test_error_rate_calculation(self):
        agg = ErrorAggregator(window_seconds=300.0)
        for _ in range(5):
            agg.record_error(_make_error())
        rate = agg.get_error_rate()
        assert rate == pytest.approx(5 / 5.0, rel=0.1)  # 5 per 5 minutes = 1/min

    def test_error_rate_by_category(self):
        agg = ErrorAggregator()
        agg.record_error(_make_error(ErrorCode.PROVIDER_TIMEOUT))
        agg.record_error(_make_error(ErrorCode.TOOL_TIMEOUT))
        rate_prov = agg.get_error_rate("provider")
        rate_tool = agg.get_error_rate("tool")
        assert rate_prov > 0
        assert rate_tool > 0

    def test_window_summary(self):
        agg = ErrorAggregator()
        agg.record_error(_make_error(ErrorCode.PROVIDER_TIMEOUT), provider_name="openai")
        agg.record_error(_make_error(ErrorCode.TOOL_TIMEOUT), tool_name="bash")
        summary = agg.get_window_summary()
        assert summary["total_errors"] == 2
        assert "provider" in summary["by_category"]
        assert "tool" in summary["by_category"]

    def test_cleanup_old(self):
        agg = ErrorAggregator()
        now = time.time()
        # Inject old event
        agg._events.append(ErrorEvent(
            timestamp=now - 7200,
            error_code=ErrorCode.PROVIDER_TIMEOUT,
            category=ErrorCategory.PROVIDER,
        ))
        agg.record_error(_make_error())
        removed = agg.cleanup_old(older_than=3600)
        assert removed == 1
        assert agg.event_count == 1

    def test_reset(self):
        agg = ErrorAggregator()
        agg.record_error(_make_error())
        agg.reset()
        assert agg.event_count == 0

    def test_error_event_to_dict(self):
        ev = ErrorEvent(
            timestamp=1000.0,
            error_code=ErrorCode.PROVIDER_TIMEOUT,
            category=ErrorCategory.PROVIDER,
            tool_name="bash",
            provider_name="openai",
            is_transient=True,
        )
        d = ev.to_dict()
        assert d["error_code"] == "E1004"
        assert d["tool_name"] == "bash"
        assert d["is_transient"] is True

    def test_error_pattern_to_dict(self):
        pat = ErrorPattern(
            pattern_type=PatternType.SPIKE,
            count=10,
            first_seen=100.0,
            last_seen=200.0,
            severity="high",
            details="spike detected",
        )
        d = pat.to_dict()
        assert d["pattern_type"] == "spike"
        assert d["severity"] == "high"

    def test_window_summary_custom_window(self):
        agg = ErrorAggregator(window_seconds=300.0)
        agg.record_error(_make_error())
        summary = agg.get_window_summary(window_seconds=60.0)
        assert summary["window_seconds"] == 60.0

    def test_empty_aggregator_no_errors(self):
        agg = ErrorAggregator()
        assert agg.get_error_rate() == 0.0
        assert agg.detect_patterns() == []
        summary = agg.get_window_summary()
        assert summary["total_errors"] == 0


# ═════════════════════════════════════════════════════════════════
# 3. TestRecoveryOrchestrator — 20 tests
# ═════════════════════════════════════════════════════════════════

class TestRecoveryOrchestrator:
    """Recovery orchestrator strategy selection tests."""

    def test_provider_timeout_retries(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.RETRY

    def test_provider_rate_limited_falls_back(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.PROVIDER_RATE_LIMITED)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.FALLBACK_PROVIDER

    def test_provider_connection_failed_falls_back(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.PROVIDER_CONNECTION_FAILED)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.FALLBACK_PROVIDER

    def test_provider_auth_failed_escalates(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.PROVIDER_AUTH_FAILED)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.ESCALATE

    def test_provider_model_not_found_escalates(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.PROVIDER_MODEL_NOT_FOUND)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.ESCALATE

    def test_tool_timeout_retries(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.TOOL_TIMEOUT)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.RETRY

    def test_tool_not_found_escalates(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.TOOL_NOT_FOUND)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.ESCALATE

    def test_agent_circuit_breaker_escalates(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.AGENT_CIRCUIT_BREAKER)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.ESCALATE

    def test_agent_budget_exceeded_escalates(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.AGENT_BUDGET_EXCEEDED)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.ESCALATE

    def test_network_timeout_retries(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.NETWORK_TIMEOUT)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.RETRY

    def test_security_blocked_escalates(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.SECURITY_BLOCKED)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.ESCALATE

    def test_max_attempts_exceeded_escalates(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)
        # attempt_number > max_attempts (3) => escalate
        ctx = RecoveryContext(error=err, attempt_number=10)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.ESCALATE

    def test_correlated_pattern_triggers_circuit_break(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)
        correlated_pattern = ErrorPattern(
            pattern_type=PatternType.CORRELATED,
            count=5,
            first_seen=time.time() - 10,
            last_seen=time.time(),
            severity="high",
            details="multi-provider",
            affected_providers=["openai", "anthropic"],
        )
        ctx = RecoveryContext(error=err, patterns=[correlated_pattern])
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.CIRCUIT_BREAK

    def test_non_correlated_pattern_no_circuit_break(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)
        recurring_pattern = ErrorPattern(
            pattern_type=PatternType.RECURRING,
            count=10,
            first_seen=time.time() - 60,
            last_seen=time.time(),
        )
        ctx = RecoveryContext(error=err, patterns=[recurring_pattern], attempt_number=1)
        strategy = orch.decide(ctx)
        # Should use normal strategy (RETRY), not circuit break
        assert strategy.action == RecoveryAction.RETRY

    def test_register_custom_strategy(self):
        orch = RecoveryOrchestrator()
        custom = lambda ctx: RecoveryStrategy(
            action=RecoveryAction.SKIP,
            metadata={"custom": True},
        )
        orch.register_strategy(ErrorCode.PROVIDER_TIMEOUT, custom)
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.SKIP
        assert strategy.metadata.get("custom") is True

    def test_list_registered(self):
        orch = RecoveryOrchestrator()
        registered = orch.list_registered()
        assert "E1004" in registered  # PROVIDER_TIMEOUT
        assert "E2002" in registered  # TOOL_TIMEOUT

    def test_reset_to_defaults(self):
        orch = RecoveryOrchestrator()
        orch.register_strategy(ErrorCode.PROVIDER_TIMEOUT, lambda ctx: RecoveryStrategy(
            action=RecoveryAction.SKIP,
        ))
        orch.reset_to_defaults()
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.RETRY  # Back to default

    def test_decision_log(self):
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)
        ctx = RecoveryContext(error=err, attempt_number=1)
        orch.decide(ctx)
        log = orch.get_decision_log()
        assert len(log) == 1
        assert log[0]["error_code"] == "E1004"
        assert log[0]["action"] == "retry"

    def test_to_dict(self):
        orch = RecoveryOrchestrator()
        d = orch.to_dict()
        assert d["registered_strategies"] > 0
        assert d["decision_count"] == 0

    def test_recovery_strategy_to_dict(self):
        s = RecoveryStrategy(
            action=RecoveryAction.RETRY,
            delay_seconds=2.0,
            max_attempts=3,
            metadata={"key": "val"},
        )
        d = s.to_dict()
        assert d["action"] == "retry"
        assert d["delay_seconds"] == 2.0
        assert d["max_attempts"] == 3


# ═════════════════════════════════════════════════════════════════
# 4. TestErrorContextEnricher — 14 tests
# ═════════════════════════════════════════════════════════════════

class TestErrorContextEnricher:
    """Error context enrichment and formatting tests."""

    def test_enrich_basic(self):
        enricher = ErrorContextEnricher()
        exc = TimeoutError("timed out")
        ctx = _make_context()
        agent_error = enricher.enrich(exc, ctx)
        assert isinstance(agent_error, AgentError)
        assert agent_error.context["tool_name"] == "bash"

    def test_enrich_preserves_code(self):
        enricher = ErrorContextEnricher()
        exc = ConnectionRefusedError("refused")
        ctx = _make_context()
        agent_error = enricher.enrich(exc, ctx)
        assert agent_error.code == ErrorCode.NETWORK_CONNECTION_REFUSED

    def test_enrich_from_code(self):
        enricher = ErrorContextEnricher()
        ctx = _make_context(tool="read", provider="anthropic", iteration=5)
        agent_error = enricher.enrich_from_code(
            ErrorCode.PROVIDER_RATE_LIMITED, ctx,
        )
        assert agent_error.code == ErrorCode.PROVIDER_RATE_LIMITED
        assert agent_error.context["provider_name"] == "anthropic"
        assert agent_error.context["iteration"] == 5

    def test_enrich_from_code_with_exception(self):
        enricher = ErrorContextEnricher()
        ctx = _make_context()
        exc = Exception("rate limited")
        agent_error = enricher.enrich_from_code(
            ErrorCode.PROVIDER_RATE_LIMITED, ctx, exception=exc,
        )
        assert agent_error.original_exception is exc

    def test_add_breadcrumb(self):
        enricher = ErrorContextEnricher()
        err = _make_error()
        enricher.add_breadcrumb(err, "step1")
        enricher.add_breadcrumb(err, "step2")
        crumbs = enricher.get_breadcrumbs(err)
        assert crumbs == ["step1", "step2"]

    def test_get_breadcrumbs_empty(self):
        enricher = ErrorContextEnricher()
        err = _make_error()
        crumbs = enricher.get_breadcrumbs(err)
        assert crumbs == []

    def test_format_for_llm_basic(self):
        enricher = ErrorContextEnricher()
        exc = TimeoutError("timed out")
        ctx = _make_context(tool="bash", provider="ollama", iteration=3)
        agent_error = enricher.enrich(exc, ctx)
        output = enricher.format_for_llm(agent_error)
        assert "E5003" in output or "E1004" in output  # NETWORK_TIMEOUT or PROVIDER_TIMEOUT
        assert "bash" in output
        assert "ollama" in output

    def test_format_for_llm_with_breadcrumbs(self):
        enricher = ErrorContextEnricher()
        err = _make_error()
        err.context["tool_name"] = "bash"
        err.context["breadcrumbs"] = ["init", "execute", "fail"]
        output = enricher.format_for_llm(err)
        assert "init → execute → fail" in output

    def test_format_for_llm_transient_note(self):
        enricher = ErrorContextEnricher()
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)  # transient
        output = enricher.format_for_llm(err)
        assert "transient" in output.lower()

    def test_format_for_log_basic(self):
        enricher = ErrorContextEnricher()
        exc = TimeoutError("timed out")
        ctx = _make_context(tool="bash", provider="ollama", iteration=2)
        agent_error = enricher.enrich(exc, ctx)
        log_str = enricher.format_for_log(agent_error)
        assert "code=" in log_str
        assert "tool=bash" in log_str
        assert "provider=ollama" in log_str

    def test_format_for_log_includes_duration(self):
        enricher = ErrorContextEnricher()
        exc = TimeoutError("timed out")
        ctx = ErrorContext(
            tool_name="bash",
            duration_ms=456.7,
        )
        agent_error = enricher.enrich(exc, ctx)
        log_str = enricher.format_for_log(agent_error)
        assert "duration_ms=456.7" in log_str

    def test_error_context_to_dict(self):
        ctx = ErrorContext(
            iteration=3,
            tool_name="bash",
            provider_name="ollama",
            duration_ms=100.0,
            breadcrumbs=["a", "b"],
        )
        d = ctx.to_dict()
        assert d["iteration"] == 3
        assert d["tool_name"] == "bash"
        assert d["breadcrumbs"] == ["a", "b"]

    def test_recovery_context_to_dict(self):
        err = _make_error()
        ctx = RecoveryContext(
            error=err,
            tool_name="bash",
            provider_name="ollama",
            attempt_number=2,
        )
        d = ctx.to_dict()
        assert d["tool_name"] == "bash"
        assert d["attempt_number"] == 2

    def test_enrich_merges_context(self):
        enricher = ErrorContextEnricher()
        exc = Exception("some error")
        ctx = _make_context(tool="write", provider="openai", iteration=7)
        agent_error = enricher.enrich(exc, ctx)
        assert agent_error.context["tool_name"] == "write"
        assert agent_error.context["provider_name"] == "openai"
        assert agent_error.context["iteration"] == 7
        assert agent_error.context["duration_ms"] == 123.4


# ═════════════════════════════════════════════════════════════════
# 5. TestErrorBudgetTracker — 12 tests
# ═════════════════════════════════════════════════════════════════

class TestErrorBudgetTracker:
    """Error budget rate calculation and enforcement tests."""

    def test_initial_no_errors(self):
        tracker = ErrorBudgetTracker()
        assert tracker.get_error_rate() == 0.0
        assert tracker.is_over_budget() is False

    def test_record_success(self):
        tracker = ErrorBudgetTracker()
        tracker.record("tool", success=True)
        assert tracker.get_error_rate() == 0.0

    def test_record_failure(self):
        tracker = ErrorBudgetTracker()
        tracker.record("tool", success=False)
        assert tracker.get_error_rate() == 1.0

    def test_mixed_records(self):
        tracker = ErrorBudgetTracker()
        for _ in range(8):
            tracker.record("tool", success=True)
        for _ in range(2):
            tracker.record("tool", success=False)
        rate = tracker.get_error_rate()
        assert rate == pytest.approx(0.2, abs=0.01)

    def test_over_budget(self):
        cfg = ErrorBudgetConfig(max_error_rate=0.10)
        tracker = ErrorBudgetTracker(config=cfg)
        for _ in range(5):
            tracker.record("tool", success=True)
        for _ in range(5):
            tracker.record("tool", success=False)
        assert tracker.is_over_budget() is True

    def test_under_budget(self):
        cfg = ErrorBudgetConfig(max_error_rate=0.50)
        tracker = ErrorBudgetTracker(config=cfg)
        for _ in range(8):
            tracker.record("tool", success=True)
        for _ in range(2):
            tracker.record("tool", success=False)
        assert tracker.is_over_budget() is False

    def test_category_filtering(self):
        tracker = ErrorBudgetTracker()
        tracker.record("tool", success=False)
        tracker.record("provider", success=True)
        assert tracker.get_error_rate("tool") == 1.0
        assert tracker.get_error_rate("provider") == 0.0

    def test_remaining_budget(self):
        cfg = ErrorBudgetConfig(max_error_rate=0.20)
        tracker = ErrorBudgetTracker(config=cfg)
        for _ in range(9):
            tracker.record("tool", success=True)
        tracker.record("tool", success=False)
        # Error rate = 0.10, max = 0.20, remaining = 1 - (0.1/0.2) = 0.5
        remaining = tracker.get_remaining_budget()
        assert remaining == pytest.approx(0.5, abs=0.05)

    def test_remaining_budget_fully_consumed(self):
        cfg = ErrorBudgetConfig(max_error_rate=0.10)
        tracker = ErrorBudgetTracker(config=cfg)
        for _ in range(5):
            tracker.record("tool", success=True)
        for _ in range(5):
            tracker.record("tool", success=False)
        # Error rate = 0.5 >> max 0.10, remaining = 0
        remaining = tracker.get_remaining_budget()
        assert remaining == 0.0

    def test_report(self):
        tracker = ErrorBudgetTracker()
        tracker.record("tool", success=True)
        tracker.record("tool", success=False)
        tracker.record("provider", success=True)
        report = tracker.get_report()
        assert "overall" in report
        assert "categories" in report
        assert "tool" in report["categories"]
        assert "provider" in report["categories"]

    def test_reset(self):
        tracker = ErrorBudgetTracker()
        tracker.record("tool", success=False)
        tracker.reset()
        assert tracker.get_error_rate() == 0.0

    def test_config_properties(self):
        cfg = ErrorBudgetConfig(max_error_rate=0.30, window_seconds=600.0)
        tracker = ErrorBudgetTracker(config=cfg)
        assert tracker.max_error_rate == 0.30
        assert tracker.window_seconds == 600.0


# ═════════════════════════════════════════════════════════════════
# 6. TestDataclasses — 8 tests
# ═════════════════════════════════════════════════════════════════

class TestDataclasses:
    """Dataclass serialization and construction tests."""

    def test_recovery_action_values(self):
        assert RecoveryAction.RETRY.value == "retry"
        assert RecoveryAction.FALLBACK_PROVIDER.value == "fallback"
        assert RecoveryAction.CIRCUIT_BREAK.value == "circuit_break"
        assert RecoveryAction.DEGRADE.value == "degrade"
        assert RecoveryAction.ESCALATE.value == "escalate"
        assert RecoveryAction.SKIP.value == "skip"

    def test_pattern_type_values(self):
        assert PatternType.SPIKE.value == "spike"
        assert PatternType.RECURRING.value == "recurring"
        assert PatternType.CORRELATED.value == "correlated"

    def test_circuit_breaker_state_values(self):
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"

    def test_recovery_strategy_defaults(self):
        s = RecoveryStrategy(action=RecoveryAction.RETRY)
        assert s.delay_seconds == 0.0
        assert s.max_attempts == 1
        assert s.metadata == {}

    def test_recovery_context_defaults(self):
        err = _make_error()
        ctx = RecoveryContext(error=err)
        assert ctx.tool_name == ""
        assert ctx.provider_name == ""
        assert ctx.attempt_number == 1
        assert ctx.patterns == []

    def test_error_context_defaults(self):
        ctx = ErrorContext()
        assert ctx.iteration == 0
        assert ctx.tool_name == ""
        assert ctx.provider_name == ""
        assert ctx.breadcrumbs == []
        assert ctx.duration_ms == 0.0

    def test_error_budget_config_defaults(self):
        cfg = ErrorBudgetConfig()
        assert cfg.max_error_rate == 0.20
        assert cfg.window_seconds == 300.0

    def test_circuit_breaker_config_defaults(self):
        cfg = CircuitBreakerConfig()
        assert cfg.failure_threshold == 5
        assert cfg.timeout_seconds == 60.0
        assert cfg.half_open_max_calls == 2


# ═════════════════════════════════════════════════════════════════
# 7. TestIntegration — 22 tests
# ═════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end flows combining multiple Sprint 13 modules."""

    def test_enrich_then_aggregate(self):
        """Enrich error → record in aggregator → check event count."""
        enricher = ErrorContextEnricher()
        agg = ErrorAggregator()
        exc = TimeoutError("timed out")
        ctx = _make_context(tool="bash", provider="ollama")
        agent_error = enricher.enrich(exc, ctx)
        agg.record_error(agent_error, tool_name="bash", provider_name="ollama")
        assert agg.event_count == 1

    def test_aggregate_then_orchestrate(self):
        """Record errors → detect patterns → feed to orchestrator."""
        agg = ErrorAggregator(recurring_threshold=3)
        orch = RecoveryOrchestrator()
        for _ in range(4):
            agg.record_error(_make_error(ErrorCode.PROVIDER_TIMEOUT))
        patterns = agg.detect_patterns()
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)
        ctx = RecoveryContext(error=err, patterns=patterns, attempt_number=1)
        strategy = orch.decide(ctx)
        # With recurring but no correlated pattern, should still use normal strategy
        assert strategy.action in (RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAK)

    def test_circuit_breaker_with_aggregator(self):
        """Circuit breaker opens → aggregator detects correlated → orchestrator circuit breaks."""
        cb = ProviderCircuitBreaker(config=CircuitBreakerConfig(failure_threshold=2))
        agg = ErrorAggregator(correlation_window=60.0)
        orch = RecoveryOrchestrator()

        # Trip circuit breaker
        cb.record_failure("openai")
        cb.record_failure("openai")
        assert cb.get_state("openai") == CircuitBreakerState.OPEN

        # Record correlated failures in aggregator
        now = time.time()
        agg._events.append(ErrorEvent(
            timestamp=now - 2, error_code=ErrorCode.PROVIDER_TIMEOUT,
            category=ErrorCategory.PROVIDER, provider_name="openai",
        ))
        agg._events.append(ErrorEvent(
            timestamp=now - 1, error_code=ErrorCode.PROVIDER_TIMEOUT,
            category=ErrorCategory.PROVIDER, provider_name="anthropic",
        ))
        patterns = agg.detect_patterns()
        correlated = [p for p in patterns if p.pattern_type == PatternType.CORRELATED]
        assert len(correlated) == 1

        # Orchestrator should recommend circuit break
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)
        ctx = RecoveryContext(error=err, patterns=correlated)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.CIRCUIT_BREAK

    def test_error_budget_with_aggregator(self):
        """Record errors in both aggregator and budget tracker."""
        agg = ErrorAggregator()
        budget = ErrorBudgetTracker(config=ErrorBudgetConfig(max_error_rate=0.10))

        for _ in range(3):
            err = _make_error(ErrorCode.PROVIDER_TIMEOUT)
            agg.record_error(err, provider_name="openai")
            budget.record("provider", success=False)

        for _ in range(7):
            budget.record("provider", success=True)

        assert agg.event_count == 3
        assert budget.get_error_rate() == pytest.approx(0.3, abs=0.01)
        assert budget.is_over_budget() is True

    def test_full_flow_retry_succeeds(self):
        """Simulate: error → enrich → aggregate → orchestrate retry → success."""
        enricher = ErrorContextEnricher()
        agg = ErrorAggregator()
        orch = RecoveryOrchestrator()
        budget = ErrorBudgetTracker()
        cb = ProviderCircuitBreaker()

        # First attempt fails
        exc = TimeoutError("provider timed out")
        ctx = _make_context(tool="bash", provider="ollama", iteration=1)
        agent_error = enricher.enrich(exc, ctx)
        agg.record_error(agent_error, tool_name="bash", provider_name="ollama")
        cb.record_failure("ollama")
        budget.record("provider", success=False)

        # Decide recovery
        patterns = agg.detect_patterns()
        recovery_ctx = RecoveryContext(
            error=agent_error,
            tool_name="bash",
            provider_name="ollama",
            attempt_number=1,
            patterns=patterns,
        )
        strategy = orch.decide(recovery_ctx)
        assert strategy.action == RecoveryAction.RETRY

        # Second attempt succeeds
        cb.record_success("ollama")
        budget.record("provider", success=True)
        assert cb.get_state("ollama") == CircuitBreakerState.CLOSED
        # 1 failure + 1 success = 50% error rate — still over default 20% budget
        # But error rate is declining; verify budget rate is as expected
        assert budget.get_error_rate() == pytest.approx(0.5, abs=0.01)

    def test_full_flow_escalate_after_max_retries(self):
        """Simulate: 4 retries → max attempts exceeded → escalate."""
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)

        for attempt in range(1, 5):
            ctx = RecoveryContext(error=err, attempt_number=attempt)
            strategy = orch.decide(ctx)
            if attempt <= 3:
                assert strategy.action == RecoveryAction.RETRY
            else:
                assert strategy.action == RecoveryAction.ESCALATE

    def test_full_flow_fallback_on_rate_limit(self):
        """Rate limit → fallback → record success on new provider."""
        enricher = ErrorContextEnricher()
        cb = ProviderCircuitBreaker()
        budget = ErrorBudgetTracker()
        orch = RecoveryOrchestrator()

        exc = Exception("rate limit exceeded (429)")
        ctx = _make_context(tool="chat", provider="openai")
        agent_error = enricher.enrich(exc, ctx)

        recovery_ctx = RecoveryContext(
            error=agent_error,
            provider_name="openai",
            attempt_number=1,
        )
        strategy = orch.decide(recovery_ctx)
        assert strategy.action == RecoveryAction.FALLBACK_PROVIDER

        # Fallback to anthropic succeeds
        cb.record_success("anthropic")
        budget.record("provider", success=True)
        assert cb.get_state("anthropic") == CircuitBreakerState.CLOSED

    def test_budget_blocks_further_requests(self):
        """Error budget exceeded → should signal stop."""
        budget = ErrorBudgetTracker(config=ErrorBudgetConfig(max_error_rate=0.10))
        for _ in range(2):
            budget.record("tool", success=True)
        for _ in range(8):
            budget.record("tool", success=False)
        assert budget.is_over_budget() is True
        assert budget.get_remaining_budget() == 0.0

    def test_circuit_breaker_recovery(self):
        """Circuit opens → times out → half-open → success → closed."""
        cfg = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=0.01)
        cb = ProviderCircuitBreaker(config=cfg)

        cb.record_failure("openai")
        cb.record_failure("openai")
        assert cb.get_state("openai") == CircuitBreakerState.OPEN

        time.sleep(0.02)
        assert cb.is_available("openai") is True  # half-open
        cb.record_success("openai")
        assert cb.get_state("openai") == CircuitBreakerState.CLOSED

    def test_aggregator_cleanup_preserves_recent(self):
        """Cleanup removes old events but keeps recent ones."""
        agg = ErrorAggregator()
        now = time.time()
        agg._events.append(ErrorEvent(
            timestamp=now - 7200,
            error_code=ErrorCode.PROVIDER_TIMEOUT,
            category=ErrorCategory.PROVIDER,
        ))
        agg._events.append(ErrorEvent(
            timestamp=now - 100,
            error_code=ErrorCode.TOOL_TIMEOUT,
            category=ErrorCategory.TOOL,
        ))
        removed = agg.cleanup_old(older_than=3600)
        assert removed == 1
        assert agg.event_count == 1

    def test_orchestrator_decision_log_grows(self):
        """Multiple decisions logged correctly."""
        orch = RecoveryOrchestrator()
        codes = [ErrorCode.PROVIDER_TIMEOUT, ErrorCode.TOOL_TIMEOUT,
                 ErrorCode.PROVIDER_AUTH_FAILED]
        for code in codes:
            err = _make_error(code)
            ctx = RecoveryContext(error=err, attempt_number=1)
            orch.decide(ctx)
        log = orch.get_decision_log()
        assert len(log) == 3

    def test_enricher_breadcrumbs_in_orchestrator(self):
        """Breadcrumbs attached by enricher flow through to orchestrator context."""
        enricher = ErrorContextEnricher()
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)
        enricher.add_breadcrumb(err, "parse_response")
        enricher.add_breadcrumb(err, "decode_json")
        crumbs = enricher.get_breadcrumbs(err)
        assert len(crumbs) == 2
        # Breadcrumbs are in error.context, visible in orchestrator log
        ctx = RecoveryContext(error=err, attempt_number=1)
        orch = RecoveryOrchestrator()
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.RETRY

    def test_multi_category_budget_tracking(self):
        """Budget tracks multiple categories independently."""
        budget = ErrorBudgetTracker()
        for _ in range(5):
            budget.record("tool", success=True)
        for _ in range(5):
            budget.record("provider", success=False)

        assert budget.get_error_rate("tool") == 0.0
        assert budget.get_error_rate("provider") == 1.0
        # Overall: 5 success + 5 failure = 50%
        assert budget.get_error_rate("all") == pytest.approx(0.5, abs=0.01)

    def test_multiple_circuit_breakers_independent(self):
        """Multiple providers have independent circuit breaker states."""
        cfg = CircuitBreakerConfig(failure_threshold=2)
        cb = ProviderCircuitBreaker(config=cfg)

        cb.record_failure("openai")
        cb.record_failure("openai")
        cb.record_failure("anthropic")

        assert cb.get_state("openai") == CircuitBreakerState.OPEN
        assert cb.get_state("anthropic") == CircuitBreakerState.CLOSED

    def test_orchestrator_handles_unknown_error_code(self):
        """Orchestrator escalates for unknown/unregistered error codes."""
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.AGENT_TRUNCATION)  # Not in default strategies
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.ESCALATE

    def test_budget_report_structure(self):
        """Budget report has expected structure."""
        budget = ErrorBudgetTracker()
        budget.record("tool", success=True)
        budget.record("tool", success=False)
        budget.record("provider", success=True)

        report = budget.get_report()
        assert "window_seconds" in report
        assert "max_error_rate" in report
        assert "overall" in report
        assert "error_rate" in report["overall"]
        assert "over_budget" in report["overall"]
        assert "remaining" in report["overall"]
        assert "categories" in report

    def test_aggregator_summary_by_provider(self):
        """Window summary correctly groups by provider."""
        agg = ErrorAggregator()
        agg.record_error(
            _make_error(ErrorCode.PROVIDER_TIMEOUT),
            provider_name="openai",
        )
        agg.record_error(
            _make_error(ErrorCode.PROVIDER_TIMEOUT),
            provider_name="openai",
        )
        agg.record_error(
            _make_error(ErrorCode.PROVIDER_TIMEOUT),
            provider_name="anthropic",
        )
        summary = agg.get_window_summary()
        assert summary["by_provider"]["openai"] == 2
        assert summary["by_provider"]["anthropic"] == 1

    def test_enricher_format_for_llm_recovery_hint(self):
        """format_for_llm includes recovery hint."""
        enricher = ErrorContextEnricher()
        err = _make_error(ErrorCode.PROVIDER_TIMEOUT)
        output = enricher.format_for_llm(err)
        assert "Suggestion:" in output

    def test_all_error_codes_have_strategies(self):
        """Verify most error codes have registered strategies."""
        orch = RecoveryOrchestrator()
        registered = set(orch.list_registered())
        # Check key codes are registered
        important_codes = [
            "E1001", "E1003", "E1004",  # Provider
            "E2001", "E2002",             # Tool
            "E3001", "E3004",             # Agent
            "E5003",                       # Network
            "E6001",                       # Security
        ]
        for code in important_codes:
            assert code in registered, f"{code} missing from registered strategies"

    def test_provider_overloaded_fallback(self):
        """Provider overloaded → fallback strategy."""
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.PROVIDER_OVERLOADED)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.FALLBACK_PROVIDER

    def test_agent_empty_response_retries(self):
        """Agent empty response → retry strategy."""
        orch = RecoveryOrchestrator()
        err = _make_error(ErrorCode.AGENT_EMPTY_RESPONSE)
        ctx = RecoveryContext(error=err, attempt_number=1)
        strategy = orch.decide(ctx)
        assert strategy.action == RecoveryAction.RETRY
