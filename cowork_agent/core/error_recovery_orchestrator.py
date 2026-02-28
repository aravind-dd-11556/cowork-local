"""
Error Recovery Orchestrator — Central coordinator for recovery strategies.

Decides the recovery action for any error based on:
  - Error code and category
  - Detected patterns (from ErrorAggregator)
  - Custom strategy overrides

Supports pluggable strategies: register custom recovery logic per error code.

Sprint 13 (Error Recovery & Resilience) Module 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .error_catalog import AgentError, ErrorCode, ErrorCategory

logger = logging.getLogger(__name__)


# ── Recovery Actions ─────────────────────────────────────────────

class RecoveryAction(Enum):
    """Possible recovery actions the orchestrator can recommend."""
    RETRY = "retry"                      # Retry the same operation
    FALLBACK_PROVIDER = "fallback"       # Switch to a fallback provider
    CIRCUIT_BREAK = "circuit_break"      # Open the circuit breaker
    DEGRADE = "degrade"                  # Use a simpler/faster model
    ESCALATE = "escalate"               # Give up and report to user
    SKIP = "skip"                        # Skip this operation


@dataclass
class RecoveryStrategy:
    """A recommended recovery strategy."""
    action: RecoveryAction
    delay_seconds: float = 0.0
    max_attempts: int = 1
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "delay_seconds": self.delay_seconds,
            "max_attempts": self.max_attempts,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryContext:
    """Context for making a recovery decision."""
    error: AgentError
    tool_name: str = ""
    provider_name: str = ""
    attempt_number: int = 1
    patterns: list = field(default_factory=list)  # list[ErrorPattern]

    def to_dict(self) -> dict:
        return {
            "error_code": self.error.code.value,
            "tool_name": self.tool_name,
            "provider_name": self.provider_name,
            "attempt_number": self.attempt_number,
            "pattern_count": len(self.patterns),
        }


# Type for strategy functions
StrategyFn = Callable[[RecoveryContext], RecoveryStrategy]


# ── Default Strategy Mappings ────────────────────────────────────

def _strategy_retry(delay: float = 2.0, max_attempts: int = 3) -> StrategyFn:
    """Create a retry strategy function."""
    def _fn(ctx: RecoveryContext) -> RecoveryStrategy:
        return RecoveryStrategy(
            action=RecoveryAction.RETRY,
            delay_seconds=delay,
            max_attempts=max_attempts,
        )
    return _fn


def _strategy_fallback(delay: float = 3.0) -> StrategyFn:
    """Create a fallback provider strategy function."""
    def _fn(ctx: RecoveryContext) -> RecoveryStrategy:
        return RecoveryStrategy(
            action=RecoveryAction.FALLBACK_PROVIDER,
            delay_seconds=delay,
            max_attempts=2,
            metadata={"reason": "provider_failure"},
        )
    return _fn


def _strategy_escalate() -> StrategyFn:
    """Create an escalation strategy function."""
    def _fn(ctx: RecoveryContext) -> RecoveryStrategy:
        return RecoveryStrategy(
            action=RecoveryAction.ESCALATE,
            delay_seconds=0.0,
            max_attempts=0,
            metadata={"reason": "unrecoverable"},
        )
    return _fn


# Default strategies keyed by ErrorCode
_DEFAULT_STRATEGIES: dict[ErrorCode, StrategyFn] = {
    # Provider errors — mostly retryable/fallback-able
    ErrorCode.PROVIDER_RATE_LIMITED: _strategy_fallback(delay=5.0),
    ErrorCode.PROVIDER_TIMEOUT: _strategy_retry(delay=2.0, max_attempts=3),
    ErrorCode.PROVIDER_CONNECTION_FAILED: _strategy_fallback(delay=3.0),
    ErrorCode.PROVIDER_OVERLOADED: _strategy_fallback(delay=5.0),
    ErrorCode.PROVIDER_INVALID_RESPONSE: _strategy_retry(delay=1.0, max_attempts=2),
    ErrorCode.PROVIDER_AUTH_FAILED: _strategy_escalate(),
    ErrorCode.PROVIDER_MODEL_NOT_FOUND: _strategy_escalate(),

    # Tool errors
    ErrorCode.TOOL_TIMEOUT: _strategy_retry(delay=5.0, max_attempts=1),
    ErrorCode.TOOL_EXECUTION_FAILED: _strategy_retry(delay=1.0, max_attempts=1),
    ErrorCode.TOOL_NOT_FOUND: _strategy_escalate(),
    ErrorCode.TOOL_VALIDATION_FAILED: _strategy_escalate(),
    ErrorCode.TOOL_PERMISSION_DENIED: _strategy_escalate(),

    # Agent errors
    ErrorCode.AGENT_CIRCUIT_BREAKER: _strategy_escalate(),
    ErrorCode.AGENT_BUDGET_EXCEEDED: _strategy_escalate(),
    ErrorCode.AGENT_MAX_ITERATIONS: _strategy_escalate(),
    ErrorCode.AGENT_LOOP_DETECTED: _strategy_escalate(),
    ErrorCode.AGENT_EMPTY_RESPONSE: _strategy_retry(delay=1.0, max_attempts=2),

    # Network errors
    ErrorCode.NETWORK_TIMEOUT: _strategy_retry(delay=3.0, max_attempts=3),
    ErrorCode.NETWORK_CONNECTION_REFUSED: _strategy_retry(delay=5.0, max_attempts=2),
    ErrorCode.NETWORK_DNS_FAILED: _strategy_retry(delay=10.0, max_attempts=2),
    ErrorCode.NETWORK_SSL_ERROR: _strategy_escalate(),

    # Security errors
    ErrorCode.SECURITY_BLOCKED: _strategy_escalate(),
    ErrorCode.SECURITY_INJECTION: _strategy_escalate(),

    # Config errors
    ErrorCode.CONFIG_MISSING_KEY: _strategy_escalate(),
    ErrorCode.CONFIG_INVALID_VALUE: _strategy_escalate(),
    ErrorCode.CONFIG_FILE_NOT_FOUND: _strategy_escalate(),
}


# ── Recovery Orchestrator ────────────────────────────────────────

class RecoveryOrchestrator:
    """
    Central coordinator for error recovery decisions.

    Maintains a registry of strategies (default + custom) and selects
    the appropriate recovery action based on error context and patterns.

    Usage:
        orch = RecoveryOrchestrator()
        ctx = RecoveryContext(error=agent_error, tool_name="bash")
        strategy = orch.decide(ctx)
        if strategy.action == RecoveryAction.RETRY:
            await asyncio.sleep(strategy.delay_seconds)
            # retry...
    """

    def __init__(self):
        # Copy defaults so custom strategies don't mutate module-level dict
        self._strategies: dict[ErrorCode, StrategyFn] = dict(_DEFAULT_STRATEGIES)
        self._decision_log: list[dict] = []  # Last N decisions for observability
        self._max_log_size = 100

    def decide(self, context: RecoveryContext) -> RecoveryStrategy:
        """
        Select a recovery strategy for the given error context.

        Decision logic:
        1. If attempt_number exceeds max_attempts from strategy → escalate
        2. If patterns indicate systemic failure → circuit_break
        3. Otherwise, use registered strategy for error code
        4. Fallback to ESCALATE if no strategy registered
        """
        error_code = context.error.code

        # Check for systemic patterns (correlated failures = circuit break)
        if context.patterns:
            from .error_aggregator import PatternType
            for pattern in context.patterns:
                if pattern.pattern_type == PatternType.CORRELATED:
                    strategy = RecoveryStrategy(
                        action=RecoveryAction.CIRCUIT_BREAK,
                        metadata={"reason": "correlated_failure", "pattern": pattern.details},
                    )
                    self._log_decision(context, strategy)
                    return strategy

        # Look up registered strategy
        strategy_fn = self._strategies.get(error_code)
        if strategy_fn is None:
            strategy = RecoveryStrategy(
                action=RecoveryAction.ESCALATE,
                metadata={"reason": f"no_strategy_for_{error_code.value}"},
            )
            self._log_decision(context, strategy)
            return strategy

        strategy = strategy_fn(context)

        # If we've exceeded max attempts, escalate
        if context.attempt_number > strategy.max_attempts:
            strategy = RecoveryStrategy(
                action=RecoveryAction.ESCALATE,
                metadata={"reason": "max_attempts_exceeded",
                          "attempts": context.attempt_number},
            )

        self._log_decision(context, strategy)
        return strategy

    def register_strategy(self, error_code: ErrorCode, strategy_fn: StrategyFn) -> None:
        """Register a custom recovery strategy for an error code."""
        self._strategies[error_code] = strategy_fn
        logger.info(f"Registered custom recovery strategy for {error_code.value}")

    def get_strategy(self, error_code: ErrorCode) -> Optional[StrategyFn]:
        """Get the registered strategy function for an error code."""
        return self._strategies.get(error_code)

    def list_registered(self) -> list[str]:
        """List all error codes with registered strategies."""
        return [code.value for code in self._strategies]

    def reset_to_defaults(self) -> None:
        """Reset all strategies to defaults."""
        self._strategies = dict(_DEFAULT_STRATEGIES)
        logger.info("Recovery strategies reset to defaults")

    def get_decision_log(self) -> list[dict]:
        """Get recent recovery decisions for observability."""
        return list(self._decision_log)

    def to_dict(self) -> dict:
        """Serialize orchestrator state for reporting."""
        return {
            "registered_strategies": len(self._strategies),
            "registered_codes": self.list_registered(),
            "decision_count": len(self._decision_log),
        }

    def _log_decision(self, context: RecoveryContext, strategy: RecoveryStrategy) -> None:
        """Log a recovery decision."""
        import time
        entry = {
            "timestamp": time.time(),
            "error_code": context.error.code.value,
            "action": strategy.action.value,
            "attempt": context.attempt_number,
            "tool": context.tool_name,
            "provider": context.provider_name,
        }
        self._decision_log.append(entry)

        # Trim log
        if len(self._decision_log) > self._max_log_size:
            self._decision_log = self._decision_log[-self._max_log_size:]

        logger.debug(
            f"Recovery decision: {context.error.code.value} → {strategy.action.value} "
            f"(attempt {context.attempt_number})"
        )
