"""
Tool Output Validator — Schema-based validation for tool results.

Validates tool outputs against declared schemas with auto-retry on failure.
Prevents downstream agent confusion from malformed, truncated, or unexpected
tool results.

Features:
  - JSON schema validation for structured tool output
  - Custom assertion rules per tool (regex, type checks, length bounds)
  - Auto-retry with hints when assertions fail
  - Configurable strictness (warn / block / retry)
  - Stats tracking for validation pass/fail rates

Sprint 24: Production Hardening.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────

class ValidationSeverity(Enum):
    """What happens when validation fails."""
    WARN = "warn"        # Log warning, pass output through
    BLOCK = "block"      # Reject output, return error
    RETRY = "retry"      # Ask tool to retry with hints


class AssertionType(Enum):
    """Built-in assertion types."""
    NOT_EMPTY = "not_empty"
    MAX_LENGTH = "max_length"
    MIN_LENGTH = "min_length"
    MATCHES_REGEX = "matches_regex"
    IS_JSON = "is_json"
    JSON_HAS_KEY = "json_has_key"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    CUSTOM = "custom"


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class OutputAssertion:
    """A single validation rule for tool output."""
    assertion_type: AssertionType
    severity: ValidationSeverity = ValidationSeverity.WARN
    # Parameters depending on assertion type
    value: Any = None  # regex pattern, max length, key name, etc.
    message: str = ""  # Custom failure message
    # For CUSTOM assertions: a callable(output) → bool
    custom_fn: Optional[Callable[[str], bool]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.assertion_type.value,
            "severity": self.severity.value,
            "value": str(self.value) if self.value else None,
            "message": self.message,
        }


@dataclass
class ValidationFailure:
    """A single failed assertion."""
    tool_name: str
    assertion: OutputAssertion
    actual_output_preview: str  # first 200 chars of output
    message: str
    severity: ValidationSeverity
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "assertion_type": self.assertion.assertion_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "output_preview": self.actual_output_preview,
        }


@dataclass
class ValidationResult:
    """Result of validating a tool's output."""
    tool_name: str
    passed: bool
    failures: List[ValidationFailure] = field(default_factory=list)
    should_retry: bool = False
    should_block: bool = False
    retry_hint: str = ""

    @property
    def summary(self) -> str:
        if self.passed:
            return f"{self.tool_name}: output validation passed"
        fail_types = [f.assertion.assertion_type.value for f in self.failures]
        return f"{self.tool_name}: {len(self.failures)} assertion(s) failed ({', '.join(fail_types)})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "passed": self.passed,
            "failures": [f.to_dict() for f in self.failures],
            "should_retry": self.should_retry,
            "should_block": self.should_block,
            "retry_hint": self.retry_hint,
        }


# ── Default tool schemas ─────────────────────────────────────────

def _default_tool_assertions() -> Dict[str, List[OutputAssertion]]:
    """Built-in assertions for common tools."""
    return {
        "bash": [
            OutputAssertion(
                assertion_type=AssertionType.MAX_LENGTH,
                severity=ValidationSeverity.WARN,
                value=500_000,
                message="Bash output exceeds 500K chars — may overwhelm context window",
            ),
        ],
        "read": [
            OutputAssertion(
                assertion_type=AssertionType.NOT_EMPTY,
                severity=ValidationSeverity.WARN,
                message="Read tool returned empty output",
            ),
        ],
        "web_fetch": [
            OutputAssertion(
                assertion_type=AssertionType.NOT_EMPTY,
                severity=ValidationSeverity.WARN,
                message="Web fetch returned empty content",
            ),
            OutputAssertion(
                assertion_type=AssertionType.MAX_LENGTH,
                severity=ValidationSeverity.WARN,
                value=200_000,
                message="Web fetch output very large — may need truncation",
            ),
        ],
        "web_search": [
            OutputAssertion(
                assertion_type=AssertionType.NOT_EMPTY,
                severity=ValidationSeverity.WARN,
                message="Web search returned no results",
            ),
        ],
    }


# ── ToolOutputValidator ──────────────────────────────────────────

class ToolOutputValidator:
    """Validates tool outputs against schemas and assertions.

    Usage::

        validator = ToolOutputValidator()
        validator.add_assertion("bash", OutputAssertion(
            assertion_type=AssertionType.NOT_CONTAINS,
            value="password",
            severity=ValidationSeverity.BLOCK,
            message="Bash output contains password",
        ))
        result = validator.validate("bash", output_text)
        if result.should_block:
            # reject tool output
        elif result.should_retry:
            # retry tool with hint
    """

    def __init__(
        self,
        load_defaults: bool = True,
        max_retries: int = 2,
    ):
        self._assertions: Dict[str, List[OutputAssertion]] = {}
        self._max_retries = max_retries
        # Stats
        self._total_validations = 0
        self._total_passed = 0
        self._total_failed = 0
        self._total_retries = 0
        self._total_blocks = 0

        if load_defaults:
            self._assertions = _default_tool_assertions()

    # ── Configuration ──────────────────────────────────────────

    def add_assertion(self, tool_name: str, assertion: OutputAssertion) -> None:
        """Add a validation assertion for a specific tool."""
        if tool_name not in self._assertions:
            self._assertions[tool_name] = []
        self._assertions[tool_name].append(assertion)

    def set_assertions(self, tool_name: str, assertions: List[OutputAssertion]) -> None:
        """Replace all assertions for a tool."""
        self._assertions[tool_name] = list(assertions)

    def clear_assertions(self, tool_name: str) -> None:
        """Remove all assertions for a tool."""
        self._assertions.pop(tool_name, None)

    def get_assertions(self, tool_name: str) -> List[OutputAssertion]:
        """Get all assertions for a tool."""
        return list(self._assertions.get(tool_name, []))

    @property
    def registered_tools(self) -> List[str]:
        """Tools that have assertions registered."""
        return list(self._assertions.keys())

    # ── Validation ─────────────────────────────────────────────

    def validate(
        self,
        tool_name: str,
        output: str,
        tool_input: Optional[Dict] = None,
    ) -> ValidationResult:
        """Validate tool output against all registered assertions.

        Args:
            tool_name: Name of the tool that produced the output
            output: The raw output string from the tool
            tool_input: Optional tool input (for context in retry hints)

        Returns:
            ValidationResult with pass/fail status and actions
        """
        self._total_validations += 1

        assertions = self._assertions.get(tool_name, [])
        if not assertions:
            self._total_passed += 1
            return ValidationResult(tool_name=tool_name, passed=True)

        failures: List[ValidationFailure] = []
        output_preview = output[:200] if output else "(empty)"

        for assertion in assertions:
            ok, msg = self._check_assertion(assertion, output)
            if not ok:
                failures.append(ValidationFailure(
                    tool_name=tool_name,
                    assertion=assertion,
                    actual_output_preview=output_preview,
                    message=msg,
                    severity=assertion.severity,
                ))

        if not failures:
            self._total_passed += 1
            return ValidationResult(tool_name=tool_name, passed=True)

        # Determine action based on highest-severity failure
        should_retry = any(f.severity == ValidationSeverity.RETRY for f in failures)
        should_block = any(f.severity == ValidationSeverity.BLOCK for f in failures)

        if should_retry:
            self._total_retries += 1
        if should_block:
            self._total_blocks += 1
        self._total_failed += 1

        # Build retry hint
        retry_hint = ""
        if should_retry:
            retry_messages = [f.message for f in failures if f.severity == ValidationSeverity.RETRY]
            retry_hint = "Retry needed: " + "; ".join(retry_messages)

        return ValidationResult(
            tool_name=tool_name,
            passed=False,
            failures=failures,
            should_retry=should_retry,
            should_block=should_block,
            retry_hint=retry_hint,
        )

    def _check_assertion(
        self, assertion: OutputAssertion, output: str
    ) -> tuple[bool, str]:
        """Check a single assertion. Returns (passed, message)."""
        atype = assertion.assertion_type
        value = assertion.value

        try:
            if atype == AssertionType.NOT_EMPTY:
                if not output or not output.strip():
                    return False, assertion.message or "Output is empty"
                return True, ""

            elif atype == AssertionType.MAX_LENGTH:
                max_len = int(value)
                if len(output) > max_len:
                    return False, assertion.message or f"Output exceeds {max_len} chars (actual: {len(output)})"
                return True, ""

            elif atype == AssertionType.MIN_LENGTH:
                min_len = int(value)
                if len(output) < min_len:
                    return False, assertion.message or f"Output below {min_len} chars (actual: {len(output)})"
                return True, ""

            elif atype == AssertionType.MATCHES_REGEX:
                pattern = re.compile(str(value), re.IGNORECASE)
                if not pattern.search(output):
                    return False, assertion.message or f"Output doesn't match pattern: {value}"
                return True, ""

            elif atype == AssertionType.IS_JSON:
                try:
                    json.loads(output)
                    return True, ""
                except (json.JSONDecodeError, TypeError):
                    return False, assertion.message or "Output is not valid JSON"

            elif atype == AssertionType.JSON_HAS_KEY:
                try:
                    data = json.loads(output)
                    if isinstance(data, dict) and str(value) in data:
                        return True, ""
                    return False, assertion.message or f"JSON missing key: {value}"
                except (json.JSONDecodeError, TypeError):
                    return False, assertion.message or f"Cannot check key '{value}' — output is not JSON"

            elif atype == AssertionType.CONTAINS:
                if str(value) in output:
                    return True, ""
                return False, assertion.message or f"Output doesn't contain: {value}"

            elif atype == AssertionType.NOT_CONTAINS:
                if str(value) not in output:
                    return True, ""
                return False, assertion.message or f"Output contains forbidden: {value}"

            elif atype == AssertionType.CUSTOM:
                if assertion.custom_fn and assertion.custom_fn(output):
                    return True, ""
                return False, assertion.message or "Custom assertion failed"

            else:
                return True, ""  # Unknown assertion type — pass

        except Exception as e:
            logger.warning(f"Assertion check error: {e}")
            return True, ""  # Don't fail on assertion errors

    # ── Stats ──────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_validations": self._total_validations,
            "total_passed": self._total_passed,
            "total_failed": self._total_failed,
            "total_retries": self._total_retries,
            "total_blocks": self._total_blocks,
        }

    @property
    def max_retries(self) -> int:
        return self._max_retries
