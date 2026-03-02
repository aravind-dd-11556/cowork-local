"""
Sprint 40 · Failure Diagnosis
==============================
Analyzes tool errors and provides structured diagnosis with
recovery suggestions (alternative tools, param fixes, rollback hints).

Signals used:
  1. ErrorCatalog classification
  2. Tool metadata / alternatives map
  3. Param-fix rule table
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .error_catalog import AgentError, ErrorCategory, ErrorCatalog, ErrorCode


# ── Diagnosis data ──────────────────────────────────────────────────

@dataclass
class FailureDiagnosis:
    """Complete diagnosis of a single tool failure."""
    error_code: ErrorCode
    error_category: ErrorCategory
    root_cause: str
    severity: str  # "low" | "medium" | "high" | "critical"

    # Recovery suggestions
    suggested_param_changes: Dict[str, Any] = field(default_factory=dict)
    suggested_alternative_tools: List[str] = field(default_factory=list)
    rollback_recommended: bool = False
    degradation_recommended: bool = False

    # Confidence
    confidence: float = 0.5  # 0.0–1.0
    is_transient: bool = False

    def to_dict(self) -> dict:
        return {
            "error_code": self.error_code.value,
            "error_category": self.error_category.value,
            "root_cause": self.root_cause,
            "severity": self.severity,
            "suggested_param_changes": self.suggested_param_changes,
            "suggested_alternative_tools": self.suggested_alternative_tools,
            "rollback_recommended": self.rollback_recommended,
            "degradation_recommended": self.degradation_recommended,
            "confidence": self.confidence,
            "is_transient": self.is_transient,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FailureDiagnosis":
        return cls(
            error_code=ErrorCode(d["error_code"]),
            error_category=ErrorCategory(d["error_category"]),
            root_cause=d.get("root_cause", ""),
            severity=d.get("severity", "medium"),
            suggested_param_changes=d.get("suggested_param_changes", {}),
            suggested_alternative_tools=d.get("suggested_alternative_tools", []),
            rollback_recommended=d.get("rollback_recommended", False),
            degradation_recommended=d.get("degradation_recommended", False),
            confidence=d.get("confidence", 0.5),
            is_transient=d.get("is_transient", False),
        )


# ── Static maps ─────────────────────────────────────────────────────

TOOL_ALTERNATIVES: Dict[str, List[str]] = {
    "write": ["edit"],
    "edit": ["write"],
    "bash": ["write"],
    "web_fetch": ["web_search"],
    "web_search": ["web_fetch"],
    "glob": ["grep", "bash"],
    "grep": ["glob", "bash"],
    "read": ["bash"],
}

# error_code → suggested parameter modifications
PARAM_FIX_RULES: Dict[ErrorCode, Dict[str, Any]] = {
    ErrorCode.TOOL_TIMEOUT: {"timeout": 120},
    ErrorCode.PROVIDER_TIMEOUT: {"timeout": 120},
    ErrorCode.NETWORK_TIMEOUT: {"timeout": 120},
    ErrorCode.PROVIDER_RATE_LIMITED: {"_delay_seconds": 2.0},
    ErrorCode.PROVIDER_OVERLOADED: {"_delay_seconds": 3.0},
}

# error_code → severity
_SEVERITY_MAP: Dict[ErrorCode, str] = {
    ErrorCode.PROVIDER_AUTH_FAILED: "critical",
    ErrorCode.SECURITY_BLOCKED: "critical",
    ErrorCode.SECURITY_INJECTION: "critical",
    ErrorCode.PROVIDER_CONNECTION_FAILED: "high",
    ErrorCode.NETWORK_CONNECTION_REFUSED: "high",
    ErrorCode.NETWORK_DNS_FAILED: "high",
    ErrorCode.NETWORK_SSL_ERROR: "high",
    ErrorCode.TOOL_EXECUTION_FAILED: "medium",
    ErrorCode.TOOL_VALIDATION_FAILED: "medium",
    ErrorCode.TOOL_PERMISSION_DENIED: "high",
    ErrorCode.TOOL_NOT_FOUND: "medium",
    ErrorCode.TOOL_TIMEOUT: "medium",
    ErrorCode.PROVIDER_TIMEOUT: "medium",
    ErrorCode.PROVIDER_RATE_LIMITED: "low",
    ErrorCode.PROVIDER_OVERLOADED: "low",
    ErrorCode.NETWORK_TIMEOUT: "medium",
}

# Transient error codes (safe to retry)
_TRANSIENT_CODES = {
    ErrorCode.PROVIDER_RATE_LIMITED,
    ErrorCode.PROVIDER_TIMEOUT,
    ErrorCode.PROVIDER_OVERLOADED,
    ErrorCode.TOOL_TIMEOUT,
    ErrorCode.NETWORK_TIMEOUT,
    ErrorCode.NETWORK_CONNECTION_REFUSED,
}


# ── Diagnostic engine ───────────────────────────────────────────────

class FailureDiagnostic:
    """
    Analyzes tool failures using error classification, tool metadata,
    and rule-based recovery suggestions.
    """

    def __init__(
        self,
        tool_registry: Any = None,  # ToolRegistry (optional)
    ):
        self.tool_registry = tool_registry
        self._cache: Dict[str, FailureDiagnosis] = {}

    # ── public API ──────────────────────────────────────────────

    def diagnose(
        self,
        tool_name: str,
        error_msg: str,
        tool_input: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> FailureDiagnosis:
        """
        Produce a *FailureDiagnosis* for the given failure.

        Uses:
          - ErrorCatalog.classify_error_from_string() for code/category
          - TOOL_ALTERNATIVES for alternative tool suggestions
          - PARAM_FIX_RULES for parameter modification hints
        """
        # cache lookup
        cache_key = self._cache_key(tool_name, error_msg)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. classify
        classified = self._classify(error_msg)
        error_code = classified.code
        error_category = classified.category

        # 2. severity
        severity = _SEVERITY_MAP.get(error_code, "medium")

        # 3. transient?
        is_transient = error_code in _TRANSIENT_CODES

        # 4. param fixes
        param_changes = dict(PARAM_FIX_RULES.get(error_code, {}))

        # 5. alternative tools
        alt_tools = list(TOOL_ALTERNATIVES.get(tool_name, []))

        # 6. rollback / degradation flags
        rollback_recommended = severity in ("high", "critical")
        degradation_recommended = is_transient or error_code in {
            ErrorCode.PROVIDER_OVERLOADED,
            ErrorCode.PROVIDER_RATE_LIMITED,
        }

        # 7. confidence
        confidence = 0.8 if is_transient else (0.6 if severity in ("low", "medium") else 0.4)

        diagnosis = FailureDiagnosis(
            error_code=error_code,
            error_category=error_category,
            root_cause=classified.message,
            severity=severity,
            suggested_param_changes=param_changes,
            suggested_alternative_tools=alt_tools,
            rollback_recommended=rollback_recommended,
            degradation_recommended=degradation_recommended,
            confidence=confidence,
            is_transient=is_transient,
        )
        self._cache[cache_key] = diagnosis
        return diagnosis

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _classify(error_msg: str) -> AgentError:
        """Classify an error string into an AgentError."""
        return ErrorCatalog.classify_error(Exception(error_msg))

    @staticmethod
    def _cache_key(tool_name: str, error_msg: str) -> str:
        raw = f"{tool_name}:{error_msg}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def clear_cache(self) -> None:
        self._cache.clear()
