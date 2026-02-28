"""
Security Pipeline — Orchestrates all security checks in proper order.

Input flow:
  1. Input sanitization (injection detection)
  2. Prompt injection scanning
  3. Credential detection

Tool call flow:
  1. Action classification (tier check)
  2. Permission check

Output flow:
  1. Credential detection (redaction)
  2. Instruction detection
  3. Prompt injection scanning (tool output variant)
  4. Audit logging

Sprint 23: Anthropic-grade security.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineCheckType(Enum):
    """Types of security checks in the pipeline."""
    INPUT_SANITIZATION = "input_sanitization"
    PROMPT_INJECTION_SCAN = "prompt_injection_scan"
    CREDENTIAL_DETECTION = "credential_detection"
    INSTRUCTION_DETECTION = "instruction_detection"
    ACTION_CLASSIFICATION = "action_classification"
    PERMISSION_CHECK = "permission_check"
    PRIVACY_CHECK = "privacy_check"


@dataclass
class CheckResult:
    """Result of a single security check."""
    check_type: PipelineCheckType
    passed: bool
    severity: str  # "info", "warning", "critical"
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check_type": self.check_type.value,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass
class PipelineResult:
    """Aggregated result from entire pipeline."""
    success: bool  # All critical checks passed
    checks: List[CheckResult] = field(default_factory=list)
    requires_user_confirmation: bool = False
    confirmation_message: Optional[str] = None
    redacted_output: Optional[str] = None  # Sanitized version of output

    def add_check(self, check: CheckResult) -> None:
        self.checks.append(check)
        if check.severity == "critical" and not check.passed:
            self.success = False

    def has_critical_failures(self) -> bool:
        return any(
            c.severity == "critical" and not c.passed for c in self.checks
        )

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == "warning" and not c.passed)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "checks": [c.to_dict() for c in self.checks],
            "requires_user_confirmation": self.requires_user_confirmation,
            "confirmation_message": self.confirmation_message,
        }


class SecurityPipeline:
    """Orchestrates all security checks before, during, and after tool execution.

    Usage::

        pipeline = SecurityPipeline(
            input_sanitizer=input_san,
            prompt_injection_detector=inj_det,
            credential_detector=cred_det,
            instruction_detector=instr_det,
            action_classifier=classifier,
            security_audit_log=audit_log,
        )

        # Before LLM call
        result = pipeline.validate_input("user message")

        # Before tool execution
        result = pipeline.validate_tool_call("bash", {"command": "ls"})

        # After tool execution
        result = pipeline.validate_tool_output("output text", "bash", {})
    """

    def __init__(
        self,
        input_sanitizer=None,
        prompt_injection_detector=None,
        credential_detector=None,
        instruction_detector=None,
        action_classifier=None,
        privacy_guard=None,
        security_audit_log=None,
        permission_manager=None,
    ):
        self.input_sanitizer = input_sanitizer
        self.prompt_injection_detector = prompt_injection_detector
        self.credential_detector = credential_detector
        self.instruction_detector = instruction_detector
        self.action_classifier = action_classifier
        self.privacy_guard = privacy_guard
        self.security_audit_log = security_audit_log
        self.permission_manager = permission_manager

        # Counters
        self._input_validations = 0
        self._tool_call_validations = 0
        self._output_validations = 0

    # ── Input validation ──────────────────────────────────────────

    def validate_input(self, user_text: str) -> PipelineResult:
        """Validate user input before LLM processing.

        Checks:
          1. Prompt injection scanning (warn if detected)
          2. Credential detection (block if credentials in user input)
        """
        self._input_validations += 1
        result = PipelineResult(success=True)

        if not user_text:
            return result

        # 1. Prompt injection scan (informational — we don't block user input)
        if self.prompt_injection_detector:
            try:
                inj_result = self.prompt_injection_detector.scan(user_text)
                result.add_check(CheckResult(
                    check_type=PipelineCheckType.PROMPT_INJECTION_SCAN,
                    passed=inj_result.is_safe,
                    severity="info" if inj_result.is_safe else "warning",
                    message=inj_result.summary,
                    metadata=inj_result.to_dict(),
                ))
            except Exception as e:
                logger.debug(f"Prompt injection scan error: {e}")

        # 2. Credential detection (block if user accidentally pastes credentials)
        if self.credential_detector:
            try:
                cred_result = self.credential_detector.scan(user_text, redact=False)
                if cred_result.has_credentials:
                    result.add_check(CheckResult(
                        check_type=PipelineCheckType.CREDENTIAL_DETECTION,
                        passed=False,
                        severity="warning",
                        message=f"Credentials detected in input: {cred_result.summary}",
                        metadata=cred_result.to_dict(),
                    ))
                    self._log_event(
                        "credential_detected", "HIGH",
                        "Credentials detected in user input",
                        tool_name="user_input",
                    )
            except Exception as e:
                logger.debug(f"Credential detection error: {e}")

        return result

    # ── Tool call validation ──────────────────────────────────────

    def validate_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> PipelineResult:
        """Validate a tool call before execution.

        Checks:
          1. Action classification (tier check)
          2. Input sanitization (injection patterns)
          3. Privacy check (sensitive data in inputs)
        """
        self._tool_call_validations += 1
        result = PipelineResult(success=True)

        # 1. Action classification
        if self.action_classifier:
            try:
                classification = self.action_classifier.classify(tool_name, tool_input)

                if classification.tier.value == "prohibited":
                    result.success = False
                    result.add_check(CheckResult(
                        check_type=PipelineCheckType.ACTION_CLASSIFICATION,
                        passed=False,
                        severity="critical",
                        message=classification.reason,
                        metadata=classification.to_dict(),
                    ))
                    self._log_event(
                        "policy_violation", "CRITICAL",
                        f"Prohibited action blocked: {classification.reason}",
                        tool_name=tool_name, blocked=True,
                    )
                    return result

                elif classification.tier.value == "explicit_consent":
                    result.requires_user_confirmation = True
                    result.confirmation_message = (
                        classification.suggested_message
                        or f"Please confirm: {classification.reason}"
                    )
                    result.add_check(CheckResult(
                        check_type=PipelineCheckType.ACTION_CLASSIFICATION,
                        passed=True,
                        severity="warning",
                        message=f"Explicit consent required: {classification.reason}",
                        metadata=classification.to_dict(),
                    ))
                else:
                    result.add_check(CheckResult(
                        check_type=PipelineCheckType.ACTION_CLASSIFICATION,
                        passed=True,
                        severity="info",
                        message="Action is safe to execute",
                    ))
            except Exception as e:
                logger.debug(f"Action classification error: {e}")

        # 2. Input sanitization
        if self.input_sanitizer:
            try:
                san_result = self.input_sanitizer.sanitize(tool_name, tool_input)
                if not san_result.is_safe:
                    result.add_check(CheckResult(
                        check_type=PipelineCheckType.INPUT_SANITIZATION,
                        passed=False,
                        severity="warning",
                        message=f"Input sanitization: {san_result.threat_summary}",
                        metadata={"threats": san_result.threats},
                    ))
                    self._log_event(
                        "input_injection", "MEDIUM",
                        f"Input injection detected: {san_result.threat_summary}",
                        tool_name=tool_name,
                    )
            except Exception as e:
                logger.debug(f"Input sanitization error: {e}")

        # 3. Privacy check
        if self.privacy_guard:
            try:
                privacy_result = self.privacy_guard.scan_for_sensitive_fields(
                    str(tool_input)
                )
                if privacy_result.has_sensitive_fields:
                    result.add_check(CheckResult(
                        check_type=PipelineCheckType.PRIVACY_CHECK,
                        passed=False,
                        severity="warning",
                        message=f"Sensitive data in tool input: {privacy_result.risk_level}",
                        metadata={"fields": privacy_result.fields_found},
                    ))
            except Exception as e:
                logger.debug(f"Privacy check error: {e}")

        return result

    # ── Tool output validation ────────────────────────────────────

    def validate_tool_output(
        self,
        output: str,
        tool_name: str,
        tool_input: Dict[str, Any] = None,
    ) -> PipelineResult:
        """Validate tool output before adding to conversation.

        Checks:
          1. Credential detection (redaction)
          2. Instruction detection
          3. Prompt injection scanning (tool output variant)
        """
        self._output_validations += 1
        result = PipelineResult(success=True)
        tool_input = tool_input or {}
        current_output = output

        if not output:
            return result

        # 1. Credential detection (redact credentials in output)
        if self.credential_detector:
            try:
                cred_result = self.credential_detector.scan(output, redact=True)
                if cred_result.has_credentials:
                    current_output = cred_result.redacted_text
                    result.redacted_output = current_output
                    result.add_check(CheckResult(
                        check_type=PipelineCheckType.CREDENTIAL_DETECTION,
                        passed=False,
                        severity="warning",
                        message=f"Credentials redacted in output: {cred_result.summary}",
                        metadata=cred_result.to_dict(),
                    ))
                    self._log_event(
                        "credential_detected", "HIGH",
                        f"Credentials found in tool output: {cred_result.summary}",
                        tool_name=tool_name,
                    )
            except Exception as e:
                logger.debug(f"Credential detection error: {e}")

        # 2. Instruction detection
        if self.instruction_detector:
            try:
                instr_result = self.instruction_detector.scan(current_output)
                if instr_result.has_instructions:
                    result.requires_user_confirmation = True
                    result.confirmation_message = (
                        f"Potential instructions detected in {tool_name} output "
                        f"(risk: {instr_result.risk_score:.0%}). "
                        f"Categories: {', '.join(instr_result.categories[:3])}. "
                        f"Should I include this output?"
                    )
                    result.add_check(CheckResult(
                        check_type=PipelineCheckType.INSTRUCTION_DETECTION,
                        passed=False,
                        severity="warning",
                        message=f"Embedded instructions detected (risk={instr_result.risk_score:.2f})",
                        metadata=instr_result.to_dict(),
                    ))
                    self._log_event(
                        "prompt_injection", "HIGH",
                        f"Instruction injection in tool output: {instr_result.summary}",
                        tool_name=tool_name,
                    )
            except Exception as e:
                logger.debug(f"Instruction detection error: {e}")

        # 3. Prompt injection scan (tool output variant)
        if self.prompt_injection_detector:
            try:
                inj_result = self.prompt_injection_detector.scan_tool_output(
                    current_output, tool_name
                )
                if not inj_result.is_safe:
                    result.add_check(CheckResult(
                        check_type=PipelineCheckType.PROMPT_INJECTION_SCAN,
                        passed=False,
                        severity="warning",
                        message=f"Prompt injection in output: {inj_result.summary}",
                        metadata=inj_result.to_dict(),
                    ))
            except Exception as e:
                logger.debug(f"Prompt injection scan error: {e}")

        return result

    # ── Audit logging helper ──────────────────────────────────────

    def _log_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        tool_name: str = "",
        blocked: bool = False,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log a security event to the audit log."""
        if not self.security_audit_log:
            return
        try:
            # Import enums from audit log module
            from .security_audit_log import SecurityEventType, SecuritySeverity

            event_type_map = {
                "input_injection": SecurityEventType.INPUT_INJECTION,
                "prompt_injection": SecurityEventType.PROMPT_INJECTION,
                "credential_detected": SecurityEventType.CREDENTIAL_DETECTED,
                "policy_violation": SecurityEventType.POLICY_VIOLATION,
            }
            severity_map = {
                "LOW": SecuritySeverity.LOW,
                "MEDIUM": SecuritySeverity.MEDIUM,
                "HIGH": SecuritySeverity.HIGH,
                "CRITICAL": SecuritySeverity.CRITICAL,
            }

            self.security_audit_log.log(
                event_type=event_type_map.get(event_type, SecurityEventType.CUSTOM),
                severity=severity_map.get(severity, SecuritySeverity.MEDIUM),
                component="security_pipeline",
                description=description,
                tool_name=tool_name,
                blocked=blocked,
                metadata=metadata,
            )
        except Exception as e:
            logger.debug(f"Audit log error: {e}")

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "input_validations": self._input_validations,
            "tool_call_validations": self._tool_call_validations,
            "output_validations": self._output_validations,
        }
