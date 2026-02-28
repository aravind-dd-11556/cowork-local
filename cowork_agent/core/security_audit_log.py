"""
Security Audit Log — structured security event logging and export.

Records security-relevant events (injection attempts, credential detections,
rate limit violations, sandbox violations) with severity levels and
query/export capabilities.

Sprint 17 (Security & Sandboxing) Module 6.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────

class SecurityEventType(Enum):
    """Types of security events."""
    INPUT_INJECTION = "input_injection"
    PROMPT_INJECTION = "prompt_injection"
    CREDENTIAL_DETECTED = "credential_detected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SANDBOX_VIOLATION = "sandbox_violation"
    EXECUTION_TIMEOUT = "execution_timeout"
    PATH_TRAVERSAL = "path_traversal"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    POLICY_VIOLATION = "policy_violation"
    CUSTOM = "custom"


class SecuritySeverity(Enum):
    """Severity levels for security events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class SecurityEvent:
    """A single security audit event."""
    event_type: SecurityEventType
    severity: SecuritySeverity
    component: str  # Which component generated the event
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = ""
    tool_name: str = ""
    blocked: bool = False  # Whether the action was blocked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "component": self.component,
            "description": self.description,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "tool_name": self.tool_name,
            "blocked": self.blocked,
        }


@dataclass
class AuditSummary:
    """Summary of security audit log."""
    total_events: int
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    blocked_count: int
    recent_critical: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "events_by_type": self.events_by_type,
            "events_by_severity": self.events_by_severity,
            "blocked_count": self.blocked_count,
            "recent_critical": self.recent_critical,
        }


# ── SecurityAuditLog ─────────────────────────────────────────────

class SecurityAuditLog:
    """
    Security event logging with query and export capabilities.

    Usage::

        audit = SecurityAuditLog()
        audit.log(
            SecurityEventType.INPUT_INJECTION,
            SecuritySeverity.HIGH,
            "input_sanitizer",
            "SQL injection detected in file_path field",
            tool_name="read",
            blocked=True,
        )
        events = audit.query(severity=SecuritySeverity.HIGH)
        report = audit.export("json")
    """

    def __init__(
        self,
        max_events: int = 10000,
        on_critical: Optional[Callable[[SecurityEvent], None]] = None,
    ):
        self._events: List[SecurityEvent] = []
        self._max_events = max_events
        self._on_critical = on_critical
        self._listeners: List[Callable[[SecurityEvent], None]] = []

    def log(
        self,
        event_type: SecurityEventType,
        severity: SecuritySeverity,
        component: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: str = "",
        tool_name: str = "",
        blocked: bool = False,
    ) -> SecurityEvent:
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            component=component,
            description=description,
            metadata=metadata or {},
            correlation_id=correlation_id,
            tool_name=tool_name,
            blocked=blocked,
        )

        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        # Log to standard logger
        log_msg = (
            f"[SECURITY] [{severity.value.upper()}] {event_type.value}: "
            f"{description} (component={component}, tool={tool_name}, blocked={blocked})"
        )
        if severity in (SecuritySeverity.CRITICAL, SecuritySeverity.HIGH):
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Fire critical callback
        if severity == SecuritySeverity.CRITICAL and self._on_critical:
            try:
                self._on_critical(event)
            except Exception as exc:
                logger.debug("Critical callback failed: %s", exc)

        # Fire listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as exc:
                logger.debug("Audit listener failed: %s", exc)

        return event

    def add_listener(self, callback: Callable[[SecurityEvent], None]) -> None:
        """Add a listener for all security events."""
        self._listeners.append(callback)

    # ── Query ──────────────────────────────────────────────────

    def query(
        self,
        event_type: Optional[SecurityEventType] = None,
        severity: Optional[SecuritySeverity] = None,
        component: Optional[str] = None,
        tool_name: Optional[str] = None,
        since: Optional[float] = None,
        blocked_only: bool = False,
        limit: int = 100,
    ) -> List[SecurityEvent]:
        """Query security events with filters."""
        results = self._events

        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        if severity is not None:
            results = [e for e in results if e.severity == severity]
        if component is not None:
            results = [e for e in results if e.component == component]
        if tool_name is not None:
            results = [e for e in results if e.tool_name == tool_name]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]
        if blocked_only:
            results = [e for e in results if e.blocked]

        return results[-limit:]

    def get_recent(self, count: int = 20) -> List[SecurityEvent]:
        """Get the most recent security events."""
        return self._events[-count:]

    # ── Summary ────────────────────────────────────────────────

    def summary(self) -> AuditSummary:
        """Generate an audit summary."""
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        blocked = 0

        for event in self._events:
            by_type[event.event_type.value] = by_type.get(event.event_type.value, 0) + 1
            by_severity[event.severity.value] = by_severity.get(event.severity.value, 0) + 1
            if event.blocked:
                blocked += 1

        # Recent critical events
        critical = [
            e.to_dict() for e in self._events
            if e.severity == SecuritySeverity.CRITICAL
        ][-5:]

        return AuditSummary(
            total_events=len(self._events),
            events_by_type=by_type,
            events_by_severity=by_severity,
            blocked_count=blocked,
            recent_critical=critical,
        )

    # ── Export ─────────────────────────────────────────────────

    def export(self, format: str = "json") -> str:
        """Export audit log. Formats: json, csv."""
        if format == "csv":
            return self._export_csv()
        return self._export_json()

    def _export_json(self) -> str:
        summary = self.summary()
        data = {
            "summary": summary.to_dict(),
            "events": [e.to_dict() for e in self._events],
        }
        return json.dumps(data, indent=2, default=str)

    def _export_csv(self) -> str:
        lines = ["timestamp,event_type,severity,component,tool_name,blocked,description"]
        for event in self._events:
            desc = event.description.replace('"', '""')
            lines.append(
                f'{event.timestamp},{event.event_type.value},{event.severity.value},'
                f'{event.component},{event.tool_name},{event.blocked},"{desc}"'
            )
        return "\n".join(lines)

    # ── Convenience logging methods ────────────────────────────

    def log_injection(
        self,
        injection_type: str,
        component: str,
        description: str,
        tool_name: str = "",
        blocked: bool = True,
        **metadata: Any,
    ) -> SecurityEvent:
        """Shortcut for logging injection events."""
        if injection_type == "prompt":
            event_type = SecurityEventType.PROMPT_INJECTION
            severity = SecuritySeverity.HIGH
        elif injection_type == "input":
            event_type = SecurityEventType.INPUT_INJECTION
            severity = SecuritySeverity.HIGH
        else:
            event_type = SecurityEventType.POLICY_VIOLATION
            severity = SecuritySeverity.MEDIUM

        return self.log(
            event_type=event_type,
            severity=severity,
            component=component,
            description=description,
            metadata=metadata,
            tool_name=tool_name,
            blocked=blocked,
        )

    def log_credential(
        self,
        credential_type: str,
        component: str = "credential_detector",
        tool_name: str = "",
        blocked: bool = True,
        **metadata: Any,
    ) -> SecurityEvent:
        """Shortcut for logging credential detection events."""
        return self.log(
            event_type=SecurityEventType.CREDENTIAL_DETECTED,
            severity=SecuritySeverity.HIGH,
            component=component,
            description=f"Credential detected: {credential_type}",
            metadata={"credential_type": credential_type, **metadata},
            tool_name=tool_name,
            blocked=blocked,
        )

    def log_rate_limit(
        self,
        resource_name: str,
        component: str = "rate_limiter",
        **metadata: Any,
    ) -> SecurityEvent:
        """Shortcut for logging rate limit events."""
        return self.log(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            severity=SecuritySeverity.MEDIUM,
            component=component,
            description=f"Rate limit exceeded for '{resource_name}'",
            metadata={"resource": resource_name, **metadata},
        )

    def log_sandbox_violation(
        self,
        tool_name: str,
        violation: str,
        component: str = "sandboxed_executor",
        **metadata: Any,
    ) -> SecurityEvent:
        """Shortcut for logging sandbox violations."""
        return self.log(
            event_type=SecurityEventType.SANDBOX_VIOLATION,
            severity=SecuritySeverity.HIGH,
            component=component,
            description=violation,
            metadata=metadata,
            tool_name=tool_name,
            blocked=True,
        )

    # ── Reset ──────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()

    @property
    def event_count(self) -> int:
        """Total number of logged events."""
        return len(self._events)
