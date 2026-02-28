"""
Persistent Audit Log — write-through wrapper for SecurityAuditLog.

Extends SecurityAuditLog to persist all logged events to SQLite while
maintaining full in-memory functionality.

Sprint 19 (Persistent Storage) Module 3.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .security_audit_log import (
    SecurityAuditLog,
    SecurityEvent,
    SecurityEventType,
    SecuritySeverity,
)
from .persistent_store import PersistentStore

logger = logging.getLogger(__name__)


class PersistentAuditLog(SecurityAuditLog):
    """
    SecurityAuditLog with SQLite write-through persistence.

    All log() calls write to both in-memory storage and SQLite.
    Adds query_db() for time-range DB queries and cleanup_old() for retention.

    Usage::

        store = PersistentStore("/path/to/db")
        audit = PersistentAuditLog(store=store)
        audit.log(
            SecurityEventType.INPUT_INJECTION,
            SecuritySeverity.HIGH,
            "sanitizer",
            "SQL injection blocked",
            blocked=True,
        )
        db_events = audit.query_db(severity="high", limit=50)
    """

    def __init__(
        self,
        store: PersistentStore,
        persist_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._store = store
        self._persist_enabled = persist_enabled

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
        """Log a security event to memory and DB."""
        event = super().log(
            event_type=event_type,
            severity=severity,
            component=component,
            description=description,
            metadata=metadata,
            correlation_id=correlation_id,
            tool_name=tool_name,
            blocked=blocked,
        )

        if self._persist_enabled:
            try:
                self._store.audit.insert_event(
                    event_type=event_type.value,
                    severity=severity.value,
                    component=component,
                    description=description,
                    tool_name=tool_name,
                    blocked=blocked,
                    correlation_id=correlation_id,
                    metadata=metadata,
                    timestamp=event.timestamp,
                )
            except Exception as exc:
                logger.debug("Failed to persist audit event: %s", exc)

        return event

    # ── DB queries ──────────────────────────────────────────────

    def query_db(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        component: Optional[str] = None,
        since: Optional[float] = None,
        blocked_only: bool = False,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query audit events from the database."""
        return self._store.audit.query_events(
            event_type=event_type,
            severity=severity,
            component=component,
            since=since,
            blocked_only=blocked_only,
            limit=limit,
        )

    def count_by_severity_db(
        self, since: Optional[float] = None,
    ) -> Dict[str, int]:
        """Count events by severity from DB."""
        return self._store.audit.count_by_severity(since=since)

    def query_historical(
        self,
        days_back: int = 7,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query historical audit events from DB."""
        since = time.time() - (days_back * 86400)
        return self._store.audit.query_events(
            severity=severity, since=since
        )

    def cleanup_old(self, retention_days: int = 90) -> int:
        """Delete audit events older than retention_days."""
        cutoff = time.time() - (retention_days * 86400)
        return self._store.audit.delete_before(cutoff)

    def export_with_history(
        self,
        format: str = "json",
        days_back: int = 7,
    ) -> str:
        """Export audit log including historical DB data."""
        import json as json_mod

        current = self.export(format)
        if format != "json":
            return current

        try:
            current_data = json_mod.loads(current)
        except Exception:
            current_data = {}

        severity_counts = self.count_by_severity_db(
            since=time.time() - (days_back * 86400)
        )

        combined = {
            "current": current_data,
            "historical": {
                "days_back": days_back,
                "severity_counts": severity_counts,
            },
        }
        return json_mod.dumps(combined, indent=2, default=str)
