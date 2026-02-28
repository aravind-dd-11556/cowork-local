"""
Observability Event Bus — central pub-sub for observability events.

External tools subscribe to typed events (tool calls, provider requests,
health checks, etc.) and receive async callbacks when events fire.

Sprint 16 (Testing & Observability Hardening) Module 1.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


# ── Event types ──────────────────────────────────────────────────

class EventType(Enum):
    """Canonical observability event types."""
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    TOOL_CALL_INITIATED = "tool_call_initiated"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"
    PROVIDER_REQUEST_SENT = "provider_request_sent"
    PROVIDER_RESPONSE_RECEIVED = "provider_response_received"
    PROVIDER_ERROR = "provider_error"
    HEALTH_CHECK_RUN = "health_check_run"
    HEALTH_CHECK_FAILED = "health_check_failed"
    TOKEN_BUDGET_WARNING = "token_budget_warning"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    COST_THRESHOLD_REACHED = "cost_threshold_reached"
    METRIC_SNAPSHOT = "metric_snapshot"
    CONTEXT_PRUNED = "context_pruned"
    CUSTOM = "custom"


# ── Event dataclass ──────────────────────────────────────────────

@dataclass
class ObservabilityEvent:
    """A single observability event emitted through the bus."""
    event_type: EventType
    component: str = ""
    trace_id: str = ""
    agent_name: str = ""
    severity: str = "info"  # info, warning, error
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "component": self.component,
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "severity": self.severity,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ── Subscription ─────────────────────────────────────────────────

@dataclass
class _Subscription:
    """Internal subscription record."""
    subscription_id: str
    event_type: EventType
    callback: Callable[[ObservabilityEvent], Any]
    created_at: float = field(default_factory=time.time)


# ── ObservabilityEventBus ────────────────────────────────────────

class ObservabilityEventBus:
    """
    Central pub-sub hub for observability events.

    Usage::

        bus = ObservabilityEventBus()

        def on_tool_call(event):
            print(f"Tool {event.component} called at {event.timestamp}")

        sub_id = bus.subscribe(EventType.TOOL_CALL_INITIATED, on_tool_call)
        bus.emit(ObservabilityEvent(event_type=EventType.TOOL_CALL_INITIATED, component="bash"))
    """

    def __init__(
        self,
        max_subscribers_per_event: int = 100,
        async_emit: bool = False,
        event_buffer_size: int = 1000,
    ):
        self._subscriptions: Dict[EventType, List[_Subscription]] = {}
        self._all_subscriptions: Dict[str, _Subscription] = {}
        self._max_per_event = max_subscribers_per_event
        self._async_emit = async_emit
        self._event_buffer_size = event_buffer_size
        self._event_history: List[ObservabilityEvent] = []
        self._emit_count: int = 0
        self._error_count: int = 0

    # ── Subscribe / Unsubscribe ───────────────────────────────

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[ObservabilityEvent], Any],
    ) -> str:
        """
        Register a callback for a specific event type.

        Returns a subscription_id for later unsubscription.
        """
        subs = self._subscriptions.setdefault(event_type, [])
        if len(subs) >= self._max_per_event:
            raise ValueError(
                f"Max subscribers ({self._max_per_event}) reached for {event_type.value}"
            )

        sub_id = uuid.uuid4().hex[:12]
        sub = _Subscription(
            subscription_id=sub_id,
            event_type=event_type,
            callback=callback,
        )
        subs.append(sub)
        self._all_subscriptions[sub_id] = sub
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription by ID.  Returns True if found and removed."""
        sub = self._all_subscriptions.pop(subscription_id, None)
        if sub is None:
            return False
        subs = self._subscriptions.get(sub.event_type, [])
        self._subscriptions[sub.event_type] = [
            s for s in subs if s.subscription_id != subscription_id
        ]
        return True

    def subscribe_all(
        self, callback: Callable[[ObservabilityEvent], Any],
    ) -> List[str]:
        """Subscribe to *every* event type.  Returns list of subscription IDs."""
        ids: List[str] = []
        for et in EventType:
            ids.append(self.subscribe(et, callback))
        return ids

    # ── Emit ──────────────────────────────────────────────────

    def emit(self, event: ObservabilityEvent) -> int:
        """
        Emit an event to all subscribers of its type.

        Returns the number of callbacks invoked.
        """
        self._emit_count += 1

        # Buffer event history
        self._event_history.append(event)
        if len(self._event_history) > self._event_buffer_size:
            self._event_history = self._event_history[-self._event_buffer_size:]

        subs = self._subscriptions.get(event.event_type, [])
        invoked = 0
        for sub in subs:
            try:
                result = sub.callback(event)
                # If it's a coroutine and we're not in async mode, log a warning
                if asyncio.iscoroutine(result):
                    result.close()  # prevent RuntimeWarning
                    logger.debug(
                        "Async callback provided to sync emit; use emit_async() instead"
                    )
                invoked += 1
            except Exception as exc:
                self._error_count += 1
                logger.warning(
                    "Event callback error for %s: %s", event.event_type.value, exc
                )
        return invoked

    async def emit_async(self, event: ObservabilityEvent) -> int:
        """
        Emit an event, awaiting any async callbacks.

        Returns the number of callbacks invoked.
        """
        self._emit_count += 1

        self._event_history.append(event)
        if len(self._event_history) > self._event_buffer_size:
            self._event_history = self._event_history[-self._event_buffer_size:]

        subs = self._subscriptions.get(event.event_type, [])
        invoked = 0
        for sub in subs:
            try:
                result = sub.callback(event)
                if asyncio.iscoroutine(result):
                    await result
                invoked += 1
            except Exception as exc:
                self._error_count += 1
                logger.warning(
                    "Async event callback error for %s: %s",
                    event.event_type.value, exc,
                )
        return invoked

    # ── Querying ──────────────────────────────────────────────

    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 50,
    ) -> List[ObservabilityEvent]:
        """Return recent events, optionally filtered by type."""
        events = self._event_history
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def subscriber_count(self, event_type: Optional[EventType] = None) -> int:
        """Count subscribers for a specific type, or total across all types."""
        if event_type is not None:
            return len(self._subscriptions.get(event_type, []))
        return sum(len(subs) for subs in self._subscriptions.values())

    def stats(self) -> Dict[str, Any]:
        """Return statistics about the event bus."""
        return {
            "total_subscribers": self.subscriber_count(),
            "subscribers_by_type": {
                et.value: len(subs) for et, subs in self._subscriptions.items() if subs
            },
            "total_events_emitted": self._emit_count,
            "event_buffer_size": len(self._event_history),
            "error_count": self._error_count,
        }

    def clear_history(self) -> None:
        """Clear the event history buffer."""
        self._event_history.clear()

    def reset(self) -> None:
        """Remove all subscriptions and clear history."""
        self._subscriptions.clear()
        self._all_subscriptions.clear()
        self._event_history.clear()
        self._emit_count = 0
        self._error_count = 0
