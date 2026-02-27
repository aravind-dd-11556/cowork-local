"""
Context Bus — Shared message bus for multi-agent communication.

Provides pub/sub messaging and a shared key-value store for agents
to exchange data, results, and status updates.

Sprint 5 (P3-Multi-Agent Orchestration) Feature 2.
"""

from __future__ import annotations
import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages on the context bus."""
    TASK_RESULT = "task_result"
    STATUS_UPDATE = "status_update"
    DATA_SHARE = "data_share"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class BusMessage:
    """A message published to the context bus."""
    msg_type: MessageType
    sender: str                    # Agent name
    content: Any = None
    topic: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.msg_type.value,
            "sender": self.sender,
            "content": self.content,
            "topic": self.topic,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ContextBus:
    """
    Shared message bus for agent communication.

    Provides:
      - Pub/sub messaging: agents publish results, others subscribe
      - Shared state store: key-value pairs accessible by all agents
      - Message history: retrievable log for debugging

    Usage:
        bus = ContextBus()

        # Subscribe to task results from agent_a
        bus.subscribe("agent_a", MessageType.TASK_RESULT, callback)

        # Publish a result
        await bus.publish(BusMessage(
            msg_type=MessageType.TASK_RESULT,
            sender="agent_a",
            content="Found 5 files",
        ))

        # Shared state
        await bus.set_shared("search_path", "/data")
        path = await bus.get_shared("search_path")
    """

    def __init__(self, max_history: int = 1000):
        # Subscribers: {sender: {msg_type: [callbacks]}}
        self._subscribers: dict[str, dict[MessageType, list[Callable]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        # Wildcard subscribers: {msg_type: [callbacks]}  (receive from ANY sender)
        self._wildcard_subscribers: dict[MessageType, list[Callable]] = defaultdict(list)

        self._shared_state: dict[str, Any] = {}
        self._message_history: list[BusMessage] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()

    # ── Pub/Sub ──

    async def publish(self, message: BusMessage) -> None:
        """Publish a message to the bus. Invokes matching subscribers."""
        async with self._lock:
            self._message_history.append(message)
            if len(self._message_history) > self._max_history:
                self._message_history = self._message_history[-self._max_history:]

        # Collect matching callbacks
        callbacks = list(self._subscribers.get(message.sender, {}).get(message.msg_type, []))
        callbacks += list(self._wildcard_subscribers.get(message.msg_type, []))

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")

    def subscribe(
        self,
        sender: str,
        msg_type: MessageType,
        callback: Callable[[BusMessage], Any],
    ) -> None:
        """Subscribe to messages from a specific sender and type."""
        self._subscribers[sender][msg_type].append(callback)

    def subscribe_all(
        self,
        msg_type: MessageType,
        callback: Callable[[BusMessage], Any],
    ) -> None:
        """Subscribe to a message type from ANY sender."""
        self._wildcard_subscribers[msg_type].append(callback)

    def unsubscribe(
        self,
        sender: str,
        msg_type: MessageType,
        callback: Callable,
    ) -> None:
        """Remove a subscription."""
        cbs = self._subscribers.get(sender, {}).get(msg_type, [])
        if callback in cbs:
            cbs.remove(callback)

    # ── Shared State ──

    async def set_shared(self, key: str, value: Any) -> None:
        """Set a shared state value (thread-safe)."""
        async with self._lock:
            self._shared_state[key] = value

    async def get_shared(self, key: str, default: Any = None) -> Any:
        """Get a shared state value."""
        async with self._lock:
            return self._shared_state.get(key, default)

    async def update_shared(self, key: str, updater: Callable[[Any], Any]) -> Any:
        """Atomically update a shared value using an updater function."""
        async with self._lock:
            old_value = self._shared_state.get(key)
            new_value = updater(old_value)
            self._shared_state[key] = new_value
            return new_value

    async def delete_shared(self, key: str) -> bool:
        """Delete a shared value. Returns True if key existed."""
        async with self._lock:
            if key in self._shared_state:
                del self._shared_state[key]
                return True
            return False

    def get_shared_snapshot(self) -> dict[str, Any]:
        """Get a snapshot of all shared state (not locked — advisory)."""
        return dict(self._shared_state)

    # ── History ──

    def get_history(
        self,
        sender: Optional[str] = None,
        msg_type: Optional[MessageType] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> list[BusMessage]:
        """Get message history, optionally filtered."""
        result = list(self._message_history)

        if sender:
            result = [m for m in result if m.sender == sender]
        if msg_type:
            result = [m for m in result if m.msg_type == msg_type]
        if topic:
            result = [m for m in result if m.topic == topic]

        return result[-limit:]

    @property
    def history_size(self) -> int:
        return len(self._message_history)

    def clear_history(self) -> None:
        """Clear all message history."""
        self._message_history.clear()

    def clear_all(self) -> None:
        """Clear history and shared state."""
        self._message_history.clear()
        self._shared_state.clear()
