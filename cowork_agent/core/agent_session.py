"""
Auto-Session Integration — wire SessionManager into Agent for auto-save.

Provides ``AgentSessionManager`` which wraps the existing SessionManager
and couples it to the Agent's run() lifecycle so that every message is
automatically persisted to disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .models import Message
from .session_manager import SessionManager, SessionMetadata

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Configuration for auto-session behaviour."""
    enabled: bool = True
    auto_create: bool = True
    session_id: Optional[str] = None   # Explicit session to resume
    provider: str = ""
    model: str = ""
    title: str = ""

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "auto_create": self.auto_create,
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class AgentSessionManager:
    """
    Couples SessionManager with Agent for auto-save conversations.

    Usage::

        session_mgr = SessionManager(workspace_dir="/tmp")
        agent_session = AgentSessionManager(session_mgr, SessionConfig(provider="ollama"))

        # On first call, creates or resumes a session
        agent_session.initialize()
        # After each message:
        agent_session.save_message(user_msg)
    """

    def __init__(self, session_manager: SessionManager, config: Optional[SessionConfig] = None):
        self._session_manager = session_manager
        self._config = config or SessionConfig()
        self._session_id: Optional[str] = None
        self._auto_save_enabled = self._config.enabled

    # ── Properties ─────────────────────────────────────────────

    @property
    def session_id(self) -> Optional[str]:
        """Current session ID (or None if disabled / not yet initialised)."""
        return self._session_id

    @property
    def is_enabled(self) -> bool:
        return self._auto_save_enabled

    @property
    def session_manager(self) -> SessionManager:
        """Access the underlying SessionManager."""
        return self._session_manager

    # ── Lifecycle ──────────────────────────────────────────────

    def initialize(self, resume_from: Optional[str] = None) -> Optional[str]:
        """
        Initialize session on Agent startup.

        If *resume_from* is given, loads that session.
        Otherwise creates a new one (if ``auto_create=True``).

        Returns the session_id or None if disabled.
        """
        if not self._auto_save_enabled:
            return None

        target = resume_from or self._config.session_id

        if target:
            meta = self._session_manager.get_metadata(target)
            if meta:
                self._session_id = target
                logger.info("Resumed session %s", target)
                return target
            logger.warning("Session %s not found; creating new.", target)

        if self._config.auto_create:
            sid = self._session_manager.create_session(
                provider=self._config.provider,
                model=self._config.model,
                title=self._config.title or "Untitled session",
            )
            self._session_id = sid
            logger.info("Created session %s", sid)
            return sid

        return None

    # ── Message persistence ────────────────────────────────────

    def save_message(self, message: Message) -> None:
        """Auto-save a message to the current session."""
        if self._auto_save_enabled and self._session_id:
            self._session_manager.save_message(self._session_id, message)

    def get_session_messages(self) -> list:
        """Load all messages from the current session."""
        if not self._session_id:
            return []
        return self._session_manager.load_messages(self._session_id)

    # ── Metadata helpers ───────────────────────────────────────

    def update_title(self, title: str) -> None:
        """Update the session title."""
        if self._session_id:
            self._session_manager.update_title(self._session_id, title)

    def get_metadata(self) -> Optional[SessionMetadata]:
        """Get metadata for the current session."""
        if not self._session_id:
            return None
        return self._session_manager.get_metadata(self._session_id)

    def list_recent(self, limit: int = 20) -> list:
        """List recent sessions (newest first)."""
        return self._session_manager.list_sessions(limit=limit)
