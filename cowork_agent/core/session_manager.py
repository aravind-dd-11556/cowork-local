"""
Session Manager — Save/load agent conversations to disk.

Mirrors real Cowork's session persistence:
  - Each session gets a unique ID
  - Conversations saved as JSONL (one message per line)
  - Can resume previous sessions
  - Stores metadata (provider, model, timestamps)
"""

from __future__ import annotations
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .models import Message, ToolCall, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Metadata about a saved session."""
    session_id: str
    created_at: float
    updated_at: float
    message_count: int
    provider: str = ""
    model: str = ""
    title: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SessionManager:
    """
    Manages saving and loading of agent conversation sessions.

    Storage layout:
        {workspace}/.cowork/sessions/
            {session_id}/
                metadata.json
                messages.jsonl
    """

    SESSIONS_DIR = ".cowork/sessions"

    def __init__(self, workspace_dir: str = ""):
        self.workspace_dir = workspace_dir
        self._base_dir = ""
        if workspace_dir:
            self._base_dir = os.path.join(workspace_dir, self.SESSIONS_DIR)
            os.makedirs(self._base_dir, exist_ok=True)

    def create_session(self, provider: str = "", model: str = "",
                       title: str = "") -> str:
        """Create a new session and return its ID."""
        session_id = uuid.uuid4().hex[:12]
        now = time.time()

        metadata = SessionMetadata(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            message_count=0,
            provider=provider,
            model=model,
            title=title or f"Session {session_id[:8]}",
        )

        session_dir = os.path.join(self._base_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)

        self._save_metadata(session_id, metadata)

        # Create empty messages file
        msg_path = os.path.join(session_dir, "messages.jsonl")
        Path(msg_path).touch()

        logger.info(f"Created session: {session_id}")
        return session_id

    def save_message(self, session_id: str, message: Message) -> None:
        """Append a message to the session's JSONL file."""
        if not self._base_dir:
            return

        session_dir = os.path.join(self._base_dir, session_id)
        if not os.path.isdir(session_dir):
            logger.warning(f"Session not found: {session_id}")
            return

        msg_data = self._message_to_dict(message)
        msg_path = os.path.join(session_dir, "messages.jsonl")

        try:
            with open(msg_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg_data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save message: {e}")

        # Update metadata
        meta = self._load_metadata(session_id)
        if meta:
            meta.updated_at = time.time()
            meta.message_count += 1
            self._save_metadata(session_id, meta)

    def load_messages(self, session_id: str) -> list[Message]:
        """Load all messages from a session."""
        if not self._base_dir:
            return []

        msg_path = os.path.join(self._base_dir, session_id, "messages.jsonl")
        if not os.path.exists(msg_path):
            return []

        messages = []
        try:
            with open(msg_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        msg_data = json.loads(line)
                        messages.append(self._dict_to_message(msg_data))
        except Exception as e:
            logger.warning(f"Failed to load messages: {e}")

        return messages

    def list_sessions(self, limit: int = 20) -> list[SessionMetadata]:
        """List recent sessions, newest first."""
        if not self._base_dir or not os.path.isdir(self._base_dir):
            return []

        sessions = []
        for entry in os.listdir(self._base_dir):
            session_dir = os.path.join(self._base_dir, entry)
            if os.path.isdir(session_dir):
                meta = self._load_metadata(entry)
                if meta:
                    sessions.append(meta)

        # Sort by updated_at, newest first
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[:limit]

    def delete_session(self, session_id: str) -> str:
        """Delete a session and its files."""
        if not self._base_dir:
            return "No workspace configured."

        session_dir = os.path.join(self._base_dir, session_id)
        if not os.path.isdir(session_dir):
            return f"Session '{session_id}' not found."

        import shutil
        try:
            shutil.rmtree(session_dir)
            return f"Session '{session_id}' deleted."
        except Exception as e:
            return f"Failed to delete session: {e}"

    def get_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Get metadata for a session."""
        return self._load_metadata(session_id)

    def update_title(self, session_id: str, title: str) -> None:
        """Update the session title."""
        meta = self._load_metadata(session_id)
        if meta:
            meta.title = title
            self._save_metadata(session_id, meta)

    # --- Internal helpers ---

    def _save_metadata(self, session_id: str, metadata: SessionMetadata) -> None:
        meta_path = os.path.join(self._base_dir, session_id, "metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def _load_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        meta_path = os.path.join(self._base_dir, session_id, "metadata.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return SessionMetadata.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load metadata for {session_id}: {e}")
            return None

    @staticmethod
    def _message_to_dict(message: Message) -> dict:
        """Serialize a Message to a JSON-safe dict."""
        data = {
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,
        }

        if message.tool_calls:
            data["tool_calls"] = [
                {"name": tc.name, "tool_id": tc.tool_id, "input": tc.input}
                for tc in message.tool_calls
            ]

        if message.tool_results:
            data["tool_results"] = [
                {
                    "tool_id": tr.tool_id,
                    "success": tr.success,
                    "output": tr.output,
                    "error": tr.error,
                    "metadata": tr.metadata,
                }
                for tr in message.tool_results
            ]

        return data

    @staticmethod
    def _dict_to_message(data: dict) -> Message:
        """Deserialize a dict back to a Message."""
        tool_calls = None
        if "tool_calls" in data and data["tool_calls"]:
            tool_calls = [
                ToolCall(name=tc["name"], tool_id=tc["tool_id"], input=tc["input"])
                for tc in data["tool_calls"]
            ]

        tool_results = None
        if "tool_results" in data and data["tool_results"]:
            tool_results = [
                ToolResult(
                    tool_id=tr["tool_id"],
                    success=tr["success"],
                    output=tr["output"],
                    error=tr.get("error"),
                    metadata=tr.get("metadata", {}),
                )
                for tr in data["tool_results"]
            ]

        return Message(
            role=data["role"],
            content=data["content"],
            tool_calls=tool_calls,
            tool_results=tool_results,
            timestamp=data.get("timestamp", 0.0),
        )
