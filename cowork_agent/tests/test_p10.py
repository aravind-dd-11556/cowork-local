"""
Sprint 10 Tests — Remote Control Interfaces

Tests for: BaseInterface, REST API, WebSocket, Telegram Bot, Slack Bot,
session management, and cross-interface integration.

Run: python -m pytest cowork_agent/tests/test_p10.py -v
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from typing import AsyncIterator, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# ── Ensure project root is on path ──────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cowork_agent.core.models import (
    Message, ToolCall, ToolResult, ToolSchema, AgentResponse,
)
from cowork_agent.interfaces.base import BaseInterface


# ── Helpers ──────────────────────────────────────────────────────


def _run(coro):
    """Run an async function synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class MockProvider:
    """Minimal mock LLM provider."""
    provider_name = "mock"
    model = "mock-model"

    def __init__(self, response_text="Mock response"):
        self._response_text = response_text

    async def send_message(self, messages, tools, system_prompt):
        return AgentResponse(text=self._response_text, stop_reason="end_turn",
                             usage={"input_tokens": 10, "output_tokens": 5})

    async def send_message_stream(self, messages, tools, system_prompt):
        response = await self.send_message(messages, tools, system_prompt)
        self._last_stream_response = response
        if response.text:
            yield response.text

    @property
    def last_stream_response(self):
        return getattr(self, "_last_stream_response", None)

    async def health_check(self):
        return {"status": "ok"}


def _make_mock_agent(response_text="Test response"):
    """Create a mock Agent with the minimum interface."""
    agent = MagicMock()
    agent.messages = []
    agent.on_tool_start = None
    agent.on_tool_end = None
    agent.on_status = None
    agent.registry = MagicMock()
    agent.registry.tool_names = ["bash", "read", "write"]
    agent.registry.get_schemas.return_value = [
        ToolSchema(name="bash", description="Execute commands", input_schema={}),
        ToolSchema(name="read", description="Read files", input_schema={}),
    ]

    async def mock_run(user_input):
        msg = Message(role="user", content=user_input)
        agent.messages.append(msg)
        resp = Message(role="assistant", content=response_text)
        agent.messages.append(resp)
        return response_text

    async def mock_run_stream(user_input):
        msg = Message(role="user", content=user_input)
        agent.messages.append(msg)
        for word in response_text.split():
            yield word + " "
        resp = Message(role="assistant", content=response_text)
        agent.messages.append(resp)

    agent.run = mock_run
    agent.run_stream = mock_run_stream
    agent.clear_history = MagicMock()
    return agent


# ═════════════════════════════════════════════════════════════════
# Test Suite 1: BaseInterface
# ═════════════════════════════════════════════════════════════════


class TestBaseInterface(unittest.TestCase):
    """Test the abstract BaseInterface contract."""

    def test_cannot_instantiate_directly(self):
        """BaseInterface is abstract — can't instantiate directly."""
        agent = _make_mock_agent()
        with self.assertRaises(TypeError):
            BaseInterface(agent)

    def test_concrete_subclass(self):
        """A concrete subclass can be instantiated."""
        class DummyInterface(BaseInterface):
            async def run(self):
                pass

        agent = _make_mock_agent()
        iface = DummyInterface(agent)
        self.assertIs(iface.agent, agent)

    def test_callbacks_wired_on_init(self):
        """Callbacks are wired to agent on construction."""
        class DummyInterface(BaseInterface):
            async def run(self):
                pass

        agent = _make_mock_agent()
        iface = DummyInterface(agent)
        self.assertEqual(agent.on_tool_start, iface._on_tool_start)
        self.assertEqual(agent.on_tool_end, iface._on_tool_end)
        self.assertEqual(agent.on_status, iface._on_status)

    def test_default_ask_user_returns_empty(self):
        """Default ask_user_handler returns empty string."""
        class DummyInterface(BaseInterface):
            async def run(self):
                pass

        agent = _make_mock_agent()
        iface = DummyInterface(agent)
        result = iface.ask_user_handler("Pick one", ["a", "b"])
        self.assertEqual(result, "")

    def test_on_tool_start_callable(self):
        """Default _on_tool_start doesn't raise."""
        class DummyInterface(BaseInterface):
            async def run(self):
                pass

        agent = _make_mock_agent()
        iface = DummyInterface(agent)
        call = ToolCall(name="bash", tool_id="t1", input={"command": "ls"})
        iface._on_tool_start(call)  # Should not raise

    def test_on_tool_end_callable(self):
        """Default _on_tool_end doesn't raise."""
        class DummyInterface(BaseInterface):
            async def run(self):
                pass

        agent = _make_mock_agent()
        iface = DummyInterface(agent)
        call = ToolCall(name="bash", tool_id="t1", input={})
        result = ToolResult(tool_id="t1", success=True, output="ok")
        iface._on_tool_end(call, result)  # Should not raise

    def test_on_status_callable(self):
        """Default _on_status doesn't raise."""
        class DummyInterface(BaseInterface):
            async def run(self):
                pass

        agent = _make_mock_agent()
        iface = DummyInterface(agent)
        iface._on_status("retrying...")  # Should not raise

    def test_custom_ask_user_override(self):
        """Subclass can override ask_user_handler."""
        class InteractiveInterface(BaseInterface):
            async def run(self):
                pass
            def ask_user_handler(self, question, options):
                return options[0] if options else ""

        agent = _make_mock_agent()
        iface = InteractiveInterface(agent)
        self.assertEqual(iface.ask_user_handler("Pick", ["x", "y"]), "x")


# ═════════════════════════════════════════════════════════════════
# Test Suite 2: APISessionManager
# ═════════════════════════════════════════════════════════════════


class TestAPISessionManager(unittest.TestCase):
    """Test session creation, retrieval, cleanup."""

    def setUp(self):
        from cowork_agent.interfaces.api import APISessionManager
        self.mgr = APISessionManager(agent_factory=_make_mock_agent)

    def test_create_session(self):
        """Creating a session returns a unique ID."""
        sid = _run(self.mgr.create_session())
        self.assertTrue(sid.startswith("sess_"))
        self.assertEqual(len(sid), 17)  # "sess_" + 12 hex chars

    def test_create_multiple_sessions(self):
        """Each session gets a unique ID."""
        s1 = _run(self.mgr.create_session())
        s2 = _run(self.mgr.create_session())
        self.assertNotEqual(s1, s2)

    def test_get_agent(self):
        """Can retrieve agent for a valid session."""
        sid = _run(self.mgr.create_session())
        agent = _run(self.mgr.get_agent(sid))
        self.assertIsNotNone(agent)

    def test_get_agent_invalid_session(self):
        """Getting an invalid session raises HTTPException."""
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            _run(self.mgr.get_agent("nonexistent"))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_close_session(self):
        """Closing a session removes it."""
        from fastapi import HTTPException
        sid = _run(self.mgr.create_session())
        _run(self.mgr.close_session(sid))
        with self.assertRaises(HTTPException):
            _run(self.mgr.get_agent(sid))

    def test_list_sessions(self):
        """list_sessions returns all active sessions."""
        _run(self.mgr.create_session("user1"))
        _run(self.mgr.create_session("user2"))
        sessions = self.mgr.list_sessions()
        self.assertEqual(len(sessions), 2)
        user_ids = {s["user_id"] for s in sessions}
        self.assertEqual(user_ids, {"user1", "user2"})

    def test_get_metadata(self):
        """Metadata includes session_id and timestamps."""
        sid = _run(self.mgr.create_session("alice"))
        meta = self.mgr.get_metadata(sid)
        self.assertEqual(meta.session_id, sid)
        self.assertEqual(meta.user_id, "alice")
        self.assertIsInstance(meta.created_at, float)

    def test_cleanup_stale(self):
        """cleanup_stale removes old sessions."""
        sid = _run(self.mgr.create_session())
        # Artificially age the session
        self.mgr._metadata[sid].last_active = time.time() - 86500  # > 24h
        removed = _run(self.mgr.cleanup_stale(max_age_hours=24))
        self.assertEqual(removed, 1)
        self.assertEqual(len(self.mgr.list_sessions()), 0)

    def test_cleanup_keeps_active(self):
        """cleanup_stale keeps recent sessions."""
        sid = _run(self.mgr.create_session())
        removed = _run(self.mgr.cleanup_stale(max_age_hours=24))
        self.assertEqual(removed, 0)
        self.assertEqual(len(self.mgr.list_sessions()), 1)

    def test_session_with_user_id(self):
        """Sessions can be tagged with a user_id."""
        sid = _run(self.mgr.create_session(user_id="bot_user"))
        meta = self.mgr.get_metadata(sid)
        self.assertEqual(meta.user_id, "bot_user")


# ═════════════════════════════════════════════════════════════════
# Test Suite 3: WebSocketConnectionManager
# ═════════════════════════════════════════════════════════════════


class TestWebSocketConnectionManager(unittest.TestCase):
    """Test WebSocket connection tracking."""

    def setUp(self):
        from cowork_agent.interfaces.api import WebSocketConnectionManager
        self.mgr = WebSocketConnectionManager()

    def test_initial_empty(self):
        """No connections initially."""
        self.assertEqual(self.mgr.connection_count("s1"), 0)

    def test_connect_and_count(self):
        """Connecting increases count."""
        ws = MagicMock()
        ws.accept = AsyncMock()
        _run(self.mgr.connect("s1", ws))
        self.assertEqual(self.mgr.connection_count("s1"), 1)

    def test_disconnect(self):
        """Disconnecting decreases count."""
        ws = MagicMock()
        ws.accept = AsyncMock()
        _run(self.mgr.connect("s1", ws))
        _run(self.mgr.disconnect("s1", ws))
        self.assertEqual(self.mgr.connection_count("s1"), 0)

    def test_multiple_connections(self):
        """Multiple clients can watch same session."""
        ws1, ws2 = MagicMock(), MagicMock()
        ws1.accept = ws2.accept = AsyncMock()
        _run(self.mgr.connect("s1", ws1))
        _run(self.mgr.connect("s1", ws2))
        self.assertEqual(self.mgr.connection_count("s1"), 2)

    def test_broadcast(self):
        """Broadcast sends to all connected clients."""
        ws1, ws2 = MagicMock(), MagicMock()
        ws1.accept = ws2.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2.send_json = AsyncMock()
        _run(self.mgr.connect("s1", ws1))
        _run(self.mgr.connect("s1", ws2))
        _run(self.mgr.broadcast("s1", {"type": "test"}))
        ws1.send_json.assert_called_once_with({"type": "test"})
        ws2.send_json.assert_called_once_with({"type": "test"})

    def test_broadcast_removes_dead_connections(self):
        """Dead connections are removed on broadcast."""
        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock(side_effect=Exception("closed"))
        _run(self.mgr.connect("s1", ws))
        _run(self.mgr.broadcast("s1", {"type": "test"}))
        self.assertEqual(self.mgr.connection_count("s1"), 0)

    def test_broadcast_no_session(self):
        """Broadcast to nonexistent session does nothing."""
        _run(self.mgr.broadcast("nonexistent", {"type": "test"}))  # No error


# ═════════════════════════════════════════════════════════════════
# Test Suite 4: REST API Interface
# ═════════════════════════════════════════════════════════════════


class TestRestAPIInterface(unittest.TestCase):
    """Test REST API routes via FastAPI TestClient."""

    def setUp(self):
        from cowork_agent.interfaces.api import RestAPIInterface
        agent = _make_mock_agent()
        self.api = RestAPIInterface(
            agent=agent,
            agent_factory=_make_mock_agent,
            port=8099,
        )
        # Use FastAPI's test client (sync wrapper)
        try:
            from fastapi.testclient import TestClient
            self.client = TestClient(self.api.app)
        except ImportError:
            self.skipTest("fastapi testclient not available")

    def test_health_check(self):
        """GET /api/health returns ok."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "cowork-agent-api")

    def test_create_session(self):
        """POST /api/sessions creates a session."""
        resp = self.client.post("/api/sessions", json={})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("session_id", data)
        self.assertTrue(data["session_id"].startswith("sess_"))

    def test_list_sessions(self):
        """GET /api/sessions lists sessions."""
        self.client.post("/api/sessions", json={})
        self.client.post("/api/sessions", json={})
        resp = self.client.get("/api/sessions")
        data = resp.json()
        self.assertEqual(len(data["sessions"]), 2)

    def test_get_session(self):
        """GET /api/sessions/{id} returns metadata."""
        create = self.client.post("/api/sessions", json={})
        sid = create.json()["session_id"]
        resp = self.client.get(f"/api/sessions/{sid}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], sid)

    def test_get_session_not_found(self):
        """GET /api/sessions/{bad_id} returns 404."""
        resp = self.client.get("/api/sessions/nonexistent")
        self.assertEqual(resp.status_code, 404)

    def test_delete_session(self):
        """DELETE /api/sessions/{id} closes session."""
        create = self.client.post("/api/sessions", json={})
        sid = create.json()["session_id"]
        resp = self.client.delete(f"/api/sessions/{sid}")
        self.assertEqual(resp.status_code, 200)
        # Verify gone
        resp2 = self.client.get(f"/api/sessions/{sid}")
        self.assertEqual(resp2.status_code, 404)

    def test_send_message(self):
        """POST /api/chat/{id} returns agent response."""
        create = self.client.post("/api/sessions", json={})
        sid = create.json()["session_id"]
        resp = self.client.post(f"/api/chat/{sid}", json={"content": "Hello"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("response", data)
        self.assertEqual(data["response"], "Test response")

    def test_send_message_empty_content(self):
        """POST /api/chat with empty content returns 400."""
        create = self.client.post("/api/sessions", json={})
        sid = create.json()["session_id"]
        resp = self.client.post(f"/api/chat/{sid}", json={"content": ""})
        self.assertEqual(resp.status_code, 400)

    def test_send_message_invalid_session(self):
        """POST /api/chat to nonexistent session returns 404."""
        resp = self.client.post("/api/chat/bad_id", json={"content": "Hi"})
        self.assertEqual(resp.status_code, 404)

    def test_get_messages(self):
        """GET /api/sessions/{id}/messages returns conversation."""
        create = self.client.post("/api/sessions", json={})
        sid = create.json()["session_id"]
        self.client.post(f"/api/chat/{sid}", json={"content": "Hello"})
        resp = self.client.get(f"/api/sessions/{sid}/messages")
        data = resp.json()
        self.assertGreater(len(data["messages"]), 0)

    def test_list_tools(self):
        """GET /api/tools returns tool list."""
        resp = self.client.get("/api/tools")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("tools", data)

    def test_dashboard_served(self):
        """GET / returns HTML."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers.get("content-type", ""))

    def test_stream_endpoint(self):
        """POST /api/chat/{id}/stream returns SSE."""
        create = self.client.post("/api/sessions", json={})
        sid = create.json()["session_id"]
        resp = self.client.post(
            f"/api/chat/{sid}/stream",
            json={"content": "Hello"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers.get("content-type", ""))

    def test_create_session_with_user_id(self):
        """Session can be tagged with user_id."""
        resp = self.client.post("/api/sessions", json={"user_id": "user123"})
        sid = resp.json()["session_id"]
        meta = self.client.get(f"/api/sessions/{sid}").json()
        self.assertEqual(meta["user_id"], "user123")


# ═════════════════════════════════════════════════════════════════
# Test Suite 5: Telegram Bot
# ═════════════════════════════════════════════════════════════════


class TestTelegramBot(unittest.TestCase):
    """Test Telegram bot components."""

    def test_session_manager_create(self):
        """TelegramSessionManager creates sessions."""
        from cowork_agent.interfaces.telegram_bot import TelegramSessionManager
        mgr = TelegramSessionManager()
        session = mgr.get_or_create(123, 456)
        self.assertEqual(session.user_id, 123)
        self.assertEqual(session.chat_id, 456)

    def test_session_manager_idempotent(self):
        """Same user returns same session."""
        from cowork_agent.interfaces.telegram_bot import TelegramSessionManager
        mgr = TelegramSessionManager()
        s1 = mgr.get_or_create(123, 456)
        s2 = mgr.get_or_create(123, 456)
        self.assertEqual(s1.user_id, s2.user_id)

    def test_session_manager_persist(self):
        """Sessions persist to and load from JSON."""
        from cowork_agent.interfaces.telegram_bot import TelegramSessionManager
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mgr1 = TelegramSessionManager(persist_path=path)
            mgr1.get_or_create(111, 222)
            mgr1.get_or_create(333, 444)

            mgr2 = TelegramSessionManager(persist_path=path)
            self.assertIn(111, mgr2._sessions)
            self.assertIn(333, mgr2._sessions)
        finally:
            os.unlink(path)

    def test_session_manager_clear(self):
        """clear() removes a user's session."""
        from cowork_agent.interfaces.telegram_bot import TelegramSessionManager
        mgr = TelegramSessionManager()
        mgr.get_or_create(123, 456)
        mgr.clear(123)
        self.assertNotIn(123, mgr._sessions)

    def test_split_message_short(self):
        """Short messages are returned as-is."""
        from cowork_agent.interfaces.telegram_bot import TelegramBotInterface
        parts = TelegramBotInterface.split_message("Hello", 100)
        self.assertEqual(parts, ["Hello"])

    def test_split_message_long(self):
        """Long messages are split correctly."""
        from cowork_agent.interfaces.telegram_bot import TelegramBotInterface
        text = "A" * 100 + "\n" + "B" * 100
        parts = TelegramBotInterface.split_message(text, max_len=120)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 120 for p in parts))

    def test_split_message_no_newline(self):
        """Messages without newlines split at max_len."""
        from cowork_agent.interfaces.telegram_bot import TelegramBotInterface
        text = "X" * 200
        parts = TelegramBotInterface.split_message(text, max_len=80)
        self.assertGreater(len(parts), 1)

    def test_get_agent_creates_new(self):
        """_get_agent creates agent for new user."""
        from cowork_agent.interfaces.telegram_bot import TelegramBotInterface
        agent = _make_mock_agent()
        bot = TelegramBotInterface(
            agent=agent, token="fake",
            agent_factory=_make_mock_agent,
        )
        user_agent = bot._get_agent(999, 888)
        self.assertIsNotNone(user_agent)

    def test_get_agent_reuses_existing(self):
        """_get_agent returns same agent for same user."""
        from cowork_agent.interfaces.telegram_bot import TelegramBotInterface
        agent = _make_mock_agent()
        bot = TelegramBotInterface(
            agent=agent, token="fake",
            agent_factory=_make_mock_agent,
        )
        a1 = bot._get_agent(999, 888)
        a2 = bot._get_agent(999, 888)
        self.assertIs(a1, a2)

    def test_different_users_different_agents(self):
        """Different users get different agents."""
        from cowork_agent.interfaces.telegram_bot import TelegramBotInterface
        agent = _make_mock_agent()
        bot = TelegramBotInterface(
            agent=agent, token="fake",
            agent_factory=_make_mock_agent,
        )
        a1 = bot._get_agent(111, 100)
        a2 = bot._get_agent(222, 200)
        self.assertIsNot(a1, a2)

    def test_session_manager_updates_last_active(self):
        """get_or_create updates last_active."""
        from cowork_agent.interfaces.telegram_bot import TelegramSessionManager
        mgr = TelegramSessionManager()
        s1 = mgr.get_or_create(123, 456)
        t1 = s1.last_active
        time.sleep(0.01)
        s2 = mgr.get_or_create(123, 456)
        self.assertGreaterEqual(s2.last_active, t1)

    def test_session_manager_persist_no_path(self):
        """Without persist_path, save/load are no-ops."""
        from cowork_agent.interfaces.telegram_bot import TelegramSessionManager
        mgr = TelegramSessionManager(persist_path=None)
        mgr.get_or_create(123, 456)
        mgr._save()  # Should not raise
        mgr._load()  # Should not raise


# ═════════════════════════════════════════════════════════════════
# Test Suite 6: Slack Bot
# ═════════════════════════════════════════════════════════════════


class TestSlackBot(unittest.TestCase):
    """Test Slack bot components."""

    def test_split_message_short(self):
        """Short messages stay intact."""
        from cowork_agent.interfaces.slack_bot import SlackBotInterface
        parts = SlackBotInterface.split_message("Hello world", 3000)
        self.assertEqual(parts, ["Hello world"])

    def test_split_message_long(self):
        """Long messages are split."""
        from cowork_agent.interfaces.slack_bot import SlackBotInterface
        text = "Line\n" * 1000  # ~5000 chars
        parts = SlackBotInterface.split_message(text, max_len=3000)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 3000 for p in parts))

    def test_get_agent_creates(self):
        """_get_agent creates for new user."""
        from cowork_agent.interfaces.slack_bot import SlackBotInterface
        agent = _make_mock_agent()
        bot = SlackBotInterface(
            agent=agent, bot_token="xoxb-fake",
            app_token="xapp-fake",
            agent_factory=_make_mock_agent,
        )
        a = bot._get_agent("U123")
        self.assertIsNotNone(a)

    def test_get_agent_reuses(self):
        """Same user gets same agent."""
        from cowork_agent.interfaces.slack_bot import SlackBotInterface
        agent = _make_mock_agent()
        bot = SlackBotInterface(
            agent=agent, bot_token="xoxb-fake",
            app_token="xapp-fake",
            agent_factory=_make_mock_agent,
        )
        a1 = bot._get_agent("U123")
        a2 = bot._get_agent("U123")
        self.assertIs(a1, a2)

    def test_different_users_different_agents(self):
        """Different Slack users get different agents."""
        from cowork_agent.interfaces.slack_bot import SlackBotInterface
        agent = _make_mock_agent()
        bot = SlackBotInterface(
            agent=agent, bot_token="xoxb-fake",
            app_token="xapp-fake",
            agent_factory=_make_mock_agent,
        )
        a1 = bot._get_agent("U111")
        a2 = bot._get_agent("U222")
        self.assertIsNot(a1, a2)

    def test_handle_action_sets_pending(self):
        """Action handler resolves pending answer."""
        from cowork_agent.interfaces.slack_bot import SlackBotInterface
        agent = _make_mock_agent()
        bot = SlackBotInterface(
            agent=agent, bot_token="xoxb-fake",
            app_token="xapp-fake",
        )
        q_id = "q_test123"
        event = asyncio.Event()
        bot._pending_answers[q_id] = event
        bot._pending_values[q_id] = ""

        body = {
            "actions": [{"action_id": f"{q_id}|0", "value": "Option A"}],
            "channel": {"id": "C123"},
            "message": {"ts": "123.456", "text": "Question?"},
        }
        mock_ack = MagicMock()
        mock_client = MagicMock()

        _run(bot._handle_action(body, mock_ack, mock_client))
        self.assertEqual(bot._pending_values[q_id], "Option A")
        self.assertTrue(event.is_set())

    def test_handle_action_no_pending(self):
        """Action for unknown question does nothing."""
        from cowork_agent.interfaces.slack_bot import SlackBotInterface
        agent = _make_mock_agent()
        bot = SlackBotInterface(
            agent=agent, bot_token="xoxb-fake",
            app_token="xapp-fake",
        )
        body = {
            "actions": [{"action_id": "q_unknown|0", "value": "X"}],
            "channel": {"id": "C123"},
            "message": {"ts": "123.456", "text": "?"},
        }
        _run(bot._handle_action(body, MagicMock(), MagicMock()))
        # Should not raise

    def test_handle_action_empty_actions(self):
        """Empty actions list does nothing."""
        from cowork_agent.interfaces.slack_bot import SlackBotInterface
        agent = _make_mock_agent()
        bot = SlackBotInterface(
            agent=agent, bot_token="xoxb-fake",
            app_token="xapp-fake",
        )
        body = {"actions": [], "channel": {"id": "C"}, "message": {"ts": "1", "text": ""}}
        _run(bot._handle_action(body, MagicMock(), MagicMock()))

    def test_max_message_len(self):
        """MAX_MESSAGE_LEN is reasonable."""
        from cowork_agent.interfaces.slack_bot import SlackBotInterface
        self.assertLessEqual(SlackBotInterface.MAX_MESSAGE_LEN, 4000)
        self.assertGreater(SlackBotInterface.MAX_MESSAGE_LEN, 1000)


# ═════════════════════════════════════════════════════════════════
# Test Suite 7: Integration
# ═════════════════════════════════════════════════════════════════


class TestSprint10Integration(unittest.TestCase):
    """Cross-interface integration tests."""

    def test_api_session_isolation(self):
        """Different sessions have isolated message history."""
        from cowork_agent.interfaces.api import APISessionManager
        mgr = APISessionManager(agent_factory=_make_mock_agent)
        s1 = _run(mgr.create_session())
        s2 = _run(mgr.create_session())

        a1 = _run(mgr.get_agent(s1))
        a2 = _run(mgr.get_agent(s2))

        _run(a1.run("Hello from session 1"))
        self.assertEqual(len(a1.messages), 2)
        self.assertEqual(len(a2.messages), 0)

    def test_telegram_session_persists(self):
        """Telegram sessions survive restart via JSON."""
        from cowork_agent.interfaces.telegram_bot import TelegramSessionManager
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            m1 = TelegramSessionManager(persist_path=path)
            m1.get_or_create(42, 100)

            m2 = TelegramSessionManager(persist_path=path)
            self.assertIn(42, m2._sessions)
            self.assertEqual(m2._sessions[42].chat_id, 100)
        finally:
            os.unlink(path)

    def test_multiple_telegram_users(self):
        """Each Telegram user gets isolated agent."""
        from cowork_agent.interfaces.telegram_bot import TelegramBotInterface
        agent = _make_mock_agent()
        bot = TelegramBotInterface(
            agent=agent, token="fake",
            agent_factory=_make_mock_agent,
        )
        a1 = bot._get_agent(1, 10)
        a2 = bot._get_agent(2, 20)
        a3 = bot._get_agent(3, 30)
        agents = {id(a1), id(a2), id(a3)}
        self.assertEqual(len(agents), 3)  # All different

    def test_multiple_slack_users(self):
        """Each Slack user gets isolated agent."""
        from cowork_agent.interfaces.slack_bot import SlackBotInterface
        agent = _make_mock_agent()
        bot = SlackBotInterface(
            agent=agent, bot_token="xoxb-fake",
            app_token="xapp-fake",
            agent_factory=_make_mock_agent,
        )
        a1 = bot._get_agent("U1")
        a2 = bot._get_agent("U2")
        self.assertIsNot(a1, a2)

    def test_api_cleanup_preserves_active(self):
        """Session cleanup only removes stale sessions."""
        from cowork_agent.interfaces.api import APISessionManager
        mgr = APISessionManager(agent_factory=_make_mock_agent)
        s_old = _run(mgr.create_session())
        s_new = _run(mgr.create_session())

        mgr._metadata[s_old].last_active = time.time() - 100000
        removed = _run(mgr.cleanup_stale(max_age_hours=24))
        self.assertEqual(removed, 1)
        sessions = mgr.list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], s_new)

    def test_rest_api_full_workflow(self):
        """Full API workflow: create → chat → messages → delete."""
        from cowork_agent.interfaces.api import RestAPIInterface
        agent = _make_mock_agent()
        api = RestAPIInterface(agent=agent, agent_factory=_make_mock_agent)
        try:
            from fastapi.testclient import TestClient
            client = TestClient(api.app)
        except ImportError:
            self.skipTest("fastapi testclient not available")

        # Create
        resp = client.post("/api/sessions", json={})
        sid = resp.json()["session_id"]

        # Chat
        resp = client.post(f"/api/chat/{sid}", json={"content": "Hi"})
        self.assertEqual(resp.json()["response"], "Test response")

        # Messages
        resp = client.get(f"/api/sessions/{sid}/messages")
        self.assertGreater(len(resp.json()["messages"]), 0)

        # Delete
        resp = client.delete(f"/api/sessions/{sid}")
        self.assertEqual(resp.json()["status"], "closed")

    def test_ws_connection_manager_multi_session(self):
        """WebSocket manager tracks connections per session."""
        from cowork_agent.interfaces.api import WebSocketConnectionManager
        mgr = WebSocketConnectionManager()
        ws1, ws2 = MagicMock(), MagicMock()
        ws1.accept = ws2.accept = AsyncMock()

        _run(mgr.connect("s1", ws1))
        _run(mgr.connect("s2", ws2))
        self.assertEqual(mgr.connection_count("s1"), 1)
        self.assertEqual(mgr.connection_count("s2"), 1)

    def test_base_interface_subclass_override(self):
        """Custom interface with overridden callbacks works."""
        events = []

        class TrackingInterface(BaseInterface):
            async def run(self):
                pass
            def _on_tool_start(self, call):
                events.append(("start", call.name))
            def _on_tool_end(self, call, result):
                events.append(("end", call.name, result.success))

        agent = _make_mock_agent()
        iface = TrackingInterface(agent)
        call = ToolCall(name="bash", tool_id="t1", input={})
        result = ToolResult(tool_id="t1", success=True, output="ok")

        iface._on_tool_start(call)
        iface._on_tool_end(call, result)
        self.assertEqual(events, [("start", "bash"), ("end", "bash", True)])

    def test_main_mode_arg_parsing(self):
        """main.py parse_args accepts --mode flag."""
        from cowork_agent.main import parse_args
        with patch("sys.argv", ["prog", "--mode", "api", "--api-port", "9000"]):
            args = parse_args()
        self.assertEqual(args.mode, "api")
        self.assertEqual(args.api_port, 9000)


# ── Remote Control Command Tests ─────────────────────────────────


class TestRemoteControlCommand(unittest.TestCase):
    """Tests for /remote-control (/rc) CLI command."""

    def _make_cli(self, agent=None, factory=None):
        """Create a CLI instance with remote control support."""
        from cowork_agent.interfaces.cli import CLI
        ag = agent or _make_mock_agent()
        cli = CLI(
            agent=ag,
            history_file="/dev/null",
            streaming=False,
            agent_factory=factory or (lambda: _make_mock_agent()),
            workspace="/tmp/test_workspace",
        )
        return cli

    def test_rc_status_empty(self):
        """Status shows no services when none running."""
        cli = self._make_cli()
        self.assertEqual(cli._remote_services, {})
        # Should not raise
        cli._rc_show_status()

    def test_rc_help(self):
        """Help text doesn't raise."""
        cli = self._make_cli()
        cli._rc_help()

    def test_rc_handle_status_default(self):
        """Empty /rc shows status."""
        cli = self._make_cli()
        _run(cli._handle_remote_control(""))

    def test_rc_handle_help(self):
        """/rc help shows help."""
        cli = self._make_cli()
        _run(cli._handle_remote_control("help"))

    def test_rc_handle_unknown(self):
        """/rc foobar shows error."""
        cli = self._make_cli()
        _run(cli._handle_remote_control("foobar"))

    def test_rc_start_no_service(self):
        """/rc start with no service name shows error."""
        cli = self._make_cli()
        _run(cli._rc_start(""))

    def test_rc_start_unknown_service(self):
        """/rc start xyz shows error."""
        cli = self._make_cli()
        _run(cli._rc_start("xyz"))

    def test_rc_stop_no_service(self):
        """/rc stop with no name shows error."""
        cli = self._make_cli()
        _run(cli._rc_stop(""))

    def test_rc_stop_not_running(self):
        """/rc stop api when not running shows warning."""
        cli = self._make_cli()
        _run(cli._rc_stop("api"))

    def test_rc_stop_all_empty(self):
        """/rc stop all when nothing running is harmless."""
        cli = self._make_cli()
        _run(cli._stop_all_remote())

    def test_rc_start_api_no_factory(self):
        """API start fails gracefully without agent factory."""
        cli = self._make_cli()
        cli._agent_factory = None
        _run(cli._start_api_server())

    def test_rc_start_telegram_no_token(self):
        """Telegram start fails gracefully without token."""
        cli = self._make_cli()
        with patch.dict(os.environ, {}, clear=True):
            # Remove TELEGRAM_BOT_TOKEN if present
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            _run(cli._start_telegram_bot(""))

    def test_rc_start_slack_no_tokens(self):
        """Slack start fails gracefully without tokens."""
        cli = self._make_cli()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SLACK_BOT_TOKEN", None)
            os.environ.pop("SLACK_APP_TOKEN", None)
            _run(cli._start_slack_bot())

    def test_rc_start_api_success(self):
        """API server can be registered as a running service."""
        cli = self._make_cli()

        # Simulate what _start_api_server does after launching
        async def test():
            task = asyncio.create_task(asyncio.sleep(100))
            cli._remote_services["api"] = {
                "task": task,
                "started_at": time.time(),
                "info": "http://localhost:9999",
                "instance": MagicMock(),
            }
            self.assertIn("api", cli._remote_services)
            self.assertFalse(task.done())
            # Clean up
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        _run(test())

    def test_rc_already_running(self):
        """Starting a service that's already running shows warning."""
        cli = self._make_cli()

        # Add a fake running task
        loop = asyncio.new_event_loop()
        task = loop.create_task(asyncio.sleep(100))
        cli._remote_services["api"] = {
            "task": task,
            "started_at": time.time(),
            "info": "http://localhost:8000",
        }
        loop.run_until_complete(cli._rc_start("api"))
        # Should still be there (not duplicated)
        self.assertIn("api", cli._remote_services)
        task.cancel()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        loop.close()

    def test_rc_stop_cancels_task(self):
        """Stopping a service cancels its asyncio task."""
        cli = self._make_cli()

        async def test():
            task = asyncio.create_task(asyncio.sleep(100))
            cli._remote_services["api"] = {
                "task": task,
                "started_at": time.time(),
                "info": "http://localhost:8000",
            }
            await cli._rc_stop("api")
            self.assertTrue(task.cancelled() or task.done())
            self.assertNotIn("api", cli._remote_services)

        _run(test())

    def test_rc_stop_all_cancels_multiple(self):
        """Stop all cancels every running service."""
        cli = self._make_cli()

        async def test():
            t1 = asyncio.create_task(asyncio.sleep(100))
            t2 = asyncio.create_task(asyncio.sleep(100))
            cli._remote_services["api"] = {"task": t1, "started_at": time.time(), "info": "api"}
            cli._remote_services["telegram"] = {"task": t2, "started_at": time.time(), "info": "tg"}
            await cli._stop_all_remote()
            self.assertEqual(len(cli._remote_services), 0)
            self.assertTrue(t1.cancelled() or t1.done())
            self.assertTrue(t2.cancelled() or t2.done())

        _run(test())

    def test_rc_status_shows_running(self):
        """Status displays running services."""
        cli = self._make_cli()

        async def test():
            task = asyncio.create_task(asyncio.sleep(100))
            cli._remote_services["api"] = {
                "task": task,
                "started_at": time.time() - 120,  # 2 min ago
                "info": "http://localhost:8000",
            }
            cli._rc_show_status()  # Should not raise, prints to stdout
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        _run(test())

    def test_rc_status_shows_stopped(self):
        """Status shows 'stopped' for completed tasks."""
        cli = self._make_cli()

        async def test():
            task = asyncio.create_task(asyncio.sleep(0))  # completes immediately
            await asyncio.sleep(0.01)
            cli._remote_services["api"] = {
                "task": task,
                "started_at": time.time(),
                "info": "http://localhost:8000",
            }
            cli._rc_show_status()  # Should show stopped

        _run(test())

    def test_handle_command_async(self):
        """/rc command is properly dispatched from async _handle_command."""
        cli = self._make_cli()
        result = _run(cli._handle_command("/rc status"))
        self.assertTrue(result)

    def test_handle_command_exit_stops_remote(self):
        """/exit stops remote services before exiting."""
        cli = self._make_cli()

        async def test():
            task = asyncio.create_task(asyncio.sleep(100))
            cli._remote_services["api"] = {
                "task": task,
                "started_at": time.time(),
                "info": "test",
            }
            result = await cli._handle_command("/exit")
            self.assertTrue(result)
            self.assertFalse(cli._running)
            self.assertEqual(len(cli._remote_services), 0)

        _run(test())

    def test_cli_init_stores_factory(self):
        """CLI stores agent_factory and workspace."""
        factory = lambda: _make_mock_agent()
        cli = self._make_cli(factory=factory)
        self.assertIs(cli._agent_factory, factory)
        self.assertEqual(cli._workspace, "/tmp/test_workspace")

    def test_rc_start_all(self):
        """/rc start all attempts to start api, telegram, and slack."""
        cli = self._make_cli()
        # All will fail gracefully (no real tokens, no imports)
        _run(cli._rc_start("all"))


if __name__ == "__main__":
    unittest.main()
