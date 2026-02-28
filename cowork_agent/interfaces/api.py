"""
REST API + WebSocket Interface — Remote control for the cowork agent.

Provides HTTP endpoints for chat, session management, and health checks,
plus a WebSocket endpoint for real-time bidirectional streaming.  Serves
the web dashboard at ``/``.

Usage::

    python -m cowork_agent --mode api --api-port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .base import BaseInterface
from ..core.agent import Agent
from ..core.models import ToolCall, ToolResult, Message


# ── Session Management ───────────────────────────────────────────


@dataclass
class SessionMetadata:
    """Metadata for a single agent session."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    message_count: int = 0


class APISessionManager:
    """Manages multiple agent sessions for concurrent users.

    Each session gets its own isolated Agent instance with shared
    configuration but independent conversation state.
    """

    def __init__(self, agent_factory):
        """Initialize with a callable that produces new Agent instances."""
        self._agent_factory = agent_factory
        self._agents: dict[str, Agent] = {}
        self._metadata: dict[str, SessionMetadata] = {}

    async def create_session(self, user_id: str | None = None) -> str:
        """Create a new session and return its ID."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        agent = self._agent_factory()
        self._agents[session_id] = agent
        self._metadata[session_id] = SessionMetadata(
            session_id=session_id, user_id=user_id,
        )
        return session_id

    async def get_agent(self, session_id: str) -> Agent:
        """Retrieve the Agent for a session.  Raises HTTPException on 404."""
        if session_id not in self._agents:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        self._metadata[session_id].last_active = time.time()
        return self._agents[session_id]

    def get_metadata(self, session_id: str) -> SessionMetadata:
        """Get metadata for a session."""
        if session_id not in self._metadata:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return self._metadata[session_id]

    async def close_session(self, session_id: str) -> None:
        """End a session and free resources."""
        self._agents.pop(session_id, None)
        self._metadata.pop(session_id, None)

    def list_sessions(self) -> list[dict]:
        """Return metadata for all active sessions."""
        return [
            {
                "session_id": m.session_id,
                "user_id": m.user_id,
                "created_at": m.created_at,
                "last_active": m.last_active,
                "message_count": m.message_count,
            }
            for m in self._metadata.values()
        ]

    async def cleanup_stale(self, max_age_hours: int = 24) -> int:
        """Remove sessions inactive for longer than *max_age_hours*."""
        cutoff = time.time() - (max_age_hours * 3600)
        stale = [
            sid for sid, m in self._metadata.items()
            if m.last_active < cutoff
        ]
        for sid in stale:
            await self.close_session(sid)
        return len(stale)


# ── WebSocket Connection Tracking ────────────────────────────────


class WebSocketConnectionManager:
    """Track active WebSocket connections grouped by session."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        """Accept and register a WebSocket."""
        await ws.accept()
        self._connections.setdefault(session_id, []).append(ws)

    async def disconnect(self, session_id: str, ws: WebSocket) -> None:
        """Unregister a WebSocket."""
        conns = self._connections.get(session_id, [])
        if ws in conns:
            conns.remove(ws)
        if not conns:
            self._connections.pop(session_id, None)

    async def broadcast(self, session_id: str, message: dict) -> None:
        """Send a JSON message to every client watching *session_id*."""
        dead: list[WebSocket] = []
        for ws in self._connections.get(session_id, []):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(session_id, ws)

    def connection_count(self, session_id: str) -> int:
        """Number of active connections for a session."""
        return len(self._connections.get(session_id, []))


# ── REST + WebSocket Interface ───────────────────────────────────


class RestAPIInterface(BaseInterface):
    """FastAPI-based REST API and WebSocket server.

    Endpoints:
        POST   /api/sessions              — create session
        GET    /api/sessions              — list sessions
        GET    /api/sessions/{id}         — session metadata
        GET    /api/sessions/{id}/messages— conversation history
        DELETE /api/sessions/{id}         — end session
        POST   /api/chat/{id}            — send message (full response)
        POST   /api/chat/{id}/stream     — send message (SSE stream)
        GET    /api/tools                 — list available tools
        GET    /api/health                — health check
        GET    /                          — web dashboard
        WS     /ws/{id}                   — WebSocket endpoint
    """

    def __init__(
        self,
        agent: Agent,
        agent_factory=None,
        host: str = "127.0.0.1",
        port: int = 8000,
        dashboard_provider=None,
    ):
        super().__init__(agent)
        self.host = host
        self.port = port
        self._dashboard_provider = dashboard_provider

        # If no factory provided, build a simple one from the prototype agent
        if agent_factory is None:
            agent_factory = lambda: agent  # noqa: E731 — single-session fallback
        self._sessions = APISessionManager(agent_factory)
        self._ws = WebSocketConnectionManager()
        self._pending_questions: dict[str, dict] = {}
        self._cancellation_tokens: dict[str, Any] = {}  # session_id -> StreamCancellationToken
        self._dashboard_ws_clients: list[WebSocket] = []

        self.app = FastAPI(
            title="Cowork Agent API",
            description="Remote control interface for the cowork agent.",
            version="1.0.0",
        )
        self._setup_routes()

    # ── Route Setup ─────────────────────────────────────────────

    def _setup_routes(self) -> None:
        """Register all API routes."""
        app = self.app

        # ── Dashboard ───────────────────────────────────────────

        @app.get("/", response_class=HTMLResponse)
        async def serve_dashboard():
            """Serve the single-file web dashboard."""
            dashboard_path = Path(__file__).parent / "web" / "dashboard.html"
            if dashboard_path.exists():
                return HTMLResponse(content=dashboard_path.read_text())
            return HTMLResponse(content="<h1>Cowork Agent API</h1><p>Dashboard not found.</p>")

        @app.get("/dashboard", response_class=HTMLResponse)
        async def serve_observability_dashboard():
            """Serve the observability dashboard."""
            obs_path = Path(__file__).parent / "web" / "observability_dashboard.html"
            if obs_path.exists():
                return HTMLResponse(content=obs_path.read_text())
            return HTMLResponse(content="<h1>Observability Dashboard</h1><p>Not found.</p>")

        @app.get("/api/dashboard/full")
        async def dashboard_full():
            """Full dashboard snapshot for initial page load."""
            if not self._dashboard_provider:
                return JSONResponse({"error": "Dashboard not configured"}, status_code=503)
            return self._dashboard_provider.get_full_dashboard()

        @app.get("/api/dashboard/metrics")
        async def dashboard_metrics():
            """Current metrics snapshot."""
            if not self._dashboard_provider:
                return JSONResponse({"error": "Dashboard not configured"}, status_code=503)
            return self._dashboard_provider.get_metrics_snapshot()

        @app.get("/api/dashboard/metrics/historical")
        async def dashboard_metrics_historical(days: int = 7):
            """Historical metrics from persistent store."""
            if not self._dashboard_provider:
                return JSONResponse({"error": "Dashboard not configured"}, status_code=503)
            return self._dashboard_provider.get_metrics_historical(days=days)

        @app.get("/api/dashboard/audit")
        async def dashboard_audit(severity: Optional[str] = None, limit: int = 100):
            """Filtered audit events."""
            if not self._dashboard_provider:
                return JSONResponse({"error": "Dashboard not configured"}, status_code=503)
            return self._dashboard_provider.get_audit_feed(severity=severity, limit=limit)

        @app.get("/api/dashboard/health")
        async def dashboard_health():
            """Health orchestrator snapshot with trends."""
            if not self._dashboard_provider:
                return JSONResponse({"error": "Dashboard not configured"}, status_code=503)
            return self._dashboard_provider.get_health_snapshot()

        @app.get("/api/dashboard/benchmarks")
        async def dashboard_benchmarks(name: Optional[str] = None):
            """Benchmark stats and recent runs."""
            if not self._dashboard_provider:
                return JSONResponse({"error": "Dashboard not configured"}, status_code=503)
            return self._dashboard_provider.get_benchmark_data(name=name)

        @app.get("/api/dashboard/store")
        async def dashboard_store_stats():
            """Persistent store statistics."""
            if not self._dashboard_provider:
                return JSONResponse({"error": "Dashboard not configured"}, status_code=503)
            return self._dashboard_provider.get_store_stats()

        # ── Dashboard WebSocket ────────────────────────────────

        @app.websocket("/ws/dashboard")
        async def dashboard_ws(ws: WebSocket):
            """WebSocket for real-time dashboard updates."""
            await ws.accept()
            self._dashboard_ws_clients.append(ws)
            try:
                while True:
                    # Keep connection alive; client doesn't send data
                    raw = await ws.receive_text()
                    # Optionally handle refresh requests
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") == "refresh":
                            if self._dashboard_provider:
                                await ws.send_json({
                                    "type": "full_refresh",
                                    "data": self._dashboard_provider.get_full_dashboard(),
                                })
                    except (json.JSONDecodeError, Exception):
                        pass
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                if ws in self._dashboard_ws_clients:
                    self._dashboard_ws_clients.remove(ws)

        # ── Sessions ────────────────────────────────────────────

        @app.post("/api/sessions")
        async def create_session(request: Request):
            """Create a new agent session."""
            body = {}
            try:
                body = await request.json()
            except Exception:
                pass
            user_id = body.get("user_id")
            session_id = await self._sessions.create_session(user_id)
            return {"session_id": session_id, "status": "created"}

        @app.get("/api/sessions")
        async def list_sessions():
            """List all active sessions."""
            return {"sessions": self._sessions.list_sessions()}

        @app.get("/api/sessions/{session_id}")
        async def get_session(session_id: str):
            """Get session metadata."""
            meta = self._sessions.get_metadata(session_id)
            agent = await self._sessions.get_agent(session_id)
            return {
                "session_id": meta.session_id,
                "user_id": meta.user_id,
                "created_at": meta.created_at,
                "last_active": meta.last_active,
                "message_count": len(agent.messages),
                "tools": agent.registry.tool_names,
            }

        @app.get("/api/sessions/{session_id}/messages")
        async def get_messages(session_id: str):
            """Get conversation history for a session."""
            agent = await self._sessions.get_agent(session_id)
            return {
                "session_id": session_id,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "timestamp": m.timestamp,
                        "has_tool_calls": bool(m.tool_calls),
                    }
                    for m in agent.messages
                ],
            }

        @app.delete("/api/sessions/{session_id}")
        async def delete_session(session_id: str):
            """End a session."""
            await self._sessions.close_session(session_id)
            return {"status": "closed", "session_id": session_id}

        # ── Chat ────────────────────────────────────────────────

        @app.post("/api/chat/{session_id}")
        async def send_message(session_id: str, request: Request):
            """Send a message and get the full response."""
            body = await request.json()
            content = body.get("content", "")
            if not content:
                raise HTTPException(status_code=400, detail="content is required")

            agent = await self._sessions.get_agent(session_id)
            self._wire_agent_callbacks(agent, session_id)
            response = await agent.run(content)
            self._sessions._metadata[session_id].message_count = len(agent.messages)
            return {
                "session_id": session_id,
                "response": response,
                "message_count": len(agent.messages),
            }

        @app.post("/api/chat/{session_id}/stream")
        async def send_message_stream(session_id: str, request: Request):
            """Send a message and stream the response via SSE.

            When the agent has ``_events_enabled``, emits structured
            ``StreamEvent`` objects.  Otherwise falls back to raw text chunks.
            """
            body = await request.json()
            content = body.get("content", "")
            if not content:
                raise HTTPException(status_code=400, detail="content is required")

            agent = await self._sessions.get_agent(session_id)
            self._wire_agent_callbacks(agent, session_id)

            async def event_generator():
                try:
                    if getattr(agent, '_events_enabled', False):
                        # Structured event stream
                        from ..core.stream_events import event_to_dict
                        from ..core.stream_cancellation import StreamCancellationToken

                        token = self._cancellation_tokens.get(session_id)
                        if token is None:
                            token = StreamCancellationToken()
                            self._cancellation_tokens[session_id] = token
                        else:
                            token.reset()

                        async for event in agent.run_stream_events(
                            content, cancellation_token=token,
                        ):
                            data = json.dumps(event_to_dict(event))
                            yield f"data: {data}\n\n"
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    else:
                        # Raw text chunks (backward compat)
                        async for chunk in agent.run_stream(content):
                            data = json.dumps({"type": "TextChunk", "text": chunk})
                            yield f"data: {data}\n\n"
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @app.post("/api/chat/{session_id}/cancel")
        async def cancel_stream(session_id: str):
            """Cancel an in-progress streaming response."""
            token = self._cancellation_tokens.get(session_id)
            if token is None:
                return {"status": "no_active_stream", "session_id": session_id}
            token.cancel(reason="Cancelled via API")
            return {"status": "cancelled", "session_id": session_id}

        # ── Tools & Health ──────────────────────────────────────

        @app.get("/api/tools")
        async def list_tools():
            """List all available tools."""
            schemas = self.agent.registry.get_schemas()
            return {
                "tools": [
                    {"name": s.name, "description": s.description}
                    for s in schemas
                ],
            }

        @app.get("/api/health")
        async def health_check():
            """System health check."""
            return {
                "status": "ok",
                "service": "cowork-agent-api",
                "sessions": len(self._sessions._metadata),
                "timestamp": time.time(),
            }

        # ── WebSocket ───────────────────────────────────────────

        @app.websocket("/ws/{session_id}")
        async def websocket_endpoint(ws: WebSocket, session_id: str):
            """Real-time bidirectional communication."""
            try:
                agent = await self._sessions.get_agent(session_id)
            except HTTPException:
                await ws.close(code=4004, reason="Session not found")
                return

            await self._ws.connect(session_id, ws)
            self._wire_agent_callbacks(agent, session_id)

            try:
                while True:
                    raw = await ws.receive_text()
                    msg = json.loads(raw)

                    if msg.get("type") == "message":
                        content = msg.get("content", "")
                        if not content:
                            await ws.send_json({"type": "error", "message": "Empty message"})
                            continue

                        try:
                            async for chunk in agent.run_stream(content):
                                await self._ws.broadcast(session_id, {
                                    "type": "message_chunk",
                                    "chunk": chunk,
                                })
                            await self._ws.broadcast(session_id, {"type": "message_end"})
                        except Exception as e:
                            await self._ws.broadcast(session_id, {
                                "type": "error",
                                "message": str(e),
                            })

                    elif msg.get("type") == "answer":
                        q_id = msg.get("question_id", "")
                        answer = msg.get("answer", "")
                        if q_id in self._pending_questions:
                            self._pending_questions[q_id]["answer"] = answer
                            self._pending_questions[q_id]["event"].set()

            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                await self._ws.disconnect(session_id, ws)

    # ── Callback Wiring ─────────────────────────────────────────

    def _wire_agent_callbacks(self, agent: Agent, session_id: str) -> None:
        """Wire tool/status callbacks to broadcast over WebSocket."""
        timers: dict[str, float] = {}

        def on_tool_start(call: ToolCall) -> None:
            timers[call.tool_id] = time.time()
            asyncio.ensure_future(self._ws.broadcast(session_id, {
                "type": "tool_start",
                "tool_name": call.name,
                "tool_id": call.tool_id,
            }))

        def on_tool_end(call: ToolCall, result: ToolResult) -> None:
            elapsed = (time.time() - timers.pop(call.tool_id, time.time())) * 1000
            asyncio.ensure_future(self._ws.broadcast(session_id, {
                "type": "tool_end",
                "tool_name": call.name,
                "tool_id": call.tool_id,
                "success": result.success,
                "duration_ms": round(elapsed, 1),
                "output_preview": (result.output or "")[:200] if result.success else None,
                "error": result.error,
            }))

        def on_status(message: str) -> None:
            asyncio.ensure_future(self._ws.broadcast(session_id, {
                "type": "status",
                "message": message,
            }))

        agent.on_tool_start = on_tool_start
        agent.on_tool_end = on_tool_end
        agent.on_status = on_status

    # ── Dashboard Broadcast ──────────────────────────────────

    async def broadcast_dashboard_event(self, event_type: str, data: dict) -> None:
        """Broadcast a dashboard event to all connected dashboard WebSocket clients."""
        message = {"type": event_type, "data": data}
        dead: list[WebSocket] = []
        for ws in self._dashboard_ws_clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._dashboard_ws_clients.remove(ws)

    # ── ask_user Handler ────────────────────────────────────────

    def ask_user_handler(self, question: str, options: list[str]) -> str:
        """Send question over WebSocket and wait for client answer."""
        q_id = f"q_{uuid.uuid4().hex[:8]}"
        event = asyncio.Event()
        self._pending_questions[q_id] = {
            "question": question,
            "options": options,
            "answer": "",
            "event": event,
        }

        # Broadcast to all sessions (simplified — in practice scope to session)
        for sid in list(self._ws._connections.keys()):
            asyncio.ensure_future(self._ws.broadcast(sid, {
                "type": "ask_user",
                "question_id": q_id,
                "question": question,
                "options": options,
            }))

        # Wait for answer with timeout
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(asyncio.wait_for(event.wait(), timeout=60))
            return self._pending_questions[q_id]["answer"]
        except (asyncio.TimeoutError, RuntimeError):
            return ""
        finally:
            self._pending_questions.pop(q_id, None)

    # ── Lifecycle ───────────────────────────────────────────────

    async def run(self) -> None:
        """Start the FastAPI server via uvicorn."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
