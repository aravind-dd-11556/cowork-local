"""
Telegram Bot Interface — Control the cowork agent from Telegram.

Uses python-telegram-bot (async v21+) with long polling.
Each Telegram user gets their own agent session with isolated state.

Usage::

    python -m cowork_agent --mode telegram --telegram-token BOT_TOKEN

Bot commands:
    /start   — Create or resume session
    /clear   — Reset conversation
    /history — Show recent messages
    /tools   — List available tools
    /help    — Show help
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .base import BaseInterface
from ..core.agent import Agent
from ..core.models import ToolCall, ToolResult

logger = logging.getLogger(__name__)


# ── Session Persistence ──────────────────────────────────────────


@dataclass
class TelegramUserSession:
    """Maps a Telegram user to an agent session."""

    user_id: int
    chat_id: int
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class TelegramSessionManager:
    """Manage Telegram user → agent session mappings.

    Persists to a JSON file so sessions survive bot restarts.
    """

    def __init__(self, persist_path: str | None = None):
        self._sessions: dict[int, TelegramUserSession] = {}
        self._persist_path = persist_path
        if persist_path:
            self._load()

    def get_or_create(self, user_id: int, chat_id: int) -> TelegramUserSession:
        """Get existing session or create a new one."""
        if user_id not in self._sessions:
            self._sessions[user_id] = TelegramUserSession(
                user_id=user_id, chat_id=chat_id,
            )
            self._save()
        session = self._sessions[user_id]
        session.last_active = time.time()
        return session

    def clear(self, user_id: int) -> None:
        """Remove a user's session."""
        self._sessions.pop(user_id, None)
        self._save()

    def _save(self) -> None:
        """Persist sessions to JSON."""
        if not self._persist_path:
            return
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                str(uid): {
                    "user_id": s.user_id,
                    "chat_id": s.chat_id,
                    "created_at": s.created_at,
                    "last_active": s.last_active,
                }
                for uid, s in self._sessions.items()
            }
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to persist Telegram sessions: %s", e)

    def _load(self) -> None:
        """Load sessions from JSON."""
        if not self._persist_path:
            return
        path = Path(self._persist_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for uid_str, info in data.items():
                uid = int(uid_str)
                self._sessions[uid] = TelegramUserSession(
                    user_id=info["user_id"],
                    chat_id=info["chat_id"],
                    created_at=info.get("created_at", time.time()),
                    last_active=info.get("last_active", time.time()),
                )
        except Exception as e:
            logger.warning("Failed to load Telegram sessions: %s", e)


# ── Telegram Bot Interface ───────────────────────────────────────


class TelegramBotInterface(BaseInterface):
    """Telegram bot for remote agent control.

    Per-user sessions, inline keyboards for ask_user, streaming
    responses chunked into Telegram's 4096-char message limit.
    """

    MAX_MESSAGE_LEN = 4000  # Leave margin below Telegram's 4096

    def __init__(
        self,
        agent: Agent,
        token: str,
        agent_factory=None,
        persist_path: str | None = None,
    ):
        super().__init__(agent)
        self.token = token
        self._agent_factory = agent_factory or (lambda: agent)
        self._user_agents: dict[int, Agent] = {}
        self._session_mgr = TelegramSessionManager(persist_path)
        self._pending_answers: dict[str, asyncio.Event] = {}
        self._pending_values: dict[str, str] = {}

    def _get_agent(self, user_id: int, chat_id: int) -> Agent:
        """Get or create an agent for a Telegram user."""
        self._session_mgr.get_or_create(user_id, chat_id)
        if user_id not in self._user_agents:
            self._user_agents[user_id] = self._agent_factory()
        return self._user_agents[user_id]

    # ── Message splitting ───────────────────────────────────────

    @staticmethod
    def split_message(text: str, max_len: int = 4000) -> list[str]:
        """Split a long message into Telegram-safe chunks."""
        if len(text) <= max_len:
            return [text]
        parts: list[str] = []
        while text:
            if len(text) <= max_len:
                parts.append(text)
                break
            chunk = text[:max_len]
            # Try to break at a newline
            last_nl = chunk.rfind("\n")
            if last_nl > max_len // 2:
                parts.append(text[:last_nl])
                text = text[last_nl + 1:]
            else:
                parts.append(chunk)
                text = text[max_len:]
        return parts

    # ── Command Handlers ────────────────────────────────────────

    async def _cmd_start(self, update, context) -> None:
        """Handle /start command."""
        user = update.effective_user
        chat_id = update.effective_chat.id
        self._get_agent(user.id, chat_id)
        await update.message.reply_text(
            f"Welcome to Cowork Agent, {user.first_name}!\n\n"
            f"Send me any message and I'll process it.\n"
            f"Use /help to see available commands."
        )

    async def _cmd_help(self, update, context) -> None:
        """Handle /help command."""
        await update.message.reply_text(
            "Cowork Agent Commands:\n\n"
            "/start   - Create or resume session\n"
            "/clear   - Reset conversation history\n"
            "/history - Show recent messages\n"
            "/tools   - List available tools\n"
            "/help    - Show this help"
        )

    async def _cmd_clear(self, update, context) -> None:
        """Handle /clear command."""
        user_id = update.effective_user.id
        if user_id in self._user_agents:
            self._user_agents[user_id].clear_history()
        await update.message.reply_text("Conversation cleared.")

    async def _cmd_history(self, update, context) -> None:
        """Handle /history command."""
        user_id = update.effective_user.id
        agent = self._user_agents.get(user_id)
        if not agent or not agent.messages:
            await update.message.reply_text("No conversation history.")
            return
        lines = []
        for m in agent.messages[-10:]:
            role = m.role.upper()
            preview = (m.content or "")[:100]
            if len(m.content or "") > 100:
                preview += "..."
            lines.append(f"[{role}] {preview}")
        await update.message.reply_text("\n\n".join(lines))

    async def _cmd_tools(self, update, context) -> None:
        """Handle /tools command."""
        schemas = self.agent.registry.get_schemas()
        tool_list = "\n".join(f"- {s.name}: {s.description[:60]}" for s in schemas)
        await update.message.reply_text(f"Available tools:\n\n{tool_list}")

    # ── Message Handler ─────────────────────────────────────────

    async def _handle_message(self, update, context) -> None:
        """Process a text message from the user."""
        user = update.effective_user
        chat_id = update.effective_chat.id
        text = update.message.text
        if not text:
            return

        agent = self._get_agent(user.id, chat_id)

        # Show typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        # Wire tool callbacks for this message
        tool_msgs: list[str] = []

        def on_tool_start(call: ToolCall) -> None:
            tool_msgs.append(f"\u2699\ufe0f Executing {call.name}...")

        def on_tool_end(call: ToolCall, result: ToolResult) -> None:
            icon = "\u2705" if result.success else "\u274c"
            tool_msgs.append(f"{icon} {call.name} done")

        agent.on_tool_start = on_tool_start
        agent.on_tool_end = on_tool_end

        # Run agent
        try:
            response_parts: list[str] = []
            async for chunk in agent.run_stream(text):
                response_parts.append(chunk)
            response = "".join(response_parts)
        except Exception as e:
            response = f"Error: {e}"

        # Send tool summary if any tools were used
        if tool_msgs:
            tool_summary = "\n".join(tool_msgs)
            await context.bot.send_message(
                chat_id=chat_id,
                text=tool_summary,
            )

        # Send response (split if needed)
        if response:
            for part in self.split_message(response):
                await context.bot.send_message(chat_id=chat_id, text=part)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="(No response from agent)"
            )

    # ── ask_user via Inline Keyboard ────────────────────────────

    async def _handle_callback(self, update, context) -> None:
        """Handle inline keyboard button press for ask_user."""
        query = update.callback_query
        await query.answer()

        data = query.data or ""
        if "|" not in data:
            return
        q_id, answer = data.split("|", 1)

        if q_id in self._pending_answers:
            self._pending_values[q_id] = answer
            self._pending_answers[q_id].set()

        await query.edit_message_text(
            text=f"{query.message.text}\n\nSelected: {answer}"
        )

    def _make_ask_user_handler(self, chat_id: int):
        """Create an ask_user callback scoped to a chat."""

        def handler(question: str, options: list[str]) -> str:
            """Send inline keyboard and wait for response."""
            try:
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            except ImportError:
                return ""

            q_id = f"q_{uuid.uuid4().hex[:8]}"
            event = asyncio.Event()
            self._pending_answers[q_id] = event
            self._pending_values[q_id] = ""

            buttons = [
                [InlineKeyboardButton(text=opt, callback_data=f"{q_id}|{opt}")]
                for opt in options[:10]  # Telegram limit
            ]
            keyboard = InlineKeyboardMarkup(buttons)

            # Schedule message send (we're in sync context from agent)
            loop = asyncio.get_event_loop()
            try:
                # This is called from within an async context
                loop.run_until_complete(
                    self._app.bot.send_message(
                        chat_id=chat_id,
                        text=question,
                        reply_markup=keyboard,
                    )
                )
            except RuntimeError:
                # Already running — use ensure_future
                asyncio.ensure_future(
                    self._app.bot.send_message(
                        chat_id=chat_id,
                        text=question,
                        reply_markup=keyboard,
                    )
                )

            # Wait for callback
            try:
                loop.run_until_complete(
                    asyncio.wait_for(event.wait(), timeout=60)
                )
                return self._pending_values.get(q_id, "")
            except (asyncio.TimeoutError, RuntimeError):
                return ""
            finally:
                self._pending_answers.pop(q_id, None)
                self._pending_values.pop(q_id, None)

        return handler

    # ── Lifecycle ───────────────────────────────────────────────

    async def run(self) -> None:
        """Start the Telegram bot with long polling."""
        try:
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                CallbackQueryHandler,
                filters,
            )
        except ImportError:
            raise ImportError(
                "python-telegram-bot is required for Telegram mode. "
                "Install it with: pip install python-telegram-bot"
            )

        self._app = Application.builder().token(self.token).build()

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("clear", self._cmd_clear))
        self._app.add_handler(CommandHandler("history", self._cmd_history))
        self._app.add_handler(CommandHandler("tools", self._cmd_tools))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        self._app.add_handler(CallbackQueryHandler(self._handle_callback))

        logger.info("Starting Telegram bot (polling)...")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        try:
            # Run until interrupted
            stop_event = asyncio.Event()
            await stop_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
