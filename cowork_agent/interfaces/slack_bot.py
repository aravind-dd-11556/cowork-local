"""
Slack Bot Interface — Control the cowork agent from Slack.

Uses slack-bolt with Socket Mode (no webhook setup needed).
Each Slack user gets their own agent session. Tool indicators
and ask_user responses are threaded.

Usage::

    python -m cowork_agent --mode slack \\
        --slack-token xoxb-... --slack-app-token xapp-...

Required Slack app permissions:
    Bot Token Scopes: chat:write, app_mentions:read, im:history, im:read
    Socket Mode: Enabled
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

from .base import BaseInterface
from ..core.agent import Agent
from ..core.models import ToolCall, ToolResult

logger = logging.getLogger(__name__)


class SlackBotInterface(BaseInterface):
    """Slack bot for remote agent control.

    Per-user sessions, threaded tool indicators, Block Kit for ask_user.
    """

    MAX_MESSAGE_LEN = 3000  # Slack limit is ~4000 but leave margin

    def __init__(
        self,
        agent: Agent,
        bot_token: str,
        app_token: str,
        agent_factory=None,
    ):
        super().__init__(agent)
        self.bot_token = bot_token
        self.app_token = app_token
        self._agent_factory = agent_factory or (lambda: agent)
        self._user_agents: dict[str, Agent] = {}
        self._pending_answers: dict[str, asyncio.Event] = {}
        self._pending_values: dict[str, str] = {}
        self._bolt_app = None

    def _get_agent(self, user_id: str) -> Agent:
        """Get or create an agent for a Slack user."""
        if user_id not in self._user_agents:
            self._user_agents[user_id] = self._agent_factory()
        return self._user_agents[user_id]

    # ── Message splitting ───────────────────────────────────────

    @staticmethod
    def split_message(text: str, max_len: int = 3000) -> list[str]:
        """Split a long message into Slack-safe chunks."""
        if len(text) <= max_len:
            return [text]
        parts: list[str] = []
        while text:
            if len(text) <= max_len:
                parts.append(text)
                break
            chunk = text[:max_len]
            last_nl = chunk.rfind("\n")
            if last_nl > max_len // 2:
                parts.append(text[:last_nl])
                text = text[last_nl + 1:]
            else:
                parts.append(chunk)
                text = text[max_len:]
        return parts

    # ── Message Handler ─────────────────────────────────────────

    async def _handle_message(self, body: dict, say, client) -> None:
        """Process an incoming Slack message."""
        event = body.get("event", {})
        user_id = event.get("user", "")
        channel = event.get("channel", "")
        text = event.get("text", "")
        thread_ts = event.get("thread_ts") or event.get("ts")
        subtype = event.get("subtype", "")

        if not text or not user_id:
            return

        # Skip bot messages and message edits/deletes
        if event.get("bot_id") or subtype in ("bot_message", "message_changed", "message_deleted"):
            return

        logger.info(f"Slack message from {user_id}: {text[:50]}...")
        agent = self._get_agent(user_id)

        # Post "thinking" message
        try:
            thinking = await say(
                text=":hourglass_flowing_sand: Processing...",
                thread_ts=thread_ts,
            )
            thinking_ts = thinking.get("ts", "") if thinking else ""
        except Exception as e:
            logger.error(f"Failed to post thinking message: {e}")
            thinking_ts = ""

        # Wire tool callbacks for this message
        tool_msgs: list[str] = []

        def on_tool_start(call: ToolCall) -> None:
            tool_msgs.append(f":hammer_and_wrench: `{call.name}` running...")

        def on_tool_end(call: ToolCall, result: ToolResult) -> None:
            icon = ":white_check_mark:" if result.success else ":x:"
            tool_msgs.append(f"{icon} `{call.name}` done")

        agent.on_tool_start = on_tool_start
        agent.on_tool_end = on_tool_end

        # Run agent
        try:
            response_parts: list[str] = []
            async for chunk in agent.run_stream(text):
                response_parts.append(chunk)
            response = "".join(response_parts)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            response = f"Error: {e}"

        # Post tool summary in thread
        if tool_msgs:
            try:
                await say(
                    text="\n".join(tool_msgs),
                    thread_ts=thread_ts,
                )
            except Exception as e:
                logger.error(f"Failed to post tool summary: {e}")

        # Update thinking message with response
        if response:
            parts = self.split_message(response)
            try:
                await client.chat_update(
                    channel=channel,
                    ts=thinking_ts,
                    text=parts[0],
                )
            except Exception:
                try:
                    await say(text=parts[0], thread_ts=thread_ts)
                except Exception as e:
                    logger.error(f"Failed to post response: {e}")

            # Additional parts in thread
            for part in parts[1:]:
                try:
                    await say(text=part, thread_ts=thread_ts)
                except Exception as e:
                    logger.error(f"Failed to post part: {e}")
        else:
            try:
                await client.chat_update(
                    channel=channel,
                    ts=thinking_ts,
                    text="(No response from agent)",
                )
            except Exception:
                pass

    # ── ask_user via Block Kit ──────────────────────────────────

    async def _handle_action(self, body: dict, ack, client) -> None:
        """Handle Block Kit button press for ask_user."""
        await ack() if asyncio.iscoroutinefunction(ack) else ack()

        actions = body.get("actions", [])
        if not actions:
            return

        action_id = actions[0].get("action_id", "")
        value = actions[0].get("value", "")

        if "|" in action_id:
            q_id = action_id.split("|")[0]
        else:
            q_id = action_id

        if q_id in self._pending_answers:
            self._pending_values[q_id] = value
            self._pending_answers[q_id].set()

        # Update the message to show selected option
        try:
            channel = body.get("channel", {}).get("id", "")
            ts = body.get("message", {}).get("ts", "")
            original_text = body.get("message", {}).get("text", "")
            if channel and ts:
                await client.chat_update(
                    channel=channel,
                    ts=ts,
                    text=f"{original_text}\n\nSelected: *{value}*",
                    blocks=[],  # Remove buttons
                )
        except Exception:
            pass

    def _make_ask_user_handler(self, channel: str, thread_ts: str):
        """Create an ask_user callback scoped to a Slack thread."""

        def handler(question: str, options: list[str]) -> str:
            """Send Block Kit buttons and wait for response."""
            q_id = f"q_{uuid.uuid4().hex[:8]}"
            event = asyncio.Event()
            self._pending_answers[q_id] = event
            self._pending_values[q_id] = ""

            # Build Block Kit
            blocks = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*{question}*"},
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": opt[:75]},
                            "value": opt,
                            "action_id": f"{q_id}|{i}",
                        }
                        for i, opt in enumerate(options[:5])  # Max 5 buttons
                    ],
                },
            ]

            try:
                self._bolt_app.client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text=question,
                    blocks=blocks,
                )
            except Exception as e:
                logger.warning("Failed to send ask_user to Slack: %s", e)
                return ""

            # Wait for callback
            loop = asyncio.get_event_loop()
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
        """Start the Slack bot with Socket Mode."""
        import re as _re

        try:
            from slack_bolt.async_app import AsyncApp
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )
        except ImportError:
            raise ImportError(
                "slack-bolt is required for Slack mode. "
                "Install it with: pip install slack-bolt"
            )

        self._bolt_app = AsyncApp(token=self.bot_token)

        # Register handlers
        @self._bolt_app.event("message")
        async def handle_message(body, say, client, logger: logging.Logger):
            """Handle incoming messages."""
            logger.info("Received message event")
            try:
                await self._handle_message(body, say, client)
            except Exception as e:
                logger.error(f"Error handling message: {e}", exc_info=True)

        @self._bolt_app.action(_re.compile(r"^q_.*"))
        async def handle_action(body, ack, client):
            """Handle Block Kit button presses."""
            await self._handle_action(body, ack, client)

        # Catch-all for unhandled events (debugging)
        @self._bolt_app.event("app_mention")
        async def handle_mention(body, say):
            """Handle @mentions in channels."""
            event = body.get("event", {})
            text = event.get("text", "")
            # Strip the bot mention from text
            import re
            text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
            if text:
                await self._handle_message(body, say, None)

        logger.info("Starting Slack bot (Socket Mode)...")
        handler = AsyncSocketModeHandler(self._bolt_app, self.app_token)
        await handler.connect_async()
        logger.info("Slack bot connected via Socket Mode")
        # Keep alive — wait forever until cancelled
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Slack bot shutting down...")
            await handler.close_async()
            raise
