"""
Chrome WebSocket Client — Sprint 46: Chrome Extension Bridge.

Low-level WebSocket JSON-RPC 2.0 client for communicating with the
Chrome extension MCP server.
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Optional websockets dependency
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    websockets = None
    HAS_WEBSOCKETS = False


@dataclass
class ChromeWSConfig:
    """WebSocket connection configuration."""
    url: str = "ws://localhost:9222"
    connect_timeout: float = 10.0
    request_timeout: float = 30.0
    auto_reconnect: bool = True
    max_reconnect_delay: float = 30.0
    heartbeat_interval: float = 15.0


class ChromeWSClient:
    """
    WebSocket JSON-RPC 2.0 client for Chrome extension communication.

    Sends method calls as JSON-RPC requests and matches responses by ID.
    Supports auto-reconnection and heartbeat pings.
    """

    def __init__(self, config: Optional[ChromeWSConfig] = None):
        self._config = config or ChromeWSConfig()
        self._ws = None  # WebSocket connection
        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._listen_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._connected = False
        self._reconnect_delay = 1.0
        self._closing = False

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket connection is currently active."""
        return self._connected and self._ws is not None

    async def connect(self) -> bool:
        """Establish WebSocket connection to Chrome extension."""
        if not HAS_WEBSOCKETS:
            logger.error("websockets library not installed. pip install websockets")
            return False

        try:
            self._closing = False
            self._ws = await asyncio.wait_for(
                websockets.connect(self._config.url),
                timeout=self._config.connect_timeout,
            )
            self._connected = True
            self._reconnect_delay = 1.0  # Reset on successful connect

            # Start background tasks
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            logger.info("Connected to Chrome extension at %s", self._config.url)
            return True

        except asyncio.TimeoutError:
            logger.error("Connection timeout after %.1fs to %s",
                        self._config.connect_timeout, self._config.url)
            return False
        except Exception as e:
            logger.error("Failed to connect to %s: %s", self._config.url, e)
            return False

    async def disconnect(self) -> None:
        """Close WebSocket connection cleanly."""
        self._closing = True
        self._connected = False

        # Cancel background tasks
        for task in [self._listen_task, self._heartbeat_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Close WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # Fail all pending requests
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionError("WebSocket disconnected"))
        self._pending.clear()

        logger.info("Disconnected from Chrome extension")

    async def call(self, method: str, params: Optional[dict] = None) -> dict:
        """
        Send a JSON-RPC request and await the response.

        Args:
            method: RPC method name (e.g., "browser/navigate")
            params: Method parameters

        Returns:
            Response result dict

        Raises:
            ConnectionError: If not connected
            TimeoutError: If response not received within timeout
            RuntimeError: If RPC returns an error
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Chrome extension")

        self._request_id += 1
        req_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        # Create future for response
        future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        try:
            await self._ws.send(json.dumps(request))
            result = await asyncio.wait_for(future, timeout=self._config.request_timeout)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"RPC call '{method}' timed out after {self._config.request_timeout}s")
        except Exception as e:
            self._pending.pop(req_id, None)
            raise

    async def send_notification(self, method: str, params: Optional[dict] = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Chrome extension")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        await self._ws.send(json.dumps(notification))

    async def _listen_loop(self) -> None:
        """Background task: read WebSocket messages and resolve pending futures."""
        try:
            async for raw_message in self._ws:
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON message: %s", raw_message[:100])
                    continue

                msg_id = message.get("id")
                if msg_id is None:
                    # Notification from extension (no ID)
                    logger.debug("Received notification: %s", message.get("method", "unknown"))
                    continue

                future = self._pending.pop(msg_id, None)
                if not future or future.done():
                    logger.warning("Received response for unknown/completed request %s", msg_id)
                    continue

                if "error" in message:
                    error = message["error"]
                    error_msg = error.get("message", "Unknown RPC error")
                    error_code = error.get("code", -1)
                    future.set_exception(
                        RuntimeError(f"RPC error {error_code}: {error_msg}")
                    )
                else:
                    future.set_result(message.get("result", {}))

        except asyncio.CancelledError:
            return
        except Exception as e:
            if not self._closing:
                logger.error("Listen loop error: %s", e)
                self._connected = False
                if self._config.auto_reconnect:
                    asyncio.create_task(self._reconnect())

    async def _heartbeat_loop(self) -> None:
        """Background task: send periodic pings to detect dead connections."""
        try:
            while self._connected:
                await asyncio.sleep(self._config.heartbeat_interval)
                if self._ws and self._connected:
                    try:
                        await self._ws.ping()
                    except Exception:
                        logger.warning("Heartbeat ping failed")
                        self._connected = False
                        if self._config.auto_reconnect:
                            asyncio.create_task(self._reconnect())
                        break
        except asyncio.CancelledError:
            return

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        if self._closing:
            return

        while not self._closing and not self._connected:
            logger.info("Reconnecting in %.1fs...", self._reconnect_delay)
            await asyncio.sleep(self._reconnect_delay)

            if await self.connect():
                logger.info("Reconnected successfully")
                return

            # Exponential backoff
            self._reconnect_delay = min(
                self._reconnect_delay * 2,
                self._config.max_reconnect_delay,
            )

        logger.info("Reconnection stopped (closing=%s, connected=%s)",
                    self._closing, self._connected)
