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
from urllib.parse import urlparse

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

    # Security constants
    MAX_PENDING_REQUESTS = 1000
    MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_RECONNECT_ATTEMPTS = 10

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
        self._reconnect_count = 0
        self._reconnect_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket connection is currently active."""
        return self._connected and self._ws is not None

    def _validate_ws_url(self, url: str) -> None:
        """
        Validate WebSocket URL for security.

        Args:
            url: WebSocket URL to validate

        Raises:
            ValueError: If URL uses unencrypted ws:// for non-localhost hosts
        """
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        scheme = parsed.scheme

        # Check if ws:// is used for non-localhost
        if scheme == "ws" and hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(
                f"WSS required for non-localhost connections. "
                f"Use wss:// instead of ws:// for host: {hostname}"
            )

    def _validate_message(self, data: dict) -> bool:
        """
        Validate JSON-RPC 2.0 message structure.

        Args:
            data: Parsed JSON message dict

        Returns:
            True if message is valid

        Raises:
            ValueError: If message does not conform to JSON-RPC 2.0
        """
        # Must have jsonrpc: "2.0"
        if data.get("jsonrpc") != "2.0":
            raise ValueError(f'Message missing or invalid "jsonrpc" field: {data}')

        # ID must be int or str if present
        msg_id = data.get("id")
        if msg_id is not None and not isinstance(msg_id, (int, str)):
            raise ValueError(f"Invalid message ID type: {type(msg_id).__name__}")

        return True

    def _cleanup_stale_requests(self) -> None:
        """Remove futures older than 60 seconds from pending requests."""
        current_time = time.time()
        stale_ids = []

        for req_id, future in self._pending.items():
            # Check if future has a creation timestamp attribute, otherwise skip cleanup
            if hasattr(future, "_created_at"):
                if current_time - future._created_at > 60:  # type: ignore
                    stale_ids.append(req_id)

        for req_id in stale_ids:
            future = self._pending.pop(req_id, None)
            if future and not future.done():
                future.set_exception(TimeoutError("Request stale (>60s)"))
                logger.warning("Cleaned up stale request %s", req_id)

    async def connect(self) -> bool:
        """Establish WebSocket connection to Chrome extension."""
        if not HAS_WEBSOCKETS:
            logger.error("websockets library not installed. pip install websockets")
            return False

        try:
            # C-1: Validate WebSocket URL (TLS for non-localhost)
            self._validate_ws_url(self._config.url)

            self._closing = False
            # M-1: Add Origin header validation
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self._config.url,
                    additional_headers={"Origin": "cowork-agent"}
                ),
                timeout=self._config.connect_timeout,
            )
            self._connected = True
            self._reconnect_delay = 1.0  # Reset on successful connect
            self._reconnect_count = 0  # H-3: Reset reconnect counter

            # Start background tasks
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            logger.info("Connected to Chrome extension at %s", self._config.url)
            return True

        except asyncio.TimeoutError:
            logger.error("Connection timeout after %.1fs to %s",
                        self._config.connect_timeout, self._config.url)
            return False
        except ValueError as e:
            # H-2: Don't leak details about URL validation failures
            logger.debug("URL validation error: %s", e)
            logger.error("Failed to connect to extension: invalid configuration")
            return False
        except Exception as e:
            # H-2: Log detailed error at debug level only
            logger.debug("Connection error details: %s", e)
            logger.error("Failed to connect to extension")
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
            RuntimeError: If RPC returns an error or limit exceeded
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Chrome extension")

        # C-2: Check pending requests limit
        if len(self._pending) >= self.MAX_PENDING_REQUESTS:
            raise RuntimeError(
                f"Too many pending requests ({len(self._pending)} >= {self.MAX_PENDING_REQUESTS})"
            )

        # L-1: Use modular arithmetic for request ID to prevent overflow
        self._request_id = (self._request_id + 1) % (2**31)
        req_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        # Create future for response
        future = asyncio.get_event_loop().create_future()
        # Track creation time for stale request cleanup
        future._created_at = time.time()  # type: ignore
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

        # H-4: Handle send errors at actual send site instead of pre-check
        try:
            await self._ws.send(json.dumps(notification))
        except (ConnectionError, Exception) as e:
            # H-2: Log detailed error at debug level only
            logger.debug("Send notification error: %s", e)
            self._connected = False
            if self._config.auto_reconnect:
                asyncio.create_task(self._reconnect())
            raise ConnectionError("Failed to send notification") from e

    async def _listen_loop(self) -> None:
        """Background task: read WebSocket messages and resolve pending futures."""
        try:
            async for raw_message in self._ws:
                # H-1: Check message size before parsing
                if len(raw_message) > self.MAX_MESSAGE_SIZE:
                    logger.error("Message exceeds size limit: %d bytes", len(raw_message))
                    continue

                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON message")
                    continue

                # H-1: Validate JSON-RPC 2.0 structure
                try:
                    self._validate_message(message)
                except ValueError as e:
                    logger.debug("Invalid JSON-RPC message: %s", e)
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
                # H-2: Log detailed error at debug level only
                logger.debug("Listen loop error details: %s", e)
                logger.error("Listen loop error")
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
        """Attempt to reconnect with exponential backoff (max attempts: H-3)."""
        # H-3: Use lock to prevent concurrent reconnects
        async with self._reconnect_lock:
            if self._closing:
                return

            while not self._closing and not self._connected:
                # H-3: Check reconnect attempt limit
                if self._reconnect_count >= self.MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        "Max reconnection attempts (%d) reached. Giving up.",
                        self.MAX_RECONNECT_ATTEMPTS
                    )
                    return

                logger.info(
                    "Reconnecting (attempt %d/%d) in %.1fs...",
                    self._reconnect_count + 1,
                    self.MAX_RECONNECT_ATTEMPTS,
                    self._reconnect_delay
                )
                await asyncio.sleep(self._reconnect_delay)

                self._reconnect_count += 1
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
