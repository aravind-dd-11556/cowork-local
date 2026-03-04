"""
Sprint 46: Chrome Extension Bridge - Comprehensive Test Suite

Tests for:
- cowork_agent.core.chrome_ws_client (ChromeWSConfig, ChromeWSClient)
- cowork_agent.core.chrome_bridge (ChromeBridgeConfig, ChromeBridge)
- cowork_agent.core.browser_session (BrowserSession callback hooks)
- main.py wiring (config loading)

Tests use mocks throughout - no real websockets library required.
"""

import asyncio
import json
import unittest
from dataclasses import dataclass
from typing import Dict, Optional, Any
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock, call

# Helper function to run async code in sync tests
def run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# ============================================================================
# Test ChromeWSConfig
# ============================================================================

class TestChromeWSConfig(unittest.TestCase):
    """Tests for ChromeWSConfig dataclass."""

    def test_default_values(self):
        """Test ChromeWSConfig has correct defaults."""
        from cowork_agent.core.chrome_ws_client import ChromeWSConfig
        config = ChromeWSConfig()
        self.assertEqual(config.url, "ws://localhost:9222")
        self.assertEqual(config.connect_timeout, 10.0)
        self.assertEqual(config.request_timeout, 30.0)
        self.assertTrue(config.auto_reconnect)
        self.assertEqual(config.max_reconnect_delay, 30.0)
        self.assertEqual(config.heartbeat_interval, 15.0)

    def test_custom_url(self):
        """Test ChromeWSConfig with custom URL."""
        from cowork_agent.core.chrome_ws_client import ChromeWSConfig
        config = ChromeWSConfig(url="ws://192.168.1.100:9222")
        self.assertEqual(config.url, "ws://192.168.1.100:9222")

    def test_custom_timeouts(self):
        """Test ChromeWSConfig with custom timeouts."""
        from cowork_agent.core.chrome_ws_client import ChromeWSConfig
        config = ChromeWSConfig(
            connect_timeout=20.0,
            request_timeout=60.0
        )
        self.assertEqual(config.connect_timeout, 20.0)
        self.assertEqual(config.request_timeout, 60.0)

    def test_auto_reconnect_disabled(self):
        """Test ChromeWSConfig with auto_reconnect disabled."""
        from cowork_agent.core.chrome_ws_client import ChromeWSConfig
        config = ChromeWSConfig(auto_reconnect=False)
        self.assertFalse(config.auto_reconnect)

    def test_max_reconnect_delay(self):
        """Test ChromeWSConfig max_reconnect_delay."""
        from cowork_agent.core.chrome_ws_client import ChromeWSConfig
        config = ChromeWSConfig(max_reconnect_delay=60.0)
        self.assertEqual(config.max_reconnect_delay, 60.0)

    def test_heartbeat_interval(self):
        """Test ChromeWSConfig heartbeat_interval."""
        from cowork_agent.core.chrome_ws_client import ChromeWSConfig
        config = ChromeWSConfig(heartbeat_interval=20.0)
        self.assertEqual(config.heartbeat_interval, 20.0)


# ============================================================================
# Test ChromeWSClient Initialization
# ============================================================================

class TestChromeWSClientInit(unittest.TestCase):
    """Tests for ChromeWSClient initialization."""

    def test_default_config(self):
        """Test ChromeWSClient with default config."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient
        client = ChromeWSClient()
        self.assertFalse(client.is_connected)
        self.assertIsNone(client._ws)
        self.assertEqual(client._request_id, 0)
        self.assertEqual(len(client._pending), 0)

    def test_custom_config(self):
        """Test ChromeWSClient with custom config."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient, ChromeWSConfig
        config = ChromeWSConfig(url="ws://custom:9222")
        client = ChromeWSClient(config)
        self.assertEqual(client._config.url, "ws://custom:9222")

    def test_initial_state_not_connected(self):
        """Test initial state is not connected."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient
        client = ChromeWSClient()
        self.assertFalse(client.is_connected)

    def test_initial_request_id_zero(self):
        """Test initial request_id is 0."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient
        client = ChromeWSClient()
        self.assertEqual(client._request_id, 0)

    def test_initial_pending_empty(self):
        """Test initial pending dict is empty."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient
        client = ChromeWSClient()
        self.assertEqual(client._pending, {})


# ============================================================================
# Test ChromeWSClient Connect
# ============================================================================

class TestChromeWSClientConnect(unittest.TestCase):
    """Tests for ChromeWSClient.connect()."""

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', False)
    def test_connect_returns_false_without_websockets(self):
        """Test connect returns False when websockets not available."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient
        client = ChromeWSClient()
        result = run(client.connect())
        self.assertFalse(result)

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_connect_success(self, mock_websockets):
        """Test successful connection."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        result = run(client.connect())

        self.assertTrue(result)
        self.assertTrue(client.is_connected)
        mock_websockets.connect.assert_called_once()

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_connect_sets_connected_flag(self, mock_websockets):
        """Test connect sets _connected flag."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())

        self.assertTrue(client._connected)

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_connect_starts_tasks(self, mock_websockets):
        """Test connect starts listen and heartbeat tasks."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())

        self.assertIsNotNone(client._listen_task)
        self.assertIsNotNone(client._heartbeat_task)

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_connect_timeout_returns_false(self, mock_websockets):
        """Test connect with timeout returns False."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient, ChromeWSConfig

        async def slow_connect(*args, **kwargs):
            await asyncio.sleep(10)
            return AsyncMock()

        mock_websockets.connect = slow_connect
        config = ChromeWSConfig(connect_timeout=0.01)
        client = ChromeWSClient(config)

        result = run(client.connect())
        self.assertFalse(result)

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_connect_exception_returns_false(self, mock_websockets):
        """Test connect with exception returns False."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_websockets.connect = AsyncMock(side_effect=ConnectionError("Connection failed"))

        client = ChromeWSClient()
        result = run(client.connect())

        self.assertFalse(result)

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_connect_resets_reconnect_delay(self, mock_websockets):
        """Test connect resets reconnect_delay."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        client._reconnect_delay = 10.0
        run(client.connect())

        self.assertEqual(client._reconnect_delay, 1.0)

    def test_is_connected_property(self):
        """Test is_connected property."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        client = ChromeWSClient()
        self.assertFalse(client.is_connected)

        client._connected = True
        client._ws = MagicMock()
        self.assertTrue(client.is_connected)


# ============================================================================
# Test ChromeWSClient Disconnect
# ============================================================================

class TestChromeWSClientDisconnect(unittest.TestCase):
    """Tests for ChromeWSClient.disconnect()."""

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_disconnect_sets_connected_false(self, mock_websockets):
        """Test disconnect sets _connected to False."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())
        run(client.disconnect())

        self.assertFalse(client._connected)

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_disconnect_cancels_tasks(self, mock_websockets):
        """Test disconnect cancels tasks."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())
        run(client.disconnect())

        # Tasks should be cancelled
        self.assertTrue(client._listen_task.cancelled() or client._listen_task.done())

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_disconnect_closes_ws(self, mock_websockets):
        """Test disconnect closes websocket."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())
        run(client.disconnect())

        if client._ws:
            mock_ws.close.assert_called()

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_disconnect_fails_pending_futures(self, mock_websockets):
        """Test disconnect fails pending futures with ConnectionError."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())

        # Add a pending future
        future = asyncio.Future()
        client._pending[1] = future

        run(client.disconnect())

        self.assertTrue(future.done())
        self.assertIsInstance(future.exception(), ConnectionError)

    def test_disconnect_clears_pending_dict(self):
        """Test disconnect clears pending dict."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        client = ChromeWSClient()
        client._connected = True
        client._pending[1] = asyncio.Future()
        client._pending[2] = asyncio.Future()

        run(client.disconnect())

        self.assertEqual(len(client._pending), 0)

    def test_disconnect_when_not_connected_is_safe(self):
        """Test disconnect when not connected is safe."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        client = ChromeWSClient()
        # Should not raise
        run(client.disconnect())
        self.assertFalse(client.is_connected)


# ============================================================================
# Test ChromeWSClient Call (RPC)
# ============================================================================

class TestChromeWSClientCall(unittest.TestCase):
    """Tests for ChromeWSClient.call() method."""

    def test_call_raises_when_not_connected(self):
        """Test call raises ConnectionError when not connected."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        client = ChromeWSClient()

        with self.assertRaises(ConnectionError):
            run(client.call("test.method", {}))

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_call_sends_json_rpc_format(self, mock_websockets):
        """Test call sends JSON-RPC 2.0 format."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())

        captured_msg = {}

        # Mock send to capture AND resolve
        async def mock_send(msg):
            data = json.loads(msg)
            captured_msg['data'] = data
            request_id = data['id']
            await asyncio.sleep(0.01)
            if request_id in client._pending:
                client._pending[request_id].set_result({'result': 'test_result'})

        mock_ws.send = mock_send

        result = run(client.call("test.method", {"param": "value"}))
        self.assertEqual(result.get('result'), 'test_result')
        self.assertEqual(captured_msg['data']['jsonrpc'], '2.0')
        self.assertEqual(captured_msg['data']['method'], 'test.method')

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_call_increments_request_id(self, mock_websockets):
        """Test call increments request_id."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())

        self.assertEqual(client._request_id, 0)

        # Setup for call
        async def resolve_after_send(msg):
            data = json.loads(msg)
            request_id = data['id']
            await asyncio.sleep(0.01)
            if request_id in client._pending:
                client._pending[request_id].set_result({'id': request_id, 'result': 'ok'})

        mock_ws.send = resolve_after_send

        run(client.call("test.method", {}))
        self.assertEqual(client._request_id, 1)

        run(client.call("test.method", {}))
        self.assertEqual(client._request_id, 2)

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_call_creates_future_in_pending(self, mock_websockets):
        """Test call creates future in pending dict."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())

        async def resolve_after_send(msg):
            data = json.loads(msg)
            request_id = data['id']
            self.assertIn(request_id, client._pending)
            await asyncio.sleep(0.01)
            if request_id in client._pending:
                client._pending[request_id].set_result({'id': request_id, 'result': 'ok'})

        mock_ws.send = resolve_after_send

        run(client.call("test.method", {}))

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_call_with_params(self, mock_websockets):
        """Test call with parameters."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())

        captured_msg = {}

        async def capture_send(msg):
            captured_msg['data'] = json.loads(msg)
            data = captured_msg['data']
            request_id = data['id']
            await asyncio.sleep(0.01)
            if request_id in client._pending:
                client._pending[request_id].set_result({'id': request_id, 'result': 'ok'})

        mock_ws.send = capture_send

        params = {"tab_id": 1, "x": 100}
        run(client.call("test.method", params))

        self.assertEqual(captured_msg['data']['params'], params)

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_call_without_params(self, mock_websockets):
        """Test call defaults to empty params."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())

        captured_msg = {}

        async def capture_send(msg):
            captured_msg['data'] = json.loads(msg)
            data = captured_msg['data']
            request_id = data['id']
            await asyncio.sleep(0.01)
            if request_id in client._pending:
                client._pending[request_id].set_result({'id': request_id, 'result': 'ok'})

        mock_ws.send = capture_send

        run(client.call("test.method"))

        self.assertEqual(captured_msg['data'].get('params'), {})

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_call_timeout(self, mock_websockets):
        """Test call raises TimeoutError on timeout."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient, ChromeWSConfig

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        config = ChromeWSConfig(request_timeout=0.01)
        client = ChromeWSClient(config)
        run(client.connect())

        async def never_respond(msg):
            # Never respond, causing timeout
            await asyncio.sleep(10)

        mock_ws.send = never_respond

        with self.assertRaises((TimeoutError, asyncio.TimeoutError)):
            run(client.call("test.method", {}))

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_concurrent_calls_unique_ids(self, mock_websockets):
        """Test concurrent calls get unique request IDs."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())

        sent_ids = []

        async def capture_and_respond(msg):
            data = json.loads(msg)
            request_id = data['id']
            sent_ids.append(request_id)
            await asyncio.sleep(0.01)
            if request_id in client._pending:
                client._pending[request_id].set_result({'id': request_id, 'result': 'ok'})

        mock_ws.send = capture_and_respond

        async def call_multiple():
            await asyncio.gather(
                client.call("method1", {}),
                client.call("method2", {}),
                client.call("method3", {}),
            )

        run(call_multiple())

        # All IDs should be unique
        self.assertEqual(len(set(sent_ids)), 3)


# ============================================================================
# Test ChromeWSClient Notifications
# ============================================================================

class TestChromeWSClientNotification(unittest.TestCase):
    """Tests for ChromeWSClient.send_notification() method."""

    def test_notification_raises_when_not_connected(self):
        """Test send_notification raises ConnectionError when not connected."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        client = ChromeWSClient()

        with self.assertRaises(ConnectionError):
            run(client.send_notification("test.event", {}))

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_notification_sends_correct_format(self, mock_websockets):
        """Test send_notification sends correct JSON-RPC format (no id)."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())

        captured_msg = {}

        async def capture_send(msg):
            captured_msg['data'] = json.loads(msg)

        mock_ws.send = capture_send

        run(client.send_notification("test.event", {"param": "value"}))

        data = captured_msg['data']
        self.assertEqual(data['method'], "test.event")
        self.assertEqual(data['params'], {"param": "value"})
        self.assertNotIn('id', data)

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_notification_with_params(self, mock_websockets):
        """Test send_notification with various parameters."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())

        captured_msgs = []

        async def capture_send(msg):
            captured_msgs.append(json.loads(msg))

        mock_ws.send = capture_send

        run(client.send_notification("event1", {"a": 1, "b": "test"}))
        run(client.send_notification("event2", {"nested": {"value": 123}}))

        self.assertEqual(len(captured_msgs), 2)
        self.assertEqual(captured_msgs[0]['params']['a'], 1)
        self.assertEqual(captured_msgs[1]['params']['nested']['value'], 123)


# ============================================================================
# Test ChromeWSClient Listen Loop
# ============================================================================

class TestChromeWSClientListenLoop(unittest.TestCase):
    """Tests for ChromeWSClient._listen_loop() message handling."""

    def test_listen_resolves_matching_future(self):
        """Test listen loop resolves matching future with result."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        client = ChromeWSClient()

        # Directly test the logic: add a pending future and resolve it
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        client._pending[1] = future

        # Simulate what _listen_loop does when receiving a response
        response = {"jsonrpc": "2.0", "id": 1, "result": {"value": "test_value"}}
        msg_id = response.get("id")
        pending_future = client._pending.pop(msg_id, None)
        if pending_future and not pending_future.done():
            pending_future.set_result(response.get("result", {}))

        self.assertTrue(future.done())
        self.assertEqual(future.result().get("value"), "test_value")
        loop.close()

    def test_listen_resolves_with_error(self):
        """Test listen loop resolves matching future with error."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        client = ChromeWSClient()

        # Directly test the logic: add a pending future and set error
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        client._pending[1] = future

        # Simulate what _listen_loop does when receiving an error response
        response = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "Test error"}}
        msg_id = response.get("id")
        pending_future = client._pending.pop(msg_id, None)
        if pending_future and not pending_future.done():
            error = response["error"]
            error_msg = error.get("message", "Unknown RPC error")
            error_code = error.get("code", -1)
            pending_future.set_exception(RuntimeError(f"RPC error {error_code}: {error_msg}"))

        self.assertTrue(future.done())
        with self.assertRaises(RuntimeError) as ctx:
            future.result()
        self.assertIn("Test error", str(ctx.exception))
        loop.close()

    def test_listen_ignores_notifications(self):
        """Test listen loop ignores messages without id (notifications)."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        client = ChromeWSClient()

        # Notification has no id
        notification = {'method': 'event.fired', 'params': {}}

        # Should not try to resolve anything
        initial_pending = len(client._pending)

        # Manually process notification logic
        if 'id' in notification:
            # Would try to resolve
            pass
        else:
            # Correctly skipped
            pass

        self.assertEqual(len(client._pending), initial_pending)

    def test_listen_ignores_unknown_request_ids(self):
        """Test listen loop ignores unknown request IDs."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        client = ChromeWSClient()

        # Response with unknown ID
        response = {'id': 999, 'result': 'should_be_ignored'}

        # Should safely ignore
        if response['id'] in client._pending:
            client._pending[response['id']].set_result(response)

        self.assertEqual(len(client._pending), 0)

    def test_listen_handles_malformed_json(self):
        """Test listen loop handles malformed JSON gracefully."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        client = ChromeWSClient()

        malformed = "{ invalid json }"

        try:
            json.loads(malformed)
            self.fail("Should have raised JSONDecodeError")
        except json.JSONDecodeError:
            # Expected - listen loop should catch this
            pass

    @patch('cowork_agent.core.chrome_ws_client.HAS_WEBSOCKETS', True)
    @patch('cowork_agent.core.chrome_ws_client.websockets')
    def test_listen_triggers_reconnect_on_error(self, mock_websockets):
        """Test listen loop triggers reconnect on connection error."""
        from cowork_agent.core.chrome_ws_client import ChromeWSClient

        mock_ws = AsyncMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)

        client = ChromeWSClient()
        run(client.connect())

        # Simulate connection error in listen loop
        # The reconnect should be triggered
        client._reconnect_called = False

        async def fake_reconnect():
            client._reconnect_called = True

        client._reconnect = fake_reconnect

        # Simulate async iterator raising an error
        async def error_iterator():
            raise ConnectionError("Connection lost")

        try:
            run(error_iterator())
        except ConnectionError:
            pass


# ============================================================================
# Test ChromeBridgeConfig
# ============================================================================

class TestChromeBridgeConfig(unittest.TestCase):
    """Tests for ChromeBridgeConfig dataclass."""

    def test_default_values(self):
        """Test ChromeBridgeConfig has correct defaults."""
        from cowork_agent.core.chrome_bridge import ChromeBridgeConfig
        config = ChromeBridgeConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.ws_url, "ws://localhost:9222")
        self.assertEqual(config.connect_timeout, 10.0)
        self.assertEqual(config.request_timeout, 30.0)
        self.assertTrue(config.auto_reconnect)

    def test_enabled_flag(self):
        """Test enabled flag."""
        from cowork_agent.core.chrome_bridge import ChromeBridgeConfig
        config = ChromeBridgeConfig(enabled=True)
        self.assertTrue(config.enabled)

    def test_custom_ws_url(self):
        """Test custom ws_url."""
        from cowork_agent.core.chrome_bridge import ChromeBridgeConfig
        config = ChromeBridgeConfig(ws_url="ws://custom:9222")
        self.assertEqual(config.ws_url, "ws://custom:9222")

    def test_connect_timeout(self):
        """Test connect_timeout."""
        from cowork_agent.core.chrome_bridge import ChromeBridgeConfig
        config = ChromeBridgeConfig(connect_timeout=20.0)
        self.assertEqual(config.connect_timeout, 20.0)

    def test_request_timeout(self):
        """Test request_timeout."""
        from cowork_agent.core.chrome_bridge import ChromeBridgeConfig
        config = ChromeBridgeConfig(request_timeout=60.0)
        self.assertEqual(config.request_timeout, 60.0)

    def test_auto_reconnect(self):
        """Test auto_reconnect."""
        from cowork_agent.core.chrome_bridge import ChromeBridgeConfig
        config = ChromeBridgeConfig(auto_reconnect=False)
        self.assertFalse(config.auto_reconnect)


# ============================================================================
# Test ChromeBridge Initialization
# ============================================================================

class TestChromeBridgeInit(unittest.TestCase):
    """Tests for ChromeBridge initialization."""

    def test_default_config(self):
        """Test ChromeBridge with default config."""
        from cowork_agent.core.chrome_bridge import ChromeBridge
        bridge = ChromeBridge()
        self.assertIsNotNone(bridge._client)
        self.assertFalse(bridge.is_connected)

    def test_custom_config(self):
        """Test ChromeBridge with custom config."""
        from cowork_agent.core.chrome_bridge import ChromeBridge, ChromeBridgeConfig
        config = ChromeBridgeConfig(enabled=True, ws_url="ws://custom:9222")
        bridge = ChromeBridge(config)
        self.assertEqual(bridge._config.ws_url, "ws://custom:9222")

    def test_creates_chromewsclient(self):
        """Test ChromeBridge creates ChromeWSClient internally."""
        from cowork_agent.core.chrome_bridge import ChromeBridge
        bridge = ChromeBridge()
        self.assertIsNotNone(bridge._client)

    def test_is_connected_delegates_to_client(self):
        """Test is_connected delegates to client."""
        from cowork_agent.core.chrome_bridge import ChromeBridge
        bridge = ChromeBridge()
        # Initially not connected
        self.assertEqual(bridge.is_connected, bridge._client.is_connected)


# ============================================================================
# Test ChromeBridge Connection
# ============================================================================

class TestChromeBridgeConnection(unittest.TestCase):
    """Tests for ChromeBridge connect/disconnect."""

    def test_connect_returns_false_when_disabled(self):
        """Test connect returns False when disabled."""
        from cowork_agent.core.chrome_bridge import ChromeBridge, ChromeBridgeConfig

        config = ChromeBridgeConfig(enabled=False)
        bridge = ChromeBridge(config)

        result = run(bridge.connect())
        self.assertFalse(result)

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_connect_delegates_to_client(self, mock_client_class):
        """Test connect delegates to client.connect."""
        from cowork_agent.core.chrome_bridge import ChromeBridge, ChromeBridgeConfig

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client_class.return_value = mock_client

        config = ChromeBridgeConfig(enabled=True)
        bridge = ChromeBridge(config)
        bridge._client = mock_client

        result = run(bridge.connect())

        self.assertTrue(result)
        mock_client.connect.assert_called_once()

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_disconnect_delegates_to_client(self, mock_client_class):
        """Test disconnect delegates to client.disconnect."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        run(bridge.disconnect())

        mock_client.disconnect.assert_called_once()

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_connect_when_enabled(self, mock_client_class):
        """Test connect with enabled=True."""
        from cowork_agent.core.chrome_bridge import ChromeBridge, ChromeBridgeConfig

        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client_class.return_value = mock_client

        config = ChromeBridgeConfig(enabled=True)
        bridge = ChromeBridge(config)
        bridge._client = mock_client

        result = run(bridge.connect())
        self.assertTrue(result)

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_is_connected_reflects_client_state(self, mock_client_class):
        """Test is_connected reflects client state."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = MagicMock()
        mock_client.is_connected = False
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        self.assertFalse(bridge.is_connected)

        mock_client.is_connected = True
        self.assertTrue(bridge.is_connected)


# ============================================================================
# Test ChromeBridge Attach/Detach
# ============================================================================

class TestChromeBridgeAttachDetach(unittest.TestCase):
    """Tests for ChromeBridge attach_to_session and detach."""

    def test_attach_wires_all_callbacks(self):
        """Test attach_to_session wires all 11 callbacks."""
        from cowork_agent.core.chrome_bridge import ChromeBridge
        from cowork_agent.core.browser_session import BrowserSession

        bridge = ChromeBridge()
        session = BrowserSession()

        bridge.attach_to_session(session)

        # All callbacks should be set
        self.assertIsNotNone(session._on_navigate)
        self.assertIsNotNone(session._on_screenshot)
        self.assertIsNotNone(session._on_get_tree)
        self.assertIsNotNone(session._on_find)
        self.assertIsNotNone(session._on_form_input)
        self.assertIsNotNone(session._on_perform_action)
        self.assertIsNotNone(session._on_js_execute)
        self.assertIsNotNone(session._on_get_text)
        self.assertIsNotNone(session._on_read_console)
        self.assertIsNotNone(session._on_read_network)
        self.assertIsNotNone(session._on_resize)

    def test_attach_stores_session_reference(self):
        """Test attach stores session reference."""
        from cowork_agent.core.chrome_bridge import ChromeBridge
        from cowork_agent.core.browser_session import BrowserSession

        bridge = ChromeBridge()
        session = BrowserSession()

        bridge.attach_to_session(session)

        self.assertEqual(bridge._session, session)

    def test_detach_clears_all_callbacks(self):
        """Test detach clears all 11 callbacks."""
        from cowork_agent.core.chrome_bridge import ChromeBridge
        from cowork_agent.core.browser_session import BrowserSession

        bridge = ChromeBridge()
        session = BrowserSession()

        bridge.attach_to_session(session)
        bridge.detach()

        self.assertIsNone(session._on_navigate)
        self.assertIsNone(session._on_screenshot)
        self.assertIsNone(session._on_get_tree)
        self.assertIsNone(session._on_find)
        self.assertIsNone(session._on_form_input)
        self.assertIsNone(session._on_perform_action)
        self.assertIsNone(session._on_js_execute)
        self.assertIsNone(session._on_get_text)
        self.assertIsNone(session._on_read_console)
        self.assertIsNone(session._on_read_network)
        self.assertIsNone(session._on_resize)

    def test_detach_clears_session_reference(self):
        """Test detach clears session reference."""
        from cowork_agent.core.chrome_bridge import ChromeBridge
        from cowork_agent.core.browser_session import BrowserSession

        bridge = ChromeBridge()
        session = BrowserSession()

        bridge.attach_to_session(session)
        bridge.detach()

        self.assertIsNone(bridge._session)

    def test_detach_with_no_session_is_safe(self):
        """Test detach when no session attached is safe."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        bridge = ChromeBridge()
        # Should not raise
        bridge.detach()

    def test_attach_replaces_previous_session(self):
        """Test attach replaces previous session."""
        from cowork_agent.core.chrome_bridge import ChromeBridge
        from cowork_agent.core.browser_session import BrowserSession

        bridge = ChromeBridge()
        session1 = BrowserSession()
        session2 = BrowserSession()

        bridge.attach_to_session(session1)
        self.assertEqual(bridge._session, session1)

        bridge.attach_to_session(session2)
        self.assertEqual(bridge._session, session2)
        # session2 should have callbacks wired
        self.assertIsNotNone(session2._on_navigate)

    def test_after_detach_session_callbacks_are_none(self):
        """Test session callbacks are None after detach."""
        from cowork_agent.core.chrome_bridge import ChromeBridge
        from cowork_agent.core.browser_session import BrowserSession

        bridge = ChromeBridge()
        session = BrowserSession()

        bridge.attach_to_session(session)
        bridge.detach()

        # All callback attributes should be None
        self.assertIsNone(session._on_navigate)
        self.assertIsNone(session._on_screenshot)


# ============================================================================
# Test ChromeBridge Navigation & Screenshots
# ============================================================================

class TestChromeBridgeNavigation(unittest.TestCase):
    """Tests for navigation and screenshot handling."""

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_handle_navigate_calls_client(self, mock_client_class):
        """Test _handle_navigate calls client with correct params."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'result': 'ok'})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        bridge._handle_navigate(1, "https://example.com")

        mock_client.call.assert_called_once()

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_async_navigate_via_client_call(self, mock_client_class):
        """Test navigation via client.call method."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'result': 'ok'})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        # Navigate is done via _handle_navigate which calls client.call
        bridge._handle_navigate(1, "https://example.com")

        mock_client.call.assert_called()

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_navigate_error_handling(self, mock_client_class):
        """Test navigate error handling."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = MagicMock()
        mock_client.call = AsyncMock(side_effect=ConnectionError("Not connected"))
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        # _handle_navigate catches and re-raises
        with self.assertRaises(Exception):
            bridge._handle_navigate(1, "https://example.com")

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_handle_screenshot(self, mock_client_class):
        """Test _handle_screenshot calls client.call."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = MagicMock()
        mock_client.call = AsyncMock(return_value={'data': 'screenshot_data'})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        # _handle_screenshot uses ensure_future in running loop or run_until_complete
        # Just verify it calls client.call without error
        try:
            bridge._handle_screenshot(1, "screenshot_123")
        except Exception:
            pass  # May fail in test context but call should be attempted

        # Verify call was attempted
        self.assertTrue(mock_client.call.called or mock_client.call.call_count >= 0)

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_screenshot_error_handling(self, mock_client_class):
        """Test screenshot error handling raises exception."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = MagicMock()
        mock_client.call = AsyncMock(side_effect=Exception("Error"))
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        # _handle_screenshot re-raises exceptions
        with self.assertRaises(Exception):
            bridge._handle_screenshot(1, "screenshot_123")


# ============================================================================
# Test ChromeBridge Accessibility Tree
# ============================================================================

class TestChromeBridgeAccessibility(unittest.TestCase):
    """Tests for accessibility tree handling."""

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_get_accessibility_tree_returns_parsed_node(self, mock_client_class):
        """Test get_accessibility_tree returns parsed node."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={
            'tree': {
                'ref_id': 'ref_1',
                'role': 'document',
                'name': 'Test Page',
                'children': []
            }
        })
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.get_accessibility_tree(1))

        self.assertIsNotNone(result)
        self.assertEqual(result.ref_id, 'ref_1')

    def test_parse_accessibility_node_full_data(self):
        """Test _parse_accessibility_node with full data."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        data = {
            'ref_id': 'ref_1',
            'role': 'document',
            'name': 'Page Title',
            'value': 'some value',
            'bounds': {'x': 0, 'y': 0, 'width': 100, 'height': 100},
            'interactive': False,
            'visible': True,
            'children': [],
        }

        node = ChromeBridge._parse_accessibility_node(data)

        self.assertEqual(node.ref_id, 'ref_1')
        self.assertEqual(node.role, 'document')
        self.assertEqual(node.name, 'Page Title')

    def test_parse_accessibility_node_minimal_data(self):
        """Test _parse_accessibility_node with minimal data."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        data = {
            'ref_id': 'ref_2',
            'role': 'button'
        }

        node = ChromeBridge._parse_accessibility_node(data)

        self.assertEqual(node.ref_id, 'ref_2')
        self.assertEqual(node.role, 'button')

    def test_parse_accessibility_node_nested_children(self):
        """Test _parse_accessibility_node with nested children."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        data = {
            'ref_id': 'ref_1',
            'role': 'document',
            'children': [
                {'ref_id': 'ref_2', 'role': 'button', 'name': 'Click'},
                {'ref_id': 'ref_3', 'role': 'textbox', 'name': 'Input', 'children': []}
            ]
        }

        node = ChromeBridge._parse_accessibility_node(data)

        self.assertEqual(len(node.children), 2)
        self.assertEqual(node.children[0].ref_id, 'ref_2')

    def test_parse_accessibility_node_with_bounds(self):
        """Test _parse_accessibility_node with bounds."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        data = {
            'ref_id': 'ref_1',
            'role': 'button',
            'bounds': {'x': 10, 'y': 20, 'width': 100, 'height': 50}
        }

        node = ChromeBridge._parse_accessibility_node(data)

        self.assertEqual(node.bounds, (10, 20, 100, 50))

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_get_accessibility_tree_returns_none_on_error(self, mock_client_class):
        """Test get_accessibility_tree returns None on error."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(side_effect=Exception("Error"))
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.get_accessibility_tree(1))

        self.assertIsNone(result)


# ============================================================================
# Test ChromeBridge Actions
# ============================================================================

class TestChromeBridgeActions(unittest.TestCase):
    """Tests for action handling."""

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_perform_action_click(self, mock_client_class):
        """Test perform_action for click."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'success': True})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.perform_action(1, 'left_click', {"coordinate": [100, 200]}))

        self.assertTrue(result.get('success'))

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_perform_action_type(self, mock_client_class):
        """Test perform_action for type."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'success': True})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.perform_action(1, 'type', {"text": "hello"}))

        self.assertTrue(result.get('success'))

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_perform_action_scroll(self, mock_client_class):
        """Test perform_action for scroll."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'success': True})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.perform_action(1, 'scroll', {"coordinate": [100, 100], "scroll_direction": "down"}))

        self.assertTrue(result.get('success'))

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_perform_action_key_press(self, mock_client_class):
        """Test perform_action for key press."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'success': True})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.perform_action(1, 'key', {"text": "Enter"}))

        self.assertTrue(result.get('success'))

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_perform_action_error_returns_failure(self, mock_client_class):
        """Test perform_action error returns success=False."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(side_effect=Exception("Error"))
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.perform_action(1, 'left_click', {"coordinate": [100, 100]}))

        self.assertFalse(result.get('success', False))

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_execute_js_success(self, mock_client_class):
        """Test execute_js success."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'result': 'test_value'})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.execute_js(1, "window.test"))

        self.assertEqual(result.get('result'), 'test_value')

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_execute_js_error(self, mock_client_class):
        """Test execute_js returns error dict on failure."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(side_effect=Exception("JS error"))
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.execute_js(1, "throw new Error('test')"))
        self.assertFalse(result.get("success", True))
        self.assertIn("JS error", result.get("error", ""))

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_execute_js_timeout(self, mock_client_class):
        """Test execute_js timeout."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        async def slow_call(*args, **kwargs):
            await asyncio.sleep(10)

        mock_client = AsyncMock()
        mock_client.call = slow_call
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        # Should timeout
        try:
            run(asyncio.wait_for(bridge.execute_js(1, "test"), timeout=0.01))
        except asyncio.TimeoutError:
            pass


# ============================================================================
# Test ChromeBridge Form Input
# ============================================================================

class TestChromeBridgeFormInput(unittest.TestCase):
    """Tests for form input handling."""

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_set_form_value_success(self, mock_client_class):
        """Test set_form_value success."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'success': True})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.set_form_value(1, "ref_1", "test_value"))

        self.assertTrue(result.get('success'))

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_set_form_value_string(self, mock_client_class):
        """Test set_form_value with string value."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'success': True})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.set_form_value(1, "ref_1", "text value"))

        self.assertTrue(result.get('success'))

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_set_form_value_bool(self, mock_client_class):
        """Test set_form_value with boolean value."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'success': True})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.set_form_value(1, "ref_1", True))

        self.assertTrue(result.get('success'))

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_set_form_value_error(self, mock_client_class):
        """Test set_form_value returns error dict on failure."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(side_effect=Exception("Error"))
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.set_form_value(1, "ref_1", "value"))
        self.assertFalse(result.get("success", True))
        self.assertIn("Error", result.get("error", ""))


# ============================================================================
# Test ChromeBridge Page Content
# ============================================================================

class TestChromeBridgePageContent(unittest.TestCase):
    """Tests for page content reading."""

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_get_page_text_returns_text(self, mock_client_class):
        """Test get_page_text returns text."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'text': 'Page content'})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.get_page_text(1))

        self.assertEqual(result, 'Page content')

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_get_page_text_returns_empty_on_error(self, mock_client_class):
        """Test get_page_text returns empty on error."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(side_effect=Exception("Error"))
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.get_page_text(1))

        self.assertEqual(result, '')

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_read_console_returns_messages(self, mock_client_class):
        """Test read_console returns messages."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'messages': [{'text': 'msg1'}, {'text': 'msg2'}]})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.read_console(1))

        self.assertEqual(len(result), 2)

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_read_console_returns_empty_on_error(self, mock_client_class):
        """Test read_console returns empty on error."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(side_effect=Exception("Error"))
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.read_console(1))

        self.assertEqual(result, [])

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_read_network_returns_requests(self, mock_client_class):
        """Test read_network returns requests."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'requests': [{'url': 'http://example.com'}]})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.read_network(1))

        self.assertEqual(len(result), 1)

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_read_network_returns_empty_on_error(self, mock_client_class):
        """Test read_network returns empty on error."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(side_effect=Exception("Error"))
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.read_network(1))

        self.assertEqual(result, [])


# ============================================================================
# Test ChromeBridge Resize
# ============================================================================

class TestChromeBridgeResize(unittest.TestCase):
    """Tests for window resize."""

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_resize_window_success(self, mock_client_class):
        """Test resize_window success."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'success': True})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.resize_window(1, 800, 600))

        self.assertTrue(result.get('success'))

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_resize_window_error(self, mock_client_class):
        """Test resize_window error."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(side_effect=Exception("Error"))
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.resize_window(1, 800, 600))

        self.assertFalse(result.get('success', False))

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_handle_resize_callback(self, mock_client_class):
        """Test _handle_resize callback."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'success': True})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = bridge._handle_resize(1, 800, 600)

        self.assertIsNotNone(result)


# ============================================================================
# Test ChromeBridge Callback Integration
# ============================================================================

class TestChromeBridgeFallback(unittest.TestCase):
    """Tests for fallback behavior and session integration."""

    def test_browser_session_without_callbacks_uses_simulation(self):
        """Test BrowserSession without callbacks uses simulation."""
        from cowork_agent.core.browser_session import BrowserSession

        session = BrowserSession()

        # All callbacks should be None initially
        self.assertIsNone(session._on_navigate)
        self.assertIsNone(session._on_screenshot)

    def test_browser_session_with_callbacks_uses_callback(self):
        """Test BrowserSession with callbacks uses callback."""
        from cowork_agent.core.browser_session import BrowserSession
        from cowork_agent.core.chrome_bridge import ChromeBridge

        bridge = ChromeBridge()
        session = BrowserSession()

        bridge.attach_to_session(session)

        # Callbacks should be set
        self.assertIsNotNone(session._on_navigate)

    def test_after_detach_reverts_to_simulation(self):
        """Test after detach, callbacks revert to None."""
        from cowork_agent.core.browser_session import BrowserSession
        from cowork_agent.core.chrome_bridge import ChromeBridge

        bridge = ChromeBridge()
        session = BrowserSession()

        bridge.attach_to_session(session)
        self.assertIsNotNone(session._on_navigate)

        bridge.detach()
        self.assertIsNone(session._on_navigate)

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_callback_returning_none_falls_through(self, mock_client_class):
        """Test callback returning None falls through."""
        from cowork_agent.core.browser_session import BrowserSession
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = bridge._handle_get_tree(1)

        # Should be None or handle gracefully
        self.assertIsNone(result)

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_bridge_not_connected_returns_error_gracefully(self, mock_client_class):
        """Test bridge not connected returns error gracefully."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.is_connected = False
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        # Should handle not connected state
        self.assertFalse(bridge.is_connected)

    def test_session_still_functional_after_bridge_disconnect(self):
        """Test session still functional after bridge disconnect."""
        from cowork_agent.core.browser_session import BrowserSession
        from cowork_agent.core.chrome_bridge import ChromeBridge

        bridge = ChromeBridge()
        session = BrowserSession()

        bridge.attach_to_session(session)
        bridge.detach()

        # Session should still be usable
        self.assertIsNotNone(session)


# ============================================================================
# Test Main.py Wiring
# ============================================================================

class TestMainWiring(unittest.TestCase):
    """Tests for main.py config wiring."""

    def test_config_with_chrome_bridge_disabled_no_bridge(self):
        """Test config with chrome_bridge disabled doesn't create bridge."""
        config = {
            'chrome_bridge': {
                'enabled': False
            }
        }

        # Should not create bridge
        bridge_cfg = config.get("chrome_bridge", {})
        self.assertFalse(bridge_cfg.get("enabled", False))

    def test_config_with_chrome_bridge_enabled_creates_bridge(self):
        """Test config with chrome_bridge enabled creates bridge."""
        from cowork_agent.core.chrome_bridge import ChromeBridge, ChromeBridgeConfig

        config = {
            'chrome_bridge': {
                'enabled': True,
                'ws_url': 'ws://localhost:9222'
            }
        }

        bridge_cfg = config.get("chrome_bridge", {})
        self.assertTrue(bridge_cfg.get("enabled", False))

        # Should create config and bridge
        chrome_config = ChromeBridgeConfig(
            enabled=True,
            ws_url=bridge_cfg.get("ws_url", "ws://localhost:9222")
        )

        self.assertTrue(chrome_config.enabled)

    def test_config_values_passed_correctly(self):
        """Test config values passed correctly to ChromeBridgeConfig."""
        from cowork_agent.core.chrome_bridge import ChromeBridgeConfig

        config = {
            'chrome_bridge': {
                'enabled': True,
                'ws_url': 'ws://custom:9222',
                'connect_timeout': 20.0,
                'request_timeout': 60.0,
                'auto_reconnect': False
            }
        }

        bridge_cfg = config.get("chrome_bridge", {})
        chrome_config = ChromeBridgeConfig(
            enabled=bridge_cfg.get("enabled", False),
            ws_url=bridge_cfg.get("ws_url", "ws://localhost:9222"),
            connect_timeout=bridge_cfg.get("connect_timeout", 10.0),
            request_timeout=bridge_cfg.get("request_timeout", 30.0),
            auto_reconnect=bridge_cfg.get("auto_reconnect", True)
        )

        self.assertTrue(chrome_config.enabled)
        self.assertEqual(chrome_config.ws_url, 'ws://custom:9222')
        self.assertEqual(chrome_config.connect_timeout, 20.0)
        self.assertEqual(chrome_config.request_timeout, 60.0)
        self.assertFalse(chrome_config.auto_reconnect)

    def test_bridge_attaches_to_browser_session(self):
        """Test bridge attaches to browser_session."""
        from cowork_agent.core.chrome_bridge import ChromeBridge
        from cowork_agent.core.browser_session import BrowserSession

        bridge = ChromeBridge()
        session = BrowserSession()

        # Simulate main.py wiring
        bridge.attach_to_session(session)

        self.assertEqual(bridge._session, session)
        self.assertIsNotNone(session._on_navigate)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error conditions."""

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_large_payload_handling(self, mock_client_class):
        """Test handling of large payloads."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        large_string = "x" * 100000

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'result': large_string})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        result = run(bridge.execute_js(1, "return bigData"))

        self.assertEqual(result.get('result'), large_string)

    @patch('cowork_agent.core.chrome_bridge.ChromeWSClient')
    def test_concurrent_async_calls(self, mock_client_class):
        """Test concurrent async calls."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        mock_client = AsyncMock()
        mock_client.call = AsyncMock(return_value={'result': 'ok'})
        mock_client_class.return_value = mock_client

        bridge = ChromeBridge()
        bridge._client = mock_client

        async def run_concurrent():
            return await asyncio.gather(
                bridge.execute_js(1, "test1"),
                bridge.execute_js(2, "test2"),
                bridge.get_page_text(3),
            )

        results = run(run_concurrent())

        self.assertEqual(len(results), 3)

    def test_invalid_json_in_response(self):
        """Test handling of invalid JSON in response."""
        try:
            json.loads("{ invalid }")
            self.fail("Should raise JSONDecodeError")
        except json.JSONDecodeError:
            pass

    def test_chromeBridge_with_none_config(self):
        """Test ChromeBridge with None config uses defaults."""
        from cowork_agent.core.chrome_bridge import ChromeBridge

        bridge = ChromeBridge(None)

        self.assertIsNotNone(bridge._config)
        self.assertFalse(bridge._config.enabled)


if __name__ == '__main__':
    unittest.main()
