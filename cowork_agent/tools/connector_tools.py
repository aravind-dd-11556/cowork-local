"""
Connector Authentication Tools — Sprint 44.

Provides CLI-friendly tools for managing connector authentication:
  - connect_connector: Authenticate a connector (/connect <name>)
  - disconnect_connector: Revoke a connector (/disconnect <name>)
  - list_connectors: Show all connectors with status (/connectors)

These tools make it easy for users to set up integrations inline
without editing config files or environment variables.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional

from .base import BaseTool

if TYPE_CHECKING:
    from ..core.connector_auth import ConnectorAuthManager, AuthMethod
    from ..core.connector_registry import ConnectorRegistry

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Tool 1: connect_connector
# ═══════════════════════════════════════════════════════════════════════

class ConnectConnectorTool(BaseTool):
    """
    Authenticate and connect to an external service.

    Handles three auth methods automatically:
      - OAuth2: Opens browser for authorization
      - API Token: Prompts user for token (masked)
      - Env Var: Reads from environment variables
    """
    name = "connect_connector"
    description = (
        "Connect to an external service by name or UUID. "
        "Handles authentication (API token, OAuth2, or env vars) "
        "and persists credentials for future sessions.\n\n"
        "Examples:\n"
        '- connect_connector(name="github")\n'
        '- connect_connector(name="slack", token="xoxb-...")\n'
        '- connect_connector(uuid="gmail-001")\n'
    )
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Service name to connect (e.g., 'github', 'slack', 'gmail'). "
                    "Case-insensitive. Either name or uuid is required."
                ),
            },
            "uuid": {
                "type": "string",
                "description": "Connector UUID if known (e.g., 'github-001').",
            },
            "token": {
                "type": "string",
                "description": (
                    "API token/key for the service. If not provided, "
                    "the tool will check environment variables."
                ),
            },
        },
        "required": [],
    }

    def __init__(
        self,
        auth_manager: Optional["ConnectorAuthManager"] = None,
        connector_registry: Optional["ConnectorRegistry"] = None,
    ):
        self._auth = auth_manager
        self._registry = connector_registry

    def _resolve_connector(self, name: str = "", uuid: str = ""):
        """Find a connector by name or UUID."""
        if not self._registry:
            return None, "No connector registry available."

        if uuid:
            conn = self._registry.get(uuid)
            if conn:
                return conn, ""
            return None, f"No connector found with UUID: {uuid}"

        if name:
            # Search by exact name match (case-insensitive)
            name_lower = name.lower().strip()
            for c in self._registry.all_connectors:
                if c.name.lower() == name_lower:
                    return c, ""
            # Try partial match
            matches = self._registry.search([name_lower])
            if matches:
                return matches[0], ""
            return None, (
                f"No connector found matching '{name}'. "
                f"Use /connectors to see available services."
            )

        return None, "Either 'name' or 'uuid' is required."

    async def execute(
        self, *, progress_callback=None,
        name="", uuid="", token="", **kwargs
    ) -> "ToolResult":
        if not self._auth:
            return self._error("Connector auth manager not available.")

        # Resolve connector
        conn, err = self._resolve_connector(name, uuid)
        if not conn:
            return self._error(err)

        # Check if already connected
        if self._auth.is_connected(conn.uuid):
            return self._success(
                f"✅ **{conn.name}** is already connected.\n"
                f"Tools: {', '.join(conn.tool_names) if conn.tool_names else 'N/A'}",
                already_connected=True,
                connector_uuid=conn.uuid,
            )

        # Get auth config
        auth_config = self._auth.get_auth_config(conn.uuid)
        if not auth_config:
            return self._error(
                f"No authentication configuration found for {conn.name}. "
                f"Please add auth config for UUID '{conn.uuid}' in settings."
            )

        # Handle based on auth method
        from ..core.connector_auth import AuthMethod
        method = auth_config.method

        if method == AuthMethod.API_TOKEN:
            return await self._connect_token(conn, auth_config, token)
        elif method == AuthMethod.OAUTH2:
            return await self._connect_oauth2(conn, auth_config, token)
        elif method == AuthMethod.ENV_VAR:
            return await self._connect_env(conn, auth_config)
        else:
            return self._error(f"Unsupported auth method: {method}")

    async def _connect_token(self, conn, auth_config, provided_token: str):
        """Connect using API token. Sprint 45: validates token, uses secure masking."""
        import os
        from ..core.connector_auth import validate_token, mask_token

        # Priority: provided token > env var
        token = provided_token
        if not token and auth_config.token_env_var:
            token = os.environ.get(auth_config.token_env_var, "")

        if not token:
            token_name = auth_config.token_name or "API Token"
            env_hint = (
                f"\n\nAlternatively, set the `{auth_config.token_env_var}` "
                f"environment variable."
            ) if auth_config.token_env_var else ""

            return self._success(
                f"🔑 **{conn.name}** requires a {token_name}.\n\n"
                f"Please provide your token:\n"
                f"  `/connect {conn.name.lower()} --token YOUR_TOKEN_HERE`"
                f"{env_hint}",
                needs_token=True,
                connector_uuid=conn.uuid,
                token_name=token_name,
            )

        # Sprint 45: Validate token before accepting
        try:
            validated = validate_token(token)
        except ValueError as e:
            return self._error(f"Invalid token: {e}")

        # Save credential
        try:
            cred = self._auth.connect_with_token(
                connector_uuid=conn.uuid,
                connector_name=conn.name,
                token=validated,
            )
        except ValueError as e:
            return self._error(f"Failed to connect: {e}")

        # Mark as connected in registry
        if self._registry:
            self._registry.mark_connected(conn.uuid)

        # Sprint 45: Use secure masking
        masked = mask_token(validated)
        return self._success(
            f"✅ **{conn.name}** connected successfully!\n"
            f"Token: {masked}\n"
            f"Credentials saved for future sessions.",
            connected=True,
            connector_uuid=conn.uuid,
            connector_name=conn.name,
        )

    async def _connect_oauth2(self, conn, auth_config, client_id: str):
        """Initiate OAuth2 connection."""
        try:
            auth_url = self._auth.initiate_oauth2(
                connector_uuid=conn.uuid,
                connector_name=conn.name,
                client_id=client_id,
            )
            return self._success(
                f"🌐 **{conn.name}** requires OAuth2 authentication.\n\n"
                f"Please open this URL in your browser to authorize:\n"
                f"  {auth_url}\n\n"
                f"After authorizing, you'll be redirected back with a code.\n"
                f"Provide the code to complete the connection.",
                needs_oauth=True,
                connector_uuid=conn.uuid,
                auth_url=auth_url,
            )
        except ValueError as e:
            return self._error(str(e))

    async def _connect_env(self, conn, auth_config):
        """Connect using environment variables."""
        import os

        env_values = {}
        missing = []
        for var in auth_config.env_vars:
            val = os.environ.get(var, "")
            if val:
                env_values[var] = val
            else:
                missing.append(var)

        if missing:
            return self._success(
                f"⚠️ **{conn.name}** requires these environment variables:\n"
                + "\n".join(f"  - `{v}` (not set)" for v in missing)
                + "\n\nPlease set them and try again.",
                needs_env_vars=True,
                missing_vars=missing,
                connector_uuid=conn.uuid,
            )

        cred = self._auth.connect_with_env(
            connector_uuid=conn.uuid,
            connector_name=conn.name,
            env_values=env_values,
        )

        if self._registry:
            self._registry.mark_connected(conn.uuid)

        return self._success(
            f"✅ **{conn.name}** connected via environment variables.\n"
            f"Credentials saved for future sessions.",
            connected=True,
            connector_uuid=conn.uuid,
        )


# ═══════════════════════════════════════════════════════════════════════
# Tool 2: disconnect_connector
# ═══════════════════════════════════════════════════════════════════════

class DisconnectConnectorTool(BaseTool):
    """Disconnect a connector and revoke its stored credentials."""

    name = "disconnect_connector"
    description = (
        "Disconnect an external service and remove its stored credentials.\n\n"
        "Examples:\n"
        '- disconnect_connector(name="github")\n'
        '- disconnect_connector(uuid="slack-001")\n'
    )
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Service name to disconnect (case-insensitive).",
            },
            "uuid": {
                "type": "string",
                "description": "Connector UUID to disconnect.",
            },
        },
        "required": [],
    }

    def __init__(
        self,
        auth_manager: Optional["ConnectorAuthManager"] = None,
        connector_registry: Optional["ConnectorRegistry"] = None,
    ):
        self._auth = auth_manager
        self._registry = connector_registry

    def _resolve_connector(self, name: str = "", uuid: str = ""):
        """Find a connector by name or UUID."""
        if not self._registry:
            return None, "No connector registry available."
        if uuid:
            conn = self._registry.get(uuid)
            if conn:
                return conn, ""
            return None, f"No connector found with UUID: {uuid}"
        if name:
            name_lower = name.lower().strip()
            for c in self._registry.all_connectors:
                if c.name.lower() == name_lower:
                    return c, ""
            return None, f"No connector found matching '{name}'."
        return None, "Either 'name' or 'uuid' is required."

    async def execute(
        self, *, progress_callback=None, name="", uuid="", **kwargs
    ) -> "ToolResult":
        if not self._auth:
            return self._error("Connector auth manager not available.")

        conn, err = self._resolve_connector(name, uuid)
        if not conn:
            return self._error(err)

        if not self._auth.is_connected(conn.uuid):
            return self._success(
                f"ℹ️ **{conn.name}** is not currently connected.",
                already_disconnected=True,
                connector_uuid=conn.uuid,
            )

        success = self._auth.disconnect(conn.uuid)
        if self._registry:
            self._registry.mark_disconnected(conn.uuid)

        if success:
            return self._success(
                f"✅ **{conn.name}** disconnected. Stored credentials removed.",
                disconnected=True,
                connector_uuid=conn.uuid,
            )
        else:
            return self._error(
                f"Failed to disconnect {conn.name}. Please try again."
            )


# ═══════════════════════════════════════════════════════════════════════
# Tool 3: list_connectors
# ═══════════════════════════════════════════════════════════════════════

class ListConnectorsTool(BaseTool):
    """List all available connectors and their connection status."""

    name = "list_connectors"
    description = (
        "List all available connectors with their connection status, "
        "auth method, and available tools.\n\n"
        "Use this to show the user what services are available and "
        "which ones are currently connected."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "string",
                "enum": ["all", "connected", "available"],
                "description": (
                    "Filter connectors: 'all' (default), 'connected' (active only), "
                    "'available' (not connected only)."
                ),
            },
        },
        "required": [],
    }

    def __init__(
        self,
        auth_manager: Optional["ConnectorAuthManager"] = None,
        connector_registry: Optional["ConnectorRegistry"] = None,
    ):
        self._auth = auth_manager
        self._registry = connector_registry

    async def execute(
        self, *, progress_callback=None, filter="all", **kwargs
    ) -> "ToolResult":
        if not self._registry:
            return self._error("No connector registry available.")

        # Get connectors based on filter
        if filter == "connected":
            connectors = self._registry.connected_connectors
        elif filter == "available":
            connectors = self._registry.available_connectors
        else:
            connectors = self._registry.all_connectors

        if not connectors:
            return self._success(
                "No connectors found." if filter == "all"
                else f"No {filter} connectors found.",
                count=0,
            )

        # Build table
        lines = []
        connected_count = 0
        available_count = 0

        for conn in connectors:
            is_connected = conn.connected or (
                self._auth and self._auth.is_connected(conn.uuid)
            )
            if is_connected:
                connected_count += 1
                status_icon = "✅"
                status_text = "Connected"
            else:
                available_count += 1
                status_icon = "⬜"
                status_text = "Available"

            # Get auth method if available
            auth_info = ""
            if self._auth:
                auth_cfg = self._auth.get_auth_config(conn.uuid)
                if auth_cfg:
                    auth_info = f" | Auth: {auth_cfg.method.value}"

            tools_text = ""
            if conn.tool_names:
                tools_text = f" | Tools: {', '.join(conn.tool_names[:3])}"
                if len(conn.tool_names) > 3:
                    tools_text += f" +{len(conn.tool_names) - 3} more"

            lines.append(
                f"  {status_icon} **{conn.name}** — {conn.description}\n"
                f"     Status: {status_text}{auth_info}{tools_text}"
            )

        header = (
            f"Connectors: {connected_count} connected, "
            f"{available_count} available\n"
        )
        separator = "─" * 50

        body = f"{header}{separator}\n" + "\n".join(lines)
        body += f"\n{separator}"
        body += "\n\nUse `/connect <name>` to connect a service."

        return self._success(
            body,
            total=len(connectors),
            connected=connected_count,
            available=available_count,
        )
