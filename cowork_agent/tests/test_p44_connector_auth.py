"""
Sprint 44 Tests — Connector Authentication & /connect CLI.

Tests the full connector auth lifecycle:
  - AuthConfig, AuthMethod, AuthStatus enums and dataclasses
  - StoredCredential creation, serialization, expiry
  - CredentialStore (save, load, delete, list)
  - ConnectorAuthManager (connect_with_token, connect_with_env, OAuth2, disconnect)
  - Auto-reconnect on startup
  - MCP env building from credentials
  - ConnectConnectorTool, DisconnectConnectorTool, ListConnectorsTool
  - CLI handler integration
  - Agent integration attributes
  - Edge cases
"""

import asyncio
import json
import os
import tempfile
import time
import shutil
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# ── Imports ──────────────────────────────────────────────────────────

from cowork_agent.core.connector_auth import (
    AuthMethod,
    AuthStatus,
    AuthConfig,
    StoredCredential,
    CredentialStore,
    ConnectorAuthManager,
    CONNECTOR_AUTH_CONFIGS,
)
from cowork_agent.core.connector_registry import (
    ConnectorInfo,
    ConnectorRegistry,
    create_default_connector_catalog,
)
from cowork_agent.tools.connector_tools import (
    ConnectConnectorTool,
    DisconnectConnectorTool,
    ListConnectorsTool,
)


# ── Helpers ──────────────────────────────────────────────────────────

def run(coro):
    """Run async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def make_registry_with_github():
    """Create a registry with a single GitHub connector."""
    reg = ConnectorRegistry()
    reg.register(ConnectorInfo(
        uuid="github-001",
        name="GitHub",
        description="Code hosting",
        keywords=["github", "git"],
    ))
    return reg


def make_registry_full():
    """Create the full default catalog."""
    return create_default_connector_catalog()


# ═══════════════════════════════════════════════════════════════════════
# 1. AuthMethod / AuthStatus / AuthConfig
# ═══════════════════════════════════════════════════════════════════════

class TestAuthEnums:
    """Test auth enums and config dataclass."""

    def test_auth_method_values(self):
        assert AuthMethod.OAUTH2.value == "oauth2"
        assert AuthMethod.API_TOKEN.value == "api_token"
        assert AuthMethod.ENV_VAR.value == "env_var"

    def test_auth_status_values(self):
        assert AuthStatus.NOT_CONFIGURED.value == "not_configured"
        assert AuthStatus.AUTHENTICATED.value == "authenticated"
        assert AuthStatus.EXPIRED.value == "expired"
        assert AuthStatus.PENDING.value == "pending"
        assert AuthStatus.FAILED.value == "failed"
        assert AuthStatus.REVOKED.value == "revoked"

    def test_auth_config_default(self):
        cfg = AuthConfig(method=AuthMethod.API_TOKEN)
        assert cfg.method == AuthMethod.API_TOKEN
        assert cfg.client_id == ""
        assert cfg.scopes == []
        assert cfg.redirect_uri == "http://localhost:9876/callback"

    def test_auth_config_oauth2(self):
        cfg = AuthConfig(
            method=AuthMethod.OAUTH2,
            client_id="my-id",
            auth_url="https://auth.example.com",
            token_url="https://token.example.com",
            scopes=["read", "write"],
        )
        assert cfg.method == AuthMethod.OAUTH2
        assert cfg.client_id == "my-id"
        assert len(cfg.scopes) == 2

    def test_auth_config_env_var(self):
        cfg = AuthConfig(
            method=AuthMethod.ENV_VAR,
            env_vars=["MY_TOKEN", "MY_SECRET"],
        )
        assert len(cfg.env_vars) == 2

    def test_builtin_configs_exist(self):
        """All 15 built-in connectors have auth configs."""
        assert len(CONNECTOR_AUTH_CONFIGS) == 15
        assert "github-001" in CONNECTOR_AUTH_CONFIGS
        assert "gmail-001" in CONNECTOR_AUTH_CONFIGS
        assert "slack-001" in CONNECTOR_AUTH_CONFIGS

    def test_github_config_is_api_token(self):
        cfg = CONNECTOR_AUTH_CONFIGS["github-001"]
        assert cfg.method == AuthMethod.API_TOKEN
        assert cfg.token_name == "Personal Access Token"
        assert cfg.token_env_var == "GITHUB_TOKEN"

    def test_gmail_config_is_oauth2(self):
        cfg = CONNECTOR_AUTH_CONFIGS["gmail-001"]
        assert cfg.method == AuthMethod.OAUTH2
        assert "google" in cfg.auth_url


# ═══════════════════════════════════════════════════════════════════════
# 2. StoredCredential
# ═══════════════════════════════════════════════════════════════════════

class TestStoredCredential:
    """Test StoredCredential dataclass."""

    def test_create_basic(self):
        cred = StoredCredential(
            connector_uuid="github-001",
            connector_name="GitHub",
            auth_method="api_token",
            tokens={"api_token": "ghp_abc123"},
            created_at=time.time(),
        )
        assert cred.connector_uuid == "github-001"
        assert cred.tokens["api_token"] == "ghp_abc123"

    def test_not_expired_by_default(self):
        cred = StoredCredential(
            connector_uuid="test",
            connector_name="Test",
            auth_method="api_token",
        )
        assert not cred.is_expired()

    def test_expired_when_past(self):
        cred = StoredCredential(
            connector_uuid="test",
            connector_name="Test",
            auth_method="oauth2",
            expires_at=time.time() - 100,
        )
        assert cred.is_expired()

    def test_not_expired_when_future(self):
        cred = StoredCredential(
            connector_uuid="test",
            connector_name="Test",
            auth_method="oauth2",
            expires_at=time.time() + 3600,
        )
        assert not cred.is_expired()

    def test_to_dict(self):
        cred = StoredCredential(
            connector_uuid="github-001",
            connector_name="GitHub",
            auth_method="api_token",
            tokens={"api_token": "ghp_abc"},
        )
        d = cred.to_dict()
        assert d["connector_uuid"] == "github-001"
        assert d["tokens"]["api_token"] == "ghp_abc"

    def test_from_dict(self):
        d = {
            "connector_uuid": "slack-001",
            "connector_name": "Slack",
            "auth_method": "api_token",
            "tokens": {"api_token": "xoxb-abc"},
            "created_at": 123456.0,
            "expires_at": 0.0,
            "last_used_at": 0.0,
            "status": "authenticated",
        }
        cred = StoredCredential.from_dict(d)
        assert cred.connector_uuid == "slack-001"
        assert cred.tokens["api_token"] == "xoxb-abc"

    def test_from_dict_ignores_extra_keys(self):
        d = {
            "connector_uuid": "test",
            "connector_name": "Test",
            "auth_method": "api_token",
            "tokens": {},
            "extra_field": "should be ignored",
            "created_at": 0.0,
            "expires_at": 0.0,
            "last_used_at": 0.0,
            "status": "authenticated",
        }
        cred = StoredCredential.from_dict(d)
        assert cred.connector_uuid == "test"
        assert not hasattr(cred, "extra_field")

    def test_roundtrip(self):
        cred = StoredCredential(
            connector_uuid="notion-001",
            connector_name="Notion",
            auth_method="api_token",
            tokens={"api_token": "secret_abc"},
            created_at=1000.0,
            expires_at=2000.0,
            last_used_at=1500.0,
            status="authenticated",
        )
        d = cred.to_dict()
        cred2 = StoredCredential.from_dict(d)
        assert cred2.connector_uuid == cred.connector_uuid
        assert cred2.tokens == cred.tokens
        assert cred2.expires_at == cred.expires_at


# ═══════════════════════════════════════════════════════════════════════
# 3. CredentialStore
# ═══════════════════════════════════════════════════════════════════════

class TestCredentialStore:
    """Test credential persistence to disk."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        cred = StoredCredential(
            connector_uuid="github-001",
            connector_name="GitHub",
            auth_method="api_token",
            tokens={"api_token": "ghp_test123"},
            created_at=time.time(),
            status="authenticated",
        )
        assert self.store.save(cred)
        loaded = self.store.load("github-001")
        assert loaded is not None
        assert loaded.connector_uuid == "github-001"
        assert loaded.tokens["api_token"] == "ghp_test123"

    def test_load_nonexistent(self):
        assert self.store.load("nonexistent") is None

    def test_delete(self):
        cred = StoredCredential(
            connector_uuid="slack-001",
            connector_name="Slack",
            auth_method="api_token",
            tokens={"api_token": "xoxb-test"},
        )
        self.store.save(cred)
        assert self.store.delete("slack-001")
        assert self.store.load("slack-001") is None

    def test_delete_nonexistent(self):
        assert not self.store.delete("nonexistent")

    def test_list_all(self):
        for i in range(3):
            cred = StoredCredential(
                connector_uuid=f"test-{i}",
                connector_name=f"Test {i}",
                auth_method="api_token",
                tokens={"api_token": f"tok-{i}"},
                status="authenticated",
            )
            self.store.save(cred)
        all_creds = self.store.list_all()
        assert len(all_creds) == 3

    def test_list_all_empty(self):
        assert self.store.list_all() == []

    def test_file_is_base64_encoded(self):
        cred = StoredCredential(
            connector_uuid="encoded-test",
            connector_name="Test",
            auth_method="api_token",
            tokens={"api_token": "secret"},
        )
        self.store.save(cred)
        path = self.store._file_path("encoded-test")
        with open(path) as f:
            raw = json.load(f)
        assert "v" in raw
        assert "data" in raw
        # The data should be base64 — not plaintext
        assert "secret" not in raw["data"]

    def test_save_creates_directory(self):
        nested = os.path.join(self.tmpdir, "sub", "dir")
        store = CredentialStore(credentials_dir=nested)
        cred = StoredCredential(
            connector_uuid="test",
            connector_name="Test",
            auth_method="api_token",
            tokens={},
        )
        assert store.save(cred)
        assert os.path.isdir(nested)

    def test_corrupt_file_returns_none(self):
        path = self.store._file_path("corrupt-test")
        with open(path, "w") as f:
            f.write("this is not json")
        assert self.store.load("corrupt-test") is None


# ═══════════════════════════════════════════════════════════════════════
# 4. ConnectorAuthManager
# ═══════════════════════════════════════════════════════════════════════

class TestConnectorAuthManager:
    """Test the auth manager lifecycle."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.mgr = ConnectorAuthManager(credential_store=self.store)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_auth_method(self):
        method = self.mgr.get_auth_method("github-001")
        assert method == AuthMethod.API_TOKEN

    def test_get_auth_method_unknown(self):
        assert self.mgr.get_auth_method("unknown-uuid") is None

    def test_connect_with_token(self):
        cred = self.mgr.connect_with_token(
            connector_uuid="github-001",
            connector_name="GitHub",
            token="ghp_abc123def456",
        )
        assert cred.connector_uuid == "github-001"
        assert cred.tokens["api_token"] == "ghp_abc123def456"
        assert self.mgr.is_connected("github-001")

    def test_connect_with_env(self):
        cred = self.mgr.connect_with_env(
            connector_uuid="searxng-001",
            connector_name="SearXNG",
            env_values={"SEARXNG_URL": "http://localhost:8888"},
        )
        assert cred.auth_method == "env_var"
        assert self.mgr.is_connected("searxng-001")

    def test_get_token(self):
        self.mgr.connect_with_token("github-001", "GitHub", "ghp_test")
        assert self.mgr.get_token("github-001") == "ghp_test"

    def test_get_token_not_connected(self):
        assert self.mgr.get_token("nonexistent") is None

    def test_disconnect(self):
        self.mgr.connect_with_token("github-001", "GitHub", "ghp_test")
        assert self.mgr.is_connected("github-001")
        assert self.mgr.disconnect("github-001")
        assert not self.mgr.is_connected("github-001")

    def test_disconnect_nonexistent(self):
        assert not self.mgr.disconnect("nonexistent")

    def test_load_saved_credentials(self):
        # Save a credential directly to store
        cred = StoredCredential(
            connector_uuid="slack-001",
            connector_name="Slack",
            auth_method="api_token",
            tokens={"api_token": "xoxb-saved"},
            created_at=time.time(),
            status="authenticated",
        )
        self.store.save(cred)

        # Create fresh manager and load
        mgr2 = ConnectorAuthManager(credential_store=self.store)
        saved = mgr2.load_saved_credentials()
        assert len(saved) == 1
        assert saved[0].connector_name == "Slack"
        assert mgr2.is_connected("slack-001")

    def test_load_skips_expired(self):
        cred = StoredCredential(
            connector_uuid="expired-001",
            connector_name="Expired",
            auth_method="oauth2",
            tokens={"access_token": "expired"},
            expires_at=time.time() - 100,
            status="authenticated",
        )
        self.store.save(cred)

        mgr2 = ConnectorAuthManager(credential_store=self.store)
        saved = mgr2.load_saved_credentials()
        assert len(saved) == 0
        assert not mgr2.is_connected("expired-001")

    def test_get_status_not_configured(self):
        assert self.mgr.get_status("unknown") == AuthStatus.NOT_CONFIGURED

    def test_get_status_authenticated(self):
        self.mgr.connect_with_token("github-001", "GitHub", "tok_valid_test_1234")
        assert self.mgr.get_status("github-001") == AuthStatus.AUTHENTICATED

    def test_get_status_expired(self):
        cred = StoredCredential(
            connector_uuid="exptest",
            connector_name="Test",
            auth_method="oauth2",
            tokens={"access_token": "tok_valid_test_1234"},
            expires_at=time.time() - 1,
            status="authenticated",
        )
        self.store.save(cred)
        assert self.mgr.get_status("exptest") == AuthStatus.NOT_CONFIGURED

    def test_list_all_statuses(self):
        self.mgr.connect_with_token("github-001", "GitHub", "tok_valid_test_1234")
        statuses = self.mgr.list_all_statuses(["github-001", "slack-001"])
        assert statuses["github-001"] == AuthStatus.AUTHENTICATED
        assert statuses["slack-001"] == AuthStatus.NOT_CONFIGURED

    def test_register_custom_auth_config(self):
        cfg = AuthConfig(
            method=AuthMethod.API_TOKEN,
            token_name="Custom Token",
            token_env_var="CUSTOM_TOKEN",
        )
        self.mgr.register_auth_config("custom-001", cfg)
        assert self.mgr.get_auth_method("custom-001") == AuthMethod.API_TOKEN

    def test_connected_connectors_property(self):
        self.mgr.connect_with_token("github-001", "GitHub", "tok1")
        self.mgr.connect_with_token("slack-001", "Slack", "tok2")
        connected = self.mgr.connected_connectors
        assert len(connected) == 2
        assert "github-001" in connected
        assert "slack-001" in connected

    def test_build_mcp_env_api_token(self):
        self.mgr.connect_with_token("github-001", "GitHub", "ghp_test")
        env = self.mgr.build_mcp_env("github-001")
        assert env.get("GITHUB_TOKEN") == "ghp_test"

    def test_build_mcp_env_env_var(self):
        self.mgr.connect_with_env(
            "custom-001", "Custom",
            {"MY_KEY": "val1", "MY_SECRET": "val2"},
        )
        env = self.mgr.build_mcp_env("custom-001")
        assert env["MY_KEY"] == "val1"
        assert env["MY_SECRET"] == "val2"

    def test_build_mcp_env_not_connected(self):
        assert self.mgr.build_mcp_env("nonexistent") == {}


# ═══════════════════════════════════════════════════════════════════════
# 5. OAuth2 Flow
# ═══════════════════════════════════════════════════════════════════════

class TestOAuth2Flow:
    """Test the OAuth2 authorization flow."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.mgr = ConnectorAuthManager(credential_store=self.store)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_initiate_oauth2(self):
        with patch.dict(os.environ, {"GMAIL_CLIENT_ID": "test-client-id"}):
            url = self.mgr.initiate_oauth2("gmail-001", "Gmail")
        assert "accounts.google.com" in url
        assert "test-client-id" in url
        assert "state=" in url

    def test_initiate_oauth2_with_explicit_client_id(self):
        url = self.mgr.initiate_oauth2(
            "gmail-001", "Gmail", client_id="explicit-id"
        )
        assert "explicit-id" in url

    def test_initiate_oauth2_no_client_id_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if present
            os.environ.pop("GMAIL_CLIENT_ID", None)
            with pytest.raises(ValueError, match="No client_id"):
                self.mgr.initiate_oauth2("gmail-001", "Gmail")

    def test_initiate_oauth2_wrong_method_raises(self):
        with pytest.raises(ValueError, match="No OAuth2 config"):
            self.mgr.initiate_oauth2("github-001", "GitHub")

    def test_complete_oauth2(self):
        with patch.dict(os.environ, {"GMAIL_CLIENT_ID": "test-id"}):
            url = self.mgr.initiate_oauth2("gmail-001", "Gmail")

        # Extract state from URL
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        state = params["state"][0]

        cred = self.mgr.complete_oauth2(
            state=state, code="auth_code_123", connector_name="Gmail"
        )
        assert cred.connector_uuid == "gmail-001"
        assert "access_token" in cred.tokens
        assert self.mgr.is_connected("gmail-001")

    def test_complete_oauth2_invalid_state(self):
        with pytest.raises(ValueError, match="Invalid or expired"):
            self.mgr.complete_oauth2(state="bad-state", code="code")

    def test_oauth2_build_mcp_env(self):
        with patch.dict(os.environ, {"GMAIL_CLIENT_ID": "test-id"}):
            url = self.mgr.initiate_oauth2("gmail-001", "Gmail")

        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        state = params["state"][0]

        self.mgr.complete_oauth2(state=state, code="code123", connector_name="Gmail")
        env = self.mgr.build_mcp_env("gmail-001")
        assert "OAUTH_ACCESS_TOKEN" in env


# ═══════════════════════════════════════════════════════════════════════
# 6. ConnectConnectorTool
# ═══════════════════════════════════════════════════════════════════════

class TestConnectConnectorTool:
    """Test the connect_connector tool."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.auth = ConnectorAuthManager(credential_store=self.store)
        self.registry = make_registry_full()
        self.tool = ConnectConnectorTool(
            auth_manager=self.auth,
            connector_registry=self.registry,
        )

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_connect_by_name_with_token(self):
        result = run(self.tool.execute(name="github", token="ghp_test123"))
        assert result.success
        assert "connected" in result.output.lower()
        assert self.auth.is_connected("github-001")

    def test_connect_by_uuid_with_token(self):
        result = run(self.tool.execute(uuid="slack-001", token="xoxb-test"))
        assert result.success
        assert "connected" in result.output.lower()

    def test_connect_already_connected(self):
        self.auth.connect_with_token("github-001", "GitHub", "tok_valid_test_1234")
        self.registry.mark_connected("github-001")
        result = run(self.tool.execute(name="github"))
        assert result.success
        assert "already connected" in result.output.lower()

    def test_connect_unknown_name(self):
        result = run(self.tool.execute(name="nonexistent_service"))
        assert not result.success

    def test_connect_no_name_or_uuid(self):
        result = run(self.tool.execute())
        assert not result.success

    def test_connect_no_auth_manager(self):
        tool = ConnectConnectorTool(auth_manager=None, connector_registry=self.registry)
        result = run(tool.execute(name="github"))
        assert not result.success

    def test_connect_needs_token_prompt(self):
        """When no token provided and no env var, tool prompts for it."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GITHUB_TOKEN", None)
            result = run(self.tool.execute(name="github"))
        assert result.success  # It returns instructions, not an error
        assert "requires" in result.output.lower() or "token" in result.output.lower()

    def test_connect_from_env_var(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_from_env"}):
            result = run(self.tool.execute(name="github"))
        assert result.success
        assert "connected" in result.output.lower()

    def test_connect_oauth2_returns_url(self):
        with patch.dict(os.environ, {"GMAIL_CLIENT_ID": "test-id"}):
            result = run(self.tool.execute(name="gmail"))
        assert result.success
        assert "oauth2" in result.output.lower() or "authorize" in result.output.lower()

    def test_connect_case_insensitive(self):
        result = run(self.tool.execute(name="GitHub", token="ghp_test"))
        assert result.success
        assert "connected" in result.output.lower()

    def test_connect_partial_match(self):
        result = run(self.tool.execute(name="git", token="ghp_test"))
        assert result.success  # Should match GitHub via search


# ═══════════════════════════════════════════════════════════════════════
# 7. DisconnectConnectorTool
# ═══════════════════════════════════════════════════════════════════════

class TestDisconnectConnectorTool:
    """Test the disconnect_connector tool."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.auth = ConnectorAuthManager(credential_store=self.store)
        self.registry = make_registry_full()
        self.tool = DisconnectConnectorTool(
            auth_manager=self.auth,
            connector_registry=self.registry,
        )

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_disconnect_connected(self):
        self.auth.connect_with_token("github-001", "GitHub", "tok_valid_test_1234")
        self.registry.mark_connected("github-001")
        result = run(self.tool.execute(name="github"))
        assert result.success
        assert "disconnected" in result.output.lower()
        assert not self.auth.is_connected("github-001")

    def test_disconnect_not_connected(self):
        result = run(self.tool.execute(name="github"))
        assert result.success
        assert "not currently connected" in result.output.lower()

    def test_disconnect_unknown(self):
        result = run(self.tool.execute(name="nonexistent"))
        assert not result.success

    def test_disconnect_by_uuid(self):
        self.auth.connect_with_token("slack-001", "Slack", "tok_valid_test_1234")
        result = run(self.tool.execute(uuid="slack-001"))
        assert result.success

    def test_disconnect_no_args(self):
        result = run(self.tool.execute())
        assert not result.success

    def test_disconnect_no_auth_manager(self):
        tool = DisconnectConnectorTool(auth_manager=None, connector_registry=self.registry)
        result = run(tool.execute(name="github"))
        assert not result.success


# ═══════════════════════════════════════════════════════════════════════
# 8. ListConnectorsTool
# ═══════════════════════════════════════════════════════════════════════

class TestListConnectorsTool:
    """Test the list_connectors tool."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.auth = ConnectorAuthManager(credential_store=self.store)
        self.registry = make_registry_full()
        self.tool = ListConnectorsTool(
            auth_manager=self.auth,
            connector_registry=self.registry,
        )

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_list_all(self):
        result = run(self.tool.execute())
        assert result.success
        assert "15" in result.output or "available" in result.output.lower()

    def test_list_connected_empty(self):
        result = run(self.tool.execute(filter="connected"))
        assert result.success

    def test_list_connected_with_some(self):
        self.auth.connect_with_token("github-001", "GitHub", "tok_valid_test_1234")
        self.registry.mark_connected("github-001")
        result = run(self.tool.execute(filter="connected"))
        assert result.success
        assert "github" in result.output.lower()

    def test_list_available(self):
        result = run(self.tool.execute(filter="available"))
        assert result.success

    def test_list_no_registry(self):
        tool = ListConnectorsTool(auth_manager=self.auth, connector_registry=None)
        result = run(tool.execute())
        assert not result.success

    def test_list_shows_auth_method(self):
        result = run(self.tool.execute())
        assert result.success
        assert "api_token" in result.output or "oauth2" in result.output


# ═══════════════════════════════════════════════════════════════════════
# 9. Agent Integration
# ═══════════════════════════════════════════════════════════════════════

class TestAgentIntegration:
    """Test that Sprint 44 attributes exist on the Agent."""

    def test_agent_has_connector_auth_attribute(self):
        from cowork_agent.core.agent import Agent
        from cowork_agent.core.tool_registry import ToolRegistry
        from cowork_agent.core.prompt_builder import PromptBuilder

        mock_provider = MagicMock()
        registry = ToolRegistry()
        pb = PromptBuilder({})
        agent = Agent(provider=mock_provider, registry=registry, prompt_builder=pb)
        assert hasattr(agent, "connector_auth")
        assert agent.connector_auth is None

    def test_agent_has_connector_registry_attribute(self):
        from cowork_agent.core.agent import Agent
        from cowork_agent.core.tool_registry import ToolRegistry
        from cowork_agent.core.prompt_builder import PromptBuilder

        mock_provider = MagicMock()
        registry = ToolRegistry()
        pb = PromptBuilder({})
        agent = Agent(provider=mock_provider, registry=registry, prompt_builder=pb)
        assert hasattr(agent, "connector_registry")
        assert agent.connector_registry is None

    def test_agent_can_receive_auth_manager(self):
        from cowork_agent.core.agent import Agent
        from cowork_agent.core.tool_registry import ToolRegistry
        from cowork_agent.core.prompt_builder import PromptBuilder

        mock_provider = MagicMock()
        registry = ToolRegistry()
        pb = PromptBuilder({})
        agent = Agent(provider=mock_provider, registry=registry, prompt_builder=pb)

        tmpdir = tempfile.mkdtemp()
        try:
            store = CredentialStore(credentials_dir=tmpdir)
            auth = ConnectorAuthManager(credential_store=store)
            agent.connector_auth = auth
            assert agent.connector_auth is auth
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_tools_register_in_registry(self):
        from cowork_agent.core.tool_registry import ToolRegistry
        tmpdir = tempfile.mkdtemp()
        try:
            store = CredentialStore(credentials_dir=tmpdir)
            auth = ConnectorAuthManager(credential_store=store)
            conn_reg = make_registry_full()
            tool_reg = ToolRegistry()

            tool_reg.register(ConnectConnectorTool(auth, conn_reg))
            tool_reg.register(DisconnectConnectorTool(auth, conn_reg))
            tool_reg.register(ListConnectorsTool(auth, conn_reg))

            assert "connect_connector" in tool_reg.tool_names
            assert "disconnect_connector" in tool_reg.tool_names
            assert "list_connectors" in tool_reg.tool_names
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
# 10. Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.mgr = ConnectorAuthManager(credential_store=self.store)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_multiple_connect_disconnect_cycles(self):
        for _ in range(3):
            self.mgr.connect_with_token("github-001", "GitHub", "tok_valid_test_1234")
            assert self.mgr.is_connected("github-001")
            self.mgr.disconnect("github-001")
            assert not self.mgr.is_connected("github-001")

    def test_overwrite_credential(self):
        self.mgr.connect_with_token("github-001", "GitHub", "old_token")
        self.mgr.connect_with_token("github-001", "GitHub", "new_token")
        assert self.mgr.get_token("github-001") == "new_token"

    def test_concurrent_connectors(self):
        """Connect multiple connectors at once."""
        self.mgr.connect_with_token("github-001", "GitHub", "tok1")
        self.mgr.connect_with_token("slack-001", "Slack", "tok2")
        self.mgr.connect_with_token("notion-001", "Notion", "tok3")
        assert len(self.mgr.connected_connectors) == 3

    def test_credential_with_special_chars(self):
        """Token with special characters saves and loads correctly."""
        token = "ghp_!@#$%^&*()=+[]{}|;:',.<>?/`~"
        self.mgr.connect_with_token("github-001", "GitHub", token)
        assert self.mgr.get_token("github-001") == token

    def test_empty_token_map(self):
        cred = StoredCredential(
            connector_uuid="empty",
            connector_name="Empty",
            auth_method="api_token",
            tokens={},
            status="authenticated",
        )
        self.store.save(cred)
        loaded = self.store.load("empty")
        assert loaded is not None
        assert loaded.tokens == {}

    def test_uuid_with_slashes_rejected(self):
        """Sprint 45: UUIDs with slashes are now rejected for security."""
        cred = StoredCredential(
            connector_uuid="ns/sub/id",
            connector_name="Namespaced",
            auth_method="api_token",
            tokens={"api_token": "tok_valid_test_1234"},
        )
        # Sprint 45 hardening: slashes in UUID rejected (path traversal prevention)
        result = self.store.save(cred)
        assert result is False  # save fails gracefully
        loaded = self.store.load("ns-sub-id")  # safe UUID format
        assert loaded is None  # nothing was saved

    def test_auth_config_scopes_preserved(self):
        cfg = AuthConfig(
            method=AuthMethod.OAUTH2,
            scopes=["read", "write", "admin"],
        )
        assert cfg.scopes == ["read", "write", "admin"]

    def test_credential_store_default_dir(self):
        """Default dir is ~/.cowork_agent/credentials."""
        store = CredentialStore()
        expected = os.path.join(
            os.path.expanduser("~"), ".cowork_agent", "credentials"
        )
        assert store._dir == expected
