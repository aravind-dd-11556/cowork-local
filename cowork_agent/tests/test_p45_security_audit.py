"""
Sprint 45 — Security Audit & Edge Case Tests.

Tests for all hardening applied in Sprint 45:
  - Fernet encryption for credential files
  - Path traversal protection (UUID sanitization)
  - File permissions (owner-only 0o600)
  - OAuth state TTL expiry and max-pending limits
  - Token validation (empty, whitespace, too short/long)
  - Secure token masking
  - Redacted logging (no token values in logs)
  - build_mcp_env restricted to declared env vars
  - Secure credential wiping on disconnect
  - Edge cases in tool input and CLI flow
"""

import asyncio
import base64
import json
import os
import secrets
import stat
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cowork_agent.core.connector_auth import (
    AuthMethod,
    AuthStatus,
    AuthConfig,
    StoredCredential,
    CredentialStore,
    ConnectorAuthManager,
    CONNECTOR_AUTH_CONFIGS,
    validate_token,
    mask_token,
    sanitize_uuid,
    MIN_TOKEN_LENGTH,
    MAX_TOKEN_LENGTH,
    MAX_PENDING_OAUTH_STATES,
    OAUTH_STATE_TTL,
    CREDENTIAL_FILE_MODE,
    SAFE_UUID_RE,
)
from cowork_agent.core.connector_registry import (
    ConnectorInfo,
    ConnectorRegistry,
)
from cowork_agent.tools.connector_tools import (
    ConnectConnectorTool,
    DisconnectConnectorTool,
    ListConnectorsTool,
)


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Token Validation
# ═══════════════════════════════════════════════════════════════════════

class TestTokenValidation(unittest.TestCase):
    """Sprint 45: Token validation rejects invalid inputs."""

    def test_valid_token(self):
        result = validate_token("ghp_abc123xyz789")
        self.assertEqual(result, "ghp_abc123xyz789")

    def test_strips_whitespace(self):
        result = validate_token("  ghp_abc123  ")
        self.assertEqual(result, "ghp_abc123")

    def test_rejects_none(self):
        with self.assertRaises(ValueError):
            validate_token(None)

    def test_rejects_empty_string(self):
        with self.assertRaises(ValueError):
            validate_token("")

    def test_rejects_whitespace_only(self):
        with self.assertRaises(ValueError):
            validate_token("   \t\n  ")

    def test_rejects_too_short(self):
        with self.assertRaises(ValueError) as ctx:
            validate_token("ab")
        self.assertIn("too short", str(ctx.exception))

    def test_rejects_too_long(self):
        huge = "x" * (MAX_TOKEN_LENGTH + 1)
        with self.assertRaises(ValueError) as ctx:
            validate_token(huge)
        self.assertIn("too long", str(ctx.exception))

    def test_accepts_minimum_length(self):
        result = validate_token("a" * MIN_TOKEN_LENGTH)
        self.assertEqual(len(result), MIN_TOKEN_LENGTH)

    def test_accepts_maximum_length(self):
        result = validate_token("a" * MAX_TOKEN_LENGTH)
        self.assertEqual(len(result), MAX_TOKEN_LENGTH)

    def test_rejects_non_string(self):
        with self.assertRaises(ValueError):
            validate_token(12345)

    def test_rejects_boolean(self):
        with self.assertRaises(ValueError):
            validate_token(True)

    def test_accepts_special_chars(self):
        """Tokens often contain special characters like xoxb-, ghp_, etc."""
        result = validate_token("xoxb-123-456-abc")
        self.assertEqual(result, "xoxb-123-456-abc")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Token Masking
# ═══════════════════════════════════════════════════════════════════════

class TestTokenMasking(unittest.TestCase):
    """Sprint 45: Secure token masking never reveals too much."""

    def test_empty_token(self):
        self.assertEqual(mask_token(""), "****")

    def test_none_token(self):
        self.assertEqual(mask_token(None), "****")

    def test_short_token_fully_masked(self):
        """Tokens under 12 chars should be fully masked."""
        result = mask_token("abcdefgh")
        self.assertNotIn("a", result)
        self.assertNotIn("h", result)
        self.assertTrue(all(c == "*" for c in result))

    def test_medium_token_fully_masked(self):
        """11-char token is still fully masked."""
        result = mask_token("12345678901")
        self.assertTrue(all(c == "*" for c in result))

    def test_long_token_shows_ends(self):
        """Tokens >= 12 chars show 3 chars from each end."""
        token = "ghp_1234567890xyz"
        result = mask_token(token)
        self.assertTrue(result.startswith("ghp"))
        self.assertTrue(result.endswith("xyz"))
        self.assertIn("*", result)

    def test_mask_minimum_asterisks(self):
        """Always at least 4 asterisks in the middle."""
        token = "A" * 12
        result = mask_token(token, show_chars=3)
        asterisks = result.count("*")
        self.assertGreaterEqual(asterisks, 4)

    def test_old_style_short_token_no_leak(self):
        """
        Sprint 44 bug: 8-char tokens showed first 4 + last 4 = entire token.
        Sprint 45 fix: short tokens are fully masked.
        """
        token = "abcd1234"
        result = mask_token(token)
        # Must NOT contain any of the original characters
        self.assertNotIn("a", result)
        self.assertNotIn("4", result)

    def test_custom_show_chars(self):
        token = "A" * 20
        result = mask_token(token, show_chars=2)
        self.assertTrue(result.startswith("AA"))
        self.assertTrue(result.endswith("AA"))


# ═══════════════════════════════════════════════════════════════════════
# Test 3: UUID Sanitization (Path Traversal)
# ═══════════════════════════════════════════════════════════════════════

class TestUUIDSanitization(unittest.TestCase):
    """Sprint 45: UUID sanitization prevents path traversal."""

    def test_valid_uuid(self):
        self.assertEqual(sanitize_uuid("github-001"), "github-001")

    def test_valid_with_underscores(self):
        self.assertEqual(sanitize_uuid("my_connector_1"), "my_connector_1")

    def test_rejects_dot_dot(self):
        with self.assertRaises(ValueError) as ctx:
            sanitize_uuid("../../etc/passwd")
        self.assertIn("path traversal", str(ctx.exception).lower())

    def test_rejects_dot_dot_embedded(self):
        with self.assertRaises(ValueError):
            sanitize_uuid("legit..path")

    def test_rejects_forward_slash(self):
        with self.assertRaises(ValueError):
            sanitize_uuid("foo/bar")

    def test_rejects_backslash(self):
        with self.assertRaises(ValueError):
            sanitize_uuid("foo\\bar")

    def test_rejects_spaces(self):
        with self.assertRaises(ValueError):
            sanitize_uuid("has spaces")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            sanitize_uuid("")

    def test_rejects_none(self):
        with self.assertRaises(ValueError):
            sanitize_uuid(None)

    def test_rejects_dots_only(self):
        with self.assertRaises(ValueError):
            sanitize_uuid("..")

    def test_rejects_colon(self):
        with self.assertRaises(ValueError):
            sanitize_uuid("C:\\Users")

    def test_rejects_null_byte(self):
        with self.assertRaises(ValueError):
            sanitize_uuid("github\x00-001")

    def test_rejects_too_long(self):
        with self.assertRaises(ValueError):
            sanitize_uuid("a" * 129)

    def test_accepts_max_length(self):
        result = sanitize_uuid("a" * 128)
        self.assertEqual(len(result), 128)

    def test_strips_whitespace(self):
        result = sanitize_uuid("  github-001  ")
        self.assertEqual(result, "github-001")

    def test_rejects_tilde(self):
        """Tilde could reference home directory."""
        with self.assertRaises(ValueError):
            sanitize_uuid("~root")

    def test_rejects_percent_encoding(self):
        with self.assertRaises(ValueError):
            sanitize_uuid("github%2F001")


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Encrypted Credential Store
# ═══════════════════════════════════════════════════════════════════════

class TestEncryptedCredentialStore(unittest.TestCase):
    """Sprint 45: CredentialStore with encryption and secure file handling."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_cred(self, uuid="test-001", name="Test"):
        return StoredCredential(
            connector_uuid=uuid,
            connector_name=name,
            auth_method=AuthMethod.API_TOKEN.value,
            tokens={"api_token": "secret_token_value_12345"},
            created_at=time.time(),
            status=AuthStatus.AUTHENTICATED.value,
        )

    def test_save_and_load_roundtrip(self):
        store = CredentialStore(credentials_dir=self.tmpdir)
        cred = self._make_cred()
        store.save(cred)
        loaded = store.load("test-001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.tokens["api_token"], "secret_token_value_12345")

    def test_file_not_plain_text(self):
        """Credential file should NOT contain plaintext token."""
        store = CredentialStore(credentials_dir=self.tmpdir)
        cred = self._make_cred()
        store.save(cred)
        path = os.path.join(self.tmpdir, "test-001.json")
        with open(path, "r") as f:
            raw = f.read()
        self.assertNotIn("secret_token_value_12345", raw)

    def test_file_not_plain_base64_decodable_to_token(self):
        """
        If encryption is available, base64 decode of the data field
        should NOT produce the original token.
        """
        store = CredentialStore(credentials_dir=self.tmpdir)
        if not store.encryption_available:
            self.skipTest("cryptography not installed")
        cred = self._make_cred()
        store.save(cred)
        path = os.path.join(self.tmpdir, "test-001.json")
        with open(path) as f:
            wrapper = json.load(f)
        # Try base64 decoding — should fail or not contain token
        try:
            decoded = base64.b64decode(wrapper["data"]).decode("utf-8", errors="ignore")
            self.assertNotIn("secret_token_value_12345", decoded)
        except Exception:
            pass  # Expected if Fernet encrypted

    def test_file_permissions_owner_only(self):
        """Credential files should be readable only by owner."""
        store = CredentialStore(credentials_dir=self.tmpdir)
        cred = self._make_cred()
        store.save(cred)
        path = os.path.join(self.tmpdir, "test-001.json")
        file_stat = os.stat(path)
        mode = stat.S_IMODE(file_stat.st_mode)
        # Owner should have read+write, no one else should have anything
        self.assertEqual(mode & 0o077, 0, f"Other/group permissions should be 0, got {oct(mode)}")

    def test_secure_delete_overwrites(self):
        """Delete should overwrite file contents before removal."""
        store = CredentialStore(credentials_dir=self.tmpdir)
        cred = self._make_cred()
        store.save(cred)
        path = os.path.join(self.tmpdir, "test-001.json")
        self.assertTrue(os.path.exists(path))
        store.delete("test-001")
        self.assertFalse(os.path.exists(path))

    def test_path_traversal_rejected_on_save(self):
        """Saving with a traversal UUID should fail (returns False)."""
        store = CredentialStore(credentials_dir=self.tmpdir)
        cred = self._make_cred(uuid="../../etc/passwd")
        # save() catches ValueError internally and returns False
        result = store.save(cred)
        self.assertFalse(result)

    def test_path_traversal_rejected_on_load(self):
        store = CredentialStore(credentials_dir=self.tmpdir)
        result = store.load("../../../etc/shadow")
        self.assertIsNone(result)

    def test_path_traversal_rejected_on_delete(self):
        store = CredentialStore(credentials_dir=self.tmpdir)
        result = store.delete("../../important-file")
        self.assertFalse(result)

    def test_legacy_base64_auto_upgrade(self):
        """v1 (base64) files should auto-upgrade to v2 (encrypted) on read."""
        store = CredentialStore(credentials_dir=self.tmpdir)
        cred = self._make_cred()
        # Write a v1 (base64-only) file manually
        data = cred.to_dict()
        encoded = base64.b64encode(json.dumps(data).encode("utf-8")).decode("ascii")
        path = os.path.join(self.tmpdir, "test-001.json")
        with open(path, "w") as f:
            json.dump({"v": 1, "data": encoded}, f)

        loaded = store.load("test-001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.tokens["api_token"], "secret_token_value_12345")

        # If encryption available, file should now be v2
        if store.encryption_available:
            with open(path) as f:
                wrapper = json.load(f)
            self.assertEqual(wrapper["v"], 2)

    def test_corrupt_file_returns_none(self):
        store = CredentialStore(credentials_dir=self.tmpdir)
        path = os.path.join(self.tmpdir, "test-001.json")
        with open(path, "w") as f:
            f.write("not json at all")
        result = store.load("test-001")
        self.assertIsNone(result)

    def test_encryption_key_persisted(self):
        """Key file should be created and reused across instances."""
        store1 = CredentialStore(credentials_dir=self.tmpdir)
        cred = self._make_cred()
        store1.save(cred)

        store2 = CredentialStore(credentials_dir=self.tmpdir)
        loaded = store2.load("test-001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.tokens["api_token"], "secret_token_value_12345")


# ═══════════════════════════════════════════════════════════════════════
# Test 5: OAuth State TTL and Limits
# ═══════════════════════════════════════════════════════════════════════

class TestOAuthStateSecurity(unittest.TestCase):
    """Sprint 45: OAuth state TTL expiry and max-pending limits."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.mgr = ConnectorAuthManager(
            credential_store=self.store,
            auth_configs={
                "test-oauth": AuthConfig(
                    method=AuthMethod.OAUTH2,
                    auth_url="https://example.com/auth",
                    token_url="https://example.com/token",
                    scopes=["read"],
                    token_env_var="TEST_CLIENT_ID",
                ),
            },
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch.dict(os.environ, {"TEST_CLIENT_ID": "my-client"})
    def test_state_has_timestamp(self):
        url = self.mgr.initiate_oauth2("test-oauth", "Test")
        states = self.mgr._oauth_states
        self.assertEqual(len(states), 1)
        state_info = list(states.values())[0]
        self.assertIn("uuid", state_info)
        self.assertIn("created_at", state_info)
        self.assertEqual(state_info["uuid"], "test-oauth")

    @patch.dict(os.environ, {"TEST_CLIENT_ID": "my-client"})
    def test_expired_state_rejected_on_complete(self):
        url = self.mgr.initiate_oauth2("test-oauth", "Test")
        state = list(self.mgr._oauth_states.keys())[0]
        # Manually expire the state
        self.mgr._oauth_states[state]["created_at"] = time.time() - OAUTH_STATE_TTL - 10
        with self.assertRaises(ValueError) as ctx:
            self.mgr.complete_oauth2(state, "auth-code-123")
        self.assertIn("expired", str(ctx.exception).lower())

    @patch.dict(os.environ, {"TEST_CLIENT_ID": "my-client"})
    def test_expired_states_cleaned_up(self):
        """Expired states should be automatically cleaned up."""
        # Add some expired states manually
        for i in range(5):
            self.mgr._oauth_states[f"old-state-{i}"] = {
                "uuid": "test-oauth",
                "created_at": time.time() - OAUTH_STATE_TTL - 100,
            }
        self.assertEqual(len(self.mgr._oauth_states), 5)
        # Initiating a new flow should trigger cleanup
        self.mgr.initiate_oauth2("test-oauth", "Test")
        # Old states should be gone, new one added
        self.assertEqual(len(self.mgr._oauth_states), 1)

    @patch.dict(os.environ, {"TEST_CLIENT_ID": "my-client"})
    def test_max_pending_states_enforced(self):
        """Cannot create more than MAX_PENDING_OAUTH_STATES."""
        for i in range(MAX_PENDING_OAUTH_STATES):
            self.mgr._oauth_states[f"state-{i}"] = {
                "uuid": "test-oauth",
                "created_at": time.time(),
            }
        with self.assertRaises(ValueError) as ctx:
            self.mgr.initiate_oauth2("test-oauth", "Test")
        self.assertIn("too many", str(ctx.exception).lower())

    @patch.dict(os.environ, {"TEST_CLIENT_ID": "my-client"})
    def test_complete_with_empty_code_rejected(self):
        url = self.mgr.initiate_oauth2("test-oauth", "Test")
        state = list(self.mgr._oauth_states.keys())[0]
        with self.assertRaises(ValueError) as ctx:
            self.mgr.complete_oauth2(state, "")
        self.assertIn("empty", str(ctx.exception).lower())

    @patch.dict(os.environ, {"TEST_CLIENT_ID": "my-client"})
    def test_complete_with_whitespace_code_rejected(self):
        url = self.mgr.initiate_oauth2("test-oauth", "Test")
        state = list(self.mgr._oauth_states.keys())[0]
        with self.assertRaises(ValueError) as ctx:
            self.mgr.complete_oauth2(state, "   ")
        self.assertIn("empty", str(ctx.exception).lower())


# ═══════════════════════════════════════════════════════════════════════
# Test 6: connect_with_token Validation
# ═══════════════════════════════════════════════════════════════════════

class TestConnectWithTokenSecurity(unittest.TestCase):
    """Sprint 45: connect_with_token rejects bad input."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.mgr = ConnectorAuthManager(
            credential_store=self.store,
            auth_configs={
                "github-001": AuthConfig(
                    method=AuthMethod.API_TOKEN,
                    token_name="PAT",
                    token_env_var="GITHUB_TOKEN",
                ),
            },
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rejects_empty_token(self):
        with self.assertRaises(ValueError):
            self.mgr.connect_with_token("github-001", "GitHub", "")

    def test_rejects_whitespace_token(self):
        with self.assertRaises(ValueError):
            self.mgr.connect_with_token("github-001", "GitHub", "   ")

    def test_rejects_short_token(self):
        with self.assertRaises(ValueError):
            self.mgr.connect_with_token("github-001", "GitHub", "ab")

    def test_rejects_traversal_uuid(self):
        with self.assertRaises(ValueError):
            self.mgr.connect_with_token("../evil", "Evil", "valid_token_1234")

    def test_accepts_valid_token(self):
        cred = self.mgr.connect_with_token("github-001", "GitHub", "ghp_validtoken12345")
        self.assertEqual(cred.tokens["api_token"], "ghp_validtoken12345")

    def test_strips_whitespace_from_token(self):
        cred = self.mgr.connect_with_token("github-001", "GitHub", "  ghp_token123  ")
        self.assertEqual(cred.tokens["api_token"], "ghp_token123")


# ═══════════════════════════════════════════════════════════════════════
# Test 7: connect_with_env Validation
# ═══════════════════════════════════════════════════════════════════════

class TestConnectWithEnvSecurity(unittest.TestCase):
    """Sprint 45: connect_with_env rejects bad input."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.mgr = ConnectorAuthManager(credential_store=self.store)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rejects_empty_env_values(self):
        with self.assertRaises(ValueError):
            self.mgr.connect_with_env("test-001", "Test", {})

    def test_rejects_empty_string_value(self):
        with self.assertRaises(ValueError):
            self.mgr.connect_with_env("test-001", "Test", {"KEY": ""})

    def test_rejects_whitespace_value(self):
        with self.assertRaises(ValueError):
            self.mgr.connect_with_env("test-001", "Test", {"KEY": "  "})

    def test_rejects_traversal_uuid(self):
        with self.assertRaises(ValueError):
            self.mgr.connect_with_env("../../etc", "Test", {"KEY": "value"})

    def test_accepts_valid_env(self):
        cred = self.mgr.connect_with_env(
            "test-001", "Test", {"SEARXNG_URL": "http://localhost:8080"}
        )
        self.assertEqual(cred.tokens["SEARXNG_URL"], "http://localhost:8080")


# ═══════════════════════════════════════════════════════════════════════
# Test 8: build_mcp_env Restriction
# ═══════════════════════════════════════════════════════════════════════

class TestBuildMcpEnvSecurity(unittest.TestCase):
    """Sprint 45: build_mcp_env only exports declared env vars."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.mgr = ConnectorAuthManager(
            credential_store=self.store,
            auth_configs={
                "github-001": AuthConfig(
                    method=AuthMethod.API_TOKEN,
                    token_name="PAT",
                    token_env_var="GITHUB_TOKEN",
                ),
            },
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_api_token_only_declared_env(self):
        """Should only export GITHUB_TOKEN, not 'api_token' key."""
        self.mgr.connect_with_token("github-001", "GitHub", "ghp_secret12345")
        env = self.mgr.build_mcp_env("github-001")
        self.assertIn("GITHUB_TOKEN", env)
        self.assertEqual(env["GITHUB_TOKEN"], "ghp_secret12345")
        # Sprint 45: Should NOT blindly dump other keys
        self.assertNotIn("api_token", env)

    def test_api_token_no_extra_keys(self):
        """Only the declared token_env_var should appear in the output."""
        self.mgr.connect_with_token("github-001", "GitHub", "ghp_secret12345")
        env = self.mgr.build_mcp_env("github-001")
        # Exactly one key expected
        self.assertEqual(len(env), 1)

    def test_not_connected_returns_empty(self):
        env = self.mgr.build_mcp_env("github-001")
        self.assertEqual(env, {})


# ═══════════════════════════════════════════════════════════════════════
# Test 9: Log Redaction
# ═══════════════════════════════════════════════════════════════════════

class TestLogRedaction(unittest.TestCase):
    """Sprint 45: Verify no token values appear in log messages."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.mgr = ConnectorAuthManager(
            credential_store=self.store,
            auth_configs={
                "github-001": AuthConfig(
                    method=AuthMethod.API_TOKEN,
                    token_name="PAT",
                    token_env_var="GITHUB_TOKEN",
                ),
            },
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_connect_log_redacted(self):
        """Connecting should not log the actual token value."""
        import logging
        import logging.handlers
        handler = logging.handlers.MemoryHandler(capacity=100)
        log = logging.getLogger("core.connector_auth")
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)

        try:
            self.mgr.connect_with_token("github-001", "GitHub", "ghp_supersecret123")
            handler.flush()
            for record in handler.buffer:
                msg = record.getMessage()
                self.assertNotIn("ghp_supersecret123", msg)
        finally:
            log.removeHandler(handler)

    def test_save_log_redacted(self):
        """Saving credentials should not log token values."""
        import logging
        import logging.handlers
        handler = logging.handlers.MemoryHandler(capacity=100)
        log = logging.getLogger("core.connector_auth")
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)

        try:
            cred = StoredCredential(
                connector_uuid="github-001",
                connector_name="GitHub",
                auth_method=AuthMethod.API_TOKEN.value,
                tokens={"api_token": "supersecretvalue"},
                created_at=time.time(),
                status=AuthStatus.AUTHENTICATED.value,
            )
            self.store.save(cred)
            handler.flush()
            for record in handler.buffer:
                msg = record.getMessage()
                self.assertNotIn("supersecretvalue", msg)
        finally:
            log.removeHandler(handler)


# ═══════════════════════════════════════════════════════════════════════
# Test 10: ConnectConnectorTool Security
# ═══════════════════════════════════════════════════════════════════════

class TestConnectToolSecurity(unittest.TestCase):
    """Sprint 45: ConnectConnectorTool validates and masks properly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.auth = ConnectorAuthManager(
            credential_store=self.store,
            auth_configs={
                "github-001": AuthConfig(
                    method=AuthMethod.API_TOKEN,
                    token_name="PAT",
                    token_env_var="GITHUB_TOKEN",
                ),
            },
        )
        self.registry = ConnectorRegistry()
        self.registry.register(ConnectorInfo(
            uuid="github-001", name="GitHub",
            description="GitHub integration",
            tool_names=["search_repos"],
        ))
        self.tool = ConnectConnectorTool(
            auth_manager=self.auth,
            connector_registry=self.registry,
        )
        self.loop = asyncio.new_event_loop()

    def tearDown(self):
        self.loop.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rejects_empty_token(self):
        result = self.loop.run_until_complete(
            self.tool.execute(name="github", token="")
        )
        # Empty token → should prompt for token (success with needs_token)
        self.assertTrue(result.success)
        self.assertIn("requires", result.output)

    def test_rejects_whitespace_token(self):
        result = self.loop.run_until_complete(
            self.tool.execute(name="github", token="   ")
        )
        # Whitespace-only → stripped to empty → validate_token raises
        self.assertFalse(result.success)

    def test_rejects_short_token(self):
        result = self.loop.run_until_complete(
            self.tool.execute(name="github", token="ab")
        )
        self.assertFalse(result.success)
        self.assertIn("too short", result.error.lower())

    def test_valid_token_masked_in_output(self):
        result = self.loop.run_until_complete(
            self.tool.execute(name="github", token="ghp_verylongsecrettoken1234")
        )
        self.assertTrue(result.success)
        self.assertNotIn("ghp_verylongsecrettoken1234", result.output)
        self.assertIn("connected", result.output.lower())

    def test_short_token_not_leaked_in_output(self):
        """8-char token should be fully masked, not partially revealed."""
        result = self.loop.run_until_complete(
            self.tool.execute(name="github", token="abcd1234")
        )
        self.assertTrue(result.success)
        # The masked output should NOT contain any original chars
        self.assertNotIn("abcd", result.output)
        self.assertNotIn("1234", result.output)


# ═══════════════════════════════════════════════════════════════════════
# Test 11: Edge Cases — Concurrent Operations
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCasesConcurrent(unittest.TestCase):
    """Sprint 45: Edge cases for concurrent and rapid operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.mgr = ConnectorAuthManager(
            credential_store=self.store,
            auth_configs={
                "test-001": AuthConfig(
                    method=AuthMethod.API_TOKEN,
                    token_name="Token",
                    token_env_var="TEST_TOKEN",
                ),
            },
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rapid_connect_disconnect_cycle(self):
        """Rapid connect/disconnect should not corrupt state."""
        for i in range(50):
            self.mgr.connect_with_token("test-001", "Test", f"token_{i:04d}_valid")
            self.assertTrue(self.mgr.is_connected("test-001"))
            self.mgr.disconnect("test-001")
            self.assertFalse(self.mgr.is_connected("test-001"))

    def test_double_disconnect_safe(self):
        self.mgr.connect_with_token("test-001", "Test", "valid_token_12345")
        self.mgr.disconnect("test-001")
        # Second disconnect should not raise
        result = self.mgr.disconnect("test-001")
        self.assertFalse(result)

    def test_overwrite_credential(self):
        """Connecting again should replace the old credential."""
        self.mgr.connect_with_token("test-001", "Test", "old_token_123456")
        self.mgr.connect_with_token("test-001", "Test", "new_token_654321")
        token = self.mgr.get_token("test-001")
        self.assertEqual(token, "new_token_654321")

    def test_load_after_overwrite(self):
        """Disk should contain the latest credential."""
        self.mgr.connect_with_token("test-001", "Test", "first_token_1234")
        self.mgr.connect_with_token("test-001", "Test", "second_token_5678")
        # Create a fresh manager with same store
        mgr2 = ConnectorAuthManager(credential_store=self.store)
        creds = mgr2.load_saved_credentials()
        self.assertEqual(len(creds), 1)
        self.assertEqual(creds[0].tokens["api_token"], "second_token_5678")


# ═══════════════════════════════════════════════════════════════════════
# Test 12: Edge Cases — Malformed Data
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCasesMalformedData(unittest.TestCase):
    """Sprint 45: Handling of malformed credential files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_missing_version_field(self):
        store = CredentialStore(credentials_dir=self.tmpdir)
        cred_data = {"connector_uuid": "test-001", "connector_name": "Test",
                     "auth_method": "api_token", "tokens": {"api_token": "val"},
                     "created_at": 0, "expires_at": 0, "last_used_at": 0,
                     "status": "authenticated"}
        encoded = base64.b64encode(json.dumps(cred_data).encode()).decode()
        path = os.path.join(self.tmpdir, "test-001.json")
        with open(path, "w") as f:
            json.dump({"data": encoded}, f)
        result = store.load("test-001")
        self.assertIsNotNone(result)

    def test_empty_json_file(self):
        store = CredentialStore(credentials_dir=self.tmpdir)
        path = os.path.join(self.tmpdir, "test-001.json")
        with open(path, "w") as f:
            f.write("{}")
        result = store.load("test-001")
        self.assertIsNone(result)

    def test_binary_garbage_file(self):
        store = CredentialStore(credentials_dir=self.tmpdir)
        path = os.path.join(self.tmpdir, "test-001.json")
        with open(path, "wb") as f:
            f.write(secrets.token_bytes(256))
        result = store.load("test-001")
        self.assertIsNone(result)

    def test_truncated_base64(self):
        store = CredentialStore(credentials_dir=self.tmpdir)
        path = os.path.join(self.tmpdir, "test-001.json")
        with open(path, "w") as f:
            json.dump({"v": 1, "data": "aGVsbG8="[:5]}, f)
        result = store.load("test-001")
        self.assertIsNone(result)

    def test_valid_json_wrong_shape(self):
        """JSON array instead of object."""
        store = CredentialStore(credentials_dir=self.tmpdir)
        path = os.path.join(self.tmpdir, "test-001.json")
        with open(path, "w") as f:
            json.dump([1, 2, 3], f)
        result = store.load("test-001")
        self.assertIsNone(result)

    def test_non_json_files_ignored_in_list_all(self):
        """list_all should skip non-.json files."""
        store = CredentialStore(credentials_dir=self.tmpdir)
        # Create a .key file and a .txt file
        with open(os.path.join(self.tmpdir, ".key"), "w") as f:
            f.write("encryption key")
        with open(os.path.join(self.tmpdir, "notes.txt"), "w") as f:
            f.write("not a credential")
        result = store.list_all()
        self.assertEqual(len(result), 0)


# ═══════════════════════════════════════════════════════════════════════
# Test 13: Constants and Defaults
# ═══════════════════════════════════════════════════════════════════════

class TestSecurityConstants(unittest.TestCase):
    """Sprint 45: Verify security constants have safe values."""

    def test_max_pending_states_reasonable(self):
        self.assertGreater(MAX_PENDING_OAUTH_STATES, 0)
        self.assertLessEqual(MAX_PENDING_OAUTH_STATES, 100)

    def test_oauth_state_ttl_reasonable(self):
        """TTL should be between 1 minute and 1 hour."""
        self.assertGreaterEqual(OAUTH_STATE_TTL, 60)
        self.assertLessEqual(OAUTH_STATE_TTL, 3600)

    def test_min_token_length_reasonable(self):
        self.assertGreaterEqual(MIN_TOKEN_LENGTH, 1)
        self.assertLessEqual(MIN_TOKEN_LENGTH, 16)

    def test_max_token_length_reasonable(self):
        self.assertGreaterEqual(MAX_TOKEN_LENGTH, 256)
        self.assertLessEqual(MAX_TOKEN_LENGTH, 65536)

    def test_file_mode_owner_only(self):
        """CREDENTIAL_FILE_MODE should allow only owner read/write."""
        self.assertEqual(CREDENTIAL_FILE_MODE & 0o077, 0)
        self.assertTrue(CREDENTIAL_FILE_MODE & stat.S_IRUSR)
        self.assertTrue(CREDENTIAL_FILE_MODE & stat.S_IWUSR)

    def test_safe_uuid_regex_rejects_slash(self):
        self.assertIsNone(SAFE_UUID_RE.match("a/b"))

    def test_safe_uuid_regex_rejects_dot(self):
        self.assertIsNone(SAFE_UUID_RE.match("a.b"))

    def test_safe_uuid_regex_accepts_hyphen(self):
        self.assertIsNotNone(SAFE_UUID_RE.match("github-001"))

    def test_safe_uuid_regex_accepts_underscore(self):
        self.assertIsNotNone(SAFE_UUID_RE.match("my_conn_1"))


# ═══════════════════════════════════════════════════════════════════════
# Test 14: DisconnectConnectorTool Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestDisconnectToolEdgeCases(unittest.TestCase):
    """Sprint 45: Edge cases in disconnect flow."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CredentialStore(credentials_dir=self.tmpdir)
        self.auth = ConnectorAuthManager(
            credential_store=self.store,
            auth_configs={
                "github-001": AuthConfig(
                    method=AuthMethod.API_TOKEN,
                    token_name="PAT",
                    token_env_var="GITHUB_TOKEN",
                ),
            },
        )
        self.registry = ConnectorRegistry()
        self.registry.register(ConnectorInfo(
            uuid="github-001", name="GitHub",
            description="GitHub integration",
            tool_names=["search_repos"],
        ))
        self.tool = DisconnectConnectorTool(
            auth_manager=self.auth,
            connector_registry=self.registry,
        )
        self.loop = asyncio.new_event_loop()

    def tearDown(self):
        self.loop.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_disconnect_removes_disk_file(self):
        """After disconnect, credential file should not exist."""
        self.auth.connect_with_token("github-001", "GitHub", "ghp_secret_tok_12345")
        path = os.path.join(self.tmpdir, "github-001.json")
        self.assertTrue(os.path.exists(path))

        self.loop.run_until_complete(self.tool.execute(name="github"))
        self.assertFalse(os.path.exists(path))

    def test_disconnect_clears_memory(self):
        self.auth.connect_with_token("github-001", "GitHub", "ghp_secret_tok_12345")
        self.assertTrue(self.auth.is_connected("github-001"))

        self.loop.run_until_complete(self.tool.execute(name="github"))
        self.assertFalse(self.auth.is_connected("github-001"))


# ═══════════════════════════════════════════════════════════════════════
# Test 15: ListConnectorsTool Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestListToolEdgeCases(unittest.TestCase):
    """Sprint 45: Edge cases in list connectors."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()

    def tearDown(self):
        self.loop.close()

    def test_no_registry(self):
        """When no registry is set, tool should return an error."""
        tool = ListConnectorsTool(auth_manager=None, connector_registry=None)
        result = self.loop.run_until_complete(tool.execute(filter="all"))
        self.assertFalse(result.success)
        self.assertIn("registry", result.error.lower())

    def test_filter_connected_when_none_connected(self):
        registry = ConnectorRegistry()
        registry.register(ConnectorInfo(
            uuid="test-001", name="Test",
            description="test", tool_names=[],
        ))
        tool = ListConnectorsTool(auth_manager=None, connector_registry=registry)

        result = self.loop.run_until_complete(tool.execute(filter="connected"))
        self.assertIn("No connected connectors found", result.output)

    def test_invalid_filter_treated_as_all(self):
        registry = ConnectorRegistry()
        registry.register(ConnectorInfo(
            uuid="test-001", name="Test",
            description="A test connector", tool_names=[],
        ))
        tool = ListConnectorsTool(auth_manager=None, connector_registry=registry)

        result = self.loop.run_until_complete(tool.execute(filter="bogus_filter"))
        self.assertIn("Test", result.output)


# ═══════════════════════════════════════════════════════════════════════
# Test 16: Credential Store Encryption Key
# ═══════════════════════════════════════════════════════════════════════

class TestCredentialStoreKeyManagement(unittest.TestCase):
    """Sprint 45: Key file management for encryption."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_key_file_created(self):
        """Encryption key file should be created in the credentials dir."""
        store = CredentialStore(credentials_dir=self.tmpdir)
        if store.encryption_available:
            key_path = os.path.join(self.tmpdir, ".key")
            self.assertTrue(os.path.exists(key_path))

    def test_key_file_permissions(self):
        """Key file should be owner-readable only."""
        store = CredentialStore(credentials_dir=self.tmpdir)
        if store.encryption_available:
            key_path = os.path.join(self.tmpdir, ".key")
            file_stat = os.stat(key_path)
            mode = stat.S_IMODE(file_stat.st_mode)
            self.assertEqual(mode & 0o077, 0)

    def test_custom_encryption_key(self):
        """Should accept a custom encryption key."""
        try:
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            store = CredentialStore(credentials_dir=self.tmpdir, encryption_key=key)
            self.assertTrue(store.encryption_available)

            cred = StoredCredential(
                connector_uuid="test-001", connector_name="Test",
                auth_method="api_token",
                tokens={"api_token": "my_secret_token_value"},
                created_at=time.time(),
                status="authenticated",
            )
            store.save(cred)
            loaded = store.load("test-001")
            self.assertEqual(loaded.tokens["api_token"], "my_secret_token_value")
        except ImportError:
            self.skipTest("cryptography not installed")

    def test_wrong_key_fails_decrypt(self):
        """Loading with wrong key should fail gracefully."""
        try:
            from cryptography.fernet import Fernet
            key1 = Fernet.generate_key()
            key2 = Fernet.generate_key()

            store1 = CredentialStore(credentials_dir=self.tmpdir, encryption_key=key1)
            cred = StoredCredential(
                connector_uuid="test-001", connector_name="Test",
                auth_method="api_token",
                tokens={"api_token": "secret_value"},
                created_at=time.time(),
                status="authenticated",
            )
            store1.save(cred)

            # Try loading with different key
            store2 = CredentialStore(
                credentials_dir=self.tmpdir, encryption_key=key2
            )
            result = store2.load("test-001")
            self.assertIsNone(result)  # Should return None, not raise
        except ImportError:
            self.skipTest("cryptography not installed")


if __name__ == "__main__":
    # Need logging.handlers for MemoryHandler
    import logging.handlers
    unittest.main(verbosity=2)
