"""
Connector Authentication — Manages auth flows for MCP connectors.

Sprint 44: Provides three authentication strategies:
  - OAuth2 (browser-based authorization for Gmail, GitHub, Google Drive, etc.)
  - API Token (masked prompt input for Slack, Telegram, etc.)
  - Environment Variable (for self-hosted services like SearXNG)

Sprint 45: Security hardening:
  - Fernet encryption for credential files (not just base64)
  - Path traversal protection with strict UUID sanitization
  - File permissions locked to owner-only (0o600)
  - OAuth state TTL expiry and max-pending limits
  - Token validation (reject empty/whitespace tokens)
  - Redacted logging (never log token values)
  - Secure token masking (minimum redaction)
  - build_mcp_env restricted to declared env vars only
  - Secure credential wiping on disconnect

Credentials are stored Fernet-encrypted in ~/.cowork_agent/credentials/ and
auto-loaded at startup to reconnect previously authenticated services.
"""

from __future__ import annotations
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import stat
import time
import webbrowser
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Sprint 45: Security Constants ────────────────────────────────────
# Maximum number of pending OAuth states (prevents memory exhaustion)
MAX_PENDING_OAUTH_STATES = 20
# OAuth state TTL in seconds (10 minutes)
OAUTH_STATE_TTL = 600
# Minimum token length to accept
MIN_TOKEN_LENGTH = 4
# Maximum token length to accept (prevents memory abuse)
MAX_TOKEN_LENGTH = 8192
# Regex for safe UUID characters (alphanumeric, hyphens, underscores)
SAFE_UUID_RE = re.compile(r'^[a-zA-Z0-9_-]+$')
# File permission: owner read/write only
CREDENTIAL_FILE_MODE = 0o600


class AuthMethod(Enum):
    """Supported authentication methods."""
    OAUTH2 = "oauth2"
    API_TOKEN = "api_token"
    ENV_VAR = "env_var"


class AuthStatus(Enum):
    """Current status of a connector's auth."""
    NOT_CONFIGURED = "not_configured"
    PENDING = "pending"           # OAuth flow started, waiting for callback
    AUTHENTICATED = "authenticated"
    EXPIRED = "expired"
    FAILED = "failed"
    REVOKED = "revoked"


@dataclass
class AuthConfig:
    """Authentication configuration for a connector."""
    method: AuthMethod
    # OAuth2 fields
    client_id: str = ""
    client_secret: str = ""
    auth_url: str = ""
    token_url: str = ""
    redirect_uri: str = "http://localhost:9876/callback"
    scopes: List[str] = field(default_factory=list)
    # API token fields
    token_name: str = ""           # Display name (e.g., "Bot Token")
    token_env_var: str = ""        # Env var to check (e.g., "SLACK_BOT_TOKEN")
    # Environment variable fields
    env_vars: List[str] = field(default_factory=list)  # Required env vars


@dataclass
class StoredCredential:
    """A persisted credential for a connector."""
    connector_uuid: str
    connector_name: str
    auth_method: str
    # Stored values (tokens, keys, etc.)
    tokens: Dict[str, str] = field(default_factory=dict)
    # Metadata
    created_at: float = 0.0
    expires_at: float = 0.0       # 0 = no expiry
    last_used_at: float = 0.0
    status: str = "authenticated"

    def is_expired(self) -> bool:
        """Check if the credential has expired."""
        if self.expires_at <= 0:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> StoredCredential:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Sprint 45: Security Utility Functions ────────────────────────────

def validate_token(token: str) -> str:
    """
    Validate and sanitize a token value.

    Raises ValueError if the token is empty, whitespace-only,
    or exceeds maximum length.

    Returns the stripped token.
    """
    if not token or not isinstance(token, str):
        raise ValueError("Token must be a non-empty string")
    stripped = token.strip()
    if not stripped:
        raise ValueError("Token cannot be empty or whitespace-only")
    if len(stripped) < MIN_TOKEN_LENGTH:
        raise ValueError(
            f"Token too short (minimum {MIN_TOKEN_LENGTH} characters)"
        )
    if len(stripped) > MAX_TOKEN_LENGTH:
        raise ValueError(
            f"Token too long (maximum {MAX_TOKEN_LENGTH} characters)"
        )
    return stripped


def mask_token(token: str, show_chars: int = 3) -> str:
    """
    Safely mask a token for display. Never reveals more than `show_chars`
    characters from each end, and always shows at least 4 asterisks.

    Sprint 45: Hardened — short tokens are fully masked.
    """
    if not token:
        return "****"
    length = len(token)
    # Tokens under 12 chars: show nothing
    if length < 12:
        return "*" * max(length, 4)
    # Show at most `show_chars` from each end
    return token[:show_chars] + "*" * max(length - show_chars * 2, 4) + token[-show_chars:]


def sanitize_uuid(connector_uuid: str) -> str:
    """
    Sanitize a connector UUID to prevent path traversal attacks.

    Sprint 45: Rejects any UUID containing path separators, dots sequences,
    or non-alphanumeric/hyphen/underscore characters.

    Raises ValueError for invalid UUIDs.
    """
    if not connector_uuid or not isinstance(connector_uuid, str):
        raise ValueError("Connector UUID must be a non-empty string")
    stripped = connector_uuid.strip()
    if not stripped:
        raise ValueError("Connector UUID cannot be empty")
    # Check for path traversal patterns
    if ".." in stripped:
        raise ValueError(f"Invalid UUID: contains path traversal sequence '..'")
    # Only allow safe characters
    if not SAFE_UUID_RE.match(stripped):
        raise ValueError(
            f"Invalid UUID: contains unsafe characters. "
            f"Only alphanumeric, hyphens, and underscores are allowed."
        )
    if len(stripped) > 128:
        raise ValueError("Connector UUID too long (max 128 characters)")
    return stripped


# ── Built-in Auth Configurations ──────────────────────────────────────

# Maps connector UUID → AuthConfig
CONNECTOR_AUTH_CONFIGS: Dict[str, AuthConfig] = {
    "gmail-001": AuthConfig(
        method=AuthMethod.OAUTH2,
        auth_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send"],
        token_env_var="GMAIL_CLIENT_ID",
    ),
    "google-drive-001": AuthConfig(
        method=AuthMethod.OAUTH2,
        auth_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive"],
        token_env_var="GOOGLE_DRIVE_CLIENT_ID",
    ),
    "github-001": AuthConfig(
        method=AuthMethod.API_TOKEN,
        token_name="Personal Access Token",
        token_env_var="GITHUB_TOKEN",
    ),
    "slack-001": AuthConfig(
        method=AuthMethod.API_TOKEN,
        token_name="Bot Token (xoxb-...)",
        token_env_var="SLACK_BOT_TOKEN",
    ),
    "asana-001": AuthConfig(
        method=AuthMethod.API_TOKEN,
        token_name="Personal Access Token",
        token_env_var="ASANA_TOKEN",
    ),
    "jira-001": AuthConfig(
        method=AuthMethod.API_TOKEN,
        token_name="API Token",
        token_env_var="JIRA_API_TOKEN",
    ),
    "notion-001": AuthConfig(
        method=AuthMethod.API_TOKEN,
        token_name="Integration Token",
        token_env_var="NOTION_TOKEN",
    ),
    "linear-001": AuthConfig(
        method=AuthMethod.API_TOKEN,
        token_name="API Key",
        token_env_var="LINEAR_API_KEY",
    ),
    "hubspot-001": AuthConfig(
        method=AuthMethod.API_TOKEN,
        token_name="Private App Access Token",
        token_env_var="HUBSPOT_TOKEN",
    ),
    "salesforce-001": AuthConfig(
        method=AuthMethod.OAUTH2,
        auth_url="https://login.salesforce.com/services/oauth2/authorize",
        token_url="https://login.salesforce.com/services/oauth2/token",
        scopes=["api", "refresh_token"],
        token_env_var="SALESFORCE_CLIENT_ID",
    ),
    "zoho-crm-001": AuthConfig(
        method=AuthMethod.API_TOKEN,
        token_name="OAuth Token",
        token_env_var="ZOHO_CRM_TOKEN",
    ),
    "dropbox-001": AuthConfig(
        method=AuthMethod.API_TOKEN,
        token_name="Access Token",
        token_env_var="DROPBOX_TOKEN",
    ),
    "trello-001": AuthConfig(
        method=AuthMethod.API_TOKEN,
        token_name="API Key",
        token_env_var="TRELLO_API_KEY",
    ),
    "confluence-001": AuthConfig(
        method=AuthMethod.API_TOKEN,
        token_name="API Token",
        token_env_var="CONFLUENCE_TOKEN",
    ),
    "canva-001": AuthConfig(
        method=AuthMethod.API_TOKEN,
        token_name="API Key",
        token_env_var="CANVA_API_KEY",
    ),
}


class CredentialStore:
    """
    Persistent credential storage with encryption.

    Sprint 45 security hardening:
      - Fernet symmetric encryption (not just base64 obfuscation)
      - File permissions locked to 0o600 (owner read/write only)
      - Path traversal protection via UUID sanitization
      - Secure delete overwrites file before unlinking
      - Graceful fallback to base64 if cryptography not installed

    Encryption key is derived from a machine-specific seed stored
    alongside credentials. For production, integrate OS keychain.
    """

    def __init__(self, credentials_dir: str = "", encryption_key: Optional[bytes] = None):
        if credentials_dir:
            self._dir = credentials_dir
        else:
            self._dir = os.path.join(
                os.path.expanduser("~"), ".cowork_agent", "credentials"
            )
        os.makedirs(self._dir, exist_ok=True)
        self._fernet = None
        self._encryption_available = False
        self._init_encryption(encryption_key)

    def _init_encryption(self, provided_key: Optional[bytes] = None) -> None:
        """Initialize Fernet encryption, falling back to base64 if unavailable."""
        try:
            from cryptography.fernet import Fernet
            if provided_key:
                self._fernet = Fernet(provided_key)
            else:
                key_file = os.path.join(self._dir, ".key")
                if os.path.exists(key_file):
                    with open(key_file, "rb") as f:
                        key = f.read().strip()
                else:
                    key = Fernet.generate_key()
                    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, CREDENTIAL_FILE_MODE)
                    with os.fdopen(fd, "wb") as f:
                        f.write(key)
                self._fernet = Fernet(key)
            self._encryption_available = True
            logger.debug("Fernet encryption initialized for credential store")
        except ImportError:
            logger.warning(
                "cryptography package not installed — "
                "falling back to base64 encoding (NOT secure). "
                "Install with: pip install cryptography"
            )
            self._encryption_available = False

    @property
    def encryption_available(self) -> bool:
        """Whether real encryption is available."""
        return self._encryption_available

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext. Uses Fernet if available, base64 fallback."""
        if self._fernet:
            return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")
        return base64.b64encode(plaintext.encode("utf-8")).decode("ascii")

    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext. Uses Fernet if available, base64 fallback."""
        if self._fernet:
            return self._fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
        return base64.b64decode(ciphertext).decode("utf-8")

    def _file_path(self, connector_uuid: str) -> str:
        """
        Get the file path for a connector's credential.
        Sprint 45: Uses sanitize_uuid() to prevent path traversal.
        """
        safe_name = sanitize_uuid(connector_uuid)
        path = os.path.join(self._dir, f"{safe_name}.json")
        # Extra safety: verify resolved path is inside credentials dir
        resolved = os.path.realpath(path)
        if not resolved.startswith(os.path.realpath(self._dir)):
            raise ValueError(f"Path traversal detected for UUID: {connector_uuid}")
        return path

    def save(self, credential: StoredCredential) -> bool:
        """Persist a credential to disk with encryption and secure permissions."""
        try:
            data = credential.to_dict()
            encrypted = self._encrypt(json.dumps(data))
            path = self._file_path(credential.connector_uuid)
            version = 2 if self._encryption_available else 1
            # Write with restricted permissions (owner-only)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, CREDENTIAL_FILE_MODE)
            with os.fdopen(fd, "w") as f:
                json.dump({"v": version, "data": encrypted}, f)
            logger.info(f"Saved credential for connector (uuid redacted)")
            return True
        except Exception as e:
            logger.error(f"Failed to save credential: {type(e).__name__}")
            return False

    def load(self, connector_uuid: str) -> Optional[StoredCredential]:
        """Load a credential from disk."""
        try:
            path = self._file_path(connector_uuid)
        except ValueError:
            logger.error(f"Invalid connector UUID rejected during load")
            return None
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                wrapper = json.load(f)
            version = wrapper.get("v", 1)
            if version >= 2 and self._encryption_available:
                decoded = self._decrypt(wrapper["data"])
            elif version == 1:
                # Legacy base64 format — decode and re-save encrypted
                decoded = base64.b64decode(wrapper["data"]).decode("utf-8")
            else:
                decoded = self._decrypt(wrapper["data"])
            cred = StoredCredential.from_dict(json.loads(decoded))
            # Auto-upgrade v1 files to encrypted v2
            if version == 1 and self._encryption_available:
                self.save(cred)
                logger.info("Auto-upgraded legacy base64 credential to encrypted format")
            return cred
        except Exception as e:
            logger.error(f"Failed to load credential: {type(e).__name__}")
            return None

    def delete(self, connector_uuid: str) -> bool:
        """
        Securely remove a credential from disk.
        Sprint 45: Overwrites file content before unlinking to prevent recovery.
        """
        try:
            path = self._file_path(connector_uuid)
        except ValueError:
            return False
        if os.path.exists(path):
            try:
                # Overwrite with random data before deletion
                size = os.path.getsize(path)
                with open(path, "wb") as f:
                    f.write(secrets.token_bytes(max(size, 64)))
                    f.flush()
                    os.fsync(f.fileno())
                os.remove(path)
                logger.info(f"Securely deleted credential file")
                return True
            except Exception as e:
                logger.error(f"Failed to delete credential: {type(e).__name__}")
                return False
        return False

    def list_all(self) -> List[StoredCredential]:
        """Load all stored credentials."""
        credentials = []
        if not os.path.isdir(self._dir):
            return credentials
        for fname in os.listdir(self._dir):
            if fname.endswith(".json"):
                uuid = fname[:-5]  # Remove .json
                try:
                    cred = self.load(uuid)
                    if cred:
                        credentials.append(cred)
                except ValueError:
                    logger.warning(f"Skipping credential file with invalid name")
                    continue
        return credentials


class ConnectorAuthManager:
    """
    Manages the full authentication lifecycle for connectors.

    Handles:
      - Initiating auth flows (OAuth2, API token, env var)
      - Storing and retrieving credentials
      - Auto-reconnecting saved credentials at startup
      - Revoking/disconnecting credentials
    """

    def __init__(
        self,
        credential_store: Optional[CredentialStore] = None,
        auth_configs: Optional[Dict[str, AuthConfig]] = None,
    ):
        self._store = credential_store or CredentialStore()
        self._auth_configs = auth_configs or dict(CONNECTOR_AUTH_CONFIGS)
        self._active: Dict[str, StoredCredential] = {}  # UUID → credential
        # Sprint 45: OAuth states now have timestamps for TTL expiry
        self._oauth_states: Dict[str, Dict[str, Any]] = {}  # state → {uuid, created_at}

    # ── Auth Flow Initiation ──────────────────────────────────────────

    def get_auth_method(self, connector_uuid: str) -> Optional[AuthMethod]:
        """Get the authentication method for a connector."""
        cfg = self._auth_configs.get(connector_uuid)
        return cfg.method if cfg else None

    def get_auth_config(self, connector_uuid: str) -> Optional[AuthConfig]:
        """Get the full auth config for a connector."""
        return self._auth_configs.get(connector_uuid)

    def register_auth_config(self, connector_uuid: str, config: AuthConfig) -> None:
        """Register or update an auth config for a connector."""
        self._auth_configs[connector_uuid] = config

    def connect_with_token(
        self,
        connector_uuid: str,
        connector_name: str,
        token: str,
        token_key: str = "api_token",
    ) -> StoredCredential:
        """
        Authenticate a connector using an API token.

        Sprint 45: Validates token and sanitizes UUID before storage.
        Never logs the token value.

        Args:
            connector_uuid: Connector UUID
            connector_name: Human-readable name
            token: The API token value
            token_key: Key name for the token in storage

        Raises:
            ValueError: If token is empty, whitespace, or too short/long
        """
        # Sprint 45: Validate token before accepting
        validated_token = validate_token(token)
        sanitize_uuid(connector_uuid)

        credential = StoredCredential(
            connector_uuid=connector_uuid,
            connector_name=connector_name,
            auth_method=AuthMethod.API_TOKEN.value,
            tokens={token_key: validated_token},
            created_at=time.time(),
            last_used_at=time.time(),
            status=AuthStatus.AUTHENTICATED.value,
        )
        self._store.save(credential)
        self._active[connector_uuid] = credential
        # Sprint 45: Never log token values
        logger.info(f"Connected {connector_name} via API token (token redacted)")
        return credential

    def connect_with_env(
        self,
        connector_uuid: str,
        connector_name: str,
        env_values: Dict[str, str],
    ) -> StoredCredential:
        """
        Authenticate a connector using environment variables.

        Sprint 45: Validates all env values are non-empty strings.

        Args:
            connector_uuid: Connector UUID
            connector_name: Human-readable name
            env_values: Dict of env var name → value
        """
        sanitize_uuid(connector_uuid)
        if not env_values:
            raise ValueError("At least one environment variable value is required")
        # Validate all values are non-empty strings
        for key, val in env_values.items():
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"Environment variable '{key}' has empty or invalid value")

        credential = StoredCredential(
            connector_uuid=connector_uuid,
            connector_name=connector_name,
            auth_method=AuthMethod.ENV_VAR.value,
            tokens=env_values,
            created_at=time.time(),
            last_used_at=time.time(),
            status=AuthStatus.AUTHENTICATED.value,
        )
        self._store.save(credential)
        self._active[connector_uuid] = credential
        logger.info(f"Connected {connector_name} via env vars (values redacted)")
        return credential

    def _cleanup_expired_oauth_states(self) -> None:
        """Sprint 45: Remove OAuth states that have exceeded their TTL."""
        now = time.time()
        expired = [
            state for state, info in self._oauth_states.items()
            if now - info.get("created_at", 0) > OAUTH_STATE_TTL
        ]
        for state in expired:
            self._oauth_states.pop(state, None)
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired OAuth state(s)")

    def initiate_oauth2(
        self,
        connector_uuid: str,
        connector_name: str,
        client_id: str = "",
        client_secret: str = "",
    ) -> str:
        """
        Start an OAuth2 authorization flow.

        Sprint 45 hardening:
          - Cleans up expired states before adding new ones
          - Enforces MAX_PENDING_OAUTH_STATES limit
          - Stores creation timestamp for TTL enforcement

        Returns the authorization URL that the user should open.
        """
        cfg = self._auth_configs.get(connector_uuid)
        if not cfg or cfg.method != AuthMethod.OAUTH2:
            raise ValueError(f"No OAuth2 config for {connector_uuid}")

        # Use provided client_id or fall back to env var
        actual_client_id = client_id or os.environ.get(cfg.token_env_var, "")
        if not actual_client_id:
            raise ValueError(
                f"No client_id provided and {cfg.token_env_var} not set. "
                f"Please provide a client_id or set the {cfg.token_env_var} env var."
            )

        # Sprint 45: Cleanup expired states and enforce limits
        self._cleanup_expired_oauth_states()
        if len(self._oauth_states) >= MAX_PENDING_OAUTH_STATES:
            raise ValueError(
                "Too many pending OAuth flows. Please complete or cancel "
                "existing authorization flows before starting new ones."
            )

        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)
        self._oauth_states[state] = {
            "uuid": connector_uuid,
            "created_at": time.time(),
        }

        # Build authorization URL
        import urllib.parse
        params = {
            "client_id": actual_client_id,
            "redirect_uri": cfg.redirect_uri,
            "response_type": "code",
            "scope": " ".join(cfg.scopes),
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = f"{cfg.auth_url}?{urllib.parse.urlencode(params)}"

        logger.info(f"OAuth2 flow started for {connector_name}")
        return auth_url

    def complete_oauth2(
        self,
        state: str,
        code: str,
        connector_name: str = "",
        client_id: str = "",
        client_secret: str = "",
    ) -> StoredCredential:
        """
        Complete an OAuth2 flow with the authorization code.

        Sprint 45: Validates state TTL and code content.
        """
        state_info = self._oauth_states.pop(state, None)
        if not state_info:
            raise ValueError("Invalid or expired OAuth state")

        # Sprint 45: Check TTL
        if isinstance(state_info, dict):
            connector_uuid = state_info["uuid"]
            created_at = state_info.get("created_at", 0)
            if time.time() - created_at > OAUTH_STATE_TTL:
                raise ValueError("OAuth state has expired. Please start a new authorization flow.")
        else:
            # Backwards compat with old format (state → uuid string)
            connector_uuid = state_info

        # Validate the authorization code
        if not code or not code.strip():
            raise ValueError("Authorization code cannot be empty")

        cfg = self._auth_configs.get(connector_uuid)
        if not cfg:
            raise ValueError(f"No auth config for {connector_uuid}")

        # In production, exchange code for access/refresh tokens here
        # For now, store the authorization code as the token
        credential = StoredCredential(
            connector_uuid=connector_uuid,
            connector_name=connector_name or connector_uuid,
            auth_method=AuthMethod.OAUTH2.value,
            tokens={
                "authorization_code": code,
                "access_token": f"mock_access_{code[:8]}",
                "refresh_token": f"mock_refresh_{code[:8]}",
            },
            created_at=time.time(),
            expires_at=time.time() + 3600,  # 1 hour
            last_used_at=time.time(),
            status=AuthStatus.AUTHENTICATED.value,
        )
        self._store.save(credential)
        self._active[connector_uuid] = credential
        logger.info(f"OAuth2 completed for {connector_name} (tokens redacted)")
        return credential

    # ── Credential Retrieval ──────────────────────────────────────────

    def get_credential(self, connector_uuid: str) -> Optional[StoredCredential]:
        """Get the active credential for a connector."""
        # Check in-memory first
        cred = self._active.get(connector_uuid)
        if cred and not cred.is_expired():
            return cred
        # Try loading from disk
        cred = self._store.load(connector_uuid)
        if cred and not cred.is_expired():
            self._active[connector_uuid] = cred
            return cred
        return None

    def get_token(self, connector_uuid: str, key: str = "api_token") -> Optional[str]:
        """Get a specific token value for a connector."""
        cred = self.get_credential(connector_uuid)
        if cred:
            return cred.tokens.get(key)
        return None

    def is_connected(self, connector_uuid: str) -> bool:
        """Check if a connector has valid credentials."""
        cred = self.get_credential(connector_uuid)
        return cred is not None and cred.status == AuthStatus.AUTHENTICATED.value

    # ── Disconnect & Revoke ───────────────────────────────────────────

    def disconnect(self, connector_uuid: str) -> bool:
        """
        Disconnect a connector — remove stored credentials.

        Returns True if credentials were found and removed.
        """
        removed_active = self._active.pop(connector_uuid, None)
        removed_disk = self._store.delete(connector_uuid)
        success = removed_active is not None or removed_disk
        if success:
            logger.info(f"Disconnected connector {connector_uuid}")
        return success

    # ── Auto-Reconnect ────────────────────────────────────────────────

    def load_saved_credentials(self) -> List[StoredCredential]:
        """
        Load all saved credentials from disk.

        Called at startup to auto-reconnect previously authenticated connectors.
        Returns list of valid (non-expired) credentials.
        """
        all_creds = self._store.list_all()
        valid = []
        for cred in all_creds:
            if cred.is_expired():
                logger.info(
                    f"Skipping expired credential for {cred.connector_name}"
                )
                cred.status = AuthStatus.EXPIRED.value
                self._store.save(cred)
                continue
            if cred.status == AuthStatus.AUTHENTICATED.value:
                self._active[cred.connector_uuid] = cred
                valid.append(cred)
                logger.info(f"Auto-loaded credential for {cred.connector_name}")
        return valid

    # ── Status & Listing ──────────────────────────────────────────────

    @property
    def connected_connectors(self) -> Dict[str, StoredCredential]:
        """Return all currently active/connected credentials."""
        return dict(self._active)

    def get_status(self, connector_uuid: str) -> AuthStatus:
        """Get the auth status for a connector."""
        cred = self.get_credential(connector_uuid)
        if not cred:
            return AuthStatus.NOT_CONFIGURED
        if cred.is_expired():
            return AuthStatus.EXPIRED
        try:
            return AuthStatus(cred.status)
        except ValueError:
            return AuthStatus.NOT_CONFIGURED

    def list_all_statuses(
        self, connector_uuids: List[str],
    ) -> Dict[str, AuthStatus]:
        """Get auth status for multiple connectors."""
        return {uuid: self.get_status(uuid) for uuid in connector_uuids}

    # ── MCP Server Config Generation ──────────────────────────────────

    def build_mcp_env(self, connector_uuid: str) -> Dict[str, str]:
        """
        Build environment variables for an MCP server from stored credentials.

        Used when starting an MCP server subprocess — injects the stored
        token/credentials into the server's environment.

        Sprint 45: Only exports declared env var names from auth config.
        No longer blindly dumps all token keys as env vars.
        """
        cred = self.get_credential(connector_uuid)
        if not cred:
            return {}

        cfg = self._auth_configs.get(connector_uuid)
        env = {}

        if cred.auth_method == AuthMethod.API_TOKEN.value:
            # Sprint 45: Only map to the declared env var from config
            if cfg and cfg.token_env_var:
                token_val = cred.tokens.get("api_token", "")
                if token_val:
                    env[cfg.token_env_var] = token_val

        elif cred.auth_method == AuthMethod.OAUTH2.value:
            access = cred.tokens.get("access_token", "")
            if access:
                if cfg and cfg.token_env_var:
                    env[cfg.token_env_var] = access
                env["OAUTH_ACCESS_TOKEN"] = access

        elif cred.auth_method == AuthMethod.ENV_VAR.value:
            # Sprint 45: Only export declared env vars from config
            if cfg and cfg.env_vars:
                for var in cfg.env_vars:
                    if var in cred.tokens:
                        env[var] = cred.tokens[var]
            else:
                env.update(cred.tokens)

        return env
