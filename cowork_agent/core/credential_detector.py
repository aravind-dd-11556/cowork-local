"""
Credential Detector — detect and redact secrets in text.

Scans for API keys, tokens, passwords, and other sensitive credentials.
Supports mask, hash, and remove redaction strategies.

Sprint 17 (Security & Sandboxing) Module 3.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Credential patterns ──────────────────────────────────────────

class CredentialType(Enum):
    """Types of credentials that can be detected."""
    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"
    AZURE_KEY = "azure_key"
    GCP_KEY = "gcp_key"
    GITHUB_TOKEN = "github_token"
    GITLAB_TOKEN = "gitlab_token"
    SLACK_TOKEN = "slack_token"
    OPENAI_KEY = "openai_key"
    ANTHROPIC_KEY = "anthropic_key"
    GENERIC_API_KEY = "generic_api_key"
    JWT_TOKEN = "jwt_token"
    SSH_PRIVATE_KEY = "ssh_private_key"
    PGP_PRIVATE_KEY = "pgp_private_key"
    DATABASE_URL = "database_url"
    PASSWORD_IN_URL = "password_in_url"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    PRIVATE_KEY_PEM = "private_key_pem"
    OAUTH_SECRET = "oauth_secret"
    GENERIC_SECRET = "generic_secret"


CREDENTIAL_PATTERNS: List[Tuple[re.Pattern, CredentialType, str]] = [
    # AWS
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), CredentialType.AWS_ACCESS_KEY, "AWS Access Key ID"),
    (re.compile(r"(?i)aws[_\-]?secret[_\-]?access[_\-]?key\s*[=:]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"),
     CredentialType.AWS_SECRET_KEY, "AWS Secret Access Key"),

    # Azure
    (re.compile(r"(?i)azure[_\-]?(?:storage|account)[_\-]?key\s*[=:]\s*['\"]?([A-Za-z0-9+/=]{44,})['\"]?"),
     CredentialType.AZURE_KEY, "Azure Storage Key"),

    # GCP
    (re.compile(r"(?i)google[_\-]?(?:api|cloud)[_\-]?key\s*[=:]\s*['\"]?(AIza[A-Za-z0-9_-]{35})['\"]?"),
     CredentialType.GCP_KEY, "GCP API Key"),

    # GitHub
    (re.compile(r"\b(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82})\b"),
     CredentialType.GITHUB_TOKEN, "GitHub Personal Access Token"),
    (re.compile(r"\b(gho_[A-Za-z0-9]{36}|ghs_[A-Za-z0-9]{36}|ghr_[A-Za-z0-9]{36})\b"),
     CredentialType.GITHUB_TOKEN, "GitHub Token"),

    # GitLab
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
     CredentialType.GITLAB_TOKEN, "GitLab Personal Access Token"),

    # Slack
    (re.compile(r"\bxox[bpsorta]-[A-Za-z0-9-]{10,}\b"),
     CredentialType.SLACK_TOKEN, "Slack Token"),

    # OpenAI
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
     CredentialType.OPENAI_KEY, "OpenAI API Key"),

    # Anthropic
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
     CredentialType.ANTHROPIC_KEY, "Anthropic API Key"),

    # JWT
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
     CredentialType.JWT_TOKEN, "JWT Token"),

    # SSH Private Key
    (re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
     CredentialType.SSH_PRIVATE_KEY, "SSH Private Key"),

    # PGP Private Key
    (re.compile(r"-----BEGIN PGP PRIVATE KEY BLOCK-----"),
     CredentialType.PGP_PRIVATE_KEY, "PGP Private Key"),

    # PEM Private Key
    (re.compile(r"-----BEGIN PRIVATE KEY-----"),
     CredentialType.PRIVATE_KEY_PEM, "PEM Private Key"),

    # Database URLs
    (re.compile(r"(?i)(mysql|postgres|postgresql|mongodb|redis|mssql)://[^\s\"']+:[^\s\"']+@[^\s\"']+"),
     CredentialType.DATABASE_URL, "Database Connection URL"),

    # Password in URL
    (re.compile(r"(?i)https?://[^:]+:[^@]+@[^\s\"']+"),
     CredentialType.PASSWORD_IN_URL, "Password in URL"),

    # Bearer token
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}"),
     CredentialType.BEARER_TOKEN, "Bearer Token"),

    # Basic auth
    (re.compile(r"(?i)basic\s+[A-Za-z0-9+/=]{20,}"),
     CredentialType.BASIC_AUTH, "Basic Auth Credentials"),

    # OAuth client secret
    (re.compile(r"(?i)client[_\-]?secret\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{20,})['\"]?"),
     CredentialType.OAUTH_SECRET, "OAuth Client Secret"),

    # Generic API key patterns
    (re.compile(r"(?i)(?:api[_\-]?key|apikey)\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{20,})['\"]?"),
     CredentialType.GENERIC_API_KEY, "Generic API Key"),

    # Generic secret
    (re.compile(r"(?i)(?:secret|password|passwd|pwd)\s*[=:]\s*['\"]?([^\s'\"]{8,})['\"]?"),
     CredentialType.GENERIC_SECRET, "Generic Secret/Password"),
]


# ── Redaction strategies ─────────────────────────────────────────

class RedactionStrategy(Enum):
    MASK = "mask"        # Replace with ***REDACTED***
    HASH = "hash"        # Replace with SHA-256 hash prefix
    REMOVE = "remove"    # Remove entirely


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class CredentialMatch:
    """A single detected credential."""
    credential_type: CredentialType
    description: str
    matched_text: str
    start: int
    end: int
    redacted_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.credential_type.value,
            "description": self.description,
            "position": {"start": self.start, "end": self.end},
            "redacted": self.redacted_text,
        }


@dataclass
class CredentialScanResult:
    """Result of credential scanning."""
    has_credentials: bool
    matches: List[CredentialMatch] = field(default_factory=list)
    redacted_text: str = ""
    credential_types_found: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.has_credentials:
            return "no credentials detected"
        types = ", ".join(self.credential_types_found[:5])
        return f"{len(self.matches)} credential(s) detected: {types}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_credentials": self.has_credentials,
            "match_count": len(self.matches),
            "credential_types": self.credential_types_found,
            "matches": [m.to_dict() for m in self.matches],
        }


# ── CredentialDetector ───────────────────────────────────────────

class CredentialDetector:
    """
    Detect and redact credentials in text.

    Usage::

        det = CredentialDetector()
        result = det.scan("My key is AKIAIOSFODNN7EXAMPLE")
        if result.has_credentials:
            safe_text = result.redacted_text
    """

    def __init__(
        self,
        strategy: RedactionStrategy = RedactionStrategy.MASK,
        enabled_types: Optional[List[CredentialType]] = None,
        max_scan_length: int = 1_000_000,
        mask_text: str = "***REDACTED***",
    ):
        self._strategy = strategy
        self._enabled_types = set(enabled_types) if enabled_types else None  # None = all
        self._max_scan_length = max_scan_length
        self._mask_text = mask_text
        self._total_scans = 0
        self._total_credentials_found = 0

    def scan(self, text: str, redact: bool = True) -> CredentialScanResult:
        """
        Scan text for credentials, optionally redacting them.

        Args:
            text: Text to scan
            redact: Whether to produce redacted text

        Returns:
            CredentialScanResult with matches and optionally redacted text
        """
        self._total_scans += 1

        if not text:
            return CredentialScanResult(has_credentials=False, redacted_text=text)

        scan_text = text[:self._max_scan_length]
        matches: List[CredentialMatch] = []

        for pattern, cred_type, description in CREDENTIAL_PATTERNS:
            if self._enabled_types is not None and cred_type not in self._enabled_types:
                continue

            for match in pattern.finditer(scan_text):
                matched_text = match.group()
                cred_match = CredentialMatch(
                    credential_type=cred_type,
                    description=description,
                    matched_text=matched_text,
                    start=match.start(),
                    end=match.end(),
                    redacted_text=self._redact(matched_text),
                )
                matches.append(cred_match)

        # Deduplicate overlapping matches (keep longer match)
        matches = self._deduplicate_matches(matches)
        self._total_credentials_found += len(matches)

        credential_types = sorted(set(m.credential_type.value for m in matches))

        # Build redacted text
        redacted_text = text
        if redact and matches:
            redacted_text = self._apply_redactions(text, matches)

        return CredentialScanResult(
            has_credentials=len(matches) > 0,
            matches=matches,
            redacted_text=redacted_text,
            credential_types_found=credential_types,
        )

    def scan_dict(self, data: Dict[str, Any], redact: bool = True) -> CredentialScanResult:
        """Scan all string values in a dictionary."""
        all_matches: List[CredentialMatch] = []
        redacted_data = dict(data)

        for key, value in data.items():
            if isinstance(value, str):
                result = self.scan(value, redact=redact)
                all_matches.extend(result.matches)
                if redact and result.has_credentials:
                    redacted_data[key] = result.redacted_text

        credential_types = sorted(set(m.credential_type.value for m in all_matches))

        return CredentialScanResult(
            has_credentials=len(all_matches) > 0,
            matches=all_matches,
            redacted_text=str(redacted_data),
            credential_types_found=credential_types,
        )

    def _redact(self, text: str) -> str:
        """Apply redaction strategy to matched text."""
        if self._strategy == RedactionStrategy.MASK:
            return self._mask_text
        elif self._strategy == RedactionStrategy.HASH:
            h = hashlib.sha256(text.encode()).hexdigest()[:12]
            return f"[HASH:{h}]"
        elif self._strategy == RedactionStrategy.REMOVE:
            return ""
        return self._mask_text

    def _deduplicate_matches(self, matches: List[CredentialMatch]) -> List[CredentialMatch]:
        """Remove overlapping matches, keeping the longest."""
        if len(matches) <= 1:
            return matches

        # Sort by start position, then by length (longest first)
        sorted_matches = sorted(matches, key=lambda m: (m.start, -(m.end - m.start)))
        result: List[CredentialMatch] = []
        last_end = -1

        for match in sorted_matches:
            if match.start >= last_end:
                result.append(match)
                last_end = match.end

        return result

    def _apply_redactions(self, text: str, matches: List[CredentialMatch]) -> str:
        """Apply redactions to text, replacing matches from end to start."""
        # Sort by position (reverse) to avoid offset issues
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)
        result = text
        for match in sorted_matches:
            result = result[:match.start] + match.redacted_text + result[match.end:]
        return result

    # ── Stats ──────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return detector statistics."""
        return {
            "total_scans": self._total_scans,
            "total_credentials_found": self._total_credentials_found,
            "strategy": self._strategy.value,
            "detection_rate": (
                self._total_credentials_found / self._total_scans
                if self._total_scans > 0 else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset scan statistics."""
        self._total_scans = 0
        self._total_credentials_found = 0
