"""
Output Sanitizer — detect and mask leaked secrets in tool output.

Complements ``SafetyChecker`` (which blocks secrets in *inputs*) by scanning
tool *outputs* before they reach conversation history or logs.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Secret patterns ────────────────────────────────────────────────
# Each tuple: (compiled regex, short label, group index for the secret)

_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Cloud provider keys
    (re.compile(r'(AKIA[0-9A-Z]{16})'), "aws_key"),
    (re.compile(r'(ASIA[0-9A-Z]{16})'), "aws_temp_key"),

    # API tokens (OpenAI, Stripe, Anthropic, etc.)
    (re.compile(r'(sk-[a-zA-Z0-9]{20,})'), "api_secret_key"),
    (re.compile(r'(sk-ant-[a-zA-Z0-9-]{20,})'), "anthropic_key"),

    # GitHub tokens
    (re.compile(r'(ghp_[a-zA-Z0-9]{36})'), "github_pat"),
    (re.compile(r'(gho_[a-zA-Z0-9]{36})'), "github_oauth"),
    (re.compile(r'(ghs_[a-zA-Z0-9]{36})'), "github_app"),
    (re.compile(r'(github_pat_[a-zA-Z0-9_]{22,})'), "github_fine_pat"),

    # Slack tokens
    (re.compile(r'(xoxb-[0-9]+-[0-9]+-[a-zA-Z0-9]+)'), "slack_bot"),
    (re.compile(r'(xoxp-[0-9]+-[0-9]+-[0-9]+-[a-f0-9]+)'), "slack_user"),

    # JWTs (three base64 segments separated by dots)
    (re.compile(r'(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})'), "jwt"),

    # Database connection URIs
    (re.compile(r'((?:postgres|mysql|mongodb|redis)://[^\s,;\"\'<>]+:[^\s,;\"\'<>]+@[^\s,;\"\'<>]+)'), "db_uri"),

    # Private keys (PEM-encoded)
    (re.compile(r'(-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----(?:.*?)-----END)', re.DOTALL), "private_key"),

    # Password / secret / key assignments
    (re.compile(r'(?i)(?:password|passwd|pwd)\s*[=:]\s*["\']([^"\']{4,})["\']'), "password_assignment"),
    (re.compile(r'(?i)api[_-]?key\s*[=:]\s*["\']([^"\']{8,})["\']'), "api_key_assignment"),
    (re.compile(r'(?i)secret[_-]?key\s*[=:]\s*["\']([^"\']{8,})["\']'), "secret_key_assignment"),
    (re.compile(r'(?i)(?:access[_-]?token|auth[_-]?token)\s*[=:]\s*["\']([^"\']{8,})["\']'), "token_assignment"),

    # Generic long hex tokens (≥32 chars, must look like a standalone token)
    (re.compile(r'(?<![a-zA-Z0-9])([a-f0-9]{40,})(?![a-zA-Z0-9])'), "hex_token"),
]


# ── Dataclass ──────────────────────────────────────────────────────

@dataclass
class SanitizationResult:
    """Result of sanitizing a string."""
    sanitized: str
    had_secrets: bool
    detected_types: List[str] = field(default_factory=list)


# ── OutputSanitizer ───────────────────────────────────────────────

class OutputSanitizer:
    """
    Scan text for leaked secrets and mask them.

    Usage::

        sanitizer = OutputSanitizer()
        result = sanitizer.sanitize(tool_output)
        if result.had_secrets:
            logger.warning("Secrets detected: %s", result.detected_types)
        safe_output = result.sanitized
    """

    def __init__(self, show_last_n: int = 4, enabled: bool = True):
        self._show_last_n = show_last_n
        self._enabled = enabled
        self._patterns = _PATTERNS

    @property
    def enabled(self) -> bool:
        """Whether output sanitization is active."""
        return self._enabled

    # ── Public API ─────────────────────────────────────────────

    def sanitize(self, text: str) -> SanitizationResult:
        """Scan *text* for secrets and return a sanitized copy."""
        if not self._enabled or not text:
            return SanitizationResult(sanitized=text, had_secrets=False)

        detected: List[str] = []
        sanitized = text

        for pattern, label in self._patterns:
            matches = list(pattern.finditer(sanitized))
            if not matches:
                continue
            if label not in detected:
                detected.append(label)
            # Replace each match (iterate in reverse to preserve offsets)
            for m in reversed(matches):
                secret = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
                masked = self.mask(secret, self._show_last_n)
                start, end = m.start(), m.end()
                # Replace the whole match region with the masked version
                sanitized = sanitized[:start] + masked + sanitized[end:]

        return SanitizationResult(
            sanitized=sanitized,
            had_secrets=len(detected) > 0,
            detected_types=detected,
        )

    def detect_secrets(self, text: str) -> List[str]:
        """Return a list of detected secret types without modifying the text."""
        if not text:
            return []
        detected: List[str] = []
        for pattern, label in self._patterns:
            if pattern.search(text) and label not in detected:
                detected.append(label)
        return detected

    def is_clean(self, text: str) -> bool:
        """Quick check: True if no secrets detected."""
        if not text:
            return True
        for pattern, _ in self._patterns:
            if pattern.search(text):
                return False
        return True

    # ── Masking ────────────────────────────────────────────────

    @staticmethod
    def mask(secret: str, show_last_n: int = 4) -> str:
        """
        Mask a secret string, showing only the last *show_last_n* characters.

        Examples:
            mask("some-long-secret-value", 4) → "***alue"
            mask("short", 4)                  → "***hort"
            mask("ab", 4)                     → "***"
        """
        if len(secret) <= show_last_n:
            return "***"
        return "***" + secret[-show_last_n:]
