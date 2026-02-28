"""
Privacy Guard — Prevents unintended exposure of sensitive PII and financial data.

Rules:
  - Never auto-fill financial data in form fields
  - Decline cookies by default
  - Refuse CAPTCHA completion
  - Flag and ask for confirmation on PII transmission
  - Detect credit card, SSN, bank account, API key, password patterns

Sprint 23: Anthropic-grade security.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Sensitive field patterns ──────────────────────────────────────

SENSITIVE_PATTERNS: Dict[str, List[Tuple[re.Pattern, str]]] = {
    "credit_card": [
        (re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"), "credit_card_number"),
        (re.compile(r"(?i)\b(?:visa|mastercard|amex|discover)\s*[:=]\s*\d"), "credit_card_label"),
    ],
    "ssn": [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "us_ssn"),
        (re.compile(r"(?i)\bssn\s*[:=]\s*\d{3}"), "ssn_label"),
    ],
    "bank_account": [
        (re.compile(r"(?i)(?:account|routing)\s*(?:number|num|#|no)\s*[:=]\s*\d{6,17}"), "bank_account"),
        (re.compile(r"(?i)\bIBAN\s*[:=]?\s*[A-Z]{2}\d{2}"), "iban"),
        (re.compile(r"(?i)\bSWIFT\s*[:=]?\s*[A-Z]{6}[A-Z0-9]{2,5}"), "swift_code"),
    ],
    "api_key": [
        (re.compile(r"(?i)(?:api[_\s]?key|api[_\s]?token|access[_\s]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}"), "api_key"),
        (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9]{20,}\b"), "secret_key_pattern"),
    ],
    "password": [
        (re.compile(r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{8,}"), "password"),
    ],
    "personal_id": [
        (re.compile(r"(?i)(?:passport|license|permit)\s*(?:number|num|#|no)\s*[:=]\s*[A-Z0-9]{5,}"), "personal_id_number"),
        (re.compile(r"(?i)(?:driver'?s?\s+license|DL)\s*[:=]?\s*[A-Z0-9]{5,}"), "drivers_license"),
    ],
}

# ── Risk level mapping ────────────────────────────────────────────

CATEGORY_RISK: Dict[str, str] = {
    "credit_card": "critical",
    "ssn": "critical",
    "bank_account": "critical",
    "personal_id": "high",
    "api_key": "high",
    "password": "high",
}


@dataclass
class SensitiveFieldDetectionResult:
    """Result of sensitive field scan."""
    has_sensitive_fields: bool
    fields_found: List[Dict[str, str]] = field(default_factory=list)
    risk_level: str = "low"  # "low", "medium", "high", "critical"

    def to_dict(self) -> dict:
        return {
            "has_sensitive_fields": self.has_sensitive_fields,
            "fields_found": self.fields_found,
            "risk_level": self.risk_level,
        }


class PrivacyGuard:
    """Prevents unintended exposure of sensitive PII and financial data.

    Usage::

        guard = PrivacyGuard()
        result = guard.scan_for_sensitive_fields("My SSN is 123-45-6789")
        if result.has_sensitive_fields:
            print(f"WARNING: {result.risk_level} — {result.fields_found}")
    """

    def __init__(
        self,
        auto_decline_cookies: bool = True,
        refuse_captcha: bool = True,
        extra_patterns: Optional[Dict[str, List[Tuple[re.Pattern, str]]]] = None,
    ):
        self._auto_decline_cookies = auto_decline_cookies
        self._refuse_captcha = refuse_captcha
        self._patterns = dict(SENSITIVE_PATTERNS)
        if extra_patterns:
            for cat, pats in extra_patterns.items():
                self._patterns.setdefault(cat, []).extend(pats)
        self._total_scans = 0
        self._total_detections = 0

    def scan_for_sensitive_fields(self, text: str) -> SensitiveFieldDetectionResult:
        """Scan text for PII / financial data patterns.

        Args:
            text: Text to scan

        Returns:
            SensitiveFieldDetectionResult with detected fields and risk level
        """
        self._total_scans += 1

        if not text:
            return SensitiveFieldDetectionResult(has_sensitive_fields=False)

        fields_found: List[Dict[str, str]] = []
        max_risk = "low"

        for category, patterns_list in self._patterns.items():
            for pattern, field_name in patterns_list:
                for match in pattern.finditer(text):
                    fields_found.append({
                        "type": field_name,
                        "category": category,
                        "snippet": match.group(0)[:30] + ("..." if len(match.group(0)) > 30 else ""),
                    })
                    category_risk = CATEGORY_RISK.get(category, "medium")
                    if _risk_order(category_risk) > _risk_order(max_risk):
                        max_risk = category_risk

        if fields_found:
            self._total_detections += 1

        return SensitiveFieldDetectionResult(
            has_sensitive_fields=len(fields_found) > 0,
            fields_found=fields_found,
            risk_level=max_risk,
        )

    def should_auto_decline_cookies(self) -> bool:
        """Return True to decline cookies by default."""
        return self._auto_decline_cookies

    def should_refuse_captcha(self) -> bool:
        """Return True to refuse CAPTCHA completion."""
        return self._refuse_captcha

    def should_auto_fill_sensitive_field(self, field_type: str) -> bool:
        """Determine if a sensitive field should be auto-filled.

        Always returns False — sensitive fields must NEVER be auto-filled.
        """
        return False

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_scans": self._total_scans,
            "total_detections": self._total_detections,
        }


def _risk_order(level: str) -> int:
    """Numeric ordering for risk levels."""
    return {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(level, 0)
