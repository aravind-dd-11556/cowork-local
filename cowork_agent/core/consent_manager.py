"""
Consent Manager — Session-scoped user confirmation for sensitive actions.

Manages user consent requests for EXPLICIT_CONSENT tier actions.
Consent is session-scoped: no carryover between sessions.

Sprint 23: Anthropic-grade security.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConsentType(Enum):
    """Types of explicit consent."""
    DOWNLOAD_FILE = "download_file"
    SEND_MESSAGE = "send_message"
    EXECUTE_DESTRUCTIVE = "execute_destructive"
    SHARE_DATA = "share_data"
    ACCEPT_TERMS = "accept_terms"
    SUBMIT_FORM = "submit_form"
    PUBLISH_CONTENT = "publish_content"
    GRANT_PERMISSION = "grant_permission"
    INSTALL_SOFTWARE = "install_software"
    NETWORK_REQUEST = "network_request"


@dataclass
class ConsentRequest:
    """A single user consent request."""
    consent_type: ConsentType
    description: str
    details: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsentResponse:
    """User's response to a consent request."""
    approved: bool
    timestamp: float = field(default_factory=time.time)
    request: Optional[ConsentRequest] = None


class ConsentManager:
    """Manages user confirmation for sensitive actions.

    Session-scoped: no carryover between sessions. Each ConsentManager
    instance starts with a clean history.

    Usage::

        manager = ConsentManager(ask_user_callback=my_callback)
        approved = manager.request_consent(
            ConsentType.DOWNLOAD_FILE,
            "Download report.pdf from example.com",
            details={"filename": "report.pdf", "source": "example.com"},
        )
        if approved:
            # Proceed with download
            ...
    """

    def __init__(self, ask_user_callback: Optional[Callable] = None):
        self._ask_user = ask_user_callback
        self._consent_history: List[ConsentResponse] = []
        self._total_requests = 0
        self._total_approved = 0
        self._total_declined = 0

    def request_consent(
        self,
        consent_type: ConsentType,
        description: str,
        details: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Request user confirmation for an action.

        Args:
            consent_type: Category of the action
            description: Human-readable description
            details: Additional context (filename, source, etc.)

        Returns:
            True if user approved, False if declined or no callback
        """
        self._total_requests += 1
        details = details or {}

        request = ConsentRequest(
            consent_type=consent_type,
            description=description,
            details=details,
        )

        if not self._ask_user:
            logger.warning(
                f"No ask_user callback — auto-declining consent for: {description}"
            )
            self._record_response(request, approved=False)
            return False

        # Format the consent message
        message = self._format_consent_message(consent_type, description, details)

        try:
            response = self._ask_user(message)
            approved = self._interpret_response(response)
        except Exception as e:
            logger.warning(f"Consent request failed: {e}")
            approved = False

        self._record_response(request, approved)
        return approved

    def _format_consent_message(
        self,
        consent_type: ConsentType,
        description: str,
        details: Dict[str, str],
    ) -> str:
        """Format a human-readable consent message."""
        templates = {
            ConsentType.DOWNLOAD_FILE: (
                f"Download {details.get('filename', 'file')} "
                f"from {details.get('source', 'unknown source')}?"
            ),
            ConsentType.SEND_MESSAGE: f"Send message: {description}?",
            ConsentType.EXECUTE_DESTRUCTIVE: (
                f"Execute potentially destructive action: {description}?"
            ),
            ConsentType.SHARE_DATA: f"Share data: {description}?",
            ConsentType.ACCEPT_TERMS: f"Accept terms: {description}?",
            ConsentType.SUBMIT_FORM: "Submit form with provided information?",
            ConsentType.PUBLISH_CONTENT: f"Publish content: {description}?",
            ConsentType.GRANT_PERMISSION: f"Grant permission: {description}?",
            ConsentType.INSTALL_SOFTWARE: f"Install software: {description}?",
            ConsentType.NETWORK_REQUEST: f"Send network request: {description}?",
        }
        return templates.get(consent_type, f"Confirm action: {description}?")

    def _interpret_response(self, response: str) -> bool:
        """Interpret user response as approval or denial."""
        if not response:
            return False
        normalized = response.strip().lower()
        approve_words = {"yes", "y", "approve", "ok", "okay", "sure", "confirm", "go ahead", "proceed"}
        return normalized in approve_words or normalized.startswith(("yes", "approve"))

    def _record_response(self, request: ConsentRequest, approved: bool) -> None:
        """Record consent response in session history."""
        response = ConsentResponse(
            approved=approved,
            request=request,
        )
        self._consent_history.append(response)
        if approved:
            self._total_approved += 1
        else:
            self._total_declined += 1

    @property
    def history(self) -> List[ConsentResponse]:
        """Get consent history for this session."""
        return list(self._consent_history)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_requests": self._total_requests,
            "total_approved": self._total_approved,
            "total_declined": self._total_declined,
        }

    def clear(self) -> None:
        """Clear consent history (for session reset)."""
        self._consent_history.clear()
