"""
Security Freeze — Prevents runtime modification of critical security settings.

Once `freeze()` is called (at the end of agent initialization), critical
security configuration keys are locked and cannot be changed. This prevents
accidental or malicious disabling of security features through config changes.

Frozen keys include:
  - security.enabled
  - security.prompt_injection.enabled
  - security.action_classifier.enabled
  - security.instruction_detector.enabled
  - security.credential_detector.enabled
  - production_hardening.auto_approve_all
  - security.invariants.enabled

Sprint 25: Immutable Security Hardening.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Default frozen keys ──────────────────────────────────────────

DEFAULT_FROZEN_KEYS: Set[str] = {
    "security.enabled",
    "security.prompt_injection.enabled",
    "security.action_classifier.enabled",
    "security.instruction_detector.enabled",
    "security.credential_detector.enabled",
    "security.privacy_guard.enabled",
    "production_hardening.auto_approve_all",
    "security.invariants.enabled",
}


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class FreezeViolation:
    """Record of an attempted config change after freeze."""
    key: str
    attempted_value: Any
    timestamp: float = field(default_factory=time.time)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "attempted_value": str(self.attempted_value),
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


# ── SecurityFreeze ───────────────────────────────────────────────

class SecurityFreeze:
    """Locks critical security configuration keys after initialization.

    Usage::

        freeze = SecurityFreeze()
        # ... initialize all security modules ...
        freeze.freeze()

        # Now attempting to change frozen keys is blocked:
        allowed, reason = freeze.validate_config_change("security.enabled", False)
        assert not allowed
    """

    def __init__(
        self,
        frozen_keys: Optional[Set[str]] = None,
    ):
        self._frozen_keys = frozen_keys or set(DEFAULT_FROZEN_KEYS)
        self._frozen = False
        self._frozen_at: Optional[float] = None
        self._violations: List[FreezeViolation] = []
        self._total_checks = 0
        self._total_blocked = 0

    # ── Freeze control ────────────────────────────────────────

    def freeze(self) -> None:
        """Lock all frozen keys. Cannot be reversed."""
        if self._frozen:
            logger.warning("SecurityFreeze: already frozen")
            return
        self._frozen = True
        self._frozen_at = time.time()
        logger.info(
            f"Security configuration frozen. {len(self._frozen_keys)} keys locked."
        )

    def is_frozen(self) -> bool:
        """Check if configuration is frozen."""
        return self._frozen

    @property
    def frozen_at(self) -> Optional[float]:
        """Timestamp when freeze was activated."""
        return self._frozen_at

    # ── Key management ────────────────────────────────────────

    def add_frozen_key(self, key: str) -> bool:
        """Add a key to the frozen set. Only works before freeze."""
        if self._frozen:
            logger.warning(f"Cannot add frozen key '{key}' — already frozen")
            return False
        self._frozen_keys.add(key)
        return True

    def remove_frozen_key(self, key: str) -> bool:
        """Remove a key from the frozen set. Only works before freeze."""
        if self._frozen:
            logger.warning(f"Cannot remove frozen key '{key}' — already frozen")
            return False
        self._frozen_keys.discard(key)
        return True

    def get_frozen_keys(self) -> List[str]:
        """Get all frozen configuration keys."""
        return sorted(self._frozen_keys)

    def is_key_frozen(self, key: str) -> bool:
        """Check if a specific key is frozen."""
        return self._frozen and key in self._frozen_keys

    # ── Validation ────────────────────────────────────────────

    def validate_config_change(
        self, key: str, value: Any
    ) -> Tuple[bool, str]:
        """Validate whether a config change is allowed.

        Returns:
            (allowed, reason) tuple. If not allowed, reason explains why.
        """
        self._total_checks += 1

        if not self._frozen:
            return True, "Configuration is not yet frozen"

        if key not in self._frozen_keys:
            return True, f"Key '{key}' is not a frozen key"

        # Key is frozen — block the change
        self._total_blocked += 1
        reason = (
            f"Configuration key '{key}' is frozen and cannot be modified. "
            f"This security setting was locked at initialization to prevent "
            f"accidental or malicious disabling of security features."
        )

        self._violations.append(FreezeViolation(
            key=key,
            attempted_value=value,
            reason=reason,
        ))

        logger.warning(f"SecurityFreeze BLOCKED config change: {key} → {value}")
        return False, reason

    # ── Queries ───────────────────────────────────────────────

    @property
    def violations(self) -> List[FreezeViolation]:
        """All recorded violations."""
        return list(self._violations)

    @property
    def violation_count(self) -> int:
        """Number of blocked config changes."""
        return len(self._violations)

    def clear_violations(self) -> None:
        """Clear violation history (audit data, doesn't unfreeze)."""
        self._violations.clear()

    # ── Stats ─────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "frozen": self._frozen,
            "frozen_keys_count": len(self._frozen_keys),
            "total_checks": self._total_checks,
            "total_blocked": self._total_blocked,
            "violations_recorded": len(self._violations),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "frozen_keys": self.get_frozen_keys(),
            "recent_violations": [
                v.to_dict() for v in self._violations[-10:]
            ],
        }
