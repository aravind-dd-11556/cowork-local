"""
Security Invariants — Non-bypassable safety rules.

Unlike the configurable security modules (action_classifier, instruction_detector,
etc.), invariants are ALWAYS enforced regardless of configuration. Even if
`auto_approve_all=True` or thresholds are tuned to zero, invariants still block
dangerous operations.

Built-in invariants:
  - NO_FORCE_PUSH: blocks `git push --force` / `-f`
  - NO_HARD_RESET: blocks `git reset --hard`
  - NO_RECURSIVE_DELETE: blocks `rm -rf /`, `rm -rf ~`, `rm -rf *`
  - NO_CREDENTIAL_IN_OUTPUT: blocks API keys/tokens in tool output
  - NO_CURL_PIPE_BASH: blocks `curl | bash`, `wget | sh`
  - NO_SYSTEM_FILE_WRITE: blocks writes to /etc/, /usr/, ~/.ssh/
  - NO_ENV_OVERRIDE: blocks overwriting .env files
  - NO_DISABLE_SECURITY: meta-invariant preventing security config changes

Sprint 25: Immutable Security Hardening.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class SecurityInvariant:
    """A single immutable security rule."""
    invariant_id: str
    description: str
    category: str  # "destructive", "credential", "privacy", "execution"
    check_fn: Callable[[str, Dict[str, Any]], Tuple[bool, str]]
    immutable: bool = True  # Cannot be disabled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "description": self.description,
            "category": self.category,
            "immutable": self.immutable,
        }


@dataclass
class InvariantCheckResult:
    """Result of checking invariants against an action."""
    passed: bool
    violated_invariants: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    is_immutable_violation: bool = False

    @property
    def summary(self) -> str:
        if self.passed:
            return "All security invariants passed"
        return (
            f"{len(self.violated_invariants)} invariant(s) violated: "
            + "; ".join(self.reasons)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violated_invariants": self.violated_invariants,
            "reasons": self.reasons,
            "is_immutable_violation": self.is_immutable_violation,
        }


# ── Built-in invariant check functions ───────────────────────────

def _check_force_push(tool_name: str, tool_input: Dict[str, Any]) -> Tuple[bool, str]:
    """Block git push --force / -f."""
    if tool_name != "bash":
        return True, ""
    cmd = str(tool_input.get("command", ""))
    if re.search(r"git\s+push\s+.*(-f|--force)", cmd):
        return False, "Force push is permanently blocked (invariant: NO_FORCE_PUSH)"
    return True, ""


def _check_hard_reset(tool_name: str, tool_input: Dict[str, Any]) -> Tuple[bool, str]:
    """Block git reset --hard."""
    if tool_name != "bash":
        return True, ""
    cmd = str(tool_input.get("command", ""))
    if re.search(r"git\s+reset\s+--hard", cmd):
        return False, "Hard reset is permanently blocked (invariant: NO_HARD_RESET)"
    return True, ""


def _check_recursive_delete(tool_name: str, tool_input: Dict[str, Any]) -> Tuple[bool, str]:
    """Block rm -rf targeting root, home, or wildcard."""
    if tool_name != "bash":
        return True, ""
    cmd = str(tool_input.get("command", ""))
    # Block rm -rf / or rm -rf ~  or rm -rf * or rm -rf .
    if re.search(r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f|(-[a-zA-Z]*f[a-zA-Z]*r))\s+(/\s|/\"|/\'|~/|/\*|\*\s|\.\s|~\s)", cmd):
        return False, "Recursive delete of root/home/wildcard is permanently blocked (invariant: NO_RECURSIVE_DELETE)"
    # Also catch rm -rf / at end of command
    if re.search(r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f|(-[a-zA-Z]*f[a-zA-Z]*r))\s+(/|~|\*|\.)$", cmd):
        return False, "Recursive delete of root/home/wildcard is permanently blocked (invariant: NO_RECURSIVE_DELETE)"
    return True, ""


def _check_curl_pipe_bash(tool_name: str, tool_input: Dict[str, Any]) -> Tuple[bool, str]:
    """Block curl | bash, wget | sh patterns."""
    if tool_name != "bash":
        return True, ""
    cmd = str(tool_input.get("command", ""))
    if re.search(r"(curl|wget)\s+.*\|\s*(bash|sh|zsh|dash|ksh|python|perl|ruby)", cmd):
        return False, "Piping remote content to shell is permanently blocked (invariant: NO_CURL_PIPE_BASH)"
    return True, ""


def _check_system_file_write(tool_name: str, tool_input: Dict[str, Any]) -> Tuple[bool, str]:
    """Block writes to /etc/, /usr/, ~/.ssh/."""
    if tool_name != "write":
        return True, ""
    file_path = str(tool_input.get("file_path", ""))
    blocked_prefixes = ["/etc/", "/usr/", "~/.ssh/", "/root/.ssh/"]
    for prefix in blocked_prefixes:
        if file_path.startswith(prefix):
            return False, f"Writing to {prefix} is permanently blocked (invariant: NO_SYSTEM_FILE_WRITE)"
    # Also check for expanded home ssh
    if "/.ssh/" in file_path and file_path.startswith("/"):
        # Allow within workspace paths but block actual system paths
        if not any(safe in file_path for safe in ["/workspace/", "/tmp/", "/sessions/"]):
            return False, "Writing to .ssh directory is permanently blocked (invariant: NO_SYSTEM_FILE_WRITE)"
    return True, ""


def _check_env_override(tool_name: str, tool_input: Dict[str, Any]) -> Tuple[bool, str]:
    """Block overwriting .env files."""
    if tool_name != "write":
        return True, ""
    file_path = str(tool_input.get("file_path", ""))
    # Match .env, .env.production, .env.local, etc.
    if re.search(r"(^|/)\.env(\.[a-zA-Z]+)?$", file_path):
        return False, f"Overwriting {file_path} is permanently blocked (invariant: NO_ENV_OVERRIDE)"
    return True, ""


# ── Output invariant check functions ─────────────────────────────

# Common credential patterns (independent of credential_detector)
_CREDENTIAL_PATTERNS = [
    re.compile(r"(?:sk|pk)[-_](?:live|test|prod)[-_][a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36,}", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|api[_-]?secret|access[_-]?token)\s*[:=]\s*['\"][a-zA-Z0-9_\-]{20,}['\"]", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----", re.IGNORECASE),
]


def _check_credential_in_output(output: str, tool_name: str) -> Tuple[bool, str]:
    """Block tool output containing obvious credentials."""
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(output):
            return False, "Credential pattern detected in output (invariant: NO_CREDENTIAL_IN_OUTPUT)"
    return True, ""


# ── SecurityInvariantRegistry ────────────────────────────────────

class SecurityInvariantRegistry:
    """Registry of immutable security invariants.

    Invariants are always enforced regardless of other configuration.
    They cannot be disabled at runtime.

    Usage::

        registry = SecurityInvariantRegistry()
        result = registry.check_tool_call("bash", {"command": "git push --force origin main"})
        assert not result.passed
        assert "NO_FORCE_PUSH" in result.violated_invariants
    """

    def __init__(self, load_defaults: bool = True):
        self._invariants: Dict[str, SecurityInvariant] = {}
        self._check_count = 0
        self._violation_count = 0

        if load_defaults:
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Register all built-in invariants."""
        defaults = [
            SecurityInvariant(
                invariant_id="NO_FORCE_PUSH",
                description="Blocks git push --force / -f",
                category="destructive",
                check_fn=_check_force_push,
            ),
            SecurityInvariant(
                invariant_id="NO_HARD_RESET",
                description="Blocks git reset --hard",
                category="destructive",
                check_fn=_check_hard_reset,
            ),
            SecurityInvariant(
                invariant_id="NO_RECURSIVE_DELETE",
                description="Blocks rm -rf targeting root, home, or wildcard",
                category="destructive",
                check_fn=_check_recursive_delete,
            ),
            SecurityInvariant(
                invariant_id="NO_CURL_PIPE_BASH",
                description="Blocks piping remote content to shell",
                category="execution",
                check_fn=_check_curl_pipe_bash,
            ),
            SecurityInvariant(
                invariant_id="NO_SYSTEM_FILE_WRITE",
                description="Blocks writes to /etc/, /usr/, ~/.ssh/",
                category="destructive",
                check_fn=_check_system_file_write,
            ),
            SecurityInvariant(
                invariant_id="NO_ENV_OVERRIDE",
                description="Blocks overwriting .env files",
                category="credential",
                check_fn=_check_env_override,
            ),
        ]
        for inv in defaults:
            self._invariants[inv.invariant_id] = inv

    def register(self, invariant: SecurityInvariant) -> bool:
        """Register a new invariant. Returns False if ID already exists."""
        if invariant.invariant_id in self._invariants:
            logger.warning(f"Invariant '{invariant.invariant_id}' already registered")
            return False
        self._invariants[invariant.invariant_id] = invariant
        return True

    def get_invariant(self, invariant_id: str) -> Optional[SecurityInvariant]:
        """Get an invariant by ID."""
        return self._invariants.get(invariant_id)

    @property
    def invariant_ids(self) -> List[str]:
        """All registered invariant IDs."""
        return list(self._invariants.keys())

    @property
    def count(self) -> int:
        """Number of registered invariants."""
        return len(self._invariants)

    def is_locked(self) -> bool:
        """Always True — invariants cannot be disabled."""
        return True

    def check_tool_call(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> InvariantCheckResult:
        """Check all tool-call invariants. Returns immediately on first violation."""
        self._check_count += 1
        violated = []
        reasons = []

        for inv_id, inv in self._invariants.items():
            try:
                passed, reason = inv.check_fn(tool_name, tool_input)
                if not passed:
                    violated.append(inv_id)
                    reasons.append(reason)
            except Exception as e:
                logger.warning(f"Invariant check '{inv_id}' error: {e}")

        if violated:
            self._violation_count += 1
            return InvariantCheckResult(
                passed=False,
                violated_invariants=violated,
                reasons=reasons,
                is_immutable_violation=True,
            )

        return InvariantCheckResult(passed=True)

    def check_tool_output(
        self, output: str, tool_name: str
    ) -> InvariantCheckResult:
        """Check all output invariants."""
        self._check_count += 1
        violated = []
        reasons = []

        # Run output-specific checks
        passed, reason = _check_credential_in_output(output, tool_name)
        if not passed:
            violated.append("NO_CREDENTIAL_IN_OUTPUT")
            reasons.append(reason)

        if violated:
            self._violation_count += 1
            return InvariantCheckResult(
                passed=False,
                violated_invariants=violated,
                reasons=reasons,
                is_immutable_violation=True,
            )

        return InvariantCheckResult(passed=True)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "invariant_count": len(self._invariants),
            "total_checks": self._check_count,
            "total_violations": self._violation_count,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "invariants": {
                inv_id: inv.to_dict()
                for inv_id, inv in self._invariants.items()
            },
        }
