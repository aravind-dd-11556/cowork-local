"""
Action Classifier — Categorizes tool calls into security tiers.

Three tiers:
  PROHIBITED       — Never auto-execute (rm -rf, force push, etc.)
  EXPLICIT_CONSENT — Requires user confirmation before executing
  REGULAR          — Safe to auto-execute

Sprint 23: Anthropic-grade security.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ActionTier(Enum):
    """Security tier for actions."""
    PROHIBITED = "prohibited"
    EXPLICIT_CONSENT = "explicit_consent"
    REGULAR = "regular"


@dataclass
class ActionClassification:
    """Classification result for a tool call."""
    tier: ActionTier
    reason: str
    blocking_rule: Optional[str] = None
    suggested_message: Optional[str] = None

    @property
    def is_allowed(self) -> bool:
        return self.tier != ActionTier.PROHIBITED

    @property
    def needs_consent(self) -> bool:
        return self.tier == ActionTier.EXPLICIT_CONSENT

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "reason": self.reason,
            "blocking_rule": self.blocking_rule,
            "suggested_message": self.suggested_message,
        }


# ── Rule definitions ──────────────────────────────────────────────

@dataclass
class ClassificationRule:
    """A single pattern-based classification rule."""
    name: str
    pattern: re.Pattern
    tier: ActionTier
    field: str = "command"  # Which input field to check
    message: str = ""

    def matches(self, input_dict: Dict[str, Any]) -> bool:
        value = input_dict.get(self.field, "")
        if not isinstance(value, str):
            return False
        return bool(self.pattern.search(value))


# ── Prohibited bash patterns ─────────────────────────────────────

PROHIBITED_BASH_RULES: List[ClassificationRule] = [
    ClassificationRule(
        name="rm_rf_root",
        pattern=re.compile(r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f|--recursive.*--force|-[a-zA-Z]*f[a-zA-Z]*r)\s"),
        tier=ActionTier.PROHIBITED,
        message="Recursive force deletion is prohibited",
    ),
    ClassificationRule(
        name="rm_rf_slash",
        pattern=re.compile(r"rm\s+.*\s+/\s*$|rm\s+.*\s+/\s+"),
        tier=ActionTier.PROHIBITED,
        message="Deleting root filesystem is prohibited",
    ),
    ClassificationRule(
        name="curl_pipe_bash",
        pattern=re.compile(r"curl\s+.*\|\s*(ba)?sh|wget\s+.*\|\s*(ba)?sh"),
        tier=ActionTier.PROHIBITED,
        message="Piping remote scripts to shell is prohibited",
    ),
    ClassificationRule(
        name="git_force_push",
        pattern=re.compile(r"git\s+push\s+.*(-f|--force)"),
        tier=ActionTier.PROHIBITED,
        message="Force push is prohibited — could destroy remote history",
    ),
    ClassificationRule(
        name="git_hard_reset",
        pattern=re.compile(r"git\s+reset\s+--hard"),
        tier=ActionTier.PROHIBITED,
        message="Hard reset is prohibited — could lose uncommitted work",
    ),
    ClassificationRule(
        name="format_disk",
        pattern=re.compile(r"mkfs\.|format\s+[A-Z]:|fdisk|dd\s+.*of=/dev/"),
        tier=ActionTier.PROHIBITED,
        message="Disk formatting/overwriting is prohibited",
    ),
    ClassificationRule(
        name="chmod_recursive_777",
        pattern=re.compile(r"chmod\s+(-R\s+)?777\s"),
        tier=ActionTier.PROHIBITED,
        message="Setting world-writable permissions is prohibited",
    ),
    ClassificationRule(
        name="eval_exec_remote",
        pattern=re.compile(r"python.*-c\s+['\"].*exec\(.*requests|eval\(.*urllib"),
        tier=ActionTier.PROHIBITED,
        message="Remote code execution via eval/exec is prohibited",
    ),
]

# ── Explicit consent bash patterns ────────────────────────────────

CONSENT_BASH_RULES: List[ClassificationRule] = [
    ClassificationRule(
        name="install_software",
        pattern=re.compile(r"pip\s+install|npm\s+install\s+-g|apt(-get)?\s+install|brew\s+install|cargo\s+install"),
        tier=ActionTier.EXPLICIT_CONSENT,
        message="Installing software requires confirmation",
    ),
    ClassificationRule(
        name="git_push",
        pattern=re.compile(r"git\s+push(?!\s+--force)"),
        tier=ActionTier.EXPLICIT_CONSENT,
        message="Pushing to remote requires confirmation",
    ),
    ClassificationRule(
        name="send_email",
        pattern=re.compile(r"sendmail|mail\s+-s|mutt|postfix"),
        tier=ActionTier.EXPLICIT_CONSENT,
        message="Sending email requires confirmation",
    ),
    ClassificationRule(
        name="network_request",
        pattern=re.compile(r"curl\s+.*(-X\s+POST|-d\s|--data)|wget\s+--post"),
        tier=ActionTier.EXPLICIT_CONSENT,
        message="Sending data via network request requires confirmation",
    ),
]

# ── Write tool patterns ──────────────────────────────────────────

CONSENT_WRITE_RULES: List[ClassificationRule] = [
    ClassificationRule(
        name="write_sensitive_file",
        pattern=re.compile(r"\.(env|pem|key|crt|p12|pfx|credentials|secret)$", re.IGNORECASE),
        tier=ActionTier.EXPLICIT_CONSENT,
        field="file_path",
        message="Writing to sensitive file requires confirmation",
    ),
    ClassificationRule(
        name="write_system_config",
        pattern=re.compile(r"/etc/|/usr/local/|~/.ssh/|~/.config/"),
        tier=ActionTier.EXPLICIT_CONSENT,
        field="file_path",
        message="Writing to system/config path requires confirmation",
    ),
]


class ActionClassifier:
    """Classifies tool calls into security tiers based on pattern matching.

    Usage::

        classifier = ActionClassifier()
        result = classifier.classify("bash", {"command": "rm -rf /"})
        if result.tier == ActionTier.PROHIBITED:
            print(f"BLOCKED: {result.reason}")
    """

    def __init__(self, extra_rules: Optional[List[ClassificationRule]] = None):
        self._rules: Dict[str, List[ClassificationRule]] = {
            "bash": PROHIBITED_BASH_RULES + CONSENT_BASH_RULES,
            "write": CONSENT_WRITE_RULES,
        }
        if extra_rules:
            for rule in extra_rules:
                # Determine tool by field; default to "bash"
                tool = "bash" if rule.field == "command" else "write"
                self._rules.setdefault(tool, []).append(rule)

        # Tools that always require explicit consent
        self._consent_tools: set[str] = set()

        # Counters for monitoring
        self._total_classifications = 0
        self._prohibited_count = 0
        self._consent_count = 0

    def classify(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> ActionClassification:
        """Classify a tool call into a security tier.

        Args:
            tool_name: Name of the tool being called
            tool_input: Input arguments to the tool

        Returns:
            ActionClassification with tier and reason
        """
        self._total_classifications += 1

        # Check pattern rules for this tool
        rules = self._rules.get(tool_name, [])
        for rule in rules:
            if rule.tier == ActionTier.PROHIBITED and rule.matches(tool_input):
                self._prohibited_count += 1
                logger.warning(f"PROHIBITED: {tool_name} matched rule '{rule.name}'")
                return ActionClassification(
                    tier=ActionTier.PROHIBITED,
                    reason=rule.message or f"Matched prohibited pattern: {rule.name}",
                    blocking_rule=rule.name,
                    suggested_message=f"This action is not allowed: {rule.message}",
                )

        # Check explicit consent rules
        for rule in rules:
            if rule.tier == ActionTier.EXPLICIT_CONSENT and rule.matches(tool_input):
                self._consent_count += 1
                return ActionClassification(
                    tier=ActionTier.EXPLICIT_CONSENT,
                    reason=rule.message or f"Requires confirmation: {rule.name}",
                    suggested_message=f"Please confirm: {rule.message}",
                )

        # Check if tool is always-consent
        if tool_name in self._consent_tools:
            self._consent_count += 1
            return ActionClassification(
                tier=ActionTier.EXPLICIT_CONSENT,
                reason=f"Tool '{tool_name}' requires explicit user consent",
            )

        # Default: regular (safe to auto-execute)
        return ActionClassification(
            tier=ActionTier.REGULAR,
            reason="Action is safe to execute",
        )

    def add_consent_tool(self, tool_name: str) -> None:
        """Mark a tool as always requiring explicit consent."""
        self._consent_tools.add(tool_name)

    def add_rule(self, tool_name: str, rule: ClassificationRule) -> None:
        """Add a custom classification rule for a tool."""
        self._rules.setdefault(tool_name, []).append(rule)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total": self._total_classifications,
            "prohibited": self._prohibited_count,
            "consent_required": self._consent_count,
            "regular": self._total_classifications - self._prohibited_count - self._consent_count,
        }
