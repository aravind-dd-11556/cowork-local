"""
Tool Permissions — per-session tool access control with built-in profiles.

Provides allow/deny lists and per-tool execution quotas. The permission
manager is consulted before every tool execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Dataclasses ────────────────────────────────────────────────────

@dataclass
class ToolQuota:
    """Per-tool call quota within a session."""
    tool_name: str
    max_calls: Optional[int] = None     # None = unlimited
    current_calls: int = 0

    @property
    def remaining(self) -> Optional[int]:
        if self.max_calls is None:
            return None
        return max(0, self.max_calls - self.current_calls)

    @property
    def exhausted(self) -> bool:
        if self.max_calls is None:
            return False
        return self.current_calls >= self.max_calls

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "max_calls": self.max_calls,
            "current_calls": self.current_calls,
            "remaining": self.remaining,
        }


@dataclass
class PermissionProfile:
    """
    A named permission profile controlling tool access.

    - ``allowed_tools``: if set (not None), only these tools can run.
    - ``denied_tools``: these tools are always blocked.
    - Deny takes precedence over allow.
    """
    name: str
    description: str = ""
    allowed_tools: Optional[Set[str]] = None   # None = all allowed
    denied_tools: Set[str] = field(default_factory=set)
    quotas: Dict[str, ToolQuota] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "allowed_tools": sorted(self.allowed_tools) if self.allowed_tools is not None else None,
            "denied_tools": sorted(self.denied_tools),
            "quotas": {k: v.to_dict() for k, v in self.quotas.items()},
        }


# ── Built-in profiles ─────────────────────────────────────────────

_BUILTIN_PROFILES: Dict[str, PermissionProfile] = {
    "full_access": PermissionProfile(
        name="full_access",
        description="All tools allowed, no restrictions.",
    ),
    "read_only": PermissionProfile(
        name="read_only",
        description="Read-only tools only — no writing, execution, or deletion.",
        denied_tools={
            "bash", "write", "edit", "delete_file", "notebook_edit",
        },
    ),
    "safe_mode": PermissionProfile(
        name="safe_mode",
        description="Only safe, non-destructive tools allowed.",
        allowed_tools={
            "read", "glob", "grep", "web_search", "web_fetch",
            "todo_write", "ask_user",
        },
    ),
}


# ── ToolPermissionManager ──────────────────────────────────────────

class ToolPermissionManager:
    """
    Manages tool access control for the current session.

    Usage::

        perm = ToolPermissionManager(default_profile="full_access")

        # Check before executing
        allowed, reason = perm.check_permission("bash")
        if not allowed:
            return ToolResult(success=False, output=reason)

        # After successful execution
        perm.record_call("bash")
    """

    def __init__(self, default_profile: str = "full_access"):
        self._profiles = dict(_BUILTIN_PROFILES)
        self._active_profile: PermissionProfile = self._profiles.get(
            default_profile, _BUILTIN_PROFILES["full_access"]
        )
        self._call_log: Dict[str, int] = {}    # tool_name → cumulative calls

    # ── Profile management ─────────────────────────────────────

    def set_profile(self, profile: PermissionProfile) -> None:
        """Activate a permission profile for this session."""
        self._active_profile = profile
        logger.info("Permission profile set to: %s", profile.name)

    def set_profile_by_name(self, name: str) -> bool:
        """Activate a built-in or custom profile by name. Returns False if not found."""
        profile = self._profiles.get(name)
        if not profile:
            return False
        self._active_profile = profile
        return True

    def get_active_profile(self) -> PermissionProfile:
        """Return the currently active permission profile."""
        return self._active_profile

    def create_profile(
        self,
        name: str,
        description: str = "",
        allowed: Optional[List[str]] = None,
        denied: Optional[List[str]] = None,
        quotas: Optional[Dict[str, int]] = None,
    ) -> PermissionProfile:
        """
        Create and register a custom permission profile.

        Args:
            name: Profile name.
            allowed: Allow-listed tool names (None = all).
            denied: Deny-listed tool names.
            quotas: Dict of tool_name → max_calls.
        """
        tool_quotas = {}
        if quotas:
            for tool_name, max_calls in quotas.items():
                tool_quotas[tool_name] = ToolQuota(tool_name=tool_name, max_calls=max_calls)

        profile = PermissionProfile(
            name=name,
            description=description,
            allowed_tools=set(allowed) if allowed is not None else None,
            denied_tools=set(denied) if denied else set(),
            quotas=tool_quotas,
        )
        self._profiles[name] = profile
        return profile

    def list_profiles(self) -> List[PermissionProfile]:
        """List all registered profiles."""
        return list(self._profiles.values())

    # ── Permission checking ────────────────────────────────────

    def check_permission(self, tool_name: str) -> Tuple[bool, str]:
        """
        Check if a tool is allowed under the active profile.

        Returns ``(True, "")`` if allowed, ``(False, reason)`` if denied.
        """
        profile = self._active_profile

        # Deny list takes precedence
        if tool_name in profile.denied_tools:
            reason = f"Tool '{tool_name}' is denied by profile '{profile.name}'."
            return False, reason

        # Allow list (if set)
        if profile.allowed_tools is not None and tool_name not in profile.allowed_tools:
            reason = f"Tool '{tool_name}' is not in the allow-list of profile '{profile.name}'."
            return False, reason

        return True, ""

    def check_quota(self, tool_name: str) -> Tuple[bool, str]:
        """
        Check if the tool is within its call quota.

        Returns ``(True, "")`` if within quota, ``(False, reason)`` if exhausted.
        """
        profile = self._active_profile
        quota = profile.quotas.get(tool_name)
        if not quota or quota.max_calls is None:
            return True, ""

        if quota.exhausted:
            reason = (
                f"Tool '{tool_name}' has reached its quota "
                f"({quota.current_calls}/{quota.max_calls} calls)."
            )
            return False, reason

        return True, ""

    def check_all(self, tool_name: str) -> Tuple[bool, str]:
        """Combined permission + quota check."""
        allowed, reason = self.check_permission(tool_name)
        if not allowed:
            return False, reason
        return self.check_quota(tool_name)

    # ── Recording ──────────────────────────────────────────────

    def record_call(self, tool_name: str) -> None:
        """Record a tool call for quota tracking."""
        # Update profile quota
        quota = self._active_profile.quotas.get(tool_name)
        if quota:
            quota.current_calls += 1

        # Global call log
        self._call_log[tool_name] = self._call_log.get(tool_name, 0) + 1

    def get_call_counts(self) -> Dict[str, int]:
        """Return cumulative call counts per tool."""
        return dict(self._call_log)

    # ── Reset ──────────────────────────────────────────────────

    def reset_quotas(self) -> None:
        """Reset all quota counters (e.g. for a new session)."""
        for quota in self._active_profile.quotas.values():
            quota.current_calls = 0
        self._call_log.clear()
