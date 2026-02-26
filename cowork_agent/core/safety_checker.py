"""
Safety Checker — Pre-screens tool calls for dangerous patterns.

Covers:
  1. Destructive bash command detection (rm -rf, dd, mkfs, force push, etc.)
  2. Path traversal prevention (../../../etc/passwd)
  3. Secrets detection (API keys, passwords in content)
  4. Tool input validation (required fields, known tool names)

The checker sits between the LLM response parser and tool execution.
It can BLOCK a call (return error), WARN (add notice), or PASS.
"""

from __future__ import annotations
import logging
import os
import re
from typing import Optional

from .models import ToolCall, ToolResult, ToolSchema

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. Destructive Command Patterns
# ─────────────────────────────────────────────────────────────

DESTRUCTIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Filesystem destruction
    (re.compile(r'\brm\s+(-\w*[rf]\w*\s+)+/(?!\w)'), "rm -rf / (recursive delete from root)"),
    (re.compile(r'\brm\s+(-\w*[rf]\w*\s+)+~'), "rm -rf ~ (recursive delete home directory)"),
    (re.compile(r'\brm\s+(-\w*[rf]\w*\s+)+\*'), "rm -rf * (recursive delete everything)"),
    (re.compile(r'\bdd\s+.*\bof=/dev/'), "dd writing to device (can destroy disk)"),
    (re.compile(r'\bmkfs\b'), "mkfs (format filesystem)"),
    (re.compile(r'\b:\s*>\s*/'), "truncating system file"),
    (re.compile(r'>\s*/dev/sd[a-z]'), "writing directly to disk device"),
    (re.compile(r'\bformat\s+[A-Za-z]:'), "formatting drive"),

    # Git destruction
    (re.compile(r'\bgit\s+push\s+.*--force\b'), "git push --force (can destroy remote history)"),
    (re.compile(r'\bgit\s+push\s+.*-f\b'), "git push -f (force push)"),
    (re.compile(r'\bgit\s+reset\s+--hard\b'), "git reset --hard (discards all local changes)"),
    (re.compile(r'\bgit\s+clean\s+.*-f'), "git clean -f (deletes untracked files)"),
    (re.compile(r'\bgit\s+checkout\s+\.\s*$'), "git checkout . (discards all changes)"),
    (re.compile(r'\bgit\s+restore\s+\.\s*$'), "git restore . (discards all changes)"),

    # Permission/security
    (re.compile(r'\bchmod\s+777\b'), "chmod 777 (opens all permissions)"),
    (re.compile(r'\bchmod\s+-R\s+777\b'), "chmod -R 777 (recursively opens all permissions)"),
    (re.compile(r'\bchown\s+-R\b.*\broot\b'), "chown -R root (changing ownership to root)"),

    # System-level danger
    (re.compile(r'\b(shutdown|reboot|halt|poweroff)\b'), "system shutdown/reboot"),
    (re.compile(r'\bsystemctl\s+(stop|disable|mask)\b'), "stopping/disabling system services"),
    (re.compile(r'\bkill\s+-9\s+-1\b'), "kill -9 -1 (kills all processes)"),
    (re.compile(r'\bkillall\b'), "killall (kills processes by name)"),
    (re.compile(r'\b:(){ :\|:& };:'), "fork bomb"),

    # Network exfiltration
    (re.compile(r'\bcurl\b.*\|\s*bash\b'), "curl | bash (executing remote scripts)"),
    (re.compile(r'\bwget\b.*\|\s*sh\b'), "wget | sh (executing remote scripts)"),
]

# Patterns that need WARNING but not blocking
WARN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\brm\s+-'), "rm with flags (double-check target)"),
    (re.compile(r'\bgit\s+push\b'), "git push (pushing to remote)"),
    (re.compile(r'\bsudo\b'), "sudo (elevated privileges)"),
    (re.compile(r'\bchmod\b'), "chmod (changing permissions)"),
    (re.compile(r'\bpip\s+install\b'), "pip install (installing packages)"),
    (re.compile(r'\bnpm\s+install\s+-g\b'), "npm install -g (global package install)"),
]


# ─────────────────────────────────────────────────────────────
# 2. Secret Patterns (for detecting secrets in file content)
# ─────────────────────────────────────────────────────────────

SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'AKIA[0-9A-Z]{16}'), "AWS Access Key ID"),
    (re.compile(r'sk-[a-zA-Z0-9]{20,}'), "API secret key (OpenAI/Stripe style)"),
    (re.compile(r'ghp_[a-zA-Z0-9]{36}'), "GitHub Personal Access Token"),
    (re.compile(r'gho_[a-zA-Z0-9]{36}'), "GitHub OAuth Token"),
    (re.compile(r'xoxb-[0-9]+-[0-9]+-[a-zA-Z0-9]+'), "Slack Bot Token"),
    (re.compile(r'xoxp-[0-9]+-[0-9]+-[0-9]+-[a-f0-9]+'), "Slack User Token"),
    (re.compile(r'eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.'), "JWT Token"),
    (re.compile(r'(?i)password\s*[=:]\s*["\'][^"\']{4,}'), "Hardcoded password"),
    (re.compile(r'(?i)api[_-]?key\s*[=:]\s*["\'][^"\']{8,}'), "Hardcoded API key"),
    (re.compile(r'(?i)secret[_-]?key\s*[=:]\s*["\'][^"\']{8,}'), "Hardcoded secret key"),
    (re.compile(r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----'), "Private key"),
]


class SafetyChecker:
    """
    Pre-execution safety checker for tool calls.

    Usage:
        checker = SafetyChecker(workspace_dir="/path/to/workspace")
        result = checker.check(tool_call, tool_schemas)
        if result.blocked:
            return result.to_tool_result(tool_call.tool_id)
        if result.warnings:
            logger.warning(result.warnings)
    """

    def __init__(self, workspace_dir: str = ""):
        self._workspace = os.path.realpath(workspace_dir) if workspace_dir else ""

    class CheckResult:
        """Result of a safety check."""

        def __init__(self):
            self.blocked: bool = False
            self.block_reason: str = ""
            self.warnings: list[str] = []

        def to_tool_result(self, tool_id: str = "") -> ToolResult:
            """Convert a blocked result to a ToolResult error."""
            return ToolResult(
                tool_id=tool_id,
                success=False,
                output="",
                error=f"[SAFETY] Blocked: {self.block_reason}",
            )

    def check(self, call: ToolCall, schemas: list[ToolSchema] | None = None) -> CheckResult:
        """
        Run all safety checks on a tool call.

        Returns CheckResult with blocked=True if the call should be prevented.
        """
        result = self.CheckResult()

        # 1. Check tool name exists (hallucination detection)
        if schemas is not None:
            known_names = {s.name for s in schemas}
            if call.name not in known_names:
                result.blocked = True
                result.block_reason = (
                    f"Unknown tool '{call.name}'. "
                    f"Available tools: {', '.join(sorted(known_names))}"
                )
                return result

        # 2. Check tool-specific safety
        if call.name == "bash":
            self._check_bash(call, result)
        if call.name in ("read", "write", "edit", "glob", "grep"):
            self._check_file_path(call, result)
        if call.name == "write" and not result.blocked:
            self._check_secrets_in_content(call, result)

        return result

    def _check_bash(self, call: ToolCall, result: CheckResult) -> None:
        """Check bash commands for destructive patterns."""
        command = call.input.get("command", "")
        if not command:
            return

        # Check BLOCK patterns first
        for pattern, description in DESTRUCTIVE_PATTERNS:
            if pattern.search(command):
                result.blocked = True
                result.block_reason = (
                    f"Destructive command detected: {description}. "
                    f"Command: {command[:100]}. "
                    f"This command was blocked for safety. If you need to run this, "
                    f"ask the user for explicit confirmation first."
                )
                logger.warning(f"BLOCKED dangerous bash command: {description} → {command[:200]}")
                return

        # Check WARN patterns
        for pattern, description in WARN_PATTERNS:
            if pattern.search(command):
                result.warnings.append(f"Caution: {description}")
                logger.info(f"Safety warning for bash command: {description}")

    def _check_file_path(self, call: ToolCall, result: CheckResult) -> None:
        """Check file paths for traversal attacks."""
        # Get the file path from various possible input keys
        file_path = (
            call.input.get("file_path", "")
            or call.input.get("path", "")
            or call.input.get("pattern", "")
        )
        if not file_path:
            return

        # Check for path traversal
        if self._workspace:
            try:
                resolved = os.path.realpath(file_path)
                if not resolved.startswith(self._workspace):
                    # Allow common system paths that are read-only safe
                    safe_prefixes = ("/usr", "/bin", "/etc", "/tmp", "/var/tmp")
                    if call.name == "read" and any(resolved.startswith(p) for p in safe_prefixes):
                        result.warnings.append(
                            f"Reading file outside workspace: {resolved}"
                        )
                    elif call.name in ("write", "edit"):
                        result.blocked = True
                        result.block_reason = (
                            f"Path traversal: {file_path} resolves to {resolved}, "
                            f"which is outside the workspace ({self._workspace}). "
                            f"Writing/editing files outside the workspace is not allowed."
                        )
                        logger.warning(f"BLOCKED path traversal: {file_path} → {resolved}")
                        return
            except (ValueError, OSError):
                pass

        # Check for suspicious patterns
        if ".." in file_path:
            result.warnings.append(
                f"Path contains '..': {file_path}. Verify this is intentional."
            )

        # Check for sensitive system files
        sensitive_paths = [
            "/etc/passwd", "/etc/shadow", "/etc/sudoers",
            ".ssh/", ".gnupg/", ".aws/credentials",
            ".env", ".netrc",
        ]
        for sensitive in sensitive_paths:
            if sensitive in file_path:
                if call.name in ("write", "edit"):
                    result.blocked = True
                    result.block_reason = (
                        f"Cannot write to sensitive path: {file_path}. "
                        f"This file may contain system credentials or security-sensitive data."
                    )
                    return
                else:
                    result.warnings.append(
                        f"Accessing sensitive path: {file_path}"
                    )

    def _check_secrets_in_content(self, call: ToolCall, result: CheckResult) -> None:
        """Check for secrets/credentials in write content."""
        content = call.input.get("content", "")
        if not content:
            return

        found_secrets = []
        for pattern, description in SECRET_PATTERNS:
            if pattern.search(content):
                found_secrets.append(description)

        if found_secrets:
            result.warnings.append(
                f"Possible secrets detected in file content: {', '.join(found_secrets)}. "
                f"Make sure you're not committing credentials."
            )
            logger.warning(f"Secrets detected in write content: {found_secrets}")

    # ── Public utility methods ──────────────────────────────

    def validate_tool_inputs(self, call: ToolCall, schemas: list[ToolSchema]) -> Optional[str]:
        """
        Validate that tool call inputs match the schema's required fields.
        Returns error message string if invalid, None if valid.
        """
        # Find the matching schema
        schema = None
        for s in schemas:
            if s.name == call.name:
                schema = s
                break

        if not schema:
            return f"Unknown tool: {call.name}"

        # Check required fields
        input_schema = schema.input_schema
        required = input_schema.get("required", [])
        properties = input_schema.get("properties", {})

        missing = [
            field for field in required
            if field not in call.input
        ]

        if missing:
            return (
                f"Missing required fields for tool '{call.name}': {', '.join(missing)}. "
                f"Expected: {required}"
            )

        # Check types for provided fields
        type_errors = []
        for field_name, field_value in call.input.items():
            if field_name in properties:
                expected_type = properties[field_name].get("type", "")
                if expected_type == "string" and not isinstance(field_value, str):
                    type_errors.append(
                        f"'{field_name}' should be string, got {type(field_value).__name__}"
                    )
                elif expected_type == "number" and not isinstance(field_value, (int, float)):
                    type_errors.append(
                        f"'{field_name}' should be number, got {type(field_value).__name__}"
                    )
                elif expected_type == "boolean" and not isinstance(field_value, bool):
                    type_errors.append(
                        f"'{field_name}' should be boolean, got {type(field_value).__name__}"
                    )

        if type_errors:
            return f"Type errors in tool '{call.name}': {'; '.join(type_errors)}"

        return None
