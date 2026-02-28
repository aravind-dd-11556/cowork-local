"""
Input Sanitizer — command injection, SQL injection, and template injection prevention.

Pre-screens tool input dictionaries for injection attacks before execution.
Complements SafetyChecker (which blocks destructive *commands*) by catching
injection vectors in all input fields.

Sprint 17 (Security & Sandboxing) Module 1.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Detection patterns ───────────────────────────────────────────

# SQL injection patterns
SQL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\b(UNION\s+SELECT)\b"), "sql_union_select"),
    (re.compile(r"(?i)\b(DROP\s+TABLE)\b"), "sql_drop_table"),
    (re.compile(r"(?i)\b(DELETE\s+FROM)\b"), "sql_delete"),
    (re.compile(r"(?i)\b(INSERT\s+INTO)\b"), "sql_insert"),
    (re.compile(r"(?i)\b(UPDATE\s+\w+\s+SET)\b"), "sql_update"),
    (re.compile(r"(?i)(\b1\s*=\s*1\b)"), "sql_tautology"),
    (re.compile(r"(?i)(--\s*$|;\s*--\s)"), "sql_comment_terminator"),
    (re.compile(r"(?i)\b(OR\s+1\s*=\s*1)\b"), "sql_or_tautology"),
    (re.compile(r"(?i)\b(EXEC\s*\(|EXECUTE\s)"), "sql_exec"),
]

# Command injection patterns (in non-bash inputs like file paths, URLs)
COMMAND_INJECTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\$\("), "shell_command_substitution"),
    (re.compile(r"`[^`]+`"), "backtick_substitution"),
    (re.compile(r";\s*(rm|cat|curl|wget|nc|python|perl|ruby|bash|sh)\b"), "semicolon_injection"),
    (re.compile(r"\|\s*(rm|cat|curl|wget|nc|python|perl|ruby|bash|sh)\b"), "pipe_injection"),
    (re.compile(r"&&\s*(rm|cat|curl|wget|nc|python|perl|ruby|bash|sh)\b"), "and_injection"),
    (re.compile(r"\|\|\s*(rm|cat|curl|wget|nc|python|perl|ruby|bash|sh)\b"), "or_injection"),
]

# Template injection patterns
TEMPLATE_INJECTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\{\{.*\}\}"), "jinja_template"),
    (re.compile(r"\{%.*%\}"), "jinja_block"),
    (re.compile(r"\$\{[^}]+\}"), "shell_variable_expansion"),
    (re.compile(r"<%.*%>"), "erb_template"),
    (re.compile(r"#\{.*\}"), "ruby_interpolation"),
]

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\.\.(/|\\)"), "path_traversal"),
    (re.compile(r"~root\b"), "home_root_access"),
    (re.compile(r"/etc/(passwd|shadow|hosts)"), "sensitive_system_file"),
    (re.compile(r"/proc/self/"), "proc_self_access"),
]

# XPath injection patterns
XPATH_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)(\bor\b\s*'[^']*'\s*=\s*'[^']*')"), "xpath_or_injection"),
    (re.compile(r"(?i)(\]\s*\[\s*1\s*=\s*1)"), "xpath_tautology"),
]


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class SanitizationResult:
    """Result of input sanitization check."""
    is_safe: bool
    threats: List[str] = field(default_factory=list)
    threat_details: List[Dict[str, str]] = field(default_factory=list)
    sanitized_input: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical_injection(self) -> bool:
        return not self.is_safe and len(self.threats) > 0

    @property
    def threat_summary(self) -> str:
        return "; ".join(self.threats[:5]) if self.threats else "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "threats": self.threats,
            "threat_count": len(self.threats),
        }


# ── InputSanitizer ──────────────────────────────────────────────

class InputSanitizer:
    """
    Detect injection attacks in tool input dictionaries.

    Usage::

        san = InputSanitizer()
        result = san.sanitize("bash", {"command": "ls; rm -rf /"})
        if not result.is_safe:
            print(result.threat_summary)
    """

    # Fields where command injection is relevant (not bash commands themselves)
    COMMAND_SENSITIVE_FIELDS = {"file_path", "path", "url", "filename", "directory", "target"}

    def __init__(
        self,
        max_input_size: int = 1_000_000,
        sql_injection: bool = True,
        command_injection: bool = True,
        template_injection: bool = True,
        xpath_injection: bool = True,
    ):
        self._max_input_size = max_input_size
        self._sql_enabled = sql_injection
        self._command_enabled = command_injection
        self._template_enabled = template_injection
        self._xpath_enabled = xpath_injection

    def sanitize(self, tool_name: str, input_dict: Dict[str, Any]) -> SanitizationResult:
        """
        Check all input fields for injection attacks.

        For bash tool: only checks non-command fields (SafetyChecker handles commands).
        For other tools: checks all string fields.
        """
        threats: List[str] = []
        details: List[Dict[str, str]] = []

        for key, value in input_dict.items():
            if not isinstance(value, str):
                continue

            # Size check
            if len(value) > self._max_input_size:
                threats.append(f"Input field '{key}' exceeds max size ({len(value)} > {self._max_input_size})")
                details.append({"field": key, "type": "size_exceeded"})
                continue

            # For bash tool, skip the 'command' field (handled by SafetyChecker)
            if tool_name == "bash" and key == "command":
                continue

            field_threats = self._check_field(key, value, tool_name)
            threats.extend(field_threats)
            for t in field_threats:
                details.append({"field": key, "type": t})

        return SanitizationResult(
            is_safe=len(threats) == 0,
            threats=threats,
            threat_details=details,
            sanitized_input=input_dict,
        )

    def _check_field(self, field_name: str, value: str, tool_name: str) -> List[str]:
        """Check a single field for injection patterns."""
        threats: List[str] = []

        # SQL injection (check all string fields)
        if self._sql_enabled:
            for pattern, label in SQL_PATTERNS:
                if pattern.search(value):
                    threats.append(f"SQL injection ({label}) in '{field_name}'")

        # Command injection (only in sensitive fields, not bash commands)
        if self._command_enabled:
            is_sensitive = (
                field_name.lower() in self.COMMAND_SENSITIVE_FIELDS
                or tool_name != "bash"
            )
            if is_sensitive:
                for pattern, label in COMMAND_INJECTION_PATTERNS:
                    if pattern.search(value):
                        threats.append(f"Command injection ({label}) in '{field_name}'")

        # Template injection
        if self._template_enabled:
            for pattern, label in TEMPLATE_INJECTION_PATTERNS:
                if pattern.search(value):
                    threats.append(f"Template injection ({label}) in '{field_name}'")

        # Path traversal
        for pattern, label in PATH_TRAVERSAL_PATTERNS:
            if pattern.search(value):
                threats.append(f"Path traversal ({label}) in '{field_name}'")

        # XPath injection
        if self._xpath_enabled:
            for pattern, label in XPATH_PATTERNS:
                if pattern.search(value):
                    threats.append(f"XPath injection ({label}) in '{field_name}'")

        return threats

    @staticmethod
    def escape_shell_arg(arg: str) -> str:
        """Escape a string for safe use as a shell argument."""
        return "'" + arg.replace("'", "'\\''") + "'"

    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalize a file path to remove traversal sequences."""
        import posixpath
        normalized = posixpath.normpath(path)
        # Remove leading .. sequences
        parts = normalized.split("/")
        safe_parts = [p for p in parts if p != ".."]
        return "/".join(safe_parts) or "."
