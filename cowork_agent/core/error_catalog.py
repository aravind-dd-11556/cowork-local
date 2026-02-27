"""
Error Catalog — Structured error codes, classification, and recovery hints.

Every error in the agent framework maps to a unique code (E1xxx–E6xxx),
a human-readable message, a recovery hint, and a transient/permanent flag
that the retry layer can inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


# ── Error categories ────────────────────────────────────────────────

class ErrorCategory(Enum):
    PROVIDER = "provider"      # E1xxx
    TOOL = "tool"              # E2xxx
    AGENT = "agent"            # E3xxx
    CONFIG = "config"          # E4xxx
    NETWORK = "network"        # E5xxx
    SECURITY = "security"      # E6xxx


# ── Error codes ─────────────────────────────────────────────────────

class ErrorCode(Enum):
    # Provider errors (E1xxx)
    PROVIDER_CONNECTION_FAILED = "E1001"
    PROVIDER_AUTH_FAILED = "E1002"
    PROVIDER_RATE_LIMITED = "E1003"
    PROVIDER_TIMEOUT = "E1004"
    PROVIDER_MODEL_NOT_FOUND = "E1005"
    PROVIDER_OVERLOADED = "E1006"
    PROVIDER_INVALID_RESPONSE = "E1007"

    # Tool errors (E2xxx)
    TOOL_EXECUTION_FAILED = "E2001"
    TOOL_TIMEOUT = "E2002"
    TOOL_NOT_FOUND = "E2003"
    TOOL_VALIDATION_FAILED = "E2004"
    TOOL_PERMISSION_DENIED = "E2005"

    # Agent errors (E3xxx)
    AGENT_MAX_ITERATIONS = "E3001"
    AGENT_BUDGET_EXCEEDED = "E3002"
    AGENT_LOOP_DETECTED = "E3003"
    AGENT_CIRCUIT_BREAKER = "E3004"
    AGENT_EMPTY_RESPONSE = "E3005"
    AGENT_TRUNCATION = "E3006"

    # Config errors (E4xxx)
    CONFIG_MISSING_KEY = "E4001"
    CONFIG_INVALID_VALUE = "E4002"
    CONFIG_FILE_NOT_FOUND = "E4003"

    # Network errors (E5xxx)
    NETWORK_DNS_FAILED = "E5001"
    NETWORK_CONNECTION_REFUSED = "E5002"
    NETWORK_TIMEOUT = "E5003"
    NETWORK_SSL_ERROR = "E5004"

    # Security errors (E6xxx)
    SECURITY_BLOCKED = "E6001"
    SECURITY_INJECTION = "E6002"


# ── Mapping: ErrorCode → ErrorCategory ──────────────────────────────

_CODE_TO_CATEGORY: Dict[str, ErrorCategory] = {}
for _code in ErrorCode:
    prefix = _code.value[1]  # second char: '1'–'6'
    _CODE_TO_CATEGORY[_code.value] = {
        "1": ErrorCategory.PROVIDER,
        "2": ErrorCategory.TOOL,
        "3": ErrorCategory.AGENT,
        "4": ErrorCategory.CONFIG,
        "5": ErrorCategory.NETWORK,
        "6": ErrorCategory.SECURITY,
    }[prefix]


def _category_for(code: ErrorCode) -> ErrorCategory:
    return _CODE_TO_CATEGORY[code.value]


# ── AgentError dataclass ────────────────────────────────────────────

@dataclass
class AgentError:
    """Rich error wrapper carrying code, category, hint, and context."""
    code: ErrorCode
    message: str
    recovery_hint: str
    category: ErrorCategory
    is_transient: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    original_exception: Optional[Exception] = None

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    def full_message(self) -> str:
        parts = [f"[{self.code.value}] {self.message}"]
        if self.recovery_hint:
            parts.append(f"Hint: {self.recovery_hint}")
        if self.context:
            parts.append(f"Context: {self.context}")
        return "\n".join(parts)


# ── Static catalog of descriptions + recovery hints ─────────────────

_CATALOG: Dict[ErrorCode, dict] = {
    # Provider
    ErrorCode.PROVIDER_CONNECTION_FAILED: {
        "description": "Failed to connect to the LLM provider.",
        "hint": "Check that the provider service is running and the base_url is correct.",
        "transient": True,
    },
    ErrorCode.PROVIDER_AUTH_FAILED: {
        "description": "Authentication with the LLM provider failed.",
        "hint": "Verify your API key is set correctly (e.g. OPENAI_API_KEY or ANTHROPIC_API_KEY).",
        "transient": False,
    },
    ErrorCode.PROVIDER_RATE_LIMITED: {
        "description": "The LLM provider rate-limited the request.",
        "hint": "Wait a moment and retry, or reduce request frequency.",
        "transient": True,
    },
    ErrorCode.PROVIDER_TIMEOUT: {
        "description": "The LLM provider did not respond within the timeout.",
        "hint": "Increase the timeout in config or try a smaller prompt.",
        "transient": True,
    },
    ErrorCode.PROVIDER_MODEL_NOT_FOUND: {
        "description": "The requested model was not found on the provider.",
        "hint": "Check the model name in your config. Run 'ollama list' for local models.",
        "transient": False,
    },
    ErrorCode.PROVIDER_OVERLOADED: {
        "description": "The LLM provider is overloaded (503/529).",
        "hint": "Wait and retry. Consider using a fallback provider.",
        "transient": True,
    },
    ErrorCode.PROVIDER_INVALID_RESPONSE: {
        "description": "The LLM provider returned an unexpected or malformed response.",
        "hint": "This may be a transient issue. Retry or check provider status.",
        "transient": True,
    },

    # Tool
    ErrorCode.TOOL_EXECUTION_FAILED: {
        "description": "A tool failed during execution.",
        "hint": "Check the tool's input parameters and try again.",
        "transient": False,
    },
    ErrorCode.TOOL_TIMEOUT: {
        "description": "A tool execution timed out.",
        "hint": "Increase the tool timeout or simplify the operation.",
        "transient": True,
    },
    ErrorCode.TOOL_NOT_FOUND: {
        "description": "The requested tool is not registered.",
        "hint": "Check available tools with the tool registry.",
        "transient": False,
    },
    ErrorCode.TOOL_VALIDATION_FAILED: {
        "description": "Tool input validation failed.",
        "hint": "Check that the tool parameters match the expected schema.",
        "transient": False,
    },
    ErrorCode.TOOL_PERMISSION_DENIED: {
        "description": "Permission denied when executing the tool.",
        "hint": "Check file permissions and workspace directory access.",
        "transient": False,
    },

    # Agent
    ErrorCode.AGENT_MAX_ITERATIONS: {
        "description": "The agent reached the maximum iteration limit.",
        "hint": "Increase max_iterations in config or break the task into smaller steps.",
        "transient": False,
    },
    ErrorCode.AGENT_BUDGET_EXCEEDED: {
        "description": "The agent exceeded its token budget.",
        "hint": "Increase the budget limit or use a more concise prompt.",
        "transient": False,
    },
    ErrorCode.AGENT_LOOP_DETECTED: {
        "description": "The agent detected a repetitive action loop.",
        "hint": "The agent is repeating itself. Try rephrasing your request.",
        "transient": False,
    },
    ErrorCode.AGENT_CIRCUIT_BREAKER: {
        "description": "The circuit breaker tripped after repeated failures.",
        "hint": "The agent encountered too many consecutive errors. Wait before retrying.",
        "transient": True,
    },
    ErrorCode.AGENT_EMPTY_RESPONSE: {
        "description": "The agent received an empty response from the provider.",
        "hint": "Retry the request. The provider may have returned a blank response.",
        "transient": True,
    },
    ErrorCode.AGENT_TRUNCATION: {
        "description": "The conversation was truncated to fit context limits.",
        "hint": "This is informational. The agent trimmed older messages to stay within limits.",
        "transient": False,
    },

    # Config
    ErrorCode.CONFIG_MISSING_KEY: {
        "description": "A required configuration key is missing.",
        "hint": "Check your config YAML file or set the corresponding environment variable.",
        "transient": False,
    },
    ErrorCode.CONFIG_INVALID_VALUE: {
        "description": "A configuration value is invalid.",
        "hint": "Check the config reference in README for valid values.",
        "transient": False,
    },
    ErrorCode.CONFIG_FILE_NOT_FOUND: {
        "description": "The configuration file was not found.",
        "hint": "Verify the config file path or use default configuration.",
        "transient": False,
    },

    # Network
    ErrorCode.NETWORK_DNS_FAILED: {
        "description": "DNS resolution failed.",
        "hint": "Check your network connection and DNS settings.",
        "transient": True,
    },
    ErrorCode.NETWORK_CONNECTION_REFUSED: {
        "description": "The connection was refused by the remote host.",
        "hint": "Verify the service is running on the expected host and port.",
        "transient": True,
    },
    ErrorCode.NETWORK_TIMEOUT: {
        "description": "The network request timed out.",
        "hint": "Check your network connection or increase the timeout.",
        "transient": True,
    },
    ErrorCode.NETWORK_SSL_ERROR: {
        "description": "An SSL/TLS error occurred.",
        "hint": "Check SSL certificates or try disabling SSL verification for local services.",
        "transient": False,
    },

    # Security
    ErrorCode.SECURITY_BLOCKED: {
        "description": "The request was blocked by security rules.",
        "hint": "The operation was blocked for safety. Review the security policy.",
        "transient": False,
    },
    ErrorCode.SECURITY_INJECTION: {
        "description": "Potential prompt injection detected.",
        "hint": "The input was flagged as a possible injection attempt.",
        "transient": False,
    },
}


# ── ErrorCatalog — main API ─────────────────────────────────────────

class ErrorCatalog:
    """Static catalog for classifying exceptions into structured AgentErrors."""

    @staticmethod
    def classify_error(exception: Exception) -> AgentError:
        """
        Pattern-match an exception to the most appropriate ErrorCode.

        Inspects exception type and message string to pick the right code.
        Falls back to PROVIDER_INVALID_RESPONSE for unknown errors.
        """
        msg = str(exception).lower()
        exc_type = type(exception)

        # ── Network-level errors ───────────────────────────────
        if isinstance(exception, ConnectionRefusedError):
            code = ErrorCode.NETWORK_CONNECTION_REFUSED
        elif isinstance(exception, TimeoutError):
            code = ErrorCode.NETWORK_TIMEOUT
        elif isinstance(exception, OSError) and "name resolution" in msg:
            code = ErrorCode.NETWORK_DNS_FAILED
        elif isinstance(exception, OSError) and "ssl" in msg:
            code = ErrorCode.NETWORK_SSL_ERROR
        elif isinstance(exception, ConnectionError):
            code = ErrorCode.PROVIDER_CONNECTION_FAILED

        # ── Provider-level heuristics (message-based) ──────────
        elif "rate limit" in msg or "429" in msg or "too many requests" in msg:
            code = ErrorCode.PROVIDER_RATE_LIMITED
        elif "authentication" in msg or "401" in msg or "unauthorized" in msg or "api key" in msg:
            code = ErrorCode.PROVIDER_AUTH_FAILED
        elif "model not found" in msg or "model_not_found" in msg or "does not exist" in msg:
            code = ErrorCode.PROVIDER_MODEL_NOT_FOUND
        elif "overloaded" in msg or "503" in msg or "529" in msg or "service unavailable" in msg:
            code = ErrorCode.PROVIDER_OVERLOADED
        elif "timeout" in msg or "timed out" in msg:
            code = ErrorCode.PROVIDER_TIMEOUT

        # ── Tool-level heuristics ──────────────────────────────
        elif "permission denied" in msg or "permission error" in msg:
            code = ErrorCode.TOOL_PERMISSION_DENIED
        elif "not found" in msg and "tool" in msg:
            code = ErrorCode.TOOL_NOT_FOUND
        elif "validation" in msg:
            code = ErrorCode.TOOL_VALIDATION_FAILED

        # ── Agent-level heuristics ─────────────────────────────
        elif "max iteration" in msg or "maximum iteration" in msg:
            code = ErrorCode.AGENT_MAX_ITERATIONS
        elif "budget" in msg or "token limit" in msg:
            code = ErrorCode.AGENT_BUDGET_EXCEEDED
        elif "loop detect" in msg or "repetitive" in msg:
            code = ErrorCode.AGENT_LOOP_DETECTED
        elif "circuit breaker" in msg or "circuit_breaker" in msg:
            code = ErrorCode.AGENT_CIRCUIT_BREAKER

        # ── Security ───────────────────────────────────────────
        elif "injection" in msg:
            code = ErrorCode.SECURITY_INJECTION
        elif "blocked" in msg and "security" in msg:
            code = ErrorCode.SECURITY_BLOCKED

        # ── Fallback ───────────────────────────────────────────
        else:
            code = ErrorCode.PROVIDER_INVALID_RESPONSE

        entry = _CATALOG[code]
        return AgentError(
            code=code,
            message=entry["description"],
            recovery_hint=entry["hint"],
            category=_category_for(code),
            is_transient=entry["transient"],
            original_exception=exception,
        )

    @staticmethod
    def get_recovery_hint(code: ErrorCode) -> str:
        """Return the recovery hint for a given error code."""
        entry = _CATALOG.get(code)
        return entry["hint"] if entry else "No recovery hint available."

    @staticmethod
    def is_transient(exception: Exception) -> bool:
        """Return True if the exception is likely transient (retryable)."""
        agent_error = ErrorCatalog.classify_error(exception)
        return agent_error.is_transient

    @staticmethod
    def wrap(exception: Exception, context: Dict[str, Any] = None) -> AgentError:
        """Classify an exception and attach additional context."""
        agent_error = ErrorCatalog.classify_error(exception)
        if context:
            agent_error.context = context
        return agent_error

    @staticmethod
    def from_code(code: ErrorCode, context: Dict[str, Any] = None,
                  exception: Optional[Exception] = None) -> AgentError:
        """Create an AgentError directly from an ErrorCode."""
        entry = _CATALOG[code]
        return AgentError(
            code=code,
            message=entry["description"],
            recovery_hint=entry["hint"],
            category=_category_for(code),
            is_transient=entry["transient"],
            context=context or {},
            original_exception=exception,
        )
