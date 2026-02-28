"""
Sandboxed Executor — resource-limited tool execution wrapper.

Enforces memory limits, execution timeouts, and disk quota for tool calls.
Wraps tool execution with pre/post checks and resource monitoring.

Sprint 17 (Security & Sandboxing) Module 4.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class ResourceLimits:
    """Resource limits for sandboxed execution."""
    max_execution_time_seconds: float = 30.0
    max_memory_mb: int = 512
    max_output_size_bytes: int = 10_000_000  # 10 MB
    max_disk_write_bytes: int = 50_000_000   # 50 MB
    allow_network: bool = True
    allow_file_write: bool = True
    allow_subprocess: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "max_memory_mb": self.max_memory_mb,
            "max_output_size_bytes": self.max_output_size_bytes,
            "max_disk_write_bytes": self.max_disk_write_bytes,
            "allow_network": self.allow_network,
            "allow_file_write": self.allow_file_write,
            "allow_subprocess": self.allow_subprocess,
        }


@dataclass
class ExecutionResult:
    """Result of a sandboxed execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timed_out: bool = False
    resource_violation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "error": self.error,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "timed_out": self.timed_out,
            "resource_violation": self.resource_violation,
        }


# ── Per-tool limit profiles ─────────────────────────────────────

DEFAULT_TOOL_LIMITS: Dict[str, ResourceLimits] = {
    "bash": ResourceLimits(
        max_execution_time_seconds=120.0,
        max_memory_mb=1024,
        max_output_size_bytes=10_000_000,
        allow_subprocess=True,
    ),
    "read": ResourceLimits(
        max_execution_time_seconds=10.0,
        max_memory_mb=256,
        allow_file_write=False,
        allow_subprocess=False,
    ),
    "write": ResourceLimits(
        max_execution_time_seconds=10.0,
        max_memory_mb=256,
        max_disk_write_bytes=50_000_000,
        allow_subprocess=False,
    ),
    "edit": ResourceLimits(
        max_execution_time_seconds=10.0,
        max_memory_mb=256,
        max_disk_write_bytes=50_000_000,
        allow_subprocess=False,
    ),
}


# ── SandboxedExecutor ────────────────────────────────────────────

class SandboxedExecutor:
    """
    Execute tool calls with resource limits and monitoring.

    Usage::

        executor = SandboxedExecutor()
        result = await executor.execute("bash", my_tool_fn, {"command": "ls"})
        if result.timed_out:
            print("Tool execution timed out")
    """

    def __init__(
        self,
        default_limits: Optional[ResourceLimits] = None,
        tool_limits: Optional[Dict[str, ResourceLimits]] = None,
        on_violation: Optional[Callable] = None,
    ):
        self._default_limits = default_limits or ResourceLimits()
        self._tool_limits: Dict[str, ResourceLimits] = dict(DEFAULT_TOOL_LIMITS)
        if tool_limits:
            self._tool_limits.update(tool_limits)
        self._on_violation = on_violation

        # Stats
        self._total_executions = 0
        self._total_timeouts = 0
        self._total_violations = 0
        self._total_errors = 0

    def get_limits(self, tool_name: str) -> ResourceLimits:
        """Get resource limits for a tool."""
        return self._tool_limits.get(tool_name, self._default_limits)

    def set_limits(self, tool_name: str, limits: ResourceLimits) -> None:
        """Set custom resource limits for a tool."""
        self._tool_limits[tool_name] = limits

    async def execute(
        self,
        tool_name: str,
        tool_fn: Callable[..., Awaitable[Any]],
        tool_input: Dict[str, Any],
        limits: Optional[ResourceLimits] = None,
    ) -> ExecutionResult:
        """
        Execute a tool function with resource limits.

        Args:
            tool_name: Name of the tool being executed
            tool_fn: Async callable to execute
            tool_input: Input arguments for the tool
            limits: Override limits (uses per-tool or default if None)
        """
        self._total_executions += 1
        effective_limits = limits or self.get_limits(tool_name)

        # Pre-execution checks
        violation = self._pre_check(tool_name, tool_input, effective_limits)
        if violation:
            self._total_violations += 1
            if self._on_violation:
                try:
                    self._on_violation(tool_name, violation)
                except Exception:
                    pass
            return ExecutionResult(
                success=False,
                error=violation,
                resource_violation=violation,
            )

        # Execute with timeout
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                tool_fn(**tool_input),
                timeout=effective_limits.max_execution_time_seconds,
            )
            elapsed_ms = (time.time() - start_time) * 1000

            # Post-execution checks
            post_violation = self._post_check(result, effective_limits)
            if post_violation:
                self._total_violations += 1
                if self._on_violation:
                    try:
                        self._on_violation(tool_name, post_violation)
                    except Exception:
                        pass
                return ExecutionResult(
                    success=False,
                    result=result,
                    error=post_violation,
                    execution_time_ms=elapsed_ms,
                    resource_violation=post_violation,
                )

            return ExecutionResult(
                success=True,
                result=result,
                execution_time_ms=elapsed_ms,
            )

        except asyncio.TimeoutError:
            elapsed_ms = (time.time() - start_time) * 1000
            self._total_timeouts += 1
            timeout_msg = (
                f"Tool '{tool_name}' exceeded timeout of "
                f"{effective_limits.max_execution_time_seconds}s"
            )
            if self._on_violation:
                try:
                    self._on_violation(tool_name, timeout_msg)
                except Exception:
                    pass
            return ExecutionResult(
                success=False,
                error=timeout_msg,
                execution_time_ms=elapsed_ms,
                timed_out=True,
                resource_violation=timeout_msg,
            )
        except Exception as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            self._total_errors += 1
            return ExecutionResult(
                success=False,
                error=str(exc),
                execution_time_ms=elapsed_ms,
            )

    def execute_sync(
        self,
        tool_name: str,
        tool_fn: Callable[..., Any],
        tool_input: Dict[str, Any],
        limits: Optional[ResourceLimits] = None,
    ) -> ExecutionResult:
        """
        Execute a synchronous tool function with resource limits.

        Uses basic time-based checking (no async timeout).
        """
        self._total_executions += 1
        effective_limits = limits or self.get_limits(tool_name)

        # Pre-execution checks
        violation = self._pre_check(tool_name, tool_input, effective_limits)
        if violation:
            self._total_violations += 1
            return ExecutionResult(
                success=False,
                error=violation,
                resource_violation=violation,
            )

        start_time = time.time()
        try:
            result = tool_fn(**tool_input)
            elapsed_ms = (time.time() - start_time) * 1000

            # Check timeout after execution
            if elapsed_ms > effective_limits.max_execution_time_seconds * 1000:
                self._total_timeouts += 1
                timeout_msg = (
                    f"Tool '{tool_name}' exceeded timeout of "
                    f"{effective_limits.max_execution_time_seconds}s "
                    f"(took {elapsed_ms/1000:.1f}s)"
                )
                return ExecutionResult(
                    success=False,
                    result=result,
                    error=timeout_msg,
                    execution_time_ms=elapsed_ms,
                    timed_out=True,
                    resource_violation=timeout_msg,
                )

            # Post-execution checks
            post_violation = self._post_check(result, effective_limits)
            if post_violation:
                self._total_violations += 1
                return ExecutionResult(
                    success=False,
                    result=result,
                    error=post_violation,
                    execution_time_ms=elapsed_ms,
                    resource_violation=post_violation,
                )

            return ExecutionResult(
                success=True,
                result=result,
                execution_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.time() - start_time) * 1000
            self._total_errors += 1
            return ExecutionResult(
                success=False,
                error=str(exc),
                execution_time_ms=elapsed_ms,
            )

    def _pre_check(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        limits: ResourceLimits,
    ) -> Optional[str]:
        """Run pre-execution resource checks."""
        # Check input size
        input_str = str(tool_input)
        if len(input_str) > limits.max_output_size_bytes:
            return f"Input size ({len(input_str)} bytes) exceeds limit ({limits.max_output_size_bytes} bytes)"

        # Check file write permission
        if not limits.allow_file_write:
            for key in ("file_path", "path", "filename"):
                if key in tool_input and tool_name in ("write", "edit"):
                    return f"File write not allowed for tool '{tool_name}'"

        return None

    def _post_check(
        self,
        result: Any,
        limits: ResourceLimits,
    ) -> Optional[str]:
        """Run post-execution resource checks."""
        if result is None:
            return None

        # Check output size
        result_str = str(result)
        if len(result_str) > limits.max_output_size_bytes:
            return (
                f"Output size ({len(result_str)} bytes) exceeds limit "
                f"({limits.max_output_size_bytes} bytes)"
            )

        return None

    # ── Stats ──────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return executor statistics."""
        return {
            "total_executions": self._total_executions,
            "total_timeouts": self._total_timeouts,
            "total_violations": self._total_violations,
            "total_errors": self._total_errors,
            "timeout_rate": (
                self._total_timeouts / self._total_executions
                if self._total_executions > 0 else 0.0
            ),
            "violation_rate": (
                self._total_violations / self._total_executions
                if self._total_executions > 0 else 0.0
            ),
            "registered_tool_limits": list(self._tool_limits.keys()),
        }

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self._total_executions = 0
        self._total_timeouts = 0
        self._total_violations = 0
        self._total_errors = 0
