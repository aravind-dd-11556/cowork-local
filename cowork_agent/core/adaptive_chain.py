"""
Adaptive Tool Chaining — intelligent tool sequence planning and execution.

Plans optimal tool sequences from templates, adapts mid-chain on failure
(retry → fallback → adapt), and learns from past patterns via
ReflectionEngine.

Sprint 28: Adaptive Tool Chaining.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tool_registry import ToolRegistry
    from .reflection_engine import ReflectionEngine
    from .rollback_journal import RollbackJournal
    from .execution_tracer import ExecutionTracer

from .models import ToolCall, ToolResult

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ADAPTED = "adapted"


# ── Dataclasses ───────────────────────────────────────────────────

@dataclass
class ChainStep:
    """A single step in a tool chain."""
    tool_name: str
    description: str = ""
    input_args: Dict[str, Any] = field(default_factory=dict)
    expected_output_type: str = "any"   # "text"|"json"|"file_path"|"code"|"any"
    fallback_tool: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    status: str = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "input_args": self.input_args,
            "expected_output_type": self.expected_output_type,
            "fallback_tool": self.fallback_tool,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChainStep:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def is_retryable(self) -> bool:
        return self.retry_count < self.max_retries

    @property
    def has_fallback(self) -> bool:
        return self.fallback_tool is not None


@dataclass
class ToolChainPlan:
    """An ordered plan of tool chain steps."""
    plan_id: str
    goal: str
    steps: List[ChainStep] = field(default_factory=list)
    current_step_index: int = 0
    status: str = PlanStatus.PENDING
    adaptations: List[str] = field(default_factory=list)
    template_used: str = ""
    reflection_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
            "status": self.status,
            "adaptations": self.adaptations,
            "template_used": self.template_used,
            "reflection_hints": self.reflection_hints,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ToolChainPlan:
        steps_data = data.pop("steps", [])
        plan = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        plan.steps = [ChainStep.from_dict(s) for s in steps_data]
        return plan

    @property
    def next_step(self) -> Optional[ChainStep]:
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)


@dataclass
class ChainExecutionResult:
    """Outcome of executing a tool chain plan."""
    plan: ToolChainPlan
    success: bool
    steps_completed: int = 0
    steps_total: int = 0
    adaptations_made: int = 0
    rollback_used: bool = False
    execution_time_ms: float = 0.0
    error: str = ""
    step_results: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan.plan_id,
            "success": self.success,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "adaptations_made": self.adaptations_made,
            "rollback_used": self.rollback_used,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "error": self.error,
            "step_results": self.step_results,
        }

    @property
    def success_rate(self) -> float:
        if self.steps_total == 0:
            return 0.0
        return self.steps_completed / self.steps_total


# ── Chain Templates ───────────────────────────────────────────────

CHAIN_TEMPLATES: Dict[str, List[dict]] = {
    "file_creation": [
        {"tool_name": "read", "description": "Read source/reference file",
         "expected_output_type": "text"},
        {"tool_name": "write", "description": "Write the new file",
         "expected_output_type": "file_path"},
        {"tool_name": "bash", "description": "Verify file was created",
         "expected_output_type": "text"},
    ],
    "code_fix": [
        {"tool_name": "read", "description": "Read the file to fix",
         "expected_output_type": "code"},
        {"tool_name": "edit", "description": "Apply the fix",
         "fallback_tool": "write", "expected_output_type": "text"},
        {"tool_name": "bash", "description": "Run tests to verify fix",
         "expected_output_type": "text"},
    ],
    "research": [
        {"tool_name": "web_search", "description": "Search for information",
         "expected_output_type": "text"},
        {"tool_name": "web_fetch", "description": "Fetch detailed content",
         "expected_output_type": "text"},
    ],
    "refactor": [
        {"tool_name": "glob", "description": "Find files to refactor",
         "expected_output_type": "text"},
        {"tool_name": "read", "description": "Read file contents",
         "expected_output_type": "code"},
        {"tool_name": "edit", "description": "Apply refactoring changes",
         "fallback_tool": "write", "expected_output_type": "text"},
        {"tool_name": "bash", "description": "Run tests after refactor",
         "expected_output_type": "text"},
    ],
    "data_pipeline": [
        {"tool_name": "read", "description": "Read input data",
         "expected_output_type": "text"},
        {"tool_name": "bash", "description": "Process/transform data",
         "expected_output_type": "text"},
        {"tool_name": "write", "description": "Write output data",
         "expected_output_type": "file_path"},
    ],
}


TEMPLATE_KEYWORDS: Dict[str, List[str]] = {
    "file_creation": ["create file", "new file", "write file", "generate file", "save file"],
    "code_fix": ["fix", "bug", "patch", "repair", "debug", "correct"],
    "research": ["search", "research", "find info", "look up", "investigate"],
    "refactor": ["refactor", "restructure", "reorganize", "rename across", "clean up code"],
    "data_pipeline": ["transform data", "process data", "convert data", "pipeline",
                      "data processing", "etl"],
}


# ── Error-to-adaptation mapping ──────────────────────────────────

ADAPTATION_MAP: Dict[str, dict] = {
    "permission": {
        "description": "Add permission fix step",
        "insert_step": {"tool_name": "bash", "description": "Fix file permissions",
                        "input_args": {"command": "chmod +w"}, "expected_output_type": "text"},
    },
    "no such file": {
        "description": "Create parent directories",
        "insert_step": {"tool_name": "bash", "description": "Create parent directories",
                        "input_args": {"command": "mkdir -p"}, "expected_output_type": "text"},
    },
    "not found": {
        "description": "Create parent directories",
        "insert_step": {"tool_name": "bash", "description": "Create missing directory",
                        "input_args": {"command": "mkdir -p"}, "expected_output_type": "text"},
    },
    "module": {
        "description": "Install missing dependency",
        "insert_step": {"tool_name": "bash", "description": "Install missing dependency",
                        "input_args": {"command": "pip install"}, "expected_output_type": "text"},
    },
    "syntax": {
        "description": "Re-read file before editing",
        "insert_step": {"tool_name": "read", "description": "Re-read file for correct context",
                        "expected_output_type": "code"},
    },
}


# ── AdaptiveChainExecutor ────────────────────────────────────────

# Tools that trigger rollback checkpoints
RISKY_CHAIN_TOOLS = {"bash", "write", "edit", "delete_file"}


class AdaptiveChainExecutor:
    """
    Plans and executes adaptive tool chains.

    Given a goal, selects a template, builds a plan, and executes it
    step-by-step. On failure, retries, falls back, or adapts the chain
    by inserting prerequisite steps.

    Usage::

        executor = AdaptiveChainExecutor(
            tool_registry=registry,
            reflection_engine=reflection_engine,
            rollback_journal=rollback_journal,
        )

        plan = executor.plan_chain("fix the test failure", available_tools=["read", "edit", "bash"])
        result = await executor.execute_chain(plan, messages=[])

    """

    # Maximum adaptations per chain to prevent infinite loops
    MAX_ADAPTATIONS = 5

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        reflection_engine: Optional[ReflectionEngine] = None,
        rollback_journal: Optional[RollbackJournal] = None,
        execution_tracer: Optional[ExecutionTracer] = None,
    ):
        self._registry = tool_registry
        self._reflection_engine = reflection_engine
        self._rollback_journal = rollback_journal
        self._tracer = execution_tracer

    # ── Plan ──────────────────────────────────────────────────

    def plan_chain(
        self,
        goal: str,
        available_tools: Optional[List[str]] = None,
        context_hints: Optional[Dict[str, Any]] = None,
    ) -> ToolChainPlan:
        """
        Build a ToolChainPlan for the given goal.

        Matches goal keywords to a template, queries reflection engine
        for past lessons, and filters steps to available tools.

        Args:
            goal: What the chain should accomplish.
            available_tools: List of tool names available (filters steps).
            context_hints: Optional context like file paths, error messages.

        Returns:
            A ToolChainPlan ready for execution.
        """
        plan_id = f"chain_{uuid.uuid4().hex[:12]}"

        # 1. Match template
        template_name = self._match_template(goal)
        template_steps = CHAIN_TEMPLATES.get(template_name, [])

        # 2. Get reflection hints
        reflection_hints = []
        if self._reflection_engine:
            try:
                relevant = self._reflection_engine.get_relevant_reflections(goal, limit=3)
                for entry in relevant:
                    if hasattr(entry, "value"):
                        reflection_hints.append(entry.value)
            except Exception as e:
                logger.debug(f"Reflection lookup failed: {e}")

        # 3. Build steps from template
        steps = []
        for step_data in template_steps:
            step = ChainStep(
                tool_name=step_data["tool_name"],
                description=step_data.get("description", ""),
                expected_output_type=step_data.get("expected_output_type", "any"),
                fallback_tool=step_data.get("fallback_tool"),
            )
            # Apply context hints to step input_args
            if context_hints:
                step.input_args = dict(context_hints)
            steps.append(step)

        # 4. Filter to available tools
        if available_tools is not None:
            available_set = set(available_tools)
            filtered_steps = []
            for step in steps:
                if step.tool_name in available_set:
                    # Also check fallback availability
                    if step.fallback_tool and step.fallback_tool not in available_set:
                        step.fallback_tool = None
                    filtered_steps.append(step)
                elif step.fallback_tool and step.fallback_tool in available_set:
                    # Primary not available, swap to fallback
                    step.tool_name = step.fallback_tool
                    step.fallback_tool = None
                    filtered_steps.append(step)
                # else: skip step entirely
            steps = filtered_steps

        plan = ToolChainPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            template_used=template_name,
            reflection_hints=reflection_hints,
        )

        logger.info(
            f"Planned chain {plan_id}: template={template_name}, "
            f"steps={len(steps)}, hints={len(reflection_hints)}"
        )
        return plan

    # ── Execute ───────────────────────────────────────────────

    async def execute_chain(
        self,
        plan: ToolChainPlan,
        messages: Optional[list] = None,
    ) -> ChainExecutionResult:
        """
        Execute a tool chain plan step-by-step.

        On failure: retry → fallback → adapt. Creates rollback checkpoints
        before risky tools.

        Args:
            plan: The ToolChainPlan to execute.
            messages: Current conversation messages (for rollback).

        Returns:
            ChainExecutionResult with outcome details.
        """
        t0 = time.time()
        plan.status = PlanStatus.RUNNING
        adaptations_made = 0
        rollback_used = False
        step_results = []

        # Start trace span
        span_id = None
        if self._tracer:
            span_id = self._tracer.start_span(
                operation=f"chain.{plan.template_used or 'custom'}",
                attributes={"goal": plan.goal, "steps": len(plan.steps)},
            )

        try:
            while plan.current_step_index < len(plan.steps):
                step = plan.steps[plan.current_step_index]
                step.status = StepStatus.RUNNING

                # Create rollback checkpoint before risky tools
                checkpoint_id = None
                if self._rollback_journal and step.tool_name in RISKY_CHAIN_TOOLS:
                    try:
                        checkpoint_id = self._rollback_journal.auto_checkpoint_before_chain(
                            tool_names=[step.tool_name],
                            messages=messages or [],
                        )
                    except Exception as e:
                        logger.debug(f"Rollback checkpoint skipped: {e}")

                # Execute step
                result = await self._execute_step(step)

                if result.success:
                    step.status = StepStatus.COMPLETED
                    step.result = result.output
                    step_results.append(result.output)

                    # Validate output type
                    if not self._validate_output_type(result.output, step.expected_output_type):
                        logger.warning(
                            f"Step {step.tool_name} output type mismatch "
                            f"(expected {step.expected_output_type})"
                        )

                    plan.current_step_index += 1

                else:
                    step.error = result.error or "Unknown error"

                    # Retry
                    if step.is_retryable:
                        step.retry_count += 1
                        logger.info(
                            f"Retrying step {step.tool_name} "
                            f"({step.retry_count}/{step.max_retries})"
                        )
                        continue

                    # Fallback
                    if step.has_fallback:
                        logger.info(
                            f"Falling back from {step.tool_name} "
                            f"to {step.fallback_tool}"
                        )
                        step.tool_name = step.fallback_tool
                        step.fallback_tool = None
                        step.retry_count = 0
                        continue

                    # Adapt
                    if adaptations_made < self.MAX_ADAPTATIONS:
                        adapted = self.adapt_chain(
                            plan, plan.current_step_index, step.error
                        )
                        if adapted:
                            adaptations_made += 1
                            plan.status = PlanStatus.ADAPTED
                            plan.adaptations.append(
                                f"Adapted at step {plan.current_step_index}: {step.error}"
                            )
                            continue

                    # Rollback if available
                    if checkpoint_id and self._rollback_journal:
                        try:
                            rb_result = self._rollback_journal.rollback(checkpoint_id)
                            rollback_used = rb_result.success
                        except Exception:
                            pass

                    # Step failed permanently
                    step.status = StepStatus.FAILED
                    plan.status = PlanStatus.FAILED

                    elapsed_ms = (time.time() - t0) * 1000
                    if span_id and self._tracer:
                        self._tracer.end_span(span_id, status="error",
                                               error_message=step.error)

                    return ChainExecutionResult(
                        plan=plan,
                        success=False,
                        steps_completed=plan.completed_steps,
                        steps_total=len(plan.steps),
                        adaptations_made=adaptations_made,
                        rollback_used=rollback_used,
                        execution_time_ms=elapsed_ms,
                        error=f"Step '{step.tool_name}' failed: {step.error}",
                        step_results=step_results,
                    )

            # All steps completed
            plan.status = PlanStatus.COMPLETED
            elapsed_ms = (time.time() - t0) * 1000

            if span_id and self._tracer:
                self._tracer.end_span(span_id, status="ok")

            return ChainExecutionResult(
                plan=plan,
                success=True,
                steps_completed=plan.completed_steps,
                steps_total=len(plan.steps),
                adaptations_made=adaptations_made,
                rollback_used=rollback_used,
                execution_time_ms=elapsed_ms,
                step_results=step_results,
            )

        except Exception as e:
            plan.status = PlanStatus.FAILED
            elapsed_ms = (time.time() - t0) * 1000

            if span_id and self._tracer:
                self._tracer.end_span(span_id, status="error", error_message=str(e))

            return ChainExecutionResult(
                plan=plan,
                success=False,
                steps_completed=plan.completed_steps,
                steps_total=len(plan.steps),
                adaptations_made=adaptations_made,
                rollback_used=rollback_used,
                execution_time_ms=elapsed_ms,
                error=f"Chain execution error: {e}",
                step_results=step_results,
            )

    # ── Adapt ─────────────────────────────────────────────────

    def adapt_chain(
        self,
        plan: ToolChainPlan,
        failed_step_index: int,
        error: str,
    ) -> bool:
        """
        Adapt the chain by inserting a prerequisite step before the failed step.

        Classifies the error and inserts an appropriate fix step (e.g., mkdir,
        chmod, pip install) before the failed step.

        Args:
            plan: The current plan.
            failed_step_index: Index of the step that failed.
            error: The error message.

        Returns:
            True if the chain was adapted, False if no adaptation found.
        """
        error_lower = error.lower()

        for keyword, adaptation in ADAPTATION_MAP.items():
            if keyword in error_lower:
                # Check that the inserted tool is available
                insert_tool = adaptation["insert_step"]["tool_name"]
                if self._registry:
                    try:
                        self._registry.get_tool(insert_tool)
                    except KeyError:
                        continue

                # Create the prerequisite step
                new_step = ChainStep(
                    tool_name=adaptation["insert_step"]["tool_name"],
                    description=adaptation["insert_step"].get("description", ""),
                    input_args=dict(adaptation["insert_step"].get("input_args", {})),
                    expected_output_type=adaptation["insert_step"].get(
                        "expected_output_type", "text"
                    ),
                    max_retries=1,
                )

                # Insert before the failed step
                plan.steps.insert(failed_step_index, new_step)

                # Reset the failed step for retry
                failed_step = plan.steps[failed_step_index + 1]
                failed_step.retry_count = 0
                failed_step.status = StepStatus.PENDING
                failed_step.error = None

                logger.info(
                    f"Adapted chain: inserted '{insert_tool}' step "
                    f"before step {failed_step_index} ({adaptation['description']})"
                )
                return True

        return False

    # ── Internal helpers ──────────────────────────────────────

    async def _execute_step(self, step: ChainStep) -> ToolResult:
        """Execute a single chain step via the tool registry."""
        if not self._registry:
            return ToolResult(
                tool_id="",
                success=False,
                output="",
                error="No tool registry available",
            )

        tool_call = ToolCall(
            name=step.tool_name,
            tool_id=ToolCall.generate_id(),
            input=dict(step.input_args),
        )

        try:
            return await self._registry.execute_tool(tool_call)
        except Exception as e:
            return ToolResult(
                tool_id=tool_call.tool_id,
                success=False,
                output="",
                error=str(e),
            )

    @staticmethod
    def _match_template(goal: str) -> str:
        """Match a goal string to a chain template name."""
        goal_lower = goal.lower()
        best_match = ""
        best_score = 0

        for template_name, keywords in TEMPLATE_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in goal_lower:
                    # Longer keyword matches are more specific
                    score += len(keyword)
            if score > best_score:
                best_score = score
                best_match = template_name

        return best_match

    @staticmethod
    def _validate_output_type(output: str, expected_type: str) -> bool:
        """
        Validate that tool output matches the expected type.

        Args:
            output: The tool's string output.
            expected_type: One of "text", "json", "file_path", "code", "any".

        Returns:
            True if output matches expected type.
        """
        if expected_type == "any" or not output:
            return True

        if expected_type == "json":
            import json
            try:
                json.loads(output)
                return True
            except (json.JSONDecodeError, TypeError):
                return False

        if expected_type == "file_path":
            # Check if output looks like a file path
            return (
                output.startswith("/")
                or output.startswith("./")
                or output.startswith("~")
                or "." in output.split("/")[-1] if "/" in output else False
            )

        if expected_type == "code":
            # Heuristic: contains common code patterns
            code_indicators = [
                "def ", "class ", "import ", "function ", "const ",
                "var ", "let ", "if ", "for ", "while ", "return ",
                "{", "}", "()", "=>",
            ]
            return any(indicator in output for indicator in code_indicators)

        # "text" — always valid for non-empty strings
        return True
