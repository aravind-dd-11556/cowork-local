"""
Reflection Engine — Agent learning from past execution patterns.

Analyzes ExecutionTracer spans to extract tool usage patterns, success/failure
rates, and error patterns. Generates actionable lessons that are stored in
KnowledgeStore for cross-session learning.

Sprint 27: Tier 2 Differentiating Feature 2.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .execution_tracer import ExecutionTracer, Span
    from .knowledge_store import KnowledgeStore, KnowledgeEntry

logger = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────

@dataclass
class Reflection:
    """A single reflection derived from an agent execution."""
    reflection_id: str
    task_description: str
    tools_used: List[str] = field(default_factory=list)
    successful_tools: List[str] = field(default_factory=list)
    failed_tools: List[str] = field(default_factory=list)
    error_patterns: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "reflection_id": self.reflection_id,
            "task_description": self.task_description,
            "tools_used": self.tools_used,
            "successful_tools": self.successful_tools,
            "failed_tools": self.failed_tools,
            "error_patterns": self.error_patterns,
            "lessons": self.lessons,
            "success_rate": self.success_rate,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Reflection:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ReflectionResult:
    """Result of a reflect() call."""
    reflection: Reflection
    lessons_stored: int = 0
    new_patterns: int = 0


# ── Error pattern templates ──────────────────────────────────────

ERROR_PATTERN_MAP = {
    "permission": "permission_denied",
    "not found": "not_found",
    "timeout": "timeout",
    "syntax error": "syntax_error",
    "import error": "import_error",
    "module not found": "module_not_found",
    "file exists": "file_exists",
    "no such file": "file_not_found",
    "connection": "connection_error",
    "memory": "memory_error",
}

# Lesson templates based on error patterns
LESSON_TEMPLATES = {
    "permission_denied": "When {tool} fails with permission error, try using elevated permissions or an alternative approach",
    "not_found": "When {tool} reports not found, verify paths exist before execution",
    "timeout": "When {tool} times out, consider breaking into smaller operations or increasing timeout",
    "syntax_error": "When {tool} hits syntax error, validate input format before execution",
    "import_error": "When {tool} fails with import error, ensure dependencies are installed first",
    "module_not_found": "When {tool} can't find module, check package installation and Python path",
    "file_exists": "When {tool} reports file exists, check for existing files before writing",
    "file_not_found": "When {tool} can't find file, verify path and create parent directories if needed",
    "connection_error": "When {tool} has connection issues, implement retry logic or check network",
    "memory_error": "When {tool} runs out of memory, process data in smaller chunks",
}


# ── ReflectionEngine ─────────────────────────────────────────────

class ReflectionEngine:
    """
    Analyzes agent execution to extract lessons and patterns.

    After each agent run, examines the ExecutionTracer spans to understand:
      - Which tools were used and their success/failure rates
      - Error patterns (e.g., "bash→permission_denied→fallback_write")
      - Generates actionable lessons stored in KnowledgeStore

    Usage::

        engine = ReflectionEngine(knowledge_store=ks)

        # After agent run completes
        result = engine.reflect(
            tracer=execution_tracer,
            messages=agent._messages,
            outcome="success",
        )

        # Before next run, get relevant past reflections
        reflections = engine.get_relevant_reflections("fix the build error")
    """

    # Minimum spans to trigger reflection (avoid noise from trivial runs)
    MIN_SPANS_FOR_REFLECTION = 2

    # Maximum reflections stored per session
    MAX_REFLECTIONS_PER_RUN = 5

    def __init__(
        self,
        knowledge_store: Optional[KnowledgeStore] = None,
        max_stored_reflections: int = 100,
    ):
        self._knowledge_store = knowledge_store
        self._max_stored = max_stored_reflections
        self._recent_reflections: List[Reflection] = []

    # ── Properties ───────────────────────────────────────────

    @property
    def recent_reflections(self) -> List[Reflection]:
        """Get recent reflections from this session."""
        return list(self._recent_reflections)

    @property
    def reflection_count(self) -> int:
        return len(self._recent_reflections)

    # ── Core: Reflect ────────────────────────────────────────

    def reflect(
        self,
        tracer: Optional[ExecutionTracer] = None,
        messages: Optional[list] = None,
        outcome: str = "success",
        task_description: str = "",
    ) -> ReflectionResult:
        """
        Analyze an execution run and generate reflections.

        Args:
            tracer: ExecutionTracer with span data from the run.
            messages: Conversation messages from the run.
            outcome: "success" or "failure".
            task_description: Description of what the agent was trying to do.

        Returns:
            ReflectionResult with the generated reflection and stats.
        """
        reflection_id = f"refl_{uuid.uuid4().hex[:12]}"

        # Extract tool usage from tracer
        tools_used = []
        successful_tools = []
        failed_tools = []
        error_patterns = []

        if tracer:
            spans = tracer.get_flat_spans()

            if len(spans) < self.MIN_SPANS_FOR_REFLECTION:
                # Too few spans — trivial run, no reflection needed
                reflection = Reflection(
                    reflection_id=reflection_id,
                    task_description=task_description or "trivial run",
                )
                return ReflectionResult(reflection=reflection)

            for span in spans:
                tool_name = self._extract_tool_name(span.operation)
                if tool_name:
                    tools_used.append(tool_name)
                    if span.status == "ok":
                        successful_tools.append(tool_name)
                    elif span.status == "error":
                        failed_tools.append(tool_name)
                        # Extract error pattern
                        pattern = self._classify_error(
                            tool_name, span.error_message or ""
                        )
                        if pattern:
                            error_patterns.append(pattern)

        # Infer task description from messages if not provided
        if not task_description and messages:
            task_description = self._infer_task_description(messages)

        # Calculate success rate
        total_tool_calls = len(tools_used)
        success_rate = (
            len(successful_tools) / total_tool_calls
            if total_tool_calls > 0
            else 1.0
        )

        # Generate lessons from error patterns
        lessons = self._generate_lessons(error_patterns, failed_tools)

        # Create reflection
        reflection = Reflection(
            reflection_id=reflection_id,
            task_description=task_description,
            tools_used=list(set(tools_used)),
            successful_tools=list(set(successful_tools)),
            failed_tools=list(set(failed_tools)),
            error_patterns=error_patterns,
            lessons=lessons,
            success_rate=round(success_rate, 2),
        )

        # Store in memory
        self._recent_reflections.append(reflection)
        if len(self._recent_reflections) > self.MAX_REFLECTIONS_PER_RUN:
            self._recent_reflections = self._recent_reflections[
                -self.MAX_REFLECTIONS_PER_RUN:
            ]

        # Store lessons in KnowledgeStore
        lessons_stored = self._store_lessons(reflection)

        logger.info(
            f"Reflection {reflection_id}: {len(tools_used)} tools, "
            f"{len(lessons)} lessons, success_rate={success_rate:.0%}"
        )

        return ReflectionResult(
            reflection=reflection,
            lessons_stored=lessons_stored,
            new_patterns=len(error_patterns),
        )

    # ── Query past reflections ───────────────────────────────

    def get_relevant_reflections(
        self,
        task_description: str,
        limit: int = 5,
    ) -> list:
        """
        Find past reflections relevant to the current task.

        Searches KnowledgeStore for reflection lessons matching the task
        description keywords.

        Args:
            task_description: Description of the current task.
            limit: Maximum reflections to return.

        Returns:
            List of KnowledgeEntry objects with stored reflection lessons.
        """
        if not self._knowledge_store:
            return []

        results = self._knowledge_store.search(task_description, limit=limit * 2)
        # Filter to only reflections category
        reflection_entries = [
            e for e in results if e.category == "reflections"
        ]
        return reflection_entries[:limit]

    # ── Generate insights ────────────────────────────────────

    def generate_insights(self, metrics_summary: Optional[dict] = None) -> List[str]:
        """
        Derive actionable insights from recent reflections and metrics.

        Args:
            metrics_summary: Optional dict from MetricsCollector.to_summary()

        Returns:
            List of insight strings.
        """
        insights = []

        # Insight from recent reflections
        if self._recent_reflections:
            # Aggregate error patterns
            all_patterns = []
            all_tools = []
            total_success = 0.0

            for r in self._recent_reflections:
                all_patterns.extend(r.error_patterns)
                all_tools.extend(r.tools_used)
                total_success += r.success_rate

            avg_success = total_success / len(self._recent_reflections)

            if avg_success < 0.7:
                insights.append(
                    f"Overall tool success rate is low ({avg_success:.0%}). "
                    f"Consider reviewing error handling strategies."
                )

            # Most common error pattern
            if all_patterns:
                pattern_counts: Dict[str, int] = {}
                for p in all_patterns:
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
                top_pattern = max(pattern_counts, key=pattern_counts.get)
                count = pattern_counts[top_pattern]
                insights.append(
                    f"Most frequent error pattern: '{top_pattern}' "
                    f"(occurred {count} time(s)). "
                    f"Address this to improve reliability."
                )

            # Most used tools
            if all_tools:
                tool_counts: Dict[str, int] = {}
                for t in all_tools:
                    tool_counts[t] = tool_counts.get(t, 0) + 1
                top_tool = max(tool_counts, key=tool_counts.get)
                insights.append(
                    f"Most used tool: '{top_tool}' "
                    f"(used {tool_counts[top_tool]} time(s))."
                )

        # Insights from metrics summary
        if metrics_summary:
            total_errors = metrics_summary.get("total_errors", 0)
            total_calls = metrics_summary.get("total_tool_calls", 0)

            if total_calls > 0 and total_errors / total_calls > 0.3:
                insights.append(
                    f"High error rate in metrics: {total_errors}/{total_calls} "
                    f"tool calls failed ({total_errors/total_calls:.0%})."
                )

        return insights

    # ── Internal helpers ─────────────────────────────────────

    @staticmethod
    def _extract_tool_name(operation: str) -> str:
        """Extract tool name from span operation string (e.g., 'tool.bash' → 'bash')."""
        if operation.startswith("tool."):
            return operation[5:]
        return ""

    @staticmethod
    def _classify_error(tool_name: str, error_message: str) -> str:
        """Classify an error message into a named pattern."""
        if not error_message:
            return ""
        lower = error_message.lower()
        for keyword, pattern_name in ERROR_PATTERN_MAP.items():
            if keyword in lower:
                return f"{tool_name}→{pattern_name}"
        return f"{tool_name}→unknown_error"

    @staticmethod
    def _generate_lessons(
        error_patterns: List[str], failed_tools: List[str]
    ) -> List[str]:
        """Generate human-readable lessons from error patterns."""
        lessons = []
        seen_patterns = set()

        for pattern in error_patterns:
            if pattern in seen_patterns:
                continue
            seen_patterns.add(pattern)

            # Parse pattern: "tool_name→error_type"
            parts = pattern.split("→", 1)
            if len(parts) != 2:
                continue
            tool_name, error_type = parts

            template = LESSON_TEMPLATES.get(error_type)
            if template:
                lessons.append(template.format(tool=tool_name))
            else:
                lessons.append(
                    f"Tool '{tool_name}' encountered '{error_type}'. "
                    f"Consider adding error handling for this case."
                )

        return lessons

    @staticmethod
    def _infer_task_description(messages: list) -> str:
        """Infer task description from the first user message."""
        for msg in messages:
            if hasattr(msg, "role") and msg.role == "user" and hasattr(msg, "content"):
                content = msg.content.strip()
                if content:
                    # Take first sentence or first 100 chars
                    first_sentence = content.split(".")[0]
                    return first_sentence[:100]
        return "unknown task"

    def _store_lessons(self, reflection: Reflection) -> int:
        """Store reflection lessons in KnowledgeStore."""
        if not self._knowledge_store or not reflection.lessons:
            return 0

        stored = 0
        for lesson in reflection.lessons:
            # Create a deterministic key from the lesson content
            lesson_key = f"reflection_{hashlib.md5(lesson.encode()).hexdigest()[:10]}"
            try:
                self._knowledge_store.remember(
                    category="reflections",
                    key=lesson_key,
                    value=lesson,
                )
                stored += 1
            except (ValueError, Exception) as e:
                logger.debug(f"Failed to store lesson: {e}")

        return stored
