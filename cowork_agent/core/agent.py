"""
Agent Loop — The core orchestrator.
Receives user messages, calls the LLM, executes tool calls, loops until done.
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from typing import AsyncIterator, Callable, Optional

from .models import Message, ToolCall, ToolResult, AgentResponse
from .providers.base import BaseLLMProvider
from .tool_registry import ToolRegistry
from .prompt_builder import PromptBuilder
from .safety_checker import SafetyChecker
from .context_manager import ContextManager
from .skill_registry import SkillRegistry
from .plan_mode import PlanManager
from .token_tracker import TokenTracker, TokenUsage, BudgetExceededError
from .execution_tracer import ExecutionTracer
from .tool_permissions import ToolPermissionManager
# Sprint 11: Advanced Memory System
from .conversation_summarizer import ConversationSummarizer
from .knowledge_store import KnowledgeStore
# Sprint 14: Streaming Events & Cancellation
from .stream_events import (
    StreamEvent, TextChunk, ToolStart, ToolProgress, ToolEnd, StatusUpdate,
)
from .stream_cancellation import StreamCancellationToken, StreamCancelledError
from .tool_progress import ProgressTracker

logger = logging.getLogger(__name__)


class Agent:
    """
    Main agent loop.

    Flow:
      user message → add to memory → build system prompt → call LLM
      → if tool_calls: execute all → add results to memory → loop
      → if no tool_calls: return final text to user
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        registry: ToolRegistry,
        prompt_builder: PromptBuilder,
        max_iterations: int = 15,
        on_tool_start: Optional[Callable] = None,
        on_tool_end: Optional[Callable] = None,
        on_status: Optional[Callable] = None,
        workspace_dir: str = "",
        max_context_tokens: int = 32000,
        skill_registry: Optional[SkillRegistry] = None,
        plan_manager: Optional[PlanManager] = None,
        token_tracker: Optional[TokenTracker] = None,
        # Sprint 11: Advanced Memory System
        summarizer: Optional[ConversationSummarizer] = None,
        knowledge_store: Optional[KnowledgeStore] = None,
    ):
        self.provider = provider
        self.registry = registry
        self.prompt_builder = prompt_builder
        self.max_iterations = max_iterations
        self.skill_registry = skill_registry
        self.plan_manager = plan_manager or PlanManager(workspace_dir=workspace_dir)
        # Sprint 4: Token tracking & budget enforcement
        self.token_tracker = token_tracker
        # Sprint 8: Execution tracer and permission manager (set by main.py)
        self.permission_manager: Optional[ToolPermissionManager] = None
        # Sprint 9: Cost and health tracking (set by main.py)
        self.cost_tracker = None      # CostTracker instance
        self.health_tracker = None    # ProviderHealthTracker instance
        # Sprint 12: Streaming & caching (set by main.py)
        self.response_cache = None    # ResponseCache instance
        self.stream_hardener = None   # StreamHardener instance
        self.retry_executor = None    # RetryExecutor instance
        # Sprint 14: Streaming events & cancellation (set by main.py)
        self._events_enabled: bool = False
        self._cancellation_token: Optional[StreamCancellationToken] = None

        # Sprint 15: Prompt optimization (set by main.py)
        self.token_estimator = None       # ModelTokenEstimator instance
        self.prompt_budget_manager = None  # PromptBudgetManager instance

        # Sprint 16: Observability (set by main.py)
        self.event_bus = None              # ObservabilityEventBus instance
        self.correlation_manager = None    # CorrelationIdManager instance
        self.metrics_registry = None       # MetricsRegistry instance
        self.benchmark = None              # PerformanceBenchmark instance
        self.health_orchestrator = None    # IntegratedHealthOrchestrator instance

        # Sprint 17: Security & Sandboxing (set by main.py)
        self.input_sanitizer = None        # InputSanitizer instance
        self.prompt_injection_detector = None  # PromptInjectionDetector instance
        self.credential_detector = None    # CredentialDetector instance
        self.sandboxed_executor = None     # SandboxedExecutor instance
        self.rate_limiter = None           # RateLimiter instance
        self.security_audit_log = None     # SecurityAuditLog instance

        # Sprint 19: Persistent Storage (set by main.py)
        self.persistent_store = None       # PersistentStore instance

        # Sprint 20: Dashboard (set by main.py)
        self.dashboard_provider = None     # DashboardDataProvider instance

        # Sprint 21: Multi-Agent Enhancement (set by main.py)
        self.specialization_registry = None  # SpecializationRegistry instance
        self.conversation_router = None      # ConversationRouter instance
        self.agent_pool = None               # AgentPool instance

        # Callbacks for UI updates
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_status = on_status  # Called with (message: str) for status updates

        # Safety & context management
        self.safety = SafetyChecker(workspace_dir=workspace_dir)

        # Sprint 11: Memory components
        self.summarizer = summarizer or ConversationSummarizer()
        self.knowledge_store = knowledge_store
        self.context_mgr = ContextManager(
            max_context_tokens=max_context_tokens,
            summarizer=self.summarizer,
        )

        # Sprint 11/15: Sliding window summary
        self._sliding_summary: Optional[str] = None
        self._summary_turn_count: int = 0
        self._SUMMARY_UPDATE_INTERVAL: int = 3  # Update every 3 user turns (was 5)

        # Conversation memory
        self._messages: list[Message] = []
        self._iteration = 0

        # ── P1: Circuit breaker — track consecutive failures per tool ──
        self._tool_failure_counts: dict[str, int] = defaultdict(int)
        self.CIRCUIT_BREAKER_THRESHOLD = 3  # Block tool after 3 consecutive failures

        # ── P1: Circular loop detection — track recent tool call signatures ──
        self._recent_tool_signatures: list[str] = []
        self.LOOP_DETECTION_WINDOW = 6  # Check last N tool calls
        self.LOOP_DETECTION_REPEAT = 3  # Flag if same signature appears N times

        # ── P1: Empty response counter ──
        self._consecutive_empty_responses = 0
        self.MAX_EMPTY_RESPONSES = 2  # Give up after N consecutive empty responses

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    # ── Sprint 16: Observability helpers ─────────────────────────

    def _emit_observability_event(
        self,
        event_type_name: str,
        component: str = "agent",
        severity: str = "info",
        metadata: dict | None = None,
    ) -> None:
        """Emit an observability event if event bus is attached."""
        if self.event_bus is None:
            return
        try:
            from .observability_event_bus import ObservabilityEvent, EventType
            et = EventType(event_type_name) if event_type_name in [e.value for e in EventType] else EventType.CUSTOM
            event = ObservabilityEvent(
                event_type=et,
                component=component,
                severity=severity,
                metadata=metadata or {},
            )
            self.event_bus.emit(event)
        except Exception as exc:
            logger.debug("Failed to emit observability event: %s", exc)

    def clear_history(self) -> None:
        """Reset conversation memory."""
        self._messages.clear()
        self._iteration = 0
        self._sliding_summary = None
        self._summary_turn_count = 0

    # Maximum retries for recovery handlers before giving up
    MAX_TRUNCATION_RETRIES = 2
    MAX_INTENT_NUDGES = 2

    async def run(self, user_input: str) -> str:
        """
        Process a user message and return the final assistant text.
        Runs the full tool-use loop until the LLM produces a text response
        or max iterations are reached.
        """
        # Add user message to memory
        self._messages.append(
            Message(role="user", content=user_input)
        )

        final_text = ""
        self._iteration = 0
        truncation_retries = 0
        intent_nudges = 0
        self._consecutive_empty_responses = 0
        self._tool_failure_counts.clear()
        self._recent_tool_signatures.clear()

        # Sprint 8: Execution tracing
        self._tracer = ExecutionTracer()
        self._root_span = self._tracer.start_span(
            "agent.run", user_input=user_input[:200]
        )

        # Sprint 16: Emit agent-started event
        self._emit_observability_event("agent_started", metadata={
            "user_input_length": len(user_input),
            "max_iterations": self.max_iterations,
        })

        while self._iteration < self.max_iterations:
            self._iteration += 1
            logger.debug(f"Agent iteration {self._iteration}")

            # Build system prompt with current context
            context = self._build_context()
            system_prompt = self.prompt_builder.build(
                tools=self.registry.get_schemas(),
                context=context,
            )

            # Sprint 15: Proactive pruning at 60% capacity
            if (self.token_estimator and
                    self.context_mgr.should_prune_proactively(self._messages, system_prompt)):
                logger.info("Proactive pruning triggered (60% capacity)")
                self._messages = self.context_mgr.prune(self._messages, system_prompt)

            # Prune context if approaching limit (75% threshold)
            if self.context_mgr.needs_pruning(self._messages, system_prompt):
                logger.info("Context pruning triggered before LLM call")
                if self.on_status:
                    self.on_status("Pruning conversation history to fit context window...")
                self._messages = self.context_mgr.prune(self._messages, system_prompt)

            # Sprint 4: Pre-call budget check
            if self.token_tracker:
                try:
                    self.token_tracker.check_budget()
                except BudgetExceededError as e:
                    budget_msg = f"[Budget exceeded: {e}]"
                    self._messages.append(Message(role="assistant", content=budget_msg))
                    return budget_msg

            # Sprint 12: Check response cache before calling LLM
            _cache_key = None
            if self.response_cache:
                _cache_key = self.response_cache.make_key(
                    model=getattr(self.provider, "model", ""),
                    messages=self._messages,
                    system_prompt=system_prompt,
                    tools=self.registry.get_schemas(),
                )
                cached = self.response_cache.get(_cache_key)
                if cached:
                    logger.info("Response cache hit — skipping LLM call")
                    response = cached
                    # Skip directly to response handling below
                else:
                    cached = None

            # Call the LLM (skip if cache hit)
            if not (self.response_cache and _cache_key and
                    self.response_cache.get(_cache_key)):
                try:
                    response = await self.provider.send_message(
                        messages=self._messages,
                        tools=self.registry.get_schemas(),
                        system_prompt=system_prompt,
                    )
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    error_text = f"Error communicating with LLM: {str(e)}"
                    self._messages.append(
                        Message(role="assistant", content=error_text)
                    )
                    return error_text

                # Sprint 12: Cache eligible responses
                if (self.response_cache and _cache_key and
                        self.response_cache._is_cacheable(response)):
                    self.response_cache.put(_cache_key, response)

            # Sprint 4: Record token usage from the response
            if self.token_tracker and response.usage:
                usage = TokenUsage(
                    input_tokens=response.usage.get("input_tokens", 0),
                    output_tokens=response.usage.get("output_tokens", 0),
                    cache_read_tokens=response.usage.get("cache_read_input_tokens", 0),
                    cache_write_tokens=response.usage.get("cache_creation_input_tokens", 0),
                    provider=self.provider.provider_name,
                    model=self.provider.model,
                )
                self.token_tracker.record(usage)

            # Sprint 9: Record cost and provider health
            if self.cost_tracker and response.usage:
                self.cost_tracker.record(
                    response.usage,
                    self.provider.provider_name,
                    self.provider.model,
                )
            if self.health_tracker:
                self.health_tracker.record_call(
                    self.provider.provider_name,
                    duration_ms=0,  # timing handled by metrics_collector
                    success=(response.stop_reason != "error"),
                )

            # ── P1: Empty response detection ──
            if not response.text and not response.tool_calls:
                self._consecutive_empty_responses += 1
                logger.warning(
                    f"Empty response from LLM ({self._consecutive_empty_responses}/{self.MAX_EMPTY_RESPONSES})"
                )
                if self._consecutive_empty_responses >= self.MAX_EMPTY_RESPONSES:
                    error_msg = (
                        "The language model returned empty responses repeatedly. "
                        "This may indicate a connectivity issue or model overload. "
                        "Please try again."
                    )
                    self._messages.append(Message(role="assistant", content=error_msg))
                    return error_msg

                # Nudge the LLM
                if self.on_status:
                    self.on_status("Received empty response, retrying...")
                self._messages.append(
                    Message(role="assistant", content="")
                )
                self._messages.append(
                    Message(
                        role="user",
                        content=(
                            "Your previous response was empty. Please respond to the "
                            "user's request. Either use a tool or provide a text answer."
                        ),
                    )
                )
                continue
            else:
                self._consecutive_empty_responses = 0

            # If the LLM returned text, accumulate it
            if response.text:
                final_text += response.text

            # ── Handle truncated response (hit token limit mid-tool-call) ──
            if response.stop_reason == "max_tokens" and not response.tool_calls:
                truncation_retries += 1
                if truncation_retries <= self.MAX_TRUNCATION_RETRIES:
                    logger.warning(
                        f"Response truncated mid-tool-call, retry {truncation_retries}/{self.MAX_TRUNCATION_RETRIES}"
                    )
                    if self.on_status:
                        self.on_status(
                            f"Response was truncated (token limit). "
                            f"Retrying with shorter content ({truncation_retries}/{self.MAX_TRUNCATION_RETRIES})..."
                        )
                    self._messages.append(
                        Message(role="assistant", content=response.text or "")
                    )
                    self._messages.append(
                        Message(
                            role="user",
                            content=(
                                "Your previous response was cut off before the tool call "
                                "could be completed. Please try again — this time, call the "
                                "tool directly with MUCH shorter content (a brief summary, "
                                "not a full article). Do NOT repeat your explanation, just "
                                "output the tool_calls JSON block immediately."
                            ),
                        )
                    )
                    final_text = ""
                    continue
                else:
                    # Exhausted retries — return what we have with an explanation
                    logger.error("Truncation retries exhausted, returning partial response")
                    fallback = (
                        "\n\n[The content was too long to fit in a single tool call. "
                        "Try asking for a shorter piece of content, or break the request "
                        "into smaller parts.]"
                    )
                    final_text = (response.text or "") + fallback
                    self._messages.append(
                        Message(role="assistant", content=final_text)
                    )
                    return final_text

            # If no tool calls, check for "intent without action"
            if not response.tool_calls:
                intent_type = self._detect_unfulfilled_intent(final_text)
                if intent_type:
                    intent_nudges += 1
                    if intent_nudges <= self.MAX_INTENT_NUDGES:
                        logger.info(
                            f"Detected unfulfilled intent ({intent_type}), nudge {intent_nudges}/{self.MAX_INTENT_NUDGES}"
                        )
                        if self.on_status:
                            self.on_status(
                                f"LLM showed content but didn't save it. "
                                f"Nudging to call the tool ({intent_nudges}/{self.MAX_INTENT_NUDGES})..."
                            )
                        self._messages.append(
                            Message(role="assistant", content=final_text)
                        )

                        # Pick a nudge message based on the type of unfulfilled intent
                        if intent_type == "code_block_dump":
                            nudge = (
                                "You showed file content as a code block in your response. "
                                "That does NOT create a file on disk. You MUST use the write "
                                "tool to actually save the file. Take the content you just "
                                "showed and call the write tool with it. Output ONLY the "
                                "```json\n{\"tool_calls\": [{\"name\": \"write\", \"id\": "
                                "\"tool_001\", \"input\": {\"file_path\": \"<path>\", "
                                "\"content\": \"<the content you showed>\"}}]}\n``` block — "
                                "no other text."
                            )
                        else:
                            nudge = (
                                "You said you would use a tool but didn't include a "
                                "tool_calls JSON block. Please proceed with the actual "
                                "tool call now. Output ONLY the ```json {\"tool_calls\": "
                                "[...]} ``` block — no other text."
                            )

                        self._messages.append(
                            Message(role="user", content=nudge)
                        )
                        final_text = ""
                        continue
                    else:
                        # Exhausted nudges — return text as-is
                        logger.warning("Intent nudge retries exhausted, returning text response")
                        self._messages.append(
                            Message(role="assistant", content=final_text)
                        )
                        return final_text

                # Genuinely done — return final text
                if final_text:
                    self._messages.append(
                        Message(role="assistant", content=final_text)
                    )
                return final_text

            # ── Tool calls found — reset recovery counters ──
            truncation_retries = 0
            intent_nudges = 0

            # Add assistant message with tool calls to memory
            self._messages.append(
                Message(
                    role="assistant",
                    content=response.text or "",
                    tool_calls=response.tool_calls,
                )
            )

            # Execute all tool calls (with circuit breaker)
            results = await self._execute_tools(response.tool_calls)

            # ── P1: Circuit breaker + loop detection (post-execution) ──
            #
            # Three categories of results, each handled differently:
            #   1. Success        → reset circuit breaker + record signature for loop detection
            #   2. Policy block   → SKIP (not a real failure — don't inflate circuit breaker)
            #   3. Execution fail → count toward circuit breaker (but not loop detection)
            #
            # Policy blocks are identified by their error prefix ([CIRCUIT BREAKER],
            # [PLAN MODE], [SAFETY], [VALIDATION]).  These are gating decisions, not
            # tool execution failures — a tool blocked by plan mode shouldn't be
            # treated as "broken" by the circuit breaker.
            for call, result in zip(response.tool_calls, results):
                if result.success:
                    self._tool_failure_counts[call.name] = 0  # Reset on success
                    self._record_tool_signature(call)
                elif self._is_policy_block(result):
                    # Policy block — don't count as execution failure
                    pass
                else:
                    # Real execution failure — count for circuit breaker
                    self._tool_failure_counts[call.name] += 1
                    if self._tool_failure_counts[call.name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                        logger.warning(
                            f"Circuit breaker tripped for tool '{call.name}' "
                            f"after {self.CIRCUIT_BREAKER_THRESHOLD} consecutive failures"
                        )

            # ── P1: Check for circular loop AFTER execution ──
            loop_detected = self._detect_circular_loop()
            if loop_detected:
                logger.warning(f"Circular loop detected: {loop_detected}")
                if self.on_status:
                    self.on_status(f"Detected repeated tool calls: {loop_detected}")
                loop_msg = (
                    f"I noticed I'm calling the same tool ({loop_detected}) repeatedly "
                    f"with the same arguments, which indicates a loop. Let me stop and "
                    f"reconsider my approach."
                )
                self._messages.append(Message(role="assistant", content=loop_msg))
                self._messages.append(
                    Message(
                        role="user",
                        content=(
                            f"You were calling '{loop_detected}' in a loop with the same "
                            f"arguments. This is not productive. Either try a DIFFERENT "
                            f"approach or explain what's blocking you."
                        ),
                    )
                )
                final_text = ""
                continue

            # Add tool results to memory
            self._messages.append(
                Message(
                    role="tool_result",
                    content="",
                    tool_results=results,
                )
            )

            # Reset final_text for next iteration (LLM will produce new text)
            final_text = ""

        # Max iterations reached
        timeout_msg = (
            f"[Agent reached maximum iterations ({self.max_iterations}). "
            "Stopping to prevent infinite loops.]"
        )
        if final_text:
            final_text += "\n\n" + timeout_msg
        else:
            final_text = timeout_msg

        self._messages.append(
            Message(role="assistant", content=final_text)
        )
        # Sprint 8: End root span
        if hasattr(self, '_tracer') and self._root_span:
            self._tracer.end_span(self._root_span, status="ok")
        return final_text

    # Prefixes that indicate a policy block (not an execution failure).
    # Circuit breaker should NOT count these as tool failures.
    POLICY_BLOCK_PREFIXES = ("[CIRCUIT BREAKER]", "[PLAN MODE]", "[SAFETY]", "[VALIDATION]", "[PERMISSION]")

    async def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """
        Execute tool calls with safety checks, circuit breaker, then in parallel.

        Returns results in the SAME ORDER as tool_calls — this is critical
        because run() zips (tool_calls, results) for tracking.
        """
        schemas = self.registry.get_schemas()

        # Pre-allocate result slots to maintain ordering.
        # Blocked calls get their result placed at their original index.
        # Safe calls are collected, executed in parallel, then placed back.
        results: list[Optional[ToolResult]] = [None] * len(tool_calls)
        safe_calls: list[tuple[int, ToolCall]] = []  # (original_index, call)

        for i, call in enumerate(tool_calls):
            # ── P1: Circuit breaker — skip tools that have failed too many times ──
            if self._tool_failure_counts.get(call.name, 0) >= self.CIRCUIT_BREAKER_THRESHOLD:
                logger.warning(f"Circuit breaker: skipping '{call.name}' (too many failures)")
                if self.on_tool_start:
                    self.on_tool_start(call)
                breaker_result = ToolResult(
                    tool_id=call.tool_id,
                    success=False,
                    output="",
                    error=(
                        f"[CIRCUIT BREAKER] Tool '{call.name}' has failed "
                        f"{self.CIRCUIT_BREAKER_THRESHOLD} consecutive times. "
                        f"Try a different approach or tool."
                    ),
                )
                if self.on_tool_end:
                    self.on_tool_end(call, breaker_result)
                results[i] = breaker_result
                continue

            # ── Sprint 8: Permission check ──
            if self.permission_manager:
                perm_ok, perm_reason = self.permission_manager.check_all(call.name)
                if not perm_ok:
                    logger.info(f"Permission denied for tool: {call.name}")
                    if self.on_tool_start:
                        self.on_tool_start(call)
                    perm_result = ToolResult(
                        tool_id=call.tool_id,
                        success=False,
                        output="",
                        error=f"[PERMISSION] {perm_reason}",
                    )
                    if self.on_tool_end:
                        self.on_tool_end(call, perm_result)
                    results[i] = perm_result
                    continue

            # ── Plan mode check — restrict to read-only tools ──
            if self.plan_manager and self.plan_manager.is_plan_mode:
                allowed, reason = self.plan_manager.is_tool_allowed(call.name)
                if not allowed:
                    logger.info(f"Plan mode blocked tool: {call.name}")
                    if self.on_tool_start:
                        self.on_tool_start(call)
                    blocked_result = ToolResult(
                        tool_id=call.tool_id,
                        success=False,
                        output="",
                        error=f"[PLAN MODE] {reason}",
                    )
                    if self.on_tool_end:
                        self.on_tool_end(call, blocked_result)
                    results[i] = blocked_result
                    continue

            # ── Safety check ──
            check = self.safety.check(call, schemas)

            if check.blocked:
                logger.warning(f"Safety BLOCKED tool call: {call.name} — {check.block_reason}")
                if self.on_tool_start:
                    self.on_tool_start(call)
                blocked_result = check.to_tool_result(call.tool_id)
                if self.on_tool_end:
                    self.on_tool_end(call, blocked_result)
                results[i] = blocked_result
                continue

            if check.warnings:
                for w in check.warnings:
                    logger.info(f"Safety warning for {call.name}: {w}")

            # ── Input validation ──
            validation_error = self.safety.validate_tool_inputs(call, schemas)
            if validation_error:
                logger.warning(f"Invalid tool inputs: {validation_error}")
                if self.on_tool_start:
                    self.on_tool_start(call)
                error_result = ToolResult(
                    tool_id=call.tool_id,
                    success=False,
                    output="",
                    error=f"[VALIDATION] {validation_error}",
                )
                if self.on_tool_end:
                    self.on_tool_end(call, error_result)
                results[i] = error_result
                continue

            safe_calls.append((i, call))

        # Execute safe calls in parallel
        if safe_calls:
            calls_only = [call for _, call in safe_calls]

            for call in calls_only:
                if self.on_tool_start:
                    self.on_tool_start(call)

            executed = await self.registry.execute_parallel(calls_only)

            for (idx, call), result in zip(safe_calls, executed):
                if self.on_tool_end:
                    self.on_tool_end(call, result)
                results[idx] = result

        # Safety: ensure no None slots remain (defensive — shouldn't happen)
        final_results: list[ToolResult] = []
        for i, r in enumerate(results):
            if r is None:
                logger.error(f"BUG: result slot {i} is None for tool '{tool_calls[i].name}'")
                final_results.append(ToolResult(
                    tool_id=tool_calls[i].tool_id,
                    success=False,
                    output="",
                    error="[INTERNAL] Tool result was not populated. This is a bug.",
                ))
            else:
                final_results.append(r)
        return final_results

    def _is_policy_block(self, result: ToolResult) -> bool:
        """
        Check if a ToolResult was blocked by policy (circuit breaker, plan mode,
        safety, validation) rather than failing during actual execution.

        Policy blocks should NOT be counted as tool failures by the circuit
        breaker, because the tool never actually ran — it was gated before
        execution.  Counting them would cause:
          - Plan mode blocking a tool 3x → circuit breaker trips when plan mode exits
          - Safety blocking a tool 3x → tool permanently unusable even if inputs change
          - Circuit breaker re-counting its own blocks → counter keeps climbing uselessly
        """
        if not result.error:
            return False
        return result.error.startswith(self.POLICY_BLOCK_PREFIXES)

    def _record_tool_signature(self, call: ToolCall) -> None:
        """
        Record a SUCCESSFUL tool call's signature for loop detection.
        Only successful calls are tracked — failed calls are handled by the
        circuit breaker instead.
        """
        sig = self._tool_call_signature(call)
        self._recent_tool_signatures.append(sig)

        # Keep only the last N signatures
        if len(self._recent_tool_signatures) > self.LOOP_DETECTION_WINDOW:
            self._recent_tool_signatures = self._recent_tool_signatures[-self.LOOP_DETECTION_WINDOW:]

    def _detect_circular_loop(self) -> Optional[str]:
        """
        Check the recent tool signature history for repeated patterns.

        Returns the tool name if a loop is detected, None otherwise.

        This is called AFTER execution so it doesn't interfere with the
        circuit breaker. Only successful calls are in the signature list,
        so a tool that keeps failing won't trigger loop detection — it will
        be caught by the circuit breaker instead.
        """
        from collections import Counter
        counts = Counter(self._recent_tool_signatures)
        for sig, count in counts.items():
            if count >= self.LOOP_DETECTION_REPEAT:
                # Extract tool name from signature
                tool_name = sig.split(":")[0] if ":" in sig else sig
                return tool_name

        return None

    @staticmethod
    def _tool_call_signature(call: ToolCall) -> str:
        """Create a deterministic signature for a tool call (name + sorted input hash)."""
        input_str = json.dumps(call.input, sort_keys=True, default=str)
        input_hash = hashlib.md5(input_str.encode()).hexdigest()[:12]
        return f"{call.name}:{input_hash}"

    @staticmethod
    def _detect_unfulfilled_intent(text: str) -> Optional[str]:
        """
        Detect if the LLM said it would use a tool but didn't actually call one.

        Returns:
            None — no unfulfilled intent detected
            "intent_phrase" — LLM said it would act but didn't
            "code_block_dump" — LLM showed a large code block instead of calling write tool
        """
        if not text:
            return None

        text_lower = text.lower()

        # Pattern 1: Large code block dumped as text instead of using write tool
        # Check this FIRST — it's the more specific and actionable pattern
        import re
        code_blocks = re.findall(r'```\w*\n([\s\S]*?)```', text)
        if code_blocks:
            largest_block = max(len(b) for b in code_blocks)
            if largest_block > 500:
                logger.info(
                    f"Detected large code block ({largest_block} chars) "
                    "dumped as text instead of using write tool"
                )
                return "code_block_dump"

        # Pattern 2: Intent phrases near the end of the text
        intent_phrases = [
            "i'll create", "i will create",
            "let me create", "let me write",
            "i'll write", "i will write",
            "now i'll", "now i will",
            "let me save", "i'll save",
            "i'll use the write", "i'll use the bash",
            "let me use the", "i'll call the",
            "here's the file", "here is the file",
            "creating the file", "writing the file",
        ]
        last_200 = text_lower[-200:]
        if any(phrase in last_200 for phrase in intent_phrases):
            return "intent_phrase"

        return None

    async def run_stream(self, user_input: str) -> AsyncIterator[str]:
        """
        Process a user message with streaming text output.

        Yields text chunks as they arrive from the LLM. Tool calls are
        handled silently between streamed segments. The full conversation
        is still tracked in memory.

        Includes the same recovery mechanisms as run():
          - Empty response detection & retry
          - Truncation detection & retry
          - Unfulfilled intent nudging
          - Circuit breaker & loop detection
        """
        # Add user message to memory
        self._messages.append(Message(role="user", content=user_input))

        self._iteration = 0
        truncation_retries = 0
        intent_nudges = 0
        self._consecutive_empty_responses = 0
        self._tool_failure_counts.clear()
        self._recent_tool_signatures.clear()

        accumulated_text = ""

        while self._iteration < self.max_iterations:
            self._iteration += 1
            logger.debug(f"Agent stream iteration {self._iteration}")

            context = self._build_context()
            system_prompt = self.prompt_builder.build(
                tools=self.registry.get_schemas(),
                context=context,
            )

            if self.context_mgr.needs_pruning(self._messages, system_prompt):
                self._messages = self.context_mgr.prune(self._messages, system_prompt)

            # Stream from the LLM (with optional stream hardener)
            full_text = ""
            try:
                raw_stream = self.provider.send_message_stream(
                    messages=self._messages,
                    tools=self.registry.get_schemas(),
                    system_prompt=system_prompt,
                )
                # Sprint 12: Wrap with stream hardener if available
                if self.stream_hardener:
                    stream = self.stream_hardener.wrap(raw_stream)
                else:
                    stream = raw_stream
                async for chunk in stream:
                    full_text += chunk
                    yield chunk
            except Exception as e:
                error_text = f"Error communicating with LLM: {str(e)}"
                self._messages.append(Message(role="assistant", content=error_text))
                yield error_text
                return

            # Get the parsed response from the provider
            response = self.provider.last_stream_response
            if response is None:
                # Fallback: treat streamed text as plain text response
                if full_text:
                    self._messages.append(Message(role="assistant", content=full_text))
                return

            # ── Empty response handling ──
            if not response.text and not response.tool_calls:
                self._consecutive_empty_responses += 1
                if self._consecutive_empty_responses >= self.MAX_EMPTY_RESPONSES:
                    msg = "The language model returned empty responses repeatedly."
                    self._messages.append(Message(role="assistant", content=msg))
                    yield msg
                    return
                self._messages.append(Message(role="assistant", content=""))
                self._messages.append(
                    Message(role="user", content="Your previous response was empty. Please respond.")
                )
                continue
            else:
                self._consecutive_empty_responses = 0

            # Accumulate text
            if response.text:
                accumulated_text += response.text

            # ── Truncation recovery (same as run()) ──
            if response.stop_reason == "max_tokens" and not response.tool_calls:
                truncation_retries += 1
                if truncation_retries <= self.MAX_TRUNCATION_RETRIES:
                    logger.warning(
                        f"Stream truncated mid-tool-call, retry {truncation_retries}/{self.MAX_TRUNCATION_RETRIES}"
                    )
                    self._messages.append(
                        Message(role="assistant", content=response.text or "")
                    )
                    self._messages.append(
                        Message(
                            role="user",
                            content=(
                                "Your previous response was cut off before the tool call "
                                "could be completed. Please try again — call the tool directly "
                                "with shorter content. Output ONLY the tool_calls JSON block."
                            ),
                        )
                    )
                    accumulated_text = ""
                    continue
                else:
                    fallback = (
                        "\n\n[The content was too long to fit in a single tool call. "
                        "Try asking for shorter content.]"
                    )
                    text = (response.text or "") + fallback
                    self._messages.append(Message(role="assistant", content=text))
                    yield fallback
                    return

            # No tool calls — check for unfulfilled intent, then return
            if not response.tool_calls:
                text = response.text or full_text

                # ── Intent nudging (same as run()) ──
                intent_type = self._detect_unfulfilled_intent(text)
                if intent_type:
                    intent_nudges += 1
                    if intent_nudges <= self.MAX_INTENT_NUDGES:
                        logger.info(f"Stream: unfulfilled intent ({intent_type}), nudge {intent_nudges}")
                        self._messages.append(Message(role="assistant", content=text))
                        if intent_type == "code_block_dump":
                            nudge = (
                                "You showed file content as a code block. That does NOT create a file. "
                                "You MUST use the write tool. Call it now — output ONLY the tool_calls JSON."
                            )
                        else:
                            nudge = (
                                "You said you would use a tool but didn't include a tool_calls JSON block. "
                                "Please proceed with the actual tool call now."
                            )
                        self._messages.append(Message(role="user", content=nudge))
                        accumulated_text = ""
                        continue
                    # Exhausted nudges — return text as-is

                self._messages.append(Message(role="assistant", content=text))
                return

            # ── Tool calls found — reset recovery counters ──
            truncation_retries = 0
            intent_nudges = 0

            # Tool calls — handle them (not streamed)
            self._messages.append(
                Message(role="assistant", content=response.text or "", tool_calls=response.tool_calls)
            )

            results = await self._execute_tools(response.tool_calls)

            # Track failures (circuit breaker) and successes (loop detection)
            for call, result in zip(response.tool_calls, results):
                if result.success:
                    self._tool_failure_counts[call.name] = 0
                    self._record_tool_signature(call)
                elif self._is_policy_block(result):
                    pass  # Policy block — don't count as execution failure
                else:
                    self._tool_failure_counts[call.name] += 1

            # Check for circular loop after execution — nudge instead of aborting
            loop_detected = self._detect_circular_loop()
            if loop_detected:
                logger.warning(f"Stream: circular loop detected: {loop_detected}")
                loop_msg = (
                    f"I noticed I'm calling the same tool ({loop_detected}) repeatedly "
                    f"with the same arguments. Let me reconsider my approach."
                )
                self._messages.append(Message(role="assistant", content=loop_msg))
                self._messages.append(
                    Message(
                        role="user",
                        content=(
                            f"You were calling '{loop_detected}' in a loop with the same "
                            f"arguments. Try a DIFFERENT approach or explain what's blocking you."
                        ),
                    )
                )
                yield f"\n{loop_msg}"
                accumulated_text = ""
                continue

            self._messages.append(
                Message(role="tool_result", content="", tool_results=results)
            )
            # Continue loop — next LLM call will be streamed

        yield f"\n[Agent reached maximum iterations ({self.max_iterations}).]"

    # ── Sprint 14: Streaming Events ─────────────────────────

    async def run_stream_events(
        self,
        user_input: str,
        cancellation_token: Optional[StreamCancellationToken] = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Process a user message with structured streaming events.

        Like run_stream(), but yields StreamEvent objects instead of raw strings.
        Supports cancellation via token and progress tracking for tools.

        Events emitted:
          - TextChunk: streaming text from LLM
          - ToolStart: before tool execution
          - ToolProgress: during long-running tool execution
          - ToolEnd: after tool execution with result and timing
          - StatusUpdate: agent status (pruning, retrying, cancellation)
        """
        token = cancellation_token or self._cancellation_token

        # Add user message to memory
        self._messages.append(Message(role="user", content=user_input))

        self._iteration = 0
        truncation_retries = 0
        intent_nudges = 0
        self._consecutive_empty_responses = 0
        self._tool_failure_counts.clear()
        self._recent_tool_signatures.clear()

        accumulated_text = ""

        while self._iteration < self.max_iterations:
            self._iteration += 1

            # Cancellation check — before LLM call
            if token and token.is_cancelled:
                yield StatusUpdate(message="Stream cancelled", severity="warning")
                return

            context = self._build_context()
            system_prompt = self.prompt_builder.build(
                tools=self.registry.get_schemas(),
                context=context,
            )

            if self.context_mgr.needs_pruning(self._messages, system_prompt):
                self._messages = self.context_mgr.prune(self._messages, system_prompt)
                yield StatusUpdate(message="Pruning conversation context...", severity="info")

            # Stream from the LLM
            full_text = ""
            try:
                raw_stream = self.provider.send_message_stream(
                    messages=self._messages,
                    tools=self.registry.get_schemas(),
                    system_prompt=system_prompt,
                )
                if self.stream_hardener:
                    stream = self.stream_hardener.wrap(raw_stream)
                else:
                    stream = raw_stream

                async for chunk in stream:
                    # Cancellation check — after each chunk
                    if token and token.is_cancelled:
                        yield StatusUpdate(message="Stream cancelled", severity="warning")
                        return
                    full_text += chunk
                    yield TextChunk(text=chunk)

            except StreamCancelledError:
                yield StatusUpdate(message="Stream cancelled", severity="warning")
                return
            except Exception as e:
                error_text = f"Error communicating with LLM: {str(e)}"
                self._messages.append(Message(role="assistant", content=error_text))
                yield StatusUpdate(message=error_text, severity="warning")
                return

            # Get the parsed response from the provider
            response = self.provider.last_stream_response
            if response is None:
                if full_text:
                    self._messages.append(Message(role="assistant", content=full_text))
                return

            # ── Empty response handling ──
            if not response.text and not response.tool_calls:
                self._consecutive_empty_responses += 1
                if self._consecutive_empty_responses >= self.MAX_EMPTY_RESPONSES:
                    msg = "The language model returned empty responses repeatedly."
                    self._messages.append(Message(role="assistant", content=msg))
                    yield StatusUpdate(message=msg, severity="warning")
                    return
                self._messages.append(Message(role="assistant", content=""))
                self._messages.append(
                    Message(role="user", content="Your previous response was empty. Please respond.")
                )
                yield StatusUpdate(message="Empty response, retrying...", severity="info")
                continue
            else:
                self._consecutive_empty_responses = 0

            if response.text:
                accumulated_text += response.text

            # ── Truncation recovery ──
            if response.stop_reason == "max_tokens" and not response.tool_calls:
                truncation_retries += 1
                if truncation_retries <= self.MAX_TRUNCATION_RETRIES:
                    self._messages.append(Message(role="assistant", content=response.text or ""))
                    self._messages.append(
                        Message(
                            role="user",
                            content=(
                                "Your previous response was cut off before the tool call "
                                "could be completed. Please try again — call the tool directly "
                                "with shorter content. Output ONLY the tool_calls JSON block."
                            ),
                        )
                    )
                    yield StatusUpdate(
                        message=f"Response truncated, retry {truncation_retries}...",
                        severity="info",
                    )
                    accumulated_text = ""
                    continue
                else:
                    fallback = (
                        "\n\n[The content was too long to fit in a single tool call. "
                        "Try asking for shorter content.]"
                    )
                    text = (response.text or "") + fallback
                    self._messages.append(Message(role="assistant", content=text))
                    yield TextChunk(text=fallback)
                    return

            # No tool calls — check intent, then return
            if not response.tool_calls:
                text = response.text or full_text
                intent_type = self._detect_unfulfilled_intent(text)
                if intent_type:
                    intent_nudges += 1
                    if intent_nudges <= self.MAX_INTENT_NUDGES:
                        self._messages.append(Message(role="assistant", content=text))
                        if intent_type == "code_block_dump":
                            nudge = (
                                "You showed file content as a code block. That does NOT create a file. "
                                "You MUST use the write tool. Call it now — output ONLY the tool_calls JSON."
                            )
                        else:
                            nudge = (
                                "You said you would use a tool but didn't include a tool_calls JSON block. "
                                "Please proceed with the actual tool call now."
                            )
                        self._messages.append(Message(role="user", content=nudge))
                        accumulated_text = ""
                        continue

                self._messages.append(Message(role="assistant", content=text))
                return

            # ── Tool calls found — execute with events ──
            truncation_retries = 0
            intent_nudges = 0

            self._messages.append(
                Message(role="assistant", content=response.text or "", tool_calls=response.tool_calls)
            )

            # Execute tools with event emission
            import time as _time
            results = []
            for call in response.tool_calls:
                # Cancellation check — before each tool
                if token and token.is_cancelled:
                    yield StatusUpdate(message="Stream cancelled during tool execution", severity="warning")
                    return

                yield ToolStart(tool_call=call)
                tool_start = _time.time()

                # Create progress callback that yields ToolProgress events
                progress_events: list[ToolProgress] = []

                def _make_progress_cb(tc: ToolCall):
                    def _cb(pct: int, msg: str):
                        progress_events.append(ToolProgress(
                            tool_call=tc,
                            progress_percent=pct,
                            message=msg,
                        ))
                    return _cb

                # Execute the single tool
                single_results = await self._execute_tools_with_progress(
                    [call], _make_progress_cb(call),
                )
                result = single_results[0]

                # Yield any accumulated progress events
                for pe in progress_events:
                    yield pe

                duration = (_time.time() - tool_start) * 1000
                yield ToolEnd(tool_call=call, result=result, duration_ms=duration)
                results.append(result)

            # Track failures and successes
            for call, result in zip(response.tool_calls, results):
                if result.success:
                    self._tool_failure_counts[call.name] = 0
                    self._record_tool_signature(call)
                elif self._is_policy_block(result):
                    pass
                else:
                    self._tool_failure_counts[call.name] += 1

            # Check for circular loop
            loop_detected = self._detect_circular_loop()
            if loop_detected:
                loop_msg = (
                    f"I noticed I'm calling the same tool ({loop_detected}) repeatedly "
                    f"with the same arguments. Let me reconsider my approach."
                )
                self._messages.append(Message(role="assistant", content=loop_msg))
                self._messages.append(
                    Message(
                        role="user",
                        content=(
                            f"You were calling '{loop_detected}' in a loop with the same "
                            f"arguments. Try a DIFFERENT approach or explain what's blocking you."
                        ),
                    )
                )
                yield StatusUpdate(message=loop_msg, severity="warning")
                accumulated_text = ""
                continue

            self._messages.append(
                Message(role="tool_result", content="", tool_results=results)
            )

        yield StatusUpdate(
            message=f"Agent reached maximum iterations ({self.max_iterations}).",
            severity="warning",
        )

    async def _execute_tools_with_progress(
        self,
        tool_calls: list[ToolCall],
        progress_callback=None,
    ) -> list[ToolResult]:
        """
        Execute tool calls with an optional progress callback.

        Wraps _execute_tools logic for single tool + progress callback support.
        Falls back to standard _execute_tools for multiple calls.
        """
        if len(tool_calls) != 1 or progress_callback is None:
            return await self._execute_tools(tool_calls)

        call = tool_calls[0]
        schemas = self.registry.get_schemas()

        # Check circuit breaker
        if self._tool_failure_counts.get(call.name, 0) >= self.CIRCUIT_BREAKER_THRESHOLD:
            return [ToolResult(
                tool_id=call.tool_id, success=False, output="",
                error=f"[CIRCUIT BREAKER] Tool '{call.name}' has failed too many times.",
            )]

        # Permission check
        if self.permission_manager:
            perm_ok, perm_reason = self.permission_manager.check_all(call.name)
            if not perm_ok:
                return [ToolResult(
                    tool_id=call.tool_id, success=False, output="",
                    error=f"[PERMISSION] {perm_reason}",
                )]

        # Plan mode check
        if self.plan_manager and self.plan_manager.is_plan_mode:
            allowed, reason = self.plan_manager.is_tool_allowed(call.name)
            if not allowed:
                return [ToolResult(
                    tool_id=call.tool_id, success=False, output="",
                    error=f"[PLAN MODE] {reason}",
                )]

        # Safety check
        check = self.safety.check(call, schemas)
        if check.blocked:
            return [check.to_tool_result(call.tool_id)]

        # Validation
        validation_error = self.safety.validate_tool_inputs(call, schemas)
        if validation_error:
            return [ToolResult(
                tool_id=call.tool_id, success=False, output="",
                error=f"[VALIDATION] {validation_error}",
            )]

        # Execute with progress callback
        try:
            result = await self.registry.execute_with_progress(call, progress_callback)
        except Exception as e:
            result = ToolResult(
                tool_id=call.tool_id, success=False, output="",
                error=f"Execution error: {str(e)}",
            )
        return [result]

    # ── Sprint 11: Sliding summary ─────────────────────────

    def _maybe_update_summary(self) -> None:
        """Update the sliding summary every N user turns."""
        user_turns = sum(1 for m in self._messages if m.role == "user")
        if user_turns <= self._summary_turn_count:
            return
        if (user_turns - self._summary_turn_count) >= self._SUMMARY_UPDATE_INTERVAL:
            recent = self._messages[-20:]  # Summarize last 20 messages
            self._sliding_summary = self.summarizer.update_sliding_summary(
                self._sliding_summary or "", recent
            )
            self._summary_turn_count = user_turns
            logger.debug(f"Sliding summary updated at turn {user_turns}")

    def _build_context(self) -> dict:
        """Build runtime context for the prompt builder."""
        # Sprint 11: Update sliding summary if interval reached
        self._maybe_update_summary()

        # Sprint 15: Deduplicate messages before building context
        if self.token_estimator:
            deduped = self.context_mgr.deduplicate_messages(self._messages)
            if len(deduped) < len(self._messages):
                self._messages = deduped

        ctx = {
            "iteration": self._iteration,
        }

        # Sprint 11/15: Memory context with relevance scoring
        if self._sliding_summary:
            ctx["memory_summary"] = self._sliding_summary

        # Get the latest user message for relevance scoring
        last_user_msg = ""
        if self._messages:
            for msg in reversed(self._messages):
                if msg.role == "user":
                    last_user_msg = msg.content
                    break

        if self.knowledge_store and self.knowledge_store.size > 0:
            if self.token_estimator and last_user_msg:
                # Sprint 15: Score-based knowledge injection
                self._inject_scored_knowledge(ctx, last_user_msg)
            else:
                # Fallback: original chronological injection
                entries = []
                for cat in ("facts", "preferences", "decisions"):
                    entries.extend(self.knowledge_store.recall_all(cat)[:5])
                if entries:
                    ctx["knowledge_entries"] = entries

        # Get todos from the todo tool if available (safe — won't crash if missing)
        try:
            todo_tool = self.registry.get_tool("todo_write")
            if hasattr(todo_tool, "get_context"):
                ctx["todos"] = todo_tool.get_context()
        except KeyError:
            pass

        # Plan mode context
        if self.plan_manager and self.plan_manager.is_plan_mode:
            ctx["plan_mode"] = True
            ctx["plan_mode_prompt"] = self.plan_manager.get_plan_mode_prompt()

        # Sprint 4: Token usage context
        if self.token_tracker and self.token_tracker.call_count > 0:
            ctx["token_usage"] = self.token_tracker.summary()

        # Match skills from the latest user message
        if self.skill_registry and self._messages:
            if last_user_msg:
                matched = self.skill_registry.match_skills(last_user_msg)
                if matched:
                    ctx["active_skills"] = matched
                    logger.info(f"Matched skills: {[s.name for s in matched]}")

        return ctx

    def _inject_scored_knowledge(self, ctx: dict, recent_user_message: str) -> None:
        """
        Sprint 15: Inject knowledge entries scored by relevance to user message.

        Replaces the chronological top-5-per-category with relevance-ranked entries.
        """
        all_entries = []
        for cat in ("decisions", "preferences", "facts"):
            all_entries.extend(self.knowledge_store.recall_all(cat))

        if not all_entries:
            return

        # Score each entry
        scored = [
            (self.context_mgr.score_knowledge_entry(entry, recent_user_message), entry)
            for entry in all_entries
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top entries (max 10)
        top_entries = [entry for _, entry in scored[:10]]
        if top_entries:
            ctx["knowledge_entries"] = top_entries
            logger.debug(
                f"Injected {len(top_entries)} scored knowledge entries "
                f"(top scores: {[f'{s:.2f}' for s, _ in scored[:5]]})"
            )
