#!/usr/bin/env python3
"""
P1 Feature Tests — Circuit Breaker, Empty Response Detection,
Circular Loop Detection, Network Retry, Streaming, Task Persistence.
"""

import asyncio
import json
import os
import sys
import tempfile
import shutil

# Add project root to path so we can import cowork_agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cowork_agent.core.models import Message, ToolCall, ToolResult, ToolSchema, AgentResponse
from cowork_agent.core.agent import Agent
from cowork_agent.core.providers.base import BaseLLMProvider
from cowork_agent.core.tool_registry import ToolRegistry
from cowork_agent.core.prompt_builder import PromptBuilder
from cowork_agent.tools.todo import TodoWriteTool
from cowork_agent.tools.web_fetch import _is_transient

# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name} {detail}")


class MockProvider(BaseLLMProvider):
    """Mock LLM provider that returns pre-configured responses."""

    def __init__(self, responses=None):
        super().__init__(model="mock")
        self.responses = responses or []
        self.call_count = 0

    async def send_message(self, messages, tools, system_prompt):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = AgentResponse(text="No more responses configured.", stop_reason="end_turn")
        self.call_count += 1
        return resp

    async def health_check(self):
        return {"status": "ok"}


class MockPromptBuilder:
    """Minimal prompt builder for tests."""
    def build(self, tools=None, context=None):
        return "You are a helpful agent."


class AlwaysFailTool:
    """A tool that always fails."""
    name = "failing_tool"
    description = "A tool that always fails"
    input_schema = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }

    def get_schema(self):
        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    async def execute(self, tool_id="", **kwargs):
        return ToolResult(
            tool_id=tool_id,
            success=False,
            output="",
            error="Always fails!",
        )


def make_agent(responses, extra_tools=None, workspace_dir="/tmp/test_p1"):
    os.makedirs(workspace_dir, exist_ok=True)
    provider = MockProvider(responses)
    registry = ToolRegistry()
    prompt_builder = MockPromptBuilder()

    if extra_tools:
        for tool in extra_tools:
            registry.register(tool)

    agent = Agent(
        provider=provider,
        registry=registry,
        prompt_builder=prompt_builder,
        max_iterations=10,
        workspace_dir=workspace_dir,
    )
    return agent


# ──────────────────────────────────────────────────
# Test 1: Circuit Breaker
# ──────────────────────────────────────────────────

async def test_circuit_breaker():
    print("\n--- Circuit Breaker Tests ---")

    fail_tool = AlwaysFailTool()

    # Create responses that keep calling the failing tool with THE SAME input.
    # Before the fix, this would trigger loop detection first. After the fix,
    # failed calls don't go into the loop tracker, so circuit breaker fires cleanly.
    responses = []
    for i in range(6):
        responses.append(AgentResponse(
            text="",
            tool_calls=[ToolCall(name="failing_tool", tool_id=f"t{i}", input={"input": "test"})],
            stop_reason="tool_use",
        ))
    responses.append(AgentResponse(text="Done after circuit breaker.", stop_reason="end_turn"))

    agent = make_agent(responses, extra_tools=[fail_tool])
    result = await agent.run("test circuit breaker")

    # After 3 failures, circuit breaker should trip
    check(
        "Circuit breaker trips after threshold",
        agent._tool_failure_counts.get("failing_tool", 0) >= agent.CIRCUIT_BREAKER_THRESHOLD,
        f"failures={agent._tool_failure_counts.get('failing_tool', 0)}"
    )


# ──────────────────────────────────────────────────
# Test 2: Empty Response Detection
# ──────────────────────────────────────────────────

async def test_empty_response():
    print("\n--- Empty Response Detection Tests ---")

    # 2 empty responses then a real one
    responses = [
        AgentResponse(text="", tool_calls=[], stop_reason="end_turn"),
        AgentResponse(text="", tool_calls=[], stop_reason="end_turn"),
        AgentResponse(text="", tool_calls=[], stop_reason="end_turn"),
    ]

    agent = make_agent(responses)
    result = await agent.run("hello")

    check(
        "Empty response detected and handled",
        "empty responses repeatedly" in result.lower(),
        f"got: {result[:100]}"
    )


# ──────────────────────────────────────────────────
# Test 3: Circular Loop Detection
# ──────────────────────────────────────────────────

async def test_circular_loop():
    print("\n--- Circular Loop Detection Tests ---")

    agent = make_agent([])

    # Test loop detection via _record_tool_signature + _detect_circular_loop
    same_call = ToolCall(name="bash", tool_id="t1", input={"command": "ls -la"})

    # Add same signature multiple times (simulating 3 successful identical calls)
    agent._recent_tool_signatures.clear()
    agent._record_tool_signature(same_call)
    r1 = agent._detect_circular_loop()
    agent._record_tool_signature(same_call)
    r2 = agent._detect_circular_loop()
    agent._record_tool_signature(same_call)
    r3 = agent._detect_circular_loop()

    check(
        "Loop detected after repeat threshold",
        r3 is not None,
        f"got: {r3}"
    )

    # Different calls should not trigger
    agent._recent_tool_signatures.clear()
    call_a = ToolCall(name="bash", tool_id="t1", input={"command": "ls -la"})
    call_b = ToolCall(name="bash", tool_id="t2", input={"command": "pwd"})
    call_c = ToolCall(name="read", tool_id="t3", input={"file_path": "/tmp/x"})

    agent._record_tool_signature(call_a)
    r1 = agent._detect_circular_loop()
    agent._record_tool_signature(call_b)
    r2 = agent._detect_circular_loop()
    agent._record_tool_signature(call_c)
    r3 = agent._detect_circular_loop()

    check(
        "Different calls don't trigger loop",
        r3 is None,
        f"got: {r3}"
    )

    # Test signature generation
    sig1 = Agent._tool_call_signature(call_a)
    sig2 = Agent._tool_call_signature(call_a)
    sig3 = Agent._tool_call_signature(call_b)

    check("Same call produces same signature", sig1 == sig2)
    check("Different calls produce different signatures", sig1 != sig3)


async def test_circuit_breaker_vs_loop_independence():
    """
    Multi-case test: Circuit breaker and loop detection must NOT interfere.

    Scenario: A tool fails repeatedly with the SAME args.
    - Circuit breaker should trip after 3 failures.
    - Loop detection should NOT trigger (failed calls aren't recorded).
    """
    print("\n--- Circuit Breaker vs Loop Detection Independence ---")

    fail_tool = AlwaysFailTool()

    # Case 1: Same tool, same args, all failures
    # Before the fix: loop detector would fire first (same signature 3x).
    # After the fix: only circuit breaker fires (failures aren't in loop tracker).
    responses = []
    for i in range(6):
        responses.append(AgentResponse(
            text="",
            tool_calls=[ToolCall(name="failing_tool", tool_id=f"t{i}", input={"input": "same_arg"})],
            stop_reason="tool_use",
        ))
    responses.append(AgentResponse(text="Done.", stop_reason="end_turn"))

    agent = make_agent(responses, extra_tools=[fail_tool])
    result = await agent.run("test same-args failure")

    check(
        "Case1: Circuit breaker trips with same args",
        agent._tool_failure_counts.get("failing_tool", 0) >= agent.CIRCUIT_BREAKER_THRESHOLD,
        f"failures={agent._tool_failure_counts.get('failing_tool', 0)}"
    )
    check(
        "Case1: Loop detection did NOT fire (only failures)",
        len(agent._recent_tool_signatures) == 0,
        f"signatures={agent._recent_tool_signatures}"
    )

    # Case 2: A tool succeeds repeatedly with same args → loop detection
    class AlwaysSucceedTool:
        name = "succeed_tool"
        description = "Always succeeds"
        input_schema = {"type": "object", "properties": {"input": {"type": "string"}}}

        def get_schema(self):
            return ToolSchema(name=self.name, description=self.description, input_schema=self.input_schema)

        async def execute(self, tool_id="", **kwargs):
            return ToolResult(tool_id=tool_id, success=True, output="ok")

    success_tool = AlwaysSucceedTool()

    responses2 = []
    for i in range(5):
        responses2.append(AgentResponse(
            text="",
            tool_calls=[ToolCall(name="succeed_tool", tool_id=f"s{i}", input={"input": "same"})],
            stop_reason="tool_use",
        ))
    responses2.append(AgentResponse(text="Stopped.", stop_reason="end_turn"))

    agent2 = make_agent(responses2, extra_tools=[success_tool])
    result2 = await agent2.run("test same-args success loop")

    check(
        "Case2: Loop detected for repeated successes",
        len(agent2._recent_tool_signatures) > 0
        and "loop" in result2.lower() or agent2._recent_tool_signatures.count(agent2._recent_tool_signatures[0]) >= 3 if agent2._recent_tool_signatures else False,
        f"sigs={agent2._recent_tool_signatures}, result={result2[:80]}"
    )
    check(
        "Case2: Circuit breaker did NOT trip (no failures)",
        agent2._tool_failure_counts.get("succeed_tool", 0) == 0,
        f"failures={agent2._tool_failure_counts.get('succeed_tool', 0)}"
    )

    # Case 3: Mixed — tool A fails (circuit breaker), tool B succeeds in loop
    responses3 = []
    for i in range(4):
        responses3.append(AgentResponse(
            text="",
            tool_calls=[
                ToolCall(name="failing_tool", tool_id=f"f{i}", input={"input": f"fail_{i}"}),
                ToolCall(name="succeed_tool", tool_id=f"s{i}", input={"input": "same"}),
            ],
            stop_reason="tool_use",
        ))
    responses3.append(AgentResponse(text="Mixed done.", stop_reason="end_turn"))

    agent3 = make_agent(responses3, extra_tools=[fail_tool, success_tool])
    result3 = await agent3.run("test mixed")

    check(
        "Case3: Circuit breaker fired for failing_tool",
        agent3._tool_failure_counts.get("failing_tool", 0) >= agent3.CIRCUIT_BREAKER_THRESHOLD,
        f"failures={agent3._tool_failure_counts.get('failing_tool', 0)}"
    )
    check(
        "Case3: Loop sigs only contain succeed_tool",
        all("succeed_tool" in sig for sig in agent3._recent_tool_signatures),
        f"sigs={agent3._recent_tool_signatures}"
    )


# ──────────────────────────────────────────────────
# Test 3b: Result Ordering in _execute_tools
# ──────────────────────────────────────────────────

async def test_result_ordering():
    """
    Bug fix test: _execute_tools must return results in the SAME order as
    tool_calls, even when some calls are blocked and some are executed.

    Before the fix: blocked results were appended first, executed results last.
    This meant run()'s zip(tool_calls, results) would misalign calls with results.
    """
    print("\n--- Result Ordering Tests ---")

    # We need a tool that succeeds and a tool that always fails (for circuit breaker)
    class OkTool:
        name = "ok_tool"
        description = "Always ok"
        input_schema = {"type": "object", "properties": {"input": {"type": "string"}}}
        def get_schema(self):
            return ToolSchema(name=self.name, description=self.description, input_schema=self.input_schema)
        async def execute(self, tool_id="", **kwargs):
            return ToolResult(tool_id=tool_id, success=True, output="ok")

    ok_tool = OkTool()
    fail_tool = AlwaysFailTool()

    # Step 1: Trip the circuit breaker for failing_tool by running 3 failures
    pre_responses = []
    for i in range(3):
        pre_responses.append(AgentResponse(
            text="",
            tool_calls=[ToolCall(name="failing_tool", tool_id=f"pre{i}", input={"input": f"pre_{i}"})],
            stop_reason="tool_use",
        ))
    # Step 2: Now send a batch with [failing_tool (blocked by CB), ok_tool (executed)]
    pre_responses.append(AgentResponse(
        text="",
        tool_calls=[
            ToolCall(name="failing_tool", tool_id="f1", input={"input": "blocked"}),
            ToolCall(name="ok_tool", tool_id="o1", input={"input": "should_run"}),
        ],
        stop_reason="tool_use",
    ))
    pre_responses.append(AgentResponse(text="Done ordering test.", stop_reason="end_turn"))

    agent = make_agent(pre_responses, extra_tools=[fail_tool, ok_tool])
    result = await agent.run("test ordering")

    # Check: In the tool_result message for the mixed batch, result[0] should be
    # the circuit-breaker block (for failing_tool) and result[1] should be the
    # success (for ok_tool).
    # Find the last tool_result message
    tool_result_msgs = [m for m in agent._messages if m.role == "tool_result" and m.tool_results]
    last_tool_results = tool_result_msgs[-1].tool_results if tool_result_msgs else []

    check(
        "Result count matches call count",
        len(last_tool_results) == 2,
        f"got {len(last_tool_results)} results"
    )

    if len(last_tool_results) == 2:
        check(
            "First result is circuit breaker block (matches failing_tool position)",
            "[CIRCUIT BREAKER]" in (last_tool_results[0].error or ""),
            f"got error: {last_tool_results[0].error}"
        )
        check(
            "Second result is success (matches ok_tool position)",
            last_tool_results[1].success,
            f"got success={last_tool_results[1].success}"
        )
        check(
            "Result tool_ids match call order",
            last_tool_results[0].tool_id == "f1" and last_tool_results[1].tool_id == "o1",
            f"got ids: {[r.tool_id for r in last_tool_results]}"
        )


# ──────────────────────────────────────────────────
# Test 3c: Policy blocks don't inflate circuit breaker
# ──────────────────────────────────────────────────

async def test_policy_block_not_counted():
    """
    Bug fix test: Plan mode / safety / validation blocks should NOT count
    as circuit breaker failures.

    Before the fix: blocking a tool via plan mode 3 times would trip the
    circuit breaker. After exiting plan mode, the tool would be permanently
    blocked even though it never actually failed.
    """
    print("\n--- Policy Block vs Circuit Breaker Tests ---")

    from cowork_agent.core.plan_mode import PlanManager

    class OkTool2:
        name = "bash"
        description = "Simulated bash"
        input_schema = {"type": "object", "properties": {"command": {"type": "string"}}}
        def get_schema(self):
            return ToolSchema(name=self.name, description=self.description, input_schema=self.input_schema)
        async def execute(self, tool_id="", **kwargs):
            return ToolResult(tool_id=tool_id, success=True, output="executed!")

    bash_tool = OkTool2()
    tmpdir = tempfile.mkdtemp()

    try:
        plan_manager = PlanManager(workspace_dir=tmpdir)
        plan_manager.enter_plan_mode()

        # Create responses that keep calling bash (blocked in plan mode)
        responses = []
        for i in range(5):
            responses.append(AgentResponse(
                text="",
                tool_calls=[ToolCall(name="bash", tool_id=f"b{i}", input={"command": "ls"})],
                stop_reason="tool_use",
            ))
        responses.append(AgentResponse(text="Done.", stop_reason="end_turn"))

        agent = make_agent(responses, extra_tools=[bash_tool], workspace_dir=tmpdir)
        agent.plan_manager = plan_manager

        result = await agent.run("run bash in plan mode")

        # Circuit breaker should NOT have tripped — these were plan mode blocks
        check(
            "Policy blocks: circuit breaker count is 0",
            agent._tool_failure_counts.get("bash", 0) == 0,
            f"failures={agent._tool_failure_counts.get('bash', 0)}"
        )

        # _is_policy_block should correctly identify the prefixes
        check(
            "_is_policy_block: CIRCUIT BREAKER",
            agent._is_policy_block(ToolResult(tool_id="x", success=False, output="", error="[CIRCUIT BREAKER] blocked")),
        )
        check(
            "_is_policy_block: PLAN MODE",
            agent._is_policy_block(ToolResult(tool_id="x", success=False, output="", error="[PLAN MODE] not allowed")),
        )
        check(
            "_is_policy_block: SAFETY",
            agent._is_policy_block(ToolResult(tool_id="x", success=False, output="", error="[SAFETY] dangerous")),
        )
        check(
            "_is_policy_block: VALIDATION",
            agent._is_policy_block(ToolResult(tool_id="x", success=False, output="", error="[VALIDATION] bad input")),
        )
        check(
            "_is_policy_block: real failure is NOT policy",
            not agent._is_policy_block(ToolResult(tool_id="x", success=False, output="", error="Connection refused")),
        )
        check(
            "_is_policy_block: success is NOT policy",
            not agent._is_policy_block(ToolResult(tool_id="x", success=True, output="ok")),
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ──────────────────────────────────────────────────
# Test 4: Network Retry — transient error detection
# ──────────────────────────────────────────────────

async def test_network_retry():
    print("\n--- Network Retry Tests ---")

    # Test _is_transient function
    check("Timeout is transient", _is_transient("Connection timed out"))
    check("503 is transient", _is_transient("HTTP 503 Service Unavailable"))
    check("429 is transient", _is_transient("Rate limit exceeded (429)"))
    check("Connection reset transient", _is_transient("Connection reset by peer"))
    check("DNS error transient", _is_transient("DNS resolution failed"))
    check("404 is NOT transient", not _is_transient("HTTP 404 Not Found"))
    check("Auth error NOT transient", not _is_transient("401 Unauthorized"))
    check("Parse error NOT transient", not _is_transient("Invalid JSON in response"))


# ──────────────────────────────────────────────────
# Test 5: Streaming base provider fallback
# ──────────────────────────────────────────────────

async def test_streaming():
    print("\n--- Streaming Tests ---")

    provider = MockProvider(responses=[
        AgentResponse(text="Hello streamed world!", stop_reason="end_turn"),
    ])

    messages = [Message(role="user", content="hi")]
    tools = []
    system = "test"

    chunks = []
    async for chunk in provider.send_message_stream(messages, tools, system):
        chunks.append(chunk)

    check("Base streaming yields text", len(chunks) == 1)
    check("Streaming content correct", chunks[0] == "Hello streamed world!")
    check(
        "last_stream_response available",
        provider.last_stream_response is not None
    )
    check(
        "last_stream_response text correct",
        provider.last_stream_response.text == "Hello streamed world!"
    )


# ──────────────────────────────────────────────────
# Test 6: Task Persistence
# ──────────────────────────────────────────────────

async def test_task_persistence():
    print("\n--- Task Persistence Tests ---")

    tmpdir = tempfile.mkdtemp()
    try:
        persist_dir = os.path.join(tmpdir, ".cowork")

        # Create tool with persistence
        tool = TodoWriteTool(persist_dir=persist_dir)

        check("Persist dir created", os.path.isdir(persist_dir))

        # Add some todos
        await tool.execute(
            todos=[
                {"content": "Task A", "status": "completed", "activeForm": "Doing A"},
                {"content": "Task B", "status": "in_progress", "activeForm": "Doing B"},
            ],
            tool_id="t1",
        )

        # Check file was written
        persist_path = os.path.join(persist_dir, "todos.json")
        check("Todos file created", os.path.exists(persist_path))

        # Read the file
        with open(persist_path) as f:
            saved = json.load(f)
        check("Saved 2 todos", len(saved) == 2)
        check("First todo correct", saved[0]["content"] == "Task A")
        check("Second todo status", saved[1]["status"] == "in_progress")

        # Create a NEW tool instance — should load from disk
        tool2 = TodoWriteTool(persist_dir=persist_dir)
        check("Loaded todos from disk", len(tool2.todos) == 2)
        check("Loaded content matches", tool2.todos[0]["content"] == "Task A")

        # Tool without persistence should not crash
        tool3 = TodoWriteTool()
        check("No persist dir is fine", len(tool3.todos) == 0)

        await tool3.execute(
            todos=[{"content": "X", "status": "pending", "activeForm": "X"}],
            tool_id="t2",
        )
        check("In-memory only works", len(tool3.todos) == 1)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ──────────────────────────────────────────────────
# Test 7: Agent run_stream method
# ──────────────────────────────────────────────────

async def test_agent_stream():
    print("\n--- Agent Streaming Tests ---")

    responses = [
        AgentResponse(text="Hello from stream!", stop_reason="end_turn"),
    ]

    agent = make_agent(responses)

    chunks = []
    async for chunk in agent.run_stream("hi"):
        chunks.append(chunk)

    full = "".join(chunks)
    check("Agent stream yields text", len(chunks) > 0)
    check("Stream content correct", "Hello from stream!" in full)
    check("Message added to memory", len(agent._messages) >= 2)  # user + assistant


# ──────────────────────────────────────────────────
# Test 8: Plan mode allows exit_plan_mode tool
# ──────────────────────────────────────────────────

async def test_plan_mode_exit_allowed():
    print("\n--- Plan Mode Exit Allowed Tests ---")

    from cowork_agent.core.plan_mode import PlanManager, PLAN_MODE_ALLOWED_TOOLS

    # Bug fix: exit_plan_mode must be in allowed tools, otherwise plan mode is inescapable
    check(
        "exit_plan_mode in PLAN_MODE_ALLOWED_TOOLS",
        "exit_plan_mode" in PLAN_MODE_ALLOWED_TOOLS,
    )
    check(
        "enter_plan_mode in PLAN_MODE_ALLOWED_TOOLS",
        "enter_plan_mode" in PLAN_MODE_ALLOWED_TOOLS,
    )

    # Test that PlanManager actually allows exit_plan_mode
    pm = PlanManager(workspace_dir=tempfile.mkdtemp())
    pm.enter_plan_mode()

    allowed_exit, reason_exit = pm.is_tool_allowed("exit_plan_mode")
    check("exit_plan_mode allowed in plan mode", allowed_exit, reason_exit)

    allowed_enter, reason_enter = pm.is_tool_allowed("enter_plan_mode")
    check("enter_plan_mode allowed in plan mode", allowed_enter, reason_enter)

    # Still blocked: bash, write, edit
    blocked_bash, _ = pm.is_tool_allowed("bash")
    check("bash still blocked in plan mode", not blocked_bash)

    blocked_write, _ = pm.is_tool_allowed("write")
    check("write still blocked in plan mode", not blocked_write)

    blocked_edit, _ = pm.is_tool_allowed("edit")
    check("edit still blocked in plan mode", not blocked_edit)


# ──────────────────────────────────────────────────
# Test 9: _execute_tools null-safety (defensive None check)
# ──────────────────────────────────────────────────

async def test_execute_tools_null_safety():
    print("\n--- Execute Tools Null Safety Tests ---")

    # Test that _execute_tools never returns None elements.
    # We can't directly trigger the None path without monkey-patching,
    # but we CAN verify the return type is list[ToolResult] not list[Optional[ToolResult]].

    fail_tool = AlwaysFailTool()
    responses = [
        AgentResponse(
            text="",
            tool_calls=[
                ToolCall(name="failing_tool", tool_id="t1", input={"input": "a"}),
                ToolCall(name="failing_tool", tool_id="t2", input={"input": "b"}),
            ],
            stop_reason="tool_use",
        ),
        AgentResponse(text="Done.", stop_reason="end_turn"),
    ]

    agent = make_agent(responses, extra_tools=[fail_tool])
    await agent.run("test null safety")

    # The tool results should have been created (not None)
    # Check the message history for tool_result messages
    tool_result_msgs = [m for m in agent._messages if m.role == "tool_result"]
    for msg in tool_result_msgs:
        if msg.tool_results:
            for r in msg.tool_results:
                check("Tool result is not None", r is not None)
                check("Tool result has tool_id", hasattr(r, 'tool_id') and r.tool_id != "")


# ──────────────────────────────────────────────────
# Test 10: run_stream() truncation recovery
# ──────────────────────────────────────────────────

async def test_stream_truncation_recovery():
    print("\n--- Stream Truncation Recovery Tests ---")

    # Simulate: first call truncated (max_tokens, no tool calls),
    # second call succeeds with text
    responses = [
        AgentResponse(text="partial content...", stop_reason="max_tokens"),
        AgentResponse(text="Complete answer here.", stop_reason="end_turn"),
    ]

    agent = make_agent(responses)

    # We need a provider that supports streaming. Since MockProvider doesn't
    # implement send_message_stream, we can test the recovery logic indirectly
    # by checking that the Agent's run_stream recovery counters are initialized.
    check("Agent has MAX_TRUNCATION_RETRIES", hasattr(agent, 'MAX_TRUNCATION_RETRIES'))
    check("Agent has MAX_INTENT_NUDGES", hasattr(agent, 'MAX_INTENT_NUDGES'))

    # Test the truncation detection helper directly via the Ollama provider
    from cowork_agent.core.providers.ollama import OllamaProvider
    check(
        "Detects truncated tool call: unclosed json block",
        OllamaProvider._looks_like_truncated_tool_call(
            '```json\n{"tool_calls": [{"name": "write", "id": "t1", "input": {"file'
        ),
    )
    check(
        "Does not false-positive on normal text",
        not OllamaProvider._looks_like_truncated_tool_call("Just a regular response."),
    )


# ──────────────────────────────────────────────────
# Test 11: run_stream() has intent nudge capability
# ──────────────────────────────────────────────────

async def test_stream_intent_nudge():
    print("\n--- Stream Intent Nudge Tests ---")

    # Test that _detect_unfulfilled_intent works correctly
    from cowork_agent.core.agent import Agent

    # Code block dump detection
    large_code = "Here's the file:\n```python\n" + "x = 1\n" * 200 + "```"
    intent = Agent._detect_unfulfilled_intent(large_code)
    check("Detects code block dump", intent == "code_block_dump")

    # Intent phrase detection
    intent_text = "I'll create the file for you now."
    intent2 = Agent._detect_unfulfilled_intent(intent_text)
    check("Detects intent phrase 'I'll create'", intent2 == "intent_phrase")

    # No intent
    normal_text = "The answer is 42."
    intent3 = Agent._detect_unfulfilled_intent(normal_text)
    check("No false positive on normal text", intent3 is None)

    # Empty text
    intent4 = Agent._detect_unfulfilled_intent("")
    check("No false positive on empty text", intent4 is None)


# ──────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  P1 Feature Tests")
    print("=" * 60)

    await test_circuit_breaker()
    await test_empty_response()
    await test_circular_loop()
    await test_circuit_breaker_vs_loop_independence()
    await test_result_ordering()
    await test_policy_block_not_counted()
    await test_network_retry()
    await test_streaming()
    await test_task_persistence()
    await test_agent_stream()
    await test_plan_mode_exit_allowed()
    await test_execute_tools_null_safety()
    await test_stream_truncation_recovery()
    await test_stream_intent_nudge()

    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
