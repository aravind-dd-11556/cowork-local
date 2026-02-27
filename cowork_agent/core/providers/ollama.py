"""
Ollama LLM Provider.

Since Ollama doesn't have universal native tool_use support, we embed
tool schemas in the system prompt and parse JSON tool_calls from the response.
Supports both streaming and non-streaming modes.
"""

from __future__ import annotations
import json
import logging
import re
import httpx
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

from .base import BaseLLMProvider, ProviderFactory
from ..models import AgentResponse, Message, ToolCall, ToolSchema


class OllamaProvider(BaseLLMProvider):
    """Ollama provider with JSON-based tool calling."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model=model, base_url=base_url, **kwargs)

    async def send_message(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system_prompt: str,
    ) -> AgentResponse:
        """Send message to Ollama with tool schemas embedded in system prompt."""

        # Build the enhanced system prompt with tool instructions
        enhanced_system = self._build_system_with_tools(system_prompt, tools)

        # Convert messages to Ollama format
        ollama_messages = self._convert_messages(messages)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": enhanced_system},
                            *ollama_messages,
                        ],
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()

            content = data.get("message", {}).get("content", "")
            logger.debug(f"Ollama raw response length: {len(content)} chars")
            logger.debug(f"Ollama response preview: {content[:300]}...")
            if '"tool_calls"' in content:
                logger.debug("Response contains 'tool_calls' keyword")
            done_reason = data.get("done_reason", "unknown")
            logger.debug(f"Ollama done_reason: {done_reason}")

            response = self._parse_response(content)

            # If Ollama reports it stopped due to length, override stop_reason
            if done_reason == "length" and not response.tool_calls:
                logger.warning("Ollama stopped due to token length limit")
                response.stop_reason = "max_tokens"

            return response

        except httpx.ConnectError:
            return AgentResponse(
                text=f"Cannot connect to Ollama at {self.base_url}. "
                     "Make sure Ollama is running (ollama serve).",
                stop_reason="error",
            )
        except Exception as e:
            return AgentResponse(text=f"Ollama error: {str(e)}", stop_reason="error")

    async def send_message_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens from Ollama.

        Yields text chunks as they arrive. After the stream ends, parses
        the complete response for tool calls. The full AgentResponse is
        available via `last_stream_response`.
        """
        enhanced_system = self._build_system_with_tools(system_prompt, tools)
        ollama_messages = self._convert_messages(messages)

        full_content = ""

        # Buffer a few chunks before yielding so we can detect tool_calls
        # JSON blocks early, before they reach the terminal.  Without this,
        # the opening `{"tool_calls` characters leak to stdout as garbled text.
        _BUFFER_WATERMARK = 40  # chars to buffer before flushing

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": enhanced_system},
                            *ollama_messages,
                        ],
                        "stream": True,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                        },
                    },
                ) as stream:
                    done_reason = "unknown"
                    pending_buffer = ""   # text waiting to be yielded
                    tool_json_started = False  # once True, stop yielding

                    async for line in stream.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            full_content += chunk

                            if tool_json_started:
                                # Already inside a tool_calls block — don't yield
                                pass
                            else:
                                pending_buffer += chunk

                                # Check if we've entered a tool_calls JSON block
                                if '"tool_calls"' in full_content or '```json' in pending_buffer:
                                    # Check more carefully — does this look like a tool call?
                                    if '"tool_calls"' in full_content:
                                        tool_json_started = True
                                        # Don't yield anything in the pending buffer
                                        # that is part of the JSON block.  Find the
                                        # start of the JSON block and yield only the
                                        # text before it.
                                        json_markers = ['```json', '{"tool_calls', "{'tool_calls"]
                                        earliest = len(full_content)
                                        for marker in json_markers:
                                            idx = full_content.find(marker)
                                            if idx >= 0 and idx < earliest:
                                                earliest = idx
                                        # Calculate how much of the buffer is safe text
                                        safe_chars = earliest - (len(full_content) - len(pending_buffer))
                                        if safe_chars > 0:
                                            yield pending_buffer[:safe_chars]
                                        pending_buffer = ""
                                        continue

                                # Flush safe text when buffer is large enough
                                if len(pending_buffer) >= _BUFFER_WATERMARK:
                                    # Keep a small tail in case a JSON block starts mid-buffer
                                    safe = pending_buffer[:-20]
                                    pending_buffer = pending_buffer[-20:]
                                    if safe:
                                        yield safe

                        if data.get("done", False):
                            done_reason = data.get("done_reason", "stop")
                            break

                    # Flush remaining buffered text (only if no tool_calls detected)
                    if pending_buffer and not tool_json_started:
                        yield pending_buffer

            # Parse the full response for tool calls
            response = self._parse_response(full_content)

            # If Ollama reports length limit
            if done_reason == "length" and not response.tool_calls:
                response.stop_reason = "max_tokens"

            self._last_stream_response = response

        except httpx.ConnectError:
            self._last_stream_response = AgentResponse(
                text=f"Cannot connect to Ollama at {self.base_url}. "
                     "Make sure Ollama is running (ollama serve).",
                stop_reason="error",
            )
            yield self._last_stream_response.text

        except Exception as e:
            self._last_stream_response = AgentResponse(
                text=f"Ollama streaming error: {str(e)}",
                stop_reason="error",
            )
            yield self._last_stream_response.text

    def _build_system_with_tools(self, base_prompt: str, tools: list[ToolSchema]) -> str:
        """Embed tool schemas and calling instructions into system prompt."""
        if not tools:
            return base_prompt

        tool_defs = json.dumps([t.to_dict() for t in tools], indent=2)

        tool_instruction = f"""
{base_prompt}

## Available Tools

You have access to the following tools:

{tool_defs}

## How to Call Tools

When you need to use a tool, include a JSON block in your response like this:

```json
{{"tool_calls": [{{"name": "tool_name", "id": "tool_001", "input": {{"param": "value"}}}}]}}
```

Rules:
- You can call multiple tools at once by including multiple items in the tool_calls array
- Each tool call must have a unique "id" (e.g., tool_001, tool_002)
- If you don't need any tools, just respond with normal text (no JSON block)
- After tool results come back, analyze them and decide your next action
- When you're done (no more tools needed), give your final text response WITHOUT any tool_calls JSON

## CRITICAL: File Creation Rule

NEVER show file content as a code block in your response. That does NOT create a file.
To create a file, you MUST use the write tool via tool_calls JSON. For example:

To create an HTML file, do this:
```json
{{"tool_calls": [{{"name": "write", "id": "tool_001", "input": {{"file_path": "/path/to/file.html", "content": "<!DOCTYPE html>..."}}}}]}}
```

Do NOT do this:
```
Here is the file:
```html
<!DOCTYPE html>...
```
```

The above just shows text — it does NOT create any file. You MUST use the write tool.
Same applies for .md, .py, .json, .css, .js, or any other file type.
If the user asks to "create", "save", "write", or "draft" a file — USE the write tool.
"""
        return tool_instruction

    # Max characters for individual message content sent to Ollama.
    # Prevents the context from exploding with large tool results or file contents.
    MAX_MSG_CHARS = 3000

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert internal messages to Ollama format, truncating large content."""
        ollama_msgs = []
        for msg in messages:
            if msg.role == "tool_result" and msg.tool_results:
                # Format tool results as a user message (truncate large outputs)
                results_text = "\n\n".join(
                    f"[Tool Result: {r.tool_id}]\n"
                    f"{'Success' if r.success else 'Error'}: "
                    f"{self._truncate(r.output or r.error or '', self.MAX_MSG_CHARS)}"
                    for r in msg.tool_results
                )
                ollama_msgs.append({"role": "user", "content": results_text})
            elif msg.role == "assistant":
                # Truncate assistant messages that contain large tool call inputs
                # (e.g., the full blog HTML inside a write call)
                content = self._truncate(msg.content, self.MAX_MSG_CHARS * 2)
                ollama_msgs.append({"role": "assistant", "content": content})
            else:
                ollama_msgs.append({"role": "user", "content": msg.content})
        return ollama_msgs

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Truncate text to max_chars, adding a notice if trimmed."""
        if not text or len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n\n[... truncated, {len(text) - max_chars} chars omitted ...]"

    def _parse_response(self, content: str) -> AgentResponse:
        """Parse LLM response for tool_calls JSON or plain text."""
        # Try to find a JSON block with tool_calls
        # Look for ```json blocks first, then raw JSON
        patterns = [
            r'```json\s*(\{[\s\S]*?"tool_calls"[\s\S]*?\})\s*```',
            r'(\{[\s\S]*?"tool_calls"\s*:\s*\[[\s\S]*?\]\s*\})',
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    raw_json = match.group(1)
                    parsed = self._safe_json_parse(raw_json)
                    if parsed is None:
                        continue

                    tool_calls = [
                        ToolCall(
                            name=tc["name"],
                            tool_id=tc.get("id", ToolCall.generate_id()),
                            input=tc.get("input", {}),
                        )
                        for tc in parsed.get("tool_calls", [])
                    ]
                    if tool_calls:
                        # Extract any text outside the JSON block
                        text = content.replace(match.group(0), "").strip()
                        return AgentResponse(
                            text=text or None,
                            tool_calls=tool_calls,
                            stop_reason="tool_use",
                        )
                except (KeyError, TypeError):
                    continue

        # ── Truncation detection ──
        # If the response contains a partial tool_calls JSON that didn't match
        # (likely truncated due to token limit), flag it so the agent can retry
        if self._looks_like_truncated_tool_call(content):
            logger.warning("Detected truncated tool_calls JSON — likely hit token limit")
            return AgentResponse(
                text=content,
                stop_reason="max_tokens",  # Signal truncation
            )

        # No tool calls found — return as plain text
        return AgentResponse(text=content, stop_reason="end_turn")

    @staticmethod
    def _looks_like_truncated_tool_call(content: str) -> bool:
        """
        Detect if the response was truncated mid-tool-call JSON.

        Signs of truncation:
        - Contains "tool_calls" keyword but no closing pattern
        - Contains ```json with tool_calls but no closing ```
        - Ends abruptly with unclosed braces/brackets
        """
        if '"tool_calls"' not in content and "'tool_calls'" not in content:
            return False

        # Check for unclosed ```json block
        json_block_opens = content.count('```json')
        json_block_closes = content.count('```') - json_block_opens
        if json_block_opens > 0 and json_block_closes < json_block_opens:
            return True

        # Check for "tool_calls" followed by unclosed brackets/braces
        tc_idx = content.rfind('"tool_calls"')
        if tc_idx >= 0:
            after_tc = content[tc_idx:]
            opens = after_tc.count('{') + after_tc.count('[')
            closes = after_tc.count('}') + after_tc.count(']')
            if opens > closes:
                return True

        return False

    @staticmethod
    def _safe_json_parse(raw: str) -> dict | None:
        """
        Parse JSON that may contain unescaped control characters.

        Local LLMs (Ollama) often output literal newlines/tabs inside
        JSON string values instead of escaped \\n / \\t. This sanitizer
        fixes those before parsing.
        """
        # Try direct parse first (fast path)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Fix: escape literal control characters inside string values
        # Walk through the string and escape unescaped control chars
        # that appear between quotes
        try:
            sanitized = _escape_json_strings(raw)
            return json.loads(sanitized)
        except (json.JSONDecodeError, Exception):
            pass

        # Last resort: try replacing common problematic patterns
        try:
            # Replace literal newlines/tabs inside the JSON
            fixed = raw.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        return None

    async def health_check(self) -> dict:
        """Check if Ollama is running and model is available."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                models = [m.get("name", "").split(":")[0] for m in resp.json().get("models", [])]
                return {
                    "status": "ok",
                    "provider": "ollama",
                    "model": self.model,
                    "model_available": self.model.split(":")[0] in models,
                    "available_models": models,
                }
        except Exception as e:
            return {"status": "error", "provider": "ollama", "error": str(e)}


def _escape_json_strings(raw: str) -> str:
    """
    Walk a JSON string and escape unescaped control characters
    that appear inside quoted string values.
    """
    result = []
    in_string = False
    i = 0
    while i < len(raw):
        ch = raw[i]

        if ch == '\\' and in_string:
            # Escaped character — pass through as-is
            result.append(ch)
            if i + 1 < len(raw):
                result.append(raw[i + 1])
                i += 2
            else:
                i += 1
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue

        if in_string and ch in ('\n', '\r', '\t'):
            # Unescaped control character inside a string — escape it
            escape_map = {'\n': '\\n', '\r': '\\r', '\t': '\\t'}
            result.append(escape_map[ch])
            i += 1
            continue

        result.append(ch)
        i += 1

    return ''.join(result)


# Register with factory
ProviderFactory.register("ollama", OllamaProvider)
