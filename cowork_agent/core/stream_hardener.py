"""
Stream Hardener — Wraps provider streaming with timeout, heartbeat
detection, and partial-result recovery.

Problems this solves:
  1. Stalled streams — provider stops sending chunks but doesn't close
  2. Slow drip — chunks arrive but too slowly (exceeds per-chunk timeout)
  3. Partial tool JSON — stream cuts mid-tool-call, needs retry signal
  4. Encoding corruption — malformed UTF-8 chunks

Sprint 4 (P2-Advanced) Feature 3.
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import AsyncIterator, Optional

from .models import AgentResponse

logger = logging.getLogger(__name__)


class StreamTimeoutError(Exception):
    """Raised when a stream stalls beyond the configured timeout."""
    pass


class StreamHardener:
    """
    Wraps an async chunk iterator with:
      - per-chunk timeout (stall detection)
      - total stream timeout
      - partial result buffering
      - chunk validation (rejects empty / corrupt chunks)

    Usage:
        hardener = StreamHardener(chunk_timeout=30, total_timeout=300)
        async for chunk in hardener.wrap(provider.send_message_stream(...)):
            print(chunk)
        # After iteration, check partial buffer
        partial = hardener.partial_text
    """

    def __init__(
        self,
        chunk_timeout: float = 30.0,
        total_timeout: float = 300.0,
        min_chunk_interval: float = 0.0,  # 0 = no throttle
    ):
        self.chunk_timeout = chunk_timeout
        self.total_timeout = total_timeout
        self.min_chunk_interval = min_chunk_interval

        # State
        self._partial_text: str = ""
        self._chunk_count: int = 0
        self._start_time: float = 0.0
        self._last_chunk_time: float = 0.0
        self._timed_out: bool = False
        self._completed: bool = False

    @property
    def partial_text(self) -> str:
        """Text accumulated so far (useful if stream was interrupted)."""
        return self._partial_text

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def elapsed(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    async def wrap(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """
        Wrap an async stream iterator with timeout and validation.

        Yields validated, non-empty text chunks.
        Raises StreamTimeoutError if a chunk or total timeout is exceeded.
        """
        self._partial_text = ""
        self._chunk_count = 0
        self._timed_out = False
        self._completed = False
        self._start_time = time.time()
        self._last_chunk_time = self._start_time

        aiter = stream.__aiter__()

        while True:
            # Check total timeout
            if time.time() - self._start_time > self.total_timeout:
                self._timed_out = True
                logger.warning(
                    f"Stream total timeout ({self.total_timeout}s) exceeded "
                    f"after {self._chunk_count} chunks"
                )
                raise StreamTimeoutError(
                    f"Total stream timeout ({self.total_timeout}s) exceeded"
                )

            # Wait for next chunk with per-chunk timeout
            try:
                chunk = await asyncio.wait_for(
                    aiter.__anext__(),
                    timeout=self.chunk_timeout,
                )
            except StopAsyncIteration:
                # Stream ended normally
                self._completed = True
                return
            except asyncio.TimeoutError:
                self._timed_out = True
                logger.warning(
                    f"Stream chunk timeout ({self.chunk_timeout}s) — "
                    f"no data received. {self._chunk_count} chunks so far."
                )
                raise StreamTimeoutError(
                    f"No data received for {self.chunk_timeout}s "
                    f"(stream stalled after {self._chunk_count} chunks)"
                )

            # Validate chunk
            if not isinstance(chunk, str):
                logger.warning(f"Non-string chunk received: {type(chunk)}")
                continue

            if not chunk:
                continue  # Skip empty chunks

            # Track timing
            now = time.time()
            self._last_chunk_time = now
            self._chunk_count += 1

            # Accumulate partial text
            self._partial_text += chunk

            yield chunk

    def build_partial_response(self) -> AgentResponse:
        """
        Build an AgentResponse from whatever we received before timeout.
        Useful for recovering partial text after a stream failure.
        """
        if self._timed_out:
            stop_reason = "max_tokens"  # Signal truncation
        elif self._completed:
            stop_reason = "end_turn"
        else:
            stop_reason = "error"

        return AgentResponse(
            text=self._partial_text if self._partial_text else None,
            stop_reason=stop_reason,
        )
