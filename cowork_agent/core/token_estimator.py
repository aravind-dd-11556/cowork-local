"""
Token Estimator — Model-aware token estimation.

Replaces the crude CHARS_PER_TOKEN=4 constant with configurable per-model
char-per-token ratios, providing more accurate token counts for context
management and budget enforcement.

Sprint 15 (Prompt Optimization & Context Management) Module 1.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Message

logger = logging.getLogger(__name__)


# Default char-per-token ratios by model family.
# Lower ratio = more tokens per character = more expensive context.
DEFAULT_MODEL_RATIOS: dict[str, float] = {
    "claude": 3.5,
    "gpt-4": 4.0,
    "gpt-3.5": 4.2,
    "ollama": 4.0,
}

FALLBACK_RATIO = 4.0


class ModelTokenEstimator:
    """
    Model-aware token estimation with configurable per-model char/token ratios.

    Usage:
        estimator = ModelTokenEstimator()
        tokens = estimator.estimate_tokens("Hello world", "claude-sonnet-4-5-20250929")
        msg_tokens = estimator.estimate_message_tokens(message, "gpt-4o")
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Optional dict of model_name → ratio overrides.
                    e.g. {"claude": 3.5, "gpt-4": 4.0, "my-custom-model": 3.8}
        """
        self._ratios = dict(DEFAULT_MODEL_RATIOS)
        if config:
            for key, val in config.items():
                try:
                    self._ratios[key] = float(val)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid ratio for {key}: {val}, skipping")

        # Cache resolved model → ratio to avoid repeated lookups
        self._model_cache: dict[str, float] = {}

    def get_ratio(self, model: str) -> float:
        """
        Get the chars-per-token ratio for a model.

        Resolution order:
          1. Exact match in ratios dict
          2. Prefix match (e.g. "gpt-4o-2024-..." → "gpt-4")
          3. Provider family match (e.g. "claude-..." → "claude")
          4. Fallback to FALLBACK_RATIO (4.0)
        """
        if model in self._model_cache:
            return self._model_cache[model]

        ratio = self._resolve_ratio(model)
        self._model_cache[model] = ratio
        return ratio

    def _resolve_ratio(self, model: str) -> float:
        """Resolve ratio using progressive matching."""
        lower = model.lower()

        # 1. Exact match
        if lower in self._ratios:
            return self._ratios[lower]

        # 2. Check if any configured key is a prefix of the model name
        for key in sorted(self._ratios.keys(), key=len, reverse=True):
            if lower.startswith(key):
                return self._ratios[key]

        # 3. Check if model name contains a known family name
        for key in self._ratios:
            if key in lower:
                return self._ratios[key]

        return FALLBACK_RATIO

    def estimate_tokens(self, text: str, model: str = "") -> int:
        """
        Estimate token count for a text string.

        Args:
            text: The text to estimate tokens for.
            model: Model name (used to select ratio). Empty = fallback ratio.

        Returns:
            Estimated token count (minimum 1 for non-empty text).
        """
        if not text:
            return 0
        ratio = self.get_ratio(model) if model else FALLBACK_RATIO
        return max(1, int(len(text) / ratio))

    def estimate_message_tokens(self, msg: Message, model: str = "") -> int:
        """
        Estimate tokens for a Message including tool calls and results.

        Accounts for:
          - Message content
          - Tool call names and JSON input
          - Tool result output and errors
          - Per-message overhead (~4 tokens for role/formatting)
        """
        tokens = self.estimate_tokens(msg.content, model)

        if msg.tool_calls:
            for call in msg.tool_calls:
                tokens += self.estimate_tokens(call.name, model)
                tokens += self.estimate_tokens(str(call.input), model)

        if msg.tool_results:
            for result in msg.tool_results:
                tokens += self.estimate_tokens(result.output, model)
                if result.error:
                    tokens += self.estimate_tokens(result.error, model)

        # Message overhead (role tag, formatting)
        tokens += 4

        return tokens

    def estimate_messages_tokens(
        self,
        messages: list[Message],
        model: str = "",
        system_prompt: str = "",
    ) -> int:
        """
        Estimate total tokens for a list of messages plus system prompt.

        Args:
            messages: Conversation history.
            model: Model name for ratio selection.
            system_prompt: System prompt text.

        Returns:
            Total estimated token count.
        """
        total = self.estimate_tokens(system_prompt, model)
        for msg in messages:
            total += self.estimate_message_tokens(msg, model)
        return total

    def to_dict(self) -> dict:
        """Return configuration as a serializable dict."""
        return {
            "ratios": dict(self._ratios),
            "fallback_ratio": FALLBACK_RATIO,
            "cached_models": list(self._model_cache.keys()),
        }
