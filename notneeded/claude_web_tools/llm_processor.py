"""
Stage 2 of WebFetch: LLM content processing via Ollama.

This is the key differentiator of WebFetch vs a simple HTTP fetch.
The raw markdown from Stage 1 is passed to a small, fast local model
along with the user's prompt. The model processes/summarizes the content
and returns a focused response.

This mirrors Claude's architecture where WebFetch uses a "small, fast model"
to process fetched content before returning it to the main Claude context.
"""

import httpx
import json
from typing import Optional

from .config import Config


class LLMProcessor:
    """
    Process fetched web content using a local Ollama model.

    Architecture (matching Claude's WebFetch Stage 2):
    - Takes raw markdown content + user prompt
    - Sends to a small, fast model (Ollama local)
    - Returns the model's processed/summarized response
    - Handles content that's too large by chunking
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = (base_url or Config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or Config.OLLAMA_MODEL
        self.timeout = Config.OLLAMA_TIMEOUT

    async def process(
        self,
        content: str,
        prompt: str,
        source_url: str,
    ) -> str:
        """
        Process web content with the LLM.

        Args:
            content: The markdown content from Stage 1
            prompt: The user's prompt describing what to extract
            source_url: The original URL (for context)

        Returns:
            The LLM's processed response string
        """
        # Build the system message (instructs the model on its role)
        system_message = (
            "You are a web content processor. You have been given the markdown "
            "content of a web page. Your job is to process this content according "
            "to the user's prompt and return a focused, relevant response. "
            "Be concise and extract only the information the user asked for. "
            "Do not reproduce large chunks of the original content — summarize "
            "and paraphrase instead. If the content doesn't contain what the "
            "user is looking for, say so clearly."
        )

        # Build the user message combining content + prompt
        user_message = (
            f"## Web Page Content (from {source_url}):\n\n"
            f"{content}\n\n"
            f"---\n\n"
            f"## Your Task:\n{prompt}"
        )

        # If content is very large, truncate to fit model context
        max_content_chars = 32000  # Conservative limit for most models
        if len(user_message) > max_content_chars:
            truncated_content = content[:max_content_chars - len(prompt) - 500]
            user_message = (
                f"## Web Page Content (from {source_url}) [truncated]:\n\n"
                f"{truncated_content}\n\n"
                f"[Content was truncated due to length]\n\n"
                f"---\n\n"
                f"## Your Task:\n{prompt}"
            )

        try:
            return await self._call_ollama(system_message, user_message)
        except httpx.ConnectError:
            return (
                f"Error: Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)."
            )
        except Exception as e:
            return f"Error processing content: {str(e)}"

    async def _call_ollama(self, system: str, user: str) -> str:
        """Make the actual API call to Ollama."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,  # Low temp for factual extraction
                "num_predict": 2048,  # Reasonable response length
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "No response from model")

    async def check_health(self) -> dict:
        """Check if Ollama is running and the model is available."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Check Ollama is running
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                model_base_names = [n.split(":")[0] for n in model_names]
                configured_base = self.model.split(":")[0]

                return {
                    "ollama_running": True,
                    "model_available": (
                        self.model in model_names or
                        configured_base in model_base_names
                    ),
                    "available_models": model_names,
                    "configured_model": self.model,
                }
        except httpx.ConnectError:
            return {
                "ollama_running": False,
                "model_available": False,
                "error": f"Cannot connect to Ollama at {self.base_url}",
            }
        except Exception as e:
            return {
                "ollama_running": False,
                "model_available": False,
                "error": str(e),
            }
