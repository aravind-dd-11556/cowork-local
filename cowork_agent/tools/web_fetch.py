"""
WebFetch Tool — Wrapper around claude_web_tools.WebFetch.
Adapts FetchResult to the agent framework's ToolResult format.
Includes retry with exponential backoff for transient network errors.
"""

from __future__ import annotations
import asyncio
import logging

from .base import BaseTool

logger = logging.getLogger(__name__)

# Retry settings
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
BACKOFF_FACTOR = 2.0

# Transient error indicators (substring matching)
TRANSIENT_ERRORS = [
    "timeout", "timed out", "connection reset", "connection refused",
    "temporary failure", "503", "502", "429", "rate limit",
    "server error", "network", "dns", "eof",
]


def _is_transient(error_str: str) -> bool:
    """Check if an error message indicates a transient/retryable failure."""
    lower = error_str.lower()
    return any(indicator in lower for indicator in TRANSIENT_ERRORS)


class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = (
        "Fetch a URL and process it with an LLM. "
        "Fetches the page content, converts HTML to markdown, "
        "then processes the content with a prompt using the configured LLM. "
        "Includes a 15-minute cache for repeated URL access. "
        "Automatically retries on transient network errors."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch (HTTP auto-upgraded to HTTPS)",
            },
            "prompt": {
                "type": "string",
                "description": "What information to extract from the page",
            },
        },
        "required": ["url", "prompt"],
    }

    def __init__(self):
        self._fetcher = None

    def _get_fetcher(self):
        """Lazy import to avoid hard dependency at tool registration."""
        if self._fetcher is None:
            try:
                from cowork_agent.vendor.claude_web_tools import WebFetch
                self._fetcher = WebFetch()
            except ImportError:
                raise ImportError(
                    "claude_web_tools package not found. "
                    "Install it or ensure it's on the Python path."
                )
        return self._fetcher

    # ── SSRF Protection ──
    # Block requests to private/internal network addresses
    BLOCKED_HOSTS = {
        "localhost", "127.0.0.1", "0.0.0.0", "::1",
        "metadata.google.internal",  # GCP metadata
        "169.254.169.254",  # AWS/GCP/Azure metadata endpoint
    }
    BLOCKED_IP_PREFIXES = (
        "10.", "172.16.", "172.17.", "172.18.", "172.19.",
        "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
        "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
        "172.30.", "172.31.", "192.168.", "169.254.",
        "fc00:", "fd00:", "fe80:",  # IPv6 private
    )

    @staticmethod
    def _extract_host(url: str) -> str:
        """Extract hostname from URL."""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            return parsed.hostname or ""
        except Exception:
            return ""

    def _is_ssrf_target(self, url: str) -> bool:
        """Check if URL targets a private/internal network address."""
        host = self._extract_host(url)
        if not host:
            return True  # Block unparseable URLs

        host_lower = host.lower()

        # Check blocked hostnames
        if host_lower in self.BLOCKED_HOSTS:
            return True

        # Check private IP ranges
        if any(host_lower.startswith(prefix) for prefix in self.BLOCKED_IP_PREFIXES):
            return True

        # Block file:// and other non-http schemes
        if not url.lower().startswith(("http://", "https://")):
            return True

        return False

    async def execute(
        self,
        url: str,
        prompt: str,
        tool_id: str = "",
        **kwargs,
    ) -> "ToolResult":
        # SSRF check: block private/internal network requests
        if self._is_ssrf_target(url):
            return self._error(
                f"Blocked: URL targets a private/internal network address. "
                f"Only public HTTP(S) URLs are allowed.",
                tool_id,
            )

        try:
            fetcher = self._get_fetcher()
        except ImportError as e:
            return self._error(str(e), tool_id)

        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = await fetcher.fetch(url=url, prompt=prompt)

                if result.error:
                    # Check if this is a transient error worth retrying
                    if _is_transient(result.error) and attempt < MAX_RETRIES:
                        delay = BASE_DELAY * (BACKOFF_FACTOR ** (attempt - 1))
                        logger.info(
                            f"WebFetch transient error (attempt {attempt}/{MAX_RETRIES}): "
                            f"{result.error}. Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        last_error = result.error
                        continue
                    return self._error(f"Fetch error: {result.error}", tool_id)

                # Handle redirect
                if result.redirect_url:
                    return self._success(
                        f"The URL redirected to a different host: {result.redirect_url}\n"
                        "Please make a new web_fetch request with the redirect URL.",
                        tool_id,
                    )

                # Use the FetchResult's built-in formatter
                output = result.to_markdown()
                return self._success(output, tool_id)

            except Exception as e:
                last_error = str(e)
                if _is_transient(last_error) and attempt < MAX_RETRIES:
                    delay = BASE_DELAY * (BACKOFF_FACTOR ** (attempt - 1))
                    logger.info(
                        f"WebFetch exception (attempt {attempt}/{MAX_RETRIES}): "
                        f"{last_error}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                return self._error(f"WebFetch error: {last_error}", tool_id)

        return self._error(
            f"WebFetch failed after {MAX_RETRIES} attempts. Last error: {last_error}",
            tool_id,
        )
