"""
WebFetch Tool — Wrapper around claude_web_tools.WebFetch.
Adapts FetchResult to the agent framework's ToolResult format.
"""

from __future__ import annotations

from .base import BaseTool


class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = (
        "Fetch a URL and process it with an LLM. "
        "Fetches the page content, converts HTML to markdown, "
        "then processes the content with a prompt using the configured LLM. "
        "Includes a 15-minute cache for repeated URL access."
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
                from claude_web_tools import WebFetch
                self._fetcher = WebFetch()
            except ImportError:
                raise ImportError(
                    "claude_web_tools package not found. "
                    "Install it or ensure it's on the Python path."
                )
        return self._fetcher

    async def execute(
        self,
        url: str,
        prompt: str,
        tool_id: str = "",
        **kwargs,
    ) -> "ToolResult":
        try:
            fetcher = self._get_fetcher()
        except ImportError as e:
            return self._error(str(e), tool_id)

        try:
            result = await fetcher.fetch(url=url, prompt=prompt)

            if result.error:
                return self._error(
                    f"Fetch error: {result.error}", tool_id
                )

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
            return self._error(f"WebFetch error: {str(e)}", tool_id)
