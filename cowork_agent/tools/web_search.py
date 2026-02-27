"""
WebSearch Tool — Wrapper around claude_web_tools.WebSearch.
Adapts SearchResponse to the agent framework's ToolResult format.
Includes retry with exponential backoff for transient network errors.
"""

from __future__ import annotations
import asyncio
import logging

from .base import BaseTool
from .web_fetch import MAX_RETRIES, BASE_DELAY, BACKOFF_FACTOR, _is_transient

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web using SearXNG and return structured results. "
        "Supports domain filtering (allow/block lists). "
        "Returns titles, URLs, and snippets for matching results. "
        "Automatically retries on transient network errors."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only include results from these domains",
            },
            "blocked_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exclude results from these domains",
            },
            "max_results": {
                "type": "number",
                "description": "Maximum number of results (default: 10)",
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        self._searcher = None

    def _get_searcher(self):
        """Lazy import to avoid hard dependency at tool registration."""
        if self._searcher is None:
            try:
                from cowork_agent.vendor.claude_web_tools import WebSearch
                self._searcher = WebSearch()
            except ImportError:
                raise ImportError(
                    "claude_web_tools package not found. "
                    "Install it or ensure it's on the Python path."
                )
        return self._searcher

    async def execute(
        self,
        query: str,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        max_results: int = 10,
        tool_id: str = "",
        **kwargs,
    ) -> "ToolResult":
        try:
            searcher = self._get_searcher()
        except ImportError as e:
            return self._error(str(e), tool_id)

        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await searcher.search(
                    query=query,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                    max_results=max_results,
                )

                if not response.results:
                    return self._success(
                        f"No results found for: {query}", tool_id
                    )

                # Format results as markdown
                output = response.to_markdown()
                return self._success(output, tool_id)

            except Exception as e:
                last_error = str(e)
                if _is_transient(last_error) and attempt < MAX_RETRIES:
                    delay = BASE_DELAY * (BACKOFF_FACTOR ** (attempt - 1))
                    logger.info(
                        f"WebSearch exception (attempt {attempt}/{MAX_RETRIES}): "
                        f"{last_error}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                return self._error(f"Search error: {last_error}", tool_id)

        return self._error(
            f"WebSearch failed after {MAX_RETRIES} attempts. Last error: {last_error}",
            tool_id,
        )
