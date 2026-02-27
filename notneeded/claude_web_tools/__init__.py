"""
Claude-style WebSearch and WebFetch tools.

A Python implementation of the two web tools used in Anthropic's Cowork mode,
built with a fully free stack: SearXNG (search) + Ollama (local LLM).

Usage:
    from claude_web_tools import WebSearch, WebFetch

    # Search the web
    searcher = WebSearch()
    results = await searcher.search("Python async programming")

    # Fetch and process a URL
    fetcher = WebFetch()
    result = await fetcher.fetch(
        "https://docs.python.org/3/library/asyncio.html",
        "Summarize the key concepts of asyncio"
    )
"""

from .web_search import WebSearch
from .web_fetch import WebFetch
from .config import Config
from .models import SearchResult, SearchResponse, FetchResult

__all__ = [
    "WebSearch",
    "WebFetch",
    "Config",
    "SearchResult",
    "SearchResponse",
    "FetchResult",
]

__version__ = "1.0.0"
