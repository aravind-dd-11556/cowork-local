"""
WebSearch Tool — Claude-style web search using SearXNG backend.

Architecture (matching Claude's implementation):
- Single API call to SearXNG instance
- Returns structured search result blocks with titles, snippets, URLs
- Supports domain filtering (allow/block lists)
- Results formatted as markdown hyperlinks
- Mandatory "Sources:" section in output

Usage:
    searcher = WebSearch()
    results = await searcher.search("Python web scraping 2026")
    print(results.to_markdown())
"""

import httpx
from urllib.parse import urlparse, urlencode
from typing import Optional

from .config import Config
from .models import SearchResult, SearchResponse


class WebSearch:
    """
    Web search tool powered by SearXNG.

    Mirrors Claude's WebSearch behavior:
    - Sends query to search backend
    - Returns structured results (title, URL, snippet)
    - Supports domain allow/block filtering
    - Formats output with Sources section
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or Config.SEARXNG_BASE_URL).rstrip("/")

    async def search(
        self,
        query: str,
        allowed_domains: Optional[list[str]] = None,
        blocked_domains: Optional[list[str]] = None,
        max_results: int = Config.SEARCH_DEFAULT_RESULTS,
    ) -> SearchResponse:
        """
        Execute a web search query.

        Args:
            query: The search query string
            allowed_domains: Only include results from these domains
            blocked_domains: Exclude results from these domains
            max_results: Maximum number of results to return

        Returns:
            SearchResponse with structured results
        """
        try:
            # Build the SearXNG query
            # If allowed_domains specified, append site: operators to query
            effective_query = self._apply_domain_filters(
                query, allowed_domains, blocked_domains
            )

            # Call SearXNG API
            raw_results = await self._call_searxng(effective_query)

            # Parse and filter results
            results = self._parse_results(raw_results, max_results)

            # Apply domain blocking on results
            if blocked_domains:
                blocked = set(d.lower() for d in blocked_domains)
                blocked.update(d.lower() for d in Config.BLOCKED_DOMAINS)
                results = [
                    r for r in results
                    if not self._domain_matches(r.url, blocked)
                ]

            if allowed_domains:
                allowed = set(d.lower() for d in allowed_domains)
                results = [
                    r for r in results
                    if self._domain_matches(r.url, allowed)
                ]

            return SearchResponse(
                query=query,
                results=results[:max_results],
                total_results=len(results),
            )

        except httpx.ConnectError:
            return SearchResponse(
                query=query,
                error=(
                    f"Cannot connect to SearXNG at {self.base_url}. "
                    "Make sure SearXNG is running (podman-compose up -d)."
                ),
            )
        except Exception as e:
            return SearchResponse(query=query, error=str(e))

    async def _call_searxng(self, query: str) -> dict:
        """Make the actual HTTP call to SearXNG."""
        params = {
            "q": query,
            "format": Config.SEARXNG_FORMAT,
            "engines": "google,bing,duckduckgo",
        }

        async with httpx.AsyncClient(timeout=Config.FETCH_TIMEOUT) as client:
            response = await client.get(
                f"{self.base_url}/search",
                params=params,
            )
            response.raise_for_status()
            return response.json()

    def _parse_results(self, raw: dict, max_results: int) -> list[SearchResult]:
        """Parse SearXNG JSON response into SearchResult objects."""
        results = []
        seen_urls = set()

        for item in raw.get("results", []):
            url = item.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            results.append(SearchResult(
                title=item.get("title", "Untitled"),
                url=url,
                snippet=item.get("content", ""),
                engine=item.get("engine", ""),
                score=item.get("score", 0.0),
            ))

            if len(results) >= max_results:
                break

        return results

    def _apply_domain_filters(
        self,
        query: str,
        allowed_domains: Optional[list[str]],
        blocked_domains: Optional[list[str]],
    ) -> str:
        """
        Modify query to apply domain filtering at the search level.
        SearXNG passes these through to underlying engines.
        """
        parts = [query]

        if allowed_domains:
            # Add site: operators for allowed domains
            site_parts = " OR ".join(f"site:{d}" for d in allowed_domains)
            parts.append(f"({site_parts})")

        if blocked_domains:
            # Add -site: operators for blocked domains
            for d in blocked_domains:
                parts.append(f"-site:{d}")

        # Also apply global blocked domains from config
        for d in Config.BLOCKED_DOMAINS:
            parts.append(f"-site:{d}")

        return " ".join(parts)

    @staticmethod
    def _domain_matches(url: str, domains: set[str]) -> bool:
        """Check if a URL's domain matches any in the given set."""
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            # Check exact match and parent domain match
            return any(
                host == d or host.endswith(f".{d}")
                for d in domains
            )
        except Exception:
            return False
