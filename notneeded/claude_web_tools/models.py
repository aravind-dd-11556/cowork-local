"""
Data models for WebSearch and WebFetch results.
Clean structured outputs matching Claude's tool response format.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    """A single search result entry."""
    title: str
    url: str
    snippet: str
    engine: str = ""
    score: float = 0.0

    def to_markdown(self) -> str:
        """Format as markdown hyperlink with snippet."""
        return f"- [{self.title}]({self.url})\n  {self.snippet}"


@dataclass
class SearchResponse:
    """Complete WebSearch response."""
    query: str
    results: list[SearchResult] = field(default_factory=list)
    total_results: int = 0
    error: Optional[str] = None

    def to_markdown(self) -> str:
        """Format full response as markdown, matching Claude's output style."""
        if self.error:
            return f"Search failed: {self.error}"
        if not self.results:
            return f"No results found for: {self.query}"

        lines = [f"### Search Results for: {self.query}\n"]
        for r in self.results:
            lines.append(r.to_markdown())
        lines.append(f"\n**Sources:**")
        for r in self.results:
            lines.append(f"- [{r.title}]({r.url})")
        return "\n".join(lines)


@dataclass
class FetchResult:
    """Complete WebFetch response."""
    url: str
    prompt: str
    raw_markdown: Optional[str] = None  # Stage 1 output
    processed_content: Optional[str] = None  # Stage 2 output (LLM processed)
    from_cache: bool = False
    redirect_url: Optional[str] = None
    error: Optional[str] = None

    def to_markdown(self) -> str:
        """Format the fetch result."""
        if self.error:
            return f"Fetch failed for {self.url}: {self.error}"
        if self.redirect_url:
            return (
                f"The URL redirected to a different host: {self.redirect_url}\n"
                f"Please make a new WebFetch request with the redirect URL."
            )
        if self.processed_content:
            cache_note = " (from cache)" if self.from_cache else ""
            return f"{self.processed_content}{cache_note}"
        return "No content retrieved."
