"""
WebFetch Tool — Claude-style two-stage web content fetcher.

Architecture (matching Claude's implementation):
┌──────────────────────────────────────────────────────────┐
│                     WebFetch Pipeline                     │
│                                                          │
│  Input: URL + Prompt                                     │
│     │                                                    │
│     ▼                                                    │
│  ┌─────────────┐    ┌──────────────┐                    │
│  │  15-min     │───▶│ Return cached │ (if cache hit)     │
│  │  Cache      │    │ result       │                     │
│  └─────────────┘    └──────────────┘                     │
│     │ (cache miss)                                       │
│     ▼                                                    │
│  ┌─────────────────────────────────┐                    │
│  │ STAGE 1: Fetch & Convert        │                    │
│  │  1. HTTP→HTTPS upgrade          │                    │
│  │  2. Fetch HTML (handle redirects)│                   │
│  │  3. Convert HTML → Markdown     │                    │
│  └─────────────────────────────────┘                    │
│     │                                                    │
│     ▼                                                    │
│  ┌─────────────────────────────────┐                    │
│  │ STAGE 2: LLM Processing         │                    │
│  │  1. Send markdown + prompt      │                    │
│  │     to small local model        │                    │
│  │  2. Model extracts/summarizes   │                    │
│  │  3. Return processed response   │                    │
│  └─────────────────────────────────┘                    │
│     │                                                    │
│     ▼                                                    │
│  Output: Processed content string                        │
└──────────────────────────────────────────────────────────┘

Usage:
    fetcher = WebFetch()
    result = await fetcher.fetch("https://example.com", "What is this page about?")
    print(result.to_markdown())
"""

from typing import Optional
from urllib.parse import urlparse

from .config import Config
from .cache import TTLCache
from .html_to_markdown import HTMLToMarkdown
from .llm_processor import LLMProcessor
from .models import FetchResult


class WebFetch:
    """
    Two-stage web content fetcher matching Claude's WebFetch architecture.

    Stage 1: Fetch URL → Convert HTML to Markdown (HTMLToMarkdown)
    Stage 2: Process Markdown with LLM using user's prompt (LLMProcessor)

    Features:
    - 15-minute self-cleaning cache (Stage 1 results cached)
    - HTTP → HTTPS auto-upgrade
    - Cross-host redirect detection (returns redirect URL, doesn't follow)
    - Content size limiting
    - Domain blocking for compliance
    """

    def __init__(
        self,
        cache_ttl: int = Config.CACHE_TTL_SECONDS,
        ollama_model: Optional[str] = None,
    ):
        self._cache = TTLCache(ttl_seconds=cache_ttl)
        self._html_converter = HTMLToMarkdown()
        self._llm = LLMProcessor(model=ollama_model)

    async def fetch(
        self,
        url: str,
        prompt: str,
    ) -> FetchResult:
        """
        Fetch a URL and process its content with the LLM.

        Args:
            url: The URL to fetch (HTTP auto-upgraded to HTTPS)
            prompt: What to extract/analyze from the page content

        Returns:
            FetchResult with processed content or error/redirect info
        """
        # ── Validate URL ──
        if not url or not isinstance(url, str):
            return FetchResult(url=url, prompt=prompt, error="Invalid URL provided")

        # ── Check domain blocking ──
        blocked_error = self._check_domain_blocked(url)
        if blocked_error:
            return FetchResult(url=url, prompt=prompt, error=blocked_error)

        # ── STAGE 1: Fetch and convert (with caching) ──
        cache_key = url.lower().strip()
        cached_markdown = self._cache.get(cache_key)

        if cached_markdown is not None:
            # Cache hit — skip Stage 1, go directly to Stage 2
            markdown = cached_markdown
            from_cache = True
        else:
            # Cache miss — run Stage 1
            markdown, redirect_url, error = await self._html_converter.fetch_and_convert(url)

            if error:
                return FetchResult(url=url, prompt=prompt, error=error)

            if redirect_url:
                return FetchResult(url=url, prompt=prompt, redirect_url=redirect_url)

            # Cache the Stage 1 result for 15 minutes
            self._cache.set(cache_key, markdown)
            from_cache = False

        # ── STAGE 2: Process with LLM ──
        processed = await self._llm.process(
            content=markdown,
            prompt=prompt,
            source_url=url,
        )

        return FetchResult(
            url=url,
            prompt=prompt,
            raw_markdown=markdown,
            processed_content=processed,
            from_cache=from_cache,
        )

    def _check_domain_blocked(self, url: str) -> Optional[str]:
        """Check if the URL's domain is in the block list."""
        try:
            parsed = urlparse(url if "://" in url else f"https://{url}")
            host = (parsed.hostname or "").lower()

            for blocked in Config.BLOCKED_DOMAINS:
                if host == blocked.lower() or host.endswith(f".{blocked.lower()}"):
                    return (
                        f"Domain '{host}' cannot be fetched due to content restrictions. "
                        "This restriction exists for legal/compliance reasons."
                    )
        except Exception:
            pass
        return None

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        return self._cache.stats()

    def clear_cache(self) -> None:
        """Clear the fetch cache."""
        self._cache.clear()

    async def health_check(self) -> dict:
        """Check if all dependencies (Ollama) are available."""
        llm_health = await self._llm.check_health()
        return {
            "cache": self._cache.stats(),
            "llm": llm_health,
        }
