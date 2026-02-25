"""
Configuration for Claude-style WebSearch and WebFetch tools.
All settings can be overridden via environment variables.
"""

import os


class Config:
    # ── SearXNG (WebSearch backend) ──
    SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8888")
    SEARXNG_FORMAT = "json"
    SEARCH_DEFAULT_RESULTS = 10

    # ── Ollama (WebFetch Stage 2 LLM) ──
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:235b-instruct-cloud")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))

    # ── WebFetch Settings ──
    FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "30"))
    FETCH_MAX_CONTENT_LENGTH = int(os.getenv("FETCH_MAX_CONTENT_LENGTH", "500000"))  # 500KB
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))  # 15 minutes
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )

    # ── Domain Filtering ──
    # Domains that should never be fetched (compliance/legal)
    BLOCKED_DOMAINS = [
        # Add domains to block here
    ]
