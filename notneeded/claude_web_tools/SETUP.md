# Claude Web Tools — Setup Guide

A Python implementation of Claude's WebSearch and WebFetch tools, built with a fully free stack.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Web Tools                          │
│                                                             │
│  ┌──────────────┐         ┌──────────────────────────────┐ │
│  │  WebSearch    │         │  WebFetch (Two-Stage)         │ │
│  │              │         │                              │ │
│  │  Query ──┐   │         │  URL + Prompt                │ │
│  │          ▼   │         │     │                        │ │
│  │  ┌────────┐  │         │     ▼                        │ │
│  │  │SearXNG │  │         │  ┌──────────┐  15-min cache  │ │
│  │  │(Podman)│  │         │  │ Stage 1  │◄──────────┐    │ │
│  │  └────────┘  │         │  │ Fetch +  │           │    │ │
│  │      │       │         │  │ HTML→MD  │───────────┘    │ │
│  │      ▼       │         │  └──────────┘                │ │
│  │  Structured  │         │      │                       │ │
│  │  Results     │         │      ▼                       │ │
│  │  (title,     │         │  ┌──────────┐                │ │
│  │   url,       │         │  │ Stage 2  │                │ │
│  │   snippet)   │         │  │ Ollama   │                │ │
│  │              │         │  │ (LLM)    │                │ │
│  └──────────────┘         │  └──────────┘                │ │
│                           │      │                       │ │
│                           │      ▼                       │ │
│                           │  Processed content           │ │
│                           └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.10+
- Podman + podman-compose (for SearXNG)
- Ollama (for local LLM)

## Step 1: Install Python Dependencies

```bash
cd claude_web_tools
pip install -r requirements.txt
```

## Step 2: Start SearXNG (Search Backend)

```bash
# Install podman-compose if you don't have it
pip install podman-compose

# Start SearXNG container
podman-compose up -d

# OR run directly without compose:
podman run -d --name searxng -p 8888:8080 \
  -v ./searxng-config:/etc/searxng:Z \
  -e SEARXNG_BASE_URL=http://localhost:8888/ \
  docker.io/searxng/searxng:latest

# Verify it's running
curl http://localhost:8888/search?q=test&format=json
```

SearXNG will be available at http://localhost:8888. The JSON API endpoint
is what WebSearch uses internally.

> **Note:** The `:Z` suffix on the volume mount is for SELinux relabeling,
> which Podman needs on Fedora/RHEL systems. It's harmless on other distros.

## Step 3: Install and Start Ollama (Local LLM)

```bash
# Install Ollama (macOS)
brew install ollama

# OR install on Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama server
ollama serve

# Pull the model (in a separate terminal)
ollama pull qwen3-vl:235b-instruct-cloud
```

Ollama will be available at http://localhost:11434.

## Step 4: Run the Tools

```bash
# Run the demo
python -m claude_web_tools

# Interactive mode
python -m claude_web_tools -i
```

## Interactive Commands

```
>>> search Python async programming best practices
>>> fetch https://docs.python.org/3 What is the latest Python version?
>>> health
>>> cache
>>> quit
```

## Using in Your Own Code

```python
import asyncio
from claude_web_tools import WebSearch, WebFetch

async def main():
    # ── WebSearch ──
    searcher = WebSearch()
    results = await searcher.search(
        "machine learning frameworks 2026",
        allowed_domains=["github.com", "arxiv.org"],
    )
    print(results.to_markdown())

    # ── WebFetch ──
    fetcher = WebFetch()
    result = await fetcher.fetch(
        "https://docs.python.org/3/library/asyncio.html",
        "Summarize the key asyncio concepts in 5 bullet points"
    )
    print(result.to_markdown())

    # Access raw data
    print(f"From cache: {result.from_cache}")
    print(f"Raw markdown length: {len(result.raw_markdown or '')}")

asyncio.run(main())
```

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| SEARXNG_BASE_URL | http://localhost:8888 | SearXNG instance URL |
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama server URL |
| OLLAMA_MODEL | qwen3-vl:235b-instruct-cloud | Model for Stage 2 processing |
| OLLAMA_TIMEOUT | 300 | LLM request timeout (seconds) |
| FETCH_TIMEOUT | 30 | HTTP fetch timeout (seconds) |
| FETCH_MAX_CONTENT_LENGTH | 500000 | Max content size (bytes) |
| CACHE_TTL_SECONDS | 900 | Cache TTL (default 15 min) |

## How It Compares to Claude's Implementation

| Feature | Claude's Tool | This Implementation |
|---------|--------------|-------------------|
| Search backend | Proprietary (likely Google/Bing wrapper) | SearXNG (open source) |
| Domain filtering | Built-in allow/block | Same — allow/block lists |
| Result format | Structured blocks + markdown links | Same format |
| Sources section | Mandatory after every search | Built into output |
| HTML fetching | Internal service | httpx with redirect handling |
| HTML→Markdown | Internal converter | trafilatura + markdownify |
| HTTP→HTTPS upgrade | Automatic | Automatic |
| Cross-host redirects | Returns redirect URL | Same behavior |
| Cache | 15-minute self-cleaning | Same — TTLCache with 15min TTL |
| Stage 2 LLM | Small fast model (likely Haiku) | Ollama local model |
| Content restrictions | Legal compliance layer | Configurable block list |
| Cost | Included in Claude subscription | Completely free |

## Project Structure

```
claude_web_tools/
├── __init__.py           # Package exports
├── __main__.py           # python -m entry point
├── config.py             # Configuration (env vars)
├── cache.py              # TTL cache (15-min self-cleaning)
├── models.py             # Data models (SearchResult, FetchResult)
├── web_search.py         # WebSearch tool (SearXNG)
├── web_fetch.py          # WebFetch orchestrator (Stage 1 + 2)
├── html_to_markdown.py   # Stage 1: Fetch + HTML→MD
├── llm_processor.py      # Stage 2: Ollama LLM processing
├── main.py               # Demo runner + interactive mode
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # SearXNG container (Podman-compatible)
├── searxng-config/
│   └── settings.yml      # SearXNG configuration
└── SETUP.md              # This file
```
