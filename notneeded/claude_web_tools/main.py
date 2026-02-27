"""
Main runner — demonstrates both WebSearch and WebFetch tools.

Run:
    python -m claude_web_tools.main

Make sure SearXNG and Ollama are running first:
    podman-compose up -d          # SearXNG
    ollama serve                  # Ollama
    ollama pull qwen3-vl:235b-instruct-cloud  # Pull the model
"""

import asyncio
import sys
import json

from .web_search import WebSearch
from .web_fetch import WebFetch


async def demo_search():
    """Demonstrate WebSearch tool."""
    print("=" * 60)
    print("  WebSearch Demo (SearXNG Backend)")
    print("=" * 60)

    searcher = WebSearch()

    # Basic search
    print("\n[1] Basic search: 'Python web scraping best practices 2026'")
    results = await searcher.search("Python web scraping best practices 2026")
    print(results.to_markdown())

    # Search with domain filtering
    print("\n[2] Search with allowed domains (only stackoverflow.com):")
    results = await searcher.search(
        "how to parse HTML in Python",
        allowed_domains=["stackoverflow.com"],
    )
    print(results.to_markdown())

    # Search with blocked domains
    print("\n[3] Search with blocked domains (no reddit.com):")
    results = await searcher.search(
        "best Python web frameworks",
        blocked_domains=["reddit.com"],
        max_results=5,
    )
    print(results.to_markdown())


async def demo_fetch():
    """Demonstrate WebFetch tool."""
    print("\n" + "=" * 60)
    print("  WebFetch Demo (Two-Stage Pipeline)")
    print("=" * 60)

    fetcher = WebFetch()

    # Check health first
    print("\n[Health Check]")
    health = await fetcher.health_check()
    print(json.dumps(health, indent=2))

    if not health.get("llm", {}).get("ollama_running"):
        print("\n⚠ Ollama is not running. Start it with: ollama serve")
        print("  Then pull a model: ollama pull llama3.2")
        return

    # Basic fetch
    print("\n[1] Fetch and process: httpbin.org")
    result = await fetcher.fetch(
        "https://httpbin.org",
        "What is this website? What services does it provide?"
    )
    print(result.to_markdown())

    # Fetch with cache demonstration
    print("\n[2] Same URL again (should be from cache):")
    result = await fetcher.fetch(
        "https://httpbin.org",
        "List the main API endpoints available."
    )
    print(f"From cache: {result.from_cache}")
    print(result.to_markdown())

    # Show cache stats
    print(f"\n[Cache Stats] {fetcher.cache_stats()}")

    # Fetch a real article
    print("\n[3] Fetch a documentation page:")
    result = await fetcher.fetch(
        "https://docs.python.org/3/library/asyncio.html",
        "What are the main components of Python's asyncio library? Summarize in 3-4 bullet points."
    )
    print(result.to_markdown())


async def interactive_mode():
    """Interactive mode — use the tools from command line."""
    searcher = WebSearch()
    fetcher = WebFetch()

    print("=" * 60)
    print("  Claude Web Tools — Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  search <query>          — WebSearch")
    print("  fetch <url> <prompt>    — WebFetch")
    print("  health                  — Check dependencies")
    print("  cache                   — Show cache stats")
    print("  quit                    — Exit")
    print()

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue

        if line.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if line.lower() == "health":
            health = await fetcher.health_check()
            print(json.dumps(health, indent=2))
            continue

        if line.lower() == "cache":
            print(json.dumps(fetcher.cache_stats(), indent=2))
            continue

        if line.lower().startswith("search "):
            query = line[7:].strip()
            if query:
                results = await searcher.search(query)
                print(results.to_markdown())
            else:
                print("Usage: search <query>")
            continue

        if line.lower().startswith("fetch "):
            parts = line[6:].strip().split(" ", 1)
            if len(parts) == 2:
                url, prompt = parts
                result = await fetcher.fetch(url, prompt)
                print(result.to_markdown())
            else:
                print("Usage: fetch <url> <prompt>")
            continue

        print(f"Unknown command: {line}")
        print("Type 'help' for available commands.")


async def main():
    if "--interactive" in sys.argv or "-i" in sys.argv:
        await interactive_mode()
    else:
        await demo_search()
        await demo_fetch()


if __name__ == "__main__":
    asyncio.run(main())
