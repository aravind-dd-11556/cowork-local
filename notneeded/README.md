# Cowork Agent — A Reverse-Engineered Cowork-Style AI Agent

A fully functional recreation of Anthropic's **Cowork mode** agent architecture, built from the ground up by studying how Cowork works. This project demonstrates a deep understanding of Cowork's internals — from its XML-tagged system prompt design to its tool-calling loop, provider abstraction, and sandbox execution model.

Built entirely with a **free, self-hosted stack**: Ollama (local LLM) + SearXNG (metasearch engine). No paid API keys required.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Cowork Agent                         │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │   CLI    │──▶│  Agent Loop  │──▶│ Prompt Builder │  │
│  │Interface │   │  (15 iter)   │   │  (XML-tagged)  │  │
│  └──────────┘   └──────┬───────┘   └────────────────┘  │
│                        │                                │
│           ┌────────────┼────────────┐                   │
│           ▼            ▼            ▼                   │
│    ┌────────────┐ ┌─────────┐ ┌──────────┐             │
│    │  Ollama    │ │ OpenAI  │ │Anthropic │             │
│    │  Provider  │ │Provider │ │ Provider │             │
│    └──────┬─────┘ └─────────┘ └──────────┘             │
│           │                                             │
│    ┌──────┴──────────────────────────────┐              │
│    │          Tool Registry              │              │
│    │  bash · read · write · edit · glob  │              │
│    │  grep · web_search · web_fetch      │              │
│    │  todo_write                         │              │
│    └─────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
  ┌──────────────┐           ┌──────────────┐
  │   Ollama     │           │   SearXNG    │
  │  (Local LLM) │           │ (Metasearch) │
  └──────────────┘           └──────────────┘
```

## Key Design Decisions (Matching Cowork)

**XML-Tagged System Prompt** — Just like Cowork, the system prompt is assembled at runtime with XML sections: `<env>`, `<tools>`, `<behavioral_rules>`, `<runtime_context>`. Date and time are injected dynamically on each prompt build (not static).

**Tool Schemas in Prompt** — For Ollama (which lacks native tool_use), tool schemas are embedded directly in the system prompt with JSON calling instructions. The agent parses `tool_calls` JSON blocks from free-form LLM output using regex + a 3-tier JSON sanitizer.

**Agent Loop with Recovery** — The core loop runs up to 15 iterations. It includes truncation detection (catches when Ollama hits `num_predict` limits mid-JSON), intent-without-action detection (catches when the LLM says "I'll create the file" but doesn't actually call a tool), and automatic retry with nudging — all with caps to prevent infinite loops.

**Provider Abstraction** — Swappable LLM backends: Ollama (JSON-in-prompt), OpenAI (native tool_use), Anthropic (native tool_use). All implement the same `BaseLLMProvider` interface via a factory pattern.

**Parallel Tool Execution** — Multiple tool calls in a single LLM response are executed concurrently using `asyncio.gather()`.

## Project Structure

```
cowork_agent/
├── __init__.py              # Package init (v0.1.0)
├── __main__.py              # python -m entry point
├── main.py                  # CLI arg parsing, workspace resolution, tool registration
├── setup.py                 # Package setup with optional extras
├── requirements.txt         # Core: pyyaml, httpx
│
├── config/
│   ├── default_config.yaml  # Default settings (provider, model, timeouts, etc.)
│   └── settings.py          # Config loader with YAML merge + env var overrides
│
├── core/
│   ├── agent.py             # Agent loop — the orchestrator
│   ├── models.py            # Data models (Message, ToolCall, ToolResult, AgentResponse, ToolSchema)
│   ├── prompt_builder.py    # XML-tagged system prompt assembly
│   ├── tool_registry.py     # Tool registry with parallel execution
│   └── providers/
│       ├── base.py          # Abstract provider + ProviderFactory
│       ├── ollama.py        # Ollama provider (JSON parsing, truncation detection)
│       ├── openai_provider.py
│       └── anthropic_provider.py
│
├── tools/
│   ├── bash.py              # Shell command execution with timeout
│   ├── read.py              # File reading with line limits
│   ├── write.py             # File writing with auto-mkdir
│   ├── edit.py              # Exact string replacement (read-first guard)
│   ├── glob_tool.py         # File pattern matching (pathlib.glob)
│   ├── grep_tool.py         # Content search (ripgrep with Python fallback)
│   ├── web_search.py        # Web search via claude_web_tools
│   ├── web_fetch.py         # URL fetch + processing via claude_web_tools
│   └── todo.py              # In-memory task tracking
│
├── prompts/
│   └── behavioral_rules.py  # Agent personality and behavioral guidelines
│
├── interfaces/
│   └── cli.py               # Interactive terminal with spinner, ANSI colors, /commands
│
└── sandbox/
    ├── Containerfile         # Ubuntu 22.04 + Python 3.11 + ripgrep
    ├── compose.yml           # Podman/Docker compose (agent + ollama + searxng)
    └── searxng-config/
        └── settings.yml      # SearXNG config (google + bing + duckduckgo)

claude_web_tools/
├── __init__.py              # WebSearch, WebFetch exports
├── web_search.py            # SearXNG-backed search with result formatting
├── web_fetch.py             # URL fetching with 4-tier HTML→Markdown conversion
├── html_to_markdown.py      # trafilatura → markdownify → BS4 → regex fallback
├── llm_processor.py         # Ollama-based content summarization
├── cache.py                 # TTL-based response cache
├── config.py                # Configuration management
├── models.py                # SearchResult, SearchResponse, FetchResult
├── requirements.txt         # httpx, trafilatura, markdownify, beautifulsoup4
└── searxng-config/
    └── settings.yml
```

## Quick Start

### Option A: Run directly (recommended for development)

**Prerequisites:** Ollama and SearXNG running locally.

```bash
# 1. Start Ollama (if not already running)
ollama serve

# 2. Pull a model
ollama pull qwen2.5:7b    # or any model you prefer

# 3. Start SearXNG (using Podman/Docker)
podman run -d -p 8888:8080 --name searxng docker.io/searxng/searxng:latest

# 4. Install dependencies
cd cowork_agent
pip install -r requirements.txt
pip install -r ../claude_web_tools/requirements.txt

# 5. Run the agent
python -m cowork_agent -p ollama -m qwen2.5:7b -v
```

### Option B: Run with containers (full stack)

```bash
cd cowork_agent/sandbox

# Start all services (Ollama + SearXNG + Agent)
podman-compose up -d

# Pull a model into the Ollama container
podman exec -it cowork-ollama ollama pull qwen2.5:7b

# Attach to the agent
podman start -ai cowork-agent
```

### CLI Usage

```
╔══════════════════════════════════════════╗
║         🤖 Cowork Agent v0.1.0          ║
╚══════════════════════════════════════════╝

  Workspace: /Users/you/Documents/project

You ▸ search about latest AI news and create a blog in html
  ⠋ Thinking... (3s)
  🌐 Executing web_search...
  ✓ ### Search Results for: latest AI news ...
  ⠙ Thinking... (12s)
  ✏️ Executing write...
  ✓ Successfully wrote 4889 bytes (113 lines) to blog.html
  ⠹ Thinking... (5s)

Agent ▸ I've created a comprehensive blog post at blog.html with the latest AI news.

You ▸ /help
You ▸ /todos
You ▸ /clear
You ▸ /exit
```

### CLI Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-p`, `--provider` | LLM provider (`ollama`, `openai`, `anthropic`) | `ollama` |
| `-m`, `--model` | Model name | `qwen3-vl:235b-instruct-cloud` |
| `-v`, `--verbose` | Enable debug logging | `false` |
| `--workspace` | Working directory path | Interactive prompt |
| `-c`, `--config` | Custom config file path | `default_config.yaml` |

## How It Works (Matching Cowork's Flow)

```
User Input
    │
    ▼
┌─────────────────┐
│ Build System     │◀── XML-tagged prompt with:
│ Prompt           │    • Date/time injection
│                  │    • Tool schemas
│                  │    • Behavioral rules
│                  │    • Todo context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Call LLM         │──── Ollama: parse tool_calls from text
│                  │     OpenAI/Anthropic: native tool_use
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 Tool      No tools
 Calls     ─────────▶ Return text to user
    │
    ▼
┌─────────────────┐
│ Execute Tools    │──── Parallel via asyncio.gather()
│ (in parallel)   │
└────────┬────────┘
         │
         ▼
   Add results to
   memory, loop ──────▶ Back to "Build System Prompt"
```

## Recovery Mechanisms

The agent handles common failure modes of local LLMs:

| Issue | Detection | Recovery |
|-------|-----------|----------|
| **Truncated JSON** | Ollama `done_reason: "length"` + unclosed brackets | Re-prompt with "output shorter content" (max 2 retries) |
| **Intent without action** | LLM says "I'll create..." but no `tool_calls` JSON | Nudge: "output the tool_calls block now" (max 2 nudges) |
| **Invalid JSON** | Literal newlines in JSON strings | 3-tier sanitizer: direct → character-walk escape → brute force |
| **Bloated context** | Large tool outputs re-sent to LLM | Auto-truncation to 3K chars per message |

## Configuration

Edit `cowork_agent/config/default_config.yaml`:

```yaml
llm:
  provider: "ollama"
  model: "qwen2.5:7b"
  temperature: 0.7
  max_tokens: 16384

providers:
  ollama:
    base_url: "http://localhost:11434"
    timeout: 300

agent:
  max_iterations: 15
  workspace_dir: "./workspace"
```

## What This Project Demonstrates

This project is a proof-of-concept showing that Cowork's architecture can be understood, replicated, and extended:

- **System prompt engineering** — XML-tagged, dynamically assembled prompts with runtime context injection (date, todos, tool schemas)
- **Tool-calling on local LLMs** — Embedding tool schemas in prompts and parsing structured JSON from free-form text, with robust error recovery for the quirks of local models
- **Agent loop design** — Iterative tool-use loop with parallel execution, conversation memory, truncation handling, and graceful degradation
- **Provider abstraction** — Clean separation between the agent logic and the LLM backend, making it trivial to swap between Ollama, OpenAI, and Anthropic
- **Self-hosted web tools** — WebSearch (SearXNG) and WebFetch (4-tier HTML→Markdown) that mirror Cowork's web capabilities without any paid APIs
- **Container-based sandbox** — Podman/Docker compose setup mirroring Cowork's isolated execution environment

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Inference | Ollama (local), OpenAI, Anthropic |
| Web Search | SearXNG (self-hosted metasearch) |
| HTML Processing | trafilatura + markdownify + BeautifulSoup4 |
| Content Summarization | Ollama (local LLM) |
| HTTP Client | httpx (async) |
| Configuration | PyYAML |
| Containerization | Podman / Docker |
| Search Backend | ripgrep (with Python re fallback) |

## License

This project was created for educational and demonstration purposes — to showcase an understanding of AI agent architecture patterns as implemented in Anthropic's Cowork mode.

---

*Built by studying Cowork, for the Cowork team.*
