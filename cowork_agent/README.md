# Cowork Agent

A modular AI agent framework inspired by Anthropic's Cowork mode — built with configurable LLM providers, a rich tool ecosystem, multi-agent orchestration, and an interactive CLI.

## Quick Start

### 1. Prerequisites

- Python 3.10+
- One of these LLM backends:
  - **Ollama** (default, free, local) — [install guide](https://ollama.com/download)
  - **OpenAI API** — requires `OPENAI_API_KEY`
  - **Anthropic API** — requires `ANTHROPIC_API_KEY`

### 2. Install Dependencies

```bash
cd cowork_agent

# Core dependencies (required)
pip install pyyaml httpx

# If using OpenAI provider
pip install openai

# If using Anthropic provider
pip install anthropic

# If using web search/fetch tools (optional)
pip install trafilatura markdownify beautifulsoup4 lxml
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### 3. Configure Your LLM Provider

The agent reads from `cowork_agent/config/default_config.yaml`. You can override settings via environment variables or a custom config file.

**Option A — Ollama (default, no API key needed):**

```bash
# Install and start Ollama, then pull a model
ollama pull qwen3-vl:235b-instruct-cloud

# Or use any other model
ollama pull llama3.1
export OLLAMA_MODEL="llama3.1"
```

**Option B — OpenAI:**

```bash
export COWORK_LLM_PROVIDER="openai"
export OPENAI_API_KEY="sk-..."
```

**Option C — Anthropic:**

```bash
export COWORK_LLM_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 4. Run the Agent

```bash
# From the project root (parent of cowork_agent/)
python -m cowork_agent

# Or with options
python -m cowork_agent --provider openai --model gpt-4o --workspace ~/my-project
python -m cowork_agent -v        # verbose (INFO logs)
python -m cowork_agent -vv       # debug (DEBUG logs)
python -m cowork_agent -c my_config.yaml  # custom config file
```

On first run, the agent will ask you to pick a workspace directory — this is where it reads/writes files.

### 5. Start Chatting

```
╭─ cowork-agent ─╮
│ Workspace: /home/user/my-project
│ Provider:  ollama (qwen3-vl:235b-instruct-cloud)
│ Tools:     bash, read, write, edit, glob, grep, todo_write, ask_user, web_search, web_fetch
╰────────────────╯

You: Can you read my project files and summarize what this codebase does?

You: Create a Python script that fetches weather data and saves it to weather.json

You: /help
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/tools` | List all registered tools |
| `/status` | Show agent status (provider, tokens, etc.) |
| `/quit` | Exit the agent |
| `Ctrl+C` | Cancel current operation |

## Project Structure

```
cowork_agent/
├── __init__.py              # Package init
├── __main__.py              # Entry point: python -m cowork_agent
├── main.py                  # CLI arg parsing, provider/tool setup, launch
├── config/
│   ├── default_config.yaml  # Default configuration
│   └── settings.py          # Config loader (YAML + env vars)
├── core/
│   ├── agent.py             # Main Agent loop (plan → call LLM → execute tools → repeat)
│   ├── models.py            # Data models (Message, ToolCall, ToolResult, AgentResponse)
│   ├── tool_registry.py     # Tool registration and dispatch
│   ├── prompt_builder.py    # System prompt assembly
│   ├── context_manager.py   # Conversation context + token window management
│   ├── safety_checker.py    # Input/output safety validation
│   ├── session_manager.py   # Conversation persistence (save/load/list)
│   ├── stream_hardener.py   # SSE streaming resilience (reconnect, circuit breaker)
│   ├── token_tracker.py     # Token usage tracking and budget enforcement
│   ├── response_cache.py    # Semantic response caching (LRU + TTL)
│   ├── provider_fallback.py # Automatic provider failover chain
│   ├── multimodal.py        # Image/PDF input handling
│   ├── plan_mode.py         # Plan mode state machine
│   ├── skill_registry.py    # Skill discovery and loading
│   ├── plugin_system.py     # Plugin loading from directories
│   ├── scheduler.py         # Cron-based task scheduling
│   ├── mcp_client.py        # MCP (Model Context Protocol) client
│   ├── worktree.py          # Git worktree management
│   ├── agent_registry.py    # [Sprint 5] Multi-agent lifecycle management
│   ├── context_bus.py       # [Sprint 5] Inter-agent pub/sub messaging
│   ├── delegate_tool.py     # [Sprint 5] Agent-to-agent task delegation
│   ├── supervisor.py        # [Sprint 5] Multi-agent orchestration
│   ├── conflict_resolver.py # [Sprint 5] Conflict detection & resolution
│   └── providers/
│       ├── base.py          # BaseLLMProvider interface + ProviderFactory
│       ├── ollama.py        # Ollama provider (local LLMs)
│       ├── openai_provider.py   # OpenAI provider
│       └── anthropic_provider.py # Anthropic provider
├── tools/
│   ├── base.py              # BaseTool interface
│   ├── bash.py              # Shell command execution (sandboxed)
│   ├── read.py              # File reading (with line limits)
│   ├── write.py             # File writing (workspace-scoped)
│   ├── edit.py              # Exact string replacement editing
│   ├── glob_tool.py         # File pattern matching
│   ├── grep_tool.py         # Content search (ripgrep-style)
│   ├── todo.py              # Todo list management
│   ├── ask_user.py          # Structured user question tool
│   ├── web_fetch.py         # URL fetching + LLM processing (with SSRF protection)
│   ├── web_search.py        # Web search via SearXNG
│   ├── notebook_edit.py     # Jupyter notebook cell editing
│   ├── plan_tools.py        # Plan mode enter/exit tools
│   ├── task_tool.py         # Subagent delegation tool
│   ├── scheduler_tools.py   # Scheduled task CRUD tools
│   ├── mcp_bridge.py        # MCP tool bridge
│   └── worktree_tool.py     # Git worktree tools
├── interfaces/
│   └── cli.py               # Interactive terminal chat interface
├── prompts/
│   └── behavioral_rules.py  # System prompt behavioral rules
├── vendor/
│   └── claude_web_tools/    # Web search & fetch backend (SearXNG + Ollama)
│       ├── web_search.py    # SearXNG search client
│       ├── web_fetch.py     # Two-stage URL fetcher (fetch → LLM process)
│       ├── html_to_markdown.py  # HTML → Markdown converter
│       ├── llm_processor.py # Ollama LLM content processor
│       ├── cache.py         # 15-minute TTL cache
│       ├── config.py        # Web tools configuration
│       └── models.py        # SearchResult, FetchResult models
├── tests/
│   ├── test_p1.py           # Sprint 1: Circuit breaker, retry, streaming (66 tests)
│   ├── test_p2.py           # Sprint 2: Skills, plugins, MCP, scheduler (56 tests)
│   ├── test_p3.py           # Sprint 3: Ask user, notebook, worktree (49 tests)
│   ├── test_p4.py           # Sprint 4: Tokens, cache, fallback, multimodal (61 tests)
│   ├── test_p4_edge.py      # Sprint 4: Edge cases (43 tests)
│   ├── test_p5.py           # Sprint 5: Multi-agent orchestration (83 tests)
│   ├── test_qa_audit.py     # QA audit tests (42 tests)
│   └── test_security_audit.py # Security audit tests (43 tests)
├── requirements.txt
└── setup.py
```

## Running Tests

```bash
# From the project root (parent of cowork_agent/)

# Run all unittest-based suites (377 tests)
python -m unittest cowork_agent.tests.test_p2 cowork_agent.tests.test_p3 \
  cowork_agent.tests.test_qa_audit cowork_agent.tests.test_security_audit \
  cowork_agent.tests.test_p4 cowork_agent.tests.test_p4_edge cowork_agent.tests.test_p5

# Run Sprint 1 tests (custom runner, 66 tests)
python cowork_agent/tests/test_p1.py

# Run a specific suite
python -m unittest cowork_agent.tests.test_p5 -v

# Total: 443 tests across 8 suites
```

## Optional: Web Search & Fetch Setup

The agent includes web search and URL fetch tools powered by SearXNG (search engine) and Ollama (content processing). These are optional — the agent works fine without them.

To enable web tools:

```bash
# 1. Start SearXNG (search backend)
cd cowork_agent/vendor/claude_web_tools
docker-compose up -d   # or: podman-compose up -d

# 2. Make sure Ollama is running (used for content processing)
ollama serve

# 3. Verify
curl http://localhost:8888/search?q=test&format=json   # SearXNG
curl http://localhost:11434/api/tags                     # Ollama
```

Environment variables for web tools:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARXNG_BASE_URL` | `http://localhost:8888` | SearXNG instance URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama instance URL |
| `OLLAMA_MODEL` | `qwen3-vl:235b-instruct-cloud` | Model for content processing |
| `FETCH_TIMEOUT` | `30` | HTTP fetch timeout (seconds) |
| `CACHE_TTL_SECONDS` | `900` | URL cache duration (15 min) |

## Configuration Reference

All settings can be set via `default_config.yaml`, a custom YAML file (`-c`), or environment variables:

| Env Variable | Config Key | Default | Description |
|-------------|------------|---------|-------------|
| `COWORK_LLM_PROVIDER` | `llm.provider` | `ollama` | LLM provider: ollama, openai, anthropic |
| `COWORK_LLM_MODEL` | `llm.model` | `qwen3-vl:235b-instruct-cloud` | Model name |
| `COWORK_LLM_TEMPERATURE` | `llm.temperature` | `0.7` | Sampling temperature |
| `OLLAMA_BASE_URL` | `providers.ollama.base_url` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_API_KEY` | `providers.openai.api_key` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | `providers.anthropic.api_key` | — | Anthropic API key |
| `COWORK_WORKSPACE` | `agent.workspace_dir` | `./workspace` | Working directory |

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        CLI Interface                          │
│  (Interactive chat, /commands, tool output, todo widget)      │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│                      Agent Loop                               │
│  1. Build system prompt (PromptBuilder + behavioral rules)    │
│  2. Send messages to LLM provider                             │
│  3. Parse tool calls from response                            │
│  4. Execute tools via ToolRegistry                            │
│  5. Safety check inputs/outputs (SafetyChecker)               │
│  6. Manage context window (ContextManager)                    │
│  7. Track tokens & budget (TokenTracker)                      │
│  8. Repeat until task complete or max iterations              │
└──────┬──────────────┬──────────────┬─────────────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────────────────────────┐
│ LLM Provider│ │ Tool       │ │ Multi-Agent Orchestration      │
│ (Ollama /   │ │ Registry   │ │ (AgentRegistry, ContextBus,    │
│  OpenAI /   │ │ (15+ tools)│ │  Supervisor, ConflictResolver) │
│  Anthropic) │ │            │ │                                │
└─────────────┘ └────────────┘ └────────────────────────────────┘
```

## License

Internal project — Zoho Corp.
