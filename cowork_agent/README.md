# Cowork Agent

A modular AI agent framework inspired by Anthropic's Cowork mode — built with configurable LLM providers, a rich tool ecosystem, multi-agent orchestration, and an interactive CLI. Developed across 22 sprints with 2,261 tests covering every subsystem.

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

# If using API/dashboard mode (optional)
pip install fastapi uvicorn websockets
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

**Option D — OpenRouter (access 200+ models via one API):**

```bash
export COWORK_LLM_PROVIDER="openrouter"
export OPENROUTER_API_KEY="sk-or-..."

# Use any model available on OpenRouter
export COWORK_LLM_MODEL="anthropic/claude-sonnet-4"     # or
export COWORK_LLM_MODEL="openai/gpt-4o"                  # or
export COWORK_LLM_MODEL="meta-llama/llama-3.1-70b-instruct"
```

### 4. Set Up Web Search (Before Launching the Agent)

The agent includes web search and URL fetch tools powered by SearXNG (search engine) and Ollama (content processing). These are optional — the agent works fine without them — but if you want web search capabilities, complete these steps **before** launching the agent.

```bash
# 1. Install web tool dependencies
pip install trafilatura markdownify beautifulsoup4 lxml

# 2. Start SearXNG (search backend)
cd cowork_agent/vendor/claude_web_tools
docker-compose up -d   # or: podman-compose up -d
cd ../../..

# 3. Make sure Ollama is running (used for content processing)
ollama serve

# 4. Verify both services are reachable
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

If SearXNG or Ollama are not running when the agent starts, the `web_search` and `web_fetch` tools will simply be unavailable — other tools will work normally.

### 5. Run the Agent

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

### 6. Start Chatting

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

## Web Dashboard & API Mode

The agent can run as a REST API with a real-time web dashboard for monitoring agent health, metrics, security events, and performance benchmarks.

### Starting the Dashboard

```bash
# Install API dependencies
pip install fastapi uvicorn websockets

# Launch in API mode (serves the dashboard + REST API)
python -m cowork_agent --mode api --api-port 8000

# Or run both CLI and API simultaneously
python -m cowork_agent --mode all --api-port 8000
```

### Dashboard URLs

Once running, open these in your browser:

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Main chat dashboard — interactive web UI for chatting with the agent |
| `http://localhost:8000/dashboard` | Observability dashboard — metrics, health, audit logs, benchmarks |

### REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/sessions` | Create a new chat session |
| `GET` | `/api/sessions` | List all active sessions |
| `POST` | `/api/chat/{session_id}` | Send a message to the agent |
| `POST` | `/api/chat/{session_id}/stream` | Send a message with SSE streaming |
| `POST` | `/api/chat/{session_id}/cancel` | Cancel an in-progress stream |
| `GET` | `/api/sessions/{session_id}/messages` | Get conversation history |
| `GET` | `/api/tools` | List available tools |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/dashboard/full` | Full dashboard snapshot (metrics, health, audit) |
| `GET` | `/api/dashboard/metrics` | Token usage and provider metrics |
| `GET` | `/api/dashboard/metrics/historical` | Historical metrics (days param) |
| `GET` | `/api/dashboard/health` | Component health status |
| `GET` | `/api/dashboard/audit` | Security audit event feed |
| `GET` | `/api/dashboard/benchmarks` | Performance benchmark data |
| `GET` | `/api/dashboard/store` | Persistent storage statistics |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `ws://localhost:8000/ws/{session_id}` | Real-time chat streaming |
| `ws://localhost:8000/ws/dashboard` | Live dashboard updates (metrics, events) |

### Telegram & Slack Bots

The agent also supports Telegram and Slack as interfaces:

```bash
# Telegram bot
export TELEGRAM_BOT_TOKEN="your-token"
python -m cowork_agent --mode telegram

# Slack bot
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_APP_TOKEN="xapp-..."
python -m cowork_agent --mode slack

# All interfaces at once
python -m cowork_agent --mode all --api-port 8000
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/history` | Show conversation history |
| `/todos` | Show current task list |
| `/config` | Show current configuration |
| `/health` | Show system health status |
| `/sessions` | List saved sessions |
| `/snapshot` | Create a state snapshot |
| `/snapshots` | List saved snapshots |
| `/metrics` | Show tool execution metrics |
| `/analytics` | Show usage analytics |
| `/rc start api` | Start REST API service (remote control) |
| `/rc start telegram` | Start Telegram bot |
| `/rc start slack` | Start Slack bot |
| `/rc status` | Show running services |
| `/rc stop all` | Stop all remote services |
| `/quit` | Exit the agent |
| `Ctrl+C` | Cancel current operation |

## Sprint Overview

The framework was built across 22 sprints, each adding a major subsystem. Here is what each sprint covers:

### Core Agent (Sprints 1–5)

| Sprint | Feature | Description |
|--------|---------|-------------|
| 1 | **Streaming & Resilience** | Stream hardener with timeout/reconnect, circuit breaker pattern, retry with exponential backoff |
| 2 | **Skills, Plugins & Scheduling** | Skill registry and discovery, plugin system, MCP (Model Context Protocol) client, cron-based task scheduler |
| 3 | **User Interaction Tools** | Ask-user structured questions, Jupyter notebook editing, git worktree management |
| 4 | **Token & Cache Management** | Token tracker with budget enforcement, semantic response cache (LRU + TTL), provider fallback chain, multimodal input (images/PDFs) |
| 5 | **Multi-Agent Orchestration** | Agent registry, context bus (pub/sub messaging), delegate tool, supervisor with strategies, conflict resolver |

### Production Hardening (Sprints 6–8)

| Sprint | Feature | Description |
|--------|---------|-------------|
| 6 | **Production Hardening** | Error catalog with classification and recovery hints, structured logging (JSON + human formats) with trace IDs, retry layer with transient-error filtering, health monitor with liveness/readiness probes, graceful shutdown manager with priority-ordered callbacks |
| 7 | **Persistence & State** | Agent session manager (create/resume sessions), conversation store (search, export, prune), token usage store (budgets, cost tracking, alerts), hybrid response cache (memory + disk spill), state snapshot manager (point-in-time capture and restore) |
| 8 | **Cross-Theme Hardening** | Output sanitizer (masks AWS keys, JWTs, DB URIs, passwords), metrics collector (per-tool/provider performance tracking with percentiles), execution tracer (hierarchical spans with parent-child relationships), tool permission manager (profiles: full_access, read_only, safe_mode with quotas), rich terminal output (tables, progress bars, formatted errors) |

### Intelligence & Interfaces (Sprints 9–11)

| Sprint | Feature | Description |
|--------|---------|-------------|
| 9 | **Multi-Provider Intelligence** | Model router (task complexity → tier selection: FAST/BALANCED/POWERFUL), cost tracker (per-model pricing, budget enforcement), provider health tracker (EWMA scoring, rankings), provider pool (tier-based provider management), usage analytics (routing decisions, efficiency scoring, recommendations) |
| 10 | **Remote Control Interfaces** | REST API with FastAPI (sessions, chat, streaming), WebSocket for real-time bidirectional streaming, Telegram bot adapter, Slack bot adapter with interactive buttons, remote control CLI commands (/rc start/stop/status), web dashboard HTML serving |
| 11 | **Advanced Memory System** | Conversation summarizer (rule-based compression), knowledge store (category-based persistent facts), memory tool (user-accessible store/retrieve/search), context manager with knowledge scoring and proactive pruning, memory-aware prompt building |

### System Integration (Sprint 12)

| Sprint | Feature | Description |
|--------|---------|-------------|
| 12 | **Wiring All Modules** | Full integration of response cache, stream hardener, multimodal support, notebook editing, scheduler, worktree, plugin system, MCP client, context bus, agent registry, supervisor, and delegate tool into the main agent loop |

### Resilience & Streaming (Sprints 13–14)

| Sprint | Feature | Description |
|--------|---------|-------------|
| 13 | **Error Recovery & Resilience** | Per-provider circuit breaker (CLOSED → OPEN → HALF_OPEN state machine), error aggregator (spike detection, recurring patterns, correlated failures), recovery orchestrator (strategy selection: retry, fallback, escalate, wait), error context enricher (breadcrumb trails), error budget tracker (rate governance per category) |
| 14 | **Streaming & Partial Output** | Stream events (TextChunk, ToolStart/End, ToolProgress, StatusUpdate), stream cancellation with tokens, tool progress reporting (percent + message), agent `run_stream_events` API, SSE endpoint for browser streaming, CLI event rendering with progress bars |

### Optimization & Observability (Sprints 15–16)

| Sprint | Feature | Description |
|--------|---------|-------------|
| 15 | **Prompt Optimization & Context** | Model-aware token estimator (per-model character ratios), prompt budget manager (allocation per model), intelligent context composition (scoring, dedup, pruning), token prediction and cost estimation |
| 16 | **Testing & Observability** | Observability event bus (typed pub/sub: AGENT_STARTED, TOOL_CALL_INITIATED, etc.), correlation ID manager (distributed tracing), centralized metrics registry, performance benchmark (statistical analysis with p99, regression detection), integrated health orchestrator (failure prediction, trend analysis) |

### Security & Git (Sprints 17–18)

| Sprint | Feature | Description |
|--------|---------|-------------|
| 17 | **Security & Sandboxing** | Input sanitizer (SQL injection, command injection, path traversal), prompt injection detector (role injection, instruction override, confidence scoring), credential detector (API keys, passwords, DB URIs with redaction), sandboxed executor (resource limits: memory, CPU, timeout), rate limiter (token bucket + sliding window), security audit log (event recording with severity levels) |
| 18 | **Git & Worktree Integration** | Git operations wrapper (status, commit, branch, diff, log, merge), file lock manager (reader-writer locks with expiry), workspace context (git-aware state tracking), git tools for users (status, diff, commit, branch, log), protected branch safety checks |

### Storage & Dashboard (Sprints 19–20)

| Sprint | Feature | Description |
|--------|---------|-------------|
| 19 | **Persistent Storage** | SQLite persistence layer with schema versioning, metrics table (token usage, errors, provider calls), audit table (security events), benchmark table (performance data), write-through wrappers for MetricsRegistry, AuditLog, and PerformanceBenchmark |
| 20 | **Web UI — Observability Dashboard** | DashboardDataProvider (aggregates all subsystems), REST endpoints for metrics/health/audit/benchmarks, WebSocket for live dashboard updates, HTML dashboard with provider status widgets, performance charts, token budget visualization, and security event timeline |

### Advanced Orchestration & Testing (Sprints 21–22)

| Sprint | Feature | Description |
|--------|---------|-------------|
| 21 | **Multi-Agent Enhancement** | Supervisor strategies (MapReduce, Debate, Voting, Sequential, Parallel), agent specialization with keyword matching, conversation router (task analysis and intelligent delegation), agent pool with auto-scaling and health-aware selection |
| 22 | **End-to-End Integration Tests** | 192 tests across 9 suites: basic agent loop, multi-turn context, tool pipeline, error recovery, multi-agent orchestration, streaming, security pipeline, observability, and full realistic workflow scenarios |

## Project Structure

```
cowork_agent/
├── __init__.py
├── __main__.py                     # Entry point: python -m cowork_agent
├── main.py                         # CLI arg parsing, provider/tool setup, launch
├── config/
│   ├── default_config.yaml         # Default configuration (all sprints)
│   └── settings.py                 # Config loader (YAML + env vars)
├── core/
│   ├── agent.py                    # Main Agent loop
│   ├── models.py                   # Data models (Message, ToolCall, ToolResult)
│   ├── tool_registry.py            # Tool registration and dispatch
│   ├── prompt_builder.py           # System prompt assembly
│   ├── context_manager.py          # Context window + knowledge scoring
│   ├── safety_checker.py           # Input/output safety validation
│   ├── session_manager.py          # Conversation persistence
│   ├── stream_hardener.py          # Streaming resilience (timeout, reconnect)
│   ├── token_tracker.py            # Token usage tracking and budgets
│   ├── response_cache.py           # Semantic response caching (LRU + TTL)
│   ├── provider_fallback.py        # Provider failover chain
│   ├── multimodal.py               # Image/PDF input handling
│   ├── plan_mode.py                # Plan mode state machine
│   ├── skill_registry.py           # Skill discovery and loading
│   ├── plugin_system.py            # Plugin loading from directories
│   ├── scheduler.py                # Cron-based task scheduling
│   ├── mcp_client.py               # MCP (Model Context Protocol) client
│   ├── worktree.py                 # Git worktree management
│   │
│   │  # Sprint 5 — Multi-Agent
│   ├── agent_registry.py           # Agent lifecycle management
│   ├── context_bus.py              # Inter-agent pub/sub messaging
│   ├── delegate_tool.py            # Agent-to-agent delegation
│   ├── supervisor.py               # Multi-agent orchestration
│   ├── conflict_resolver.py        # Conflict detection & resolution
│   │
│   │  # Sprint 6 — Production Hardening
│   ├── error_catalog.py            # Error classification + recovery hints
│   ├── structured_logger.py        # JSON/human logging with trace IDs
│   ├── retry.py                    # Retry layer with backoff + jitter
│   ├── health_monitor.py           # Component health + liveness probes
│   ├── shutdown_manager.py         # Graceful shutdown orchestration
│   │
│   │  # Sprint 7 — Persistence & State
│   ├── agent_session.py            # Session lifecycle (create/resume)
│   ├── conversation_store.py       # Search, export, prune conversations
│   ├── token_usage_store.py        # Token budgets, cost tracking, alerts
│   ├── hybrid_cache.py             # Memory + disk response cache
│   ├── state_snapshot.py           # Point-in-time state capture/restore
│   │
│   │  # Sprint 8 — Cross-Theme Hardening
│   ├── output_sanitizer.py         # Secret masking (AWS, JWT, DB URIs)
│   ├── metrics_collector.py        # Per-tool/provider performance metrics
│   ├── execution_tracer.py         # Hierarchical span tracing
│   ├── tool_permissions.py         # Access control profiles + quotas
│   │
│   │  # Sprint 9 — Multi-Provider Intelligence
│   ├── model_router.py             # Task complexity → model tier routing
│   ├── cost_tracker.py             # LLM API cost calculation + budgets
│   ├── provider_health_tracker.py  # EWMA health scoring + rankings
│   ├── provider_pool.py            # Tier-based provider management
│   ├── usage_analytics.py          # Session-level usage analysis
│   │
│   │  # Sprint 11 — Advanced Memory
│   ├── conversation_summarizer.py  # Intelligent conversation compression
│   ├── knowledge_store.py          # Category-based persistent facts
│   ├── memory_tool.py              # User-accessible memory store/retrieve
│   │
│   │  # Sprint 13 — Error Recovery
│   ├── provider_circuit_breaker.py # Per-provider circuit breaker
│   ├── error_aggregator.py         # Spike + pattern detection
│   ├── error_recovery_orchestrator.py # Recovery strategy selection
│   ├── error_context.py            # Error breadcrumb trails
│   ├── error_budget.py             # Error rate governance
│   │
│   │  # Sprint 14 — Streaming & Progress
│   ├── stream_events.py            # TextChunk, ToolStart/End, ToolProgress
│   ├── stream_cancellation.py      # Cancellation tokens
│   ├── tool_progress.py            # Progress reporting
│   │
│   │  # Sprint 15 — Prompt Optimization
│   ├── token_estimator.py          # Model-aware token estimation
│   ├── prompt_budget.py            # System prompt budget management
│   │
│   │  # Sprint 16 — Observability
│   ├── observability_event_bus.py  # Typed event pub/sub
│   ├── correlation_id_manager.py   # Distributed tracing IDs
│   ├── metrics_registry.py         # Centralized metrics storage
│   ├── performance_benchmark.py    # Statistical benchmarking
│   ├── integrated_health_orchestrator.py # System-wide health aggregation
│   ├── test_coverage_collector.py  # Test coverage tracking
│   │
│   │  # Sprint 17 — Security
│   ├── input_sanitizer.py          # SQL/command/path injection detection
│   ├── prompt_injection_detector.py # LLM prompt attack detection
│   ├── credential_detector.py      # API key/password detection + redaction
│   ├── sandboxed_executor.py       # Resource-limited tool execution
│   ├── rate_limiter.py             # Token bucket + sliding window
│   ├── security_audit_log.py       # Security event recording
│   │
│   │  # Sprint 18 — Git Integration
│   ├── git_ops.py                  # Git command wrapper
│   ├── file_lock.py                # Reader-writer file locks
│   ├── workspace_context.py        # Git-aware workspace state
│   │
│   │  # Sprint 19 — Persistent Storage
│   ├── persistent_store.py         # SQLite persistence layer
│   ├── persistent_metrics_registry.py # Write-through metrics persistence
│   ├── persistent_audit_log.py     # Write-through audit log persistence
│   ├── persistent_benchmark.py     # Write-through benchmark persistence
│   │
│   │  # Sprint 20 — Dashboard Data
│   ├── dashboard_data_provider.py  # Aggregated dashboard snapshots
│   │
│   │  # Sprint 21 — Orchestration Enhancement
│   ├── supervisor_strategies.py    # MapReduce, Debate, Voting, etc.
│   ├── agent_specialization.py     # Role-based agent matching
│   ├── conversation_router.py      # Intelligent task delegation
│   ├── agent_pool.py               # Auto-scaling agent pool
│   │
│   └── providers/
│       ├── base.py                 # BaseLLMProvider interface + factory
│       ├── ollama.py               # Ollama provider (local LLMs)
│       ├── openai_provider.py      # OpenAI provider
│       ├── anthropic_provider.py   # Anthropic provider
│       └── openrouter_provider.py  # OpenRouter provider (200+ models)
├── tools/
│   ├── base.py                     # BaseTool interface
│   ├── bash.py                     # Shell command execution (sandboxed)
│   ├── read.py                     # File reading
│   ├── write.py                    # File writing (workspace-scoped)
│   ├── edit.py                     # Exact string replacement editing
│   ├── glob_tool.py                # File pattern matching
│   ├── grep_tool.py                # Content search
│   ├── todo.py                     # Todo list management
│   ├── ask_user.py                 # Structured user questions
│   ├── web_fetch.py                # URL fetching + LLM processing
│   ├── web_search.py               # Web search via SearXNG
│   ├── notebook_edit.py            # Jupyter notebook editing
│   ├── plan_tools.py               # Plan mode enter/exit
│   ├── task_tool.py                # Subagent delegation
│   ├── scheduler_tools.py          # Scheduled task CRUD
│   ├── mcp_bridge.py               # MCP tool bridge
│   ├── worktree_tool.py            # Git worktree tools
│   ├── git_tools.py                # Git status/diff/commit/branch/log
│   └── memory_tool.py              # Knowledge store/retrieve/search
├── interfaces/
│   ├── base.py                     # Abstract interface contract
│   ├── cli.py                      # Interactive terminal chat
│   ├── api.py                      # REST API + WebSocket (FastAPI)
│   ├── telegram_bot.py             # Telegram bot adapter
│   ├── slack_bot.py                # Slack bot adapter
│   ├── rich_output.py              # Formatted terminal display (tables, progress bars)
│   └── web/
│       ├── dashboard.html          # Main chat web dashboard
│       └── observability_dashboard.html  # Observability dashboard
├── prompts/
│   └── behavioral_rules.py        # System prompt behavioral rules
├── vendor/
│   └── claude_web_tools/           # Web search & fetch backend
│       ├── web_search.py           # SearXNG search client
│       ├── web_fetch.py            # Two-stage URL fetcher
│       ├── html_to_markdown.py     # HTML → Markdown converter
│       ├── llm_processor.py        # Ollama content processor
│       ├── cache.py                # 15-minute TTL cache
│       ├── config.py               # Web tools configuration
│       └── models.py               # SearchResult, FetchResult models
├── tests/                          # 2,261 tests across 22 sprints
│   ├── test_p1.py                  # Sprint 1: Streaming & resilience (66)
│   ├── test_p2.py                  # Sprint 2: Skills, plugins, MCP (56)
│   ├── test_p3.py                  # Sprint 3: User interaction tools (49)
│   ├── test_p4.py                  # Sprint 4: Token & cache management (61)
│   ├── test_p4_edge.py             # Sprint 4: Edge cases (43)
│   ├── test_p5.py                  # Sprint 5: Multi-agent orchestration (83)
│   ├── test_p6.py                  # Sprint 6: Production hardening (70)
│   ├── test_p7.py                  # Sprint 7: Persistence & state (70)
│   ├── test_p8.py                  # Sprint 8: Cross-theme hardening (68)
│   ├── test_p9.py                  # Sprint 9: Multi-provider intelligence (70)
│   ├── test_p10.py                 # Sprint 10: Remote control interfaces (140)
│   ├── test_p11.py                 # Sprint 11: Advanced memory system (100)
│   ├── test_p12.py                 # Sprint 12: Module integration (150)
│   ├── test_p13.py                 # Sprint 13: Error recovery (120)
│   ├── test_p14.py                 # Sprint 14: Streaming & progress (120)
│   ├── test_p15.py                 # Sprint 15: Prompt optimization (120)
│   ├── test_p16.py                 # Sprint 16: Observability hardening (150)
│   ├── test_p17.py                 # Sprint 17: Security & sandboxing (160)
│   ├── test_p18.py                 # Sprint 18: Git & worktree (120)
│   ├── test_p19.py                 # Sprint 19: Persistent storage (137)
│   ├── test_p20.py                 # Sprint 20: Observability dashboard (96)
│   ├── test_p21.py                 # Sprint 21: Multi-agent enhancement (104)
│   ├── test_p22.py                 # Sprint 22: E2E integration tests (192)
│   ├── e2e_helpers.py              # E2E test helpers (MockLLMProvider, etc.)
│   ├── test_qa_audit.py            # QA audit tests (42)
│   └── test_security_audit.py      # Security audit tests (43)
├── requirements.txt
└── setup.py
```

## Running Tests

```bash
# From the project root (parent of cowork_agent/)

# Run all tests with pytest (recommended, 2,261 tests)
python -m pytest cowork_agent/tests/ --ignore=cowork_agent/tests/test_p1.py -q

# Run Sprint 1 tests separately (custom runner, 66 tests)
python cowork_agent/tests/test_p1.py

# Run a specific sprint
python -m pytest cowork_agent/tests/test_p22.py -v

# Run a specific test class
python -m pytest cowork_agent/tests/test_p22.py::TestE2EFullScenarios -v

# Total: 2,261 tests across 22 sprints
```

## Configuration Reference

All settings can be set via `default_config.yaml`, a custom YAML file (`-c`), or environment variables:

| Env Variable | Config Key | Default | Description |
|-------------|------------|---------|-------------|
| `COWORK_LLM_PROVIDER` | `llm.provider` | `ollama` | LLM provider: ollama, openai, anthropic, openrouter |
| `COWORK_LLM_MODEL` | `llm.model` | `qwen3-vl:235b-instruct-cloud` | Model name |
| `COWORK_LLM_TEMPERATURE` | `llm.temperature` | `0.7` | Sampling temperature |
| `OLLAMA_BASE_URL` | `providers.ollama.base_url` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_API_KEY` | `providers.openai.api_key` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | `providers.anthropic.api_key` | — | Anthropic API key |
| `OPENROUTER_API_KEY` | `providers.openrouter.api_key` | — | OpenRouter API key |
| `OPENROUTER_BASE_URL` | `providers.openrouter.base_url` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `COWORK_WORKSPACE` | `agent.workspace_dir` | `./workspace` | Working directory |

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Interfaces Layer                              │
│  CLI (terminal) │ REST API (FastAPI) │ WebSocket │ Telegram │ Slack  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│                          Agent Loop                                   │
│  PromptBuilder → LLM Provider → Parse Tool Calls → Execute Tools     │
│  → Safety Check → Context Management → Token Tracking → Repeat       │
└──┬──────────┬──────────┬──────────┬──────────┬───────────────────────┘
   │          │          │          │          │
┌──▼────┐ ┌──▼──────┐ ┌─▼────────┐│  ┌───────▼──────────────────────┐
│  LLM  │ │  Tool   │ │ Security ││  │   Multi-Agent Orchestration  │
│Providers│ │Registry │ │ Pipeline ││  │  Supervisor, AgentPool,      │
│Ollama/ │ │ 20+    │ │Sanitizer,││  │  Strategies (MapReduce,      │
│OpenAI/ │ │ tools  │ │Injection,││  │  Debate, Voting), Router,    │
│Anthropic│ │        │ │RateLimiter│  │  Specialization, ContextBus  │
└─────────┘ └────────┘ └──────────┘│  └──────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────┐
│                      Supporting Subsystems                            │
│  Error Recovery (circuit breaker, error budget, aggregator)          │
│  Observability (event bus, metrics, benchmarks, correlation IDs)     │
│  Persistence (SQLite store, session manager, knowledge store)        │
│  Streaming (events, cancellation, progress bars)                     │
│  Web Dashboard (real-time metrics, health, audit, benchmarks)        │
└──────────────────────────────────────────────────────────────────────┘
```

## License

Internal project — Zoho Corp.
