# Cowork Agent

A modular AI agent framework inspired by Anthropic's Cowork mode — built from the ground up with configurable LLM providers, a rich tool ecosystem, multi-agent orchestration, browser automation, CRM integration, and an interactive CLI.

**4,916 tests** | **62 tools** | **45 development sprints** | **4 LLM providers** | **5 interface modes**

---

## Features

### Core Agent Loop
- Plan → Call LLM → Parse tool calls → Execute → Safety check → Repeat
- Context window management with automatic summarization and knowledge scoring
- Token tracking with budget enforcement and cost optimization
- Configurable max iterations and stop conditions
- Adaptive tool chaining with automatic retry and fallback
- Reflection engine for learning from past executions

### LLM Providers
- **Ollama** — Free, local inference (default). Pull, list, and manage models directly
- **OpenAI** — GPT-4o, GPT-4o-mini, o1, o3-mini, etc.
- **Anthropic** — Claude Opus 4.5, Sonnet 4.5, Haiku 4.5, and legacy models
- **OpenRouter** — Access 200+ models from all major providers via a single API
- **Interactive Model Selector** — `/model` command to discover, list, test, and switch models at runtime
- Automatic provider fallback chain with health-aware routing
- Multi-provider pool with tier-based model selection (FAST / BALANCED / POWERFUL)
- Per-provider cost tracking and budget enforcement

### Tool Ecosystem (62 built-in tools)
- **File ops**: `bash`, `read`, `write`, `edit`, `glob`, `grep`, `file_management`
- **Interaction**: `todo_write`, `ask_user`
- **Web**: `web_search` (SearXNG), `web_fetch` (URL + LLM processing)
- **Browser**: `browser_click`, `browser_navigate`, `browser_screenshot`, `browser_type`, `browser_scroll`, `browser_read_page`, `browser_find`, `browser_form_input`, `browser_javascript`, plus extended actions (tabs, cookies, network, console, drag-drop)
- **Git**: `git_status`, `git_diff`, `git_commit`, `git_branch`, `git_log`
- **CRM**: Zoho CRM integration (search, create, update, delete records; manage tags and territories)
- **Advanced**: `notebook_edit`, `plan_mode`, `task` (subagent delegation with agent types), `scheduled_tasks`, `chain` (adaptive tool chaining), `skill`, `generate_tool`
- **Memory**: `memory_store`, `memory_retrieve`, `memory_search`
- **Extensible**: MCP bridge, connector registry, plugin marketplace, skill registry

### Browser Automation (Sprint 33–35)
- Full browser session management with Playwright integration
- Page navigation, clicking, typing, scrolling, screenshots
- DOM reading with accessibility tree and element finding
- JavaScript execution in page context
- Form filling and file upload
- Network request monitoring and console message reading
- Tab management (create, switch, close)
- Cookie management and drag-and-drop support

### Multi-Agent Orchestration
- Agent registry with lifecycle management
- Inter-agent pub/sub messaging via context bus
- Supervisor strategies: MapReduce, Debate, Voting, Sequential, Parallel
- Agent specialization with keyword matching and intelligent routing
- Auto-scaling agent pool with health-aware selection
- Conflict detection and resolution across concurrent agents

### Connector Authentication & Security (Sprint 44–45)
- Connector authentication framework with OAuth2 PKCE, API token, and environment variable flows
- Credential store with Fernet encryption (AES-128-CBC), file permissions (0o600), secure deletion
- Path traversal prevention via strict UUID sanitization
- Token validation (min/max length, whitespace stripping) and secure masking
- OAuth state management with TTL expiry (600s) and max-pending limits (20)
- Log redaction — no token values ever logged
- CLI credential protection with getpass and readline history scrubbing
- Legacy credential auto-upgrade (base64 → Fernet encryption)

### Security & Safety (Sprint 23–25, 37)
- Anthropic-grade security pipeline with 7-layer validation
- Input sanitization (SQL injection, command injection, path traversal)
- Prompt injection detection with confidence scoring
- Credential detection with automatic redaction
- Instruction detector for untrusted content verification
- Consent management and trust context tracking
- Security freeze capability and invariant enforcement
- Privacy guard for sensitive data protection
- Tool permission profiles with allow/deny lists and rate quotas
- SSRF protection on web fetch
- Workspace-scoped file operations
- Faithful system prompt with behavioral rules matching Anthropic's standards

### Observability & Persistence
- Execution tracing with hierarchical spans and correlation IDs
- Per-tool metrics (latency percentiles, success rates, call counts)
- SQLite persistence for metrics, audit logs, and benchmarks
- Session usage analytics with efficiency scoring
- Observability event bus with typed pub/sub
- Performance benchmarking with regression detection
- Rich CLI output with ANSI formatting

### Remote Control Interfaces (Sprint 10)
- **REST API** — FastAPI server with session management, SSE streaming, tool listing
- **WebSocket** — Real-time bidirectional chat with JSON protocol
- **Web Dashboard** — Browser-based chat UI with dark/light theme, observability dashboard
- **Telegram Bot** — Per-user sessions, inline keyboards for ask_user, message splitting
- **Slack Bot** — Socket Mode, Block Kit buttons, progressive streaming, live tool status
- **`/remote-control` CLI command** — Start/stop any interface live from the running CLI

![Remote Control Demo](https://bucketest-development.zohostratus.com/RemoteControl-Cowork.gif)

### Developer Experience
- Automated code review via pre-commit hook (AST analysis, secret scanning, quality checks)
- Skill system with discovery, loading, and enforcement
- Plugin marketplace with connector registry
- Plan mode for complex task decomposition
- Conversation persistence (save/load/list sessions)
- Rollback journal for safe undo of tool executions
- Tool output validation with schema checking
- Artifact system for managing generated files

---

## Quick Start

### Prerequisites

- Python 3.10+
- One LLM backend:
  - **Ollama** (default, free, local) — [install guide](https://ollama.com/download)
  - **OpenAI API** — requires `OPENAI_API_KEY`
  - **Anthropic API** — requires `ANTHROPIC_API_KEY`
  - **OpenRouter API** — requires `OPENROUTER_API_KEY` (access 200+ models)

### Install

```bash
cd cowork_agent
pip install -r requirements.txt
```

### Configure

**Ollama (default, no API key needed):**
```bash
ollama pull llama3.1
```

**OpenAI:**
```bash
export COWORK_LLM_PROVIDER="openai"
export OPENAI_API_KEY="sk-..."
```

**Anthropic:**
```bash
export COWORK_LLM_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-..."
```

**OpenRouter (200+ models via one API):**
```bash
export COWORK_LLM_PROVIDER="openrouter"
export OPENROUTER_API_KEY="sk-or-..."
export COWORK_LLM_MODEL="anthropic/claude-sonnet-4"
```

### Run

```bash
python -m cowork_agent

# With options
python -m cowork_agent --provider openai --model gpt-4o --workspace ~/my-project
python -m cowork_agent -v        # verbose logging
python -m cowork_agent -vv       # debug logging
python -m cowork_agent -c my_config.yaml

# Model discovery (no interactive session needed)
python -m cowork_agent --list-models              # List all provider models
python -m cowork_agent --list-models openai        # List OpenAI models
python -m cowork_agent --test-model anthropic claude-sonnet-4-5-20250929
python -m cowork_agent --model-status              # Check provider availability
```

### Remote Control Modes

**REST API + Web Dashboard:**
```bash
python -m cowork_agent --mode api --api-port 8000
# Dashboard: http://localhost:8000/
# Observability: http://localhost:8000/dashboard
```

**Telegram Bot:**
```bash
export TELEGRAM_BOT_TOKEN="your-bot-token"
python -m cowork_agent --mode telegram
```

**Slack Bot:**
```bash
export SLACK_BOT_TOKEN="xoxb-your-bot-token"
export SLACK_APP_TOKEN="xapp-your-app-token"
python -m cowork_agent --mode slack
```

**All services at once:**
```bash
python -m cowork_agent --mode all --api-port 8000
```

**Or start from within the CLI:**
```bash
python -m cowork_agent    # starts in CLI mode
# Then type: /rc start slack
# Or:        /rc start api 9000
# Or:        /rc start all
```

---

### Slack Bot Setup

1. Create a Slack app at [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From scratch**
2. **Socket Mode** → Toggle **ON** → Generate an App-Level Token with `connections:write` scope → copy it as `SLACK_APP_TOKEN`
3. **OAuth & Permissions** → Add Bot Token Scopes: `chat:write`, `app_mentions:read`, `im:history`, `im:read`
4. **Event Subscriptions** → Toggle **ON** → Subscribe to bot events: `message.im`, `app_mention`
5. **App Home** → Enable **Messages Tab** → Check **Allow users to send messages**
6. **Install App** → Install to Workspace → Copy Bot User OAuth Token as `SLACK_BOT_TOKEN`

### Telegram Bot Setup

1. Message [@BotFather](https://t.me/BotFather) on Telegram → `/newbot` → follow the prompts
2. Copy the bot token as `TELEGRAM_BOT_TOKEN`
3. Optionally send `/setprivacy` → **Disable** (so the bot can read group messages)

---

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
| `/snapshot` | Create state snapshot |
| `/metrics` | Show per-tool performance metrics |
| `/analytics` | Show session usage analytics and cost breakdown |
| `/model status` | Show provider availability (API keys, Ollama status) |
| `/model current` | Show the active provider and model |
| `/model list [provider]` | List available models from a provider |
| `/model select [provider]` | Interactive numbered model selection |
| `/model test <provider> <model>` | Test a model connection with latency check |
| `/model use <provider> <model>` | Switch to a different model at runtime |
| `/model popular [provider]` | Show recommended models per provider |
| `/rc status` | Show running remote services |
| `/rc start api` | Start REST API + WebSocket server (default port 8000) |
| `/rc start telegram` | Start Telegram bot |
| `/rc start slack` | Start Slack bot |
| `/rc start all` | Start all available remote services |
| `/rc stop <service>` | Stop a running remote service |
| `/exit` | Exit the agent (stops all remote services) |
| `Ctrl+C` | Cancel current operation |

---

## Project Structure

```
cowork_agent/
├── __main__.py                # Entry point: python -m cowork_agent
├── main.py                    # CLI arg parsing, provider/tool setup, launch
├── config/
│   ├── default_config.yaml    # Default configuration
│   └── settings.py            # Config loader (YAML + env vars)
├── core/                      # 95+ modules
│   ├── agent.py               # Main agent loop
│   ├── models.py              # Data models (Message, ToolCall, ToolResult)
│   ├── model_selector.py      # Interactive model discovery & selection
│   ├── tool_registry.py       # Tool registration and dispatch
│   ├── adaptive_chain.py      # Adaptive tool chaining with retry/fallback
│   ├── reflection_engine.py   # Learning from past executions
│   ├── rollback_journal.py    # Safe undo for tool executions
│   ├── browser_session.py     # Browser automation session manager
│   ├── artifact_system.py     # Generated file management
│   ├── security_pipeline.py   # 7-layer security validation
│   ├── connector_registry.py  # MCP connector/plugin catalog
│   └── providers/
│       ├── base.py            # BaseLLMProvider interface + factory
│       ├── ollama.py          # Ollama provider (local LLMs)
│       ├── openai_provider.py # OpenAI provider
│       ├── anthropic_provider.py  # Anthropic provider
│       └── openrouter_provider.py # OpenRouter provider (200+ models)
├── tools/                     # 30 tool files → 62 registered tools
│   ├── bash.py, read.py, write.py, edit.py
│   ├── glob_tool.py, grep_tool.py
│   ├── todo.py, ask_user.py
│   ├── web_search.py, web_fetch.py
│   ├── browser_tools.py, browser_tools_ext.py, browser_tools_extra.py
│   ├── git_tools.py, crm_tools.py
│   ├── notebook_edit.py, plan_tools.py
│   ├── task_tool.py, scheduler_tools.py, scheduler_tools_ext.py
│   ├── chain_tool.py, skill_tool.py, generate_tool.py
│   ├── memory_tool.py, mcp_bridge.py, mcp_registry_tools.py
│   └── file_management_tools.py, worktree_tool.py
├── interfaces/
│   ├── base.py                # BaseInterface ABC
│   ├── cli.py                 # Interactive terminal + /model + /rc commands
│   ├── api.py                 # REST API + WebSocket (FastAPI)
│   ├── telegram_bot.py        # Telegram bot adapter
│   ├── slack_bot.py           # Slack bot adapter
│   ├── rich_output.py         # ANSI-formatted output helpers
│   └── web/
│       ├── dashboard.html     # Browser-based chat UI
│       └── observability_dashboard.html  # Metrics/health dashboard
├── prompts/
│   └── behavioral_rules.py    # Anthropic-faithful system prompt rules
├── vendor/
│   └── claude_web_tools/      # Vendored web search/fetch backend
├── sandbox/
│   ├── Containerfile          # Podman/Docker container image
│   ├── compose.yml            # Compose stack (Ollama + SearXNG + Agent)
│   └── searxng-config/        # SearXNG search engine config
└── tests/                     # 4,727 tests across 43 sprints
    ├── test_p1.py → test_p38_model_selector.py
    ├── test_p37_edge.py
    ├── test_qa_audit.py
    └── test_security_audit.py
```

---

## Running Tests

```bash
# All tests via pytest (4,661 tests)
python -m pytest cowork_agent/tests/ --ignore=cowork_agent/tests/test_p1.py -q

# Sprint 1 tests (custom runner, 66 tests)
python cowork_agent/tests/test_p1.py

# A specific suite
python -m pytest cowork_agent/tests/test_p43_crew_mode.py -v

# Full regression (all 4,727 tests)
python cowork_agent/tests/test_p1.py && python -m pytest cowork_agent/tests/ --ignore=cowork_agent/tests/test_p1.py -q
```

---

## Configuration

All settings can be set via `default_config.yaml`, a custom YAML file (`-c`), or environment variables:

| Env Variable | Config Key | Default | Description |
|---|---|---|---|
| `COWORK_LLM_PROVIDER` | `llm.provider` | `ollama` | Provider: ollama, openai, anthropic, openrouter |
| `COWORK_LLM_MODEL` | `llm.model` | `qwen3-vl:235b-instruct-cloud` | Model name |
| `COWORK_LLM_TEMPERATURE` | `llm.temperature` | `0.7` | Sampling temperature |
| `OLLAMA_BASE_URL` | `providers.ollama.base_url` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_API_KEY` | `providers.openai.api_key` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | `providers.anthropic.api_key` | — | Anthropic API key |
| `OPENROUTER_API_KEY` | `providers.openrouter.api_key` | — | OpenRouter API key |
| `COWORK_WORKSPACE` | `agent.workspace_dir` | `./workspace` | Working directory |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Interfaces Layer                              │
│  CLI (terminal) │ REST API (FastAPI) │ WebSocket │ Telegram │ Slack  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│                          Agent Loop                                   │
│  PromptBuilder → LLM Provider → Parse Tool Calls → Execute Tools     │
│  → Safety Pipeline → Context Management → Token Tracking → Repeat    │
│  Adaptive Chain Executor │ Reflection Engine │ Rollback Journal       │
└──┬──────────┬──────────┬──────────┬──────────┬───────────────────────┘
   │          │          │          │          │
┌──▼────┐ ┌──▼──────┐ ┌─▼────────┐│  ┌───────▼──────────────────────┐
│  LLM  │ │  Tool   │ │ Security ││  │   Multi-Agent Orchestration  │
│Providers│ │Registry │ │ Pipeline ││  │  Supervisor, AgentPool,      │
│Ollama/ │ │ 62     │ │7-layer   ││  │  Strategies (MapReduce,      │
│OpenAI/ │ │ tools  │ │validation││  │  Debate, Voting), Router,    │
│Anthropic│ │+browser│ │+privacy  ││  │  Specialization, ContextBus  │
│OpenRouter│ │+CRM   │ │guard     ││  │                              │
└─────────┘ └────────┘ └──────────┘│  └──────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────┐
│                      Supporting Subsystems                            │
│  Error Recovery (circuit breaker, error budget, aggregator)          │
│  Observability (event bus, metrics, benchmarks, correlation IDs)     │
│  Persistence (SQLite store, session manager, knowledge store)        │
│  Streaming (events, cancellation, progress bars)                     │
│  Browser Automation (Playwright sessions, DOM, screenshots)          │
│  Web Dashboard (real-time metrics, health, audit, benchmarks)        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Development Sprints

| Sprint | Focus | Tests |
|--------|-------|-------|
| 1 | Core agent loop, streaming, circuit breaker, retry | 66 |
| 2 | Skills, plugins, MCP client, scheduler | 56 |
| 3 | Ask user, notebook edit, worktree, providers | 49 |
| 4 | Token tracking, response cache, fallback, multimodal | 104 |
| 5 | Multi-agent orchestration, delegation, conflict resolution | 83 |
| 6 | Session management, health monitor, state snapshots, logging | 70 |
| 7 | Hybrid cache, error catalog, context bus enhancements | 70 |
| 8 | Output sanitizer, metrics, execution tracer, permissions, rich output | 68 |
| 9 | Model router, cost tracker, provider health, provider pool, analytics | 70 |
| 10 | Remote control: REST API, WebSocket, Telegram, Slack, /rc command | 140 |
| 11 | Advanced memory system: summarizer, knowledge store, memory tool | 100 |
| 12 | Full module integration wiring | 150 |
| 13 | Error recovery: circuit breaker, error budget, aggregator, orchestrator | 120 |
| 14 | Streaming events, cancellation, tool progress | 120 |
| 15 | Prompt optimization, token estimation, budget management | 120 |
| 16 | Observability: event bus, metrics registry, benchmarks, health | 150 |
| 17 | Security: input sanitizer, prompt injection, credential detection, sandbox | 160 |
| 18 | Git integration: operations, file locks, workspace context | 120 |
| 19 | Persistent storage: SQLite, metrics, audit, benchmarks | 137 |
| 20 | Web dashboard: observability UI, live updates | 96 |
| 21 | Multi-agent: strategies, specialization, routing, auto-scaling pool | 104 |
| 22 | End-to-end integration tests | 192 |
| 23 | Anthropic-grade security pipeline (7-layer validation) | 166 |
| 24 | Production hardening (action classifier, approval workflow) | 123 |
| 25 | Immutable security hardening + scheduler activation | 120 |
| 26 | Module wiring completion + Sprint 21 integration | 50 |
| 27 | Tier 2 features: reflection engine, cost optimizer, rollback journal | 113 |
| 28 | Adaptive tool chaining with retry/fallback/adaptation | 150 |
| 29 | Skill tool + skill content + skill-before-work enforcement | 166 |
| 30 | Task tool agent types, worktree isolation, resume | 85 |
| 31 | MCP registry + plugin marketplace + connector catalog | 92 |
| 32 | File management tools + artifact system | 76 |
| 33 | Browser automation core (Playwright integration) | 139 |
| 34 | Browser automation extended (tabs, cookies, network) | 76 |
| 35 | Browser automation extras (drag-drop, console, advanced) | 53 |
| 36 | CRM integration tools (Zoho CRM) | 72 |
| 37 | Faithful system prompt: behavioral rules + prompt builder | 333 |
| 38 | Interactive model selector: multi-provider discovery & switching | 71 |
| 39 | Web search (SearXNG) & fetch (two-stage pipeline), Podman sandbox | 100 |
| 40 | Self-healing pipelines: failure diagnosis, recovery strategies, auto-rollback | 100 |
| 41 | Cross-session task continuity: pause/resume, task queue, checkpoints | 100 |
| 42 | Live workspace awareness: file watcher, analyzer, suggestions, git monitor | 109 |
| 43 | Multi-agent crew mode: roles, task decomposition, result aggregation | 95 |
| 44 | Connector authentication: OAuth2 PKCE, API token, env var flows | 86 |
| 45 | Security audit: Fernet encryption, path traversal, token validation | 103 |
| QA | Quality assurance audit | 42 |
| Security | Security audit and hardening | 43 |
| **Total** | | **4,916** |

---

## Pre-commit Code Review

The repo includes an automated code review hook that runs on every commit:

- **AST-based analysis**: Missing docstrings, mutable defaults, bare excepts, long functions
- **Secret scanning**: AWS keys, API tokens, GitHub PATs, Slack tokens, private keys, DB URIs
- **Dangerous pattern detection**: eval(), exec(), pickle.load(), subprocess shell=True
- **Test coverage check**: Flags new modules without corresponding test files

Critical/high severity issues block the commit. Medium/low issues are reported but non-blocking.
