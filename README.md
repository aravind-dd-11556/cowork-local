# Cowork Agent

A modular AI agent framework inspired by Anthropic's Cowork mode — built from the ground up with configurable LLM providers, a rich tool ecosystem, multi-agent orchestration, and an interactive CLI.

**721 tests** | **84 files** | **9 development sprints** | **3 LLM providers**

---

## Features

### Core Agent Loop
- Plan → Call LLM → Parse tool calls → Execute → Safety check → Repeat
- Context window management with automatic summarization
- Token tracking with budget enforcement
- Configurable max iterations and stop conditions

### LLM Providers
- **Ollama** — Free, local inference (default)
- **OpenAI** — GPT-4o, GPT-4o-mini, etc.
- **Anthropic** — Claude Sonnet, Haiku, Opus
- Automatic provider fallback chain with health-aware routing
- Multi-provider pool with tier-based model selection (FAST / BALANCED / POWERFUL)
- Per-provider cost tracking and budget enforcement

### Tool Ecosystem (15+ built-in tools)
- **File ops**: `bash`, `read`, `write`, `edit`, `glob`, `grep`
- **Interaction**: `todo_write`, `ask_user`
- **Web**: `web_search` (SearXNG), `web_fetch` (URL + LLM processing)
- **Advanced**: `notebook_edit`, `plan_mode`, `task` (subagent delegation), `scheduled_tasks`
- **Extensible**: MCP bridge, plugin system, skill registry

### Multi-Agent Orchestration
- Agent registry with lifecycle management
- Inter-agent pub/sub messaging via context bus
- Supervisor-driven task delegation
- Conflict detection and resolution across concurrent agents

### Safety & Security
- Input/output safety validation (prompt injection, secret detection, dangerous commands)
- Output sanitization (auto-masks API keys, tokens, credentials)
- Tool permission profiles with allow/deny lists and rate quotas
- SSRF protection on web fetch
- Workspace-scoped file operations

### Observability
- Execution tracing with hierarchical spans
- Per-tool metrics (latency percentiles, success rates, call counts)
- Session usage analytics with efficiency scoring
- Rich CLI output with ANSI formatting

### Developer Experience
- Automated code review via pre-commit hook (AST analysis, secret scanning, quality checks)
- Skill and plugin system for extensibility
- Plan mode for complex task decomposition
- Conversation persistence (save/load/list sessions)

---

## Quick Start

### Prerequisites

- Python 3.10+
- One LLM backend:
  - **Ollama** (default, free, local) — [install guide](https://ollama.com/download)
  - **OpenAI API** — requires `OPENAI_API_KEY`
  - **Anthropic API** — requires `ANTHROPIC_API_KEY`

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

### Run

```bash
python -m cowork_agent

# With options
python -m cowork_agent --provider openai --model gpt-4o --workspace ~/my-project
python -m cowork_agent -v        # verbose logging
python -m cowork_agent -vv       # debug logging
python -m cowork_agent -c my_config.yaml
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/tools` | List all registered tools |
| `/status` | Show agent status (provider, tokens, etc.) |
| `/metrics` | Show per-tool performance metrics |
| `/analytics` | Show session usage analytics and cost breakdown |
| `/quit` | Exit the agent |
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
├── core/
│   ├── agent.py               # Main agent loop
│   ├── models.py              # Data models (Message, ToolCall, ToolResult)
│   ├── tool_registry.py       # Tool registration and dispatch
│   ├── prompt_builder.py      # System prompt assembly
│   ├── context_manager.py     # Conversation context + token windowing
│   ├── safety_checker.py      # Input/output safety validation
│   ├── output_sanitizer.py    # Secret masking in tool outputs
│   ├── token_tracker.py       # Token usage tracking + budgets
│   ├── cost_tracker.py        # Per-model cost tracking
│   ├── model_router.py        # Task-aware model tier routing
│   ├── provider_pool.py       # Multi-provider management
│   ├── provider_fallback.py   # Automatic provider failover
│   ├── provider_health_tracker.py  # EWMA-based provider health scoring
│   ├── metrics_collector.py   # Per-tool latency/success metrics
│   ├── execution_tracer.py    # Hierarchical execution tracing
│   ├── usage_analytics.py     # Session analytics + recommendations
│   ├── tool_permissions.py    # Tool allow/deny + rate limits
│   ├── plan_mode.py           # Plan mode state machine
│   ├── session_manager.py     # Conversation persistence
│   ├── skill_registry.py      # Skill discovery and loading
│   ├── plugin_system.py       # Plugin loading
│   ├── supervisor.py          # Multi-agent orchestration
│   ├── context_bus.py         # Inter-agent messaging
│   ├── conflict_resolver.py   # Multi-agent conflict resolution
│   └── providers/
│       ├── base.py            # BaseLLMProvider interface
│       ├── ollama.py          # Ollama provider
│       ├── openai_provider.py # OpenAI provider
│       └── anthropic_provider.py  # Anthropic provider
├── tools/                     # 15+ built-in tools
│   ├── bash.py, read.py, write.py, edit.py
│   ├── glob_tool.py, grep_tool.py
│   ├── todo.py, ask_user.py
│   ├── web_search.py, web_fetch.py
│   ├── notebook_edit.py, plan_tools.py
│   └── task_tool.py, scheduler_tools.py
├── interfaces/
│   ├── cli.py                 # Interactive terminal interface
│   └── rich_output.py         # ANSI-formatted output helpers
├── prompts/
│   └── behavioral_rules.py    # System prompt behavioral rules
├── vendor/
│   └── claude_web_tools/      # Vendored web search/fetch backend
└── tests/                     # 721 tests across 13 suites
    ├── test_p1.py             # Sprint 1: Streaming, circuit breaker, retry (66)
    ├── test_p2.py             # Sprint 2: Skills, plugins, MCP, scheduler (56)
    ├── test_p3.py             # Sprint 3: Ask user, notebook, worktree (49)
    ├── test_p4.py             # Sprint 4: Tokens, cache, fallback, multimodal (61)
    ├── test_p4_edge.py        # Sprint 4: Edge cases (43)
    ├── test_p5.py             # Sprint 5: Multi-agent orchestration (83)
    ├── test_p6.py             # Sprint 6: Session, health, snapshot, logging (70)
    ├── test_p7.py             # Sprint 7: Hybrid cache, error catalog, context bus (72)
    ├── test_p8.py             # Sprint 8: Sanitizer, metrics, tracer, permissions (85)
    ├── test_p9.py             # Sprint 9: Model router, cost, health, pool, analytics (70)
    ├── test_qa_audit.py       # QA audit (42)
    └── test_security_audit.py # Security audit (43)
```

---

## Running Tests

```bash
# All P2-P9 tests via pytest (655 tests)
python -m pytest cowork_agent/tests/ --ignore=cowork_agent/tests/test_p1.py -q

# Sprint 1 tests (custom runner, 66 tests)
python cowork_agent/tests/test_p1.py

# A specific suite
python -m pytest cowork_agent/tests/test_p9.py -v

# Full regression: 721 tests total
python cowork_agent/tests/test_p1.py && python -m pytest cowork_agent/tests/ --ignore=cowork_agent/tests/test_p1.py -q
```

---

## Configuration

All settings can be set via `default_config.yaml`, a custom YAML file (`-c`), or environment variables:

| Env Variable | Config Key | Default | Description |
|---|---|---|---|
| `COWORK_LLM_PROVIDER` | `llm.provider` | `ollama` | Provider: ollama, openai, anthropic |
| `COWORK_LLM_MODEL` | `llm.model` | `qwen3-vl:235b-instruct-cloud` | Model name |
| `COWORK_LLM_TEMPERATURE` | `llm.temperature` | `0.7` | Sampling temperature |
| `OLLAMA_BASE_URL` | `providers.ollama.base_url` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_API_KEY` | `providers.openai.api_key` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | `providers.anthropic.api_key` | — | Anthropic API key |
| `COWORK_WORKSPACE` | `agent.workspace_dir` | `./workspace` | Working directory |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        CLI Interface                         │
│  (Interactive chat, /commands, tool output, todo widget)     │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│                       Agent Loop                             │
│  PromptBuilder → LLM Provider → Tool Execution → Safety     │
│  ContextManager ← TokenTracker ← MetricsCollector            │
│  CostTracker ← ModelRouter ← ProviderPool                   │
└──────┬──────────────┬──────────────┬─────────────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────────────────────────┐
│ LLM Provider│ │   Tools    │ │   Multi-Agent Orchestration   │
│  Ollama     │ │ bash, read │ │  Supervisor, ContextBus,      │
│  OpenAI     │ │ write,edit │ │  AgentRegistry, Delegation,   │
│  Anthropic  │ │ web, plan  │ │  ConflictResolver             │
└─────────────┘ └────────────┘ └───────────────────────────────┘
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
| 7 | Hybrid cache, error catalog, context bus enhancements | 72 |
| 8 | Output sanitizer, metrics, execution tracer, permissions, rich output | 85 |
| 9 | Model router, cost tracker, provider health, provider pool, analytics | 70 |
| QA | Quality assurance audit | 42 |
| Security | Security audit and hardening | 43 |
| **Total** | | **721** |

---

## Pre-commit Code Review

The repo includes an automated code review hook that runs on every commit:

- **AST-based analysis**: Missing docstrings, mutable defaults, bare excepts, long functions
- **Secret scanning**: AWS keys, API tokens, GitHub PATs, Slack tokens, private keys, DB URIs
- **Dangerous pattern detection**: eval(), exec(), pickle.load(), subprocess shell=True
- **Test coverage check**: Flags new modules without corresponding test files

Critical/high severity issues block the commit. Medium/low issues are reported but non-blocking.

---

## License

Internal project — Zoho Corp.
