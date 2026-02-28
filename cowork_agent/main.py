"""
Main entry point — parse args, load config, initialize everything, launch CLI.
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import sys
import os

from .config.settings import load_config
from .core.providers.base import ProviderFactory
from .core.tool_registry import ToolRegistry
from .core.prompt_builder import PromptBuilder
from .core.agent import Agent
from .core.structured_logger import setup_structured_logging
from .core.health_monitor import HealthMonitor
from .core.shutdown_manager import ShutdownManager
from .core.agent_session import AgentSessionManager, SessionConfig
from .core.conversation_store import ConversationStore
from .core.state_snapshot import StateSnapshotManager
from .core.metrics_collector import MetricsCollector
from .core.output_sanitizer import OutputSanitizer
from .core.tool_permissions import ToolPermissionManager
from .core.cost_tracker import CostTracker
from .core.provider_health_tracker import ProviderHealthTracker
from .core.provider_pool import ProviderPool
from .core.model_router import ModelRouter, ModelTier, TierConfig
from .core.usage_analytics import UsageAnalytics
from .interfaces.cli import CLI

# Register providers
from .core.providers.ollama import OllamaProvider
from .core.providers.openai_provider import OpenAIProvider
from .core.providers.anthropic_provider import AnthropicProvider
from .core.providers.openrouter_provider import OpenRouterProvider

ProviderFactory.register("ollama", OllamaProvider)
ProviderFactory.register("openai", OpenAIProvider)
ProviderFactory.register("anthropic", AnthropicProvider)
ProviderFactory.register("openrouter", OpenRouterProvider)


def setup_logging(level: str = "WARNING") -> None:
    """Configure logging based on verbosity."""
    numeric = getattr(logging, level.upper(), logging.WARNING)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def register_tools(registry: ToolRegistry, config: dict) -> None:
    """Register all available tools."""
    workspace = config.get("agent.workspace_dir", os.getcwd())

    # Core tools
    from .tools.bash import BashTool
    from .tools.read import ReadTool
    from .tools.write import WriteTool
    from .tools.edit import EditTool
    from .tools.glob_tool import GlobTool
    from .tools.grep_tool import GrepTool
    from .tools.todo import TodoWriteTool

    registry.register(BashTool(workspace_dir=workspace))
    registry.register(ReadTool())
    registry.register(WriteTool(workspace_dir=workspace))

    edit_tool = EditTool(registry=registry)
    registry.register(edit_tool)

    registry.register(GlobTool(default_dir=workspace))
    registry.register(GrepTool(default_dir=workspace))

    # TodoWrite with persistence to .cowork/ directory
    persist_dir = os.path.join(workspace, ".cowork")
    registry.register(TodoWriteTool(persist_dir=persist_dir))

    # Interactive tool — ask user questions mid-task
    from .tools.ask_user import AskUserTool
    registry.register(AskUserTool())

    # Notebook editing tool
    from .tools.notebook_edit import NotebookEditTool
    registry.register(NotebookEditTool())

    # Web tools (optional — only if claude_web_tools is available)
    try:
        from .tools.web_search import WebSearchTool
        from .tools.web_fetch import WebFetchTool
        registry.register(WebSearchTool())
        registry.register(WebFetchTool())
        logging.getLogger(__name__).info("Web tools (WebSearch, WebFetch) registered.")
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Web tools not available: {e}. "
            "Install claude_web_tools for web search/fetch support."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cowork-agent",
        description="Cowork-like AI Agent — CLI interface",
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to config YAML file",
        default=None,
    )
    parser.add_argument(
        "-p", "--provider",
        help="LLM provider (ollama, openai, anthropic)",
        default=None,
    )
    parser.add_argument(
        "-m", "--model",
        help="Model name to use",
        default=None,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    parser.add_argument(
        "--workspace",
        help="Working directory for the agent",
        default=None,
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "api", "telegram", "slack", "all"],
        default="cli",
        help="Interface mode (default: cli)",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="REST API port (for api/all modes)",
    )
    parser.add_argument(
        "--telegram-token",
        default=os.environ.get("TELEGRAM_BOT_TOKEN"),
        help="Telegram bot token (for telegram/all modes)",
    )
    parser.add_argument(
        "--slack-token",
        default=os.environ.get("SLACK_BOT_TOKEN"),
        help="Slack bot token (for slack/all modes)",
    )
    parser.add_argument(
        "--slack-app-token",
        default=os.environ.get("SLACK_APP_TOKEN"),
        help="Slack app-level token (for slack/all modes)",
    )
    return parser.parse_args()


def resolve_workspace(config, cli_workspace: str | None) -> str:
    """
    Resolve and validate the workspace directory.

    Priority: CLI --workspace arg > config file > interactive prompt > cwd fallback.
    Creates the directory if it doesn't exist (with user confirmation).
    """
    # 1. CLI argument takes highest priority
    if cli_workspace:
        workspace = os.path.abspath(cli_workspace)
    else:
        # 2. Config value
        configured = config.get("agent.workspace_dir", "")
        if configured and configured != "./workspace":
            workspace = os.path.abspath(configured)
        else:
            # 3. Interactive prompt — ask the user to pick a workspace
            workspace = _prompt_workspace()

    # Validate / create
    if not os.path.exists(workspace):
        print(f"\n  Workspace directory does not exist: {workspace}")
        try:
            answer = input("  Create it? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer in ("", "y", "yes"):
            os.makedirs(workspace, exist_ok=True)
            print(f"  ✓ Created: {workspace}\n")
        else:
            print("  Using current directory as workspace instead.\n")
            workspace = os.getcwd()

    elif not os.path.isdir(workspace):
        print(f"\n  Warning: {workspace} exists but is not a directory.")
        print("  Using current directory as workspace instead.\n")
        workspace = os.getcwd()

    return workspace


def _prompt_workspace() -> str:
    """Ask the user to enter a workspace path interactively."""
    print("\n  No workspace directory configured.")
    print("  The workspace is where the agent reads/writes files.\n")

    try:
        path = input("  Enter workspace path (or press Enter for current dir): ").strip()
    except (EOFError, KeyboardInterrupt):
        path = ""

    if not path:
        workspace = os.getcwd()
        print(f"  → Using current directory: {workspace}\n")
    else:
        workspace = os.path.abspath(os.path.expanduser(path))

    return workspace


def main() -> None:
    args = parse_args()

    # Logging level
    if args.verbose >= 2:
        log_level = "DEBUG"
    elif args.verbose >= 1:
        log_level = "INFO"
    else:
        log_level = "WARNING"
    setup_structured_logging(level=log_level)

    logger = logging.getLogger(__name__)

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.provider:
        config.set("llm.provider", args.provider)
    if args.model:
        provider_name = config.get("llm.provider", "ollama")
        config.set(f"llm.{provider_name}.model", args.model)

    # Resolve workspace (interactive prompt if needed)
    workspace = resolve_workspace(config, args.workspace)
    config.set("agent.workspace_dir", workspace)
    logger.info(f"Workspace: {workspace}")

    # Create LLM provider
    try:
        provider = ProviderFactory.create(config._data)
    except Exception as e:
        print(f"Error creating LLM provider: {e}", file=sys.stderr)
        sys.exit(1)

    # Create health monitor and register provider
    health_monitor = HealthMonitor()
    if hasattr(provider, "health_check"):
        health_monitor.register_component("provider", provider.health_check)

    # Create shutdown manager
    shutdown_mgr = ShutdownManager()
    shutdown_mgr.register_callback(
        "health_shutdown",
        lambda: health_monitor.set_shutting_down(True),
        priority=100,
    )

    # Create tool registry and register tools
    registry = ToolRegistry()
    register_tools(registry, config)

    # Create plan manager and register plan tools
    from .core.plan_mode import PlanManager
    from .tools.plan_tools import EnterPlanModeTool, ExitPlanModeTool
    plan_manager = PlanManager(workspace_dir=workspace)
    registry.register(EnterPlanModeTool(plan_manager))
    registry.register(ExitPlanModeTool(plan_manager))

    # Create skill registry and discover skills
    from .core.skill_registry import SkillRegistry
    skill_registry = SkillRegistry(workspace_dir=workspace)
    num_skills = skill_registry.discover()
    if num_skills > 0:
        logger.info(f"Discovered {num_skills} skills: {skill_registry.skill_names}")

    # Create prompt builder with skill registry
    prompt_builder = PromptBuilder(config._data, skill_registry=skill_registry)

    # ── Sprint 11: Advanced Memory System ─────────────────────────────
    from .core.conversation_summarizer import ConversationSummarizer
    from .core.knowledge_store import KnowledgeStore
    from .tools.memory_tool import MemoryTool

    knowledge_store = KnowledgeStore(workspace_dir=workspace)
    summarizer = ConversationSummarizer()

    # Register memory tool
    registry.register(MemoryTool(knowledge_store=knowledge_store))

    logger.info(
        f"Memory system initialized: {knowledge_store.size} knowledge entries loaded"
    )

    # Create agent with safety checker, context manager, skill registry, and plan manager
    agent = Agent(
        provider=provider,
        registry=registry,
        prompt_builder=prompt_builder,
        max_iterations=config.get("agent.max_iterations", 15),
        workspace_dir=workspace,
        max_context_tokens=config.get("llm.max_tokens", 32000) * 2,  # context > output
        skill_registry=skill_registry,
        plan_manager=plan_manager,
        # Sprint 11: Memory components
        summarizer=summarizer,
        knowledge_store=knowledge_store,
    )

    # ── Sprint 7: Persistence & State ─────────────────────────────
    # Auto-session integration
    from .core.session_manager import SessionManager
    session_manager = SessionManager(workspace_dir=workspace)
    agent_session = AgentSessionManager(
        session_manager,
        SessionConfig(
            provider=config.get("llm.provider", ""),
            model=config.get(f"llm.{config.get('llm.provider', 'ollama')}.model", ""),
        ),
    )
    agent_session.initialize()

    # Conversation store (for search/export)
    conversation_store = ConversationStore(session_manager)

    # State snapshot manager
    snapshot_manager = StateSnapshotManager(workspace_dir=workspace)

    # ── Sprint 8: Cross-Theme Hardening ───────────────────────────────
    metrics_collector = MetricsCollector()
    output_sanitizer = OutputSanitizer()
    permission_manager = ToolPermissionManager(default_profile="full_access")

    # Attach to registry and agent for automatic use
    registry.metrics_collector = metrics_collector
    registry.output_sanitizer = output_sanitizer
    agent.permission_manager = permission_manager

    # ── Sprint 9: Multi-Provider Intelligence ─────────────────────────
    cost_tracker = CostTracker()
    health_tracker = ProviderHealthTracker()
    provider_pool = ProviderPool(health_tracker=health_tracker)
    provider_pool.register(
        config.get("llm.provider", "ollama"), provider, ModelTier.BALANCED,
    )
    model_router = ModelRouter(enabled=True)
    usage_analytics = UsageAnalytics(
        cost_tracker=cost_tracker,
        metrics_collector=metrics_collector,
        health_tracker=health_tracker,
    )

    # Attach to agent for automatic use
    agent.cost_tracker = cost_tracker
    agent.health_tracker = health_tracker

    # ── Sprint 12: Wire Response Cache + Stream Hardener + Retry ────────
    try:
        from .core.response_cache import ResponseCache
        rc_cfg = config.get("agent.response_cache", {})
        if rc_cfg.get("enabled", True):
            agent.response_cache = ResponseCache(
                max_size=rc_cfg.get("max_size", 100),
                ttl=rc_cfg.get("ttl", 3600),
            )
            logger.info("Response cache initialized")
    except Exception as e:
        logger.warning(f"Response cache not available: {e}")

    try:
        from .core.stream_hardener import StreamHardener
        sh_cfg = config.get("agent.stream_hardener", {})
        agent.stream_hardener = StreamHardener(
            chunk_timeout=sh_cfg.get("chunk_timeout", 30.0),
            total_timeout=sh_cfg.get("total_timeout", 300.0),
        )
        logger.info("Stream hardener initialized")
    except Exception as e:
        logger.warning(f"Stream hardener not available: {e}")

    try:
        from .core.retry import RetryExecutor
        agent.retry_executor = RetryExecutor()
        logger.info("Retry executor initialized")
    except Exception as e:
        logger.warning(f"Retry executor not available: {e}")

    # ── Sprint 13: Error Recovery & Resilience ──────────────────────────
    try:
        er_cfg = config.get("error_recovery", {})
        if er_cfg.get("enabled", True):
            from .core.provider_circuit_breaker import ProviderCircuitBreaker, CircuitBreakerConfig
            from .core.error_aggregator import ErrorAggregator
            from .core.error_recovery_orchestrator import RecoveryOrchestrator
            from .core.error_context import ErrorContextEnricher
            from .core.error_budget import ErrorBudgetTracker, ErrorBudgetConfig

            # Provider circuit breaker
            cb_cfg = er_cfg.get("circuit_breaker", {})
            cb_config = CircuitBreakerConfig(
                failure_threshold=cb_cfg.get("failure_threshold", 5),
                timeout_seconds=cb_cfg.get("timeout_seconds", 60),
                half_open_max_calls=cb_cfg.get("half_open_max_calls", 2),
            )
            provider_circuit_breaker = ProviderCircuitBreaker(config=cb_config)

            # Error aggregator
            agg_cfg = er_cfg.get("aggregator", {})
            error_aggregator = ErrorAggregator(
                window_seconds=agg_cfg.get("window_seconds", 300),
                spike_threshold=agg_cfg.get("spike_threshold", 3.0),
            )

            # Recovery orchestrator
            recovery_orchestrator = RecoveryOrchestrator()

            # Error context enricher
            error_enricher = ErrorContextEnricher()

            # Error budget tracker
            eb_cfg = er_cfg.get("error_budget", {})
            eb_config = ErrorBudgetConfig(
                max_error_rate=eb_cfg.get("max_error_rate", 0.20),
                window_seconds=eb_cfg.get("window_seconds", 300),
            )
            error_budget = ErrorBudgetTracker(config=eb_config)

            # Attach all to agent
            agent.provider_circuit_breaker = provider_circuit_breaker
            agent.error_aggregator = error_aggregator
            agent.recovery_orchestrator = recovery_orchestrator
            agent.error_enricher = error_enricher
            agent.error_budget = error_budget

            # Register aggregator health check with HealthMonitor
            health_monitor.register_component(
                "error_aggregator",
                lambda: error_aggregator.get_error_rate() < eb_config.max_error_rate,
            )

            logger.info("Error recovery & resilience modules initialized")
    except Exception as e:
        logger.warning(f"Error recovery modules not available: {e}")

    # ── Sprint 14: Streaming & Partial Output ───────────────────────────
    try:
        stream_cfg = config.get("streaming", {})
        if stream_cfg.get("events_enabled", True):
            agent._events_enabled = True
            logger.info("Streaming events enabled")

        cancel_cfg = config.get("cancellation", {})
        if cancel_cfg.get("enabled", True):
            from .core.stream_cancellation import StreamCancellationToken
            agent._cancellation_token = StreamCancellationToken()
            logger.info("Stream cancellation enabled")
    except Exception as e:
        logger.warning(f"Sprint 14 streaming/cancellation not available: {e}")

    # ── Sprint 15: Prompt Optimization & Context Management ────────────
    try:
        po_cfg = config.get("prompt_optimization", {})
        if po_cfg.get("enabled", True):
            from .core.token_estimator import ModelTokenEstimator
            from .core.prompt_budget import PromptBudgetManager

            # Initialize token estimator with model ratios
            est_cfg = po_cfg.get("token_estimator", {})
            if est_cfg.get("enabled", True):
                model_ratios = est_cfg.get("model_ratios", {})
                token_estimator = ModelTokenEstimator(config=model_ratios)
                agent.token_estimator = token_estimator

                # Update context manager to use model-aware estimator
                model_name = getattr(provider, "model", "") or config.get(
                    f"llm.{config.get('llm.provider', 'ollama')}.model", ""
                )
                agent.context_mgr.token_estimator = token_estimator
                agent.context_mgr.model = model_name
                logger.info("Token estimator initialized (model-aware)")
            else:
                token_estimator = None

            # Initialize prompt budget manager
            pb_cfg = po_cfg.get("prompt_budget", {})
            if pb_cfg.get("enabled", True) and token_estimator:
                agent.prompt_budget_manager = PromptBudgetManager(
                    max_system_prompt_tokens=pb_cfg.get("max_system_prompt_tokens", 8000),
                    estimator=token_estimator,
                    model=getattr(provider, "model", "claude"),
                )
                logger.info("Prompt budget manager initialized")

            # Configure context assembly settings
            ca_cfg = po_cfg.get("context_assembly", {})
            if ca_cfg.get("proactive_prune_threshold"):
                agent.context_mgr.PROACTIVE_PRUNE_RATIO = float(
                    ca_cfg["proactive_prune_threshold"]
                )
            if ca_cfg.get("dedup_window"):
                agent.context_mgr.DEDUP_WINDOW = int(ca_cfg["dedup_window"])
            if ca_cfg.get("summary_update_interval"):
                agent._SUMMARY_UPDATE_INTERVAL = int(ca_cfg["summary_update_interval"])

            # Configure budget warning thresholds
            warn_cfg = po_cfg.get("budget_warnings", {})
            if warn_cfg.get("enabled", True) and agent.token_tracker:
                thresholds = warn_cfg.get("thresholds", [50, 75, 90])
                for threshold in thresholds:
                    def _make_cb(pct):
                        def _cb(reached_pct, remaining):
                            logger.warning(
                                f"Token budget {reached_pct}% reached — "
                                f"remaining: {remaining.get('tokens_remaining', '?')} tokens"
                            )
                        return _cb
                    agent.token_tracker.on_threshold_reached(threshold, _make_cb(threshold))
                logger.info(f"Token budget warnings configured at {thresholds}%")

            logger.info("Sprint 15: Prompt optimization initialized")
    except Exception as e:
        logger.warning(f"Sprint 15 prompt optimization not available: {e}")

    # ── Sprint 16: Testing & Observability Hardening ─────────────────
    try:
        obs_cfg = config.get("observability", {})
        if obs_cfg.get("enabled", True):
            # Event bus
            eb_cfg = obs_cfg.get("event_bus", {})
            if eb_cfg.get("enabled", True):
                from .core.observability_event_bus import ObservabilityEventBus
                event_bus = ObservabilityEventBus(
                    max_subscribers_per_event=eb_cfg.get("max_subscribers_per_event", 100),
                    async_emit=eb_cfg.get("async_emit", False),
                    event_buffer_size=eb_cfg.get("event_buffer_size", 1000),
                )
                agent.event_bus = event_bus
                logger.info("Observability event bus initialized")
            else:
                event_bus = None

            # Correlation ID manager
            cid_cfg = obs_cfg.get("correlation_ids", {})
            if cid_cfg.get("enabled", True):
                from .core.correlation_id_manager import CorrelationIdManager
                correlation_mgr = CorrelationIdManager(
                    header_name=cid_cfg.get("header_name", "X-Correlation-ID"),
                )
                agent.correlation_manager = correlation_mgr
                logger.info("Correlation ID manager initialized")
            else:
                correlation_mgr = None

            # Metrics registry (extends existing MetricsCollector)
            mr_cfg = obs_cfg.get("metrics_registry", {})
            if mr_cfg.get("enabled", True):
                from .core.metrics_registry import MetricsRegistry
                metrics_registry = MetricsRegistry(
                    error_rate_window_seconds=mr_cfg.get("error_rate_window_seconds", 300),
                    token_usage_tracking=mr_cfg.get("token_usage_tracking", True),
                    detailed_latency_tracking=mr_cfg.get("detailed_latency_tracking", False),
                )
                agent.metrics_registry = metrics_registry
                logger.info("Metrics registry initialized")

            # Performance benchmark
            pb_cfg = obs_cfg.get("performance_benchmarking", {})
            if pb_cfg.get("enabled", True):
                from .core.performance_benchmark import PerformanceBenchmark
                benchmark = PerformanceBenchmark(
                    max_runs=pb_cfg.get("max_runs", 1000),
                )
                agent.benchmark = benchmark
                logger.info("Performance benchmark initialized")

            # Integrated health orchestrator
            ho_cfg = obs_cfg.get("health_orchestrator", {})
            if ho_cfg.get("enabled", True):
                from .core.integrated_health_orchestrator import IntegratedHealthOrchestrator
                health_orch = IntegratedHealthOrchestrator(
                    max_trend_history=ho_cfg.get("max_trend_history", 100),
                    event_bus=event_bus if eb_cfg.get("enabled", True) else None,
                    correlation_manager=correlation_mgr if cid_cfg.get("enabled", True) else None,
                )
                agent.health_orchestrator = health_orch
                logger.info("Health orchestrator initialized")

            logger.info("Sprint 16: Observability hardening initialized")
    except Exception as e:
        logger.warning(f"Sprint 16 observability not available: {e}")

    # ── Sprint 17: Security & Sandboxing ──────────────────────────────
    try:
        security_cfg = config.get("security", {})
        if security_cfg.get("enabled", True):
            # Input Sanitizer
            if security_cfg.get("input_sanitizer", {}).get("enabled", True):
                from .core.input_sanitizer import InputSanitizer
                san_cfg = security_cfg.get("input_sanitizer", {})
                agent.input_sanitizer = InputSanitizer(
                    max_input_size=san_cfg.get("max_input_size", 1_000_000),
                    sql_injection=san_cfg.get("sql_injection", True),
                    command_injection=san_cfg.get("command_injection", True),
                    template_injection=san_cfg.get("template_injection", True),
                    xpath_injection=san_cfg.get("xpath_injection", True),
                )
                logger.info("Input sanitizer initialized")

            # Prompt Injection Detector
            if security_cfg.get("prompt_injection_detector", {}).get("enabled", True):
                from .core.prompt_injection_detector import PromptInjectionDetector
                pid_cfg = security_cfg.get("prompt_injection_detector", {})
                agent.prompt_injection_detector = PromptInjectionDetector(
                    risk_threshold=pid_cfg.get("risk_threshold", 0.4),
                    enabled_categories=pid_cfg.get("enabled_categories"),
                    max_scan_length=pid_cfg.get("max_scan_length", 500_000),
                )
                logger.info("Prompt injection detector initialized")

            # Credential Detector
            if security_cfg.get("credential_detector", {}).get("enabled", True):
                from .core.credential_detector import CredentialDetector, RedactionStrategy
                cred_cfg = security_cfg.get("credential_detector", {})
                strategy_name = cred_cfg.get("strategy", "mask")
                strategy = RedactionStrategy(strategy_name)
                agent.credential_detector = CredentialDetector(
                    strategy=strategy,
                    max_scan_length=cred_cfg.get("max_scan_length", 1_000_000),
                )
                logger.info("Credential detector initialized")

            # Sandboxed Executor
            if security_cfg.get("sandboxed_executor", {}).get("enabled", True):
                from .core.sandboxed_executor import SandboxedExecutor, ResourceLimits
                se_cfg = security_cfg.get("sandboxed_executor", {})
                default_limits = ResourceLimits(
                    max_execution_time_seconds=se_cfg.get("default_timeout_seconds", 30.0),
                    max_memory_mb=se_cfg.get("default_max_memory_mb", 512),
                    max_output_size_bytes=se_cfg.get("default_max_output_bytes", 10_000_000),
                )
                agent.sandboxed_executor = SandboxedExecutor(default_limits=default_limits)
                logger.info("Sandboxed executor initialized")

            # Rate Limiter
            if security_cfg.get("rate_limiter", {}).get("enabled", True):
                from .core.rate_limiter import RateLimiter, RateLimitConfig
                rl_cfg = security_cfg.get("rate_limiter", {})
                default_rl_config = RateLimitConfig(
                    max_requests=rl_cfg.get("default_max_requests", 60),
                    window_seconds=rl_cfg.get("default_window_seconds", 60),
                    burst_limit=rl_cfg.get("default_burst_limit", 10),
                )
                rate_limiter = RateLimiter(default_config=default_rl_config)
                # Configure per-tool limits
                tool_limits = rl_cfg.get("tool_limits", {})
                for tool_name, tool_cfg in tool_limits.items():
                    rate_limiter.configure(tool_name, RateLimitConfig(
                        max_requests=tool_cfg.get("max_requests", 60),
                        window_seconds=tool_cfg.get("window_seconds", 60),
                        burst_limit=tool_cfg.get("burst_limit", 10),
                    ))
                agent.rate_limiter = rate_limiter
                logger.info("Rate limiter initialized")

            # Security Audit Log
            if security_cfg.get("audit_log", {}).get("enabled", True):
                from .core.security_audit_log import SecurityAuditLog
                al_cfg = security_cfg.get("audit_log", {})
                agent.security_audit_log = SecurityAuditLog(
                    max_events=al_cfg.get("max_events", 10000),
                )
                logger.info("Security audit log initialized")

            logger.info("Sprint 17: Security & sandboxing initialized")
    except Exception as e:
        logger.warning(f"Sprint 17 security not available: {e}")

    # ── Sprint 19: Persistent Storage ─────────────────────────────────
    try:
        ps_cfg = config.get("persistent_storage", {})
        if ps_cfg.get("enabled", True):
            from .core.persistent_store import PersistentStore

            store_path = os.path.join(workspace, ".cowork", "metrics")
            db_name = ps_cfg.get("db_name", "metrics.db")
            persistent_store = PersistentStore(base_path=store_path, db_name=db_name)
            agent.persistent_store = persistent_store

            # Upgrade metrics_registry to persistent version if available
            if agent.metrics_registry is not None:
                from .core.persistent_metrics_registry import PersistentMetricsRegistry
                mr_cfg = config.get("observability", {}).get("metrics_registry", {})
                persistent_metrics = PersistentMetricsRegistry(
                    store=persistent_store,
                    error_rate_window_seconds=mr_cfg.get("error_rate_window_seconds", 300),
                    token_usage_tracking=mr_cfg.get("token_usage_tracking", True),
                    detailed_latency_tracking=mr_cfg.get("detailed_latency_tracking", False),
                )
                agent.metrics_registry = persistent_metrics
                logger.info("Metrics registry upgraded to persistent")

            # Upgrade security_audit_log to persistent version if available
            if agent.security_audit_log is not None:
                from .core.persistent_audit_log import PersistentAuditLog
                al_cfg = config.get("security", {}).get("audit_log", {})
                persistent_audit = PersistentAuditLog(
                    store=persistent_store,
                    max_events=al_cfg.get("max_events", 10000),
                )
                agent.security_audit_log = persistent_audit
                logger.info("Security audit log upgraded to persistent")

            # Upgrade benchmark to persistent version if available
            if agent.benchmark is not None:
                from .core.persistent_benchmark import PersistentPerformanceBenchmark
                pb_cfg = config.get("observability", {}).get("performance_benchmarking", {})
                persistent_bench = PersistentPerformanceBenchmark(
                    store=persistent_store,
                    max_runs=pb_cfg.get("max_runs", 1000),
                )
                agent.benchmark = persistent_bench
                logger.info("Performance benchmark upgraded to persistent")

            # Auto-cleanup if configured
            retention_days = ps_cfg.get("retention_days", 90)
            if ps_cfg.get("auto_cleanup", True) and retention_days > 0:
                import time as _time
                cutoff = _time.time() - (retention_days * 86400)
                deleted = persistent_store.metrics.delete_before(cutoff)
                deleted += persistent_store.audit.delete_before(cutoff)
                deleted += persistent_store.benchmarks.delete_before(cutoff)
                if deleted > 0:
                    logger.info(f"Auto-cleanup: removed {deleted} old records (>{retention_days}d)")

            logger.info("Sprint 19: Persistent storage initialized")
    except Exception as e:
        logger.warning(f"Sprint 19 persistent storage not available: {e}")

    # ── Sprint 20: Web UI — Observability Dashboard ──────────────────
    dashboard_provider = None
    try:
        dash_cfg = config.get("web_dashboard", {})
        if dash_cfg.get("enabled", True):
            from .core.dashboard_data_provider import DashboardDataProvider
            dashboard_provider = DashboardDataProvider(
                metrics_registry=getattr(agent, "metrics_registry", None),
                audit_log=getattr(agent, "security_audit_log", None),
                health_orchestrator=getattr(agent, "health_orchestrator", None),
                benchmark=getattr(agent, "benchmark", None),
                persistent_store=getattr(agent, "persistent_store", None),
            )
            agent.dashboard_provider = dashboard_provider
            logger.info("Sprint 20: Dashboard data provider initialized")
    except Exception as e:
        logger.warning(f"Sprint 20 dashboard not available: {e}")

    # ── Sprint 18: Wire Git Operations, File Locks, Workspace Context ──
    try:
        if config.get("git.enabled", True):
            from .core.git_ops import GitOperations
            from .core.workspace_context import WorkspaceContext
            from .tools.git_tools import (
                GitStatusTool, GitDiffTool, GitCommitTool, GitBranchTool, GitLogTool,
            )

            git_ops = GitOperations(workspace_dir=workspace)
            workspace_context = WorkspaceContext(workspace_dir=workspace, git_ops=git_ops)
            workspace_context.refresh()

            # Register 5 git tools
            protected = config.get("git.protected_branches", ["main", "master"])
            max_log = config.get("git.max_log_entries", 50)
            registry.register(GitStatusTool(git_ops=git_ops))
            registry.register(GitDiffTool(git_ops=git_ops))
            registry.register(GitCommitTool(git_ops=git_ops))
            registry.register(GitBranchTool(git_ops=git_ops, protected_branches=set(protected)))
            registry.register(GitLogTool(git_ops=git_ops, max_entries=max_log))

            # Attach to agent for workspace awareness
            agent.workspace_context = workspace_context
            logger.info("Git operations initialized — 5 git tools registered")
    except Exception as e:
        logger.warning(f"Git operations not available: {e}")

    try:
        if config.get("concurrency.file_lock_enabled", True):
            from .core.file_lock import FileLockManager
            lock_timeout = config.get("concurrency.lock_timeout_seconds", 300)
            file_lock_manager = FileLockManager(lock_timeout=float(lock_timeout))
            agent.file_lock_manager = file_lock_manager
            logger.info("File lock manager initialized")
    except Exception as e:
        logger.warning(f"File lock manager not available: {e}")

    # ── Sprint 12: Wire Worktree Tools ────────────────────────────────
    try:
        from .tools.worktree_tool import EnterWorktreeTool, ListWorktreesTool, RemoveWorktreeTool
        enter_wt = EnterWorktreeTool(workspace_dir=workspace)
        # Late-bind BashTool reference so worktree can switch cwd
        try:
            bash_tool = registry.get_tool("bash")
            enter_wt.set_bash_tool(bash_tool)
        except KeyError:
            pass
        registry.register(enter_wt)
        registry.register(ListWorktreesTool(workspace_dir=workspace))
        registry.register(RemoveWorktreeTool(workspace_dir=workspace))
        logger.info("Worktree tools registered")
    except Exception as e:
        logger.warning(f"Worktree tools not available: {e}")

    # ── Sprint 12: Wire Task Scheduler ────────────────────────────────
    try:
        if config.get("scheduler.enabled", True):
            from .core.scheduler import TaskScheduler
            from .tools.scheduler_tools import (
                CreateScheduledTaskTool, ListScheduledTasksTool, UpdateScheduledTaskTool,
            )
            task_scheduler = TaskScheduler(workspace_dir=workspace)
            task_scheduler.load()
            registry.register(CreateScheduledTaskTool(scheduler=task_scheduler))
            registry.register(ListScheduledTasksTool(scheduler=task_scheduler))
            registry.register(UpdateScheduledTaskTool(scheduler=task_scheduler))
            logger.info("Task scheduler initialized")
    except Exception as e:
        logger.warning(f"Task scheduler not available: {e}")

    # ── Sprint 12: Wire Plugin System ─────────────────────────────────
    try:
        if config.get("plugins.enabled", True):
            from .core.plugin_system import PluginSystem
            plugin_system = PluginSystem(
                workspace_dir=workspace,
                user_plugins_dir=config.get("plugins.user_plugins_dir", ""),
            )
            loaded_plugins = plugin_system.discover_and_load()
            for info, tools in loaded_plugins:
                for tool in tools:
                    registry.register(tool)
            if loaded_plugins:
                logger.info(
                    f"Loaded {len(loaded_plugins)} plugins: {plugin_system.plugin_names}"
                )
    except Exception as e:
        logger.warning(f"Plugin system not available: {e}")

    # ── Sprint 12: Wire MCP Client (config-gated) ────────────────────
    try:
        if config.get("mcp.enabled", False):
            from .core.mcp_client import MCPClient, MCPServerConfig
            from .tools.mcp_bridge import register_mcp_tools

            mcp_servers = config.get("mcp.servers", [])
            if mcp_servers:
                server_configs = [
                    MCPServerConfig(
                        name=s.get("name", f"server_{i}"),
                        command=s.get("command", ""),
                        args=s.get("args", []),
                        env=s.get("env", {}),
                        transport=s.get("transport", "stdio"),
                        url=s.get("url", ""),
                    )
                    for i, s in enumerate(mcp_servers)
                ]
                mcp_client = MCPClient()
                for sc in server_configs:
                    mcp_client.add_server(sc)
                num_mcp = register_mcp_tools(registry, mcp_client)
                logger.info(f"Registered {num_mcp} MCP tools")
    except Exception as e:
        logger.warning(f"MCP client not available: {e}")

    # ── Sprint 12: Wire Multi-Agent System (config-gated) ─────────────
    try:
        if config.get("multi_agent.enabled", False):
            from .core.context_bus import ContextBus
            from .core.agent_registry import AgentRegistry
            from .core.supervisor import Supervisor, SupervisorConfig
            from .core.delegate_tool import DelegateTaskTool

            context_bus = ContextBus()
            agent_registry = AgentRegistry()
            supervisor = Supervisor(
                config=SupervisorConfig(),
                agent_registry=agent_registry,
                context_bus=context_bus,
            )
            registry.register(DelegateTaskTool(
                agent_registry=agent_registry,
                context_bus=context_bus,
            ))
            logger.info("Multi-agent system initialized")
    except Exception as e:
        logger.warning(f"Multi-agent system not available: {e}")

    # Register Task tool (subagent delegation)
    from .tools.task_tool import TaskTool

    def _create_subagent():
        """Factory for creating fresh subagent instances."""
        sub_registry = ToolRegistry()
        register_tools(sub_registry, config)
        sub_builder = PromptBuilder(config._data, skill_registry=skill_registry)
        return Agent(
            provider=provider,
            registry=sub_registry,
            prompt_builder=sub_builder,
            max_iterations=10,
            workspace_dir=workspace,
            max_context_tokens=config.get("llm.max_tokens", 32000) * 2,
            skill_registry=skill_registry,
        )

    registry.register(TaskTool(agent_factory=_create_subagent))

    # ── Sprint 10: Interface Mode Selection ─────────────────────────
    mode = getattr(args, "mode", "cli")

    logger.info(
        f"Starting agent with provider={config.get('llm.provider')}, "
        f"mode={mode}, tools={registry.tool_names}"
    )

    def _make_cli():
        """Build and wire the CLI interface."""
        history_file = config.get("cli.history_file", "~/.cowork_agent_history")
        cli = CLI(agent=agent, history_file=history_file,
                  health_monitor=health_monitor, shutdown_manager=shutdown_mgr,
                  agent_session=agent_session, conversation_store=conversation_store,
                  snapshot_manager=snapshot_manager,
                  usage_analytics=usage_analytics,
                  agent_factory=_create_subagent,
                  workspace=workspace)
        try:
            ask_tool = registry.get_tool("ask_user")
            ask_tool.set_input_callback(cli.ask_user_handler)
        except KeyError:
            pass
        return cli

    if mode == "cli":
        cli = _make_cli()
        try:
            asyncio.run(cli.run())
        except KeyboardInterrupt:
            print("\nGoodbye!")

    elif mode == "api":
        from .interfaces.api import RestAPIInterface
        api = RestAPIInterface(
            agent=agent,
            agent_factory=_create_subagent,
            host="0.0.0.0",
            port=getattr(args, "api_port", 8000),
            dashboard_provider=dashboard_provider,
        )
        print(f"Starting API server on http://0.0.0.0:{args.api_port}")
        print(f"Dashboard: http://localhost:{args.api_port}/")
        try:
            asyncio.run(api.run())
        except KeyboardInterrupt:
            print("\nServer stopped.")

    elif mode == "telegram":
        token = getattr(args, "telegram_token", None)
        if not token:
            print("Error: --telegram-token or TELEGRAM_BOT_TOKEN required", file=sys.stderr)
            sys.exit(1)
        from .interfaces.telegram_bot import TelegramBotInterface
        bot = TelegramBotInterface(
            agent=agent,
            token=token,
            agent_factory=_create_subagent,
            persist_path=os.path.join(workspace, ".cowork", "telegram_sessions.json"),
        )
        print("Starting Telegram bot...")
        try:
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            print("\nBot stopped.")

    elif mode == "slack":
        bot_token = getattr(args, "slack_token", None)
        app_token = getattr(args, "slack_app_token", None)
        if not bot_token or not app_token:
            print("Error: --slack-token and --slack-app-token required", file=sys.stderr)
            sys.exit(1)
        from .interfaces.slack_bot import SlackBotInterface
        bot = SlackBotInterface(
            agent=agent,
            bot_token=bot_token,
            app_token=app_token,
            agent_factory=_create_subagent,
        )
        print("Starting Slack bot...")
        try:
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            print("\nBot stopped.")

    elif mode == "all":
        from .interfaces.api import RestAPIInterface
        api = RestAPIInterface(
            agent=agent,
            agent_factory=_create_subagent,
            host="0.0.0.0",
            port=getattr(args, "api_port", 8000),
        )

        async def run_all():
            tasks = [asyncio.create_task(api.run())]
            tg_token = getattr(args, "telegram_token", None)
            if tg_token:
                from .interfaces.telegram_bot import TelegramBotInterface
                tg = TelegramBotInterface(
                    agent=agent, token=tg_token,
                    agent_factory=_create_subagent,
                    persist_path=os.path.join(workspace, ".cowork", "telegram_sessions.json"),
                )
                tasks.append(asyncio.create_task(tg.run()))
            slack_token = getattr(args, "slack_token", None)
            slack_app = getattr(args, "slack_app_token", None)
            if slack_token and slack_app:
                from .interfaces.slack_bot import SlackBotInterface
                slack = SlackBotInterface(
                    agent=agent, bot_token=slack_token,
                    app_token=slack_app, agent_factory=_create_subagent,
                )
                tasks.append(asyncio.create_task(slack.run()))
            print(f"Running API on http://0.0.0.0:{args.api_port}")
            if tg_token:
                print("Running Telegram bot")
            if slack_token:
                print("Running Slack bot")
            await asyncio.gather(*tasks)

        try:
            asyncio.run(run_all())
        except KeyboardInterrupt:
            print("\nAll services stopped.")


if __name__ == "__main__":
    main()
