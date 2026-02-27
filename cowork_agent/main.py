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

ProviderFactory.register("ollama", OllamaProvider)
ProviderFactory.register("openai", OpenAIProvider)
ProviderFactory.register("anthropic", AnthropicProvider)


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
