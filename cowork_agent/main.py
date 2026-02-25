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
    registry.register(TodoWriteTool())

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
    setup_logging(log_level)

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

    # Create tool registry and register tools
    registry = ToolRegistry()
    register_tools(registry, config)

    # Create prompt builder
    prompt_builder = PromptBuilder(config._data)

    # Create agent
    agent = Agent(
        provider=provider,
        registry=registry,
        prompt_builder=prompt_builder,
        max_iterations=config.get("agent.max_iterations", 15),
    )

    # Create and run CLI
    history_file = config.get("cli.history_file", "~/.cowork_agent_history")
    cli = CLI(agent=agent, history_file=history_file)

    logger.info(
        f"Starting agent with provider={config.get('llm.provider')}, "
        f"tools={registry.tool_names}"
    )

    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
