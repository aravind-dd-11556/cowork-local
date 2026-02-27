"""
CLI Interface — Interactive terminal chat with the agent.
Supports /commands, streaming display, tool execution indicators, and todo widget.
"""

from __future__ import annotations
import asyncio
import logging
import os
import sys
import signal
import readline
import threading
import time
from datetime import datetime
from typing import Optional, Callable

from ..core.agent import Agent
from ..core.models import ToolCall, ToolResult
from ..core.health_monitor import HealthMonitor
from ..core.shutdown_manager import ShutdownManager
from ..core.agent_session import AgentSessionManager
from ..core.conversation_store import ConversationStore
from ..core.state_snapshot import StateSnapshotManager
from .rich_output import RichOutput


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_DARK = "\033[48;5;236m"


# Thread-safe stdout lock — prevents spinner, streaming, tool callbacks
# and logging from interleaving on the terminal.
_stdout_lock = threading.Lock()


class Spinner:
    """A simple terminal spinner that runs in a background thread."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Thinking"):
        self._message = message
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()  # Signals spinner thread has fully exited

    def start(self, message: str = None) -> None:
        if message:
            self._message = message
        if self._running:
            return
        self._stopped.clear()
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running and self._thread is None:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        # Wait for the thread to signal it's truly done writing
        self._stopped.wait(timeout=1)
        # Clear the spinner line
        with _stdout_lock:
            sys.stdout.write(f"\r{' ' * 80}\r")
            sys.stdout.flush()

    def _spin(self) -> None:
        idx = 0
        start = time.time()
        try:
            while self._running:
                elapsed = int(time.time() - start)
                frame = self.FRAMES[idx % len(self.FRAMES)]
                with _stdout_lock:
                    if not self._running:
                        break
                    sys.stdout.write(
                        f"\r  {Colors.DIM}{frame} {self._message}... ({elapsed}s){Colors.RESET}"
                    )
                    sys.stdout.flush()
                idx += 1
                time.sleep(0.1)
        finally:
            self._stopped.set()


class CLI:
    """Interactive terminal interface for the agent."""

    def __init__(
        self,
        agent: Agent,
        history_file: str = "~/.cowork_agent_history",
        streaming: bool = True,
        health_monitor: Optional[HealthMonitor] = None,
        shutdown_manager: Optional[ShutdownManager] = None,
        agent_session: Optional[AgentSessionManager] = None,
        conversation_store: Optional[ConversationStore] = None,
        snapshot_manager: Optional[StateSnapshotManager] = None,
        usage_analytics=None,
        agent_factory: Optional[Callable] = None,
        workspace: Optional[str] = None,
    ):
        self.agent = agent
        self.history_file = os.path.expanduser(history_file)
        self._running = False
        self._spinner = Spinner()
        self._streaming_enabled = streaming
        self._health_monitor = health_monitor
        self._shutdown_manager = shutdown_manager
        self._agent_session = agent_session
        self._conversation_store = conversation_store
        self._snapshot_manager = snapshot_manager
        self._usage_analytics = usage_analytics
        self._rich = RichOutput(width=100)
        self._tool_timers: dict[str, float] = {}  # tool_id -> start_time
        self._agent_factory = agent_factory
        self._workspace = workspace or os.getcwd()

        # Remote control state
        self._remote_services: dict[str, dict] = {}  # name -> {task, started_at, info}

        # Track whether we're in a streaming response (tool callbacks
        # should NOT restart the spinner during streaming — the stream
        # loop itself drives the next LLM call).
        self._is_streaming = False

        # Wire up callbacks
        self.agent.on_tool_start = self._on_tool_start
        self.agent.on_tool_end = self._on_tool_end
        self.agent.on_status = self._on_status

    def _on_tool_start(self, call: ToolCall) -> None:
        """Display tool execution indicator."""
        self._spinner.stop()
        self._tool_timers[call.tool_id] = time.time()
        icon = self._tool_icon(call.name)
        with _stdout_lock:
            print(
                f"  {Colors.DIM}{icon} Executing {Colors.CYAN}{call.name}"
                f"{Colors.RESET}{Colors.DIM}...{Colors.RESET}"
            )

    def _on_tool_end(self, call: ToolCall, result: ToolResult) -> None:
        """Display tool result status with timing via RichOutput."""
        self._spinner.stop()
        # Calculate duration
        start = self._tool_timers.pop(call.tool_id, None)
        duration_ms = (time.time() - start) * 1000 if start else 0
        output_lines = len(result.output.split("\n")) if result.output else 0
        # Use RichOutput for formatted tool result line
        with _stdout_lock:
            print(self._rich.tool_result(call.name, result.success, duration_ms, output_lines))
            if not result.success and result.error:
                print(self._rich.error(result.error))
        # Only restart spinner if NOT in a streaming response — during
        # streaming the stream loop drives the next LLM call directly
        # and a spinner would race with the streamed text.
        if not self._is_streaming:
            self._spinner.start("Thinking")

    def _on_status(self, message: str) -> None:
        """Display agent status updates (retries, nudges, etc.)."""
        self._spinner.stop()
        with _stdout_lock:
            print(
                f"  {Colors.YELLOW}⟳ {message}{Colors.RESET}"
            )
        if not self._is_streaming:
            self._spinner.start("Retrying")

    def ask_user_handler(self, question: str, options: list[str]) -> str:
        """
        Handle AskUser tool calls — display question and get user input.
        Called from the AskUser tool via callback.
        """
        self._spinner.stop()
        with _stdout_lock:
            print(f"\n  {Colors.BOLD}{Colors.MAGENTA}❓ Agent asks:{Colors.RESET} {question}")

        if options:
            for i, opt in enumerate(options, 1):
                print(f"     {Colors.CYAN}{i}.{Colors.RESET} {opt}")
            print(f"     {Colors.DIM}(Enter number or type your own answer){Colors.RESET}")

        try:
            response = input(f"  {Colors.BOLD}Your answer ▸ {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

        # If they entered a number and we have options, resolve it
        if options and response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(options):
                response = options[idx]

        print()  # Blank line after answer
        self._spinner.start("Thinking")
        return response

    @staticmethod
    def _tool_icon(name: str) -> str:
        icons = {
            "bash": "⚡",
            "read": "📄",
            "write": "✏️",
            "edit": "🔧",
            "glob": "🔍",
            "grep": "🔎",
            "web_search": "🌐",
            "web_fetch": "🌍",
            "todo_write": "📋",
            "ask_user": "❓",
        }
        return icons.get(name, "🔨")

    def _setup_readline(self) -> None:
        """Configure readline for command history."""
        try:
            readline.set_history_length(1000)
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
        except Exception:
            pass

    def _save_readline(self) -> None:
        """Save command history."""
        try:
            readline.write_history_file(self.history_file)
        except Exception:
            pass

    def _print_banner(self) -> None:
        """Print welcome banner."""
        # Show workspace path from the bash tool's cwd
        bash_tool = self.agent.registry.get_tool("bash") if "bash" in self.agent.registry.tool_names else None
        workspace = bash_tool._cwd if bash_tool else os.getcwd()

        print(f"""
{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════╗
║         🤖 Cowork Agent v0.1.0          ║
╚══════════════════════════════════════════╝{Colors.RESET}

  {Colors.BOLD}Workspace:{Colors.RESET} {workspace}
  {Colors.DIM}Type your message and press Enter.
  Use /help for available commands.
  Press Ctrl+C to cancel, Ctrl+D to exit.{Colors.RESET}
""")

    async def _handle_command(self, cmd: str) -> bool:
        """
        Handle slash commands. Returns True if the command was handled,
        False if it should be sent to the agent.
        """
        parts = cmd.strip().split(None, 1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            self._print_help()
            return True

        elif command == "/clear":
            self.agent.clear_history()
            print(f"  {Colors.GREEN}✓ Conversation history cleared.{Colors.RESET}\n")
            return True

        elif command == "/history":
            self._show_history()
            return True

        elif command == "/todos":
            self._show_todos()
            return True

        elif command == "/config":
            self._show_config()
            return True

        elif command == "/health":
            await self._show_health()
            return True

        elif command == "/sessions":
            self._show_sessions()
            return True

        elif command == "/snapshot":
            self._create_snapshot(arg)
            return True

        elif command == "/snapshots":
            self._show_snapshots()
            return True

        elif command == "/metrics":
            self._show_metrics()
            return True

        elif command == "/analytics":
            self._show_analytics()
            return True

        elif command in ("/remote-control", "/rc"):
            await self._handle_remote_control(arg)
            return True

        elif command in ("/exit", "/quit", "/q"):
            # Stop all remote services before exiting
            if self._remote_services:
                await self._stop_all_remote()
            self._running = False
            return True

        return False

    def _print_help(self) -> None:
        print(f"""
{Colors.BOLD}Available Commands:{Colors.RESET}

  {Colors.CYAN}/help{Colors.RESET}      Show this help message
  {Colors.CYAN}/clear{Colors.RESET}     Clear conversation history
  {Colors.CYAN}/history{Colors.RESET}   Show conversation history
  {Colors.CYAN}/todos{Colors.RESET}     Show current task list
  {Colors.CYAN}/config{Colors.RESET}    Show current configuration
  {Colors.CYAN}/health{Colors.RESET}    Show system health status
  {Colors.CYAN}/sessions{Colors.RESET}  List saved sessions
  {Colors.CYAN}/snapshot{Colors.RESET}  Create state snapshot
  {Colors.CYAN}/snapshots{Colors.RESET} List saved snapshots
  {Colors.CYAN}/metrics{Colors.RESET}   Show tool execution metrics
  {Colors.CYAN}/analytics{Colors.RESET} Show session usage analytics
  {Colors.CYAN}/remote-control{Colors.RESET} Manage remote interfaces (API, Telegram, Slack)
  {Colors.CYAN}/rc{Colors.RESET}        Shortcut for /remote-control
  {Colors.CYAN}/exit{Colors.RESET}      Exit the agent
""")

    def _show_history(self) -> None:
        msgs = self.agent.messages
        if not msgs:
            print(f"  {Colors.DIM}No messages yet.{Colors.RESET}\n")
            return

        print()
        for msg in msgs:
            if msg.role == "user":
                print(f"  {Colors.BLUE}You:{Colors.RESET} {msg.content[:100]}")
            elif msg.role == "assistant":
                preview = msg.content[:100] if msg.content else "[tool calls]"
                print(f"  {Colors.GREEN}Agent:{Colors.RESET} {preview}")
            elif msg.role == "tool_result":
                count = len(msg.tool_results) if msg.tool_results else 0
                print(f"  {Colors.DIM}[{count} tool result(s)]{Colors.RESET}")
        print()

    def _show_todos(self) -> None:
        try:
            todo_tool = self.agent.registry.get_tool("todo_write")
        except KeyError:
            todo_tool = None
        if not todo_tool or not hasattr(todo_tool, "todos"):
            print(f"  {Colors.DIM}No todo tool available.{Colors.RESET}\n")
            return

        todos = todo_tool.todos
        if not todos:
            print(f"  {Colors.DIM}No tasks tracked yet.{Colors.RESET}\n")
            return

        status_colors = {
            "pending": Colors.DIM,
            "in_progress": Colors.YELLOW,
            "completed": Colors.GREEN,
        }
        status_icons = {
            "pending": "⬜",
            "in_progress": "🔄",
            "completed": "✅",
        }

        print()
        for todo in todos:
            color = status_colors.get(todo["status"], "")
            icon = status_icons.get(todo["status"], "•")
            label = todo.get("activeForm", todo["content"]) if todo["status"] == "in_progress" else todo["content"]
            print(f"  {icon} {color}{label}{Colors.RESET}")
        print()

    def _show_sessions(self) -> None:
        """Display recent saved sessions."""
        if not self._agent_session:
            print(f"  {Colors.DIM}Session manager not available.{Colors.RESET}\n")
            return
        sessions = self._agent_session.list_recent(limit=10)
        if not sessions:
            print(f"  {Colors.DIM}No saved sessions.{Colors.RESET}\n")
            return
        print()
        for s in sessions:
            from datetime import datetime
            ts = datetime.fromtimestamp(s.updated_at).strftime("%Y-%m-%d %H:%M")
            current = " ◀" if s.session_id == self._agent_session.session_id else ""
            print(
                f"  {Colors.CYAN}{s.session_id[:8]}{Colors.RESET} "
                f"{s.title}  {Colors.DIM}({ts}, {s.message_count} msgs){Colors.RESET}"
                f"{Colors.GREEN}{current}{Colors.RESET}"
            )
        print()

    def _create_snapshot(self, label: str = "") -> None:
        """Create a state snapshot."""
        if not self._snapshot_manager:
            print(f"  {Colors.DIM}Snapshot manager not available.{Colors.RESET}\n")
            return
        snap_id = self._snapshot_manager.create_snapshot(
            messages=self.agent.messages,
            label=label or "Manual snapshot",
            session_id=self._agent_session.session_id if self._agent_session else None,
        )
        print(f"  {Colors.GREEN}✓ Snapshot created: {snap_id}{Colors.RESET}\n")

    def _show_snapshots(self) -> None:
        """List saved snapshots."""
        if not self._snapshot_manager:
            print(f"  {Colors.DIM}Snapshot manager not available.{Colors.RESET}\n")
            return
        snaps = self._snapshot_manager.list_snapshots(limit=10)
        if not snaps:
            print(f"  {Colors.DIM}No snapshots saved.{Colors.RESET}\n")
            return
        print()
        for s in snaps:
            from datetime import datetime
            ts = datetime.fromtimestamp(s.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"  {Colors.CYAN}{s.snapshot_id[:16]}{Colors.RESET} "
                f"{s.label}  {Colors.DIM}({ts}){Colors.RESET}"
            )
        print()

    def _show_metrics(self) -> None:
        """Display tool execution metrics via RichOutput."""
        registry = self.agent.registry
        collector = getattr(registry, "metrics_collector", None)
        if not collector:
            print(f"  {Colors.DIM}Metrics collector not available.{Colors.RESET}\n")
            return
        summary = collector.summary()
        if summary.get("total_tool_calls", 0) == 0:
            print(f"  {Colors.DIM}No tool calls recorded yet.{Colors.RESET}\n")
            return
        print()
        print(self._rich.metrics_table(summary))
        print()

    def _show_analytics(self) -> None:
        """Display session usage analytics."""
        if not self._usage_analytics:
            print(f"  {Colors.DIM}Usage analytics not available.{Colors.RESET}\n")
            return
        report = self._usage_analytics.session_report()
        print()
        # Cost summary
        cost = report.get("cost", {})
        total = cost.get("total_cost", 0)
        calls = cost.get("call_count", 0)
        savings = cost.get("total_cache_savings", 0)
        print(f"  {Colors.BOLD}Cost:{Colors.RESET} ${total:.4f} ({calls} calls, ${savings:.4f} cache savings)")
        budget = cost.get("budget_limit")
        if budget is not None:
            remaining = cost.get("remaining_budget", 0)
            print(f"  {Colors.BOLD}Budget:{Colors.RESET} ${remaining:.4f} remaining of ${budget:.4f}")
        # Routing summary
        routing = report.get("routing", {})
        dist = routing.get("tier_distribution", {})
        if dist:
            parts = [f"{k}: {v}" for k, v in dist.items()]
            print(f"  {Colors.BOLD}Routing:{Colors.RESET} {', '.join(parts)}")
        esc = routing.get("escalation_count", 0)
        if esc:
            print(f"  {Colors.YELLOW}Escalations: {esc}{Colors.RESET}")
        # Efficiency
        score = self._usage_analytics.efficiency_score()
        print(f"  {Colors.BOLD}Efficiency:{Colors.RESET} {score:.0f}/100")
        # Recommendations
        recs = report.get("recommendations", [])
        if recs:
            print(f"  {Colors.BOLD}Recommendations:{Colors.RESET}")
            for r in recs:
                print(f"    {Colors.DIM}• {r}{Colors.RESET}")
        print()

    async def _run_streaming(self, user_input: str) -> str:
        """Run agent with streaming display — print tokens as they arrive."""
        self._spinner.stop()
        self._is_streaming = True

        # Suppress noisy logging during streaming so debug logs don't
        # interleave with the response text on the terminal.
        root_logger = logging.getLogger()
        saved_level = root_logger.level
        root_logger.setLevel(logging.WARNING)

        with _stdout_lock:
            sys.stdout.write(f"\n{Colors.BOLD}{Colors.GREEN}Agent ▸{Colors.RESET} ")
            sys.stdout.flush()

        full_response = ""
        try:
            async for chunk in self.agent.run_stream(user_input):
                with _stdout_lock:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                full_response += chunk
        finally:
            # Restore logging level and streaming flag
            root_logger.setLevel(saved_level)
            self._is_streaming = False

        with _stdout_lock:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return full_response

    async def _show_health(self) -> None:
        """Display health report."""
        if not self._health_monitor:
            print(f"  {Colors.DIM}Health monitor not available.{Colors.RESET}\n")
            return

        report = await self._health_monitor.check_health()
        status_colors = {
            "healthy": Colors.GREEN,
            "degraded": Colors.YELLOW,
            "unhealthy": Colors.RED,
        }
        color = status_colors.get(report.status.value, Colors.DIM)
        print(f"\n  {Colors.BOLD}System Health:{Colors.RESET} {color}{report.status.value.upper()}{Colors.RESET}")
        print(f"  {Colors.DIM}Uptime: {report.uptime_seconds:.1f}s{Colors.RESET}")

        for comp in report.components:
            c = status_colors.get(comp.status.value, Colors.DIM)
            print(f"    {c}●{Colors.RESET} {comp.name}: {comp.status.value} ({comp.response_time_ms:.1f}ms)")
        print()

    def _show_config(self) -> None:
        provider = self.agent.provider
        print(f"""
  {Colors.BOLD}Provider:{Colors.RESET} {provider.__class__.__name__}
  {Colors.BOLD}Tools:{Colors.RESET} {', '.join(self.agent.registry.tool_names)}
  {Colors.BOLD}Max iterations:{Colors.RESET} {self.agent.max_iterations}
""")

    # ── Remote Control ─────────────────────────────────────────────

    async def _handle_remote_control(self, arg: str) -> None:
        """Handle /remote-control subcommands."""
        parts = arg.strip().split(None, 1)
        subcmd = parts[0].lower() if parts else ""
        sub_arg = parts[1] if len(parts) > 1 else ""

        if subcmd in ("status", ""):
            self._rc_show_status()
        elif subcmd == "start":
            await self._rc_start(sub_arg)
        elif subcmd == "stop":
            await self._rc_stop(sub_arg)
        elif subcmd == "help":
            self._rc_help()
        else:
            print(f"  {Colors.RED}Unknown subcommand '{subcmd}'.{Colors.RESET}")
            self._rc_help()

    def _rc_help(self) -> None:
        """Show remote control help."""
        print(f"""
{Colors.BOLD}Remote Control — Manage remote interfaces{Colors.RESET}

  {Colors.CYAN}/rc status{Colors.RESET}                Show running services
  {Colors.CYAN}/rc start api{Colors.RESET}             Start REST API + WebSocket server
  {Colors.CYAN}/rc start api 9000{Colors.RESET}        Start API on custom port
  {Colors.CYAN}/rc start telegram{Colors.RESET}        Start Telegram bot
  {Colors.CYAN}/rc start telegram <token>{Colors.RESET} Start with explicit token
  {Colors.CYAN}/rc start slack{Colors.RESET}           Start Slack bot
  {Colors.CYAN}/rc start all{Colors.RESET}             Start all available services
  {Colors.CYAN}/rc stop api{Colors.RESET}              Stop the API server
  {Colors.CYAN}/rc stop all{Colors.RESET}              Stop all running services
""")

    def _rc_show_status(self) -> None:
        """Display status of all remote services."""
        if not self._remote_services:
            print(f"\n  {Colors.DIM}No remote services running.{Colors.RESET}")
            print(f"  {Colors.DIM}Use /rc start api|telegram|slack|all to start.{Colors.RESET}\n")
            return

        print(f"\n{Colors.BOLD}  Remote Services:{Colors.RESET}\n")
        for name, svc in self._remote_services.items():
            elapsed = time.time() - svc["started_at"]
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            task = svc["task"]
            if task.done():
                status = f"{Colors.RED}● stopped{Colors.RESET}"
            else:
                status = f"{Colors.GREEN}● running{Colors.RESET}"
            info = svc.get("info", "")
            print(f"    {status} {Colors.CYAN}{name}{Colors.RESET}  {info}  ({mins}m {secs}s)")
        print()

    async def _rc_start(self, arg: str) -> None:
        """Start a remote service."""
        parts = arg.strip().split(None, 1)
        service = parts[0].lower() if parts else ""
        extra = parts[1].strip() if len(parts) > 1 else ""

        if not service:
            print(f"  {Colors.RED}Specify a service: api, telegram, slack, or all{Colors.RESET}\n")
            return

        if service == "all":
            await self._rc_start("api" + (" " + extra if extra else ""))
            await self._rc_start("telegram")
            await self._rc_start("slack")
            return

        if service in self._remote_services:
            task = self._remote_services[service]["task"]
            if not task.done():
                print(f"  {Colors.YELLOW}⚠ {service} is already running.{Colors.RESET}\n")
                return

        if service == "api":
            await self._start_api_server(extra)
        elif service == "telegram":
            await self._start_telegram_bot(extra)
        elif service == "slack":
            await self._start_slack_bot(extra)
        else:
            print(f"  {Colors.RED}Unknown service '{service}'. Use api, telegram, slack, or all.{Colors.RESET}\n")

    async def _start_api_server(self, port_arg: str = "") -> None:
        """Launch the API server as a background task."""
        port = int(port_arg) if port_arg.isdigit() else 8000
        try:
            from .api import RestAPIInterface
        except ImportError:
            print(f"  {Colors.RED}FastAPI not installed. Run: pip install fastapi uvicorn websockets{Colors.RESET}\n")
            return

        factory = self._agent_factory
        if not factory:
            print(f"  {Colors.RED}No agent factory available. Start with --mode cli to enable remote control.{Colors.RESET}\n")
            return

        api = RestAPIInterface(
            agent=self.agent,
            agent_factory=factory,
            host="0.0.0.0",
            port=port,
        )

        task = asyncio.create_task(api.run())
        self._remote_services["api"] = {
            "task": task,
            "started_at": time.time(),
            "info": f"http://localhost:{port}",
            "instance": api,
        }
        # Give server a moment to start
        await asyncio.sleep(0.5)
        print(f"  {Colors.GREEN}✓ API server started on http://localhost:{port}{Colors.RESET}")
        print(f"  {Colors.DIM}  Dashboard: http://localhost:{port}/{Colors.RESET}\n")

    async def _start_telegram_bot(self, token_arg: str = "") -> None:
        """Launch the Telegram bot as a background task."""
        token = token_arg or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            print(f"  {Colors.YELLOW}⚠ No Telegram token found.{Colors.RESET}")
            print(f"  {Colors.DIM}  Set TELEGRAM_BOT_TOKEN env var or: /rc start telegram <token>{Colors.RESET}\n")
            return
        try:
            from .telegram_bot import TelegramBotInterface
        except ImportError:
            print(f"  {Colors.RED}python-telegram-bot not installed. Run: pip install python-telegram-bot{Colors.RESET}\n")
            return

        factory = self._agent_factory
        if not factory:
            print(f"  {Colors.RED}No agent factory available.{Colors.RESET}\n")
            return

        persist = os.path.join(self._workspace, ".cowork", "telegram_sessions.json")
        bot = TelegramBotInterface(
            agent=self.agent,
            token=token,
            agent_factory=factory,
            persist_path=persist,
        )

        task = asyncio.create_task(bot.run())
        self._remote_services["telegram"] = {
            "task": task,
            "started_at": time.time(),
            "info": "polling",
            "instance": bot,
        }
        await asyncio.sleep(0.5)
        print(f"  {Colors.GREEN}✓ Telegram bot started.{Colors.RESET}\n")

    async def _start_slack_bot(self, tokens_arg: str = "") -> None:
        """Launch the Slack bot as a background task."""
        bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
        app_token = os.environ.get("SLACK_APP_TOKEN", "")
        if not bot_token or not app_token:
            print(f"  {Colors.YELLOW}⚠ Slack tokens not found.{Colors.RESET}")
            print(f"  {Colors.DIM}  Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN env vars.{Colors.RESET}\n")
            return
        try:
            from .slack_bot import SlackBotInterface
        except ImportError:
            print(f"  {Colors.RED}slack-bolt not installed. Run: pip install slack-bolt{Colors.RESET}\n")
            return

        factory = self._agent_factory
        if not factory:
            print(f"  {Colors.RED}No agent factory available.{Colors.RESET}\n")
            return

        bot = SlackBotInterface(
            agent=self.agent,
            bot_token=bot_token,
            app_token=app_token,
            agent_factory=factory,
        )

        task = asyncio.create_task(bot.run())
        self._remote_services["slack"] = {
            "task": task,
            "started_at": time.time(),
            "info": "socket mode",
            "instance": bot,
        }
        await asyncio.sleep(0.5)
        print(f"  {Colors.GREEN}✓ Slack bot started.{Colors.RESET}\n")

    async def _rc_stop(self, arg: str) -> None:
        """Stop a running remote service."""
        service = arg.strip().lower()
        if not service:
            print(f"  {Colors.RED}Specify a service: api, telegram, slack, or all{Colors.RESET}\n")
            return

        if service == "all":
            await self._stop_all_remote()
            return

        if service not in self._remote_services:
            print(f"  {Colors.YELLOW}⚠ {service} is not running.{Colors.RESET}\n")
            return

        svc = self._remote_services.pop(service)
        task = svc["task"]
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        print(f"  {Colors.GREEN}✓ {service} stopped.{Colors.RESET}\n")

    async def _stop_all_remote(self) -> None:
        """Stop all running remote services."""
        if not self._remote_services:
            print(f"  {Colors.DIM}No remote services running.{Colors.RESET}\n")
            return

        names = list(self._remote_services.keys())
        for name in names:
            svc = self._remote_services.pop(name)
            task = svc["task"]
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            print(f"  {Colors.GREEN}✓ {name} stopped.{Colors.RESET}")
        print()

    def _blocking_input(self, prompt: str) -> str:
        """Read input from stdin (runs in thread to avoid blocking event loop)."""
        return input(prompt).strip()

    async def run(self) -> None:
        """Main interactive loop."""
        self._running = True
        self._setup_readline()
        self._print_banner()

        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Get user input in a thread so background tasks (Slack, API) keep running
                user_input = await loop.run_in_executor(
                    None,
                    self._blocking_input,
                    f"{Colors.BOLD}{Colors.BLUE}You ▸ {Colors.RESET}",
                )

                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    handled = await self._handle_command(user_input)
                    if handled:
                        continue

                # Send to agent
                print()
                self._spinner.start("Thinking")
                try:
                    if self._streaming_enabled:
                        response = await self._run_streaming(user_input)
                        # Streaming already printed the response inline
                        print()  # Just add spacing
                    else:
                        response = await self.agent.run(user_input)
                        # Display response
                        print(f"\n{Colors.BOLD}{Colors.GREEN}Agent ▸{Colors.RESET} {response}\n")
                finally:
                    self._spinner.stop()

            except KeyboardInterrupt:
                print(f"\n  {Colors.DIM}(Cancelled){Colors.RESET}\n")
                continue

            except EOFError:
                print(f"\n{Colors.DIM}Goodbye!{Colors.RESET}")
                self._running = False

            except Exception as e:
                print(f"\n  {Colors.RED}Error: {str(e)}{Colors.RESET}\n")

        self._save_readline()
        print(f"\n{Colors.DIM}Session ended.{Colors.RESET}")
