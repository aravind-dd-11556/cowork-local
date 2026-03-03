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
from ..core.model_selector import ModelSelector, ModelInfo
from ..core.providers.base import ProviderFactory
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
        self._model_selector: Optional[ModelSelector] = None  # Lazy init

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
        """Configure readline for command history and key bindings."""
        try:
            readline.set_history_length(1000)

            # Enable standard key bindings (arrow keys, Home/End, etc.)
            readline.parse_and_bind("set editing-mode emacs")
            # Tab completion for /commands
            readline.parse_and_bind("tab: complete")
            readline.set_completer(self._readline_completer)
            readline.set_completer_delims(" \t\n")

            # Ensure history directory exists
            history_dir = os.path.dirname(self.history_file)
            if history_dir and not os.path.exists(history_dir):
                os.makedirs(history_dir, exist_ok=True)

            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
        except Exception:
            pass

    def _readline_completer(self, text: str, state: int) -> Optional[str]:
        """Tab-complete /commands."""
        commands = [
            "/help", "/clear", "/history", "/todos", "/config",
            "/health", "/sessions", "/snapshot", "/snapshots",
            "/metrics", "/analytics", "/model",
            "/remote-control", "/rc",
            "/connect", "/disconnect", "/connectors",
            "/exit",
        ]
        if text.startswith("/"):
            matches = [c for c in commands if c.startswith(text)]
        else:
            matches = []
        if state < len(matches):
            return matches[state]
        return None

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

        elif command == "/model":
            await self._handle_model_command(arg)
            return True

        elif command in ("/remote-control", "/rc"):
            await self._handle_remote_control(arg)
            return True

        elif command == "/connect":
            await self._handle_connect(arg)
            return True

        elif command == "/disconnect":
            await self._handle_disconnect(arg)
            return True

        elif command == "/connectors":
            await self._handle_list_connectors(arg)
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
  {Colors.CYAN}/model{Colors.RESET}     Model selection (list, select, test, info)
  {Colors.CYAN}/remote-control{Colors.RESET} Manage remote interfaces (API, Telegram, Slack)
  {Colors.CYAN}/rc{Colors.RESET}        Shortcut for /remote-control
  {Colors.CYAN}/connect{Colors.RESET}   Connect an external service (/connect <name> [--token TOKEN])
  {Colors.CYAN}/disconnect{Colors.RESET} Disconnect a service (/disconnect <name>)
  {Colors.CYAN}/connectors{Colors.RESET} List all connectors and their status
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
        """Run agent with streaming display — print tokens as they arrive.

        When the agent has ``run_stream_events()`` enabled, consumes
        structured StreamEvent objects and renders each type appropriately
        (progress bars for tools, status messages, etc.).  Otherwise falls
        back to the raw ``run_stream()`` text-chunk path.
        """
        self._spinner.stop()
        self._is_streaming = True

        # Suppress noisy logging during streaming so debug logs don't
        # interleave with the response text on the terminal.
        root_logger = logging.getLogger()
        saved_level = root_logger.level
        root_logger.setLevel(logging.WARNING)

        full_response = ""
        try:
            # Prefer structured events when the agent supports them
            if getattr(self.agent, '_events_enabled', False):
                full_response = await self._run_stream_events(user_input)
            else:
                with _stdout_lock:
                    sys.stdout.write(f"\n{Colors.BOLD}{Colors.GREEN}Agent ▸{Colors.RESET} ")
                    sys.stdout.flush()

                async for chunk in self.agent.run_stream(user_input):
                    with _stdout_lock:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                    full_response += chunk

                with _stdout_lock:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
        finally:
            # Restore logging level and streaming flag
            root_logger.setLevel(saved_level)
            self._is_streaming = False

        return full_response

    # ── Structured-event streaming ────────────────────────────

    async def _run_stream_events(self, user_input: str) -> str:
        """Consume ``run_stream_events()`` and render each event type."""
        from ..core.stream_events import (
            TextChunk, ToolStart, ToolProgress, ToolEnd, StatusUpdate,
        )

        with _stdout_lock:
            sys.stdout.write(f"\n{Colors.BOLD}{Colors.GREEN}Agent ▸{Colors.RESET} ")
            sys.stdout.flush()

        full_response = ""
        cancellation_token = getattr(self.agent, '_cancellation_token', None)

        async for event in self.agent.run_stream_events(
            user_input, cancellation_token=cancellation_token,
        ):
            if isinstance(event, TextChunk):
                with _stdout_lock:
                    sys.stdout.write(event.text)
                    sys.stdout.flush()
                full_response += event.text

            elif isinstance(event, ToolStart):
                icon = self._tool_icon(event.tool_call.name)
                with _stdout_lock:
                    sys.stdout.write(
                        f"\n  {Colors.DIM}{icon} Executing "
                        f"{Colors.CYAN}{event.tool_call.name}{Colors.RESET}"
                        f"{Colors.DIM}...{Colors.RESET}"
                    )
                    sys.stdout.flush()

            elif isinstance(event, ToolProgress):
                bar = self._rich.stream_progress_bar(
                    event.progress_percent,
                    label=event.message[:30],
                )
                with _stdout_lock:
                    # Overwrite the current line with the progress bar
                    sys.stdout.write(f"\r{bar}")
                    sys.stdout.flush()

            elif isinstance(event, ToolEnd):
                duration = event.duration_ms
                result = event.result
                output_lines = len(result.output.split("\n")) if result.output else 0
                line = self._rich.tool_result(
                    event.tool_call.name, result.success, duration, output_lines,
                )
                with _stdout_lock:
                    # Clear progress bar line and print result
                    sys.stdout.write(f"\r{' ' * 80}\r{line}\n")
                    if not result.success and result.error:
                        sys.stdout.write(self._rich.error(result.error) + "\n")
                    # Resume text on new line
                    sys.stdout.flush()

            elif isinstance(event, StatusUpdate):
                color = Colors.YELLOW if event.severity == "warning" else Colors.DIM
                with _stdout_lock:
                    sys.stdout.write(
                        f"\n  {color}⟳ {event.message}{Colors.RESET}\n"
                    )
                    sys.stdout.flush()

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
                # Show crash reason if the task died with an exception
                error_detail = ""
                try:
                    exc = task.exception()
                    if exc:
                        error_detail = f"\n      {Colors.RED}Error: {exc}{Colors.RESET}"
                        # Print traceback for debugging
                        import traceback as _tb
                        tb_lines = _tb.format_exception(type(exc), exc, exc.__traceback__)
                        for line in tb_lines[-3:]:
                            for sub in line.rstrip().split("\n"):
                                error_detail += f"\n      {Colors.DIM}{sub}{Colors.RESET}"
                except asyncio.CancelledError:
                    error_detail = f"\n      {Colors.DIM}(cancelled){Colors.RESET}"
            else:
                status = f"{Colors.GREEN}● running{Colors.RESET}"
                error_detail = ""
            info = svc.get("info", "")
            print(f"    {status} {Colors.CYAN}{name}{Colors.RESET}  {info}  ({mins}m {secs}s){error_detail}")
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

        def _on_telegram_done(t: asyncio.Task) -> None:
            try:
                exc = t.exception()
                if exc:
                    print(f"\n  {Colors.RED}✗ Telegram bot crashed: {exc}{Colors.RESET}")
                    print(f"  {Colors.DIM}Use /rc start telegram to restart.{Colors.RESET}\n")
            except asyncio.CancelledError:
                pass

        task.add_done_callback(_on_telegram_done)

        self._remote_services["telegram"] = {
            "task": task,
            "started_at": time.time(),
            "info": "polling",
            "instance": bot,
        }
        await asyncio.sleep(1)
        if task.done():
            print(f"  {Colors.RED}✗ Telegram bot failed to start. See error above.{Colors.RESET}\n")
        else:
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

        def _on_slack_done(t: asyncio.Task) -> None:
            """Callback when Slack task finishes — surface errors immediately."""
            try:
                exc = t.exception()
                if exc:
                    print(f"\n  {Colors.RED}✗ Slack bot crashed: {exc}{Colors.RESET}")
                    import traceback as _tb
                    tb_lines = _tb.format_exception(type(exc), exc, exc.__traceback__)
                    for line in tb_lines[-4:]:
                        for sub in line.rstrip().split("\n"):
                            print(f"    {Colors.DIM}{sub}{Colors.RESET}")
                    print(f"  {Colors.DIM}Use /rc start slack to restart.{Colors.RESET}\n")
            except asyncio.CancelledError:
                print(f"\n  {Colors.DIM}Slack bot stopped.{Colors.RESET}\n")

        task.add_done_callback(_on_slack_done)

        self._remote_services["slack"] = {
            "task": task,
            "started_at": time.time(),
            "info": "socket mode",
            "instance": bot,
        }
        # Wait a bit longer to catch immediate startup failures
        await asyncio.sleep(2)
        if task.done():
            # Task already died — the callback will have printed the error
            print(f"  {Colors.RED}✗ Slack bot failed to start. See error above.{Colors.RESET}\n")
        else:
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

    # ── Connector Management (Sprint 44) ──────────────────────────

    async def _handle_connect(self, arg: str) -> None:
        """Handle /connect <name> [--token TOKEN]."""
        if not arg.strip():
            print(f"""
  {Colors.BOLD}Usage:{Colors.RESET} /connect <service-name> [--token TOKEN]

  {Colors.BOLD}Examples:{Colors.RESET}
    /connect github --token ghp_abc123...
    /connect slack --token xoxb-...
    /connect gmail
    /connect notion

  Use {Colors.CYAN}/connectors{Colors.RESET} to see available services.
""")
            return

        # Parse --token if present
        parts = arg.strip().split()
        service_name = parts[0]
        token = ""
        for i, p in enumerate(parts):
            if p == "--token" and i + 1 < len(parts):
                token = parts[i + 1]
                break

        # Get auth manager from agent
        auth_mgr = getattr(self.agent, "connector_auth", None)
        conn_registry = getattr(self.agent, "connector_registry", None)

        if not auth_mgr or not conn_registry:
            print(f"  {Colors.YELLOW}⚠ Connector system not initialized.{Colors.RESET}")
            print(f"  {Colors.DIM}Ensure mcp_registry is enabled in config.{Colors.RESET}\n")
            return

        # Resolve connector by name
        conn = None
        name_lower = service_name.lower()
        for c in conn_registry.all_connectors:
            if c.name.lower() == name_lower:
                conn = c
                break

        if not conn:
            # Try partial match
            matches = conn_registry.search([name_lower])
            if matches:
                conn = matches[0]

        if not conn:
            print(f"  {Colors.RED}✗ No connector found matching '{service_name}'.{Colors.RESET}")
            print(f"  Use {Colors.CYAN}/connectors{Colors.RESET} to see available services.\n")
            return

        # Check if already connected
        if auth_mgr.is_connected(conn.uuid):
            print(f"  {Colors.GREEN}✅ {conn.name} is already connected.{Colors.RESET}\n")
            return

        # Get auth config
        auth_config = auth_mgr.get_auth_config(conn.uuid)
        if not auth_config:
            print(f"  {Colors.RED}✗ No auth configuration for {conn.name}.{Colors.RESET}\n")
            return

        from ..core.connector_auth import AuthMethod

        if auth_config.method == AuthMethod.API_TOKEN:
            # Try to get token from: argument > env var > prompt
            import os
            from ..core.connector_auth import validate_token, mask_token
            actual_token = token
            if not actual_token and auth_config.token_env_var:
                actual_token = os.environ.get(auth_config.token_env_var, "")

            if not actual_token:
                token_name = auth_config.token_name or "API Token"
                print(f"  {Colors.BOLD}{conn.name}{Colors.RESET} requires a {token_name}.")
                if auth_config.token_env_var:
                    print(f"  {Colors.DIM}(or set {auth_config.token_env_var} env var){Colors.RESET}")
                try:
                    # Sprint 45: Use getpass to avoid echoing token to terminal
                    import getpass
                    actual_token = getpass.getpass(
                        f"  {Colors.BOLD}Enter token ▸ {Colors.RESET}"
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    print(f"\n  {Colors.DIM}Cancelled.{Colors.RESET}\n")
                    return

            if not actual_token:
                print(f"  {Colors.RED}✗ No token provided.{Colors.RESET}\n")
                return

            # Sprint 45: Validate token
            try:
                actual_token = validate_token(actual_token)
            except ValueError as e:
                print(f"  {Colors.RED}✗ Invalid token: {e}{Colors.RESET}\n")
                return

            try:
                cred = auth_mgr.connect_with_token(
                    connector_uuid=conn.uuid,
                    connector_name=conn.name,
                    token=actual_token,
                )
            except ValueError as e:
                print(f"  {Colors.RED}✗ {e}{Colors.RESET}\n")
                return

            conn_registry.mark_connected(conn.uuid)
            # Sprint 45: Secure token masking
            masked = mask_token(actual_token)
            print(f"  {Colors.GREEN}✅ {conn.name} connected!{Colors.RESET} Token: {masked}")
            print(f"  {Colors.DIM}Credentials saved for future sessions.{Colors.RESET}\n")

            # Sprint 45: Clear the token from readline history
            try:
                hist_len = readline.get_current_history_length()
                for i in range(hist_len, 0, -1):
                    item = readline.get_history_item(i)
                    if item and "--token" in item:
                        readline.remove_history_item(i - 1)
                        break
            except Exception:
                pass  # readline history manipulation is best-effort

        elif auth_config.method == AuthMethod.OAUTH2:
            try:
                auth_url = auth_mgr.initiate_oauth2(
                    connector_uuid=conn.uuid,
                    connector_name=conn.name,
                )
                print(f"  {Colors.BOLD}🌐 {conn.name} requires browser authorization.{Colors.RESET}")
                print(f"  Open this URL: {Colors.CYAN}{auth_url}{Colors.RESET}")
                print(f"  {Colors.DIM}After authorizing, enter the code below.{Colors.RESET}")
                try:
                    code = input(f"  {Colors.BOLD}Authorization code ▸ {Colors.RESET}").strip()
                except (EOFError, KeyboardInterrupt):
                    print(f"\n  {Colors.DIM}Cancelled.{Colors.RESET}\n")
                    return
                if code:
                    # Find the state from the oauth_states
                    states = list(auth_mgr._oauth_states.keys())
                    if states:
                        auth_mgr.complete_oauth2(
                            state=states[-1],
                            code=code,
                            connector_name=conn.name,
                        )
                        conn_registry.mark_connected(conn.uuid)
                        print(f"  {Colors.GREEN}✅ {conn.name} connected via OAuth2!{Colors.RESET}\n")
                    else:
                        print(f"  {Colors.RED}✗ OAuth state expired. Try again.{Colors.RESET}\n")
            except ValueError as e:
                print(f"  {Colors.RED}✗ {e}{Colors.RESET}\n")

        elif auth_config.method == AuthMethod.ENV_VAR:
            import os
            env_values = {}
            missing = []
            for var in auth_config.env_vars:
                val = os.environ.get(var, "")
                if val:
                    env_values[var] = val
                else:
                    missing.append(var)

            if missing:
                print(f"  {Colors.YELLOW}⚠ Missing env vars for {conn.name}:{Colors.RESET}")
                for v in missing:
                    print(f"    - {v}")
                print()
                return

            auth_mgr.connect_with_env(conn.uuid, conn.name, env_values)
            conn_registry.mark_connected(conn.uuid)
            print(f"  {Colors.GREEN}✅ {conn.name} connected via env vars!{Colors.RESET}\n")

    async def _handle_disconnect(self, arg: str) -> None:
        """Handle /disconnect <name>."""
        if not arg.strip():
            print(f"  {Colors.BOLD}Usage:{Colors.RESET} /disconnect <service-name>\n")
            return

        service_name = arg.strip().split()[0]
        auth_mgr = getattr(self.agent, "connector_auth", None)
        conn_registry = getattr(self.agent, "connector_registry", None)

        if not auth_mgr or not conn_registry:
            print(f"  {Colors.YELLOW}⚠ Connector system not initialized.{Colors.RESET}\n")
            return

        # Find connector
        conn = None
        for c in conn_registry.all_connectors:
            if c.name.lower() == service_name.lower():
                conn = c
                break

        if not conn:
            print(f"  {Colors.RED}✗ No connector found matching '{service_name}'.{Colors.RESET}\n")
            return

        if not auth_mgr.is_connected(conn.uuid):
            print(f"  {Colors.DIM}{conn.name} is not currently connected.{Colors.RESET}\n")
            return

        auth_mgr.disconnect(conn.uuid)
        conn_registry.mark_disconnected(conn.uuid)
        print(f"  {Colors.GREEN}✅ {conn.name} disconnected. Credentials removed.{Colors.RESET}\n")

    async def _handle_list_connectors(self, arg: str) -> None:
        """Handle /connectors [connected|available]."""
        conn_registry = getattr(self.agent, "connector_registry", None)
        auth_mgr = getattr(self.agent, "connector_auth", None)

        if not conn_registry:
            print(f"  {Colors.YELLOW}⚠ Connector registry not initialized.{Colors.RESET}\n")
            return

        filter_type = arg.strip().lower() if arg.strip() else "all"

        if filter_type == "connected":
            connectors = conn_registry.connected_connectors
        elif filter_type == "available":
            connectors = conn_registry.available_connectors
        else:
            connectors = conn_registry.all_connectors

        if not connectors:
            print(f"  {Colors.DIM}No connectors found.{Colors.RESET}\n")
            return

        connected_count = sum(1 for c in connectors if c.connected or
                              (auth_mgr and auth_mgr.is_connected(c.uuid)))
        available_count = len(connectors) - connected_count

        print(f"\n  {Colors.BOLD}Connectors:{Colors.RESET} "
              f"{Colors.GREEN}{connected_count} connected{Colors.RESET}, "
              f"{available_count} available")
        print(f"  {'─' * 50}")

        for conn in connectors:
            is_connected = conn.connected or (
                auth_mgr and auth_mgr.is_connected(conn.uuid)
            )
            if is_connected:
                icon = f"{Colors.GREEN}✅{Colors.RESET}"
                status = f"{Colors.GREEN}Connected{Colors.RESET}"
            else:
                icon = f"⬜"
                status = f"{Colors.DIM}Available{Colors.RESET}"

            auth_hint = ""
            if auth_mgr:
                cfg = auth_mgr.get_auth_config(conn.uuid)
                if cfg:
                    auth_hint = f" {Colors.DIM}({cfg.method.value}){Colors.RESET}"

            print(f"  {icon} {Colors.BOLD}{conn.name:<15}{Colors.RESET} "
                  f"{status}{auth_hint}")
            print(f"     {Colors.DIM}{conn.description}{Colors.RESET}")

        print(f"  {'─' * 50}")
        print(f"  Use {Colors.CYAN}/connect <name>{Colors.RESET} to connect a service.\n")

    # ── Model Selection ─────────────────────────────────────────

    def _get_model_selector(self) -> ModelSelector:
        """Lazy-init the model selector from the agent's config."""
        if self._model_selector is None:
            pb = getattr(self.agent, "prompt_builder", None)
            config_data = getattr(pb, "config", {}) if pb else {}
            self._model_selector = ModelSelector(config_data)
        return self._model_selector

    async def _handle_model_command(self, arg: str) -> None:
        """Handle /model subcommands."""
        parts = arg.strip().split(None, 1)
        subcmd = parts[0].lower() if parts else ""
        sub_arg = parts[1] if len(parts) > 1 else ""

        if subcmd in ("", "help"):
            self._model_help()
        elif subcmd == "status":
            await self._model_status()
        elif subcmd == "current":
            self._model_current()
        elif subcmd == "list":
            await self._model_list(sub_arg)
        elif subcmd == "select":
            await self._model_select(sub_arg)
        elif subcmd == "test":
            await self._model_test(sub_arg)
        elif subcmd == "popular":
            self._model_popular(sub_arg)
        elif subcmd == "use":
            self._model_use(sub_arg)
        else:
            print(f"  {Colors.RED}Unknown subcommand '{subcmd}'.{Colors.RESET}")
            self._model_help()

    def _model_help(self) -> None:
        print(f"""
{Colors.BOLD}Model Selection — Switch between LLM providers and models{Colors.RESET}

  {Colors.CYAN}/model status{Colors.RESET}              Show which providers are available
  {Colors.CYAN}/model current{Colors.RESET}             Show current provider & model
  {Colors.CYAN}/model list <provider>{Colors.RESET}     List models from a provider
  {Colors.CYAN}/model list all{Colors.RESET}            List models from ALL providers
  {Colors.CYAN}/model popular [prov]{Colors.RESET}      Show popular/recommended models
  {Colors.CYAN}/model select <provider>{Colors.RESET}   Interactive model selection
  {Colors.CYAN}/model test <prov> <model>{Colors.RESET} Test a model connection
  {Colors.CYAN}/model use <prov> <model>{Colors.RESET}  Switch to a model immediately

{Colors.BOLD}Examples:{Colors.RESET}
  /model use ollama llama3.1:8b
  /model use anthropic claude-sonnet-4-5-20250929
  /model use openai gpt-4o
  /model use openrouter anthropic/claude-sonnet-4
  /model list ollama
  /model test openai gpt-4o
  /model select anthropic
""")

    async def _model_status(self) -> None:
        """Show provider availability."""
        selector = self._get_model_selector()
        print(f"\n{Colors.BOLD}  Provider Status:{Colors.RESET}\n")

        self._spinner.start("Checking providers")
        try:
            status = await selector.get_provider_status()
        finally:
            self._spinner.stop()

        print(selector.format_provider_status(status))

        # Show current
        prov, model = selector.get_current_model()
        print(f"\n  {Colors.BOLD}Current:{Colors.RESET} {Colors.CYAN}{prov}{Colors.RESET} / {Colors.GREEN}{model}{Colors.RESET}\n")

    def _model_current(self) -> None:
        """Show current model config."""
        provider = self.agent.provider
        pname = getattr(provider, "provider_name", provider.__class__.__name__)
        print(f"""
  {Colors.BOLD}Provider:{Colors.RESET}    {pname}
  {Colors.BOLD}Model:{Colors.RESET}       {provider.model}
  {Colors.BOLD}Temperature:{Colors.RESET} {provider.temperature}
  {Colors.BOLD}Max Tokens:{Colors.RESET}  {provider.max_tokens}
  {Colors.BOLD}Base URL:{Colors.RESET}    {getattr(provider, 'base_url', 'N/A')}
""")

    async def _model_list(self, arg: str) -> None:
        """List models from a provider or all providers."""
        selector = self._get_model_selector()
        parts = arg.strip().split(None, 1)
        provider_name = parts[0].lower() if parts else ""
        filter_text = parts[1] if len(parts) > 1 else ""

        if not provider_name:
            print(f"  {Colors.YELLOW}Specify a provider: ollama, openai, anthropic, openrouter, or 'all'{Colors.RESET}\n")
            return

        if provider_name == "all":
            self._spinner.start("Fetching models from all providers")
            try:
                all_models = await selector.list_all_models(filter_text)
            finally:
                self._spinner.stop()

            for pname, models in all_models.items():
                if models:
                    print(f"\n  {Colors.BOLD}{Colors.CYAN}{pname.upper()}{Colors.RESET} ({len(models)} models):\n")
                    show_pricing = pname == "openrouter"
                    print(selector.format_model_table(models[:20], show_pricing=show_pricing))
                    if len(models) > 20:
                        print(f"  {Colors.DIM}  ... and {len(models) - 20} more. Use /model list {pname} <filter> to narrow down.{Colors.RESET}")
                else:
                    print(f"\n  {Colors.DIM}{pname}: No models available or provider not configured.{Colors.RESET}")
            print()
        else:
            self._spinner.start(f"Fetching models from {provider_name}")
            try:
                models = await selector.list_models(provider_name, filter_text)
            finally:
                self._spinner.stop()

            if not models:
                print(f"  {Colors.YELLOW}No models found for {provider_name}. Is it configured?{Colors.RESET}\n")
                return

            show_pricing = provider_name == "openrouter"
            print(f"\n  {Colors.BOLD}{provider_name.upper()}{Colors.RESET} — {len(models)} models:\n")
            print(selector.format_model_table(models[:50], show_pricing=show_pricing))
            if len(models) > 50:
                print(f"\n  {Colors.DIM}Showing 50 of {len(models)}. Add a filter: /model list {provider_name} <search>{Colors.RESET}")
            print()

    async def _model_select(self, arg: str) -> None:
        """Interactive model selection for a provider."""
        selector = self._get_model_selector()
        provider_name = arg.strip().lower()

        if not provider_name:
            print(f"  {Colors.YELLOW}Specify a provider: ollama, openai, anthropic, openrouter{Colors.RESET}\n")
            return

        self._spinner.start(f"Fetching {provider_name} models")
        try:
            models = await selector.list_models(provider_name)
        finally:
            self._spinner.stop()

        if not models:
            print(f"  {Colors.YELLOW}No models available from {provider_name}.{Colors.RESET}\n")
            return

        print(f"\n  {Colors.BOLD}Select a model from {provider_name}:{Colors.RESET}\n")
        print(selector.format_model_table(models[:30], numbered=True))
        if len(models) > 30:
            print(f"  {Colors.DIM}  ... showing 30 of {len(models)}{Colors.RESET}")

        print(f"\n  {Colors.DIM}Enter number, model name, or 'cancel':{Colors.RESET}")
        try:
            choice = input(f"  {Colors.BOLD}Select ▸ {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {Colors.DIM}Cancelled.{Colors.RESET}\n")
            return

        if not choice or choice.lower() == "cancel":
            print(f"  {Colors.DIM}Cancelled.{Colors.RESET}\n")
            return

        # Resolve selection
        selected_model = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected_model = models[idx]
        else:
            # Search by name/id
            for m in models:
                if choice.lower() in m.id.lower() or choice.lower() in m.name.lower():
                    selected_model = m
                    break

        if not selected_model:
            print(f"  {Colors.RED}No model matching '{choice}'.{Colors.RESET}\n")
            return

        # Apply selection
        self._apply_model_switch(provider_name, selected_model.id)
        print(f"  {Colors.GREEN}✓ Switched to {provider_name}/{selected_model.id}{Colors.RESET}\n")

    async def _model_test(self, arg: str) -> None:
        """Test a specific model."""
        selector = self._get_model_selector()
        parts = arg.strip().split(None, 1)

        if len(parts) < 2:
            print(f"  {Colors.YELLOW}Usage: /model test <provider> <model_id>{Colors.RESET}\n")
            return

        provider_name = parts[0].lower()
        model_id = parts[1]

        self._spinner.start(f"Testing {provider_name}/{model_id}")
        try:
            result = await selector.test_model(provider_name, model_id)
        finally:
            self._spinner.stop()

        if result.success:
            print(f"\n  {Colors.GREEN}✅ Model test passed!{Colors.RESET}")
            print(f"  {Colors.BOLD}Latency:{Colors.RESET} {result.latency_ms:.0f}ms")
            if result.output_preview:
                print(f"  {Colors.BOLD}Response:{Colors.RESET} {result.output_preview[:100]}")
            if result.tokens_used:
                in_t = result.tokens_used.get("input_tokens", 0)
                out_t = result.tokens_used.get("output_tokens", 0)
                print(f"  {Colors.BOLD}Tokens:{Colors.RESET} {in_t} in / {out_t} out")
        else:
            print(f"\n  {Colors.RED}❌ Model test failed{Colors.RESET}")
            print(f"  {Colors.BOLD}Error:{Colors.RESET} {result.error}")
            if result.latency_ms > 0:
                print(f"  {Colors.BOLD}Latency:{Colors.RESET} {result.latency_ms:.0f}ms")
        print()

    def _model_popular(self, arg: str = "") -> None:
        """Show popular/recommended models."""
        provider_name = arg.strip().lower() if arg else ""

        if provider_name and provider_name in ModelSelector.POPULAR_MODELS:
            providers = {provider_name: ModelSelector.POPULAR_MODELS[provider_name]}
        else:
            providers = ModelSelector.POPULAR_MODELS

        print(f"\n{Colors.BOLD}  Popular Models:{Colors.RESET}\n")
        for pname, models in providers.items():
            print(f"  {Colors.CYAN}{pname.upper()}:{Colors.RESET}")
            for m in models:
                print(f"    • {m}")
            print()

        print(f"  {Colors.DIM}Use: /model use <provider> <model> to switch{Colors.RESET}\n")

    def _model_use(self, arg: str) -> None:
        """Immediately switch to a specific provider/model."""
        parts = arg.strip().split(None, 1)
        if len(parts) < 2:
            print(f"  {Colors.YELLOW}Usage: /model use <provider> <model_id>{Colors.RESET}")
            print(f"  {Colors.DIM}Example: /model use anthropic claude-sonnet-4-5-20250929{Colors.RESET}\n")
            return

        provider_name = parts[0].lower()
        model_id = parts[1]

        if provider_name not in ProviderFactory._providers:
            print(f"  {Colors.RED}Unknown provider '{provider_name}'. Available: {', '.join(ProviderFactory._providers.keys())}{Colors.RESET}\n")
            return

        self._apply_model_switch(provider_name, model_id)
        print(f"  {Colors.GREEN}✓ Switched to {provider_name}/{model_id}{Colors.RESET}\n")

    def _apply_model_switch(self, provider_name: str, model_id: str) -> None:
        """
        Actually switch the agent's provider and model at runtime.
        Creates a new provider instance and hot-swaps it on the agent.
        """
        # Build config for new provider
        pb = getattr(self.agent, "prompt_builder", None)
        config_data = getattr(pb, "config", {}) if pb else {}
        prov_config = config_data.get("providers", {}).get(provider_name, {})

        old_provider = self.agent.provider

        temp_config = {
            "llm": {
                "provider": provider_name,
                "model": model_id,
                "temperature": old_provider.temperature,
                "max_tokens": old_provider.max_tokens,
            },
            "providers": {provider_name: prov_config},
        }

        try:
            new_provider = ProviderFactory.create(temp_config)
            self.agent.provider = new_provider
            # Update config in prompt builder too
            if pb:
                pb.config.setdefault("llm", {})["provider"] = provider_name
                pb.config.setdefault("llm", {})["model"] = model_id
        except Exception as e:
            print(f"  {Colors.RED}Failed to switch: {e}{Colors.RESET}")

    @staticmethod
    def _rl_prompt(text: str) -> str:
        """Wrap ANSI escape codes with readline invisible markers.

        Readline counts every character for cursor positioning. Without
        \\001 / \\002 wrappers around non-printing ANSI escapes, it
        miscalculates the cursor column and garbles the line when the
        user presses arrow keys or edits text.
        """
        import re
        # Replace each ANSI escape sequence with \001...\002 wrapped version
        return re.sub(
            r'(\033\[[0-9;]*m)',
            lambda m: f'\001{m.group(1)}\002',
            text,
        )

    async def _async_input(self, prompt: str) -> str:
        """Read user input without blocking the async event loop.

        Readline (arrow keys, history, tab completion) only works on the
        main thread.  We run input() in a *dedicated* thread so the event
        loop can continue processing background tasks (Slack bot, API server)
        while still preserving full readline functionality.

        The key insight: run_in_executor gives us a Future that the event
        loop awaits, so background coroutines keep running.  And because
        we always use the *same* thread (the default executor), readline's
        terminal state stays consistent.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: input(prompt).strip(),
        )

    async def run(self) -> None:
        """Main interactive loop."""
        self._running = True
        self._setup_readline()
        self._print_banner()

        # Sprint 26: Start scheduler background loop if configured
        scheduler_task = None
        if hasattr(self.agent, 'task_scheduler') and self.agent.task_scheduler:
            async def _scheduler_agent_runner(prompt_text: str) -> str:
                return await self.agent.run(prompt_text)
            scheduler_task = asyncio.create_task(
                self.agent.task_scheduler.start(agent_runner=_scheduler_agent_runner)
            )
            # Also late-bind agent runner on the run-now tool
            if hasattr(self.agent, '_scheduler_run_tool') and self.agent._scheduler_run_tool:
                self.agent._scheduler_run_tool.set_agent_runner(_scheduler_agent_runner)

        # Build the prompt once with proper readline escape wrapping
        prompt = self._rl_prompt(
            f"{Colors.BOLD}{Colors.BLUE}You ▸ {Colors.RESET}"
        )

        while self._running:
            try:
                user_input = await self._async_input(prompt)

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
                        with _stdout_lock:
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

        # Sprint 26: Stop scheduler background loop
        if scheduler_task and not scheduler_task.done():
            if hasattr(self.agent, 'task_scheduler') and self.agent.task_scheduler:
                self.agent.task_scheduler.stop()
            scheduler_task.cancel()
            try:
                await scheduler_task
            except asyncio.CancelledError:
                pass

        # Stop any running remote services before exit
        if self._remote_services:
            await self._stop_all_remote()
        self._save_readline()
        print(f"\n{Colors.DIM}Session ended.{Colors.RESET}")
