"""
CLI Interface — Interactive terminal chat with the agent.
Supports /commands, streaming display, tool execution indicators, and todo widget.
"""

from __future__ import annotations
import asyncio
import os
import sys
import signal
import readline
import threading
import time
from datetime import datetime
from typing import Optional

from ..core.agent import Agent
from ..core.models import ToolCall, ToolResult


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


class Spinner:
    """A simple terminal spinner that runs in a background thread."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Thinking"):
        self._message = message
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, message: str = None) -> None:
        if message:
            self._message = message
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        # Clear the spinner line
        sys.stdout.write(f"\r{' ' * 60}\r")
        sys.stdout.flush()

    def _spin(self) -> None:
        idx = 0
        start = time.time()
        while self._running:
            elapsed = int(time.time() - start)
            frame = self.FRAMES[idx % len(self.FRAMES)]
            sys.stdout.write(
                f"\r  {Colors.DIM}{frame} {self._message}... ({elapsed}s){Colors.RESET}"
            )
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)


class CLI:
    """Interactive terminal interface for the agent."""

    def __init__(
        self,
        agent: Agent,
        history_file: str = "~/.cowork_agent_history",
    ):
        self.agent = agent
        self.history_file = os.path.expanduser(history_file)
        self._running = False
        self._spinner = Spinner()

        # Wire up callbacks
        self.agent.on_tool_start = self._on_tool_start
        self.agent.on_tool_end = self._on_tool_end
        self.agent.on_status = self._on_status

    def _on_tool_start(self, call: ToolCall) -> None:
        """Display tool execution indicator."""
        self._spinner.stop()  # Stop spinner before printing
        icon = self._tool_icon(call.name)
        print(
            f"  {Colors.DIM}{icon} Executing {Colors.CYAN}{call.name}"
            f"{Colors.RESET}{Colors.DIM}...{Colors.RESET}"
        )

    def _on_tool_end(self, call: ToolCall, result: ToolResult) -> None:
        """Display tool result status."""
        self._spinner.stop()
        if result.success:
            # Show truncated output
            preview = result.output[:120].replace("\n", " ")
            if len(result.output) > 120:
                preview += "..."
            print(
                f"  {Colors.GREEN}✓{Colors.RESET} {Colors.DIM}{preview}{Colors.RESET}"
            )
        else:
            print(
                f"  {Colors.RED}✗ {result.error}{Colors.RESET}"
            )
        # Restart spinner — agent will call LLM again after tool results
        self._spinner.start("Thinking")

    def _on_status(self, message: str) -> None:
        """Display agent status updates (retries, nudges, etc.)."""
        self._spinner.stop()
        print(
            f"  {Colors.YELLOW}⟳ {message}{Colors.RESET}"
        )
        self._spinner.start("Retrying")

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

    def _handle_command(self, cmd: str) -> bool:
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

        elif command in ("/exit", "/quit", "/q"):
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

    def _show_config(self) -> None:
        provider = self.agent.provider
        print(f"""
  {Colors.BOLD}Provider:{Colors.RESET} {provider.__class__.__name__}
  {Colors.BOLD}Tools:{Colors.RESET} {', '.join(self.agent.registry.tool_names)}
  {Colors.BOLD}Max iterations:{Colors.RESET} {self.agent.max_iterations}
""")

    async def run(self) -> None:
        """Main interactive loop."""
        self._running = True
        self._setup_readline()
        self._print_banner()

        while self._running:
            try:
                # Get user input
                user_input = input(
                    f"{Colors.BOLD}{Colors.BLUE}You ▸ {Colors.RESET}"
                ).strip()

                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    handled = self._handle_command(user_input)
                    if handled:
                        continue

                # Send to agent
                print()
                self._spinner.start("Thinking")
                try:
                    response = await self.agent.run(user_input)
                finally:
                    self._spinner.stop()

                # Display response
                print(f"\n{Colors.BOLD}{Colors.GREEN}Agent ▸{Colors.RESET} {response}\n")

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
