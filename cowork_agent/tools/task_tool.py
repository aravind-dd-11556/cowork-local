"""
Task Tool — Subagent delegation for complex, multi-step tasks.

Mirrors real Cowork's Task tool:
  - Spawns a new agent instance to handle a sub-task
  - The subagent has its own conversation loop
  - Results are returned to the parent agent
  - Supports isolation (separate conversation context)
  - Recursion depth capped to prevent infinite nesting

Sprint 30 additions:
  - subagent_type: 6 specialized agent profiles (Bash, Explore, Plan,
    general-purpose, claude-code-guide, statusline-setup)
  - isolation: "worktree" mode — run subagent in an isolated git worktree
  - resume: Continue a previous agent by ID instead of starting fresh
  - model: Optional model override for the subagent
"""

from __future__ import annotations
import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .base import BaseTool

if TYPE_CHECKING:
    from ..core.agent import Agent

logger = logging.getLogger(__name__)

# Module-level depth tracker (thread-safe via asyncio single-thread model)
_current_depth = 0


# ── Agent Type Profiles ─────────────────────────────────────────────────

@dataclass
class AgentTypeProfile:
    """
    Defines a specialized agent type with its own tool filter,
    system instructions, and capabilities.
    """
    name: str
    description: str
    tool_filter: Optional[List[str]] = None   # None = all tools
    excluded_tools: List[str] = field(default_factory=list)
    system_instructions: str = ""
    max_turns_default: int = 10

    def filter_tools(self, available_tools: List[str]) -> List[str]:
        """Return the subset of tools this agent type should have access to."""
        if self.tool_filter is not None:
            # Whitelist mode: only these tools
            return [t for t in available_tools if t in self.tool_filter]
        elif self.excluded_tools:
            # Blacklist mode: all except these
            return [t for t in available_tools if t not in self.excluded_tools]
        return available_tools


# The 6 agent types from real Cowork
AGENT_TYPE_PROFILES: Dict[str, AgentTypeProfile] = {
    "Bash": AgentTypeProfile(
        name="Bash",
        description="Command execution specialist for running bash commands. "
                    "Use this for git operations, command execution, and other terminal tasks.",
        tool_filter=["bash"],
        system_instructions=(
            "You are a Bash command execution specialist. Focus on running "
            "shell commands efficiently. You have access only to the Bash tool."
        ),
        max_turns_default=10,
    ),
    "Explore": AgentTypeProfile(
        name="Explore",
        description="Fast agent specialized for exploring codebases. Use this when you "
                    "need to quickly find files by patterns, search code for keywords, "
                    "or answer questions about the codebase.",
        excluded_tools=["task", "edit", "write", "notebook_edit"],
        system_instructions=(
            "You are a codebase exploration specialist. Your job is to search, "
            "read, and understand code. You cannot edit or write files — only "
            "read and search. Be thorough but efficient."
        ),
        max_turns_default=15,
    ),
    "Plan": AgentTypeProfile(
        name="Plan",
        description="Software architect agent for designing implementation plans. "
                    "Use this when you need to plan the implementation strategy for a task.",
        excluded_tools=["task", "edit", "write", "notebook_edit"],
        system_instructions=(
            "You are a software architect. Design implementation plans by "
            "exploring the codebase, understanding patterns, and proposing "
            "step-by-step strategies. You cannot modify files — only read and plan."
        ),
        max_turns_default=15,
    ),
    "general-purpose": AgentTypeProfile(
        name="general-purpose",
        description="General-purpose agent for researching complex questions, "
                    "searching for code, and executing multi-step tasks.",
        # All tools available
        tool_filter=None,
        system_instructions="",
        max_turns_default=10,
    ),
    "claude-code-guide": AgentTypeProfile(
        name="claude-code-guide",
        description="Use this agent when the user asks questions about Claude Code, "
                    "Claude Agent SDK, or Claude API features, hooks, settings, etc.",
        tool_filter=["glob_tool", "grep_tool", "read", "web_fetch", "web_search"],
        system_instructions=(
            "You are a Claude Code documentation specialist. Help users "
            "understand Claude Code features, hooks, slash commands, MCP servers, "
            "settings, IDE integrations, and keyboard shortcuts. Also covers "
            "Claude Agent SDK and Claude API usage."
        ),
        max_turns_default=10,
    ),
    "statusline-setup": AgentTypeProfile(
        name="statusline-setup",
        description="Use this agent to configure the user's status line setting.",
        tool_filter=["read", "edit"],
        system_instructions=(
            "You are a configuration specialist. Help the user set up and "
            "configure their status line settings by reading and editing config files."
        ),
        max_turns_default=5,
    ),
}


# ── Agent Session Store (for resume) ────────────────────────────────────

@dataclass
class AgentSession:
    """Stores a completed or suspended agent session for potential resume."""
    agent_id: str
    subagent_type: str
    description: str
    prompt: str
    result: str
    messages: List[Any] = field(default_factory=list)  # Conversation history
    worktree_path: Optional[str] = None
    worktree_branch: Optional[str] = None
    completed: bool = True


class AgentSessionStore:
    """In-memory store for agent sessions, enabling resume by ID."""

    def __init__(self, max_sessions: int = 100):
        self._sessions: Dict[str, AgentSession] = {}
        self._max_sessions = max_sessions

    def save(self, session: AgentSession) -> str:
        """Save a session, evicting oldest if at capacity."""
        if len(self._sessions) >= self._max_sessions:
            # Evict oldest
            oldest_id = next(iter(self._sessions))
            del self._sessions[oldest_id]
        self._sessions[session.agent_id] = session
        return session.agent_id

    def get(self, agent_id: str) -> Optional[AgentSession]:
        """Retrieve a session by ID."""
        return self._sessions.get(agent_id)

    def remove(self, agent_id: str) -> bool:
        """Remove a session."""
        if agent_id in self._sessions:
            del self._sessions[agent_id]
            return True
        return False

    @property
    def session_ids(self) -> List[str]:
        return list(self._sessions.keys())

    def __len__(self) -> int:
        return len(self._sessions)


# Module-level session store (shared across TaskTool instances)
_session_store = AgentSessionStore()


def get_session_store() -> AgentSessionStore:
    """Access the global session store."""
    return _session_store


def reset_session_store() -> None:
    """Reset the global session store (for testing)."""
    global _session_store
    _session_store = AgentSessionStore()


# ── Task Tool ───────────────────────────────────────────────────────────

class TaskTool(BaseTool):
    """
    Spawn a subagent to handle a complex sub-task autonomously.

    The subagent gets a fresh conversation context, executes the task,
    and returns its final response to the parent agent.

    Sprint 30 features:
      - subagent_type: Choose a specialized agent profile
      - isolation: "worktree" runs in an isolated git worktree
      - resume: Continue a previous agent session by ID
      - model: Optional model override

    Recursion depth is capped at MAX_DEPTH to prevent infinite nesting.
    """
    name = "task"
    MAX_DEPTH = 3  # Maximum subagent nesting depth
    description = (
        "Launch a new agent to handle complex, multi-step tasks autonomously.\n\n"
        "Available agent types:\n"
        "- Bash: Command execution specialist for running bash commands.\n"
        "- Explore: Fast agent for exploring codebases (read-only).\n"
        "- Plan: Software architect for designing implementation plans.\n"
        "- general-purpose: General agent for research and multi-step tasks.\n"
        "- claude-code-guide: Documentation specialist for Claude Code/API.\n"
        "- statusline-setup: Configure status line settings.\n\n"
        "Set isolation='worktree' to run in an isolated git worktree.\n"
        "Use resume=<agent_id> to continue a previous agent's work."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short (3-5 word) description of the task",
            },
            "prompt": {
                "type": "string",
                "description": "Detailed instructions for the subagent",
            },
            "subagent_type": {
                "type": "string",
                "description": (
                    "The type of specialized agent to use. "
                    "Options: Bash, Explore, Plan, general-purpose, "
                    "claude-code-guide, statusline-setup"
                ),
                "enum": list(AGENT_TYPE_PROFILES.keys()),
            },
            "max_turns": {
                "type": "integer",
                "description": "Maximum iterations for the subagent",
                "exclusiveMinimum": 0,
            },
            "isolation": {
                "type": "string",
                "description": (
                    'Set to "worktree" to run the agent in a temporary git worktree, '
                    "giving it an isolated copy of the repository."
                ),
                "enum": ["worktree"],
            },
            "resume": {
                "type": "string",
                "description": (
                    "Optional agent ID to resume from. If provided, the agent "
                    "will continue from the previous execution transcript."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional model to use for this agent. "
                    'Options: "sonnet", "opus", "haiku"'
                ),
                "enum": ["sonnet", "opus", "haiku"],
            },
        },
        "required": ["description", "prompt", "subagent_type"],
    }

    def __init__(
        self,
        agent_factory: Callable[..., "Agent"],
        worktree_manager=None,
        workspace_dir: str = "",
    ):
        """
        Args:
            agent_factory: A callable that creates a fresh Agent instance.
                Accepts optional keyword args: tool_filter, system_instructions,
                max_iterations, workspace_dir.
            worktree_manager: Optional WorktreeManager for worktree isolation.
            workspace_dir: Default workspace directory.
        """
        self._agent_factory = agent_factory
        self._worktree_manager = worktree_manager
        self._workspace_dir = workspace_dir

    async def execute(
        self,
        *,
        progress_callback=None,
        description: str = "",
        prompt: str = "",
        subagent_type: str = "general-purpose",
        max_turns: int = 0,
        isolation: str = "",
        resume: str = "",
        model: str = "",
        **kwargs,
    ) -> "ToolResult":
        global _current_depth

        if not prompt:
            return self._error("Prompt is required for task delegation.")

        # ── Resume mode ────────────────────────────────────────────
        if resume:
            return await self._handle_resume(
                resume, prompt, description, progress_callback
            )

        # ── Resolve agent type profile ─────────────────────────────
        profile = AGENT_TYPE_PROFILES.get(subagent_type)
        if not profile:
            available = ", ".join(sorted(AGENT_TYPE_PROFILES.keys()))
            return self._error(
                f"Unknown subagent_type '{subagent_type}'. "
                f"Available types: {available}"
            )

        # ── Recursion depth check ──────────────────────────────────
        if _current_depth >= self.MAX_DEPTH:
            return self._error(
                f"Maximum subagent nesting depth ({self.MAX_DEPTH}) reached. "
                f"Cannot spawn another subagent. Complete the task directly instead."
            )

        # ── Determine max turns ────────────────────────────────────
        effective_max_turns = max_turns if max_turns > 0 else profile.max_turns_default

        # ── Worktree isolation setup ───────────────────────────────
        worktree_info = None
        subagent_workspace = self._workspace_dir
        if isolation == "worktree":
            worktree_info = self._setup_worktree(description)
            if worktree_info is None:
                return self._error(
                    "Failed to create isolated worktree. "
                    "Ensure you are in a git repository."
                )
            subagent_workspace = worktree_info.get("path", self._workspace_dir)

        # ── Build enriched prompt ──────────────────────────────────
        enriched_prompt = prompt
        if profile.system_instructions:
            enriched_prompt = (
                f"[Agent Type: {profile.name}]\n"
                f"{profile.system_instructions}\n\n"
                f"Task: {prompt}"
            )

        # ── Spawn and run subagent ─────────────────────────────────
        agent_id = f"agent_{uuid.uuid4().hex[:12]}"
        logger.info(
            f"Spawning {profile.name} subagent ({agent_id}) for: {description} "
            f"(depth {_current_depth + 1}/{self.MAX_DEPTH}, "
            f"max_turns={effective_max_turns})"
        )

        if progress_callback:
            progress_callback(-1, f"Launching {profile.name} agent...")

        try:
            # Create subagent with profile constraints
            subagent = self._agent_factory(
                tool_filter=profile.filter_tools,
                system_instructions=profile.system_instructions,
                max_iterations=effective_max_turns,
                workspace_dir=subagent_workspace,
            )

            # Track recursion depth
            _current_depth += 1
            try:
                result = await subagent.run(enriched_prompt)
            finally:
                _current_depth -= 1

            # Save session for potential resume
            session = AgentSession(
                agent_id=agent_id,
                subagent_type=subagent_type,
                description=description,
                prompt=prompt,
                result=result,
                messages=getattr(subagent, '_messages', []),
                worktree_path=worktree_info.get("path") if worktree_info else None,
                worktree_branch=worktree_info.get("branch") if worktree_info else None,
            )
            _session_store.save(session)

            # Cleanup worktree if no changes
            wt_note = ""
            if worktree_info:
                wt_note = self._cleanup_worktree(worktree_info, session)

            logger.info(
                f"Subagent {agent_id} completed task: {description} "
                f"({len(result)} chars)"
            )

            output_parts = [f"[Subagent result for: {description}]"]
            if wt_note:
                output_parts.append(wt_note)
            output_parts.append("")
            output_parts.append(result)

            return self._success(
                "\n".join(output_parts),
                agent_id=agent_id,
                subagent_type=subagent_type,
                worktree_path=worktree_info.get("path") if worktree_info else None,
                worktree_branch=worktree_info.get("branch") if worktree_info else None,
            )

        except Exception as e:
            logger.error(f"Subagent error: {e}")
            # Cleanup worktree on failure
            if worktree_info:
                self._force_cleanup_worktree(worktree_info)
            return self._error(f"Subagent failed: {str(e)}")

    async def _handle_resume(
        self,
        agent_id: str,
        prompt: str,
        description: str,
        progress_callback=None,
    ) -> "ToolResult":
        """Resume a previous agent session by ID."""
        session = _session_store.get(agent_id)
        if not session:
            available = _session_store.session_ids
            if available:
                return self._error(
                    f"Agent '{agent_id}' not found. "
                    f"Available agent IDs: {', '.join(available[-5:])}"
                )
            return self._error(
                f"Agent '{agent_id}' not found. No previous sessions available."
            )

        logger.info(f"Resuming agent {agent_id} (type={session.subagent_type})")
        if progress_callback:
            progress_callback(-1, f"Resuming agent {agent_id}...")

        profile = AGENT_TYPE_PROFILES.get(
            session.subagent_type, AGENT_TYPE_PROFILES["general-purpose"]
        )

        # Determine workspace — use worktree if session had one
        ws = session.worktree_path or self._workspace_dir

        try:
            subagent = self._agent_factory(
                tool_filter=profile.filter_tools,
                system_instructions=profile.system_instructions,
                max_iterations=profile.max_turns_default,
                workspace_dir=ws,
            )

            # Inject previous conversation context
            if session.messages:
                subagent._messages = list(session.messages)

            resume_prompt = (
                f"[Resuming previous session {agent_id}]\n"
                f"Previous task: {session.description}\n\n"
                f"Continue with: {prompt}"
            )

            global _current_depth
            _current_depth += 1
            try:
                result = await subagent.run(resume_prompt)
            finally:
                _current_depth -= 1

            # Update session
            session.result = result
            session.messages = getattr(subagent, '_messages', [])
            _session_store.save(session)

            return self._success(
                f"[Resumed agent {agent_id}: {description}]\n\n{result}",
                agent_id=agent_id,
                subagent_type=session.subagent_type,
                resumed=True,
            )

        except Exception as e:
            logger.error(f"Resume failed for {agent_id}: {e}")
            return self._error(f"Resume failed: {str(e)}")

    def _setup_worktree(self, description: str) -> Optional[Dict[str, str]]:
        """Create an isolated worktree for the subagent."""
        if not self._worktree_manager:
            logger.warning("Worktree isolation requested but no WorktreeManager available")
            return None

        try:
            # Generate a short name from the description
            safe_name = "".join(
                c if c.isalnum() or c == '-' else '-'
                for c in description.lower()[:30]
            ).strip('-') or "task"
            wt_name = f"{safe_name}-{uuid.uuid4().hex[:6]}"

            if not self._worktree_manager.is_git_repo():
                logger.warning("Not in a git repo — cannot create worktree")
                return None

            info = self._worktree_manager.create(name=wt_name)
            if info is None:
                return None

            logger.info(f"Created worktree '{info.name}' at {info.path}")
            return {
                "name": info.name,
                "path": info.path,
                "branch": info.branch,
                "created_from": getattr(info, 'created_from', 'HEAD'),
            }
        except Exception as e:
            logger.error(f"Worktree creation failed: {e}")
            return None

    def _cleanup_worktree(
        self, worktree_info: Dict[str, str], session: AgentSession
    ) -> str:
        """
        Cleanup worktree after subagent completes.
        If no changes were made, remove it automatically.
        If changes exist, keep it and return a note.
        """
        if not self._worktree_manager:
            return ""

        wt_name = worktree_info.get("name", "")
        try:
            worktrees = self._worktree_manager.list_worktrees()
            for wt in worktrees:
                if wt.name == wt_name:
                    if not wt.has_changes:
                        self._worktree_manager.remove(wt_name)
                        logger.info(f"Auto-removed clean worktree '{wt_name}'")
                        return "(Worktree auto-cleaned — no changes detected)"
                    else:
                        logger.info(f"Keeping worktree '{wt_name}' — has changes")
                        return (
                            f"(Worktree kept at {worktree_info['path']} "
                            f"on branch {worktree_info['branch']} — changes detected)"
                        )
        except Exception as e:
            logger.debug(f"Worktree cleanup check failed: {e}")
        return ""

    def _force_cleanup_worktree(self, worktree_info: Dict[str, str]) -> None:
        """Force-remove worktree on error."""
        if not self._worktree_manager:
            return
        try:
            self._worktree_manager.remove(worktree_info.get("name", ""), force=True)
        except Exception:
            pass

    @staticmethod
    def get_agent_types() -> Dict[str, AgentTypeProfile]:
        """Return all available agent type profiles."""
        return dict(AGENT_TYPE_PROFILES)

    @staticmethod
    def get_agent_type(name: str) -> Optional[AgentTypeProfile]:
        """Look up a specific agent type by name."""
        return AGENT_TYPE_PROFILES.get(name)
