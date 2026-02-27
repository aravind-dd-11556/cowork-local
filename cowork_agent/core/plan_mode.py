"""
Plan Mode — Allows the agent to think through complex tasks before executing.

Mirrors real Cowork's EnterPlanMode/ExitPlanMode:
  - Agent explores codebase, reads files, searches for patterns (read-only tools)
  - Agent writes a plan to a plan file
  - User approves or rejects the plan
  - On approval, agent switches to execution mode and follows the plan

States:
  NORMAL  — default, all tools available
  PLAN    — read-only tools + plan writing, no writes/edits/bash mutations
"""

from __future__ import annotations
import logging
import os
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    NORMAL = "normal"
    PLAN = "plan"


# Tools allowed in plan mode (read-only exploration + mode transition tools)
PLAN_MODE_ALLOWED_TOOLS = {
    "read", "glob", "grep", "web_search", "web_fetch",
    "todo_write", "ask_user",
    "enter_plan_mode", "exit_plan_mode",  # Must be able to exit plan mode!
}

# Tools that are never allowed in plan mode
PLAN_MODE_BLOCKED_TOOLS = {
    "bash", "write", "edit",
}


class PlanManager:
    """
    Manages plan mode state and plan file.

    The plan file is stored at workspace/.cowork/plan.md
    and contains the agent's implementation plan.
    """

    def __init__(self, workspace_dir: str = ""):
        self.workspace_dir = workspace_dir
        self._mode = AgentMode.NORMAL
        self._plan_content: str = ""
        self._plan_file = ""

        if workspace_dir:
            cowork_dir = os.path.join(workspace_dir, ".cowork")
            os.makedirs(cowork_dir, exist_ok=True)
            self._plan_file = os.path.join(cowork_dir, "plan.md")

    @property
    def mode(self) -> AgentMode:
        return self._mode

    @property
    def is_plan_mode(self) -> bool:
        return self._mode == AgentMode.PLAN

    @property
    def plan_content(self) -> str:
        return self._plan_content

    def enter_plan_mode(self) -> str:
        """Switch to plan mode. Returns status message."""
        if self._mode == AgentMode.PLAN:
            return "Already in plan mode."

        self._mode = AgentMode.PLAN
        self._plan_content = ""
        logger.info("Entered plan mode")
        return (
            "Entered plan mode. You can now explore the codebase using read-only tools "
            "(read, glob, grep, web_search, web_fetch). "
            "When your plan is ready, call exit_plan_mode with your plan."
        )

    def exit_plan_mode(self, plan: str = "") -> str:
        """
        Exit plan mode and return to normal execution.

        Args:
            plan: The implementation plan text. If provided, it's saved
                  and presented to the user for approval.

        Returns:
            Status message with the plan summary.
        """
        if self._mode != AgentMode.PLAN:
            return "Not in plan mode."

        self._mode = AgentMode.NORMAL

        if plan:
            self._plan_content = plan
            self._save_plan(plan)
            logger.info(f"Plan saved ({len(plan)} chars)")
            return f"Plan saved. Exiting plan mode.\n\n{plan}"
        else:
            logger.info("Exited plan mode without a plan")
            return "Exited plan mode without saving a plan."

    def is_tool_allowed(self, tool_name: str) -> tuple[bool, str]:
        """
        Check if a tool is allowed in the current mode.

        Returns:
            (allowed, reason) — reason is empty string if allowed
        """
        if self._mode == AgentMode.NORMAL:
            return True, ""

        # Plan mode — only read-only tools
        if tool_name in PLAN_MODE_BLOCKED_TOOLS:
            return False, (
                f"Tool '{tool_name}' is not available in plan mode. "
                f"Plan mode only allows read-only exploration tools: "
                f"{', '.join(sorted(PLAN_MODE_ALLOWED_TOOLS))}. "
                f"Call exit_plan_mode first to switch back to normal mode."
            )

        if tool_name not in PLAN_MODE_ALLOWED_TOOLS:
            return False, (
                f"Tool '{tool_name}' is not available in plan mode. "
                f"Allowed tools: {', '.join(sorted(PLAN_MODE_ALLOWED_TOOLS))}."
            )

        return True, ""

    def _save_plan(self, plan: str) -> None:
        """Save plan to disk."""
        if not self._plan_file:
            return
        try:
            with open(self._plan_file, "w") as f:
                f.write(plan)
        except Exception as e:
            logger.warning(f"Failed to save plan: {e}")

    def get_plan_mode_prompt(self) -> str:
        """Get additional system prompt instructions for plan mode."""
        if not self.is_plan_mode:
            return ""

        return """
<plan_mode>
You are currently in PLAN MODE. This means:

1. You can ONLY use read-only tools: read, glob, grep, web_search, web_fetch, todo_write, ask_user
2. You CANNOT use: bash, write, edit (these modify files)
3. Your goal is to:
   - Explore the codebase thoroughly
   - Understand existing patterns and architecture
   - Design an implementation approach
   - Present your plan for user approval

4. When your plan is ready, call the exit_plan_mode tool with your complete plan.
5. The plan should include:
   - Files to modify/create
   - Step-by-step implementation approach
   - Any architectural decisions or trade-offs
   - Potential risks or issues

Do NOT attempt to write any code or modify any files until the plan is approved.
</plan_mode>
"""
