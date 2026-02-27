"""
Agent Registry — Lifecycle management for multiple named agents.

Manages a pool of Agent instances with state tracking, capability-based
tool filtering, and async lifecycle control (create, start, pause, resume,
terminate).

Sprint 5 (P3-Multi-Agent Orchestration) Feature 1.
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from .agent import Agent
from .models import AgentResponse
from .providers.base import BaseLLMProvider
from .tool_registry import ToolRegistry
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Lifecycle states for a managed agent."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Declarative agent definition."""
    name: str
    role: str                          # e.g. "code_reader", "analyzer"
    description: str = ""
    capabilities: list[str] = field(default_factory=list)  # Tool names allowed
    provider_config: dict = field(default_factory=dict)
    max_iterations: int = 15
    timeout_seconds: float = 300.0
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentInstance:
    """Runtime wrapper tracking an agent's lifecycle."""
    config: AgentConfig
    agent: Agent
    state: AgentState = AgentState.IDLE
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self.state == AgentState.RUNNING

    @property
    def is_completed(self) -> bool:
        return self.state in (AgentState.COMPLETED, AgentState.ERROR)

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at

    def to_dict(self) -> dict:
        return {
            "name": self.config.name,
            "role": self.config.role,
            "state": self.state.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed": round(self.elapsed, 2),
            "error": self.error,
            "has_result": self.result is not None,
        }


class AgentRegistry:
    """
    Manages a pool of named agents with lifecycle control.

    Usage:
        registry = AgentRegistry()

        config = AgentConfig(name="reader", role="file_reader",
                             capabilities=["read", "glob"])
        instance = await registry.create_agent(config, provider, tools, prompt_builder)

        task = await registry.start_agent("reader", "Read all .py files")
        result = await registry.get_result("reader", timeout=60)
    """

    def __init__(self):
        self._agents: dict[str, AgentInstance] = {}
        self._agent_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def create_agent(
        self,
        config: AgentConfig,
        provider: BaseLLMProvider,
        registry: ToolRegistry,
        prompt_builder: PromptBuilder,
        **agent_kwargs,
    ) -> AgentInstance:
        """Create and register a new agent from config."""
        async with self._lock:
            if config.name in self._agents:
                raise ValueError(f"Agent '{config.name}' already exists")

        # Filter tool registry to only allowed capabilities
        filtered_registry = self._filter_registry(registry, config.capabilities)

        agent = Agent(
            provider=provider,
            registry=filtered_registry,
            prompt_builder=prompt_builder,
            max_iterations=config.max_iterations,
            **agent_kwargs,
        )

        instance = AgentInstance(config=config, agent=agent)

        async with self._lock:
            self._agents[config.name] = instance

        logger.info(f"Created agent '{config.name}' (role={config.role}, "
                     f"tools={config.capabilities})")
        return instance

    async def start_agent(self, agent_name: str, task: str) -> asyncio.Task:
        """Start an agent on a task. Returns the asyncio.Task."""
        async with self._lock:
            if agent_name not in self._agents:
                raise KeyError(f"Agent '{agent_name}' not found")
            instance = self._agents[agent_name]
            if instance.state == AgentState.RUNNING:
                raise RuntimeError(f"Agent '{agent_name}' is already running")
            instance.state = AgentState.RUNNING
            instance.started_at = time.time()
            instance.completed_at = None
            instance.result = None
            instance.error = None

        async def _run_wrapper():
            try:
                result = await instance.agent.run(task)
                async with self._lock:
                    instance.state = AgentState.COMPLETED
                    instance.result = result
                    instance.completed_at = time.time()
                return result
            except asyncio.CancelledError:
                async with self._lock:
                    if instance.state == AgentState.RUNNING:
                        instance.state = AgentState.PAUSED
                raise
            except Exception as e:
                async with self._lock:
                    instance.state = AgentState.ERROR
                    instance.error = str(e)
                    instance.completed_at = time.time()
                raise

        atask = asyncio.create_task(_run_wrapper())
        self._agent_tasks[agent_name] = atask
        logger.info(f"Started agent '{agent_name}' on task: {task[:80]}...")
        return atask

    async def pause_agent(self, agent_name: str) -> None:
        """Pause a running agent by cancelling its task."""
        async with self._lock:
            if agent_name not in self._agents:
                raise KeyError(f"Agent '{agent_name}' not found")
            instance = self._agents[agent_name]
            if instance.state != AgentState.RUNNING:
                raise RuntimeError(f"Agent '{agent_name}' is not running (state={instance.state.value})")
            instance.state = AgentState.PAUSED

        atask = self._agent_tasks.get(agent_name)
        if atask and not atask.done():
            atask.cancel()
            try:
                await atask
            except (asyncio.CancelledError, Exception):
                pass

        logger.info(f"Paused agent '{agent_name}'")

    async def resume_agent(self, agent_name: str, task: str) -> asyncio.Task:
        """Resume a paused agent with a new task."""
        async with self._lock:
            if agent_name not in self._agents:
                raise KeyError(f"Agent '{agent_name}' not found")
            instance = self._agents[agent_name]
            if instance.state not in (AgentState.PAUSED, AgentState.COMPLETED, AgentState.ERROR):
                raise RuntimeError(
                    f"Agent '{agent_name}' cannot be resumed (state={instance.state.value})")

        return await self.start_agent(agent_name, task)

    async def get_result(self, agent_name: str, timeout: Optional[float] = None) -> str:
        """Wait for agent to complete and return its result."""
        atask = self._agent_tasks.get(agent_name)
        if not atask:
            # Check if already completed
            async with self._lock:
                instance = self._agents.get(agent_name)
                if instance and instance.result is not None:
                    return instance.result
            raise KeyError(f"No running task for agent '{agent_name}'")

        try:
            result = await asyncio.wait_for(asyncio.shield(atask), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            async with self._lock:
                instance = self._agents.get(agent_name)
                if instance:
                    instance.state = AgentState.ERROR
                    instance.error = f"Timeout after {timeout}s"
                    instance.completed_at = time.time()
            raise

    async def terminate_agent(self, agent_name: str) -> None:
        """Force-terminate an agent."""
        async with self._lock:
            if agent_name not in self._agents:
                raise KeyError(f"Agent '{agent_name}' not found")
            instance = self._agents[agent_name]
            instance.state = AgentState.ERROR
            instance.error = "Terminated"
            instance.completed_at = time.time()

        atask = self._agent_tasks.get(agent_name)
        if atask and not atask.done():
            atask.cancel()
            try:
                await atask
            except (asyncio.CancelledError, Exception):
                pass

        logger.info(f"Terminated agent '{agent_name}'")

    def list_agents(self) -> list[str]:
        """Get all registered agent names."""
        return list(self._agents.keys())

    def get_agent_status(self, agent_name: str) -> AgentInstance:
        """Get the AgentInstance for inspection."""
        if agent_name not in self._agents:
            raise KeyError(f"Agent '{agent_name}' not found")
        return self._agents[agent_name]

    def get_all_statuses(self) -> dict[str, dict]:
        """Get status dicts for all agents."""
        return {name: inst.to_dict() for name, inst in self._agents.items()}

    async def remove_agent(self, agent_name: str) -> None:
        """Remove a non-running agent from the registry."""
        async with self._lock:
            instance = self._agents.get(agent_name)
            if not instance:
                raise KeyError(f"Agent '{agent_name}' not found")
            if instance.state == AgentState.RUNNING:
                raise RuntimeError("Cannot remove a running agent")
            del self._agents[agent_name]
            self._agent_tasks.pop(agent_name, None)

    @staticmethod
    def _filter_registry(
        registry: ToolRegistry,
        capabilities: list[str],
    ) -> ToolRegistry:
        """Create a filtered ToolRegistry with only allowed tools."""
        if not capabilities:
            return registry  # Empty capabilities = all tools allowed

        filtered = ToolRegistry()
        for tool_name in capabilities:
            try:
                tool = registry.get_tool(tool_name)
                filtered.register(tool)
            except KeyError:
                logger.warning(f"Capability tool '{tool_name}' not found in registry")
        return filtered
