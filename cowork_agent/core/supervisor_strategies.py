"""
Supervisor Strategies — MAP_REDUCE, DEBATE, VOTING execution strategies.

Extends the Supervisor with three advanced multi-agent orchestration patterns:
  - MAP_REDUCE: Split task → parallel execution → merge results
  - DEBATE: Round-robin argumentation → scoring → winner selection
  - VOTING: Independent solutions → cross-voting → consensus

Sprint 21 (Multi-Agent Orchestration Enhancement) Module 1.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


# ── Strategy Interface ────────────────────────────────────────────


class BaseStrategy(ABC):
    """Abstract strategy interface for multi-agent execution."""

    @abstractmethod
    async def execute(
        self,
        task: str,
        agent_runner: Callable,
        agent_names: List[str],
        **kwargs,
    ) -> "StrategyResult":
        """
        Execute the strategy.

        Args:
            task: The task description
            agent_runner: Async callable (agent_name, prompt) -> result_str
            agent_names: List of agent names to use
        """
        ...


@dataclass
class StrategyResult:
    """Result from a strategy execution."""
    final_output: str
    agent_outputs: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    strategy: str = ""
    elapsed_seconds: float = 0.0
    rounds: int = 0


# ── MAP_REDUCE Strategy ──────────────────────────────────────────


@dataclass
class MapReduceConfig:
    """Configuration for MAP_REDUCE strategy."""
    subtask_template: str = "Subtask for {agent}: {task}"
    merge_template: str = (
        "Merge and synthesize the following results into a coherent output:\n\n{results}"
    )
    max_parallel: int = 5
    merge_agent: Optional[str] = None  # If None, use simple concatenation


class MapReduceStrategy(BaseStrategy):
    """
    MAP_REDUCE: Split task into subtasks, execute in parallel, merge results.

    1. MAP phase: Each agent receives the task (or a subtask variant)
    2. REDUCE phase: Results are merged by a designated agent or concatenated

    Usage::

        strategy = MapReduceStrategy(config=MapReduceConfig(max_parallel=3))
        result = await strategy.execute(task, runner, agents)
    """

    def __init__(self, config: Optional[MapReduceConfig] = None):
        self.config = config or MapReduceConfig()

    async def execute(
        self,
        task: str,
        agent_runner: Callable,
        agent_names: List[str],
        **kwargs,
    ) -> StrategyResult:
        start = time.time()

        # MAP phase: run all agents in parallel
        semaphore = asyncio.Semaphore(self.config.max_parallel)

        async def _run_agent(name: str) -> tuple:
            async with semaphore:
                prompt = self.config.subtask_template.format(agent=name, task=task)
                try:
                    result = await agent_runner(name, prompt)
                    return (name, result, None)
                except Exception as e:
                    return (name, None, str(e))

        tasks = [_run_agent(name) for name in agent_names]
        raw_results = await asyncio.gather(*tasks)

        agent_outputs = {}
        errors = {}
        for name, result, error in raw_results:
            if error:
                errors[name] = error
                agent_outputs[name] = f"[ERROR: {error}]"
            else:
                agent_outputs[name] = result or ""

        # REDUCE phase: merge results
        successful_outputs = {
            k: v for k, v in agent_outputs.items() if k not in errors
        }

        if self.config.merge_agent and self.config.merge_agent in agent_names:
            # Use a designated agent to merge
            combined = "\n\n".join(
                f"=== {name} ===\n{output}"
                for name, output in successful_outputs.items()
            )
            merge_prompt = self.config.merge_template.format(results=combined)
            try:
                final_output = await agent_runner(self.config.merge_agent, merge_prompt)
            except Exception as e:
                final_output = f"Merge failed: {e}\n\n{combined}"
        else:
            # Simple concatenation
            final_output = "\n\n".join(
                f"=== {name} ===\n{output}"
                for name, output in agent_outputs.items()
            )

        return StrategyResult(
            final_output=final_output,
            agent_outputs=agent_outputs,
            metadata={
                "errors": errors,
                "successful": len(successful_outputs),
                "failed": len(errors),
                "merge_agent": self.config.merge_agent,
            },
            strategy="map_reduce",
            elapsed_seconds=time.time() - start,
            rounds=1,
        )


# ── DEBATE Strategy ──────────────────────────────────────────────


@dataclass
class DebateConfig:
    """Configuration for DEBATE strategy."""
    max_rounds: int = 3
    opening_template: str = "Present your argument on: {task}"
    rebuttal_template: str = (
        "The previous argument was:\n{previous}\n\n"
        "Present your counter-argument or build upon this:"
    )
    judge_template: str = (
        "As a judge, evaluate the following debate arguments and pick the best one. "
        "Explain your reasoning.\n\n{arguments}"
    )
    judge_agent: Optional[str] = None  # If None, last agent is judge


class DebateStrategy(BaseStrategy):
    """
    DEBATE: Agents argue in rounds, a judge picks the winner.

    1. Round 1: Each agent presents an opening argument
    2. Rounds 2+: Each agent rebuts the previous argument
    3. Judging: A designated agent (or the supervisor) scores and picks a winner

    Usage::

        strategy = DebateStrategy(config=DebateConfig(max_rounds=3))
        result = await strategy.execute(task, runner, agents)
    """

    def __init__(self, config: Optional[DebateConfig] = None):
        self.config = config or DebateConfig()

    async def execute(
        self,
        task: str,
        agent_runner: Callable,
        agent_names: List[str],
        **kwargs,
    ) -> StrategyResult:
        if len(agent_names) < 2:
            raise ValueError("DEBATE requires at least 2 agents")

        start = time.time()
        agent_outputs = {name: [] for name in agent_names}
        all_rounds: List[Dict[str, str]] = []

        # Round 1: Opening arguments
        round_results = {}
        for name in agent_names:
            prompt = self.config.opening_template.format(task=task)
            try:
                result = await agent_runner(name, prompt)
                round_results[name] = result or ""
                agent_outputs[name].append(result or "")
            except Exception as e:
                round_results[name] = f"[ERROR: {e}]"
                agent_outputs[name].append(f"[ERROR: {e}]")
        all_rounds.append(dict(round_results))

        # Subsequent rounds: rebuttals
        for round_num in range(1, self.config.max_rounds):
            round_results_new = {}
            for i, name in enumerate(agent_names):
                # Each agent rebuts the previous agent's argument
                prev_idx = (i - 1) % len(agent_names)
                prev_name = agent_names[prev_idx]
                prev_arg = round_results.get(prev_name, "")

                prompt = self.config.rebuttal_template.format(previous=prev_arg)
                try:
                    result = await agent_runner(name, prompt)
                    round_results_new[name] = result or ""
                    agent_outputs[name].append(result or "")
                except Exception as e:
                    round_results_new[name] = f"[ERROR: {e}]"
                    agent_outputs[name].append(f"[ERROR: {e}]")

            round_results = round_results_new
            all_rounds.append(dict(round_results))

        # Judging phase
        arguments_text = "\n\n".join(
            f"=== {name} (final round) ===\n{round_results.get(name, '')}"
            for name in agent_names
        )
        judge_prompt = self.config.judge_template.format(arguments=arguments_text)

        judge = self.config.judge_agent or agent_names[-1]
        try:
            judgment = await agent_runner(judge, judge_prompt)
        except Exception as e:
            judgment = f"Judging failed: {e}"

        # Combine final output
        final_lines = [f"=== DEBATE: {task} ===\n"]
        for r_idx, rd in enumerate(all_rounds):
            final_lines.append(f"\n--- Round {r_idx + 1} ---")
            for name, arg in rd.items():
                final_lines.append(f"[{name}]: {arg}")
        final_lines.append(f"\n--- Judge ({judge}) ---")
        final_lines.append(judgment or "[No judgment]")

        return StrategyResult(
            final_output="\n".join(final_lines),
            agent_outputs={
                name: "\n---\n".join(outputs)
                for name, outputs in agent_outputs.items()
            },
            metadata={
                "rounds": len(all_rounds),
                "judge": judge,
                "judgment": judgment,
            },
            strategy="debate",
            elapsed_seconds=time.time() - start,
            rounds=len(all_rounds),
        )


# ── VOTING Strategy ──────────────────────────────────────────────


@dataclass
class VotingConfig:
    """Configuration for VOTING strategy."""
    solution_template: str = "Provide your solution for: {task}"
    vote_template: str = (
        "Review these solutions and vote for the best one (respond with the "
        "solution number only, e.g. '1' or '2'):\n\n{solutions}"
    )
    consensus_threshold: float = 0.5  # Fraction needed for consensus
    allow_self_vote: bool = False


class VotingStrategy(BaseStrategy):
    """
    VOTING: Agents generate solutions, then vote on the best one.

    1. Generation: Each agent independently produces a solution
    2. Voting: Each agent votes on all solutions (optionally excluding self)
    3. Consensus: The solution with the most votes wins

    Usage::

        strategy = VotingStrategy(config=VotingConfig())
        result = await strategy.execute(task, runner, agents)
    """

    def __init__(self, config: Optional[VotingConfig] = None):
        self.config = config or VotingConfig()

    async def execute(
        self,
        task: str,
        agent_runner: Callable,
        agent_names: List[str],
        **kwargs,
    ) -> StrategyResult:
        if len(agent_names) < 2:
            raise ValueError("VOTING requires at least 2 agents")

        start = time.time()

        # Phase 1: Generate solutions
        solutions: Dict[str, str] = {}
        for name in agent_names:
            prompt = self.config.solution_template.format(task=task)
            try:
                result = await agent_runner(name, prompt)
                solutions[name] = result or ""
            except Exception as e:
                solutions[name] = f"[ERROR: {e}]"

        # Phase 2: Voting
        # Build solutions text for voting prompt
        solution_list = list(solutions.items())
        solutions_text = "\n\n".join(
            f"Solution {i+1} (by {name}):\n{sol}"
            for i, (name, sol) in enumerate(solution_list)
        )
        vote_prompt = self.config.vote_template.format(solutions=solutions_text)

        votes: Dict[str, int] = {}  # agent_name -> solution_index (1-based)
        vote_counts: Dict[int, int] = {i+1: 0 for i in range(len(solution_list))}

        for voter_name in agent_names:
            try:
                vote_result = await agent_runner(voter_name, vote_prompt)
                # Parse the vote — look for a number
                vote_num = self._parse_vote(vote_result, len(solution_list))
                if vote_num is not None:
                    # Check self-vote
                    voted_for = solution_list[vote_num - 1][0]
                    if not self.config.allow_self_vote and voted_for == voter_name:
                        # Skip self-votes
                        votes[voter_name] = -1
                    else:
                        votes[voter_name] = vote_num
                        vote_counts[vote_num] = vote_counts.get(vote_num, 0) + 1
                else:
                    votes[voter_name] = -1  # Invalid vote
            except Exception:
                votes[voter_name] = -1

        # Phase 3: Determine winner
        if vote_counts:
            winner_idx = max(vote_counts, key=lambda k: vote_counts[k])
            winner_votes = vote_counts[winner_idx]
            total_valid = sum(1 for v in votes.values() if v > 0)
            consensus = (winner_votes / max(total_valid, 1))
        else:
            winner_idx = 1
            consensus = 0.0

        winner_name, winner_solution = solution_list[winner_idx - 1]
        has_consensus = consensus >= self.config.consensus_threshold

        # Build final output
        final_lines = [f"=== VOTING: {task} ===\n"]
        final_lines.append(f"Winner: {winner_name} (Solution {winner_idx})")
        final_lines.append(
            f"Votes: {vote_counts[winner_idx]}/{sum(vote_counts.values())} "
            f"({'consensus' if has_consensus else 'no consensus'})"
        )
        final_lines.append(f"\n--- Winning Solution ---\n{winner_solution}")

        return StrategyResult(
            final_output="\n".join(final_lines),
            agent_outputs=solutions,
            metadata={
                "votes": votes,
                "vote_counts": vote_counts,
                "winner_idx": winner_idx,
                "winner_name": winner_name,
                "consensus": consensus,
                "has_consensus": has_consensus,
            },
            strategy="voting",
            elapsed_seconds=time.time() - start,
            rounds=2,  # generate + vote
        )

    @staticmethod
    def _parse_vote(vote_text: str, max_choices: int) -> Optional[int]:
        """Parse a vote response into a valid choice number."""
        if not vote_text:
            return None
        # Look for digits in the response
        for char in vote_text.strip():
            if char.isdigit():
                num = int(char)
                if 1 <= num <= max_choices:
                    return num
        return None
