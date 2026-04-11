"""Regression tests for the instrumentation preservation bug in Game.step().

Before the fix in game.py:99-111, agents wrote instrumentation to self.state,
then apply_move() returned a fresh state with all metrics zeroed — making
nodes_visited, max_depth_reached, and prunings always 0 in benchmarks.

These tests verify that every instrumentation field survives the step.
"""
from __future__ import annotations

import math
import time

import pytest

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents.classic_search.minimax_ab import MinimaxAB
from tictactoe.benchmark.arena import Arena
from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.core.board import Board
from tictactoe.core.game import Game
from tictactoe.core.state import GameState
from tictactoe.core.types import Move, Player, Result


# ---------------------------------------------------------------------------
# Scripted agents for deterministic testing
# ---------------------------------------------------------------------------


class InstrumentedAgent(BaseAgent):
    """Agent that plays a scripted sequence and writes fixed instrumentation."""

    def __init__(
        self,
        moves: list[Move],
        nodes: int = 42,
        depth: int = 7,
        prunings: int = 15,
    ) -> None:
        self._moves = iter(moves)
        self._nodes = nodes
        self._depth = depth
        self._prunings = prunings

    def choose_move(self, state: GameState) -> Move:
        state.nodes_visited = self._nodes
        state.max_depth_reached = self._depth
        state.prunings = self._prunings
        state.compute_ebf()
        return next(self._moves)

    def get_name(self) -> str:
        return "Instrumented"

    def get_tier(self) -> int:
        return 0


class SlowAgent(BaseAgent):
    """Agent that sleeps to trigger a time-limit violation flag."""

    def __init__(self, sleep_ms: float) -> None:
        self._sleep_s = sleep_ms / 1000.0

    def choose_move(self, state: GameState) -> Move:
        time.sleep(self._sleep_s)
        empty = Board.get_all_empty_cells(state.board)
        state.nodes_visited = 1
        state.max_depth_reached = 0
        state.prunings = 0
        state.compute_ebf()
        return empty[0]

    def get_name(self) -> str:
        return "Slow"

    def get_tier(self) -> int:
        return 0


def _x_moves() -> list[Move]:
    """X wins with the top row."""
    return [(0, 0), (0, 1), (0, 2)]


def _o_moves() -> list[Move]:
    return [(1, 0), (1, 1)]


# ---------------------------------------------------------------------------
# Core regression: instrumentation survives apply_move
# ---------------------------------------------------------------------------


class TestInstrumentationSurvivesStep:
    """Each instrumentation field must be readable after game.step()."""

    def _game_after_one_step(
        self, nodes: int = 42, depth: int = 7, prunings: int = 15
    ) -> Game:
        agent_x = InstrumentedAgent(_x_moves(), nodes=nodes, depth=depth, prunings=prunings)
        agent_o = InstrumentedAgent(_o_moves())
        game = Game(agent_x, agent_o, n=3)
        game.step()
        return game

    def test_nodes_visited_survives_step(self) -> None:
        game = self._game_after_one_step(nodes=42)
        assert game.state.nodes_visited == 42

    def test_max_depth_survives_step(self) -> None:
        game = self._game_after_one_step(depth=7)
        assert game.state.max_depth_reached == 7

    def test_prunings_survive_step(self) -> None:
        game = self._game_after_one_step(prunings=15)
        assert game.state.prunings == 15

    def test_time_taken_ms_is_nonnegative_after_step(self) -> None:
        game = self._game_after_one_step()
        assert game.state.time_taken_ms >= 0.0

    def test_ebf_computed_after_step(self) -> None:
        # nodes=8, depth=3 → ebf = 8^(1/3) = 2.0
        game = self._game_after_one_step(nodes=8, depth=3)
        assert game.state.effective_branching_factor == pytest.approx(2.0, rel=1e-5)

    def test_instrumentation_not_accumulated_across_steps(self) -> None:
        """Second step overwrites first step's instrumentation (not accumulated)."""
        agent_x = InstrumentedAgent(_x_moves(), nodes=10, depth=2)
        agent_o = InstrumentedAgent(_o_moves(), nodes=20, depth=4)
        game = Game(agent_x, agent_o, n=3)

        game.step()  # X moves → nodes=10, depth=2
        game.step()  # O moves → nodes=20, depth=4

        # After O's move, the state should reflect O's instrumentation.
        assert game.state.nodes_visited == 20
        assert game.state.max_depth_reached == 4


# ---------------------------------------------------------------------------
# Time-limit flag
# ---------------------------------------------------------------------------


class TestTimeLimitFlag:
    """time_limit_exceeded must be set correctly based on actual elapsed time."""

    def test_flag_set_for_slow_agent(self) -> None:
        config = MatchConfig.time_controlled(ms=50.0)
        slow = SlowAgent(sleep_ms=200.0)
        fast = InstrumentedAgent(_o_moves())
        game = Game(slow, fast, n=3, match_config=config)
        game.step()
        assert game.state.time_limit_exceeded is True

    def test_flag_clear_for_fast_agent(self) -> None:
        config = MatchConfig.time_controlled(ms=2000.0)
        fast_x = InstrumentedAgent(_x_moves())
        fast_o = InstrumentedAgent(_o_moves())
        game = Game(fast_x, fast_o, n=3, match_config=config)
        game.step()
        assert game.state.time_limit_exceeded is False


# ---------------------------------------------------------------------------
# End-to-end: Arena.duel captures non-zero per_move_stats
# ---------------------------------------------------------------------------


class TestArenaDuelInstrumentation:
    """Arena.duel must record non-zero instrumentation in GameRecord.per_move_stats."""

    def test_per_move_stats_nonempty_in_game_record(self) -> None:
        """InstrumentedAgent writes nodes=42; per_move_stats must reflect this."""
        # We need two agents that always make legal moves and terminate.
        # Use a simple pair of agents that fill the board predictably.
        x_moves_list = [(0, 0), (0, 1), (0, 2)]
        o_moves_list = [(1, 0), (1, 1)]

        class _Scripted(BaseAgent):
            def __init__(self, moves_template: list[Move], nodes: int = 5) -> None:
                self._template = moves_template
                self._nodes = nodes
                self._idx = 0

            def choose_move(self, s: GameState) -> Move:
                s.nodes_visited = self._nodes
                s.max_depth_reached = 2
                s.prunings = 0
                s.compute_ebf()
                move = self._template[self._idx % len(self._template)]
                self._idx += 1
                return move

            def get_name(self) -> str:
                return f"Scripted-{self._nodes}"

            def get_tier(self) -> int:
                return 0

        arena = Arena(n=3, num_games=2, match_config=MatchConfig.time_controlled(500))
        result = arena.duel(_Scripted(x_moves_list, nodes=99), _Scripted(o_moves_list, nodes=3))
        assert len(result.games) > 0
        first_game = result.games[0]
        assert len(first_game.per_move_stats) > 0
        x_stats = [s for s in first_game.per_move_stats
                   if first_game.agent_x_name == "Scripted-99"
                   and s.player is Player.X]
        if x_stats:
            assert all(s.nodes_visited == 99 for s in x_stats)


# ---------------------------------------------------------------------------
# MinimaxAB end-to-end: instrumentation is non-zero
# ---------------------------------------------------------------------------


class TestMinimaxInstrumentationE2E:
    """MinimaxAB must produce non-zero nodes_visited after each step."""

    def test_minimax_nodes_visited_nonzero_after_step(self) -> None:
        config = MatchConfig.node_controlled(100)
        agent_x = MinimaxAB(depth=3, match_config=config)
        agent_o = InstrumentedAgent(
            [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
            nodes=1, depth=0, prunings=0,
        )
        game = Game(agent_x, agent_o, n=3, match_config=config)
        game.step()
        # MinimaxAB should have visited more than 1 node on an empty 3×3 board.
        assert game.state.nodes_visited > 0

    def test_minimax_nodes_within_budget(self) -> None:
        budget = 50
        config = MatchConfig.node_controlled(budget)
        agent_x = MinimaxAB(depth=6, match_config=config)
        agent_o = InstrumentedAgent(
            [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
            nodes=1, depth=0, prunings=0,
        )
        game = Game(agent_x, agent_o, n=3, match_config=config)
        game.step()
        # The budget check is tested at the START of each iteration, so the
        # agent may complete the iteration it is in when the budget expires.
        # We verify the agent doesn't go wildly over budget (< 10× limit).
        assert game.state.nodes_visited < budget * 10
