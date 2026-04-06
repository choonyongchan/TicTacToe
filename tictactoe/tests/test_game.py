"""Tests for the Game session manager."""

from __future__ import annotations

import time

import pytest

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.benchmark.metrics import MatchConfig
from tictactoe.core.board import Board
from tictactoe.core.game import Game
from tictactoe.core.state import GameState
from tictactoe.core.types import MatchMode, Move, Player, Result


# ---------------------------------------------------------------------------
# Scripted agents for deterministic testing
# ---------------------------------------------------------------------------


class ScriptedAgent(BaseAgent):
    """Agent that plays a pre-determined sequence of moves."""

    def __init__(self, moves: list[Move], name: str = "Scripted") -> None:
        self._moves = iter(moves)
        self._name = name

    def choose_move(self, state: GameState) -> Move:
        state.nodes_visited = 1
        state.max_depth_reached = 0
        state.prunings = 0
        state.compute_ebf()
        return next(self._moves)

    def get_name(self) -> str:
        return self._name

    def get_tier(self) -> int:
        return 0


class SlowAgent(BaseAgent):
    """Agent that deliberately sleeps to trigger a time-limit violation."""

    def __init__(self, sleep_ms: float = 200.0) -> None:
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
        return "SlowAgent"

    def get_tier(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Full game with scripted moves
# ---------------------------------------------------------------------------


class TestFullGame:
    """Integration tests for a complete game run."""

    def test_x_wins_scripted(self) -> None:
        # X plays top row; O plays middle row (not completing 3-in-a-row).
        x_moves = [(0, 0), (0, 1), (0, 2)]
        o_moves = [(1, 0), (1, 1)]
        agent_x = ScriptedAgent(x_moves)
        agent_o = ScriptedAgent(o_moves)
        game = Game(agent_x, agent_o, n=3)
        result = game.run()
        assert result is Result.X_WINS

    def test_draw_scripted(self) -> None:
        # A sequence that fills the board without a winner.
        # X X O / O O X / X O X
        x_moves = [(0, 0), (0, 1), (1, 2), (2, 0), (2, 2)]
        o_moves = [(0, 2), (1, 0), (1, 1), (2, 1)]
        agent_x = ScriptedAgent(x_moves)
        agent_o = ScriptedAgent(o_moves)
        game = Game(agent_x, agent_o, n=3)
        result = game.run()
        assert result is Result.DRAW


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for Game.reset."""

    def test_reset_clears_state(self) -> None:
        x_moves = [(0, 0), (0, 1), (0, 2)]
        o_moves = [(1, 0), (1, 1)]
        game = Game(ScriptedAgent(x_moves), ScriptedAgent(o_moves), n=3)
        game.run()
        game.reset()
        state = game.get_state()
        assert state.move_number == 0
        assert state.result is Result.IN_PROGRESS
        assert state.current_player is Player.X


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------


class TestStep:
    """Tests for Game.step."""

    def test_step_alternates_player(self) -> None:
        agent_x = ScriptedAgent([(0, 0), (0, 1), (0, 2)])
        agent_o = ScriptedAgent([(1, 0), (1, 1)])
        game = Game(agent_x, agent_o, n=3)
        assert game.state.current_player is Player.X
        game.step()
        assert game.state.current_player is Player.O
        game.step()
        assert game.state.current_player is Player.X

    def test_step_updates_result(self) -> None:
        # After X fills the top row, result should be X_WINS.
        x_moves = [(0, 0), (0, 1), (0, 2)]
        o_moves = [(1, 0), (1, 1)]
        game = Game(ScriptedAgent(x_moves), ScriptedAgent(o_moves), n=3)
        for _ in range(5):
            game.step()
        assert game.state.result is Result.X_WINS


# ---------------------------------------------------------------------------
# time_limit_exceeded
# ---------------------------------------------------------------------------


class TestTimeLimitExceeded:
    """Tests for the time_limit_exceeded flag."""

    def test_flag_set_when_agent_is_slow(self) -> None:
        # SlowAgent sleeps 200 ms; time limit is 50 ms.
        config = MatchConfig.time_controlled(ms=50.0)
        slow = SlowAgent(sleep_ms=200.0)
        fast = ScriptedAgent([(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)])
        game = Game(slow, fast, n=3, match_config=config)
        game.step()  # SlowAgent makes first move.
        assert game.state.time_limit_exceeded is True

    def test_flag_not_set_for_fast_agent(self) -> None:
        config = MatchConfig.time_controlled(ms=2000.0)
        fast_x = ScriptedAgent([(0, 0), (0, 1), (0, 2)])
        fast_o = ScriptedAgent([(1, 0), (1, 1)])
        game = Game(fast_x, fast_o, n=3, match_config=config)
        game.step()
        assert game.state.time_limit_exceeded is False
