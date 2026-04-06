"""Tests for the BruteForceOracle and correctness verification utilities."""

from __future__ import annotations

import pytest

from tictactoe.agents.random_agent import RandomAgent
from tictactoe.benchmark.correctness import (
    KNOWN_POSITIONS,
    BruteForceOracle,
    verify_agent_on_known_positions,
    verify_oracle_never_loses,
    _board_from_ints,
)
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def state_from_position(position: dict) -> GameState:
    """Build a GameState from a KNOWN_POSITIONS entry."""
    board = _board_from_ints(position["board"])
    player = Player.X if position["current_player"] == "X" else Player.O
    state = GameState(board=board, current_player=player, n=3, k=3)
    state.result = Board.is_terminal(board, 3, 3, None)
    return state


def x_count(rows: list[list[int]]) -> int:
    return sum(v == 1 for row in rows for v in row)


def o_count(rows: list[list[int]]) -> int:
    return sum(v == 2 for row in rows for v in row)


# ---------------------------------------------------------------------------
# BruteForceOracle basic behaviour
# ---------------------------------------------------------------------------


class TestBruteForceOracle:
    """Tests for BruteForceOracle."""

    def test_never_loses_to_itself(self) -> None:
        result = verify_oracle_never_loses(BruteForceOracle(), games=20)
        assert result["agent_losses"] == 0

    def test_oracle_vs_oracle_all_draws(self) -> None:
        result = verify_oracle_never_loses(BruteForceOracle(), games=20)
        assert result["draws"] == 20

    def test_takes_winning_move(self) -> None:
        # X X . / . O . / . . O — X can win at (0,2).
        position = {
            "board": [[1, 1, 0], [0, 2, 0], [0, 0, 2]],
            "current_player": "X",
            "expected_move": (0, 2),
            "expected_result": "X_WINS",
            "must_not_lose": False,
            "description": "X wins at (0,2)",
        }
        oracle = BruteForceOracle()
        state = state_from_position(position)
        move = oracle.choose_move(state)
        assert move == (0, 2)

    def test_blocks_opponent_win(self) -> None:
        # O O . / X X . / . . . — X must block at (0,2).
        position = {
            "board": [[2, 2, 0], [1, 1, 0], [0, 0, 0]],
            "current_player": "X",
            "expected_move": (0, 2),
            "expected_result": "DRAW",
            "must_not_lose": False,
            "description": "X blocks O at (0,2)",
        }
        oracle = BruteForceOracle()
        state = state_from_position(position)
        move = oracle.choose_move(state)
        assert move == (0, 2)

    def test_raises_on_wrong_board_size(self) -> None:
        oracle = BruteForceOracle()
        state = GameState(
            board=Board.create(5), current_player=Player.X, n=5, k=5
        )
        with pytest.raises(ValueError):
            oracle.choose_move(state)


# ---------------------------------------------------------------------------
# verify_agent_on_known_positions
# ---------------------------------------------------------------------------


class TestVerifyAgentOnKnownPositions:
    """Tests for verify_agent_on_known_positions."""

    def test_oracle_passes_all_positions(self) -> None:
        report = verify_agent_on_known_positions(BruteForceOracle())
        assert report["passed"] is True
        assert report["correct"] == report["total"]

    def test_random_agent_fails_many_positions(self) -> None:
        # RandomAgent will fail forced-win positions much of the time.
        # Run many seeds and assert that at least one fails.
        any_failed = False
        for seed in range(10):
            report = verify_agent_on_known_positions(RandomAgent(seed=seed))
            if not report["passed"]:
                any_failed = True
                break
        assert any_failed, "RandomAgent should fail at least some known positions."


# ---------------------------------------------------------------------------
# verify_oracle_never_loses
# ---------------------------------------------------------------------------


class TestVerifyOracleNeverLoses:
    """Tests for verify_oracle_never_loses."""

    def test_oracle_vs_itself_zero_losses(self) -> None:
        result = verify_oracle_never_loses(BruteForceOracle(), games=20)
        assert result["passed"] is True
        assert result["agent_losses"] == 0

    def test_report_structure(self) -> None:
        result = verify_oracle_never_loses(BruteForceOracle(), games=10)
        required_keys = {"passed", "agent_losses", "draws", "agent_wins", "games"}
        assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# KNOWN_POSITIONS validity
# ---------------------------------------------------------------------------


class TestKnownPositionsValidity:
    """Tests that KNOWN_POSITIONS entries are well-formed board states."""

    @pytest.mark.parametrize("position", KNOWN_POSITIONS)
    def test_piece_counts_are_valid(self, position: dict) -> None:
        x_pieces = x_count(position["board"])
        o_pieces = o_count(position["board"])
        # X moves first: x_count == o_count or x_count == o_count + 1.
        assert x_pieces == o_pieces or x_pieces == o_pieces + 1, (
            f"Invalid piece counts in '{position['description']}': "
            f"X={x_pieces} O={o_pieces}"
        )

    @pytest.mark.parametrize("position", KNOWN_POSITIONS)
    def test_board_is_3x3(self, position: dict) -> None:
        board = position["board"]
        assert len(board) == 3
        assert all(len(row) == 3 for row in board)

    @pytest.mark.parametrize("position", KNOWN_POSITIONS)
    def test_player_field_valid(self, position: dict) -> None:
        assert position["current_player"] in {"X", "O"}

    @pytest.mark.parametrize("position", KNOWN_POSITIONS)
    def test_expected_result_valid(self, position: dict) -> None:
        assert position["expected_result"] in {"X_WINS", "O_WINS", "DRAW"}
