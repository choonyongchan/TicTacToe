"""Tests for the GameState dataclass and its methods."""

from __future__ import annotations

import json
import math

import pytest

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fresh_state(n: int = 3) -> GameState:
    """Return a blank GameState for an n×n game."""
    return GameState(
        board=Board.create(n),
        current_player=Player.X,
        n=n,
        k=n,
    )


# ---------------------------------------------------------------------------
# apply_move
# ---------------------------------------------------------------------------


class TestApplyMove:
    """Tests for GameState.apply_move."""

    def test_does_not_mutate_original(self) -> None:
        state = fresh_state()
        original_board_copy = [row[:] for row in state.board]
        state.apply_move((0, 0))
        assert state.board == original_board_copy

    def test_updates_move_history(self) -> None:
        state = fresh_state()
        next_state = state.apply_move((1, 1))
        assert (1, 1) in next_state.move_history

    def test_updates_last_move(self) -> None:
        state = fresh_state()
        next_state = state.apply_move((2, 2))
        assert next_state.last_move == (2, 2)

    def test_increments_move_number(self) -> None:
        state = fresh_state()
        next_state = state.apply_move((0, 0))
        assert next_state.move_number == 1

    def test_alternates_current_player(self) -> None:
        state = fresh_state()
        assert state.current_player is Player.X
        next_state = state.apply_move((0, 0))
        assert next_state.current_player is Player.O
        next_next = next_state.apply_move((1, 1))
        assert next_next.current_player is Player.X

    def test_places_correct_cell(self) -> None:
        state = fresh_state()
        next_state = state.apply_move((0, 0))
        assert next_state.board[0][0] is Cell.X

    def test_history_grows_each_move(self) -> None:
        state = fresh_state()
        state = state.apply_move((0, 0))
        state = state.apply_move((1, 1))
        assert len(state.move_history) == 2


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


class TestCopy:
    """Tests for GameState.copy."""

    def test_copy_is_equal(self) -> None:
        state = fresh_state()
        state.move_history = [(0, 0)]
        copied = state.copy()
        assert copied.move_history == state.move_history

    def test_copy_is_deep(self) -> None:
        state = fresh_state()
        copied = state.copy()
        copied.board[0][0] = Cell.X
        assert state.board[0][0] is Cell.EMPTY

    def test_move_history_not_shared(self) -> None:
        state = fresh_state()
        state.move_history = [(0, 0)]
        copied = state.copy()
        copied.move_history.append((1, 1))
        assert len(state.move_history) == 1


# ---------------------------------------------------------------------------
# compute_ebf
# ---------------------------------------------------------------------------


class TestComputeEBF:
    """Tests for GameState.compute_ebf."""

    def test_zero_depth_returns_zero(self) -> None:
        state = fresh_state()
        state.nodes_visited = 100
        state.max_depth_reached = 0
        ebf = state.compute_ebf()
        assert ebf == 0.0
        assert state.effective_branching_factor == 0.0

    def test_known_pair(self) -> None:
        state = fresh_state()
        state.nodes_visited = 27
        state.max_depth_reached = 3
        ebf = state.compute_ebf()
        expected = 27 ** (1 / 3)
        assert math.isclose(ebf, expected, rel_tol=1e-6)

    def test_stores_result(self) -> None:
        state = fresh_state()
        state.nodes_visited = 16
        state.max_depth_reached = 2
        state.compute_ebf()
        assert math.isclose(state.effective_branching_factor, 4.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    """Tests for GameState.to_dict."""

    def test_json_serialisable(self) -> None:
        state = fresh_state()
        as_dict = state.to_dict()
        json_str = json.dumps(as_dict)
        assert isinstance(json_str, str)

    def test_contains_all_fields(self) -> None:
        state = fresh_state()
        as_dict = state.to_dict()
        required_keys = [
            "board", "current_player", "n", "k",
            "move_history", "last_move", "result", "move_number",
            "nodes_visited", "max_depth_reached", "time_taken_ms",
            "prunings", "effective_branching_factor", "time_limit_exceeded",
        ]
        for key in required_keys:
            assert key in as_dict, f"Missing key: {key}"

    def test_board_encoded_as_strings(self) -> None:
        state = fresh_state()
        as_dict = state.to_dict()
        # Board must be a list of lists of strings.
        assert isinstance(as_dict["board"][0][0], str)

    def test_result_is_string(self) -> None:
        state = fresh_state()
        as_dict = state.to_dict()
        assert isinstance(as_dict["result"], str)
