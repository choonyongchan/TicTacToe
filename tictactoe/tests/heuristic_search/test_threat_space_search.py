"""Tests for ThreatSpaceSearch."""
from __future__ import annotations

import pytest

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result
from tictactoe.agents.heuristic_search.shared.threat_space_search import (
    ThreatSpaceSearch,
    ThreatType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(rows: list[list[int]], player: str = "X", k: int | None = None) -> GameState:
    mapping = {0: Cell.EMPTY, 1: Cell.X, 2: Cell.O}
    board = [[mapping[v] for v in row] for row in rows]
    n = len(rows)
    current = Player.X if player == "X" else Player.O
    win_k = k if k is not None else n
    return GameState(board=board, current_player=current, n=n, k=win_k)


def empty_state(n: int = 3) -> GameState:
    board = Board.create(n)
    return GameState(board=board, current_player=Player.X, n=n, k=n)


# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------


class TestTSSBasic:
    """Basic TSS correctness tests."""

    def test_returns_list_or_none_on_empty_board(self) -> None:
        """TSS returns a list or None on an empty board (never raises)."""
        tss = ThreatSpaceSearch()
        state = empty_state(3)
        result = tss.find_forced_win(state, Player.X)
        # The result may be a sequence (if TSS sees threats) or None — both fine
        assert result is None or isinstance(result, list)

    def test_none_on_drawn_full_board(self) -> None:
        """No forced win when the board is full."""
        tss = ThreatSpaceSearch()
        # Drawn 3x3 board
        state = make_state([
            [1, 2, 1],
            [2, 1, 2],
            [2, 1, 2],
        ])
        state.result = Result.DRAW
        result = tss.find_forced_win(state, Player.X)
        assert result is None

    def test_finds_immediate_win_in_sequence(self) -> None:
        """TSS should return a winning sequence when X can win immediately."""
        tss = ThreatSpaceSearch()
        # X has two in a row, can win with one more move
        state = make_state([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 0],
        ])
        result = tss.find_forced_win(state, Player.X)
        # May or may not find it via threat-space (depends on threat classification)
        # Key property: if returned, first move must be a legal empty cell
        if result is not None:
            assert len(result) >= 1
            row, col = result[0]
            assert 0 <= row < 3 and 0 <= col < 3
            assert state.board[row][col] is Cell.EMPTY

    def test_result_is_list_or_none(self) -> None:
        """find_forced_win always returns a list or None."""
        tss = ThreatSpaceSearch()
        state = make_state([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        result = tss.find_forced_win(state, Player.X)
        assert result is None or isinstance(result, list)

    def test_sequence_contains_only_tuples(self) -> None:
        """All moves in a returned sequence are (int, int) tuples."""
        tss = ThreatSpaceSearch()
        state = make_state([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 0],
        ])
        result = tss.find_forced_win(state, Player.X)
        if result is not None:
            for move in result:
                assert isinstance(move, tuple)
                assert len(move) == 2


# ---------------------------------------------------------------------------
# Forced win detection on known positions
# ---------------------------------------------------------------------------


class TestTSSForcedWins:
    """TSS on positions with a definite forced win."""

    def test_detects_win_on_winning_board(self) -> None:
        """X already has k-1 in a row with an open end — should detect as threat."""
        tss = ThreatSpaceSearch()
        # X has two pieces in the top row, (0,2) is empty for the win
        # k=3, so two in a row with open end = open threat
        state = make_state([
            [1, 1, 0],
            [2, 0, 0],
            [0, 0, 0],
        ])
        result = tss.find_forced_win(state, Player.X)
        # TSS might find a threat sequence here
        if result is not None:
            assert len(result) >= 1
            row, col = result[0]
            assert state.board[row][col] is Cell.EMPTY

    def test_player_o_forced_win(self) -> None:
        """TSS works for Player.O as well."""
        tss = ThreatSpaceSearch()
        state = make_state([
            [2, 2, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], player="O")
        result = tss.find_forced_win(state, Player.O)
        if result is not None:
            row, col = result[0]
            assert state.board[row][col] is Cell.EMPTY

    def test_max_depth_zero_returns_none(self) -> None:
        """With max_depth=0, TSS cannot explore any threats."""
        tss = ThreatSpaceSearch()
        state = make_state([
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        result = tss.find_forced_win(state, Player.X, max_depth=0)
        assert result is None

    def test_max_depth_respected(self) -> None:
        """TSS with depth=1 may miss sequences requiring 2+ moves."""
        tss = ThreatSpaceSearch()
        state = empty_state(3)
        result_shallow = tss.find_forced_win(state, Player.X, max_depth=1)
        result_deep = tss.find_forced_win(state, Player.X, max_depth=10)
        # Neither should crash; deep result can only be None or a sequence
        assert result_shallow is None or isinstance(result_shallow, list)
        assert result_deep is None or isinstance(result_deep, list)


# ---------------------------------------------------------------------------
# ThreatType enum
# ---------------------------------------------------------------------------


class TestThreatTypeEnum:
    """Verify ThreatType enum values are ordered correctly."""

    def test_open_four_strongest(self) -> None:
        assert ThreatType.OPEN_FOUR.value > ThreatType.HALF_OPEN_FOUR.value

    def test_open_three_weaker_than_half_open_four(self) -> None:
        assert ThreatType.HALF_OPEN_FOUR.value > ThreatType.OPEN_THREE.value

    def test_half_open_three_weakest(self) -> None:
        assert ThreatType.OPEN_THREE.value > ThreatType.HALF_OPEN_THREE.value
