"""Tests for the evaluation/move_ordering module."""

from __future__ import annotations

import pytest

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, INF, Player
from tictactoe.evaluation.move_ordering import (
    HistoryTable,
    KillerMoveTable,
    order_moves,
    score_move_statically,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(rows: list[list[int]], player: str = "X") -> GameState:
    mapping = {0: Cell.EMPTY, 1: Cell.X, 2: Cell.O}
    board = [[mapping[v] for v in row] for row in rows]
    n = len(rows)
    current = Player.X if player == "X" else Player.O
    return GameState(board=board, current_player=current, n=n, k=n)


# ---------------------------------------------------------------------------
# score_move_statically
# ---------------------------------------------------------------------------


class TestScoreMoveStatically:
    """Tests for score_move_statically."""

    def test_winning_move_scores_inf(self) -> None:
        # X X . — (0,2) wins immediately for X.
        state = make_state([[1, 1, 0], [0, 2, 0], [0, 0, 2]])
        score = score_move_statically(state, (0, 2))
        assert score == INF

    def test_blocking_move_scores_1e6(self) -> None:
        # O O . — X must block at (0,2).
        state = make_state([[2, 2, 0], [1, 0, 0], [0, 0, 0]])
        score = score_move_statically(state, (0, 2))
        assert score == pytest.approx(1e6)

    def test_non_critical_move_below_winning_threshold(self) -> None:
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        score = score_move_statically(state, (1, 1))
        assert score < 1e6


# ---------------------------------------------------------------------------
# order_moves
# ---------------------------------------------------------------------------


class TestOrderMoves:
    """Tests for order_moves."""

    def test_winning_move_comes_first(self) -> None:
        # X X . — (0,2) should be first in ordering.
        state = make_state([[1, 1, 0], [0, 2, 0], [0, 0, 2]])
        moves = [(0, 2), (1, 0), (2, 0), (2, 1)]
        ordered = order_moves(state, moves)
        assert ordered[0] == (0, 2)

    def test_blocking_move_before_random_moves(self) -> None:
        # O O . — blocking move should rank high.
        state = make_state([[2, 2, 0], [1, 0, 0], [0, 0, 0]])
        moves = Board.get_all_empty_cells(state.board)
        ordered = order_moves(state, moves)
        assert ordered[0] == (0, 2)


# ---------------------------------------------------------------------------
# KillerMoveTable
# ---------------------------------------------------------------------------


class TestKillerMoveTable:
    """Tests for KillerMoveTable."""

    def test_stores_and_retrieves(self) -> None:
        table = KillerMoveTable()
        table.store(3, (0, 0))
        assert (0, 0) in table.get(3)

    def test_keeps_top_two(self) -> None:
        table = KillerMoveTable()
        table.store(2, (0, 0))
        table.store(2, (1, 1))
        table.store(2, (2, 2))
        assert len(table.get(2)) == 2
        # Oldest entry should be evicted.
        assert (0, 0) not in table.get(2)

    def test_no_duplicates(self) -> None:
        table = KillerMoveTable()
        table.store(1, (0, 0))
        table.store(1, (0, 0))
        assert table.get(1).count((0, 0)) == 1

    def test_clear_resets(self) -> None:
        table = KillerMoveTable()
        table.store(0, (0, 0))
        table.clear()
        assert table.get(0) == []


# ---------------------------------------------------------------------------
# HistoryTable
# ---------------------------------------------------------------------------


class TestHistoryTable:
    """Tests for HistoryTable."""

    def test_update_increments_with_depth(self) -> None:
        table = HistoryTable()
        table.update((0, 0), depth=3)
        assert table.score((0, 0)) == 2.0 ** 3

    def test_multiple_updates_accumulate(self) -> None:
        table = HistoryTable()
        table.update((1, 1), depth=1)
        table.update((1, 1), depth=2)
        assert table.score((1, 1)) == pytest.approx(2.0 + 4.0)

    def test_unseen_move_scores_zero(self) -> None:
        table = HistoryTable()
        assert table.score((9, 9)) == 0.0

    def test_clear_resets_scores(self) -> None:
        table = HistoryTable()
        table.update((0, 0), depth=5)
        table.clear()
        assert table.score((0, 0)) == 0.0

    def test_get_top_n(self) -> None:
        table = HistoryTable()
        table.update((0, 0), depth=1)
        table.update((1, 1), depth=3)
        table.update((2, 2), depth=2)
        top = table.get_top_n(2)
        assert top[0] == (1, 1)
        assert len(top) == 2
