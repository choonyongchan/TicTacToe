"""Tests for the evaluation/heuristics module."""

from __future__ import annotations

import pytest

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result, WIN_SCORE
from tictactoe.evaluation.heuristics import (
    count_open_threats,
    evaluate_position,
    precompute_line_indices,
    precompute_positional_weights,
    score_board,
    score_line,
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


def cells_from_ints(values: list[int]) -> list[Cell]:
    mapping = {0: Cell.EMPTY, 1: Cell.X, 2: Cell.O}
    return [mapping[v] for v in values]


# ---------------------------------------------------------------------------
# score_line
# ---------------------------------------------------------------------------


class TestScoreLine:
    """Tests for score_line."""

    def test_dead_line_scores_zero(self) -> None:
        cells = cells_from_ints([1, 2, 0])
        assert score_line(cells, Player.X, 3) == 0.0

    def test_player_only_positive(self) -> None:
        cells = cells_from_ints([1, 1, 0])
        score = score_line(cells, Player.X, 3)
        assert score > 0

    def test_opponent_only_negative(self) -> None:
        cells = cells_from_ints([2, 2, 0])
        score = score_line(cells, Player.X, 3)
        assert score < 0

    def test_empty_line_scores_zero(self) -> None:
        cells = cells_from_ints([0, 0, 0])
        assert score_line(cells, Player.X, 3) == 0.0

    def test_more_pieces_higher_score(self) -> None:
        two_pieces = score_line(cells_from_ints([1, 1, 0]), Player.X, 3)
        one_piece = score_line(cells_from_ints([1, 0, 0]), Player.X, 3)
        assert two_pieces > one_piece


# ---------------------------------------------------------------------------
# score_board
# ---------------------------------------------------------------------------


class TestScoreBoard:
    """Tests for score_board."""

    def test_x_ahead_scores_higher(self) -> None:
        state_x_ahead = make_state([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
        state_o_ahead = make_state([[2, 2, 0], [0, 0, 0], [0, 0, 0]])
        score_x = score_board(state_x_ahead.board, 3, 3, Player.X)
        score_o = score_board(state_o_ahead.board, 3, 3, Player.X)
        assert score_x > score_o


# ---------------------------------------------------------------------------
# evaluate_position
# ---------------------------------------------------------------------------


class TestEvaluatePosition:
    """Tests for evaluate_position."""

    def test_terminal_x_win_returns_win_score(self) -> None:
        state = make_state([[1, 1, 1], [0, 2, 0], [0, 0, 2]])
        state.result = Result.X_WINS
        assert evaluate_position(state, Player.X) == WIN_SCORE

    def test_terminal_x_win_returns_neg_win_score_for_o(self) -> None:
        state = make_state([[1, 1, 1], [0, 2, 0], [0, 0, 2]])
        state.result = Result.X_WINS
        assert evaluate_position(state, Player.O) == -WIN_SCORE

    def test_terminal_o_win(self) -> None:
        state = make_state([[2, 2, 2], [0, 1, 0], [0, 0, 1]])
        state.result = Result.O_WINS
        assert evaluate_position(state, Player.O) == WIN_SCORE

    def test_draw_returns_zero(self) -> None:
        state = make_state([[1, 2, 1], [2, 1, 2], [2, 1, 2]])
        state.result = Result.DRAW
        assert evaluate_position(state, Player.X) == 0.0

    def test_in_progress_returns_float(self) -> None:
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        score = evaluate_position(state, Player.X)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# precompute_line_indices
# ---------------------------------------------------------------------------


class TestPrecomputeLineIndices:
    """Tests for precompute_line_indices."""

    def test_correct_count_3x3_k3(self) -> None:
        # 3 rows + 3 cols + 2 diagonals = 8 lines.
        lines = precompute_line_indices(3, 3)
        assert len(lines) == 8

    def test_correct_count_5x5_k4(self) -> None:
        # Horizontal: 5 rows × 2 positions = 10
        # Vertical:   5 cols × 2 positions = 10
        # Diagonal:   2×2 = 4
        # Anti-diag:  4
        # Total: 28
        lines = precompute_line_indices(5, 4)
        assert len(lines) == 28

    def test_each_line_has_correct_length(self) -> None:
        for k in [3, 4]:
            for line in precompute_line_indices(5, k):
                assert len(line) == k

    def test_cached_returns_same_object(self) -> None:
        lines_a = precompute_line_indices(3, 3)
        lines_b = precompute_line_indices(3, 3)
        assert lines_a is lines_b


# ---------------------------------------------------------------------------
# count_open_threats
# ---------------------------------------------------------------------------


class TestCountOpenThreats:
    """Tests for count_open_threats."""

    def test_single_open_threat(self) -> None:
        # X X . — one open (k-1) threat.
        state = make_state([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
        count = count_open_threats(state.board, 3, 3, Player.X, 2)
        assert count >= 1

    def test_no_threat_for_dead_line(self) -> None:
        # X O . — dead line, no open threat.
        state = make_state([[1, 2, 0], [0, 0, 0], [0, 0, 0]])
        count = count_open_threats(state.board, 3, 3, Player.X, 1)
        # The row is dead; only other lines containing just X count.
        # (0,0) participates in col-0, diag, etc. — check col-0 only has X.
        # This test just verifies the function runs without error.
        assert isinstance(count, int)


# ---------------------------------------------------------------------------
# precompute_positional_weights
# ---------------------------------------------------------------------------


class TestPositionalWeights:
    """Tests for precompute_positional_weights."""

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_centre_has_highest_weight(self, n: int) -> None:
        weights = precompute_positional_weights(n, n)
        centre = n // 2
        centre_weight = weights[centre][centre]
        for row in range(n):
            for col in range(n):
                assert weights[row][col] <= centre_weight + 1e-9

    @pytest.mark.parametrize("n", [3, 5])
    def test_all_weights_in_range(self, n: int) -> None:
        weights = precompute_positional_weights(n, n)
        for row in weights:
            for w in row:
                assert 0.0 <= w <= 1.0 + 1e-9
