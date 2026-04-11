"""Boundary and edge-case tests spanning core modules.

Covers k=1, k>n, n=1, large boards, EBF edge values, multi-move chains,
and win detection in all four directions on larger boards.
"""
from __future__ import annotations

import math

import pytest

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh(n: int, k: int | None = None) -> GameState:
    """Return a blank GameState for an n×n game."""
    if k is None:
        k = n
    return GameState(board=Board.create(n), current_player=Player.X, n=n, k=k)


def _state_from_rows(rows: list[str], player: Player = Player.X, k: int | None = None) -> GameState:
    n = len(rows)
    if k is None:
        k = n
    board = Board.create(n)
    last_move = None
    move_history: list = []
    for r, row_str in enumerate(rows):
        for c, ch in enumerate(row_str):
            if ch == "X":
                board[r][c] = Cell.X
                last_move = (r, c)
                move_history.append((r, c))
            elif ch == "O":
                board[r][c] = Cell.O
                last_move = (r, c)
                move_history.append((r, c))
    return GameState(board=board, current_player=player, n=n, k=k,
                     last_move=last_move, move_history=move_history)


# ---------------------------------------------------------------------------
# k = 1 (first move always wins)
# ---------------------------------------------------------------------------


class TestKEqualsOne:
    """With k=1, placing any piece immediately wins."""

    def test_first_move_wins_on_3x3(self) -> None:
        board = Board.create(3)
        board[1][1] = Cell.X
        result = Board.is_terminal(board, 3, 1, (1, 1))
        assert result is Result.X_WINS

    def test_check_win_single_cell(self) -> None:
        board = Board.create(3)
        board[0][0] = Cell.X
        winner = Board.check_win(board, 3, 1, (0, 0))
        assert winner is Player.X

    def test_o_first_move_wins_with_k1(self) -> None:
        board = Board.create(3)
        board[2][2] = Cell.O
        result = Board.is_terminal(board, 3, 1, (2, 2))
        assert result is Result.O_WINS


# ---------------------------------------------------------------------------
# k > n (win is impossible; only draw or in-progress)
# ---------------------------------------------------------------------------


class TestKGreaterThanN:
    """When k exceeds the board size, no player can ever win."""

    def test_full_board_is_draw_when_k_exceeds_n(self) -> None:
        # Fill a 3×3 board with alternating pieces — no k=4 line possible.
        board = Board.create(3)
        cells = [Cell.X, Cell.O, Cell.X, Cell.O, Cell.X, Cell.O, Cell.X, Cell.O, Cell.X]
        for i in range(9):
            board[i // 3][i % 3] = cells[i]
        # k=4 on a 3×3 board → impossible to win → draw
        result = Board.is_terminal(board, 3, 4, (2, 2))
        assert result is Result.DRAW

    def test_partial_board_is_in_progress(self) -> None:
        # Only a few pieces placed — no win, not full.
        board = Board.create(3)
        board[0][0] = Cell.X
        board[0][1] = Cell.X
        board[0][2] = Cell.X
        result = Board.is_terminal(board, 3, 4, (0, 2))
        assert result is Result.IN_PROGRESS


# ---------------------------------------------------------------------------
# n = 1 (trivially small board)
# ---------------------------------------------------------------------------


class TestOneCellBoard:
    """A 1×1 board with k=1 ends on the very first move."""

    def test_create_1x1_board(self) -> None:
        board = Board.create(1)
        assert len(board) == 1
        assert len(board[0]) == 1
        assert board[0][0] is Cell.EMPTY

    def test_apply_move_on_1x1(self) -> None:
        state = _fresh(1, k=1)
        next_state = state.apply_move((0, 0))
        assert next_state.board[0][0] is Cell.X

    def test_first_move_on_1x1_is_terminal(self) -> None:
        board = Board.create(1)
        board[0][0] = Cell.X
        result = Board.is_terminal(board, 1, 1, (0, 0))
        assert result is Result.X_WINS


# ---------------------------------------------------------------------------
# Large board operations
# ---------------------------------------------------------------------------


class TestLargeBoardOperations:
    """Sanity checks that core operations work on boards up to n=50."""

    def test_n50_board_create(self) -> None:
        board = Board.create(50)
        assert len(board) == 50
        assert len(board[0]) == 50
        assert board[25][25] is Cell.EMPTY

    def test_n50_candidate_moves_nonempty_after_first_move(self) -> None:
        state = _fresh(50, k=5)
        # Place one piece at the centre.
        state = state.apply_move((25, 25))
        candidates = Board.get_candidate_moves(state, radius=2)
        assert len(candidates) > 0

    def test_n50_is_terminal_returns_in_progress(self) -> None:
        board = Board.create(50)
        board[25][25] = Cell.X
        result = Board.is_terminal(board, 50, 5, (25, 25))
        assert result is Result.IN_PROGRESS

    def test_n20_candidate_moves_all_empty(self) -> None:
        state = _fresh(20, k=5)
        # One move placed.
        state = state.apply_move((10, 10))
        candidates = Board.get_candidate_moves(state, radius=2)
        for r, c in candidates:
            assert state.board[r][c] is Cell.EMPTY


# ---------------------------------------------------------------------------
# EBF boundary values
# ---------------------------------------------------------------------------


class TestEBFBoundary:
    """Edge-case inputs to GameState.compute_ebf()."""

    def test_zero_depth_returns_zero(self) -> None:
        state = _fresh(3)
        state.nodes_visited = 1000
        state.max_depth_reached = 0
        ebf = state.compute_ebf()
        assert ebf == 0.0
        assert state.effective_branching_factor == 0.0

    def test_depth_one_nodes_one(self) -> None:
        state = _fresh(3)
        state.nodes_visited = 1
        state.max_depth_reached = 1
        ebf = state.compute_ebf()
        assert ebf == pytest.approx(1.0)

    def test_depth_two_nodes_four(self) -> None:
        # 4^(1/2) = 2.0
        state = _fresh(3)
        state.nodes_visited = 4
        state.max_depth_reached = 2
        ebf = state.compute_ebf()
        assert ebf == pytest.approx(2.0)

    def test_large_nodes_no_overflow(self) -> None:
        state = _fresh(3)
        state.nodes_visited = 10 ** 9
        state.max_depth_reached = 10
        ebf = state.compute_ebf()
        assert math.isfinite(ebf)
        assert ebf > 0.0


# ---------------------------------------------------------------------------
# apply_move chain
# ---------------------------------------------------------------------------


class TestApplyMoveChain:
    """Chaining apply_move() through a full 3×3 game."""

    _MOVES = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2),
              (2, 0), (1, 2), (2, 1), (2, 2)]

    def test_9_moves_increments_move_number(self) -> None:
        state = _fresh(3)
        for move in self._MOVES:
            state = state.apply_move(move)
        assert state.move_number == 9

    def test_9_moves_history_length(self) -> None:
        state = _fresh(3)
        for move in self._MOVES:
            state = state.apply_move(move)
        assert len(state.move_history) == 9

    def test_board_all_non_empty_after_9_moves(self) -> None:
        state = _fresh(3)
        for move in self._MOVES:
            state = state.apply_move(move)
        for row in state.board:
            for cell in row:
                assert cell is not Cell.EMPTY

    def test_apply_move_does_not_mutate_original(self) -> None:
        state = _fresh(3)
        original_cell = state.board[0][0]
        state.apply_move((0, 0))
        # Original state must be unchanged.
        assert state.board[0][0] is original_cell


# ---------------------------------------------------------------------------
# Win detection in all four directions on a 5×5 board (k=3)
# ---------------------------------------------------------------------------


class TestCheckWinDirections:
    """Verify that check_win detects wins in all four directions on a 5×5 board."""

    def test_horizontal_win(self) -> None:
        board = Board.create(5)
        for c in range(3):
            board[2][c] = Cell.X
        winner = Board.check_win(board, 5, 3, (2, 2))
        assert winner is Player.X

    def test_vertical_win(self) -> None:
        board = Board.create(5)
        for r in range(3):
            board[r][2] = Cell.X
        winner = Board.check_win(board, 5, 3, (2, 2))
        assert winner is Player.X

    def test_main_diagonal_win(self) -> None:
        board = Board.create(5)
        for i in range(3):
            board[i][i] = Cell.O
        winner = Board.check_win(board, 5, 3, (2, 2))
        assert winner is Player.O

    def test_anti_diagonal_win(self) -> None:
        board = Board.create(5)
        # Anti-diagonal: (0,4), (1,3), (2,2)
        for i in range(3):
            board[i][4 - i] = Cell.O
        winner = Board.check_win(board, 5, 3, (2, 2))
        assert winner is Player.O

    def test_k_minus_one_does_not_win(self) -> None:
        """k-1 consecutive pieces must NOT trigger a win."""
        board = Board.create(5)
        for c in range(2):
            board[0][c] = Cell.X
        winner = Board.check_win(board, 5, 3, (0, 1))
        assert winner is None
