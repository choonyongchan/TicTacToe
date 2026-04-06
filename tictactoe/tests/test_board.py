"""Tests for the Board stateless utility class."""

from __future__ import annotations

import pytest

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(rows: list[list[int]], player: str = "X") -> GameState:
    """Build a GameState from a list-of-ints board (0=E, 1=X, 2=O)."""
    mapping = {0: Cell.EMPTY, 1: Cell.X, 2: Cell.O}
    board = [[mapping[v] for v in row] for row in rows]
    n = len(rows)
    current = Player.X if player == "X" else Player.O
    state = GameState(board=board, current_player=current, n=n, k=n)
    return state


# ---------------------------------------------------------------------------
# Board creation
# ---------------------------------------------------------------------------


class TestCreate:
    """Tests for Board.create."""

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_creates_empty_board(self, n: int) -> None:
        board = Board.create(n)
        assert len(board) == n
        assert all(len(row) == n for row in board)
        assert all(cell is Cell.EMPTY for row in board for cell in row)


# ---------------------------------------------------------------------------
# Board queries
# ---------------------------------------------------------------------------


class TestBoardQueries:
    """Tests for is_empty, is_full, get_cell, count_pieces, get_all_empty_cells."""

    def test_is_empty_true(self) -> None:
        board = Board.create(3)
        assert Board.is_empty(board, 1, 1) is True

    def test_is_empty_false(self) -> None:
        board = Board.create(3)
        board[0][0] = Cell.X
        assert Board.is_empty(board, 0, 0) is False

    def test_is_full_empty_board(self) -> None:
        assert Board.is_full(Board.create(3)) is False

    def test_is_full_full_board(self) -> None:
        board = [[Cell.X] * 3 for _ in range(3)]
        assert Board.is_full(board) is True

    def test_get_cell(self) -> None:
        board = Board.create(3)
        board[2][1] = Cell.O
        assert Board.get_cell(board, 2, 1) is Cell.O

    def test_count_pieces(self) -> None:
        board = Board.create(3)
        board[0][0] = Cell.X
        board[1][1] = Cell.X
        board[2][0] = Cell.O
        assert Board.count_pieces(board, Player.X) == 2
        assert Board.count_pieces(board, Player.O) == 1

    def test_get_all_empty_cells(self) -> None:
        board = Board.create(3)
        board[0][0] = Cell.X
        empty = Board.get_all_empty_cells(board)
        assert len(empty) == 8
        assert (0, 0) not in empty


# ---------------------------------------------------------------------------
# Win detection
# ---------------------------------------------------------------------------


class TestCheckWin:
    """Tests for Board.check_win and Board.check_win_full."""

    def test_horizontal_win(self) -> None:
        board = Board.create(3)
        for col in range(3):
            board[0][col] = Cell.X
        assert Board.check_win(board, 3, 3, (0, 2)) is Player.X

    def test_vertical_win(self) -> None:
        board = Board.create(3)
        for row in range(3):
            board[row][1] = Cell.O
        assert Board.check_win(board, 3, 3, (2, 1)) is Player.O

    def test_diagonal_win(self) -> None:
        board = Board.create(3)
        for i in range(3):
            board[i][i] = Cell.X
        assert Board.check_win(board, 3, 3, (2, 2)) is Player.X

    def test_anti_diagonal_win(self) -> None:
        board = Board.create(3)
        board[0][2] = Cell.O
        board[1][1] = Cell.O
        board[2][0] = Cell.O
        assert Board.check_win(board, 3, 3, (1, 1)) is Player.O

    def test_no_win_empty_board(self) -> None:
        board = Board.create(3)
        board[0][0] = Cell.X
        assert Board.check_win(board, 3, 3, (0, 0)) is None

    def test_k_minus_1_not_win(self) -> None:
        board = Board.create(3)
        board[0][0] = Cell.X
        board[0][1] = Cell.X
        # Only k-1=2 pieces in a row for k=3 — not a win.
        assert Board.check_win(board, 3, 3, (0, 1)) is None

    def test_check_win_full_matches_incremental(self) -> None:
        board = Board.create(3)
        for col in range(3):
            board[2][col] = Cell.O
        incremental = Board.check_win(board, 3, 3, (2, 1))
        full = Board.check_win_full(board, 3, 3)
        assert incremental == full == Player.O


# ---------------------------------------------------------------------------
# is_terminal
# ---------------------------------------------------------------------------


class TestIsTerminal:
    """Tests for Board.is_terminal."""

    def test_in_progress(self) -> None:
        board = Board.create(3)
        board[0][0] = Cell.X
        result = Board.is_terminal(board, 3, 3, (0, 0))
        assert result is Result.IN_PROGRESS

    def test_x_wins(self) -> None:
        board = Board.create(3)
        for col in range(3):
            board[1][col] = Cell.X
        result = Board.is_terminal(board, 3, 3, (1, 2))
        assert result is Result.X_WINS

    def test_o_wins(self) -> None:
        board = Board.create(3)
        for row in range(3):
            board[row][2] = Cell.O
        result = Board.is_terminal(board, 3, 3, (2, 2))
        assert result is Result.O_WINS

    def test_draw(self) -> None:
        # X X O / O O X / X O X — fully filled, no winner.
        rows = [[1, 1, 2], [2, 2, 1], [1, 2, 1]]
        board = [[{0: Cell.EMPTY, 1: Cell.X, 2: Cell.O}[v] for v in row] for row in rows]
        result = Board.is_terminal(board, 3, 3, (2, 2))
        assert result is Result.DRAW

    def test_no_last_move(self) -> None:
        board = Board.create(3)
        result = Board.is_terminal(board, 3, 3, None)
        assert result is Result.IN_PROGRESS


# ---------------------------------------------------------------------------
# Candidate moves
# ---------------------------------------------------------------------------


class TestGetCandidateMoves:
    """Tests for Board.get_candidate_moves."""

    def test_centre_fallback_on_empty_board(self) -> None:
        state = make_state([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        moves = Board.get_candidate_moves(state)
        assert (1, 1) in moves

    def test_only_neighbours_returned(self) -> None:
        # Place one piece at (1,1) and expect only neighbours.
        state = make_state([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        moves = Board.get_candidate_moves(state, radius=1)
        # The 8 cells around (1,1) minus (1,1) itself.
        for row, col in moves:
            assert abs(row - 1) <= 1 and abs(col - 1) <= 1
            assert (row, col) != (1, 1)


# ---------------------------------------------------------------------------
# Winning and blocking moves
# ---------------------------------------------------------------------------


class TestWinningBlockingMoves:
    """Tests for Board.get_winning_move and Board.get_blocking_move."""

    def test_get_winning_move(self) -> None:
        # X X . / . O . / . . .
        board = Board.create(3)
        board[0][0] = Cell.X
        board[0][1] = Cell.X
        board[1][1] = Cell.O
        winning = Board.get_winning_move(board, 3, 3, Player.X)
        assert winning == (0, 2)

    def test_no_winning_move_returns_none(self) -> None:
        board = Board.create(3)
        board[0][0] = Cell.X
        assert Board.get_winning_move(board, 3, 3, Player.X) is None

    def test_get_blocking_move(self) -> None:
        # O O . / . X . / . . .
        board = Board.create(3)
        board[0][0] = Cell.O
        board[0][1] = Cell.O
        board[1][1] = Cell.X
        block = Board.get_blocking_move(board, 3, 3, Player.X)
        assert block == (0, 2)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


class TestRender:
    """Tests for Board.render."""

    def test_render_contains_indices(self) -> None:
        board = Board.create(3)
        output = Board.render(board, 3)
        for i in range(3):
            assert str(i) in output

    def test_render_highlights_last_move(self) -> None:
        board = Board.create(3)
        board[1][1] = Cell.X
        output = Board.render(board, 3, last_move=(1, 1))
        assert "[X]" in output

    def test_render_shows_pieces(self) -> None:
        board = Board.create(3)
        board[0][0] = Cell.X
        board[2][2] = Cell.O
        output = Board.render(board, 3)
        assert "X" in output
        assert "O" in output
