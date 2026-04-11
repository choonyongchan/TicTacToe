"""Tests for the _shared_utils module.

Directly tests check_forced_move(), which is otherwise only covered
indirectly through agent implementations.
"""
from __future__ import annotations

import pytest

from tictactoe.agents._shared_utils import check_forced_move
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player, Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(rows: list[str], player: Player = Player.X, k: int | None = None) -> GameState:
    """Build a GameState from a list of strings like ['XX.', '...', '..O']."""
    n = len(rows)
    if k is None:
        k = n
    board = Board.create(n)
    last_move = None
    move_history = []
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
    return GameState(
        board=board,
        current_player=player,
        n=n,
        k=k,
        last_move=last_move,
        move_history=move_history,
    )


# ---------------------------------------------------------------------------
# Winning move detection
# ---------------------------------------------------------------------------


class TestWinningMove:
    """check_forced_move returns the winning move when one exists."""

    def test_x_completes_top_row(self) -> None:
        # X has (0,0) and (0,1); winning move is (0,2).
        state = _make_state(["XX.", "...", "..."], player=Player.X, k=3)
        move = check_forced_move(state)
        assert move == (0, 2), f"Expected (0,2), got {move}"

    def test_x_completes_column(self) -> None:
        # X has (0,0) and (1,0); winning move is (2,0).
        state = _make_state(["X..", "X..", "..."], player=Player.X, k=3)
        move = check_forced_move(state)
        assert move == (2, 0), f"Expected (2,0), got {move}"

    def test_o_completes_diagonal(self) -> None:
        # O has (0,0) and (1,1); winning move is (2,2).
        state = _make_state(["O..", ".O.", "..."], player=Player.O, k=3)
        move = check_forced_move(state)
        assert move == (2, 2), f"Expected (2,2), got {move}"


# ---------------------------------------------------------------------------
# Blocking move detection
# ---------------------------------------------------------------------------


class TestBlockingMove:
    """check_forced_move returns the blocking move when the opponent threatens."""

    def test_x_blocks_o_top_row(self) -> None:
        # O has (0,0) and (0,1); X must block at (0,2).
        state = _make_state(["OO.", "X..", "..."], player=Player.X, k=3)
        move = check_forced_move(state)
        assert move == (0, 2), f"Expected (0,2), got {move}"

    def test_o_blocks_x_column(self) -> None:
        # X has (0,0) and (1,0); O must block at (2,0).
        state = _make_state(["X..", "X..", "..."], player=Player.O, k=3)
        move = check_forced_move(state)
        assert move == (2, 0), f"Expected (2,0), got {move}"


# ---------------------------------------------------------------------------
# No forced move
# ---------------------------------------------------------------------------


class TestNoForcedMove:
    """check_forced_move returns None when there is no immediate win or block."""

    def test_empty_board_returns_none(self) -> None:
        state = _make_state(["...", "...", "..."], player=Player.X, k=3)
        assert check_forced_move(state) is None

    def test_scattered_pieces_returns_none(self) -> None:
        # No player has k-1 in a row on any line.
        state = _make_state(["X..", ".O.", "..X"], player=Player.X, k=3)
        assert check_forced_move(state) is None

    def test_single_piece_returns_none(self) -> None:
        state = _make_state(["X..", "...", "..."], player=Player.O, k=3)
        assert check_forced_move(state) is None


# ---------------------------------------------------------------------------
# Winning takes priority over blocking
# ---------------------------------------------------------------------------


class TestPriority:
    """Winning move is returned when both a win and a block are available."""

    def test_win_over_block(self) -> None:
        # X can win at (0,2) and must also block O at (1,2), but win comes first.
        # Board:  XX. / OO. / ...
        state = _make_state(["XX.", "OO.", "..."], player=Player.X, k=3)
        move = check_forced_move(state)
        assert move == (0, 2), (
            f"Expected winning move (0,2), got {move}"
        )


# ---------------------------------------------------------------------------
# Returned move is always a legal empty cell
# ---------------------------------------------------------------------------


class TestMoveLegality:
    """Forced move must always be an empty cell within board bounds."""

    def test_winning_move_is_empty_cell(self) -> None:
        state = _make_state(["XX.", "...", "..."], player=Player.X, k=3)
        move = check_forced_move(state)
        assert move is not None
        r, c = move
        assert 0 <= r < state.n
        assert 0 <= c < state.n
        assert state.board[r][c] is Cell.EMPTY

    def test_blocking_move_is_empty_cell(self) -> None:
        state = _make_state(["OO.", "X..", "..."], player=Player.X, k=3)
        move = check_forced_move(state)
        assert move is not None
        r, c = move
        assert 0 <= r < state.n
        assert 0 <= c < state.n
        assert state.board[r][c] is Cell.EMPTY
