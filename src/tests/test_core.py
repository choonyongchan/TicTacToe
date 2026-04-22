"""Tests for src/core: types, board, state."""
from __future__ import annotations

import numpy as np

from src.core.types import Player, DIRECTIONS, ZOBRIST_TABLE
from src.core.board import Board as board_mod
from src.core.state import State


# ---------------------------------------------------------------------------
# types.py
# ---------------------------------------------------------------------------

class TestTypes:
    def test_player_values(self):
        assert int(Player._) == 0
        assert int(Player.X) == 1
        assert int(Player.O) == 2

    def test_player_opponent(self):
        assert Player.X.opponent() is Player.O
        assert Player.O.opponent() is Player.X

    def test_directions_count(self):
        assert len(DIRECTIONS) == 4

    def test_direction_tuples(self):
        assert (0, 1) in [(d.dr, d.dc) for d in DIRECTIONS]  # horizontal
        assert (1, 0) in [(d.dr, d.dc) for d in DIRECTIONS]  # vertical

    def test_zobrist_table_shape(self):
        assert ZOBRIST_TABLE.shape == (3, 3, 3)

    def test_zobrist_table_dtype(self):
        assert ZOBRIST_TABLE.dtype == np.uint64

    def test_zobrist_values_nonzero(self):
        assert np.all(ZOBRIST_TABLE != 0)


# ---------------------------------------------------------------------------
# board.py — unit tests
# ---------------------------------------------------------------------------

class TestBoardCreate:
    def test_create_shape(self):
        b = board_mod.create()
        assert b.shape == (3, 3)

    def test_create_dtype(self):
        b = board_mod.create()
        assert b.dtype == np.uint8

    def test_create_all_empty(self):
        b = board_mod.create()
        assert (b == Player._).all()


class TestBoardReset:
    def test_reset_clears_board(self):
        b = board_mod.create()
        board_mod.set(b, 0, 0, Player.X)
        board_mod.reset(b)
        assert (b == Player._).all()


class TestBoardGetSet:
    def test_set_and_get(self):
        b = board_mod.create()
        board_mod.set(b, 1, 2, Player.X)
        assert board_mod.get(b, 1, 2) is Player.X

    def test_set_o(self):
        b = board_mod.create()
        board_mod.set(b, 0, 0, Player.O)
        assert board_mod.get(b, 0, 0) is Player.O

    def test_get_empty(self):
        b = board_mod.create()
        assert board_mod.get(b, 2, 2) is Player._


class TestBoardIsEmpty:
    def test_empty_cell(self):
        b = board_mod.create()
        assert board_mod.is_empty(b, 0, 0) is True

    def test_occupied_cell(self):
        b = board_mod.create()
        board_mod.set(b, 0, 0, Player.X)
        assert board_mod.is_empty(b, 0, 0) is False


class TestBoardIsFull:
    def test_empty_board_not_full(self):
        b = board_mod.create()
        assert board_mod.is_full(b) is False

    def test_full_board(self):
        b = board_mod.create()
        for r in range(3):
            for c in range(3):
                board_mod.set(b, r, c, Player.X)
        assert board_mod.is_full(b) is True

    def test_one_empty_not_full(self):
        b = board_mod.create()
        for r in range(3):
            for c in range(3):
                board_mod.set(b, r, c, Player.X)
        b[2, 2] = Player._
        assert board_mod.is_full(b) is False


class TestBoardGetEmptyCells:
    def test_all_empty(self):
        b = board_mod.create()
        cells = board_mod.get_empty_cells(b)
        assert len(cells) == 9

    def test_row_major_order(self):
        b = board_mod.create()
        cells = board_mod.get_empty_cells(b)
        assert cells[0] == (0, 0)
        assert cells[-1] == (2, 2)

    def test_partial_filled(self):
        b = board_mod.create()
        board_mod.set(b, 0, 0, Player.X)
        board_mod.set(b, 1, 1, Player.O)
        cells = board_mod.get_empty_cells(b)
        assert len(cells) == 7
        assert (0, 0) not in cells
        assert (1, 1) not in cells


class TestBoardCheckWin:
    def test_no_win_empty(self):
        b = board_mod.create()
        board_mod.set(b, 0, 0, Player.X)
        assert board_mod.check_win(b, 0, 0) is False

    def test_horizontal_win(self):
        b = board_mod.create()
        for c in range(3):
            board_mod.set(b, 0, c, Player.X)
        assert board_mod.check_win(b, 0, 1) is True

    def test_vertical_win(self):
        b = board_mod.create()
        for r in range(3):
            board_mod.set(b, r, 0, Player.O)
        assert board_mod.check_win(b, 1, 0) is True

    def test_diagonal_win(self):
        b = board_mod.create()
        for i in range(3):
            board_mod.set(b, i, i, Player.X)
        assert board_mod.check_win(b, 1, 1) is True

    def test_anti_diagonal_win(self):
        b = board_mod.create()
        board_mod.set(b, 0, 2, Player.O)
        board_mod.set(b, 1, 1, Player.O)
        board_mod.set(b, 2, 0, Player.O)
        assert board_mod.check_win(b, 1, 1) is True

    def test_no_win_two_in_a_row(self):
        b = board_mod.create()
        board_mod.set(b, 0, 0, Player.X)
        board_mod.set(b, 0, 1, Player.X)
        assert board_mod.check_win(b, 0, 1) is False

    def test_empty_cell_no_win(self):
        b = board_mod.create()
        assert board_mod.check_win(b, 0, 0) is False

    def test_mixed_row_no_win(self):
        b = board_mod.create()
        board_mod.set(b, 0, 0, Player.X)
        board_mod.set(b, 0, 1, Player.O)
        board_mod.set(b, 0, 2, Player.X)
        assert board_mod.check_win(b, 0, 0) is False


class TestBoardRender:
    def test_render_empty(self):
        b = board_mod.create()
        result = board_mod.render(b)
        assert "." in result
        assert result.count(".") == 9

    def test_render_with_last_move(self):
        b = board_mod.create()
        board_mod.set(b, 1, 1, Player.X)
        result = board_mod.render(b, 1, 1)
        assert "[X]" in result

    def test_render_no_last_move(self):
        b = board_mod.create()
        board_mod.set(b, 0, 0, Player.X)
        result = board_mod.render(b)
        assert " X " in result
        assert "[X]" not in result


# ---------------------------------------------------------------------------
# state.py — unit, bound, and e2e tests
# ---------------------------------------------------------------------------

class TestStateInit:
    def test_initial_board_empty(self):
        s = State()
        assert (s.board == Player._).all()

    def test_initial_player_x(self):
        s = State()
        assert s.current_player is Player.X

    def test_initial_history_empty(self):
        s = State()
        assert s.history == []

    def test_initial_state_count_zero(self):
        s = State()
        assert s.state_count == 0

    def test_initial_hash_zero(self):
        s = State()
        assert s._hash == 0


class TestStateApply:
    def test_apply_sets_board(self):
        s = State()
        s.apply(0, 0)
        assert board_mod.get(s.board, 0, 0) is Player.X

    def test_apply_switches_player(self):
        s = State()
        s.apply(0, 0)
        assert s.current_player is Player.O

    def test_apply_appends_history(self):
        s = State()
        s.apply(1, 1)
        assert s.history == [(1, 1)]

    def test_apply_increments_state_count_first_move(self):
        s = State()
        s.apply(0, 0)
        assert s.state_count == 1

    def test_apply_updates_hash(self):
        s = State()
        s.apply(0, 0)
        expected = int(ZOBRIST_TABLE[0, 0, int(Player.X)])
        assert s._hash == expected

    def test_apply_does_not_double_count_same_hash(self):
        # Same position reached in one move then undone and redone
        s = State()
        s.apply(0, 0)
        s.undo()
        s.apply(0, 0)
        # visited already has this hash, state_count stays at 1
        assert s.state_count == 1

    def test_apply_multiple_moves(self):
        s = State()
        s.apply(0, 0)
        s.apply(1, 1)
        assert board_mod.get(s.board, 0, 0) is Player.X
        assert board_mod.get(s.board, 1, 1) is Player.O
        assert s.current_player is Player.X


class TestStateUndo:
    def test_undo_clears_cell(self):
        s = State()
        s.apply(0, 0)
        s.undo()
        assert board_mod.get(s.board, 0, 0) is Player._

    def test_undo_restores_player(self):
        s = State()
        s.apply(0, 0)
        s.undo()
        assert s.current_player is Player.X

    def test_undo_pops_history(self):
        s = State()
        s.apply(0, 0)
        s.undo()
        assert s.history == []

    def test_undo_restores_hash(self):
        s = State()
        original_hash = s._hash
        s.apply(0, 0)
        s.undo()
        assert s._hash == original_hash

    def test_undo_does_not_decrement_state_count(self):
        s = State()
        s.apply(0, 0)
        assert s.state_count == 1
        s.undo()
        # state_count only grows — undo does not remove from visited
        assert s.state_count == 1

    def test_undo_multiple_moves(self):
        s = State()
        s.apply(0, 0)
        s.apply(1, 1)
        s.undo()
        assert board_mod.get(s.board, 1, 1) is Player._
        assert s.current_player is Player.O
        s.undo()
        assert board_mod.get(s.board, 0, 0) is Player._
        assert s.current_player is Player.X


class TestStateIsTerminal:
    def test_not_terminal_start(self):
        s = State()
        assert s.is_terminal() is False

    def test_terminal_on_win(self):
        s = State()
        # X wins top row
        s.apply(0, 0)  # X
        s.apply(1, 0)  # O
        s.apply(0, 1)  # X
        s.apply(1, 1)  # O
        s.apply(0, 2)  # X wins
        assert s.is_terminal() is True

    def test_not_terminal_partial(self):
        s = State()
        s.apply(0, 0)
        s.apply(1, 1)
        assert s.is_terminal() is False

    def test_terminal_draw(self):
        # Fill board without a winner: X O X / X O X / O X O
        moves = [(0,0),(0,1),(0,2),(1,1),(1,0),(1,2),(2,1),(2,0),(2,2)]
        s = State()
        for m in moves:
            s.apply(*m)
        assert s.is_terminal() is True
        assert s.winner() is None


class TestStateWinner:
    def test_winner_none_start(self):
        s = State()
        assert s.winner() is None

    def test_winner_x(self):
        s = State()
        s.apply(0, 0)  # X
        s.apply(1, 0)  # O
        s.apply(0, 1)  # X
        s.apply(1, 1)  # O
        s.apply(0, 2)  # X wins
        assert s.winner() is Player.X

    def test_winner_o(self):
        s = State()
        s.apply(0, 0)  # X
        s.apply(1, 0)  # O
        s.apply(0, 1)  # X
        s.apply(1, 1)  # O
        s.apply(2, 2)  # X
        s.apply(1, 2)  # O wins col 1
        assert s.winner() is Player.O

    def test_no_winner_mid_game(self):
        s = State()
        s.apply(0, 0)
        assert s.winner() is None


class TestStateReset:
    def test_reset_clears_board(self):
        s = State()
        s.apply(0, 0)
        s.reset()
        assert (s.board == Player._).all()

    def test_reset_player_is_x(self):
        s = State()
        s.apply(0, 0)
        s.reset()
        assert s.current_player is Player.X

    def test_reset_clears_history(self):
        s = State()
        s.apply(0, 0)
        s.reset()
        assert s.history == []

    def test_reset_clears_state_count(self):
        s = State()
        s.apply(0, 0)
        s.reset()
        assert s.state_count == 0

    def test_reset_clears_visited(self):
        s = State()
        s.apply(0, 0)
        s.reset()
        assert len(s.visited) == 0

    def test_reset_clears_hash(self):
        s = State()
        s.apply(0, 0)
        s.reset()
        assert s._hash == 0


# ---------------------------------------------------------------------------
# e2e: Spec verification tests
# ---------------------------------------------------------------------------

class TestSpecVerification:
    def test_three_in_a_row_terminal_and_winner(self):
        """Spec v2: Apply 3-in-a-row, assert is_terminal and winner == Player.X."""
        s = State()
        s.apply(0, 0)  # X
        s.apply(1, 0)  # O
        s.apply(0, 1)  # X
        s.apply(1, 1)  # O
        s.apply(0, 2)  # X — top row complete
        assert s.is_terminal() is True
        assert s.winner() is Player.X

    def test_apply_undo_board_unchanged(self):
        """Spec v3: Apply then undo, board unchanged and state_count stays at 1."""
        s = State()
        s.apply(1, 1)
        assert s.state_count == 1
        s.undo()
        assert (s.board == Player._).all()
        assert s.state_count == 1  # undo never removes from visited

    def test_two_paths_same_hash(self):
        """Spec v4: Two paths to the same board produce the same _hash."""
        s1 = State()
        s1.apply(0, 0)  # X
        s1.apply(1, 1)  # O

        s2 = State()
        s2.apply(0, 0)  # X
        s2.apply(1, 1)  # O

        assert s1._hash == s2._hash

    def test_different_positions_different_hash(self):
        s1 = State()
        s1.apply(0, 0)

        s2 = State()
        s2.apply(1, 1)

        assert s1._hash != s2._hash
