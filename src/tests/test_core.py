"""Tests for src/core: types, board, state."""

from __future__ import annotations

import numpy as np

from src.core.types import Player, DIRECTIONS
from src.core.board import Board
from src.core.state import State
from src.core.zobrist import ZobristTable


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
        assert ZobristTable(3)._table.shape == (3, 3, 3)

    def test_zobrist_table_dtype(self):
        assert ZobristTable(3)._table.dtype == np.uint64

    def test_zobrist_values_nonzero(self):
        assert np.all(ZobristTable(3)._table != 0)

    def test_zobrist_table_nxn_shape(self):
        assert ZobristTable(5)._table.shape == (5, 5, 3)

    def test_zobrist_hash_move_xor_inverse(self):
        zt = ZobristTable(3)
        h = zt.hash_move(0, 0, 0, int(Player.X))
        assert zt.hash_move(h, 0, 0, int(Player.X)) == 0


# ---------------------------------------------------------------------------
# board.py — unit tests
# ---------------------------------------------------------------------------


class TestBoardCreate:
    def test_create_shape(self):
        b = Board(3, 3)
        assert b._grid.shape == (3, 3)

    def test_create_dtype(self):
        b = Board(3, 3)
        assert b._grid.dtype == np.uint8

    def test_create_all_empty(self):
        b = Board(3, 3)
        assert (b._grid == Player._).all()

    def test_create_nxn_shape(self):
        b = Board(5, 3)
        assert b._grid.shape == (5, 5)

    def test_board_stores_n_and_k(self):
        b = Board(4, 3)
        assert b.n == 4
        assert b.k == 3


class TestBoardReset:
    def test_reset_clears_board(self):
        b = Board(3, 3)
        b.set(0, 0, Player.X)
        b.reset()
        assert (b._grid == Player._).all()


class TestBoardGetSet:
    def test_set_and_get(self):
        b = Board(3, 3)
        b.set(1, 2, Player.X)
        assert b.get(1, 2) is Player.X

    def test_set_o(self):
        b = Board(3, 3)
        b.set(0, 0, Player.O)
        assert b.get(0, 0) is Player.O

    def test_get_empty(self):
        b = Board(3, 3)
        assert b.get(2, 2) is Player._


class TestBoardIsEmpty:
    def test_empty_cell(self):
        b = Board(3, 3)
        assert b.is_empty(0, 0) is True

    def test_occupied_cell(self):
        b = Board(3, 3)
        b.set(0, 0, Player.X)
        assert b.is_empty(0, 0) is False


class TestBoardIsFull:
    def test_empty_board_not_full(self):
        b = Board(3, 3)
        assert b.is_full() is False

    def test_full_board(self):
        b = Board(3, 3)
        for r in range(3):
            for c in range(3):
                b.set(r, c, Player.X)
        assert b.is_full() is True

    def test_one_empty_not_full(self):
        b = Board(3, 3)
        for r in range(3):
            for c in range(3):
                b.set(r, c, Player.X)
        b._grid[2, 2] = Player._
        assert b.is_full() is False


class TestBoardGetEmptyCells:
    def test_all_empty(self):
        b = Board(3, 3)
        cells = b.get_empty_cells()
        assert len(cells) == 9

    def test_row_major_order(self):
        b = Board(3, 3)
        cells = b.get_empty_cells()
        assert cells[0] == (0, 0)
        assert cells[-1] == (2, 2)

    def test_partial_filled(self):
        b = Board(3, 3)
        b.set(0, 0, Player.X)
        b.set(1, 1, Player.O)
        cells = b.get_empty_cells()
        assert len(cells) == 7
        assert (0, 0) not in cells
        assert (1, 1) not in cells


class TestBoardCheckWin:
    def test_no_win_empty(self):
        b = Board(3, 3)
        b.set(0, 0, Player.X)
        assert b.check_win(0, 0) is False

    def test_horizontal_win(self):
        b = Board(3, 3)
        for c in range(3):
            b.set(0, c, Player.X)
        assert b.check_win(0, 1) is True

    def test_vertical_win(self):
        b = Board(3, 3)
        for r in range(3):
            b.set(r, 0, Player.O)
        assert b.check_win(1, 0) is True

    def test_diagonal_win(self):
        b = Board(3, 3)
        for i in range(3):
            b.set(i, i, Player.X)
        assert b.check_win(1, 1) is True

    def test_anti_diagonal_win(self):
        b = Board(3, 3)
        b.set(0, 2, Player.O)
        b.set(1, 1, Player.O)
        b.set(2, 0, Player.O)
        assert b.check_win(1, 1) is True

    def test_no_win_two_in_a_row(self):
        b = Board(3, 3)
        b.set(0, 0, Player.X)
        b.set(0, 1, Player.X)
        assert b.check_win(0, 1) is False

    def test_empty_cell_no_win(self):
        b = Board(3, 3)
        assert b.check_win(0, 0) is False

    def test_mixed_row_no_win(self):
        b = Board(3, 3)
        b.set(0, 0, Player.X)
        b.set(0, 1, Player.O)
        b.set(0, 2, Player.X)
        assert b.check_win(0, 0) is False

    def test_4x4_board_k4_win(self):
        b = Board(4, 4)
        for c in range(4):
            b.set(0, c, Player.X)
        assert b.check_win(0, 2) is True

    def test_4x4_board_k4_no_win_on_3(self):
        b = Board(4, 4)
        for c in range(3):
            b.set(0, c, Player.X)
        assert b.check_win(0, 1) is False


class TestBoardRender:
    def test_render_empty(self):
        b = Board(3, 3)
        result = b.render()
        assert "." in result
        assert result.count(".") == 9

    def test_render_with_last_move(self):
        b = Board(3, 3)
        b.set(1, 1, Player.X)
        result = b.render(1, 1)
        assert "[X]" in result

    def test_render_no_last_move(self):
        b = Board(3, 3)
        b.set(0, 0, Player.X)
        result = b.render()
        assert " X " in result
        assert "[X]" not in result


# ---------------------------------------------------------------------------
# state.py — unit, bound, and e2e tests
# ---------------------------------------------------------------------------


class TestStateInit:
    def test_initial_board_empty(self):
        s = State()
        assert (s.board._grid == Player._).all()

    def test_initial_player_x(self):
        s = State()
        assert s.current_player is Player.X

    def test_initial_history_empty(self):
        s = State()
        assert s.history == []

    def test_initial_state_count_zero(self):
        s = State()
        assert s._state_count == 0

    def test_initial_hash_zero(self):
        s = State()
        assert s._hash == 0

    def test_state_nxn(self):
        s = State(n=4, k=3)
        assert s.board._grid.shape == (4, 4)
        assert s.board.n == 4
        assert s.board.k == 3


class TestStateApply:
    def test_apply_sets_board(self):
        s = State()
        s.apply(0, 0)
        assert s.board.get(0, 0) is Player.X

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
        assert s._state_count == 1

    def test_apply_updates_hash(self):
        s = State()
        s.apply(0, 0)
        expected = s._zobrist.hash_move(0, 0, 0, int(Player.X))
        assert s._hash == expected

    def test_apply_does_not_double_count_same_hash(self):
        # Same position reached in one move then undone and redone
        s = State()
        s.apply(0, 0)
        s.undo()
        s.apply(0, 0)
        # visited already has this hash, state_count stays at 1
        assert s._state_count == 1

    def test_apply_multiple_moves(self):
        s = State()
        s.apply(0, 0)
        s.apply(1, 1)
        assert s.board.get(0, 0) is Player.X
        assert s.board.get(1, 1) is Player.O
        assert s.current_player is Player.X


class TestStateUndo:
    def test_undo_clears_cell(self):
        s = State()
        s.apply(0, 0)
        s.undo()
        assert s.board.get(0, 0) is Player._

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
        assert s._state_count == 1
        s.undo()
        # state_count only grows — undo does not remove from visited
        assert s._state_count == 1

    def test_undo_multiple_moves(self):
        s = State()
        s.apply(0, 0)
        s.apply(1, 1)
        s.undo()
        assert s.board.get(1, 1) is Player._
        assert s.current_player is Player.O
        s.undo()
        assert s.board.get(0, 0) is Player._
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
        moves = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (1, 2), (2, 1), (2, 0), (2, 2)]
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
        assert (s.board._grid == Player._).all()

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
        assert s._state_count == 0

    def test_reset_clears_visited(self):
        s = State()
        s.apply(0, 0)
        s.reset()
        assert len(s._visited) == 0

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
        assert s._state_count == 1
        s.undo()
        assert (s.board._grid == Player._).all()
        assert s._state_count == 1  # undo never removes from visited

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

    def test_4x4_k3_game(self):
        """4x4 board with k=3: 3-in-a-row wins."""
        s = State(n=4, k=3)
        s.apply(0, 0)  # X
        s.apply(1, 0)  # O
        s.apply(0, 1)  # X
        s.apply(1, 1)  # O
        s.apply(0, 2)  # X wins (3 in a row)
        assert s.is_terminal() is True
        assert s.winner() is Player.X


# ---------------------------------------------------------------------------
# board.py — get_candidate_cells
# ---------------------------------------------------------------------------


class TestGetCandidateCells:
    def test_empty_board_returns_center(self):
        b = Board(5, 3)
        cells = b.get_candidate_cells([], d=1)
        assert cells == [(2, 2)]

    def test_empty_board_even_n_returns_center(self):
        b = Board(4, 4)
        cells = b.get_candidate_cells([], d=2)
        assert cells == [(2, 2)]

    def test_single_stone_candidates_are_chebyshev_ring(self):
        b = Board(10, 5)
        d = 3
        history = [(5, 5)]
        candidates = set(b.get_candidate_cells(history, d=d))
        expected = {
            (r, c)
            for r in range(10)
            for c in range(10)
            if max(abs(r - 5), abs(c - 5)) <= d and b.is_empty(r, c)
        }
        assert candidates == expected

    def test_candidates_subset_of_empty_cells(self):
        b = Board(10, 5)
        b.set(3, 3, Player.X)
        b.set(4, 4, Player.O)
        history = [(3, 3), (4, 4)]
        candidates = set(b.get_candidate_cells(history, d=3))
        empty = set(b.get_empty_cells())
        assert candidates <= empty

    def test_no_occupied_cell_in_candidates(self):
        b = Board(10, 5)
        b.set(3, 3, Player.X)
        b.set(4, 4, Player.O)
        history = [(3, 3), (4, 4)]
        candidates = b.get_candidate_cells(history, d=3)
        assert (3, 3) not in candidates
        assert (4, 4) not in candidates

    def test_small_board_fallback_to_empty_cells(self):
        # 4x4 with d=2: (2*2+1)^2=25 > 16=n^2, always falls back
        b = Board(4, 4)
        b.set(0, 0, Player.X)
        history = [(0, 0)]
        candidates = set(b.get_candidate_cells(history, d=2))
        assert candidates == set(b.get_empty_cells())


# ---------------------------------------------------------------------------
# tt_state.py
# ---------------------------------------------------------------------------

from src.core.tt_state import TTState
from src.core.manipulator import Manipulator


class TestTTStateInit:
    def test_hashes_length(self):
        s = TTState(3, 3)
        assert len(s._hashes) == Manipulator.TRANSFORM_COUNT

    def test_hashes_all_zero(self):
        s = TTState(3, 3)
        assert s._hashes == [0] * Manipulator.TRANSFORM_COUNT

    def test_inherits_state_behaviour(self):
        s = TTState(3, 3)
        assert s.current_player is Player.X
        assert s.history == []
        assert s._hash == 0
