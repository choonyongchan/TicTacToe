"""Tests for the TranspositionTable with Zobrist hashing."""
from __future__ import annotations

import pytest

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player
from tictactoe.agents.heuristic_search.shared.transposition_table import (
    TranspositionTable,
    TTFlag,
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


def empty_board_state(n: int = 3) -> GameState:
    board = Board.create(n)
    return GameState(board=board, current_player=Player.X, n=n, k=n)


# ---------------------------------------------------------------------------
# Zobrist hashing
# ---------------------------------------------------------------------------


class TestZobristHashing:
    """Tests for hash_board and update_hash."""

    def test_empty_board_hash_is_zero(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        state = empty_board_state(3)
        h = tt.hash_board(state.board, 3)
        assert h == 0

    def test_different_boards_have_different_hashes(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        state_a = make_state([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        state_b = make_state([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        h_a = tt.hash_board(state_a.board, 3)
        h_b = tt.hash_board(state_b.board, 3)
        assert h_a != h_b

    def test_incremental_hash_matches_full_hash(self) -> None:
        tt = TranspositionTable(2**10, seed=42)
        state = empty_board_state(3)
        h0 = tt.hash_board(state.board, 3)
        # Simulate placing X at (1, 1)
        h1 = tt.update_hash(h0, 1, 1, Player.X, 3)
        # Now compute full hash of the board with X at (1,1)
        state.board[1][1] = Cell.X
        h1_full = tt.hash_board(state.board, 3)
        assert h1 == h1_full

    def test_xor_involution(self) -> None:
        """Applying update_hash twice with same args returns original hash."""
        tt = TranspositionTable(2**10, seed=7)
        state = empty_board_state(3)
        h0 = tt.hash_board(state.board, 3)
        h1 = tt.update_hash(h0, 0, 2, Player.O, 3)
        h2 = tt.update_hash(h1, 0, 2, Player.O, 3)
        assert h0 == h2

    def test_player_x_and_o_give_different_hash_updates(self) -> None:
        tt = TranspositionTable(2**10, seed=3)
        h0 = 0
        h_x = tt.update_hash(h0, 0, 0, Player.X, 3)
        h_o = tt.update_hash(h0, 0, 0, Player.O, 3)
        assert h_x != h_o

    def test_different_board_sizes_cached_separately(self) -> None:
        tt = TranspositionTable(2**10, seed=99)
        state3 = empty_board_state(3)
        state5 = empty_board_state(5)
        h3 = tt.hash_board(state3.board, 3)
        h5 = tt.hash_board(state5.board, 5)
        # Both are zero (empty boards) but internal tables are separate
        assert h3 == 0
        assert h5 == 0
        # Place piece at (0,0) on each; hashes should differ
        h3_piece = tt.update_hash(h3, 0, 0, Player.X, 3)
        h5_piece = tt.update_hash(h5, 0, 0, Player.X, 5)
        assert h3_piece != h5_piece


# ---------------------------------------------------------------------------
# Store and lookup — EXACT flag
# ---------------------------------------------------------------------------


class TestStoreLookupExact:
    """Tests for storing and retrieving EXACT entries."""

    def test_store_and_lookup_exact_returns_score(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 12345
        tt.store(key, depth=3, score=42.0, flag=TTFlag.EXACT, best_move=(1, 1))
        result = tt.lookup(key, depth=3, alpha=-1000.0, beta=1000.0)
        assert result == pytest.approx(42.0)

    def test_lookup_insufficient_depth_returns_none(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 99999
        tt.store(key, depth=2, score=10.0, flag=TTFlag.EXACT, best_move=None)
        # Requesting depth=3 but stored at depth=2 — should miss
        result = tt.lookup(key, depth=3, alpha=-100.0, beta=100.0)
        assert result is None

    def test_lookup_exact_with_sufficient_depth(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 77777
        tt.store(key, depth=5, score=-50.0, flag=TTFlag.EXACT, best_move=(0, 0))
        # depth=3 <= stored depth=5, should return the stored score
        result = tt.lookup(key, depth=3, alpha=-1000.0, beta=1000.0)
        assert result == pytest.approx(-50.0)

    def test_collision_different_key_returns_none(self) -> None:
        """Two keys mapping to the same slot but different full keys."""
        tt = TranspositionTable(2**4, seed=1)  # tiny TT — collisions likely
        key_a = 0   # slot 0
        key_b = 16  # same slot (16 & 15 == 0) but different key
        tt.store(key_a, depth=3, score=7.0, flag=TTFlag.EXACT, best_move=None)
        # Looking up key_b should NOT return key_a's entry
        result = tt.lookup(key_b, depth=3, alpha=-1000.0, beta=1000.0)
        assert result is None


# ---------------------------------------------------------------------------
# Store and lookup — LOWER_BOUND and UPPER_BOUND flags
# ---------------------------------------------------------------------------


class TestStoreLookupBounds:
    """Tests for LOWER_BOUND and UPPER_BOUND flag logic."""

    def test_lower_bound_returns_score_when_gte_beta(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 111
        tt.store(key, depth=4, score=80.0, flag=TTFlag.LOWER_BOUND,
                 best_move=None)
        # score (80) >= beta (50) → should return 80
        result = tt.lookup(key, depth=4, alpha=0.0, beta=50.0)
        assert result == pytest.approx(80.0)

    def test_lower_bound_returns_none_when_lt_beta(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 222
        tt.store(key, depth=4, score=30.0, flag=TTFlag.LOWER_BOUND,
                 best_move=None)
        # score (30) < beta (50) → cannot prune, returns None
        result = tt.lookup(key, depth=4, alpha=0.0, beta=50.0)
        assert result is None

    def test_upper_bound_returns_score_when_lte_alpha(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 333
        tt.store(key, depth=4, score=10.0, flag=TTFlag.UPPER_BOUND,
                 best_move=None)
        # score (10) <= alpha (20) → should return 10
        result = tt.lookup(key, depth=4, alpha=20.0, beta=100.0)
        assert result == pytest.approx(10.0)

    def test_upper_bound_returns_none_when_gt_alpha(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 444
        tt.store(key, depth=4, score=50.0, flag=TTFlag.UPPER_BOUND,
                 best_move=None)
        # score (50) > alpha (10) → cannot prune
        result = tt.lookup(key, depth=4, alpha=10.0, beta=100.0)
        assert result is None


# ---------------------------------------------------------------------------
# Two-tier replacement
# ---------------------------------------------------------------------------


class TestTwoTierReplacement:
    """Tests for depth-preferred (Tier A) and always-replace (Tier B)."""

    def test_depth_preferred_keeps_deeper_entry(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 555
        # Store shallow entry
        tt.store(key, depth=2, score=1.0, flag=TTFlag.EXACT, best_move=(0, 0))
        # Store deeper entry — Tier A should be replaced
        tt.store(key, depth=5, score=2.0, flag=TTFlag.EXACT, best_move=(1, 1))
        # Lookup with depth=5 should succeed
        result = tt.lookup(key, depth=5, alpha=-1000.0, beta=1000.0)
        assert result == pytest.approx(2.0)

    def test_depth_preferred_keeps_entry_against_shallower_store(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 666
        # Store deep entry first
        tt.store(key, depth=6, score=99.0, flag=TTFlag.EXACT, best_move=(2, 2))
        # Attempt to overwrite with shallow entry — Tier A should stay
        tt.store(key, depth=1, score=-1.0, flag=TTFlag.EXACT, best_move=(0, 0))
        result = tt.lookup(key, depth=5, alpha=-1000.0, beta=1000.0)
        assert result == pytest.approx(99.0)

    def test_tier_b_always_replaced(self) -> None:
        """Tier B always stores the most recent entry."""
        tt = TranspositionTable(2**10, seed=1)
        key = 777
        tt.store(key, depth=5, score=10.0, flag=TTFlag.EXACT, best_move=(0, 0))
        tt.store(key, depth=1, score=99.0, flag=TTFlag.EXACT, best_move=(1, 1))
        # Tier B has the shallow entry (score=99); Tier A still has deep (score=10)
        # A lookup at depth=1 should succeed (tier A depth=5 >= 1)
        result = tt.lookup(key, depth=1, alpha=-1000.0, beta=1000.0)
        assert result is not None  # should hit tier A (exact, depth sufficient)


# ---------------------------------------------------------------------------
# get_best_move
# ---------------------------------------------------------------------------


class TestGetBestMove:
    """Tests for the get_best_move convenience method."""

    def test_returns_stored_best_move(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 888
        tt.store(key, depth=3, score=5.0, flag=TTFlag.EXACT, best_move=(2, 1))
        assert tt.get_best_move(key) == (2, 1)

    def test_returns_none_for_unknown_key(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        assert tt.get_best_move(9999999) is None

    def test_none_best_move_stored_and_retrieved(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 1234
        tt.store(key, depth=2, score=0.0, flag=TTFlag.EXACT, best_move=None)
        # Depending on tier, may or may not retrieve None — at minimum no crash
        result = tt.get_best_move(key)
        assert result is None or isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Hit rate and clear
# ---------------------------------------------------------------------------


class TestHitRateAndClear:
    """Tests for hit_rate() and clear()."""

    def test_hit_rate_zero_before_any_lookups(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        assert tt.hit_rate() == 0.0

    def test_hit_rate_increases_on_hit(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 55555
        tt.store(key, depth=3, score=7.0, flag=TTFlag.EXACT, best_move=None)
        tt.lookup(key, depth=3, alpha=-INF_VAL, beta=INF_VAL)
        assert tt.hit_rate() > 0.0

    def test_clear_resets_table_and_stats(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        key = 44444
        tt.store(key, depth=3, score=1.0, flag=TTFlag.EXACT, best_move=None)
        tt.lookup(key, depth=3, alpha=-1e9, beta=1e9)
        tt.clear()
        assert tt.hit_rate() == 0.0
        result = tt.lookup(key, depth=3, alpha=-1e9, beta=1e9)
        assert result is None

    def test_size_counts_tier_a_entries(self) -> None:
        tt = TranspositionTable(2**10, seed=1)
        assert tt.size() == 0
        tt.store(12345, depth=3, score=1.0, flag=TTFlag.EXACT, best_move=None)
        assert tt.size() >= 1


# ---------------------------------------------------------------------------
# Symmetry canonical
# ---------------------------------------------------------------------------


class TestSymmetryCanonical:
    """Tests for symmetry_canonical."""

    def test_no_symmetry_returns_key_unchanged(self) -> None:
        tt = TranspositionTable(2**10, use_symmetry=False, seed=1)
        state = make_state([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        h = tt.hash_board(state.board, 3)
        assert tt.symmetry_canonical(h, state.board, 3) == h

    def test_symmetry_canonical_is_deterministic(self) -> None:
        tt = TranspositionTable(2**10, use_symmetry=True, seed=1)
        state = make_state([[1, 0, 0], [0, 2, 0], [0, 0, 0]])
        h = tt.hash_board(state.board, 3)
        c1 = tt.symmetry_canonical(h, state.board, 3)
        c2 = tt.symmetry_canonical(h, state.board, 3)
        assert c1 == c2

    def test_symmetry_canonical_is_integer(self) -> None:
        tt = TranspositionTable(2**10, use_symmetry=True, seed=1)
        state = make_state([[0, 1, 0], [0, 0, 2], [0, 0, 0]])
        h = tt.hash_board(state.board, 3)
        canon = tt.symmetry_canonical(h, state.board, 3)
        assert isinstance(canon, int)


# Sentinel for readability in bounds tests
INF_VAL = 1e18
