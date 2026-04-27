import pytest

from src.core.transposition_table import TranspositionTable
from src.core.types import NEGATIVE_INFINITY

INF = -NEGATIVE_INFINITY  # float("inf")


class TestInit:
    def test_empty_on_creation(self):
        assert len(TranspositionTable()) == 0


class TestStore:
    def test_len_increases_on_new_key(self):
        tt = TranspositionTable()
        tt.store(1, NEGATIVE_INFINITY, INF, None)
        assert len(tt) == 1

    def test_len_two_distinct_keys(self):
        tt = TranspositionTable()
        tt.store(1, NEGATIVE_INFINITY, INF, None)
        tt.store(2, NEGATIVE_INFINITY, INF, None)
        assert len(tt) == 2

    def test_overwrite_same_key_no_len_change(self):
        tt = TranspositionTable()
        tt.store(1, NEGATIVE_INFINITY, INF, None)
        tt.store(1, 0.5, 0.5, (0, 0))
        assert len(tt) == 1


class TestLookup:
    def test_miss_returns_none(self):
        assert TranspositionTable().lookup(42) is None

    def test_hit_returns_full_tuple(self):
        tt = TranspositionTable()
        tt.store(7, -0.5, 0.8, (1, 2))
        assert tt.lookup(7) == (-0.5, 0.8, (1, 2))

    def test_lower_bound_stored(self):
        tt = TranspositionTable()
        tt.store(1, 0.3, INF, None)
        lb, ub, _ = tt.lookup(1)
        assert lb == pytest.approx(0.3)
        assert ub == INF

    def test_upper_bound_stored(self):
        tt = TranspositionTable()
        tt.store(1, NEGATIVE_INFINITY, 0.3, None)
        lb, ub, _ = tt.lookup(1)
        assert lb == NEGATIVE_INFINITY
        assert ub == pytest.approx(0.3)

    def test_exact_stored(self):
        tt = TranspositionTable()
        tt.store(1, 0.3, 0.3, (0, 1))
        lb, ub, bm = tt.lookup(1)
        assert lb == pytest.approx(0.3)
        assert ub == pytest.approx(0.3)
        assert bm == (0, 1)

    def test_none_best_move_round_trips(self):
        tt = TranspositionTable()
        tt.store(5, 0.0, 0.0, None)
        assert tt.lookup(5)[2] is None


class TestBestMove:
    def test_miss_returns_none(self):
        assert TranspositionTable().best_move(99) is None

    def test_hit_returns_move(self):
        tt = TranspositionTable()
        tt.store(3, 0.0, 0.5, (1, 2))
        assert tt.best_move(3) == (1, 2)

    def test_none_best_move_returns_none(self):
        tt = TranspositionTable()
        tt.store(3, 0.0, 0.5, None)
        assert tt.best_move(3) is None


class TestOverwrite:
    def test_overwrite_updates_lower(self):
        tt = TranspositionTable()
        tt.store(1, NEGATIVE_INFINITY, 0.5, None)
        tt.store(1, 0.3, 0.5, None)
        lb, _, _ = tt.lookup(1)
        assert lb == pytest.approx(0.3)

    def test_overwrite_updates_upper(self):
        tt = TranspositionTable()
        tt.store(1, NEGATIVE_INFINITY, 0.5, None)
        tt.store(1, NEGATIVE_INFINITY, 0.3, None)
        _, ub, _ = tt.lookup(1)
        assert ub == pytest.approx(0.3)

    def test_overwrite_updates_best_move(self):
        tt = TranspositionTable()
        tt.store(1, 0.0, 0.0, (0, 0))
        tt.store(1, 0.0, 0.0, (1, 1))
        assert tt.best_move(1) == (1, 1)
