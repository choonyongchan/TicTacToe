from __future__ import annotations

import pytest

from src.heuristics.distance_heuristic import DistanceHeuristic
from src.tests.test_helper import fresh_state, state_with_moves


class TestDistanceHeuristic:
    def test_empty_board_returns_zero(self):
        # Both players equidistant on an empty board.
        h = DistanceHeuristic()
        assert h.evaluate(fresh_state(3, 3)) == pytest.approx(0.0)

    def test_current_player_ahead_returns_positive(self):
        # X has (0,0),(0,1) — gap 1. O has (0,3),(3,0) — corner pieces with no shared k-window.
        # After 4 moves, current=X, n=4 k=3. dist_me(X)=1, dist_opp(O)=2. score=(2-1)/2=0.5
        state = state_with_moves([(0, 0), (0, 3), (0, 1), (3, 0)], n=4, k=3)
        h = DistanceHeuristic()
        assert h.evaluate(state) == pytest.approx(0.5)

    def test_current_player_behind_returns_negative(self):
        # X has (0,0),(0,1) — gap 1. Current player = O (3 moves done, O goes next).
        # dist_me(O): O has (3,3), best unblocked window gap = 2.
        # dist_opp(X) = 1. score=(1-2)/(3-1)=-0.5
        state = state_with_moves([(0, 0), (3, 3), (0, 1)], n=4, k=3)
        h = DistanceHeuristic()
        assert h.evaluate(state) == pytest.approx(-0.5)

    def test_symmetric_position_returns_zero(self):
        # X at (0,0), O at (3,3) — both have gap 2 on n=4 k=3.
        state = state_with_moves([(0, 0), (3, 3)], n=4, k=3)
        h = DistanceHeuristic()
        assert h.evaluate(state) == pytest.approx(0.0)

    def test_result_in_bounds(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)], n=5, k=3)
        h = DistanceHeuristic()
        score = h.evaluate(state)
        assert -1.0 <= score <= 1.0


import math
from src.heuristics.taxonomy_heuristic import TaxonomyHeuristic


class TestTaxonomyHeuristic:
    def test_empty_board_returns_zero(self):
        h = TaxonomyHeuristic()
        assert h.evaluate(fresh_state(3, 3)) == pytest.approx(0.0)

    def test_current_player_with_open_run_returns_positive(self):
        # X has (1,0),(1,1) — middle-row run, both ends open (2 open ends). k=3, n=4. current=X.
        # X run: m=2, open_ends=2 → weight=2*4^1=8. O at corners: only single pieces.
        # X score=14, O score=6 → positive result.
        state = state_with_moves([(1, 0), (0, 3), (1, 1), (3, 0)], n=4, k=3)
        h = TaxonomyHeuristic()
        assert h.evaluate(state) > 0.0

    def test_symmetric_position_returns_zero(self):
        # Mirror: X at (0,0), O at (3,3) — both single pieces, same weight structure.
        # n=4, k=3. current=O after 2 moves. O's (3,3) mirrors X's (0,0).
        # Both have equivalent open runs in all directions → score=0.
        state = state_with_moves([(0, 0), (3, 3)], n=4, k=3)
        h = TaxonomyHeuristic()
        assert h.evaluate(state) == pytest.approx(0.0)

    def test_result_in_bounds(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1)], n=5, k=3)
        h = TaxonomyHeuristic()
        score = h.evaluate(state)
        assert -1.0 <= score <= 1.0

    def test_opponent_ahead_returns_negative(self):
        # O has (1,0),(1,1) — middle-row run, both ends open. X at corners. k=3, n=4. current=X.
        # O score=14, X score=6 → negative result from X's perspective.
        state = state_with_moves([(0, 3), (1, 0), (3, 0), (1, 1)], n=4, k=3)
        h = TaxonomyHeuristic()
        assert h.evaluate(state) < 0.0


from src.heuristics.fork_heuristic import ForkHeuristic


class TestForkHeuristic:
    def test_empty_board_returns_zero(self):
        # No k-1 sequences exist yet → no forks.
        h = ForkHeuristic()
        assert h.evaluate(fresh_state(3, 3)) == pytest.approx(0.0)

    def test_no_sequences_returns_zero(self):
        # Single scattered pieces do not form k-1=2 sequences.
        # n=5 k=3. X=(0,0), O=(4,4). No direction from any empty cell creates
        # length-2 runs in 2 directions simultaneously.
        state = state_with_moves([(0, 0), (4, 4)], n=5, k=3)
        h = ForkHeuristic()
        assert h.evaluate(state) == pytest.approx(0.0)

    def test_current_player_has_fork_returns_positive(self):
        # n=5, k=3. X at (0,0),(0,2). Playing at (1,1) would create:
        #   diagonal (0,0)-(1,1): length 2, forward open → THREAT
        #   anti-diagonal (0,2)-(1,1): length 2, forward open → THREAT
        # → fork at (1,1) for X only (O pieces at (4,0),(4,4) cannot form a fork).
        # moves: X=(0,0), O=(4,0), X=(0,2), O=(4,4). current=X.
        state = state_with_moves([(0, 0), (4, 0), (0, 2), (4, 4)], n=5, k=3)
        h = ForkHeuristic()
        assert h.evaluate(state) > 0.0

    def test_opponent_has_fork_returns_negative(self):
        # Mirror: O at (0,0),(0,2), X is scattered. current=X.
        # moves: X=(4,0), O=(0,0), X=(4,4), O=(0,2). current=X.
        state = state_with_moves([(4, 0), (0, 0), (4, 4), (0, 2)], n=5, k=3)
        h = ForkHeuristic()
        assert h.evaluate(state) < 0.0

    def test_result_in_bounds(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1)], n=5, k=3)
        h = ForkHeuristic()
        score = h.evaluate(state)
        assert -1.0 <= score <= 1.0
