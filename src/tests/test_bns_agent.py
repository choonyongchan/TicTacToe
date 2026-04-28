import pytest

from src.agents.bns_agent import BNSAgent
from src.core.types import NEGATIVE_INFINITY
from src.tests.test_helper import (
    PUZZLE_3X3,
    fresh_state,
    state_with_moves,
)

_EPS = 1.0 / 10  # max_depth=9


class TestInit:
    def test_name(self):
        assert BNSAgent(9).name == "BNSAgent"

    def test_epsilon_depth_9(self):
        assert BNSAgent(9)._epsilon == pytest.approx(0.1)

    def test_epsilon_depth_4(self):
        assert BNSAgent(4)._epsilon == pytest.approx(1.0 / 5)


class TestTerminalScore:
    def test_draw_returns_zero(self):
        state = fresh_state()
        assert BNSAgent(9)._terminal_score(state) == 0.0

    def test_win_returns_positive(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        assert BNSAgent(9)._terminal_score(state) > 0.0

    def test_win_score_is_depth_discounted(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        assert BNSAgent(9)._terminal_score(state) == pytest.approx(1.0 - _EPS * 5)

    def test_earlier_win_scores_higher(self):
        state_d5 = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        state_d7 = state_with_moves(
            PUZZLE_3X3.moves + ((1, 2),), PUZZLE_3X3.n, PUZZLE_3X3.k
        )
        assert BNSAgent(9)._terminal_score(state_d5) > BNSAgent(9)._terminal_score(state_d7)


class TestAlphabeta:
    def _puzzle_state(self):
        return state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)

    def test_full_window_returns_exact_value(self):
        # PUZZLE_3X3 true value: 1.0 - 0.1*7 = 0.3 (X wins at depth 7)
        state = self._puzzle_state()
        score = BNSAgent(9)._alphabeta(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_fail_low_upper_bound(self):
        # Window above true value → fail-low, returns <= alpha
        state = self._puzzle_state()
        score = BNSAgent(9)._alphabeta(state, 0.5, 1.0)
        assert score <= 0.5

    def test_fail_high_lower_bound(self):
        # Window below true value → fail-high (beta cutoff), returns >= beta
        state = self._puzzle_state()
        score = BNSAgent(9)._alphabeta(state, NEGATIVE_INFINITY, 0.1)
        assert score >= 0.1


class TestBns:
    def test_returns_best_move_puzzle_3x3(self):
        # PUZZLE_3X3: X to move, best move is (1,2) — wins row 1
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        assert BNSAgent(9)._bns(state, -1.0, 1.0) == (1, 2)

    def test_beta_lowered_when_no_child_passes(self):
        # Window [-1, -0.5] is entirely below the true value (+0.3).
        # The winning child passes on the first probe (test=-0.75, val≈0.3 >= -0.75),
        # so alpha keeps rising until beta-alpha < 2*epsilon, then the passing move is returned.
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = BNSAgent(9)
        result = agent._bns(state, -1.0, -0.5)
        assert result is not None
        assert state.board.is_empty(*result)


class TestActWinningMove:
    def test_takes_immediate_win_row(self):
        # X X .
        # O O .
        # . . .  → X plays (0,2) to complete row 0
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        assert BNSAgent(9).act(state) == (0, 2)

    def test_takes_immediate_win_column(self):
        # X O .
        # X O .
        # . . .  → X plays (2,0) to complete col 0
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1)])
        assert BNSAgent(9).act(state) == (2, 0)


class TestActBlockingMove:
    def test_blocks_opponent_win(self):
        # O O .  → X must block at (0,2)
        # X . .
        # X . .
        state = state_with_moves([(1, 0), (0, 0), (2, 0), (0, 1)])
        assert BNSAgent(9).act(state) == (0, 2)


class TestActSmallTree:
    def test_picks_best_move_puzzle_3x3(self):
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        assert BNSAgent(9).act(state) == PUZZLE_3X3.best_move


class TestActValidMove:
    def test_returns_in_bounds_move(self):
        state = fresh_state()
        row, col = BNSAgent(9).act(state)
        assert 0 <= row < state.board.n
        assert 0 <= col < state.board.n

    def test_returns_empty_cell(self):
        state = fresh_state()
        row, col = BNSAgent(9).act(state)
        assert state.board.is_empty(row, col)
