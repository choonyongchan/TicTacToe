import pytest

from src.agents.negascout_agent import NegascoutAgent
from src.agents.negamax_agent import NegamaxAgent
from src.core.types import NEGATIVE_INFINITY
from src.tests.test_helper import (
    PUZZLE_3X3,
    PUZZLE_4X4,
    PUZZLE_5X5,
    fresh_state,
    state_with_moves,
)

_EPS = 1.0 / 10  # max_depth=9 → epsilon=0.1


class TestInit:
    def test_name(self):
        assert NegascoutAgent(9).name == "NegascoutAgent"

    def test_epsilon_depth_9(self):
        assert NegascoutAgent(9)._epsilon == pytest.approx(0.1)

    def test_epsilon_depth_4(self):
        assert NegascoutAgent(4)._epsilon == pytest.approx(1.0 / 5)


class TestTerminalScore:
    def test_draw_returns_zero(self):
        state = fresh_state()
        agent = NegascoutAgent(9)
        assert agent._terminal_score(state) == 0.0

    def test_win_returns_positive(self):
        # X wins on move 5 (history length = 5)
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        agent = NegascoutAgent(9)
        assert agent._terminal_score(state) > 0.0

    def test_win_score_is_depth_discounted(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        agent = NegascoutAgent(9)
        assert agent._terminal_score(state) == pytest.approx(1.0 - _EPS * 5)

    def test_earlier_win_scores_higher(self):
        state_d5 = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        state_d7 = state_with_moves(
            PUZZLE_3X3.moves + ((1, 2),), PUZZLE_3X3.n, PUZZLE_3X3.k
        )
        agent = NegascoutAgent(9)
        assert agent._terminal_score(state_d5) > agent._terminal_score(state_d7)


class TestBoundedSearch:
    """Test alpha-beta window semantics: exact / fail-low (upper bound) / fail-high (lower bound).

    PUZZLE_3X3 true value from X's perspective = 1.0 - 0.1*7 = 0.3
    (X wins at (1,2), bringing history length to 7)
    """

    def _puzzle_state(self):
        return state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)

    def test_full_window_returns_exact_value(self):
        # Full window [-inf, +inf] must return the exact minimax score.
        state = self._puzzle_state()
        agent = NegascoutAgent(9)
        score = agent._negascout(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_fail_low_upper_bound(self):
        # alpha=0.5 > true value=0.3 → fail-low.
        # Returned value ≤ alpha proves the true score is at most alpha.
        state = self._puzzle_state()
        agent = NegascoutAgent(9)
        score = agent._negascout(state, 0.5, 1.0)
        assert score <= 0.5

    def test_fail_high_lower_bound(self):
        # beta=0.1 < true value=0.3 → fail-high (beta cutoff).
        # Returned value ≥ beta proves the true score is at least beta.
        state = self._puzzle_state()
        agent = NegascoutAgent(9)
        score = agent._negascout(state, NEGATIVE_INFINITY, 0.1)
        assert score >= 0.1

    def test_tight_window_spanning_value_returns_exact(self):
        # Window [0.25, 0.35] contains the true value 0.3 → exact result.
        state = self._puzzle_state()
        agent = NegascoutAgent(9)
        score = agent._negascout(state, 0.25, 0.35)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_null_window_at_true_value_is_lower_bound(self):
        # Null window [0.3, 0.3+ε]: true value=0.3 ≥ beta → fail-high → score ≥ beta.
        state = self._puzzle_state()
        agent = NegascoutAgent(9)
        true_val = 1.0 - _EPS * 7
        score = agent._negascout(state, true_val, true_val + 1e-9)
        assert score >= true_val

    def test_null_window_below_true_value_is_lower_bound(self):
        # Null window [0.1, 0.1+ε]: true value=0.3 > beta → fail-high → score ≥ beta.
        state = self._puzzle_state()
        agent = NegascoutAgent(9)
        beta = 0.1 + 1e-9
        score = agent._negascout(state, 0.1, beta)
        assert score >= beta

    def test_null_window_above_true_value_is_upper_bound(self):
        # Null window [0.5, 0.5+ε]: true value=0.3 < alpha=0.5 → fail-low → score ≤ alpha.
        state = self._puzzle_state()
        agent = NegascoutAgent(9)
        score = agent._negascout(state, 0.5, 0.5 + 1e-9)
        assert score <= 0.5


class TestNegaScoutSmallTree:
    def test_picks_best_move_puzzle_3x3(self):
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = NegascoutAgent(9)
        assert agent.act(state) == PUZZLE_3X3.best_move

    def test_best_move_value_is_positive(self):
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = NegascoutAgent(9)
        score = agent._negascout(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY)
        assert score > 0


class TestActWinningMove:
    def test_takes_immediate_win_row(self):
        # X X .  → X plays (0, 2) to complete row
        # O O .
        # . . .
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        agent = NegascoutAgent(9)
        assert agent.act(state) == (0, 2)

    def test_takes_immediate_win_column(self):
        # X O .
        # X O .
        # . . .  → X plays (2, 0) to complete column 0
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1)])
        agent = NegascoutAgent(9)
        assert agent.act(state) == (2, 0)


class TestActBlockingMove:
    def test_blocks_opponent_win(self):
        # O O .  → X must block at (0, 2)
        # X . .
        # X . .
        state = state_with_moves([(1, 0), (0, 0), (2, 0), (0, 1)])
        agent = NegascoutAgent(9)
        assert agent.act(state) == (0, 2)


class TestActLargerBoards:
    def test_4x4_picks_best_move(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        agent = NegascoutAgent(16)
        assert agent.act(state) == PUZZLE_4X4.best_move

    def test_5x5_picks_best_move(self):
        state = state_with_moves(PUZZLE_5X5.moves, PUZZLE_5X5.n, PUZZLE_5X5.k)
        agent = NegascoutAgent(25)
        assert agent.act(state) == PUZZLE_5X5.best_move


class TestActValidMove:
    def test_returns_in_bounds_move(self):
        state = fresh_state()
        agent = NegascoutAgent(9)
        row, col = agent.act(state)
        assert 0 <= row < state.board.n
        assert 0 <= col < state.board.n

    def test_returns_empty_cell(self):
        state = fresh_state()
        agent = NegascoutAgent(9)
        row, col = agent.act(state)
        assert state.board.is_empty(row, col)


class TestAgreesWithNegamax:
    """NegaScout must choose identical moves to NegamaxAgent (same depth)."""

    def test_agrees_on_puzzle_3x3(self):
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        assert NegascoutAgent(9).act(state) == NegamaxAgent(9).act(state)

    def test_agrees_on_empty_board(self):
        state = fresh_state()
        assert NegascoutAgent(9).act(state) == NegamaxAgent(9).act(state)

    def test_agrees_on_puzzle_4x4(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        assert NegascoutAgent(16).act(state) == NegamaxAgent(16).act(state)
