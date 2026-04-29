import pytest

from src.agents.mtdf_agent import MTDfAgent
from src.agents.negamax_agent import NegamaxAgent
from src.core.transposition_table import TranspositionTable
from src.core.types import NEGATIVE_INFINITY
from src.tests.test_helper import (
    PUZZLE_3X3,
    PUZZLE_4X4,
    PUZZLE_4X4_BLOCK,
    PUZZLE_5X5,
    fresh_state,
    state_with_moves,
)

_EPS = 1.0 / 10  # max_depth=9 → epsilon=0.1


class TestInit:
    def test_name(self):
        assert MTDfAgent(9).name == "MTDfAgent"

    def test_epsilon_depth_9(self):
        assert MTDfAgent(9)._epsilon == pytest.approx(0.1)

    def test_epsilon_depth_4(self):
        assert MTDfAgent(4)._epsilon == pytest.approx(1.0 / 5)


class TestTerminalScore:
    def test_draw_returns_zero(self):
        state = fresh_state()
        agent = MTDfAgent(9)
        assert agent._terminal_score(state) == 0.0

    def test_win_returns_positive(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        agent = MTDfAgent(9)
        assert agent._terminal_score(state) > 0.0

    def test_win_score_is_depth_discounted(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        agent = MTDfAgent(9)
        assert agent._terminal_score(state) == pytest.approx(1.0 - _EPS * 5)

    def test_earlier_win_scores_higher(self):
        state_d5 = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        state_d7 = state_with_moves(
            PUZZLE_3X3.moves + ((1, 2),), PUZZLE_3X3.n, PUZZLE_3X3.k
        )
        agent = MTDfAgent(9)
        assert agent._terminal_score(state_d5) > agent._terminal_score(state_d7)


class TestMtdf:
    """MTD(f) outer loop must converge to the exact minimax value.

    PUZZLE_3X3 true value from X's perspective = 1.0 - 0.1*7 = 0.3
    (X wins at (1,2), bringing history length to 7)
    """

    def _puzzle_state(self):
        return state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)

    def test_converges_from_zero_guess(self):
        state = self._puzzle_state()
        agent = MTDfAgent(9)
        tt = TranspositionTable()
        score = agent._mtdf(state, 0.0, tt)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_converges_from_high_guess(self):
        state = self._puzzle_state()
        agent = MTDfAgent(9)
        tt = TranspositionTable()
        score = agent._mtdf(state, 1.0, tt)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_converges_from_low_guess(self):
        state = self._puzzle_state()
        agent = MTDfAgent(9)
        tt = TranspositionTable()
        score = agent._mtdf(state, -1.0, tt)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_positive_value_on_winning_position(self):
        state = self._puzzle_state()
        agent = MTDfAgent(9)
        tt = TranspositionTable()
        score = agent._mtdf(state, 0.0, tt)
        assert score > 0

    def test_tt_populated_after_search(self):
        state = self._puzzle_state()
        agent = MTDfAgent(9)
        tt = TranspositionTable()
        agent._mtdf(state, 0.0, tt)
        assert len(tt) > 0


class TestNegamaxTt:
    """Underlying _negamax_tt must respect TT bounds and alpha-beta window."""

    def _puzzle_state(self):
        return state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)

    def test_full_window_returns_exact_value(self):
        state = self._puzzle_state()
        agent = MTDfAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, tt)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_fail_low_upper_bound(self):
        # alpha=0.5 > true value=0.3 → fail-low
        state = self._puzzle_state()
        agent = MTDfAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, 0.5, 1.0, tt)
        assert score <= 0.5

    def test_fail_high_lower_bound(self):
        # beta=0.1 < true value=0.3 → fail-high (beta cutoff)
        state = self._puzzle_state()
        agent = MTDfAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, NEGATIVE_INFINITY, 0.1, tt)
        assert score >= 0.1

    def test_tight_window_spanning_value_returns_exact(self):
        # Window [0.25, 0.35] contains the true value 0.3 → exact result
        state = self._puzzle_state()
        agent = MTDfAgent(9)
        tt = TranspositionTable()
        score = agent._negamax_tt(state, 0.25, 0.35, tt)
        assert score == pytest.approx(1.0 - _EPS * 7)

    def test_tt_hit_avoids_re_search(self):
        # Pre-seed TT with exact bound; second call must return same value
        state = self._puzzle_state()
        agent = MTDfAgent(9)
        tt = TranspositionTable()
        first = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, tt)
        second = agent._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, tt)
        assert first == pytest.approx(second)


class TestActWinningMove:
    def test_takes_immediate_win_row(self):
        # X X .  → X plays (0, 2) to complete row
        # O O .
        # . . .
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        agent = MTDfAgent(9)
        assert agent.act(state) == (0, 2)

    def test_takes_immediate_win_column(self):
        # X O .
        # X O .
        # . . .  → X plays (2, 0) to complete column 0
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1)])
        agent = MTDfAgent(9)
        assert agent.act(state) == (2, 0)


class TestActBlockingMove:
    def test_blocks_opponent_win(self):
        # O O .  → X must block at (0, 2)
        # X . .
        # X . .
        state = state_with_moves([(1, 0), (0, 0), (2, 0), (0, 1)])
        agent = MTDfAgent(9)
        assert agent.act(state) == (0, 2)

    def test_4x4_o_blocks_x_column(self):
        state = state_with_moves(
            PUZZLE_4X4_BLOCK.moves, PUZZLE_4X4_BLOCK.n, PUZZLE_4X4_BLOCK.k
        )
        assert MTDfAgent(PUZZLE_4X4_BLOCK.n ** 2).act(state) == PUZZLE_4X4_BLOCK.best_move


class TestActSmallTree:
    def test_picks_best_move_puzzle_3x3(self):
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        agent = MTDfAgent(9)
        assert agent.act(state) == PUZZLE_3X3.best_move


class TestActLargerBoards:
    def test_4x4_picks_best_move(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        agent = MTDfAgent(16)
        assert agent.act(state) == PUZZLE_4X4.best_move

    def test_5x5_picks_best_move(self):
        state = state_with_moves(PUZZLE_5X5.moves, PUZZLE_5X5.n, PUZZLE_5X5.k)
        agent = MTDfAgent(25)
        assert agent.act(state) == PUZZLE_5X5.best_move


class TestActValidMove:
    def test_returns_in_bounds_move(self):
        state = fresh_state()
        agent = MTDfAgent(9)
        row, col = agent.act(state)
        assert 0 <= row < state.board.n
        assert 0 <= col < state.board.n

    def test_returns_empty_cell(self):
        state = fresh_state()
        agent = MTDfAgent(9)
        row, col = agent.act(state)
        assert state.board.is_empty(row, col)


class TestAgreesWithNegamax:
    """MTDfAgent must choose identical moves to NegamaxAgent (same depth)."""

    def test_agrees_on_puzzle_3x3(self):
        state = state_with_moves(PUZZLE_3X3.moves, PUZZLE_3X3.n, PUZZLE_3X3.k)
        assert MTDfAgent(9).act(state) == NegamaxAgent(9).act(state)

    def test_agrees_on_empty_board(self):
        state = fresh_state()
        assert MTDfAgent(9).act(state) == NegamaxAgent(9).act(state)

    def test_agrees_on_puzzle_4x4(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        assert MTDfAgent(16).act(state) == NegamaxAgent(16).act(state)
