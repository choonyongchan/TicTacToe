import pytest

from src.agents.bns_agent import BNSAgent
from src.agents.negamax_agent import NegamaxAgent
from src.core.types import NEGATIVE_INFINITY
from src.tests.test_helper import (
    PUZZLE_3X3,
    PUZZLE_4X4,
    PUZZLE_5X5,
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
