"""Tests for src/agents/negamax_agent.py."""

from __future__ import annotations

import pytest

from src.agents.minimax_rewards_alphabeta_agent import MinimaxRewardsAlphaBetaAgent
from src.agents.negamax_agent import NegamaxAgent
from src.core.types import Player
from src.tests.test_helper import (
    PUZZLE_3X3,
    PUZZLE_4X4,
    PUZZLE_5X5,
    fresh_state,
    state_with_moves,
)

_DUMMY_TREE_MOVES = PUZZLE_3X3.moves
_EPS = 1.0 / 10  # epsilon for default max_depth=9


class TestInit:
    def test_name(self):
        assert NegamaxAgent(max_depth=9).name == "NegamaxAgent"

    def test_default_epsilon(self):
        assert NegamaxAgent(max_depth=9)._epsilon == pytest.approx(1.0 / 10)

    def test_custom_max_depth(self):
        assert NegamaxAgent(max_depth=15)._epsilon == pytest.approx(1.0 / 16)


class TestTerminalScore:
    """_terminal_score is player-agnostic: positive for any win, 0 for draw."""

    def test_win_at_depth_5(self):
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
        agent = NegamaxAgent(max_depth=9)
        assert agent._terminal_score(state) == pytest.approx(1.0 - _EPS * 5)

    def test_any_win_at_depth_6(self):
        # O wins: _terminal_score is still positive (player-agnostic win reward)
        state = state_with_moves([(0, 1), (0, 0), (0, 2), (1, 0), (2, 1), (2, 0)])
        agent = NegamaxAgent(max_depth=9)
        assert agent._terminal_score(state) == pytest.approx(1.0 - _EPS * 6)

    def test_draw_returns_zero(self):
        state = state_with_moves(
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (2, 0),
                (1, 2),
                (2, 2),
                (2, 1),
            ]
        )
        assert state.is_terminal() and state.winner() is None
        assert NegamaxAgent(max_depth=9)._terminal_score(state) == 0.0

    def test_any_win_score_is_positive(self):
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
        assert NegamaxAgent(max_depth=9)._terminal_score(state) > 0

    def test_opponent_win_score_is_also_positive(self):
        state = state_with_moves([(0, 1), (0, 0), (0, 2), (1, 0), (2, 1), (2, 0)])
        assert NegamaxAgent(max_depth=9)._terminal_score(state) > 0

    def test_earlier_win_scores_higher(self):
        state_d5 = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
        state_d7 = state_with_moves(
            [(0, 0), (2, 2), (0, 1), (2, 1), (1, 0), (1, 2), (2, 0)]
        )
        agent = NegamaxAgent(max_depth=9)
        assert agent._terminal_score(state_d5) > agent._terminal_score(state_d7)

    def test_earlier_win_scores_higher_regardless_of_winner(self):
        # Depth-scaling applies identically regardless of which player won
        state_d6 = state_with_moves([(0, 1), (0, 0), (0, 2), (1, 0), (2, 1), (2, 0)])
        state_d8 = state_with_moves(_DUMMY_TREE_MOVES + ((0, 2), (2, 1)))
        agent = NegamaxAgent(max_depth=9)
        assert agent._terminal_score(state_d6) > agent._terminal_score(state_d8)

    def test_custom_max_depth_scales_epsilon(self):
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
        agent = NegamaxAgent(max_depth=19)
        assert agent._terminal_score(state) == pytest.approx(1.0 - (1.0 / 20) * 5)


class TestNegamaxSmallTree:
    """Hand-traced depth-scaled scores on the dummy search tree.

    _negamax returns the score from the CURRENT PLAYER's perspective.
    At a terminal node the current player is the loser, so the score is negative.
    """

    def test_terminal_winner_scores_negative(self):
        # X(1,2) → terminal, X wins row 1, depth 7.
        # Current player is O (the loser); score = -(1.0 - ε×7)
        state = state_with_moves(_DUMMY_TREE_MOVES + ((1, 2),))
        agent = NegamaxAgent(max_depth=9)
        assert state.is_terminal()
        assert agent._negamax(state, float("-inf"), float("inf")) == pytest.approx(
            -(1.0 - _EPS * 7)
        )

    def test_current_player_winning_by_force_scores_positive(self):
        # X(0,2) → O to move; O forces O(2,1) win at depth 8.
        # Current player is O (the winner-to-be); score = +(1.0 - ε×8)
        state = state_with_moves(_DUMMY_TREE_MOVES)
        state.apply(0, 2)
        agent = NegamaxAgent(max_depth=9)
        assert agent._negamax(state, float("-inf"), float("inf")) == pytest.approx(
            1.0 - _EPS * 8
        )

    def test_draw_branch_scores_0(self):
        # X(2,1) → forced draw
        state = state_with_moves(_DUMMY_TREE_MOVES)
        state.apply(2, 1)
        agent = NegamaxAgent(max_depth=9)
        assert agent._negamax(state, float("-inf"), float("inf")) == pytest.approx(0.0)

    def test_root_score_is_winning_path(self):
        # X to move; best outcome is X wins at depth 7
        state = state_with_moves(_DUMMY_TREE_MOVES)
        agent = NegamaxAgent(max_depth=9)
        assert agent._negamax(state, float("-inf"), float("inf")) == pytest.approx(
            1.0 - _EPS * 7
        )

    def test_winning_path_beats_draw(self):
        agent = NegamaxAgent(max_depth=9)
        win_score = agent._negamax(
            state_with_moves(_DUMMY_TREE_MOVES), float("-inf"), float("inf")
        )
        draw_state = state_with_moves(_DUMMY_TREE_MOVES)
        draw_state.apply(2, 1)
        draw_score = agent._negamax(draw_state, float("-inf"), float("inf"))
        assert win_score > draw_score


class TestActWinningMove:
    """act() must prefer an immediate win."""

    def test_wins_in_one_row(self):
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        assert NegamaxAgent(max_depth=9).act(state) == (0, 2)

    def test_wins_in_one_column(self):
        state = state_with_moves([(0, 0), (0, 2), (1, 0), (1, 2)])
        assert NegamaxAgent(max_depth=9).act(state) == (2, 0)

    def test_picks_winning_move_on_dummy_tree(self):
        state = state_with_moves(_DUMMY_TREE_MOVES)
        assert NegamaxAgent(max_depth=9).act(state) == (1, 2)

    def test_does_not_play_losing_move(self):
        state = state_with_moves(_DUMMY_TREE_MOVES)
        assert NegamaxAgent(max_depth=9).act(state) != (0, 2)


class TestActBlockingMove:
    """act() must block the opponent's immediate win when no own win exists."""

    def test_blocks_opponent_row_win(self):
        # X:(0,0),(2,0)  O:(1,0),(1,1) — O threatens (1,2)
        state = state_with_moves([(0, 0), (1, 0), (2, 0), (1, 1)])
        assert NegamaxAgent(max_depth=9).act(state) == (1, 2)

    def test_blocks_opponent_diagonal_win(self):
        state = state_with_moves([(0, 2), (0, 0), (1, 2), (1, 1)])
        assert NegamaxAgent(max_depth=9).act(state) == (2, 2)


class TestPrefersFasterWin:
    """act() prefers an immediate win over a delayed one (depth-scaled rewards)."""

    def test_prefers_immediate_win(self):
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        assert NegamaxAgent(max_depth=9).act(state) == (0, 2)


class TestActValidMove:
    """act() always returns a legal move."""

    def test_returns_tuple(self):
        move = NegamaxAgent(max_depth=9).act(fresh_state())
        assert isinstance(move, tuple) and len(move) == 2

    def test_returns_in_bounds(self):
        row, col = NegamaxAgent(max_depth=9).act(fresh_state())
        assert 0 <= row <= 2 and 0 <= col <= 2

    def test_returns_empty_cell(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0)])
        row, col = NegamaxAgent(max_depth=9).act(state)
        assert state.board.is_empty(row, col)

    def test_only_one_cell_left(self):
        state = state_with_moves(
            [
                (1, 0),
                (0, 0),
                (0, 1),
                (1, 1),
                (0, 2),
                (1, 2),
                (2, 0),
                (2, 1),
            ]
        )
        assert not state.is_terminal()
        assert NegamaxAgent(max_depth=9).act(state) == (2, 2)


class TestActLargerBoards:
    """act() picks the correct move on 4×4 and 5×5 puzzle positions."""

    def test_4x4_picks_best_move(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        agent = NegamaxAgent(max_depth=PUZZLE_4X4.n**2)
        assert agent.act(state) == PUZZLE_4X4.best_move

    def test_5x5_picks_best_move(self):
        state = state_with_moves(PUZZLE_5X5.moves, PUZZLE_5X5.n, PUZZLE_5X5.k)
        agent = NegamaxAgent(max_depth=PUZZLE_5X5.n**2)
        assert agent.act(state) == PUZZLE_5X5.best_move


class TestAgreesWithMinimaxRewards:
    """NegamaxAgent produces identical act() results to MinimaxRewardsAlphaBetaAgent."""

    def _minimax(
        self, player: Player, max_depth: int = 9
    ) -> MinimaxRewardsAlphaBetaAgent:
        return MinimaxRewardsAlphaBetaAgent(player, max_depth=max_depth)

    def test_agrees_on_dummy_tree(self):
        state = state_with_moves(_DUMMY_TREE_MOVES)
        assert NegamaxAgent(max_depth=9).act(state) == self._minimax(Player.X).act(
            state
        )

    def test_agrees_on_win_in_one_row(self):
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        assert NegamaxAgent(max_depth=9).act(state) == self._minimax(Player.X).act(
            state
        )

    def test_agrees_on_block(self):
        state = state_with_moves([(0, 0), (1, 0), (2, 0), (1, 1)])
        assert NegamaxAgent(max_depth=9).act(state) == self._minimax(Player.X).act(
            state
        )

    def test_agrees_on_4x4_puzzle(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        md = PUZZLE_4X4.n**2
        assert NegamaxAgent(max_depth=md).act(state) == self._minimax(Player.X, md).act(
            state
        )

    def test_agrees_on_5x5_puzzle(self):
        state = state_with_moves(PUZZLE_5X5.moves, PUZZLE_5X5.n, PUZZLE_5X5.k)
        md = PUZZLE_5X5.n**2
        assert NegamaxAgent(max_depth=md).act(state) == self._minimax(Player.X, md).act(
            state
        )

    def test_two_negamax_agents_draw(self):
        state = fresh_state()
        agent_x = NegamaxAgent(max_depth=9)
        agent_o = NegamaxAgent(max_depth=9)
        while not state.is_terminal():
            agent = agent_x if state.current_player == Player.X else agent_o
            state.apply(*agent.act(state))
        assert state.winner() is None
