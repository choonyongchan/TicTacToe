"""Tests for src/agents/minimax_rewards_alphabeta_agent.py."""
from __future__ import annotations

import pytest

from src.agents.minimax_rewards_alphabeta_agent import MinimaxRewardsAlphaBetaAgent
from src.core.board import Board
from src.core.state import State
from src.core.types import Player


def fresh_state() -> State:
    return State()


def state_with_moves(moves: list[tuple[int, int]]) -> State:
    s = State()
    for row, col in moves:
        s.apply(row, col)
    return s


# Shared dummy search tree (mirrored from test_minimax_alphabeta_agent.py).
# Board after: X(0,0), O(0,1), X(1,1), O(2,2), X(1,0), O(2,0)
#   X | O | .
#   X | X | .
#   O | . | O
# X to move. Empty cells (in order): (0,2), (1,2), (2,1).
#
# Hand-traced scores with ε = 0.1 (max_depth=9), maximizer=X:
#
#   X(1,2) → X wins row 1 immediately          depth=7  → 1.0 - 0.1×7 =  0.3
#
#   X(2,1) → O to move, empty: (0,2),(1,2)
#               O(0,2) → X(1,2) → X wins row 1  depth=9  → 0.1
#               O(1,2) → X(0,2) → board full/draw          → 0.0
#             O minimises → 0.0
#
#   X(0,2) → O to move, empty: (1,2),(2,1)
#               O(1,2) → X(2,1) → board full/draw          → 0.0
#               O(2,1) → O wins row 2            depth=8  → -1.0 + 0.1×8 = -0.2
#             O minimises → -0.2
#
# Root _maximize picks max(-0.2, 0.3, 0.0) = 0.3 → best move (1,2).
_DUMMY_TREE_MOVES = [(0, 0), (0, 1), (1, 1), (2, 2), (1, 0), (2, 0)]
_EPS = 1.0 / 10  # epsilon for default max_depth=9


class TestInit:
    def test_name(self):
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent.name == "MinimaxRewardsAlphaBetaAgent"

    def test_maximizer_stored(self):
        agent = MinimaxRewardsAlphaBetaAgent(Player.O)
        assert agent.maximizer == Player.O

    def test_default_max_depth(self):
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent._epsilon == pytest.approx(1.0 / 10)

    def test_custom_max_depth(self):
        agent = MinimaxRewardsAlphaBetaAgent(Player.X, max_depth=15)
        assert agent._epsilon == pytest.approx(1.0 / 16)


class TestTerminalScore:
    """_terminal_score returns depth-scaled floats with win>0, loss<0, draw=0."""

    def test_win_at_depth_5(self):
        # X wins: (0,0),(1,0),(0,1),(1,1),(0,2) — 5 moves played
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        eps = agent._epsilon
        assert agent._terminal_score(state) == pytest.approx(1.0 - eps * 5)

    def test_loss_at_depth_6(self):
        # O wins: (0,1),(0,0),(0,2),(1,0),(2,1),(2,0) — 6 moves played
        state = state_with_moves([(0, 1), (0, 0), (0, 2), (1, 0), (2, 1), (2, 0)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        eps = agent._epsilon
        assert agent._terminal_score(state) == pytest.approx(-1.0 + eps * 6)

    def test_draw_returns_0(self):
        state = state_with_moves([
            (0, 0), (0, 1),
            (0, 2), (1, 0),
            (1, 1), (2, 0),
            (1, 2), (2, 2),
            (2, 1),
        ])
        assert state.is_terminal()
        assert state.winner() is None
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent._terminal_score(state) == 0.0

    def test_win_score_is_positive(self):
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent._terminal_score(state) > 0

    def test_loss_score_is_negative(self):
        state = state_with_moves([(0, 1), (0, 0), (0, 2), (1, 0), (2, 1), (2, 0)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent._terminal_score(state) < 0

    def test_custom_max_depth_scales_epsilon(self):
        # With max_depth=19, epsilon=1/20=0.05; win at depth 5 → 1.0 - 0.05*5 = 0.75
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X, max_depth=19)
        assert agent._terminal_score(state) == pytest.approx(1.0 - (1.0 / 20) * 5)

    def test_earlier_win_scores_higher_than_later_win(self):
        # Win at depth 5 vs win at depth 7
        state_d5 = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
        # X wins at depth 7: X(0,0),O(2,2),X(0,1),O(2,1),X(1,0),O(1,2),X(2,0)
        state_d7 = state_with_moves([(0, 0), (2, 2), (0, 1), (2, 1), (1, 0), (1, 2), (2, 0)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent._terminal_score(state_d5) > agent._terminal_score(state_d7)

    def test_later_loss_scores_higher_than_earlier_loss(self):
        # O wins at depth 6: existing 6-move sequence
        state_d6 = state_with_moves([(0, 1), (0, 0), (0, 2), (1, 0), (2, 1), (2, 0)])
        # O wins at depth 8: from dummy tree root, X plays (0,2) then O plays (2,1) → O row 2
        # _DUMMY_TREE_MOVES(6) + X(0,2)(7) + O(2,1)(8) → O has (2,0),(2,2),(2,1) = row 2 win
        state_d8 = state_with_moves(_DUMMY_TREE_MOVES + [(0, 2), (2, 1)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent._terminal_score(state_d8) > agent._terminal_score(state_d6)


class TestMinimaxRewardsSmallTree:
    """Hand-traced depth-scaled scores on the shared dummy search tree."""

    def test_winning_branch_scores_depth7(self):
        # X(1,2) → X wins row 1, depth 7 → 1.0 - ε×7
        state = state_with_moves(_DUMMY_TREE_MOVES + [(1, 2)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert state.is_terminal()
        assert agent._minimax(state, float("-inf"), float("inf")) == pytest.approx(1.0 - _EPS * 7)

    def test_losing_branch_scores_neg02(self):
        # X(0,2) → O minimises to O(2,1) win at depth 8 → -1.0 + ε×8 = -0.2
        state = state_with_moves(_DUMMY_TREE_MOVES)
        state.apply(0, 2)
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent._minimax(state, float("-inf"), float("inf")) == pytest.approx(-1.0 + _EPS * 8)

    def test_draw_branch_scores_0(self):
        # X(2,1) → O minimises to draw → 0.0
        state = state_with_moves(_DUMMY_TREE_MOVES)
        state.apply(2, 1)
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent._minimax(state, float("-inf"), float("inf")) == pytest.approx(0.0)

    def test_maximize_picks_best_child(self):
        # max(-0.2, 0.3, 0.0) = 0.3
        state = state_with_moves(_DUMMY_TREE_MOVES)
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent._maximize(state, float("-inf"), float("inf")) == pytest.approx(1.0 - _EPS * 7)

    def test_minimize_at_draw_branch(self):
        # After X(2,1): O sees O(0,2)→0.1, O(1,2)→0.0 → minimise = 0.0
        state = state_with_moves(_DUMMY_TREE_MOVES)
        state.apply(2, 1)
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent._minimize(state, float("-inf"), float("inf")) == pytest.approx(0.0)

    def test_minimize_at_losing_branch(self):
        # After X(0,2): O sees O(1,2)→0.0, O(2,1)→-0.2 → minimise = -0.2
        state = state_with_moves(_DUMMY_TREE_MOVES)
        state.apply(0, 2)
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent._minimize(state, float("-inf"), float("inf")) == pytest.approx(-1.0 + _EPS * 8)

    def test_depth_scaled_win_beats_draw(self):
        # 0.3 > 0.0 — confirms the agent strictly prefers the winning path
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        win_score = agent._minimax(
            state_with_moves(_DUMMY_TREE_MOVES + [(1, 2)]), float("-inf"), float("inf")
        )
        draw_state = state_with_moves(_DUMMY_TREE_MOVES)
        draw_state.apply(2, 1)
        draw_score = agent._minimax(draw_state, float("-inf"), float("inf"))
        assert win_score > draw_score


class TestActWinningMove:
    """act() must prefer an immediate win over any other move."""

    def test_wins_in_one_row(self):
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent.act(state) == (0, 2)

    def test_wins_in_one_column(self):
        state = state_with_moves([(0, 0), (0, 2), (1, 0), (1, 2)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent.act(state) == (2, 0)

    def test_picks_winning_move_on_dummy_tree(self):
        state = state_with_moves(_DUMMY_TREE_MOVES)
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent.act(state) == (1, 2)

    def test_does_not_play_losing_move(self):
        state = state_with_moves(_DUMMY_TREE_MOVES)
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent.act(state) != (0, 2)


class TestActBlockingMove:
    """act() must block the opponent's immediate win when no own win exists."""

    def test_blocks_opponent_row_win(self):
        # X:(0,0),(2,0)  O:(1,0),(1,1) — O threatens (1,2); X has no immediate win
        # Board: X|.|. / O|O|. / X|.|.
        state = state_with_moves([(0, 0), (1, 0), (2, 0), (1, 1)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent.act(state) == (1, 2)

    def test_blocks_opponent_diagonal_win(self):
        state = state_with_moves([(0, 2), (0, 0), (1, 2), (1, 1)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent.act(state) == (2, 2)


class TestPrefersFasterWin:
    """Agent prefers a shallower win over an equivalent but deeper one."""

    def test_prefers_immediate_win_over_delayed(self):
        # X can win immediately at (0,2), or win later via another path.
        # With depth-scaled rewards, immediate win must be chosen.
        state = state_with_moves([(0, 0), (1, 0), (0, 1), (1, 1)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        move = agent.act(state)
        assert move == (0, 2), f"Expected immediate win (0,2), got {move}"


class TestTwoAgentsDraw:
    """Two MinimaxRewardsAlphaBetaAgents playing against each other must always draw."""

    def _play_game(
        self,
        agent_x: MinimaxRewardsAlphaBetaAgent,
        agent_o: MinimaxRewardsAlphaBetaAgent,
    ) -> Player | None:
        state = fresh_state()
        agents = {Player.X: agent_x, Player.O: agent_o}
        while not state.is_terminal():
            agent = agents[state.current_player]
            row, col = agent.act(state)
            state.apply(row, col)
        return state.winner()

    def test_two_agents_draw(self):
        agent_x = MinimaxRewardsAlphaBetaAgent(Player.X)
        agent_o = MinimaxRewardsAlphaBetaAgent(Player.O)
        result = self._play_game(agent_x, agent_o)
        assert result is None, f"Expected draw, got winner={result}"


class TestActValidMove:
    """act() always returns a legal move regardless of board state."""

    def test_returns_tuple(self):
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        move = agent.act(fresh_state())
        assert isinstance(move, tuple) and len(move) == 2

    def test_returns_in_bounds(self):
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        row, col = agent.act(fresh_state())
        assert 0 <= row <= 2 and 0 <= col <= 2

    def test_returns_empty_cell(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0)])
        agent = MinimaxRewardsAlphaBetaAgent(Player.O)
        row, col = agent.act(state)
        assert Board.is_empty(state.board, row, col)

    def test_only_one_cell_left(self):
        state = state_with_moves([
            (1, 0), (0, 0),
            (0, 1), (1, 1),
            (0, 2), (1, 2),
            (2, 0), (2, 1),
        ])
        assert not state.is_terminal()
        agent = MinimaxRewardsAlphaBetaAgent(Player.X)
        assert agent.act(state) == (2, 2)
