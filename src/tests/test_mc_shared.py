"""Tests for src/agents/mc_shared.py."""

from __future__ import annotations

import math

from src.agents.mc_shared import MCNode, MonteCarloAgent
from src.core.types import Player
from src.tests.test_helper import PUZZLE_3X3, fresh_state, state_with_moves


class _DummyMCAgent(MonteCarloAgent):
    """Minimal concrete MonteCarloAgent for exercising shared helper methods."""

    def act(self, state):  # pragma: no cover - not exercised directly
        return state.board.get_empty_cells()[0]


class TestMCNodeExpansion:
    def test_is_fully_expanded_false_with_untried_moves(self):
        node = MCNode(None, None, Player.X, [(0, 0), (0, 1)])
        assert node.is_fully_expanded() is False

    def test_is_fully_expanded_true_when_empty(self):
        node = MCNode(None, None, Player.X, [])
        assert node.is_fully_expanded() is True


class TestMCNodeQ:
    def test_q_zero_when_unvisited(self):
        node = MCNode(None, None, Player.X, [])
        assert node.q() == 0.0

    def test_q_averages_value_sum_over_visits(self):
        node = MCNode(None, None, Player.X, [])
        node.visits = 4
        node.value_sum = 2.0
        assert node.q() == 0.5


class TestMCNodeUctScore:
    def test_unvisited_child_is_infinite(self):
        parent = MCNode(None, None, Player.X, [])
        parent.visits = 10
        child = MCNode(parent, (0, 0), Player.O, [])
        assert child.uct_score(c=1.0) == math.inf

    def test_visited_child_matches_ucb1_formula(self):
        parent = MCNode(None, None, Player.X, [])
        parent.visits = 9
        child = MCNode(parent, (0, 0), Player.O, [])
        child.visits = 3
        child.value_sum = 1.5
        expected = 0.5 + 2.0 * math.sqrt(math.log(9) / 3)
        assert child.uct_score(c=2.0) == expected


class TestReward:
    def test_reward_win_for_mover(self):
        agent = _DummyMCAgent("Dummy", n_simulations=1)
        state = state_with_moves(PUZZLE_3X3.moves + (PUZZLE_3X3.best_move,))
        assert state.winner() is Player.X
        assert agent._reward(state, Player.X) == 1.0

    def test_reward_loss_for_opponent(self):
        agent = _DummyMCAgent("Dummy", n_simulations=1)
        state = state_with_moves(PUZZLE_3X3.moves + (PUZZLE_3X3.best_move,))
        assert agent._reward(state, Player.O) == -1.0

    def test_reward_draw(self):
        agent = _DummyMCAgent("Dummy", n_simulations=1)
        # Full board, no winner: classic draw layout.
        moves = (
            (0, 0), (0, 1), (0, 2),
            (1, 1), (1, 0), (1, 2),
            (2, 1), (2, 0), (2, 2),
        )
        state = state_with_moves(moves)
        assert state.is_terminal()
        assert state.winner() is None
        assert agent._reward(state, Player.X) == 0.0
        assert agent._reward(state, Player.O) == 0.0


class TestRandomRollout:
    def test_restores_state_fully(self):
        agent = _DummyMCAgent("Dummy", n_simulations=1, seed=1)
        state = fresh_state(n=3, k=3)
        history_before = list(state.history)
        empty_before = set(state.board.get_empty_cells())

        reward, moves_played = agent._random_rollout(state, Player.X)

        assert state.history == history_before
        assert state.current_player is Player.X
        assert set(state.board.get_empty_cells()) == empty_before
        assert reward in (-1.0, 0.0, 1.0)
        assert 1 <= len(moves_played) <= 9

    def test_reward_is_deterministic_one_move_from_win(self):
        agent = _DummyMCAgent("Dummy", n_simulations=1, seed=0)
        # Only (2,2) is empty; playing it completes column 2 for X.
        #   X | O | X
        #   O | O | X
        #   O | X | .
        moves = (
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 2), (1, 1),
            (2, 1), (2, 0),
        )
        state = state_with_moves(moves, n=3, k=3)
        assert state.board.get_empty_cells() == [(2, 2)]

        reward, moves_played = agent._random_rollout(state, Player.X)

        assert moves_played == [(Player.X, (2, 2))]
        assert reward == 1.0


class TestExpandOne:
    def test_pops_move_and_appends_child(self):
        agent = _DummyMCAgent("Dummy", n_simulations=1)
        state = fresh_state(n=3, k=3)
        moves = state.board.get_empty_cells()
        node = MCNode(None, None, state.current_player, list(moves))
        expected_move = moves[-1]

        child = agent._expand_one(node, state)

        assert child.move == expected_move
        assert child.parent is node
        assert node.children == [child]
        assert expected_move not in node.untried_moves
        assert state.history[-1] == expected_move
        state.undo()


class TestBackpropagate:
    def test_alternates_sign_up_the_path(self):
        agent = _DummyMCAgent("Dummy", n_simulations=1)
        root = MCNode(None, None, Player.X, [])
        mid = MCNode(root, (0, 0), Player.O, [])
        leaf = MCNode(mid, (1, 1), Player.X, [])
        path = [root, mid, leaf]

        agent._backpropagate(path, reward=1.0)

        assert leaf.visits == mid.visits == root.visits == 1
        assert leaf.value_sum == 1.0
        assert mid.value_sum == -1.0
        assert root.value_sum == 1.0
