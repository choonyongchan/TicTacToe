"""Tests for tictactoe/agents/mc_rave_agent.py."""

from __future__ import annotations

from tictactoe.agents.mc_rave_agent import MCRaveAgent
from tictactoe.agents.mc_shared import MCRaveNode
from tictactoe.core.types import Player
from tests.test_helper import PUZZLE_3X3, fresh_state, state_with_moves


class TestInit:
    def test_defaults(self):
        agent = MCRaveAgent()
        assert agent.name == "MCRaveAgent"
        assert agent.rave_k == 500.0


class TestSelectChild:
    def test_prefers_unvisited_child(self):
        agent = MCRaveAgent(rave_k=10.0)
        parent = MCRaveNode(None, None, Player.X, [])
        parent.visits = 5
        visited = MCRaveNode(parent, (0, 0), Player.O, [])
        visited.visits = 5
        visited.value_sum = 5.0
        unvisited = MCRaveNode(parent, (0, 1), Player.O, [])
        parent.children = [visited, unvisited]

        assert agent._select_child(parent) is unvisited

    def test_blends_uct_and_amaf(self):
        agent = MCRaveAgent(c=0.0, rave_k=10.0)
        parent = MCRaveNode(None, None, Player.X, [])
        parent.visits = 20
        child = MCRaveNode(parent, (0, 0), Player.O, [])
        child.visits = 5  # alpha = (10-5)/10 = 0.5
        child.value_sum = 2.5  # uct_score (c=0) == q() == 0.5
        parent.amaf_visits[(0, 0)] = 4
        parent.amaf_values[(0, 0)] = 2.0  # amaf_q = 0.5
        other = MCRaveNode(parent, (0, 1), Player.O, [])
        other.visits = 5
        other.value_sum = -5.0
        parent.children = [child, other]

        # Both terms equal 0.5 here, so combined score is also 0.5.
        assert agent._select_child(parent) is child


class TestBackpropagate:
    def test_updates_amaf_for_matching_player_moves(self):
        agent = MCRaveAgent()
        root = MCRaveNode(None, None, Player.X, [])
        leaf = MCRaveNode(root, (0, 0), Player.O, [])
        path = [root, leaf]
        rollout_moves = [(Player.O, (1, 1)), (Player.X, (2, 2))]

        agent._backpropagate(path, reward=1.0, rollout_moves=rollout_moves)

        # leaf.player is O: only the O move from the rollout updates leaf's AMAF.
        assert leaf.amaf_visits == {(1, 1): 1}
        assert leaf.amaf_values[(1, 1)] == -1.0  # -reward * sign(=1) at the leaf
        # root.player is X: only the X move from the rollout updates root's AMAF.
        assert root.amaf_visits == {(2, 2): 1}
        assert root.amaf_values[(2, 2)] == 1.0  # -reward * sign(=-1) at root


class TestActWinningMove:
    def test_takes_immediate_win(self):
        agent = MCRaveAgent(n_simulations=300, seed=11)
        state = state_with_moves(PUZZLE_3X3.moves, n=PUZZLE_3X3.n, k=PUZZLE_3X3.k)
        assert agent.act(state) == PUZZLE_3X3.best_move


class TestActFullGame:
    def test_self_play_terminates(self):
        agent = MCRaveAgent(n_simulations=30, seed=13)
        state = fresh_state(n=3, k=3)
        while not state.is_terminal():
            state.apply(*agent.act(state))
        assert state.is_terminal()
        assert state.board.is_full() or state.winner() is not None
