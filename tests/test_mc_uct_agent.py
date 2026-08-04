"""Tests for tictactoe/agents/mc_uct_agent.py."""

from __future__ import annotations

import math

from tictactoe.agents.mc_shared import MCNode
from tictactoe.agents.mc_uct_agent import MCUCTAgent
from tictactoe.core.types import Player
from tests.test_helper import PUZZLE_3X3, fresh_state, state_with_moves


class TestInit:
    def test_defaults(self):
        agent = MCUCTAgent()
        assert agent.name == "MCUCTAgent"
        assert agent.n_simulations == 200
        assert agent.c == math.sqrt(2)


class TestSelectChild:
    def test_prefers_unvisited_child(self):
        agent = MCUCTAgent()
        parent = MCNode(None, None, Player.X, [])
        parent.visits = 5
        visited = MCNode(parent, (0, 0), Player.O, [])
        visited.visits = 5
        visited.value_sum = 5.0
        unvisited = MCNode(parent, (0, 1), Player.O, [])
        parent.children = [visited, unvisited]

        assert agent._select_child(parent) is unvisited

    def test_prefers_higher_ucb1_score(self):
        agent = MCUCTAgent(c=0.0)  # c=0 reduces UCT to pure exploitation
        parent = MCNode(None, None, Player.X, [])
        parent.visits = 10
        strong = MCNode(parent, (0, 0), Player.O, [])
        strong.visits = 1
        strong.value_sum = 1.0
        weak = MCNode(parent, (0, 1), Player.O, [])
        weak.visits = 1
        weak.value_sum = -1.0
        parent.children = [weak, strong]

        assert agent._select_child(parent) is strong


class TestActWinningMove:
    def test_takes_immediate_win(self):
        agent = MCUCTAgent(n_simulations=300, seed=7)
        state = state_with_moves(PUZZLE_3X3.moves, n=PUZZLE_3X3.n, k=PUZZLE_3X3.k)
        assert agent.act(state) == PUZZLE_3X3.best_move


class TestActRestoresState:
    def test_state_unchanged_after_act(self):
        agent = MCUCTAgent(n_simulations=20, seed=1)
        state = fresh_state(n=3, k=3)
        history_before = list(state.history)
        empty_before = set(state.board.get_empty_cells())

        agent.act(state)

        assert state.history == history_before
        assert set(state.board.get_empty_cells()) == empty_before


class TestActFullGame:
    def test_self_play_terminates(self):
        agent = MCUCTAgent(n_simulations=30, seed=9)
        state = fresh_state(n=3, k=3)
        while not state.is_terminal():
            state.apply(*agent.act(state))
        assert state.is_terminal()
        assert state.board.is_full() or state.winner() is not None
