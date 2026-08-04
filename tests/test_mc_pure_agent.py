"""Tests for tictactoe/agents/mc_pure_agent.py."""

from __future__ import annotations

from tictactoe.agents.mc_pure_agent import MCPureAgent
from tests.test_helper import PUZZLE_3X3, fresh_state, state_with_moves


class TestInit:
    def test_defaults(self):
        agent = MCPureAgent()
        assert agent.name == "MCPureAgent"
        assert agent.n_simulations == 200

    def test_custom_simulations(self):
        agent = MCPureAgent(n_simulations=50, seed=1)
        assert agent.n_simulations == 50


class TestActWinningMove:
    def test_takes_immediate_win(self):
        # Any rollout starting from the winning move scores exactly 1.0 (it
        # terminates immediately), which no other move's average can beat.
        agent = MCPureAgent(n_simulations=60, seed=3)
        state = state_with_moves(PUZZLE_3X3.moves, n=PUZZLE_3X3.n, k=PUZZLE_3X3.k)
        assert agent.act(state) == PUZZLE_3X3.best_move


class TestActFullGame:
    def test_self_play_terminates(self):
        agent = MCPureAgent(n_simulations=30, seed=5)
        state = fresh_state(n=3, k=3)
        while not state.is_terminal():
            state.apply(*agent.act(state))
        assert state.is_terminal()
        assert state.board.is_full() or state.winner() is not None
