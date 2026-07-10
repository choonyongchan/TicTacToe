"""Tests for src/agents/mc_informed_agent.py."""

from __future__ import annotations

from src.agents.mc_informed_agent import MCInformedAgent
from src.tests.test_helper import (
    PUZZLE_3X3,
    PUZZLE_4X4_BLOCK,
    fresh_state,
    state_with_moves,
)


class TestInit:
    def test_defaults(self):
        agent = MCInformedAgent()
        assert agent.name == "MCInformedAgent"
        assert agent.n_simulations == 200


class TestRolloutMove:
    def test_plays_forced_move_when_available(self):
        agent = MCInformedAgent(seed=1)
        state = state_with_moves(
            PUZZLE_4X4_BLOCK.moves, n=PUZZLE_4X4_BLOCK.n, k=PUZZLE_4X4_BLOCK.k
        )
        assert agent._rollout_move(state) == PUZZLE_4X4_BLOCK.best_move

    def test_falls_back_to_heuristic_best_move_when_no_forced_move(self):
        agent = MCInformedAgent(seed=1)
        state = fresh_state(n=3, k=3)
        move = agent._rollout_move(state)
        assert move in state.board.get_empty_cells()

    def test_heuristics_toggle_restricts_registry(self):
        agent = MCInformedAgent(seed=1, heuristics=["fork"])
        assert agent.heuristics == ("fork",)


class TestActWinningMove:
    def test_takes_immediate_win(self):
        agent = MCInformedAgent(n_simulations=200, seed=2)
        state = state_with_moves(PUZZLE_3X3.moves, n=PUZZLE_3X3.n, k=PUZZLE_3X3.k)
        assert agent.act(state) == PUZZLE_3X3.best_move


class TestActFullGame:
    def test_self_play_terminates(self):
        agent = MCInformedAgent(n_simulations=30, seed=4)
        state = fresh_state(n=3, k=3)
        while not state.is_terminal():
            state.apply(*agent.act(state))
        assert state.is_terminal()
        assert state.board.is_full() or state.winner() is not None
