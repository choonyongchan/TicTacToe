"""Tests for src/agents/mc_puct_agent.py."""

from __future__ import annotations

from src.agents.mc_puct_agent import MCPUCTAgent
from src.agents.mc_shared import MCPUCTNode
from src.core.types import Player
from src.tests.test_helper import fresh_state


class TestInit:
    def test_defaults(self):
        agent = MCPUCTAgent()
        assert agent.name == "MCPUCTAgent"
        assert agent.n_simulations == 200
        assert agent.net is not None


class TestExpand:
    def test_creates_one_child_per_legal_move_with_normalised_priors(self):
        agent = MCPUCTAgent(seed=0)
        state = fresh_state(n=3, k=3)
        history_before = list(state.history)
        root = MCPUCTNode(None, None, state.current_player, [])

        agent._expand(root, state)

        legal_moves = {(r, c) for r in range(3) for c in range(3)}
        assert {child.move for child in root.children} == legal_moves
        assert sum(child.prior for child in root.children) == 1.0 or abs(
            sum(child.prior for child in root.children) - 1.0
        ) < 1e-9
        assert state.history == history_before  # _expand restores state


class TestPuctScore:
    def test_rewards_high_prior_when_unvisited(self):
        agent = MCPUCTAgent(c_puct=1.0)
        parent = MCPUCTNode(None, None, Player.X, [])
        parent.visits = 4
        high_prior = MCPUCTNode(parent, (0, 0), Player.O, [])
        high_prior.prior = 0.9
        low_prior = MCPUCTNode(parent, (0, 1), Player.O, [])
        low_prior.prior = 0.1

        assert agent._puct_score(parent, high_prior) > agent._puct_score(parent, low_prior)


class TestActRestoresState:
    def test_returns_legal_move_and_restores_state(self):
        agent = MCPUCTAgent(n_simulations=10, seed=0)
        state = fresh_state(n=3, k=3)
        history_before = list(state.history)

        move = agent.act(state)

        assert move in state.board.get_empty_cells()
        assert state.history == history_before


class TestActFullGame:
    def test_self_play_terminates(self):
        agent = MCPUCTAgent(n_simulations=10, seed=2)
        state = fresh_state(n=3, k=3)
        while not state.is_terminal():
            state.apply(*agent.act(state))
        assert state.is_terminal()
        assert state.board.is_full() or state.winner() is not None
