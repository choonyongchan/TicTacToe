"""Tests for PPOSelfPlayAgent."""
import numpy as np
import pytest

from tictactoe.agents.reinforcement_learning.ppo_selfplay import PPOSelfPlayAgent
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player


def make_empty_state(n=3):
    return GameState(board=Board.create(n), current_player=Player.X, n=n, k=n)


def test_returns_legal_move():
    agent = PPOSelfPlayAgent(n=3)
    state = make_empty_state()
    move = agent.choose_move(state)
    r, c = move
    assert state.board[r][c] is Cell.EMPTY


def test_get_name_and_tier():
    agent = PPOSelfPlayAgent(n=3)
    assert "PPO" in agent.get_name()
    assert agent.get_tier() == 4


def test_save_load_roundtrip(tmp_path):
    import numpy as np
    agent = PPOSelfPlayAgent(n=3)
    path = str(tmp_path / "ppo_model")
    agent.save(path)
    agent2 = PPOSelfPlayAgent(n=3)
    agent2.load(path)
    for w1, w2 in zip(agent._net.get_weights(), agent2._net.get_weights()):
        assert np.allclose(w1, w2)


def test_policy_is_valid_distribution():
    import numpy as np
    from tictactoe.agents.reinforcement_learning.shared.neural_net import softmax, encode_board_flat
    agent = PPOSelfPlayAgent(n=3)
    state = make_empty_state()
    x = encode_board_flat(state.board, state.current_player, state.n)
    policy_logits, _ = agent._net.forward(x)
    policy = softmax(policy_logits)
    assert abs(policy.sum() - 1.0) < 1e-5
    assert all(p >= 0 for p in policy)


def test_nodes_visited_is_1():
    agent = PPOSelfPlayAgent(n=3)
    state = make_empty_state()
    agent.choose_move(state)
    assert state.nodes_visited == 1
