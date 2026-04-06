"""Tests for AlphaZeroAgent."""
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

import pytest
pytestmark = pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")

from tictactoe.agents.reinforcement_learning.alphazero import AlphaZeroAgent
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Player


def make_empty_state(n=3):
    return GameState(board=Board.create(n), current_player=Player.X, n=n, k=n)


def test_returns_legal_move():
    agent = AlphaZeroAgent(n=3, num_simulations=10)
    state = make_empty_state()
    move = agent.choose_move(state)
    r, c = move
    assert state.board[r][c] is Cell.EMPTY


def test_get_name_and_tier():
    agent = AlphaZeroAgent(n=3)
    assert "AlphaZero" in agent.get_name()
    assert agent.get_tier() == 4


def test_picks_immediate_winning_move():
    import numpy as np
    from tictactoe.core.types import Cell as C
    board = Board.create(3)
    board[0][0] = C.X
    board[0][1] = C.X
    state = GameState(board=board, current_player=Player.X, n=3, k=3, last_move=(0, 1))
    agent = AlphaZeroAgent(n=3, num_simulations=10)
    move = agent.choose_move(state)
    assert move == (0, 2)


def test_data_augmentation_produces_8_variants():
    import numpy as np
    agent = AlphaZeroAgent(n=3)
    board = Board.create(3)
    board[0][0] = Cell.X
    policy = np.ones(9, dtype=np.float32) / 9
    variants = agent._generate_symmetries(board, 3, policy, 1.0)
    assert len(variants) == 8


def test_save_load_roundtrip(tmp_path):
    import numpy as np
    agent = AlphaZeroAgent(n=3)
    path = str(tmp_path / "az_model")
    agent.save(path)
    agent2 = AlphaZeroAgent(n=3)
    agent2._net.load(path + '.npz')
    for w1, w2 in zip(agent._net.get_weights(), agent2._net.get_weights()):
        assert np.allclose(w1, w2)
