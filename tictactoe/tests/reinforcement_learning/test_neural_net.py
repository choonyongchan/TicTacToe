"""Tests for neural_net module."""
from __future__ import annotations

import pytest

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

pytestmark = pytest.mark.skipif(not HAS_NUMPY, reason="numpy required")


def _make_board_and_state():
    from tictactoe.core.board import Board
    from tictactoe.core.state import GameState
    from tictactoe.core.types import Cell, Player
    board = Board.create(3)
    board[0][0] = Cell.X
    board[1][1] = Cell.O
    return board, Player.X, 3


def test_encode_board_channel_0_own_pieces():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board
    from tictactoe.core.types import Cell, Player
    board, player, n = _make_board_and_state()
    arr = encode_board(board, player, n)
    # Channel 0 should have 1.0 at (0,0) and 0.0 elsewhere
    assert arr[0, 0, 0] == pytest.approx(1.0)
    assert arr[0, 1, 1] == pytest.approx(0.0)  # O's piece
    assert arr[0, 0, 1] == pytest.approx(0.0)  # empty


def test_encode_board_channel_1_opponent_pieces():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board
    from tictactoe.core.types import Cell, Player
    board, player, n = _make_board_and_state()
    arr = encode_board(board, player, n)
    # Channel 1 should have 1.0 at (1,1) (O's piece) and 0.0 at (0,0)
    assert arr[1, 1, 1] == pytest.approx(1.0)
    assert arr[1, 0, 0] == pytest.approx(0.0)


def test_encode_board_channel_2_all_ones():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board
    from tictactoe.core.types import Player
    board, player, n = _make_board_and_state()
    arr = encode_board(board, player, n)
    assert np.all(arr[2] == 1.0)


def test_policy_value_net_policy_sums_to_1():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import (
        PolicyValueNetwork, encode_board_flat, softmax
    )
    from tictactoe.core.types import Player
    board, player, n = _make_board_and_state()
    net = PolicyValueNetwork(n)
    x = encode_board_flat(board, player, n)
    policy_logits, _ = net.forward(x)
    policy = softmax(policy_logits)
    assert abs(float(policy.sum()) - 1.0) < 1e-5
    assert all(p >= 0 for p in policy)


def test_policy_value_net_value_in_range():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import (
        PolicyValueNetwork, encode_board_flat
    )
    from tictactoe.core.types import Player
    board, player, n = _make_board_and_state()
    net = PolicyValueNetwork(n)
    x = encode_board_flat(board, player, n)
    _, value = net.forward(x)
    assert -1.0 <= value <= 1.0


def test_q_network_output_shape():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import (
        QNetwork, encode_board_flat
    )
    from tictactoe.core.types import Player
    board, player, n = _make_board_and_state()
    net = QNetwork(n)
    x = encode_board_flat(board, player, n)
    q_vals = net.forward(x)
    assert q_vals.shape == (n * n,)


def test_get_set_weights_roundtrip():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
    net = PolicyValueNetwork(3)
    weights = net.get_weights()
    # Modify the network
    net._w1 += 1.0
    # Restore
    net.set_weights(weights)
    restored = net.get_weights()
    for w, r in zip(weights, restored):
        assert np.allclose(w, r)


def test_copy_is_independent():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
    net = PolicyValueNetwork(3)
    net_copy = net.copy()
    # Modify original
    net._w1 += 100.0
    # Copy should be unaffected
    original_w1 = net_copy._w1
    assert not np.allclose(net._w1, original_w1)


def test_softmax_sums_to_1():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import softmax
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = softmax(x)
    assert abs(float(result.sum()) - 1.0) < 1e-6


def test_relu_zeros_negatives():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import relu
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    result = relu(x)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(1.0)


def test_save_load_roundtrip(tmp_path):
    from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
    net = PolicyValueNetwork(3)
    path = str(tmp_path / "net")
    net.save(path)
    net2 = PolicyValueNetwork(3)
    net2.load(path + '.npz')
    for w1, w2 in zip(net.get_weights(), net2.get_weights()):
        assert np.allclose(w1, w2)
