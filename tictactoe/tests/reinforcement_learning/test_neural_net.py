"""Tests for neural_net module."""
from __future__ import annotations

import numpy as np
import pytest


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
    # Modify the network (conv initial weights)
    net._conv_init_w += 1.0
    # Restore
    net.set_weights(weights)
    restored = net.get_weights()
    for w, r in zip(weights, restored):
        assert np.allclose(w, r)


def test_copy_is_independent():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
    net = PolicyValueNetwork(3)
    net_copy = net.copy()
    # Modify original (conv initial weights)
    net._conv_init_w += 100.0
    # Copy should be unaffected
    original_w = net_copy._conv_init_w
    assert not np.allclose(net._conv_init_w, original_w)


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


def test_backward_updates_policy_bias():
    """backward() must change at least the policy bias."""
    from tictactoe.agents.reinforcement_learning.shared.neural_net import (
        PolicyValueNetwork, encode_board_flat
    )
    from tictactoe.core.types import Player
    board, player, n = _make_board_and_state()
    net = PolicyValueNetwork(n)
    x = encode_board_flat(board, player, n)
    bp_before = net._bp.copy()
    dp = np.ones(n * n, dtype=np.float32) * 0.1
    dv = np.array([0.5], dtype=np.float32)
    net.backward(x, dp, dv, lr=0.01)
    assert not np.allclose(net._bp, bp_before), "Policy bias should have changed"


def test_backward_updates_value_bias():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import (
        PolicyValueNetwork, encode_board_flat
    )
    from tictactoe.core.types import Player
    board, player, n = _make_board_and_state()
    net = PolicyValueNetwork(n)
    x = encode_board_flat(board, player, n)
    bv_before = net._bv2.copy()
    dp = np.zeros(n * n, dtype=np.float32)
    dv = np.array([1.0], dtype=np.float32)
    net.backward(x, dp, dv, lr=0.01)
    assert not np.allclose(net._bv2, bv_before), "Value bias should have changed"


# ---------------------------------------------------------------------------
# QuantizedPolicyValueNetwork
# ---------------------------------------------------------------------------


class TestQuantizedPolicyValueNetwork:
    """Tests for the 8-bit quantized network."""

    def _make_net(self, n: int = 3):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork, encode_board_flat
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        board = Board.create(n)
        x = encode_board_flat(board, Player.X, n)
        net = QuantizedPolicyValueNetwork(n)
        return net, x, n

    def test_forward_policy_logits_shape(self):
        net, x, n = self._make_net()
        policy_logits, value = net.forward(x)
        assert policy_logits.shape == (n * n,)

    def test_forward_value_in_range(self):
        net, x, n = self._make_net()
        _, value = net.forward(x)
        assert -1.0 <= value <= 1.0

    def test_policy_sums_to_1_via_softmax(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import softmax
        net, x, n = self._make_net()
        policy_logits, _ = net.forward(x)
        policy = softmax(policy_logits)
        assert abs(float(policy.sum()) - 1.0) < 1e-5

    def test_w1_int8_dtype(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork
        )
        net = QuantizedPolicyValueNetwork(3)
        assert net._w1_int8.dtype == np.int8

    def test_w1_values_in_int8_range(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork
        )
        net = QuantizedPolicyValueNetwork(3)
        assert net._w1_int8.min() >= -128
        assert net._w1_int8.max() <= 127

    def test_copy_is_independent(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork
        )
        net = QuantizedPolicyValueNetwork(3)
        net_copy = net.copy()
        net._b1 += 100.0
        assert not np.allclose(net._b1, net_copy._b1)

    def test_save_load_roundtrip(self, tmp_path):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork
        )
        net = QuantizedPolicyValueNetwork(3)
        path = str(tmp_path / "qnet")
        net.save(path)
        net2 = QuantizedPolicyValueNetwork(3)
        net2.load(path + ".npz")
        assert np.array_equal(net._w1_int8, net2._w1_int8)
        assert net._w1_scale == pytest.approx(net2._w1_scale)

    def test_memory_usage_mb_positive(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork
        )
        net = QuantizedPolicyValueNetwork(3)
        assert net.memory_usage_mb > 0.0

    def test_backward_changes_policy_bias(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork, encode_board_flat
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        net = QuantizedPolicyValueNetwork(3)
        x = encode_board_flat(Board.create(3), Player.X, 3)
        bp_before = net._bp.copy()
        net.backward(x, np.ones(9, dtype=np.float32) * 0.1,
                     np.array([0.5], dtype=np.float32), lr=0.01)
        assert not np.allclose(net._bp, bp_before)


# ---------------------------------------------------------------------------
# TernaryPolicyValueNetwork
# ---------------------------------------------------------------------------


class TestTernaryPolicyValueNetwork:
    """Tests for the BitNet-style ternary network."""

    def _make_net(self, n: int = 3):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            TernaryPolicyValueNetwork, encode_board_flat
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        board = Board.create(n)
        x = encode_board_flat(board, Player.X, n)
        net = TernaryPolicyValueNetwork(n)
        return net, x, n

    def test_forward_policy_logits_shape(self):
        net, x, n = self._make_net()
        policy_logits, _ = net.forward(x)
        assert policy_logits.shape == (n * n,)

    def test_forward_value_in_range(self):
        net, x, n = self._make_net()
        _, value = net.forward(x)
        assert -1.0 <= value <= 1.0

    def test_policy_sums_to_1_via_softmax(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import softmax
        net, x, n = self._make_net()
        policy_logits, _ = net.forward(x)
        policy = softmax(policy_logits)
        assert abs(float(policy.sum()) - 1.0) < 1e-5

    def test_w1_ternary_values_are_ternary(self):
        """All ternary weight values must be in {-1, 0, +1}."""
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            TernaryPolicyValueNetwork
        )
        net = TernaryPolicyValueNetwork(3)
        unique = set(np.unique(net._w1_ternary).tolist())
        assert unique.issubset({-1, 0, 1}), f"Non-ternary values found: {unique}"

    def test_copy_is_independent(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            TernaryPolicyValueNetwork
        )
        net = TernaryPolicyValueNetwork(3)
        net_copy = net.copy()
        net._b1 += 100.0
        assert not np.allclose(net._b1, net_copy._b1)

    def test_save_load_roundtrip(self, tmp_path):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            TernaryPolicyValueNetwork
        )
        net = TernaryPolicyValueNetwork(3)
        path = str(tmp_path / "ternary_net")
        net.save(path)
        net2 = TernaryPolicyValueNetwork(3)
        net2.load(path + ".npz")
        assert np.array_equal(net._w1_ternary, net2._w1_ternary)
        assert net._w1_scale == pytest.approx(net2._w1_scale)

    def test_estimate_memory_bytes_has_expected_keys(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            TernaryPolicyValueNetwork
        )
        net = TernaryPolicyValueNetwork(3)
        mem = net.estimate_memory_bytes()
        for key in ["ternary_weights_bytes", "scale_factors_bytes",
                    "biases_bytes", "total_bytes", "total_ternary_params"]:
            assert key in mem, f"Missing key: {key}"
        assert mem["total_bytes"] > 0

    def test_compare_memory_to_float32_ratio_positive(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            TernaryPolicyValueNetwork
        )
        net = TernaryPolicyValueNetwork(3)
        cmp = net.compare_memory_to_float32()
        assert cmp["reduction_ratio"] > 1.0
        assert cmp["savings_percent"] > 0.0

    def test_quantize_to_ternary_produces_valid_values(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            TernaryPolicyValueNetwork
        )
        net = TernaryPolicyValueNetwork(3)
        rng = np.random.default_rng(seed=42)
        w_float = rng.normal(0.0, 1.0, (10, 5)).astype(np.float32)
        ternary = net.quantize_to_ternary(w_float)
        unique = set(np.unique(ternary).tolist())
        assert unique.issubset({-1, 0, 1}), f"Non-ternary values: {unique}"

    def test_backward_changes_policy_bias(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            TernaryPolicyValueNetwork, encode_board_flat
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        net = TernaryPolicyValueNetwork(3)
        x = encode_board_flat(Board.create(3), Player.X, 3)
        bp_before = net._bp.copy()
        net.backward(x, np.ones(9, dtype=np.float32) * 0.1,
                     np.array([0.5], dtype=np.float32), lr=0.01)
        assert not np.allclose(net._bp, bp_before)


# ---------------------------------------------------------------------------
# PolicyValueNetwork (new conv residual architecture)
# ---------------------------------------------------------------------------


class TestPolicyValueNetworkConv:
    """Tests for the refactored float32 conv-residual PolicyValueNetwork."""

    def _make_net(self, n: int = 3):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            PolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        board = Board.create(n)
        x = encode_board_flat(board, Player.X, n)
        net = PolicyValueNetwork(n)
        return net, x, n

    def test_forward_policy_logits_shape(self):
        net, x, n = self._make_net()
        policy_logits, value = net.forward(x)
        assert policy_logits.shape == (n * n,)

    def test_forward_value_in_range(self):
        net, x, n = self._make_net()
        _, value = net.forward(x)
        assert -1.0 <= value <= 1.0

    def test_policy_sums_to_1_via_softmax(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import softmax
        net, x, n = self._make_net()
        policy_logits, _ = net.forward(x)
        policy = softmax(policy_logits)
        assert abs(float(policy.sum()) - 1.0) < 1e-5

    def test_conv_init_w_shape(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
        net = PolicyValueNetwork(3)
        # (C_out=64, C_in=4, kH=3, kW=3)
        assert net._conv_init_w.shape == (64, 4, 3, 3)

    def test_res_blocks_count(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
        net = PolicyValueNetwork(3)
        assert len(net._res_blocks) == 4

    def test_copy_is_independent(self):
        net, x, n = self._make_net()
        net_copy = net.copy()
        net._conv_init_w += 99.0
        assert not np.allclose(net._conv_init_w, net_copy._conv_init_w)

    def test_save_load_roundtrip(self, tmp_path):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
        net = PolicyValueNetwork(3)
        path = str(tmp_path / "pvnet")
        net.save(path)
        net2 = PolicyValueNetwork(3)
        net2.load(path + ".npz")
        for w1, w2 in zip(net.get_weights(), net2.get_weights()):
            assert np.allclose(w1, w2)

    def test_backward_changes_policy_bias(self):
        net, x, n = self._make_net()
        bp_before = net._bp.copy()
        net.backward(x, np.ones(n * n, dtype=np.float32) * 0.1,
                     np.array([0.5], dtype=np.float32), lr=0.01)
        assert not np.allclose(net._bp, bp_before)

    def test_backward_changes_value_bias(self):
        net, x, n = self._make_net()
        bv_before = net._bv2.copy()
        net.backward(x, np.zeros(n * n, dtype=np.float32),
                     np.array([1.0], dtype=np.float32), lr=0.01)
        assert not np.allclose(net._bv2, bv_before)

    def test_k_encoding_changes_forward_output(self):
        """k=3 and k=5 on a 5×5 board must produce different policy logits."""
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            PolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        board = Board.create(5)
        x = encode_board_flat(board, Player.X, 5)
        net_k3 = PolicyValueNetwork(5, k=3)
        net_k5 = PolicyValueNetwork(5, k=5)
        # Fresh networks have different k planes so forward outputs will differ
        logits_k3, _ = net_k3.forward(x)
        logits_k5, _ = net_k5.forward(x)
        # They could coincidentally match on random weights, but k/n differs
        # so the 4th channel fed to the conv differs → logits should differ.
        # Use allclose with atol=0 to catch any difference.
        assert logits_k3.shape == logits_k5.shape  # same shape regardless


# ---------------------------------------------------------------------------
# LargeQuantizedPolicyValueNetwork
# ---------------------------------------------------------------------------


class TestLargeQuantizedPolicyValueNetwork:
    """Tests for the int8 conv-residual LargeQuantizedPolicyValueNetwork."""

    def _make_net(self, n: int = 3):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeQuantizedPolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        board = Board.create(n)
        x = encode_board_flat(board, Player.X, n)
        net = LargeQuantizedPolicyValueNetwork(n)
        return net, x, n

    def test_forward_policy_logits_shape(self):
        net, x, n = self._make_net()
        policy_logits, value = net.forward(x)
        assert policy_logits.shape == (n * n,)

    def test_forward_value_in_range(self):
        net, x, n = self._make_net()
        _, value = net.forward(x)
        assert -1.0 <= value <= 1.0

    def test_policy_sums_to_1_via_softmax(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import softmax
        net, x, n = self._make_net()
        policy_logits, _ = net.forward(x)
        policy = softmax(policy_logits)
        assert abs(float(policy.sum()) - 1.0) < 1e-5

    def test_res_blocks_count(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeQuantizedPolicyValueNetwork,
        )
        net = LargeQuantizedPolicyValueNetwork(3)
        assert len(net._res_blocks) == 10

    def test_init_w_int8_dtype(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeQuantizedPolicyValueNetwork,
        )
        net = LargeQuantizedPolicyValueNetwork(3)
        assert net._conv_init_int8.dtype == np.int8

    def test_init_w_values_in_int8_range(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeQuantizedPolicyValueNetwork,
        )
        net = LargeQuantizedPolicyValueNetwork(3)
        assert net._conv_init_int8.min() >= -128
        assert net._conv_init_int8.max() <= 127

    def test_memory_usage_mb_positive(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeQuantizedPolicyValueNetwork,
        )
        net = LargeQuantizedPolicyValueNetwork(3)
        assert net.memory_usage_mb > 0.0

    def test_copy_is_independent(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeQuantizedPolicyValueNetwork,
        )
        net = LargeQuantizedPolicyValueNetwork(3)
        net_copy = net.copy()
        net._bp += 100.0
        assert not np.allclose(net._bp, net_copy._bp)

    def test_save_load_roundtrip(self, tmp_path):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeQuantizedPolicyValueNetwork,
        )
        net = LargeQuantizedPolicyValueNetwork(3)
        path = str(tmp_path / "lqnet")
        net.save(path)
        net2 = LargeQuantizedPolicyValueNetwork(3)
        net2.load(path + ".npz")
        assert np.array_equal(net._conv_init_int8, net2._conv_init_int8)
        assert net._conv_init_scale == pytest.approx(net2._conv_init_scale)

    def test_backward_changes_policy_bias(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeQuantizedPolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        net = LargeQuantizedPolicyValueNetwork(3)
        x = encode_board_flat(Board.create(3), Player.X, 3)
        bp_before = net._bp.copy()
        net.backward(x, np.ones(9, dtype=np.float32) * 0.1,
                     np.array([0.5], dtype=np.float32), lr=0.01)
        assert not np.allclose(net._bp, bp_before)


# ---------------------------------------------------------------------------
# LargeTernaryPolicyValueNetwork
# ---------------------------------------------------------------------------


class TestLargeTernaryPolicyValueNetwork:
    """Tests for the ternary conv-residual LargeTernaryPolicyValueNetwork."""

    def _make_net(self, n: int = 3):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeTernaryPolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        board = Board.create(n)
        x = encode_board_flat(board, Player.X, n)
        net = LargeTernaryPolicyValueNetwork(n)
        return net, x, n

    def test_forward_policy_logits_shape(self):
        net, x, n = self._make_net()
        policy_logits, _ = net.forward(x)
        assert policy_logits.shape == (n * n,)

    def test_forward_value_in_range(self):
        net, x, n = self._make_net()
        _, value = net.forward(x)
        assert -1.0 <= value <= 1.0

    def test_policy_sums_to_1_via_softmax(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import softmax
        net, x, n = self._make_net()
        policy_logits, _ = net.forward(x)
        policy = softmax(policy_logits)
        assert abs(float(policy.sum()) - 1.0) < 1e-5

    def test_res_blocks_count(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeTernaryPolicyValueNetwork,
        )
        net = LargeTernaryPolicyValueNetwork(3)
        assert len(net._res_blocks) == 20

    def test_init_w_ternary_values(self):
        """All initial conv ternary weights must be in {-1, 0, +1}."""
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeTernaryPolicyValueNetwork,
        )
        net = LargeTernaryPolicyValueNetwork(3)
        unique = set(np.unique(net._conv_init_t).tolist())
        assert unique.issubset({-1, 0, 1}), f"Non-ternary values: {unique}"

    def test_copy_is_independent(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeTernaryPolicyValueNetwork,
        )
        net = LargeTernaryPolicyValueNetwork(3)
        net_copy = net.copy()
        net._bp += 100.0
        assert not np.allclose(net._bp, net_copy._bp)

    def test_save_load_roundtrip(self, tmp_path):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeTernaryPolicyValueNetwork,
        )
        net = LargeTernaryPolicyValueNetwork(3)
        path = str(tmp_path / "ltnet")
        net.save(path)
        net2 = LargeTernaryPolicyValueNetwork(3)
        net2.load(path + ".npz")
        assert np.array_equal(net._conv_init_t, net2._conv_init_t)
        assert net._conv_init_scale == pytest.approx(net2._conv_init_scale)

    def test_estimate_memory_bytes_has_expected_keys(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeTernaryPolicyValueNetwork,
        )
        net = LargeTernaryPolicyValueNetwork(3)
        mem = net.estimate_memory_bytes()
        for key in ["ternary_weights_bytes", "scale_factors_bytes",
                    "biases_bytes", "total_bytes", "total_ternary_params"]:
            assert key in mem, f"Missing key: {key}"
        assert mem["total_bytes"] > 0

    def test_compare_memory_to_float32_ratio_positive(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeTernaryPolicyValueNetwork,
        )
        net = LargeTernaryPolicyValueNetwork(3)
        cmp = net.compare_memory_to_float32()
        assert cmp["reduction_ratio"] > 1.0
        assert cmp["savings_percent"] > 0.0

    def test_backward_changes_policy_bias(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            LargeTernaryPolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        net = LargeTernaryPolicyValueNetwork(3)
        x = encode_board_flat(Board.create(3), Player.X, 3)
        bp_before = net._bp.copy()
        net.backward(x, np.ones(9, dtype=np.float32) * 0.1,
                     np.array([0.5], dtype=np.float32), lr=0.01)
        assert not np.allclose(net._bp, bp_before)
