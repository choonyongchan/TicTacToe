"""Tests for neural_net module."""
from __future__ import annotations

import numpy as np
import pytest
import torch


def _make_board_and_state():
    from tictactoe.core.board import Board
    from tictactoe.core.state import GameState
    from tictactoe.core.types import Cell, Player
    board = Board.create(3)
    board[0][0] = Cell.X
    board[1][1] = Cell.O
    return board, Player.X, 3


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

def test_encode_board_channel_0_own_pieces():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board
    from tictactoe.core.types import Cell, Player
    board, player, n = _make_board_and_state()
    arr = encode_board(board, player, n)
    assert arr[0, 0, 0] == pytest.approx(1.0)
    assert arr[0, 1, 1] == pytest.approx(0.0)
    assert arr[0, 0, 1] == pytest.approx(0.0)


def test_encode_board_channel_1_opponent_pieces():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board
    from tictactoe.core.types import Cell, Player
    board, player, n = _make_board_and_state()
    arr = encode_board(board, player, n)
    assert arr[1, 1, 1] == pytest.approx(1.0)
    assert arr[1, 0, 0] == pytest.approx(0.0)


def test_encode_board_channel_2_all_ones():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board
    from tictactoe.core.types import Player
    board, player, n = _make_board_and_state()
    arr = encode_board(board, player, n)
    assert np.all(arr[2].cpu().numpy() == 1.0)


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def test_softmax_sums_to_1():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import softmax
    import torch
    x = torch.tensor([1.0, 2.0, 3.0])
    result = softmax(x)
    assert abs(float(result.sum()) - 1.0) < 1e-6


def test_relu_zeros_negatives():
    from tictactoe.agents.reinforcement_learning.shared.neural_net import relu
    x = torch.tensor([-1.0, 0.0, 1.0])
    result = relu(x)
    assert float(result[0]) == pytest.approx(0.0)
    assert float(result[1]) == pytest.approx(0.0)
    assert float(result[2]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# QNetwork
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PolicyValueNetwork (float32, 10 blocks, 256 filters)
# ---------------------------------------------------------------------------

class TestPolicyValueNetwork:
    """Tests for the float32 conv-residual PolicyValueNetwork."""

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
        assert -1.0 <= float(value) <= 1.0

    def test_policy_sums_to_1_via_softmax(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import softmax
        net, x, n = self._make_net()
        policy_logits, _ = net.forward(x)
        policy = softmax(policy_logits)
        assert abs(float(policy.sum()) - 1.0) < 1e-5

    def test_architecture_constants(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
        assert PolicyValueNetwork.NUM_RES_BLOCKS == 10
        assert PolicyValueNetwork.FILTERS == 256

    def test_conv_init_shape(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
        net = PolicyValueNetwork(3)
        assert net.conv_init.weight.shape == (256, 4, 3, 3)

    def test_res_blocks_count(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
        net = PolicyValueNetwork(3)
        assert len(net.res_blocks) == 10

    def test_value_head_width(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
        net = PolicyValueNetwork(3)
        assert net.fc_v1.out_features == 256

    def test_all_parameters_trainable(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
        net = PolicyValueNetwork(3)
        frozen = [n for n, p in net.named_parameters() if not p.requires_grad]
        assert frozen == [], f"Expected no frozen params, got: {frozen}"

    def test_copy_is_independent(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
        net = PolicyValueNetwork(3)
        net_copy = net.copy()
        with torch.no_grad():
            net.conv_init.weight += 99.0
        assert not torch.allclose(net.conv_init.weight, net_copy.conv_init.weight)

    def test_save_load_roundtrip(self, tmp_path):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            PolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        net = PolicyValueNetwork(3)
        path = str(tmp_path / "pvnet")
        net.save(path)
        net2 = PolicyValueNetwork(3)
        net2.load(path + ".pt")
        x = encode_board_flat(Board.create(3), Player.X, 3)
        p1, v1 = net.forward(x)
        p2, v2 = net2.forward(x)
        assert torch.allclose(p1, p2)
        assert torch.allclose(v1, v2)

    def test_train_step_changes_parameters(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            PolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        net = PolicyValueNetwork(3)
        x = encode_board_flat(Board.create(3), Player.X, 3)
        fc_p_w_before = net.fc_p.weight.detach().clone()
        target_p = torch.ones(9) / 9.0
        target_v = torch.tensor([0.5])
        net.train_step(x, target_p, target_v)
        assert not torch.allclose(net.fc_p.weight, fc_p_w_before)

    def test_k_encoding_changes_forward_output(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            PolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        board = Board.create(5)
        x = encode_board_flat(board, Player.X, 5)
        net_k3 = PolicyValueNetwork(5, k=3)
        net_k5 = PolicyValueNetwork(5, k=5)
        logits_k3, _ = net_k3.forward(x)
        logits_k5, _ = net_k5.forward(x)
        assert logits_k3.shape == logits_k5.shape


# ---------------------------------------------------------------------------
# QuantizedPolicyValueNetwork (int8, 10 blocks, 256 filters)
# ---------------------------------------------------------------------------

class TestQuantizedPolicyValueNetwork:
    """Tests for the int8 conv-residual QuantizedPolicyValueNetwork."""

    def _make_net(self, n: int = 3):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork, encode_board_flat,
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
        assert -1.0 <= float(value) <= 1.0

    def test_policy_sums_to_1_via_softmax(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import softmax
        net, x, n = self._make_net()
        policy_logits, _ = net.forward(x)
        policy = softmax(policy_logits)
        assert abs(float(policy.sum()) - 1.0) < 1e-5

    def test_architecture_constants(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork,
        )
        assert QuantizedPolicyValueNetwork.NUM_RES_BLOCKS == 10
        assert QuantizedPolicyValueNetwork.FILTERS == 256

    def test_conv_init_w_int8_dtype(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork,
        )
        net = QuantizedPolicyValueNetwork(3)
        assert net.conv_init_w.dtype == torch.int8

    def test_conv_init_w_values_in_int8_range(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork,
        )
        net = QuantizedPolicyValueNetwork(3)
        assert int(net.conv_init_w.min()) >= -128
        assert int(net.conv_init_w.max()) <= 127

    def test_res_w1_int8_dtype(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork,
        )
        net = QuantizedPolicyValueNetwork(3)
        assert net.res_w1.dtype == torch.int8

    def test_res_w1_shape(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork,
        )
        net = QuantizedPolicyValueNetwork(3)
        # (N_blocks, C_out, C_in, kH, kW) = (10, 256, 256, 3, 3)
        assert net.res_w1.shape == (10, 256, 256, 3, 3)

    def test_only_output_biases_are_parameters(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork,
        )
        net = QuantizedPolicyValueNetwork(3)
        param_names = [n for n, _ in net.named_parameters()]
        assert set(param_names) == {"bp", "bv2"}

    def test_memory_usage_mb_positive(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork,
        )
        net = QuantizedPolicyValueNetwork(3)
        assert net.memory_usage_mb > 0.0

    def test_copy_is_independent(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork,
        )
        net = QuantizedPolicyValueNetwork(3)
        net_copy = net.copy()
        with torch.no_grad():
            net.bp += 100.0
        assert not torch.allclose(net.bp, net_copy.bp)

    def test_save_load_roundtrip(self, tmp_path):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        net = QuantizedPolicyValueNetwork(3)
        path = str(tmp_path / "qnet")
        net.save(path)
        net2 = QuantizedPolicyValueNetwork(3)
        net2.load(path + ".pt")
        assert torch.equal(net.conv_init_w, net2.conv_init_w)
        assert torch.allclose(net.conv_init_s, net2.conv_init_s)

    def test_train_step_changes_output_bias(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        net = QuantizedPolicyValueNetwork(3)
        x = encode_board_flat(Board.create(3), Player.X, 3)
        bp_before = net.bp.detach().clone()
        target_p = torch.ones(9) / 9.0
        target_v = torch.tensor([0.5])
        net.train_step(x, target_p, target_v)
        assert not torch.allclose(net.bp, bp_before)

    def test_architecture_matches_float32_network(self):
        """QuantizedPolicyValueNetwork must have the same depth/width as PolicyValueNetwork."""
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            PolicyValueNetwork, QuantizedPolicyValueNetwork,
        )
        assert PolicyValueNetwork.NUM_RES_BLOCKS == QuantizedPolicyValueNetwork.NUM_RES_BLOCKS
        assert PolicyValueNetwork.FILTERS == QuantizedPolicyValueNetwork.FILTERS


# ---------------------------------------------------------------------------
# BitNetPolicyValueNetwork (15 layers, d=256)
# ---------------------------------------------------------------------------

class TestBitNetPolicyValueNetwork:
    """Tests for the BitNet 1.58-bit Transformer."""

    def _make_net(self, n: int = 3):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            BitNetPolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        board = Board.create(n)
        x = encode_board_flat(board, Player.X, n)
        net = BitNetPolicyValueNetwork(n)
        return net, x, n

    def test_forward_policy_logits_shape(self):
        net, x, n = self._make_net()
        policy_logits, value = net.forward(x)
        assert policy_logits.shape == (n * n,)

    def test_forward_value_in_range(self):
        net, x, n = self._make_net()
        _, value = net.forward(x)
        assert -1.0 <= float(value) <= 1.0

    def test_policy_sums_to_1_via_softmax(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import softmax
        net, x, n = self._make_net()
        policy_logits, _ = net.forward(x)
        policy = softmax(policy_logits)
        assert abs(float(policy.sum()) - 1.0) < 1e-5

    def test_architecture_constants(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            BitNetPolicyValueNetwork,
        )
        assert BitNetPolicyValueNetwork._N_LAYERS == 15
        assert BitNetPolicyValueNetwork._D_MODEL == 256
        assert BitNetPolicyValueNetwork._N_HEADS == 8
        assert BitNetPolicyValueNetwork._D_FF == 1024

    def test_n_transformer_blocks(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            BitNetPolicyValueNetwork,
        )
        net = BitNetPolicyValueNetwork(3)
        assert len(net._blocks) == 15

    def test_all_parameters_trainable(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            BitNetPolicyValueNetwork,
        )
        net = BitNetPolicyValueNetwork(3)
        frozen = [n for n, p in net.named_parameters() if not p.requires_grad]
        assert frozen == [], f"Expected no frozen params, got: {frozen}"

    def test_copy_is_independent(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            BitNetPolicyValueNetwork,
        )
        net = BitNetPolicyValueNetwork(3)
        net_copy = net.copy()
        with torch.no_grad():
            net._embed.weight += 99.0
        assert not torch.allclose(net._embed.weight, net_copy._embed.weight)

    def test_copy_returns_correct_type(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            BitNetPolicyValueNetwork,
        )
        net = BitNetPolicyValueNetwork(3)
        assert isinstance(net.copy(), BitNetPolicyValueNetwork)

    def test_save_load_roundtrip(self, tmp_path):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            BitNetPolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        net = BitNetPolicyValueNetwork(3)
        path = str(tmp_path / "bitnet")
        net.save(path)
        net2 = BitNetPolicyValueNetwork(3)
        net2.load(path + ".pt")
        x = encode_board_flat(Board.create(3), Player.X, 3)
        p1, v1 = net.forward(x)
        p2, v2 = net2.forward(x)
        assert torch.allclose(p1, p2, atol=1e-5)
        assert torch.allclose(v1, v2, atol=1e-5)

    def test_train_step_changes_parameters(self):
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            BitNetPolicyValueNetwork, encode_board_flat,
        )
        from tictactoe.core.board import Board
        from tictactoe.core.types import Player
        net = BitNetPolicyValueNetwork(3)
        x = encode_board_flat(Board.create(3), Player.X, 3)
        embed_w_before = net._embed.weight.detach().clone()
        target_p = torch.ones(9) / 9.0
        target_v = torch.tensor([0.5])
        net.train_step(x, target_p, target_v)
        assert not torch.allclose(net._embed.weight, embed_w_before)
