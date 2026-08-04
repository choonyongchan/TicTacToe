"""Tests for tictactoe/agents/mc_policy_value_net.py."""

from __future__ import annotations

import numpy as np

from tictactoe.agents.mc_policy_value_net import PolicyValueNet
from tests.test_helper import fresh_state, state_with_moves


class TestLazyBuild:
    def test_n_is_none_until_first_encode(self):
        net = PolicyValueNet(seed=0)
        assert net.n is None
        net.encode(fresh_state(n=4, k=3))
        assert net.n == 4


class TestEncode:
    def test_shape_and_planes(self):
        net = PolicyValueNet(seed=0)
        state = state_with_moves([(0, 0), (1, 1)], n=3, k=3)  # X at (0,0), O at (1,1)

        x = net.encode(state)  # X to move next: "own" plane is X's pieces

        assert x.shape == (2 * 3 * 3,)
        own_plane = x[: 9].reshape(3, 3)
        opp_plane = x[9:].reshape(3, 3)
        assert own_plane[0, 0] == 1.0
        assert own_plane.sum() == 1.0
        assert opp_plane[1, 1] == 1.0
        assert opp_plane.sum() == 1.0


class TestForward:
    def test_output_shapes_and_ranges(self):
        net = PolicyValueNet(seed=0)
        state = fresh_state(n=3, k=3)
        x = net.encode(state)

        policy_logits, value = net.forward(x)

        assert policy_logits.shape == (9,)
        assert -1.0 <= value <= 1.0

    def test_deterministic_given_seed(self):
        state = fresh_state(n=3, k=3)
        net_a = PolicyValueNet(seed=42)
        net_b = PolicyValueNet(seed=42)

        policy_a, value_a = net_a.forward(net_a.encode(state))
        policy_b, value_b = net_b.forward(net_b.encode(state))

        assert np.allclose(policy_a, policy_b)
        assert value_a == value_b
