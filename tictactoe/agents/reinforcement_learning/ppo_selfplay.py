"""PPO (Proximal Policy Optimization) self-play agent for n×n Tic-Tac-Toe.

Uses a PolicyValueNetwork, computes GAE advantages, and applies the clipped
surrogate objective for policy updates.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Move, Player


class PPOSelfPlayAgent(BaseAgent):
    """PPO agent trained via self-play.

    At inference time, the policy is evaluated by the PolicyValueNetwork and
    illegal moves are masked out before sampling.

    Attributes:
        n: Board dimension.
        _net: PolicyValueNetwork.
        epsilon_clip: PPO clipping parameter.
        gamma: Discount factor.
        lam: GAE lambda.
    """

    def __init__(
        self,
        n: int = 3,
        k: int | None = None,
        epsilon_clip: float | None = None,
        gamma: float | None = None,
        lam: float | None = None,
        lr: float | None = None,
        seed: int | None = None,
    ) -> None:
        from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
        from tictactoe.config import get_config as _cfg, ConfigError as _CE
        try:
            _c = _cfg().rl
            self.epsilon_clip = epsilon_clip if epsilon_clip is not None else _c.ppo_epsilon_clip
            self.gamma = gamma if gamma is not None else _c.ppo_gamma
            self.lam = lam if lam is not None else _c.ppo_lam
            self.lr = lr if lr is not None else _c.ppo_lr
        except _CE:
            self.epsilon_clip = epsilon_clip if epsilon_clip is not None else 0.2
            self.gamma = gamma if gamma is not None else 0.99
            self.lam = lam if lam is not None else 0.95
            self.lr = lr if lr is not None else 1e-3
        self.n = n
        self.k = k if k is not None else n
        self._net = PolicyValueNetwork(n, k=self.k)
        self._rng = random.Random(seed)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def choose_move(self, state: GameState) -> Move:
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            encode_board_flat, DEVICE
        )
        state.nodes_visited = 1
        state.max_depth_reached = 0
        state.prunings = 0
        state.compute_ebf()

        empty_cells = Board.get_all_empty_cells(state.board)
        if not empty_cells:
            return (0, 0)

        x = encode_board_flat(state.board, state.current_player, state.n)
        with torch.no_grad():
            policy_logits, _ = self._net.forward(x)

        # Mask illegal moves: set logits for occupied cells to very negative
        n = state.n
        masked_logits = torch.full((n * n,), -1e9, dtype=torch.float32, device=DEVICE)
        for r, c in empty_cells:
            masked_logits[r * n + c] = policy_logits[r * n + c]

        probs = F.softmax(masked_logits, dim=-1)

        # Sample from policy distribution
        action_idx = int(torch.multinomial(probs, 1).item())
        return (action_idx // n, action_idx % n)

    def get_name(self) -> str:
        return f"PPO-SelfPlay(n={self.n}, k={self.k}, clip={self.epsilon_clip})"

    def get_tier(self) -> int:
        return 4

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
    ) -> tuple[list[float], list[float]]:
        """Compute GAE advantages and returns.

        Args:
            rewards: Per-step rewards.
            values: Value estimates from the network.
            dones: Whether each step was terminal.

        Returns:
            (advantages, returns) as lists of floats.
        """
        advantages = []
        gae = 0.0
        values_extended = values + [0.0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_extended[t + 1] * (1.0 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.lam * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def ppo_update(
        self,
        states: list,
        actions: list[int],
        old_log_probs: list[float],
        advantages: list[float],
        returns: list[float],
    ) -> float:
        """Perform one PPO policy and value update step via autograd.

        Args:
            states: Encoded state vectors (tensors or arrays).
            actions: Action indices chosen.
            old_log_probs: Log probabilities under the old policy.
            advantages: GAE advantage estimates.
            returns: Target returns for value function.

        Returns:
            The combined PPO loss value.
        """
        from tictactoe.agents.reinforcement_learning.shared.neural_net import DEVICE

        old_lp_t = torch.tensor(old_log_probs, dtype=torch.float32, device=DEVICE)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        self._optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=DEVICE)

        for i, (x, a) in enumerate(zip(states, actions)):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            policy_logits, value = self._net.forward(x)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            log_prob = log_probs[a]
            ratio = torch.exp(log_prob - old_lp_t[i])
            clipped = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip)
            policy_loss = -torch.min(ratio * adv_t[i], clipped * adv_t[i])
            value_loss = F.mse_loss(value.reshape(1), ret_t[i].reshape(1))
            total_loss = total_loss + policy_loss + 0.5 * value_loss

        mean_loss = total_loss / max(len(states), 1)
        mean_loss.backward()
        self._optimizer.step()

        return mean_loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save network weights to path (no extension)."""
        self._net.save(path)

    def load(self, path: str) -> None:
        """Load network weights from path.pt."""
        self._net.load(path)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
