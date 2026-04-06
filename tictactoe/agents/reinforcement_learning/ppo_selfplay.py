"""PPO (Proximal Policy Optimization) self-play agent for n×n Tic-Tac-Toe.

Uses a PolicyValueNetwork, computes GAE advantages, and applies the clipped
surrogate objective for policy updates.

Requires numpy; raises ImportError at instantiation otherwise.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import random

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

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
        epsilon_clip: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        if not _HAS_NUMPY:
            raise ImportError(
                "numpy is required for PPOSelfPlayAgent. Install it with: pip install numpy"
            )
        import numpy as np
        from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
        self.n = n
        self.epsilon_clip = epsilon_clip
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self._net = PolicyValueNetwork(n)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def choose_move(self, state: GameState) -> Move:
        import numpy as np
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            encode_board_flat, softmax
        )
        state.nodes_visited = 1
        state.max_depth_reached = 0
        state.prunings = 0
        state.compute_ebf()

        empty_cells = Board.get_all_empty_cells(state.board)
        if not empty_cells:
            return (0, 0)

        x = encode_board_flat(state.board, state.current_player, state.n)
        policy_logits, _ = self._net.forward(x)

        # Mask illegal moves: set logits for occupied cells to very negative
        n = state.n
        legal_set = {r * n + c for r, c in empty_cells}
        masked_logits = np.full(n * n, -1e9, dtype=np.float32)
        for idx in legal_set:
            masked_logits[idx] = policy_logits[idx]

        probs = softmax(masked_logits)

        # Sample from policy distribution
        action_idx = int(np.random.choice(n * n, p=probs))
        return (action_idx // n, action_idx % n)

    def get_name(self) -> str:
        return f"PPO-SelfPlay(n={self.n}, clip={self.epsilon_clip})"

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
        """Perform one PPO policy and value update step.

        Args:
            states: Encoded state vectors (numpy arrays).
            actions: Action indices chosen.
            old_log_probs: Log probabilities under the old policy.
            advantages: GAE advantage estimates.
            returns: Target returns for value function.

        Returns:
            The combined PPO loss value.
        """
        import numpy as np
        from tictactoe.agents.reinforcement_learning.shared.neural_net import softmax
        total_loss = 0.0
        for x, a, old_lp, adv, ret in zip(states, actions, old_log_probs, advantages, returns):
            policy_logits, value = self._net.forward(x)
            probs = softmax(policy_logits)
            log_prob = float(np.log(np.clip(probs[a], 1e-9, 1.0)))
            ratio = np.exp(log_prob - old_lp)
            clipped_ratio = np.clip(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip)
            policy_loss = -min(float(ratio * adv), float(clipped_ratio * adv))
            value_loss = (value - ret) ** 2
            total_loss += policy_loss + 0.5 * value_loss

        return total_loss / max(len(states), 1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save network weights to path (no extension)."""
        self._net.save(path)

    def load(self, path: str) -> None:
        """Load network weights from path.npz."""
        self._net.load(path + '.npz')
