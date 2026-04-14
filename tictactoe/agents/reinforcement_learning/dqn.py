"""Deep Q-Network (DQN) agent for n×n Tic-Tac-Toe.

Uses a QNetwork with experience replay and a target network updated
periodically. Inference masks illegal moves with -inf.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents.reinforcement_learning.shared.replay_buffer import (
    ReplayBuffer, Experience
)
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Move, Player


class DQNAgent(BaseAgent):
    """DQN agent: epsilon-greedy Q-learning with experience replay.

    Attributes:
        n: Board dimension.
        epsilon: Current exploration rate.
        _online: Online Q-network.
        _target: Target Q-network (updated less frequently).
        _buffer: Experience replay buffer.
    """

    def __init__(
        self,
        n: int = 3,
        k: int | None = None,
        epsilon: float | None = None,
        buffer_capacity: int | None = None,
        batch_size: int | None = None,
        gamma: float | None = None,
        lr: float | None = None,
        target_update_freq: int | None = None,
        seed: int | None = None,
    ) -> None:
        from tictactoe.agents.reinforcement_learning.shared.neural_net import QNetwork
        from tictactoe.config import get_config as _cfg, ConfigError as _CE
        try:
            _c = _cfg().rl
            self.epsilon = epsilon if epsilon is not None else _c.dqn_epsilon
            _cap = buffer_capacity if buffer_capacity is not None else _c.dqn_buffer_capacity
            self.batch_size = batch_size if batch_size is not None else _c.dqn_batch_size
            self.gamma = gamma if gamma is not None else _c.dqn_gamma
            self.lr = lr if lr is not None else _c.dqn_lr
            self.target_update_freq = target_update_freq if target_update_freq is not None \
                else _c.dqn_target_update_freq
        except _CE:
            self.epsilon = epsilon if epsilon is not None else 0.1
            _cap = buffer_capacity if buffer_capacity is not None else 10_000
            self.batch_size = batch_size if batch_size is not None else 32
            self.gamma = gamma if gamma is not None else 0.95
            self.lr = lr if lr is not None else 1e-3
            self.target_update_freq = target_update_freq if target_update_freq is not None else 100
        self.n = n
        self.k = k if k is not None else n
        self._steps = 0
        self._rng = random.Random(seed)
        self._online = QNetwork(n, self.k)
        self._target = self._online.copy()
        self._buffer = ReplayBuffer(_cap)
        self._optimizer = torch.optim.Adam(self._online.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def choose_move(self, state: GameState) -> Move:
        from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board_flat, DEVICE
        state.nodes_visited = 1
        state.max_depth_reached = 0
        state.prunings = 0
        state.compute_ebf()

        empty_cells = Board.get_all_empty_cells(state.board)
        if not empty_cells:
            return (0, 0)

        # Epsilon-greedy
        if self._rng.random() < self.epsilon:
            return self._rng.choice(empty_cells)

        x = encode_board_flat(state.board, state.current_player, state.n)
        with torch.no_grad():
            q_vals = self._online.forward(x)

        # Mask illegal moves with -inf
        mask = torch.full((state.n * state.n,), float('-inf'), dtype=torch.float32, device=DEVICE)
        for r, c in empty_cells:
            mask[r * state.n + c] = q_vals[r * state.n + c]

        best_idx = int(torch.argmax(mask).item())
        return (best_idx // state.n, best_idx % state.n)

    def get_name(self) -> str:
        return f"DQN(n={self.n}, k={self.k}, eps={self.epsilon})"

    def get_tier(self) -> int:
        return 4

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def store(self, experience: Experience) -> None:
        """Push one experience to the replay buffer."""
        self._buffer.push(experience)

    def train_step(self) -> float | None:
        """Sample a mini-batch and perform one gradient update.

        Returns:
            The loss value, or None if the buffer is not ready.
        """
        if not self._buffer.is_ready(self.batch_size):
            return None

        batch = self._buffer.sample(self.batch_size)
        loss = self._update_weights(batch)
        self._steps += 1

        if self._steps % self.target_update_freq == 0:
            self._target = self._online.copy()
            # keep optimizer in sync with lr
            for pg in self._optimizer.param_groups:
                pg['lr'] = self.lr

        return loss

    def _update_weights(self, batch: list[Experience]) -> float:
        """Compute DQN targets and update online network via autograd."""
        from tictactoe.agents.reinforcement_learning.shared.neural_net import DEVICE

        states = torch.stack([
            e.state_tensor if isinstance(e.state_tensor, torch.Tensor)
            else torch.tensor(e.state_tensor, dtype=torch.float32)
            for e in batch
        ]).to(DEVICE)
        next_states = torch.stack([
            e.next_state_tensor if isinstance(e.next_state_tensor, torch.Tensor)
            else torch.tensor(e.next_state_tensor, dtype=torch.float32)
            for e in batch
        ]).to(DEVICE)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=DEVICE)
        dones = torch.tensor([float(e.done) for e in batch], dtype=torch.float32, device=DEVICE)
        actions = torch.tensor([e.action for e in batch], dtype=torch.long, device=DEVICE)

        # Compute targets using the frozen target network
        with torch.no_grad():
            q_next = torch.stack([self._target.forward(next_states[i]) for i in range(len(batch))])
            max_q_next = q_next.max(dim=1).values
            q_targets_sa = rewards + self.gamma * max_q_next * (1.0 - dones)

        # Predict Q-values for chosen actions
        q_pred_all = torch.stack([self._online.forward(states[i]) for i in range(len(batch))])
        q_pred_sa = q_pred_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(q_pred_sa, q_targets_sa)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save online network weights to path (no extension needed)."""
        self._online.save(path)

    def load(self, path: str) -> None:
        """Load online network weights from path.pt and sync target."""
        self._online.load(path)
        self._target = self._online.copy()
        self._optimizer = torch.optim.Adam(self._online.parameters(), lr=self.lr)
