"""Deep Q-Network (DQN) agent for n×n Tic-Tac-Toe.

Uses a QNetwork with experience replay and a target network updated
periodically. Inference masks illegal moves with -inf.

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
        epsilon: float = 0.1,
        buffer_capacity: int = 10_000,
        batch_size: int = 32,
        gamma: float = 0.95,
        lr: float = 1e-3,
        target_update_freq: int = 100,
        seed: int | None = None,
    ) -> None:
        if not _HAS_NUMPY:
            raise ImportError(
                "numpy is required for DQNAgent. Install it with: pip install numpy"
            )
        from tictactoe.agents.reinforcement_learning.shared.neural_net import QNetwork
        self.n = n
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.target_update_freq = target_update_freq
        self._steps = 0
        self._rng = random.Random(seed)
        self._online = QNetwork(n)
        self._target = self._online.copy()
        self._buffer = ReplayBuffer(buffer_capacity)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def choose_move(self, state: GameState) -> Move:
        import numpy as np
        from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board_flat
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
        q_vals = self._online.forward(x)

        # Mask illegal moves with -inf
        mask = np.full(state.n * state.n, -np.inf, dtype=np.float32)
        for r, c in empty_cells:
            mask[r * state.n + c] = q_vals[r * state.n + c]

        best_idx = int(np.argmax(mask))
        return (best_idx // state.n, best_idx % state.n)

    def get_name(self) -> str:
        return f"DQN(n={self.n}, eps={self.epsilon})"

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

        return loss

    def _update_weights(self, batch: list[Experience]) -> float:
        """Compute DQN targets and update online network via SGD on output layer."""
        import numpy as np
        from tictactoe.agents.reinforcement_learning.shared.neural_net import relu

        states = np.array([e.state_tensor for e in batch], dtype=np.float32)
        next_states = np.array([e.next_state_tensor for e in batch], dtype=np.float32)
        rewards = np.array([e.reward for e in batch], dtype=np.float32)
        dones = np.array([float(e.done) for e in batch], dtype=np.float32)
        actions = [e.action for e in batch]

        q_current_all = np.array([self._online.forward(states[i]) for i in range(len(batch))])
        q_next_all = np.array([self._target.forward(next_states[i]) for i in range(len(batch))])
        max_q_next = q_next_all.max(axis=1)
        targets = rewards + self.gamma * max_q_next * (1.0 - dones)

        q_targets = q_current_all.copy()
        for i, a in enumerate(actions):
            q_targets[i, a] = targets[i]

        loss = float(np.mean((q_current_all - q_targets) ** 2))
        delta = (q_current_all - q_targets)

        h2_all = np.array([
            relu(relu(states[i] @ self._online._w1 + self._online._b1) @ self._online._w2 + self._online._b2)
            for i in range(len(batch))
        ])
        dw3 = (h2_all.T @ delta) / len(batch)
        db3 = delta.mean(axis=0)
        self._online._w3 -= self.lr * dw3
        self._online._b3 -= self.lr * db3

        return loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save online network weights to path (no extension needed)."""
        self._online.save(path)

    def load(self, path: str) -> None:
        """Load online network weights from path.npz and sync target."""
        self._online.load(path + '.npz')
        self._target = self._online.copy()
