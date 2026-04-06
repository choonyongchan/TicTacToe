"""Tabular Q-learning agent for n=3 Tic-Tac-Toe.

Represents Q-values in a dictionary keyed by a base-3 integer hash of the
board state. Only supports n=3 (9 cells → 3^9 = 19683 states).

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import pickle
import random
from typing import Any

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Cell, Move, Player


class TabularQAgent(BaseAgent):
    """Q-learning agent using an explicit state → action → value table.

    Only supports n=3 (the state space is manageable: 3^9 = 19 683 states).
    Raises NotImplementedError for any other board size.

    Attributes:
        n: Board dimension (must be 3).
        epsilon: Exploration rate for epsilon-greedy inference.
        _q_table: Nested dict mapping state_hash → {action_idx: q_value}.
    """

    def __init__(
        self,
        n: int = 3,
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if n != 3:
            raise NotImplementedError(
                f"TabularQAgent only supports n=3, got n={n}. "
                "The state space is too large for tabular Q-learning on larger boards."
            )
        self.n = n
        self.epsilon = epsilon
        self._q_table: dict[int, dict[int, float]] = {}
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def choose_move(self, state: GameState) -> Move:
        if state.n != 3:
            raise NotImplementedError(
                f"TabularQAgent only supports n=3, got n={state.n}."
            )
        state.nodes_visited = 1
        state.max_depth_reached = 0
        state.prunings = 0
        state.compute_ebf()

        empty_cells = Board.get_all_empty_cells(state.board)
        if not empty_cells:
            return (0, 0)  # Fallback; should not happen in valid games

        # Epsilon-greedy: with probability epsilon, pick randomly
        if self._rng.random() < self.epsilon:
            return self._rng.choice(empty_cells)

        state_hash = self._hash_board(state.board)
        q_values = self._q_table.get(state_hash, {})

        if not q_values:
            return self._rng.choice(empty_cells)

        # Choose action with highest Q-value among legal moves
        best_action = max(
            empty_cells,
            key=lambda m: q_values.get(m[0] * 3 + m[1], 0.0),
        )
        return best_action

    def get_name(self) -> str:
        return f"TabularQ(n={self.n}, eps={self.epsilon})"

    def get_tier(self) -> int:
        return 4

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def update(
        self,
        state_hash: int,
        action: int,
        reward: float,
        next_state_hash: int,
        next_legal_actions: list[int],
        done: bool,
        alpha: float = 0.1,
        gamma: float = 0.9,
    ) -> None:
        """Perform one Q-learning update.

        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]

        Args:
            state_hash: Hash of the current state.
            action: Action index (row * n + col).
            reward: Reward received.
            next_state_hash: Hash of the next state.
            next_legal_actions: Legal action indices in the next state.
            done: Whether the episode ended.
            alpha: Learning rate.
            gamma: Discount factor.
        """
        if state_hash not in self._q_table:
            self._q_table[state_hash] = {}
        q_sa = self._q_table[state_hash].get(action, 0.0)

        if done or not next_legal_actions:
            target = reward
        else:
            next_q = self._q_table.get(next_state_hash, {})
            max_next = max(next_q.get(a, 0.0) for a in next_legal_actions)
            target = reward + gamma * max_next

        self._q_table[state_hash][action] = q_sa + alpha * (target - q_sa)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save Q-table to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self._q_table, f)

    def load(self, path: str) -> None:
        """Load Q-table from a pickle file."""
        with open(path, 'rb') as f:
            self._q_table = pickle.load(f)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_board(self, board: "list[list[Cell]]") -> int:
        """Encode the 3×3 board as a base-3 integer.

        Cell.EMPTY → 0, Cell.X → 1, Cell.O → 2.
        Cells are read in row-major order.
        """
        _cell_to_int = {Cell.EMPTY: 0, Cell.X: 1, Cell.O: 2}
        value = 0
        for row in board:
            for cell in row:
                value = value * 3 + _cell_to_int[cell]
        return value
