"""Pure-numpy policy/value network used by MCPUCTAgent.

Forward-pass only: weights are randomly initialised and never trained. The
purpose is to demonstrate how PUCT blends a learned policy prior and value
estimate into tree search, not to produce a strong player.
"""

from __future__ import annotations

import numpy as np

from src.core.state import State


class PolicyValueNet:
    """Single-hidden-layer MLP mapping an encoded board to (policy, value).

    Weight shapes depend on the board size, which isn't known until the
    first state is encoded, so the weights are built lazily on first use
    instead of requiring n up front.

    Attributes:
        hidden: Hidden layer width.
        n: Board side length, set on first encode() call (None until then).
    """

    def __init__(self, hidden: int = 64, seed: int | None = None) -> None:
        self.hidden = hidden
        self.n: int | None = None
        self._rng = np.random.default_rng(seed)

    def _build(self, n: int) -> None:
        """Initialise weight matrices sized for an n x n board.

        Args:
            n: Board side length.
        """
        self.n = n
        in_dim = 2 * n * n
        self._w1 = self._rng.standard_normal((in_dim, self.hidden)) * np.sqrt(2.0 / in_dim)
        self._b1 = np.zeros(self.hidden)
        self._w_policy = self._rng.standard_normal((self.hidden, n * n)) * np.sqrt(2.0 / self.hidden)
        self._b_policy = np.zeros(n * n)
        self._w_value = self._rng.standard_normal((self.hidden, 1)) * np.sqrt(2.0 / self.hidden)
        self._b_value = np.zeros(1)

    def encode(self, state: State) -> np.ndarray:
        """Encode state as a flat (own-plane, opponent-plane) input vector.

        Args:
            state: Game state to encode.

        Returns:
            Float array of length 2*n*n: 1s where the current player has a
            piece, followed by 1s where the opponent has a piece.
        """
        if self.n is None:
            self._build(state.board.n)
        grid = state.board._grid
        me = int(state.current_player)
        opp = int(state.current_player.opponent())
        own = (grid == me).astype(np.float64).ravel()
        other = (grid == opp).astype(np.float64).ravel()
        return np.concatenate([own, other])

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Run the forward pass.

        Args:
            x: Encoded board, length 2*n*n (see encode()).

        Returns:
            (policy_logits, value): policy_logits has length n*n; value is a
            scalar in [-1, 1] from a tanh output, both from the perspective
            of the player who was "own" in the encoding.
        """
        hidden = np.maximum(0.0, x @ self._w1 + self._b1)
        policy_logits = hidden @ self._w_policy + self._b_policy
        value = float(np.tanh(hidden @ self._w_value + self._b_value)[0])
        return policy_logits, value
