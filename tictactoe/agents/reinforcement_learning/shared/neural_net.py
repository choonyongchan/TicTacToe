"""Numpy-only neural network implementations for RL agents.

Provides PolicyValueNetwork (policy + value head) and QNetwork (action-value).
Raises ImportError with a helpful message if numpy is not installed.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None  # type: ignore[assignment]

from tictactoe.core.types import Board2D, Cell, Player


# ---------------------------------------------------------------------------
# Activation functions and losses
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit."""
    return np.maximum(0.0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def tanh_act(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation."""
    return np.tanh(x)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Cross-entropy loss between logits and probability targets."""
    probs = softmax(logits)
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(targets * np.log(probs)))


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error loss."""
    return float(np.mean((predictions - targets) ** 2))


# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

def encode_board(board: Board2D, current_player: Player, n: int) -> np.ndarray:
    """Encode board as 3×n×n float32 array (3-channel representation).

    Channel 0: 1.0 where current_player has a piece.
    Channel 1: 1.0 where the opponent has a piece.
    Channel 2: all 1.0 (constant plane marking board size).

    Args:
        board: The current board.
        current_player: The player whose turn it is.
        n: Board dimension.

    Returns:
        A (3, n, n) float32 numpy array.
    """
    own_cell = current_player.to_cell()
    opp_cell = current_player.opponent().to_cell()
    arr = np.zeros((3, n, n), dtype=np.float32)
    for r in range(n):
        for c in range(n):
            cell = board[r][c]
            if cell is own_cell:
                arr[0, r, c] = 1.0
            elif cell is opp_cell:
                arr[1, r, c] = 1.0
        arr[2, :, :] = 1.0
    return arr


def encode_board_flat(board: Board2D, current_player: Player, n: int) -> np.ndarray:
    """Encode board as flat 3*n*n float32 vector.

    Same channel layout as encode_board but flattened to 1-D.

    Args:
        board: The current board.
        current_player: The player whose turn it is.
        n: Board dimension.

    Returns:
        A (3*n*n,) float32 numpy array.
    """
    return encode_board(board, current_player, n).flatten()


# ---------------------------------------------------------------------------
# PolicyValueNetwork: 3n²→256→128→policy(n²)+value(1)
# ---------------------------------------------------------------------------

class PolicyValueNetwork:
    """Two-headed network: policy (softmax over n² moves) + value (tanh scalar).

    Architecture:
        Input:  3*n*n flat vector
        Hidden: Linear(3n², 256) → ReLU → Linear(256, 128) → ReLU
        Policy: Linear(128, n²)   (logits; apply softmax externally)
        Value:  Linear(128, 1)    → tanh

    Weights are initialised with He-normal initialisation.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        input_size = 3 * n * n
        self._build(input_size, n * n)

    def _build(self, input_size: int, output_size: int) -> None:
        rng = np.random.default_rng(seed=0)

        def he(fan_in: int, fan_out: int) -> np.ndarray:
            std = np.sqrt(2.0 / fan_in)
            return rng.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)

        self._w1 = he(input_size, 256)
        self._b1 = np.zeros(256, dtype=np.float32)
        self._w2 = he(256, 128)
        self._b2 = np.zeros(128, dtype=np.float32)
        # Policy head
        self._wp = he(128, output_size)
        self._bp = np.zeros(output_size, dtype=np.float32)
        # Value head
        self._wv = he(128, 1)
        self._bv = np.zeros(1, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward pass.

        Args:
            x: Flat input vector of length 3*n*n.

        Returns:
            (policy_logits, value) where policy_logits is length n*n and
            value is a scalar in [-1, 1].
        """
        h1 = relu(x @ self._w1 + self._b1)
        h2 = relu(h1 @ self._w2 + self._b2)
        policy_logits = h2 @ self._wp + self._bp
        value = float(tanh_act(h2 @ self._wv + self._bv)[0])
        return policy_logits, value

    def get_weights(self) -> list[np.ndarray]:
        """Return all weight arrays in order."""
        return [self._w1, self._b1, self._w2, self._b2,
                self._wp, self._bp, self._wv, self._bv]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Set all weight arrays from a list (same order as get_weights)."""
        (self._w1, self._b1, self._w2, self._b2,
         self._wp, self._bp, self._wv, self._bv) = weights

    def copy(self) -> PolicyValueNetwork:
        """Return a deep copy of this network."""
        net = PolicyValueNetwork.__new__(PolicyValueNetwork)
        net.n = self.n
        net._w1 = self._w1.copy()
        net._b1 = self._b1.copy()
        net._w2 = self._w2.copy()
        net._b2 = self._b2.copy()
        net._wp = self._wp.copy()
        net._bp = self._bp.copy()
        net._wv = self._wv.copy()
        net._bv = self._bv.copy()
        return net

    def save(self, path: str) -> None:
        """Save weights to a .npz file (path without extension)."""
        np.savez(
            path,
            w1=self._w1, b1=self._b1,
            w2=self._w2, b2=self._b2,
            wp=self._wp, bp=self._bp,
            wv=self._wv, bv=self._bv,
        )

    def load(self, path: str) -> None:
        """Load weights from a .npz file."""
        data = np.load(path)
        self._w1 = data['w1']
        self._b1 = data['b1']
        self._w2 = data['w2']
        self._b2 = data['b2']
        self._wp = data['wp']
        self._bp = data['bp']
        self._wv = data['wv']
        self._bv = data['bv']


# ---------------------------------------------------------------------------
# QNetwork: 3n²→128→64→n²
# ---------------------------------------------------------------------------

class QNetwork:
    """Q-network: maps board state to action-values for each cell.

    Architecture:
        Input:  3*n*n flat vector
        Hidden: Linear(3n², 128) → ReLU → Linear(128, 64) → ReLU
        Output: Linear(64, n²)  (Q-values for each cell)
    """

    def __init__(self, n: int) -> None:
        self.n = n
        input_size = 3 * n * n
        output_size = n * n
        rng = np.random.default_rng(seed=0)

        def he(fan_in: int, fan_out: int) -> np.ndarray:
            std = np.sqrt(2.0 / fan_in)
            return rng.normal(0.0, std, (fan_in, fan_out)).astype(np.float32)

        self._w1 = he(input_size, 128)
        self._b1 = np.zeros(128, dtype=np.float32)
        self._w2 = he(128, 64)
        self._b2 = np.zeros(64, dtype=np.float32)
        self._w3 = he(64, output_size)
        self._b3 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: Flat input vector of length 3*n*n.

        Returns:
            Q-value array of length n*n.
        """
        h1 = relu(x @ self._w1 + self._b1)
        h2 = relu(h1 @ self._w2 + self._b2)
        return h2 @ self._w3 + self._b3

    def get_weights(self) -> list[np.ndarray]:
        """Return all weight arrays in order."""
        return [self._w1, self._b1, self._w2, self._b2, self._w3, self._b3]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Set weights from list (same order as get_weights)."""
        self._w1, self._b1, self._w2, self._b2, self._w3, self._b3 = weights

    def copy(self) -> QNetwork:
        """Return a deep copy of this network."""
        net = QNetwork.__new__(QNetwork)
        net.n = self.n
        net._w1 = self._w1.copy()
        net._b1 = self._b1.copy()
        net._w2 = self._w2.copy()
        net._b2 = self._b2.copy()
        net._w3 = self._w3.copy()
        net._b3 = self._b3.copy()
        return net

    def save(self, path: str) -> None:
        """Save weights to a .npz file."""
        np.savez(path, w1=self._w1, b1=self._b1,
                 w2=self._w2, b2=self._b2,
                 w3=self._w3, b3=self._b3)

    def load(self, path: str) -> None:
        """Load weights from a .npz file."""
        data = np.load(path)
        self._w1 = data['w1']
        self._b1 = data['b1']
        self._w2 = data['w2']
        self._b2 = data['b2']
        self._w3 = data['w3']
        self._b3 = data['b3']
