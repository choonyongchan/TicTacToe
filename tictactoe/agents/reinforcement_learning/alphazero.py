"""AlphaZero agent: full self-play + MCTS (PUCT) + training loop.

Uses a PolicyValueNetwork to guide MCTS with policy priors and value
estimates, collecting self-play data for supervised training.

Requires numpy; raises ImportError at instantiation otherwise.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import math
import random

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents._shared_utils import check_forced_move
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Board2D, Cell, Move, Player, Result


# ---------------------------------------------------------------------------
# PUCT Node
# ---------------------------------------------------------------------------

class PUCTNode:
    """Node in the AlphaZero MCTS tree."""

    def __init__(
        self,
        state: GameState,
        parent: PUCTNode | None = None,
        move: Move | None = None,
        prior: float = 0.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: list[PUCTNode] = []
        self.visits: int = 0
        self.value_sum: float = 0.0

    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def puct_score(self, c_puct: float = 1.0) -> float:
        if self.parent is None:
            return 0.0
        N_parent = max(self.parent.visits, 1)
        return self.q_value() + c_puct * self.prior * math.sqrt(N_parent) / (1 + self.visits)

    def is_terminal(self) -> bool:
        return self.state.result != Result.IN_PROGRESS

    def is_expanded(self) -> bool:
        return len(self.children) > 0


# ---------------------------------------------------------------------------
# AlphaZeroAgent
# ---------------------------------------------------------------------------

class AlphaZeroAgent(BaseAgent):
    """AlphaZero-style agent: PUCT MCTS guided by a PolicyValueNetwork.

    Attributes:
        n: Board dimension.
        num_simulations: MCTS simulations per move.
        c_puct: PUCT exploration constant.
        _net: PolicyValueNetwork for policy and value evaluation.
    """

    def __init__(
        self,
        n: int = 3,
        num_simulations: int | None = None,
        c_puct: float | None = None,
        temperature: float | None = None,
        lr: float | None = None,
        seed: int | None = None,
    ) -> None:
        if not _HAS_NUMPY:
            raise ImportError(
                "numpy is required for AlphaZeroAgent. Install it with: pip install numpy"
            )
        from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
        from tictactoe.config import get_config as _cfg, ConfigError as _CE
        try:
            _c = _cfg().rl
            self.num_simulations = num_simulations if num_simulations is not None \
                else _c.alphazero_simulations
            self.c_puct = c_puct if c_puct is not None else _c.alphazero_c_puct
            self.temperature = temperature if temperature is not None else _c.alphazero_temperature
            self.lr = lr if lr is not None else _c.alphazero_lr
        except _CE:
            self.num_simulations = num_simulations if num_simulations is not None else 50
            self.c_puct = c_puct if c_puct is not None else 1.0
            self.temperature = temperature if temperature is not None else 1.0
            self.lr = lr if lr is not None else 1e-3
        self.n = n
        self._net = PolicyValueNetwork(n)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def choose_move(self, state: GameState) -> Move:
        # Forced move check
        forced = check_forced_move(state)
        if forced is not None:
            state.nodes_visited = 1
            state.max_depth_reached = 0
            state.prunings = 0
            state.compute_ebf()
            return forced

        root_state = state.copy()
        root_state.result = Board.is_terminal(
            root_state.board, root_state.n, root_state.k, root_state.last_move
        )
        root = PUCTNode(root_state)
        self._expand(root)

        max_depth = 0
        for sim in range(self.num_simulations):
            node = root
            depth = 0

            # Selection
            while node.is_expanded() and not node.is_terminal():
                node = max(node.children, key=lambda c: c.puct_score(self.c_puct))
                depth += 1
            max_depth = max(max_depth, depth)

            # Expansion + Evaluation
            if not node.is_terminal():
                self._expand(node)
                # Evaluate leaf
                import numpy as np
                from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board_flat
                x = encode_board_flat(node.state.board, node.state.current_player, self.n)
                _, value = self._net.forward(x)
            else:
                value = self._terminal_value(node.state)

            # Backpropagation
            self._backpropagate(node, float(value))

        if not root.children:
            candidates = Board.get_candidate_moves(state, radius=2)
            move = candidates[0] if candidates else (0, 0)
        else:
            # Select most-visited child
            move = max(root.children, key=lambda c: c.visits).move

        state.nodes_visited = self.num_simulations
        state.max_depth_reached = max_depth
        state.prunings = 0
        state.compute_ebf()
        return move

    def get_name(self) -> str:
        return f"AlphaZero(n={self.n}, sims={self.num_simulations})"

    def get_tier(self) -> int:
        return 4

    # ------------------------------------------------------------------
    # MCTS helpers
    # ------------------------------------------------------------------

    def _expand(self, node: PUCTNode) -> None:
        """Expand node using network policy priors."""
        import numpy as np
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            encode_board_flat, softmax
        )
        if node.is_terminal() or node.is_expanded():
            return
        candidates = Board.get_candidate_moves(node.state, radius=2)
        if not candidates:
            return

        x = encode_board_flat(node.state.board, node.state.current_player, self.n)
        policy_logits, _ = self._net.forward(x)
        policy = softmax(policy_logits)

        n = self.n
        masked = np.zeros(n * n, dtype=np.float32)
        for r, c in candidates:
            masked[r * n + c] = policy[r * n + c]
        total = masked.sum()
        if total > 0:
            masked /= total
        else:
            for r, c in candidates:
                masked[r * n + c] = 1.0 / len(candidates)

        for move in candidates:
            r, c = move
            child_state = node.state.apply_move(move)
            child_state.result = Board.is_terminal(
                child_state.board, child_state.n, child_state.k, move
            )
            prior = float(masked[r * n + c])
            child = PUCTNode(child_state, parent=node, move=move, prior=prior)
            node.children.append(child)

    def _backpropagate(self, node: PUCTNode, value: float) -> None:
        """Backpropagate value up to root, flipping sign at each level."""
        current = node
        current_value = value
        while current is not None:
            current.visits += 1
            current.value_sum += current_value
            current_value = -current_value
            current = current.parent

    def _terminal_value(self, state: GameState) -> float:
        if state.result == Result.DRAW:
            return 0.0
        if state.result == Result.X_WINS:
            return 1.0 if state.current_player == Player.O else -1.0
        if state.result == Result.O_WINS:
            return 1.0 if state.current_player == Player.X else -1.0
        return 0.0

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def _generate_symmetries(
        self,
        board: Board2D,
        n: int,
        policy,
        value: float,
    ) -> list:
        """Generate all 8 dihedral symmetries of a board/policy pair.

        Args:
            board: Current board (n×n Cell grid).
            n: Board dimension.
            policy: Flat policy array of length n*n.
            value: Scalar value.

        Returns:
            List of 8 (encoded_state, policy_variant, value) tuples.
        """
        import numpy as np
        cell_to_int = {Cell.EMPTY: 0, Cell.X: 1, Cell.O: 2}
        board_arr = np.array([[cell_to_int[board[r][c]] for c in range(n)] for r in range(n)])
        policy_2d = np.asarray(policy).reshape(n, n)

        variants = []
        for rot in range(4):
            b_rot = np.rot90(board_arr, rot)
            p_rot = np.rot90(policy_2d, rot)
            b_enc = self._board_arr_to_flat(b_rot, n)
            variants.append((b_enc, p_rot.flatten(), value))
            b_flip = np.fliplr(b_rot)
            p_flip = np.fliplr(p_rot)
            b_enc_f = self._board_arr_to_flat(b_flip, n)
            variants.append((b_enc_f, p_flip.flatten(), value))

        return variants

    def _board_arr_to_flat(self, board_arr, n: int):
        """Convert an (n,n) int array to 3-channel flat encoding."""
        import numpy as np
        own = (board_arr == 1).astype(np.float32)
        opp = (board_arr == 2).astype(np.float32)
        const = np.ones((n, n), dtype=np.float32)
        return np.stack([own, opp, const]).flatten()

    def train_on_example(
        self,
        state_enc,
        target_policy,
        target_value: float,
    ) -> float:
        """Perform one supervised update step.

        Args:
            state_enc: Flat encoded state (numpy array).
            target_policy: Target policy distribution (sums to 1).
            target_value: Target value in [-1, 1].

        Returns:
            Combined loss (policy + value).
        """
        import numpy as np
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            softmax, cross_entropy_loss, mse_loss, relu
        )
        policy_logits, value = self._net.forward(state_enc)
        p_loss = cross_entropy_loss(policy_logits, target_policy)
        v_loss = mse_loss(np.array([value]), np.array([target_value]))
        total_loss = p_loss + v_loss

        policy_probs = softmax(policy_logits)
        dp = policy_probs - target_policy
        dv = np.array([2.0 * (value - target_value)])

        h1 = relu(state_enc @ self._net._w1 + self._net._b1)
        h2 = relu(h1 @ self._net._w2 + self._net._b2)

        self._net._wp -= self.lr * np.outer(h2, dp)
        self._net._bp -= self.lr * dp
        self._net._wv -= self.lr * np.outer(h2, dv)
        self._net._bv -= self.lr * dv

        return float(total_loss)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save network weights to path (no extension)."""
        self._net.save(path)

    def load(self, path: str) -> None:
        """Load network weights from path.npz."""
        self._net.load(path + '.npz')
