"""AlphaZero agent: full self-play + MCTS (PUCT) + training loop.

Uses a PolicyValueNetwork to guide MCTS with policy priors and value
estimates, collecting self-play data for supervised training.

Optimisations implemented:
- Dirichlet noise at the root for exploration diversity.
- Temperature annealing: high temperature for early moves, low for endgame.
- MCTS tree reuse: the subtree rooted at the chosen child is kept for
  the next call, preserving visit statistics.
- Mini-batch SGD via ``train_on_batch()`` for efficient replay training.

Requires numpy.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents._search_budget import SearchBudget
from tictactoe.agents._shared_utils import check_forced_move
from tictactoe.benchmark.metrics import MatchConfig
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
    """AlphaZero-style agent: PUCT MCTS guided by a neural network.

    Supports multiple network architectures for memory efficiency:
    - ``"quantized"``: 8-bit quantized FC network (~4× memory reduction).
    - ``"ternary"``: Ternary FC network (~16× memory reduction).
    - ``"quantized_large"``: 10-block int8 conv residual network (~2.9 MB at n=3).
    - ``"ternary_large"``: 20-block ternary conv residual network (~22.5 MB at n=3),
      matching the original AlphaZero depth.
    - ``"ternary_bitnet"``: 4-layer BitNet 1.58-bit Transformer (d_model=64),
      W1.58A8 quantization, RoPE, squared ReLU, SubLN, no biases, STE training.
    - ``"ternary_bitnet_large"``: 12-layer BitNet 1.58-bit Transformer (d_model=256),
      same BitNet characteristics at AlphaZero scale.
    - ``"default"`` / ``"float32"``: 4-block float32 conv residual network (~1.1 MB).

    Attributes:
        n: Board dimension.
        k: Winning-run-length (3 ≤ k ≤ n). Passed to conv networks as a
            4th encoding channel so the network can learn k-conditioned
            strategies.
        num_simulations: MCTS simulations per move (upper bound; the
            match_config budget may stop the search earlier).
        c_puct: PUCT exploration constant.
        temperature: Sampling temperature for the first ``_TEMP_MOVES``
            moves; annealed toward ``_TEMP_FLOOR`` thereafter.
        match_config: Budget configuration (time/node/depth controlled).
        _net: Neural network for policy and value evaluation.
        _trained: True when a pre-built network was injected at construction.
        _cached_root: Subtree kept from the previous call for tree reuse.
    """

    # Temperature annealing parameters
    _TEMP_MOVES: int = 10        # Use self.temperature for moves 1-10
    _TEMP_FLOOR: float = 0.1     # Minimum temperature after annealing

    def __init__(
        self,
        n: int = 3,
        k: int | None = None,
        num_simulations: int | None = None,
        c_puct: float | None = None,
        temperature: float | None = None,
        lr: float | None = None,
        seed: int | None = None,
        match_config: MatchConfig | None = None,
        net=None,
        network_type: str = "quantized",
    ) -> None:
        """Initialise AlphaZeroAgent.

        Args:
            n: Board dimension.
            k: Winning-run-length (3 ≤ k ≤ n). Defaults to n (standard rules).
                Passed to conv networks as a 4th encoding channel.
            num_simulations: Maximum MCTS simulations per move.
            c_puct: PUCT exploration constant (higher = more exploration).
            temperature: Initial sampling temperature. High values explore more
                in the opening; annealed toward _TEMP_FLOOR after _TEMP_MOVES moves.
            lr: Learning rate for network weight updates.
            seed: Random seed for reproducibility.
            match_config: Budget configuration (time, node, or depth controlled).
            net: Pre-built network to inject (skips network_type selection).
            network_type: Architecture selector. One of:
                ``"quantized"``, ``"ternary"``, ``"quantized_large"``,
                ``"ternary_large"``, ``"ternary_bitnet"``,
                ``"ternary_bitnet_large"``, ``"default"`` / ``"float32"``.
        """
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            QuantizedPolicyValueNetwork,
            TernaryPolicyValueNetwork,
            PolicyValueNetwork,
            LargeQuantizedPolicyValueNetwork,
            LargeTernaryPolicyValueNetwork,
            TernaryBitNetPolicyValueNetwork,
            LargeTernaryBitNetPolicyValueNetwork,
        )
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
        self.k = k if k is not None else n
        self.match_config = match_config
        self._cached_root: PUCTNode | None = None

        # Select network type
        if net is not None:
            self._net = net
            self._network_type = "injected"
        elif network_type in ("default", "float32"):
            self._net = PolicyValueNetwork(n, k=self.k)
            self._network_type = "float32"
        elif network_type == "quantized":
            self._net = QuantizedPolicyValueNetwork(n, k=self.k)
            self._network_type = "quantized"
        elif network_type == "ternary":
            self._net = TernaryPolicyValueNetwork(n, k=self.k)
            self._network_type = "ternary"
        elif network_type == "quantized_large":
            self._net = LargeQuantizedPolicyValueNetwork(n, k=self.k)
            self._network_type = "quantized_large"
        elif network_type == "ternary_large":
            self._net = LargeTernaryPolicyValueNetwork(n, k=self.k)
            self._network_type = "ternary_large"
        elif network_type == "ternary_bitnet":
            self._net = TernaryBitNetPolicyValueNetwork(n, k=self.k)
            self._network_type = "ternary_bitnet"
        elif network_type == "ternary_bitnet_large":
            self._net = LargeTernaryBitNetPolicyValueNetwork(n, k=self.k)
            self._network_type = "ternary_bitnet_large"
        else:
            raise ValueError(
                f"Unknown network_type: {network_type!r}. "
                "Valid options: 'quantized', 'ternary', 'quantized_large', "
                "'ternary_large', 'ternary_bitnet', 'ternary_bitnet_large', "
                "'default', 'float32'."
            )

        self._trained = net is not None
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def choose_move(self, state: GameState) -> Move:
        """Select a move using PUCT MCTS guided by the neural network.

        Applies Dirichlet noise at the root for exploration, temperature
        annealing for move selection, and tree reuse to preserve visit
        statistics from the previous call.

        Args:
            state: Current game state. Instrumentation fields
                (nodes_visited, max_depth_reached, etc.) are populated
                before returning.

        Returns:
            The chosen (row, col) move.
        """
        # Forced move check
        forced = check_forced_move(state)
        if forced is not None:
            state.nodes_visited = 1
            state.max_depth_reached = 0
            state.prunings = 0
            state.compute_ebf()
            self._cached_root = None
            return forced

        from tictactoe.agents.reinforcement_learning.shared.neural_net import encode_board_flat

        # Tree reuse: attempt to reuse a cached subtree
        root = self._find_reusable_root(state.last_move)
        if root is None:
            root_state = state.copy()
            root_state.result = Board.is_terminal(
                root_state.board, root_state.n, root_state.k, root_state.last_move
            )
            root = PUCTNode(root_state)
            self._expand(root)

        # Apply Dirichlet noise to root priors
        if root.children:
            priors = np.array([c.prior for c in root.children], dtype=np.float32)
            noisy = self._add_dirichlet_noise(priors, self._np_rng)
            for child, new_prior in zip(root.children, noisy):
                child.prior = float(new_prior)

        budget = SearchBudget(self.match_config, time.perf_counter_ns())
        max_depth = 0
        sim = 0
        while sim < self.num_simulations and not budget.exhausted(sim, 0):
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
                x = encode_board_flat(node.state.board, node.state.current_player, self.n)
                _, value = self._net.forward(x)
            else:
                value = self._terminal_value(node.state)

            # Backpropagation
            self._backpropagate(node, float(value))
            sim += 1

        if not root.children:
            candidates = Board.get_candidate_moves(state, radius=2)
            move = candidates[0] if candidates else (0, 0)
            self._cached_root = None
        else:
            # Temperature-annealed move selection
            temp = self._effective_temperature(state.move_number)
            if temp < 1e-3:
                best_child = max(root.children, key=lambda c: c.visits)
            else:
                visits = np.array([c.visits for c in root.children], dtype=np.float32)
                visits_t = visits ** (1.0 / temp)
                total = visits_t.sum()
                probs = visits_t / total if total > 0 else visits_t
                idx = int(self._np_rng.choice(len(root.children), p=probs))
                best_child = root.children[idx]
            move = best_child.move

            # Cache the selected subtree for tree reuse on the next call
            best_child.parent = None
            self._cached_root = best_child

        state.nodes_visited = sim
        state.max_depth_reached = max_depth
        state.prunings = 0
        state.compute_ebf()
        return move

    def get_name(self) -> str:
        return f"AlphaZero-{self._network_type}(n={self.n}, sims={self.num_simulations})"

    def get_tier(self) -> int:
        return 4

    # ------------------------------------------------------------------
    # MCTS helpers
    # ------------------------------------------------------------------

    def _expand(self, node: PUCTNode) -> None:
        """Expand node using network policy priors."""
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
    # Optimisation helpers
    # ------------------------------------------------------------------

    def _add_dirichlet_noise(
        self,
        priors: np.ndarray,
        rng: np.random.Generator,
        epsilon: float = 0.25,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """Mix network priors with Dirichlet noise at the root node.

        Standard AlphaZero exploration technique. Adds noise drawn from a
        Dirichlet distribution to the root prior, encouraging the agent to
        explore moves that the network initially undervalues.

        Args:
            priors: Network policy distribution over legal moves (length m).
            rng: Numpy random generator (seeded via self._np_rng).
            epsilon: Noise mixture weight (0 = no noise, 1 = pure noise).
                AlphaZero uses 0.25.
            alpha: Dirichlet concentration parameter. Lower values produce
                more peaked noise. AlphaZero uses 0.3 for chess/Go.

        Returns:
            Mixed prior distribution of the same length as ``priors``.
        """
        if len(priors) == 0:
            return priors
        noise = rng.dirichlet(alpha=np.full(len(priors), alpha))
        return (1.0 - epsilon) * priors + epsilon * noise

    def _effective_temperature(self, move_number: int) -> float:
        """Compute the effective sampling temperature for the current move.

        Uses ``self.temperature`` for moves 1–``_TEMP_MOVES``, then linearly
        decays toward ``_TEMP_FLOOR`` for later moves to encourage exploitation
        over exploration in endgames.

        Args:
            move_number: 1-indexed move number from ``state.move_number``.
                Move 0 (before the first move) is treated as move 1.

        Returns:
            Effective temperature scalar in [``_TEMP_FLOOR``, ``self.temperature``].
        """
        move = max(move_number, 1)
        if move <= self._TEMP_MOVES:
            return self.temperature
        # Linear decay from self.temperature toward _TEMP_FLOOR
        progress = min((move - self._TEMP_MOVES) / self._TEMP_MOVES, 1.0)
        return max(self._TEMP_FLOOR, self.temperature + progress * (self._TEMP_FLOOR - self.temperature))

    def _find_reusable_root(
        self,
        last_move: tuple[int, int] | None,
    ) -> PUCTNode | None:
        """Find a cached subtree matching the opponent's last move.

        After each ``choose_move`` call, the selected child node is cached as
        ``self._cached_root``. On the next call, this method searches the
        cached root's children for a node whose move matches the opponent's
        last move, promoting it to root and preserving its visit statistics.

        Args:
            last_move: The (row, col) of the opponent's last move, or None
                if this is the first move of the game.

        Returns:
            The reusable ``PUCTNode`` (detached from its parent), or None if
            no match is found (caller must build a fresh root).
        """
        if self._cached_root is None or last_move is None:
            return None
        for child in self._cached_root.children:
            if child.move == last_move:
                child.parent = None
                return child
        return None

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def train_on_example(
        self,
        state_enc,
        target_policy,
        target_value: float,
    ) -> float:
        """Perform one supervised update step.

        Computes policy and value losses, derives output-head gradients, then
        delegates the weight update to the network's own ``backward()`` method.
        This keeps ``AlphaZeroAgent`` decoupled from the network's internal
        storage format (float32, int8-quantized, or ternary).

        Args:
            state_enc: Flat encoded state (numpy array of length 3*n*n).
            target_policy: Target policy distribution (sums to 1, length n*n).
            target_value: Target value in [-1, 1].

        Returns:
            Combined loss (policy cross-entropy + value MSE).
        """
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            softmax, cross_entropy_loss, mse_loss,
        )
        policy_logits, value = self._net.forward(state_enc)
        p_loss = cross_entropy_loss(policy_logits, target_policy)
        v_loss = mse_loss(np.array([value]), np.array([target_value]))
        total_loss = p_loss + v_loss

        dp = softmax(policy_logits) - target_policy
        dv = np.array([2.0 * (value - target_value)])
        self._net.backward(state_enc, dp, dv, self.lr)

        return float(total_loss)

    def train_on_batch(
        self,
        examples: list[tuple[np.ndarray, np.ndarray, float]],
        lr: float | None = None,
    ) -> float:
        """Train on a mini-batch of (state, policy, value) examples.

        Computes policy and value losses for each example, derives output-head
        gradients, then delegates one batched weight update to the network's
        ``backward_batch()`` method. Mini-batch training is more sample-efficient
        than single-example updates because gradients are averaged across the
        batch before the weight update, reducing gradient noise.

        Args:
            examples: List of (state_enc, target_policy, target_value) tuples.
                state_enc: Flat encoded state (numpy array of length 3*n*n).
                target_policy: Target distribution (sums to 1, length n*n).
                target_value: Target value in [-1, 1].
            lr: Learning rate. Defaults to ``self.lr``.

        Returns:
            Mean combined loss (policy cross-entropy + value MSE) over the
            batch.

        Raises:
            ValueError: If ``examples`` is empty.
        """
        if not examples:
            raise ValueError("examples must be non-empty.")
        from tictactoe.agents.reinforcement_learning.shared.neural_net import (
            softmax, cross_entropy_loss, mse_loss,
        )
        learning_rate = lr if lr is not None else self.lr
        batch = []
        total_loss = 0.0
        for state_enc, target_policy, target_value in examples:
            policy_logits, value = self._net.forward(state_enc)
            p_loss = cross_entropy_loss(policy_logits, target_policy)
            v_loss = mse_loss(np.array([value]), np.array([target_value]))
            total_loss += p_loss + v_loss

            dp = softmax(policy_logits) - target_policy
            dv = np.array([2.0 * (value - target_value)])
            batch.append((state_enc, dp, dv))

        self._net.backward_batch(batch, learning_rate)
        return float(total_loss) / len(examples)

    @staticmethod
    def _generate_symmetries(
        board: Board2D,
        n: int,
        policy,
        value: float,
    ) -> list:
        """Generate all 8 dihedral symmetries of a board/policy pair.

        The dihedral group D4 has 8 elements: 4 rotations × 2 reflections.
        Augmenting training data with these symmetries is free for square boards
        and significantly improves sample efficiency.

        Args:
            board: Current board (n×n Cell grid).
            n: Board dimension.
            policy: Flat policy array of length n*n.
            value: Scalar value target.

        Returns:
            List of 8 (encoded_state, policy_variant, value) tuples.
        """
        cell_to_int = {Cell.EMPTY: 0, Cell.X: 1, Cell.O: 2}
        board_arr = np.array([[cell_to_int[board[r][c]] for c in range(n)] for r in range(n)])
        policy_2d = np.asarray(policy).reshape(n, n)

        variants = []
        for rot in range(4):
            b_rot = np.rot90(board_arr, rot)
            p_rot = np.rot90(policy_2d, rot)
            b_enc = AlphaZeroAgent._board_arr_to_flat(b_rot, n)
            variants.append((b_enc, p_rot.flatten(), value))
            b_flip = np.fliplr(b_rot)
            p_flip = np.fliplr(p_rot)
            b_enc_f = AlphaZeroAgent._board_arr_to_flat(b_flip, n)
            variants.append((b_enc_f, p_flip.flatten(), value))

        return variants

    @staticmethod
    def _board_arr_to_flat(board_arr: np.ndarray, n: int) -> np.ndarray:
        """Convert an (n, n) int array to a 3-channel flat encoding.

        Channel 0: own pieces (value==1), Channel 1: opponent pieces (value==2),
        Channel 2: all ones (constant plane). Matches ``encode_board_flat`` layout.

        Args:
            board_arr: Integer array of shape (n, n) with values in {0, 1, 2}.
            n: Board dimension (used to build the constant plane).

        Returns:
            Flat float32 array of length 3*n*n.
        """
        own = (board_arr == 1).astype(np.float32)
        opp = (board_arr == 2).astype(np.float32)
        const = np.ones((n, n), dtype=np.float32)
        return np.stack([own, opp, const]).flatten()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save network weights to path (no extension)."""
        self._net.save(path)

    def load(self, path: str) -> None:
        """Load network weights from path.npz."""
        self._net.load(path + '.npz')
