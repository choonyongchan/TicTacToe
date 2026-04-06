"""MCTS with neural network policy and value (AlphaZero-lite).

Implements a simplified version of the AlphaZero algorithm. Unlike vanilla
MCTS, there is NO rollout/simulation phase. Instead, leaf nodes are evaluated
directly by a neural network that provides:
    - A policy prior P(s, a): a probability distribution over legal moves.
    - A value estimate V(s): the expected outcome from state s.

The policy prior is used to bias PUCT (Polynomial Upper Confidence bounds
for Trees) selection toward moves the network considers promising, while the
value estimate replaces the random rollout.

PUCT selection formula (per child):
    PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N_parent) / (1 + N(s, a))

where:
    Q(s, a)  — average value of child (exploitation).
    P(s, a)  — network policy prior for action a from state s.
    N_parent — visit count of the parent node.
    N(s, a)  — visit count of child node.
    c_puct   — exploration constant (higher = more exploration).

Network availability:
    When a network is not provided (or numpy is unavailable), the agent falls
    back gracefully to uniform priors (P = 1/|legal_moves|) and value = 0.0.
    This allows the agent to run in any environment, at the cost of reduced
    playing strength.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import math
import time

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents._search_budget import SearchBudget
from tictactoe.agents._shared_utils import check_forced_move
from tictactoe.agents.monte_carlo.node import MCTSNode
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import INF, Move, Player, Result
from tictactoe.benchmark.metrics import MatchConfig


class AlphaZeroLiteNode(MCTSNode):
    """MCTSNode extended with a network policy prior for PUCT selection.

    Inherits all standard MCTS node fields. Adds the prior probability
    P(s, a) produced by the neural network for the move that led to this
    node, used in the PUCT selection formula.

    Attributes:
        prior: P(s, a) — the network's prior probability for the action
            that led to this node. Set to a uniform value when the network
            is unavailable.
    """

    def __init__(self, state: GameState, parent: AlphaZeroLiteNode | None = None,
                 move: Move | None = None, prior: float = 0.0) -> None:
        """Initialise an AlphaZero-lite MCTS node.

        Args:
            state: The game state at this node.
            parent: Parent node, or None for the root.
            move: The action that led to this node, or None for the root.
            prior: Network policy probability P(s, a) for this node's move.
                Defaults to 0.0 (updated when the parent is expanded).
        """
        super().__init__(state, parent, move)
        self.prior = prior  # P(s, a) from neural network

    def puct_score(self, c_puct: float = 1.0) -> float:
        """Compute the PUCT score for this node.

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N_parent) / (1 + N)

        where Q = average value, P = prior from the network, N = visit count
        of this child, and N_parent = visit count of the parent.

        Args:
            c_puct: Exploration constant. Higher values give more weight to
                the network prior relative to the Monte Carlo value.

        Returns:
            The PUCT score for this node. Returns 0.0 if there is no parent.
        """
        if self.parent is None:
            return 0.0
        N_parent = self.parent.visits
        N = self.visits
        Q = self.value / max(self.visits, 1)
        U = c_puct * self.prior * math.sqrt(N_parent) / (1 + N)
        return Q + U


class MCTSAlphaZeroLite(BaseAgent):
    """MCTS agent guided by a neural network policy and value function.

    Implements a lightweight AlphaZero-style agent:
    - No random rollouts. Leaf nodes are evaluated by a neural network.
    - PUCT selection biases exploration toward high-prior moves.
    - Expansion creates all children at once (full expansion), unlike vanilla
      MCTS which expands one child per simulation.
    - The final move is the child with the highest visit count.

    When no network is available (or numpy is missing), the agent operates
    with uniform priors and zero value estimates — equivalent to vanilla MCTS
    selection with a slight PUCT bias from the c_puct * prior term.

    Attributes:
        net: The policy-value network, or None if untrained/unavailable.
        c_puct: PUCT exploration constant.
        num_simulations: Hard cap on the number of MCTS iterations. The
            actual number may be lower if the time/node budget is smaller.
        match_config: Budget configuration (time/node/depth controlled).
        _trained: True if a network was provided at construction time.
    """

    def __init__(self, net=None, c_puct: float = 1.0,
                 num_simulations: int = 100,
                 match_config: MatchConfig | None = None) -> None:
        """Initialise the AlphaZero-lite agent.

        Args:
            net: A PolicyValueNetwork instance, or None to use uniform priors.
                The network is called lazily on the first choose_move call.
            c_puct: PUCT exploration constant. Higher values increase the
                influence of the network prior on selection.
            num_simulations: Maximum simulations per move (additional hard
                cap on top of the budget from match_config).
            match_config: Budget configuration. None defaults to TIME_CONTROLLED
                with a 1000 ms limit.
        """
        self.net = net
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.match_config = match_config
        self._trained = net is not None

    def _get_net(self, n: int):
        """Lazily initialise the network for board size n.

        Attempts to import and construct a PolicyValueNetwork if none was
        provided at construction. Falls back to None if the import fails.

        Args:
            n: Board dimension needed to configure the network input size.

        Returns:
            The network instance, or None if unavailable.
        """
        if self.net is not None:
            return self.net
        try:
            from tictactoe.agents.reinforcement_learning.shared.neural_net import PolicyValueNetwork
            self.net = PolicyValueNetwork(n)
            return self.net
        except (ImportError, Exception):
            return None

    def choose_move(self, state: GameState) -> Move:
        """Select the best move using AlphaZero-lite MCTS with PUCT.

        Runs MCTS simulations until both the simulation cap and the budget
        are exhausted, then returns the root child with the highest visit
        count.

        Args:
            state: The current game state.

        Returns:
            The chosen (row, col) move.
        """
        forced = check_forced_move(state)
        if forced is not None:
            state.nodes_visited = 1
            state.max_depth_reached = 0
            state.prunings = 0
            state.compute_ebf()
            return forced

        net = self._get_net(state.n)
        budget = SearchBudget(self.match_config, time.perf_counter_ns())

        root_state = state.copy()
        root_state.result = Board.is_terminal(
            root_state.board, root_state.n, root_state.k, root_state.last_move
        )
        root = AlphaZeroLiteNode(root_state)

        # Expand root with network priors (or uniform)
        root_value = self._expand_with_network(root, net, state.n)

        simulations = 0
        max_depth = 0

        while not budget.exhausted(simulations, 0) and simulations < self.num_simulations:
            node = root
            depth = 0

            # --- Selection via PUCT ---
            while node.children and not node.is_terminal():
                unvisited = [c for c in node.children if c.visits == 0]
                if unvisited:
                    node = unvisited[0]
                    depth += 1
                    break
                node = max(node.children, key=lambda c: c.puct_score(self.c_puct))
                depth += 1
            max_depth = max(max_depth, depth)

            # --- Evaluate leaf ---
            if node.is_terminal():
                # Terminal value from the mover's perspective
                if node.parent is not None:
                    mover = node.parent.state.current_player
                else:
                    mover = node.state.current_player
                value = self._terminal_value_for(node.state, mover)
                node.backpropagate(value)
            else:
                # Expand with network (creates children); returns value estimate
                value = self._expand_with_network(node, net, state.n)
                node.backpropagate(value)

            simulations += 1

        if not root.children:
            candidates = Board.get_candidate_moves(state, radius=2)
            move = candidates[0] if candidates else (0, 0)
        else:
            move = max(root.children, key=lambda c: c.visits).move

        state.nodes_visited = simulations
        state.max_depth_reached = max_depth
        state.prunings = 0
        state.compute_ebf()
        return move

    def _expand_with_network(self, node: AlphaZeroLiteNode, net, board_n: int) -> float:
        """Run network forward pass; populate node.children with priors; return value.

        If the network is unavailable, falls back to uniform priors and 0.0 value.
        Children are only created if node has no children yet (idempotent).
        """
        if node.children:
            # Already expanded
            return 0.0

        candidates = Board.get_candidate_moves(node.state, radius=2)
        if not candidates:
            return 0.0

        value = 0.0
        priors: dict[Move, float] = {}

        if net is not None:
            try:
                from tictactoe.agents.reinforcement_learning.shared.neural_net import (
                    encode_board_flat, softmax
                )
                import numpy as np

                x = encode_board_flat(node.state.board, node.state.current_player, board_n)
                policy_logits, net_value = net.forward(x)
                policy = softmax(policy_logits)
                value = float(net_value)

                n = board_n
                legal_set = set(candidates)

                # Mask and renormalize over legal moves
                masked: list[float] = []
                for i in range(n * n):
                    r, c = divmod(i, n)
                    masked.append(policy[i] if (r, c) in legal_set else 0.0)

                total = sum(masked)
                if total > 0:
                    masked = [p / total for p in masked]
                else:
                    uniform = 1.0 / len(candidates)
                    masked = [uniform if (divmod(i, n) in legal_set) else 0.0
                              for i in range(n * n)]

                for move in candidates:
                    r, c = move
                    priors[move] = masked[r * n + c]

            except (ImportError, Exception):
                # Fallback to uniform
                net = None

        if not priors:
            uniform = 1.0 / len(candidates)
            for move in candidates:
                priors[move] = uniform

        # Create child nodes
        for move in candidates:
            child_state = node.state.apply_move(move)
            child_state.result = Board.is_terminal(
                child_state.board, child_state.n, child_state.k, move
            )
            child = AlphaZeroLiteNode(
                child_state, parent=node, move=move, prior=priors.get(move, 0.0)
            )
            node.children.append(child)
        node.untried_moves.clear()

        return value

    def _terminal_value_for(self, state: GameState, player: Player) -> float:
        """Return the game outcome as a reward from a specific player's perspective.

        Args:
            state: A terminal game state.
            player: The player whose perspective determines the sign of the reward.

        Returns:
            +1.0 if player won, -1.0 if player lost, 0.0 for a draw.
        """
        if state.result == Result.DRAW:
            return 0.0
        if (state.result == Result.X_WINS and player == Player.X) or \
           (state.result == Result.O_WINS and player == Player.O):
            return 1.0
        return -1.0

    def get_name(self) -> str:
        """Return the agent's display name indicating whether a network is loaded.

        Returns:
            A string of the form "MCTS-AlphaZeroLite(trained=True/False)".
        """
        return f"MCTS-AlphaZeroLite(trained={self._trained})"

    def get_tier(self) -> int:
        """Return the benchmark tier for this agent.

        Returns:
            3 — Monte Carlo search.
        """
        return 3
