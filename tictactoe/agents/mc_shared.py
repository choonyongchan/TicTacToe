"""Shared building blocks for Monte Carlo search agents (mc_*).

Value convention used by every tree-based mc_ agent: a node's value_sum is
the cumulative reward from the perspective of the player who chose to make
that move, i.e. node.parent.player (equivalently node.player.opponent()).
That is exactly the player deciding between node and its siblings, so
selection always does a plain argmax regardless of whose turn it is.
"""

from __future__ import annotations

import math
import random

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.core.state import State
from tictactoe.core.types import Player


class MCNode:
    """Node in a Monte Carlo search tree.

    Attributes:
        parent: Parent node, or None for the root.
        move: Move applied to the parent's state to reach this node, or None
            for the root.
        player: Player to move from this node's position.
        children: Expanded child nodes.
        untried_moves: Legal moves not yet expanded into children. Unused
            (left empty) by agents that expand all children at once.
        visits: Number of backpropagations through this node.
        value_sum: Cumulative reward from the perspective of node.parent.player.
    """

    def __init__(
        self,
        parent: MCNode | None,
        move: tuple[int, int] | None,
        player: Player,
        untried_moves: list[tuple[int, int]],
    ) -> None:
        self.parent = parent
        self.move = move
        self.player = player
        self.children: list[MCNode] = []
        self.untried_moves = untried_moves
        self.visits = 0
        self.value_sum = 0.0

    def is_fully_expanded(self) -> bool:
        """Return True if every legal move from this node has a child."""
        return not self.untried_moves

    def q(self) -> float:
        """Return the average reward for this node, or 0.0 if unvisited."""
        return self.value_sum / self.visits if self.visits else 0.0

    def uct_score(self, c: float) -> float:
        """Return the UCB1 (UCT) selection score.

        Args:
            c: Exploration constant.

        Returns:
            math.inf for an unvisited node (always selected first), otherwise
            q() + c * sqrt(ln(parent.visits) / visits).
        """
        if self.visits == 0:
            return math.inf
        return self.q() + c * math.sqrt(math.log(self.parent.visits) / self.visits)


class MCPUCTNode(MCNode):
    """MCNode with a policy-network prior, used by MCPUCTAgent.

    Attributes:
        prior: Policy network prior for the move leading to this node.
    """

    def __init__(
        self,
        parent: MCNode | None,
        move: tuple[int, int] | None,
        player: Player,
        untried_moves: list[tuple[int, int]],
    ) -> None:
        super().__init__(parent, move, player, untried_moves)
        self.prior = 0.0


class MCRaveNode(MCNode):
    """MCNode with per-move AMAF tables, used by MCRaveAgent.

    Attributes:
        amaf_visits: Per-move AMAF visit counts.
        amaf_values: Per-move AMAF cumulative reward.
    """

    def __init__(
        self,
        parent: MCNode | None,
        move: tuple[int, int] | None,
        player: Player,
        untried_moves: list[tuple[int, int]],
    ) -> None:
        super().__init__(parent, move, player, untried_moves)
        self.amaf_visits: dict[tuple[int, int], int] = {}
        self.amaf_values: dict[tuple[int, int], float] = {}


class MonteCarloAgent(BaseAgent):
    """Base class for Monte Carlo search agents.

    Provides terminal-state reward scoring and uniform-random rollouts that
    mutate a single shared State via apply()/undo() rather than cloning it,
    matching how the minimax/negamax agents in this codebase walk the tree.

    Attributes:
        n_simulations: Number of simulations run per act() call.
        _node_cls: Node class used to build the search tree. Subclasses that
            need extra per-node state (MCPUCTAgent, MCRaveAgent) override
            this with their own MCNode subclass.
    """

    _node_cls: type[MCNode] = MCNode

    def __init__(self, name: str, n_simulations: int, seed: int | None = None) -> None:
        super().__init__(name)
        self.n_simulations = n_simulations
        self._rng = random.Random(seed)

    def _reward(self, state: State, mover: Player) -> float:
        """Return the terminal outcome of state from mover's perspective.

        Args:
            state: Terminal game state (is_terminal() must be True).
            mover: Player whose perspective determines the sign of the reward.

        Returns:
            +1.0 if mover won, -1.0 if mover lost, 0.0 for a draw.
        """
        winner = state.winner()
        if winner is None:
            return 0.0
        return 1.0 if winner is mover else -1.0

    def _rollout_move(self, state: State) -> tuple[int, int]:
        """Choose the next move during a rollout. Default: uniform random.

        Args:
            state: Current (non-terminal) game state.

        Returns:
            (row, col) of the chosen move.
        """
        return self._rng.choice(state.board.get_empty_cells())

    def _random_rollout(
        self, state: State, mover: Player
    ) -> tuple[float, list[tuple[Player, tuple[int, int]]]]:
        """Play state to completion via _rollout_move, then undo every move played.

        Args:
            state: State to roll out from; mutated and fully restored in place.
            mover: Player whose perspective determines the reward sign.

        Returns:
            (reward, moves_played): reward is mover's terminal outcome;
            moves_played is the ordered (player, move) list applied during
            the rollout, used by MCRaveAgent for AMAF updates.
        """
        moves_played: list[tuple[Player, tuple[int, int]]] = []
        while not state.is_terminal():
            player = state.current_player
            move = self._rollout_move(state)
            state.apply(*move)
            moves_played.append((player, move))
        reward = self._reward(state, mover)
        for _ in moves_played:
            state.undo()
        return reward, moves_played

    def _expand_one(self, node: MCNode, state: State) -> MCNode:
        """Pop one untried move from node, apply it, and append the new child.

        Args:
            node: Node being expanded (state must match node's position).
            state: Shared game state, mutated in place (left applied).

        Returns:
            The newly created child node.
        """
        move = node.untried_moves.pop()
        state.apply(*move)
        child = self._node_cls(node, move, state.current_player, state.board.get_empty_cells())
        node.children.append(child)
        return child

    def _backpropagate(
        self,
        path: list[MCNode],
        reward: float,
        rollout_moves: list[tuple[Player, tuple[int, int]]] = (),
    ) -> None:
        """Propagate reward up path, flipping sign at each ancestor (zero-sum).

        Args:
            path: Nodes from root to the evaluated leaf, in order.
            reward: Reward from the perspective of the leaf's parent's player.
            rollout_moves: Unused here; MCRaveAgent overrides this method to
                also update AMAF tables from the (player, move) pairs played
                during the rollout.
        """
        sign = 1
        for node in reversed(path):
            node.visits += 1
            node.value_sum += reward * sign
            sign *= -1
