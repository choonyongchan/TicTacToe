"""MCTS augmented with RAVE (Rapid Action Value Estimation / AMAF)."""

from __future__ import annotations

import math

from src.agents.mc_shared import MCRaveNode
from src.agents.mc_uct_agent import MCUCTAgent
from src.core.types import Player


class MCRaveAgent(MCUCTAgent):
    """MCUCTAgent with a blended UCT + AMAF selection score.

    Overrides only child selection (UCT blended with each child's AMAF
    estimate) and backpropagation (also updates the AMAF tables). Tree
    structure, expansion, and rollout are inherited from MCUCTAgent.

    Attributes:
        rave_k: Crossover visit count where UCT and AMAF weigh equally.
    """

    _node_cls = MCRaveNode

    def __init__(
        self,
        n_simulations: int = 200,
        c: float = math.sqrt(2),
        rave_k: float = 500.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_simulations, c, seed, name="MCRaveAgent")
        self.rave_k = rave_k

    def _combined_score(self, node: MCRaveNode, child: MCRaveNode) -> float:
        """Return child's blended UCT + AMAF score under node's AMAF tables.

        combined = (1 - alpha) * uct + alpha * amaf, where
        alpha = max(0, (rave_k - visits) / rave_k) decays from 1 to 0 as a
        child accumulates its own Monte Carlo visits.

        Args:
            node: Parent node whose AMAF tables back the amaf term.
            child: Candidate child.

        Returns:
            math.inf for an unvisited child, otherwise the blended score.
        """
        if child.visits == 0:
            return math.inf
        amaf_n = node.amaf_visits.get(child.move, 0)
        amaf_q = node.amaf_values.get(child.move, 0.0) / amaf_n if amaf_n else 0.0
        alpha = max(0.0, (self.rave_k - child.visits) / self.rave_k)
        return (1.0 - alpha) * child.uct_score(self.c) + alpha * amaf_q

    def _select_child(self, node: MCRaveNode) -> MCRaveNode:
        """Return node's child with the highest blended UCT + AMAF score.

        Args:
            node: Fully-expanded node to select from.

        Returns:
            The child maximizing _combined_score.
        """
        return max(node.children, key=lambda child: self._combined_score(node, child))

    def _backpropagate(
        self,
        path: list[MCRaveNode],
        reward: float,
        rollout_moves: list[tuple[Player, tuple[int, int]]] = (),
    ) -> None:
        """Propagate reward as MCUCTAgent does, plus AMAF updates per node.

        For each node on path, every (player, move) played later in this
        simulation by the same player as node.player updates that move's
        AMAF entry at node, using the same perspective/sign as node.player's
        hypothetical children would use.

        Args:
            path: Nodes from root to the evaluated leaf, in order.
            reward: Reward from the perspective of the leaf's parent's player.
            rollout_moves: (player, move) pairs played during the rollout.
        """
        sign = 1
        for node in reversed(path):
            node.visits += 1
            node.value_sum += reward * sign
            for player, move in rollout_moves:
                if player is node.player:
                    node.amaf_visits[move] = node.amaf_visits.get(move, 0) + 1
                    node.amaf_values[move] = (
                        node.amaf_values.get(move, 0.0) - reward * sign
                    )
            sign *= -1
