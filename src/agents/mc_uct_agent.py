"""Monte Carlo Tree Search with UCT (Upper Confidence bounds for Trees)."""

from __future__ import annotations

import math

from src.agents.mc_shared import MCNode, MonteCarloAgent
from src.core.state import State


class MCUCTAgent(MonteCarloAgent):
    """MCTS agent using UCB1 (UCT) for selection and random rollouts for evaluation.

    Runs the four MCTS phases (select, expand, simulate, backpropagate) for
    n_simulations iterations, building the tree by mutating a single shared
    State via apply()/undo() instead of cloning states.

    Attributes:
        c: UCB1 exploration constant.
    """

    def __init__(
        self,
        n_simulations: int = 200,
        c: float = math.sqrt(2),
        seed: int | None = None,
        name: str = "MCUCTAgent",
    ) -> None:
        super().__init__(name, n_simulations, seed)
        self.c = c

    def act(self, state: State) -> tuple[int, int]:
        """Choose a move by running MCTS and returning the most-visited root child.

        Args:
            state: Current game state.

        Returns:
            (row, col) of the chosen move.
        """
        root = self._node_cls(None, None, state.current_player, state.board.get_empty_cells())
        for _ in range(self.n_simulations):
            self._simulate(root, state)
        return max(root.children, key=lambda child: child.visits).move

    def _simulate(self, root: MCNode, state: State) -> None:
        """Run one select/expand/simulate/backpropagate iteration from root.

        Args:
            root: Root node of the search tree (state must match root's position).
            state: Shared game state; mutated during the walk and restored
                before returning.
        """
        node = root
        path = [root]
        depth = 0

        while not state.is_terminal() and node.is_fully_expanded():
            node = self._select_child(node)
            state.apply(*node.move)
            path.append(node)
            depth += 1

        if not state.is_terminal():
            node = self._expand_one(node, state)
            path.append(node)
            depth += 1

        mover = node.parent.player
        if state.is_terminal():
            reward, rollout_moves = self._reward(state, mover), []
        else:
            reward, rollout_moves = self._random_rollout(state, mover)

        self._backpropagate(path, reward, rollout_moves)
        for _ in range(depth):
            state.undo()

    def _select_child(self, node: MCNode) -> MCNode:
        """Return node's child with the highest UCB1 score.

        Args:
            node: Fully-expanded node to select from.

        Returns:
            The child maximizing uct_score.
        """
        return max(node.children, key=lambda child: child.uct_score(self.c))
