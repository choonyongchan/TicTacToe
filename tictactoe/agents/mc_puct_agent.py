"""MCTS guided by a learned policy/value network (PUCT selection)."""

from __future__ import annotations

import math

import numpy as np

from tictactoe.agents.mc_policy_value_net import PolicyValueNet
from tictactoe.agents.mc_shared import MCPUCTNode, MonteCarloAgent
from tictactoe.core.state import State


class MCPUCTAgent(MonteCarloAgent):
    """MCTS agent using PUCT selection and a policy/value net instead of rollouts.

    Differs from MCUCTAgent in two ways: a node's children are all created at
    once at expansion time, weighted by the net's policy prior, and leaf
    values come directly from the net's value head rather than a random
    rollout to a terminal state.

    Attributes:
        c_puct: PUCT exploration constant.
        net: Policy/value network providing priors and leaf value estimates.
    """

    _node_cls = MCPUCTNode

    def __init__(
        self,
        n_simulations: int = 200,
        c_puct: float = 1.0,
        net: PolicyValueNet | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__("MCPUCTAgent", n_simulations, seed)
        self.c_puct = c_puct
        self.net = net if net is not None else PolicyValueNet(seed=seed)

    def act(self, state: State) -> tuple[int, int]:
        """Choose a move by running PUCT-guided MCTS.

        Args:
            state: Current game state.

        Returns:
            (row, col) of the chosen move.
        """
        root = self._node_cls(None, None, state.current_player, [])
        self._expand(root, state)
        for _ in range(self.n_simulations):
            self._simulate(root, state)
        return max(root.children, key=lambda child: child.visits).move

    def _expand(self, node: MCPUCTNode, state: State) -> None:
        """Create all of node's children at once, weighted by the net's policy prior.

        Args:
            node: Leaf node to expand (state must match node's position).
            state: Shared game state at node's position.
        """
        moves = state.board.get_empty_cells()
        policy_logits, _ = self.net.forward(self.net.encode(state))
        exp_logits = np.exp(policy_logits - policy_logits.max())
        n = self.net.n
        raw_scores = {move: float(exp_logits[move[0] * n + move[1]]) for move in moves}
        total = sum(raw_scores.values())

        for move, score in raw_scores.items():
            prior = score / total if total > 0 else 1.0 / len(moves)
            state.apply(*move)
            child = self._node_cls(node, move, state.current_player, [])
            child.prior = prior
            node.children.append(child)
            state.undo()

    def _puct_score(self, parent: MCPUCTNode, child: MCPUCTNode) -> float:
        """Return the PUCT selection score for child from parent.

        Args:
            parent: Node choosing among its children.
            child: Candidate child.

        Returns:
            child.q() + c_puct * child.prior * sqrt(parent.visits) / (1 + child.visits).
        """
        return child.q() + self.c_puct * child.prior * math.sqrt(parent.visits) / (1 + child.visits)

    def _simulate(self, root: MCPUCTNode, state: State) -> None:
        """Run one select/expand/evaluate/backpropagate PUCT iteration.

        Args:
            root: Root node of the search tree (already expanded).
            state: Shared game state; mutated during the walk and restored
                before returning.
        """
        node = root
        path = [root]
        depth = 0

        while node.children and not state.is_terminal():
            parent = node
            node = max(node.children, key=lambda child: self._puct_score(parent, child))
            state.apply(*node.move)
            path.append(node)
            depth += 1

        mover = node.parent.player
        if state.is_terminal():
            value = self._reward(state, mover)
        else:
            self._expand(node, state)
            _, net_value = self.net.forward(self.net.encode(state))
            # net_value is from state.current_player's (node.player's)
            # perspective; flip to mover's perspective to match _backpropagate.
            value = -net_value

        self._backpropagate(path, value)
        for _ in range(depth):
            state.undo()
