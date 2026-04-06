"""MCTS tree node — the fundamental building block of Monte Carlo Tree Search.

Each MCTSNode represents a single game state in the search tree. Nodes track
the statistics needed for the UCB1 (UCT) selection policy and for RAVE/AMAF
updates used by the MCTSRave agent.

Visit and value semantics:
    visits  — number of times this node has been selected and backed up.
    value   — cumulative reward from the perspective of the player who MADE
              the move to reach this node (the parent's current_player).
              Positive = good for that player.

Backpropagation convention:
    result = +1 if the mover (parent's current_player) wins the simulation.
    Each ancestor flips the sign because the game is zero-sum: what is good
    for one player is bad for the other.

RAVE / AMAF fields:
    amaf_values and amaf_visits are populated by MCTSRave during
    backpropagation. They store the average reward observed when a particular
    move was played at ANY point after this node in the rollout (All Moves As
    First heuristic).

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import math
from tictactoe.core.state import GameState
from tictactoe.core.types import INF, Move, MoveList, Player, Result
from tictactoe.core.board import Board


class MCTSNode:
    """Single node in the MCTS search tree.

    Stores the game state, parent/child links, visit statistics, and RAVE/AMAF
    data required by the four MCTS phases (Selection, Expansion, Simulation,
    Backpropagation).

    Attributes:
        state: The game state at this node.
        parent: Parent node, or None for the root.
        children: List of expanded child nodes.
        untried_moves: Candidate moves not yet expanded into child nodes.
            As expansion proceeds, moves are popped from this list.
        move: The move played by the parent to reach this node (None at root).
        visits: Number of times this node has been traversed during selection.
        value: Cumulative reward from the perspective of the player who made
            the move to reach this node. Positive values are favourable.
        player: The player who will move from this node's state
            (state.current_player).
        rave_visits: Unused legacy field; AMAF statistics are in amaf_visits.
        rave_value: Unused legacy field; AMAF statistics are in amaf_values.
        amaf_values: Cumulative AMAF reward per move (used by MCTSRave).
        amaf_visits: AMAF visit count per move (used by MCTSRave).
    """

    def __init__(self, state: GameState, parent: MCTSNode | None = None,
                 move: Move | None = None) -> None:
        """Initialise a new MCTS node.

        Args:
            state: The game state represented by this node.
            parent: The parent node in the tree, or None if this is the root.
            move: The move played from the parent's state to reach this state,
                or None for the root node.
        """
        self.state = state
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.untried_moves: MoveList = Board.get_candidate_moves(state, radius=2)
        self.move = move  # Move that led to this node
        self.visits: int = 0
        self.value: float = 0.0
        self.player = state.current_player  # Player who will move at this node
        # RAVE fields
        self.rave_visits: int = 0
        self.rave_value: float = 0.0
        # Per-move AMAF tables (used by RAVE agent)
        self.amaf_values: dict[Move, float] = {}
        self.amaf_visits: dict[Move, int] = {}

    def is_fully_expanded(self) -> bool:
        """Return True if all candidate moves have been expanded into children.

        Returns:
            True when untried_moves is empty, meaning every legal move from
            this node has a corresponding child node.
        """
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Return True if this node represents a finished game.

        Returns:
            True when state.result is not IN_PROGRESS (win or draw).
        """
        return self.state.result != Result.IN_PROGRESS

    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """Compute the UCB1 (UCT) score for selection.

        UCB1 = Q(s,a) + c * sqrt(ln(N_parent) / N(s,a))

        where Q is the average reward, N(s,a) is the visit count of this node,
        and N_parent is the visit count of the parent.

        Args:
            exploration_constant: Controls the exploration-exploitation
                trade-off. Higher values favour less-visited nodes.
                The theoretically optimal value for rewards in [0, 1] is
                sqrt(2) ≈ 1.414.

        Returns:
            INF for unvisited nodes (guaranteeing they are selected first).
            Otherwise the UCB1 score from the current player's perspective.
        """
        if self.visits == 0:
            return INF
        if self.parent is None or self.parent.visits == 0:
            return self.value / self.visits
        return (self.value / self.visits +
                exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits))

    def best_child(self, c: float = 1.414) -> MCTSNode:
        """Return the child with the highest UCB1 score.

        Used during the Selection phase to traverse already-expanded nodes.

        Args:
            c: Exploration constant passed to ucb1_score.

        Returns:
            The child node with the maximum UCB1 score.
        """
        return max(self.children, key=lambda n: n.ucb1_score(c))

    def most_visited_child(self) -> MCTSNode:
        """Return the child with the highest visit count.

        Used at the root after the simulation budget is exhausted to select
        the final move. Most-visited is more robust than best-score for the
        final decision because it is less sensitive to outlier simulations.

        Returns:
            The child node with the greatest visits count.
        """
        return max(self.children, key=lambda n: n.visits)

    def expand(self) -> MCTSNode:
        """Pop one untried move, create and return a child node."""
        if not self.untried_moves:
            raise ValueError("No untried moves to expand.")
        move = self.untried_moves.pop(0)
        child_state = self.state.apply_move(move)
        child_state.result = Board.is_terminal(
            child_state.board, child_state.n, child_state.k, child_state.last_move
        )
        child = MCTSNode(child_state, parent=self, move=move)
        self.children.append(child)
        return child

    def backpropagate(self, result: float) -> None:
        """Propagate result up to root, flipping sign at each level.

        Convention: result = +1 means the player who MADE the move to reach
        this node wins. Each ancestor flips the sign because the game is
        zero-sum.
        """
        self.visits += 1
        self.value += result
        if self.parent is not None:
            self.parent.backpropagate(-result)

    def win_rate(self) -> float:
        """Return the average reward (win rate) for this node.

        Returns:
            value / visits, or 0.0 if the node has never been visited.
        """
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    def rave_score(self, move: Move) -> float:
        """Return the AMAF (All Moves As First) average value for a specific move.

        The AMAF value at a node for move m is the average outcome of all
        simulations where m was played by the current player at ANY point
        after this node — not necessarily as the immediate next move.

        Args:
            move: The candidate move whose AMAF value is requested.

        Returns:
            Average AMAF reward for the move, or 0.0 if no data exists.
        """
        n = self.amaf_visits.get(move, 0)
        if n == 0:
            return 0.0
        return self.amaf_values.get(move, 0.0) / n
