"""Vanilla MCTS with UCT (Upper Confidence Bound for Trees).

Monte Carlo Tree Search builds a partial game tree by repeatedly running
four phases until the search budget is exhausted, then selects the most-
visited child of the root as the final move.

Four phases:
    Selection:
        Starting from the root, traverse fully-expanded nodes using the
        UCB1 (UCT) formula to balance exploration and exploitation, until
        reaching a node with at least one untried move or a terminal node.
    Expansion:
        If the selected node is not terminal, expand one untried move by
        creating a new child node.
    Simulation (rollout):
        From the newly-expanded node, play a random game to completion (or
        until the rollout depth limit is reached). Every move is chosen
        uniformly at random from all empty cells.
    Backpropagation:
        Propagate the simulation result back up the tree, incrementing
        visit counts and adding the reward at each node. Signs are flipped
        at each level because the game is zero-sum.

Final move selection:
    The root's child with the highest visit count (most_visited_child) is
    returned. Visit count is more robust than win rate for the final decision.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import random
import time

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents._search_budget import SearchBudget
from tictactoe.agents._shared_utils import check_forced_move
from tictactoe.agents.monte_carlo.node import MCTSNode
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Move, Player, Result
from tictactoe.benchmark.metrics import MatchConfig


class MCTSVanilla(BaseAgent):
    """Vanilla MCTS agent using UCT for tree policy and random rollouts.

    Uses UCB1 (UCT) for Selection and uniformly-random play for the Simulation
    phase. No domain knowledge is applied during rollouts.

    Attributes:
        c: UCB1 exploration constant (sqrt(2) ≈ 1.414 is theoretically
            optimal for rewards in [0, 1]).
        match_config: Budget configuration (time/node/depth controlled).
        rollout_depth_limit: Maximum number of moves per random rollout.
            Limits computation on large boards where games rarely terminate.
        _rng: Seeded random number generator for reproducible rollouts.
    """

    def __init__(self, exploration_constant: float = 1.414,
                 match_config: MatchConfig | None = None,
                 rollout_depth_limit: int = 200,
                 seed: int | None = None) -> None:
        """Initialise the vanilla MCTS agent.

        Args:
            exploration_constant: UCB1 exploration constant c. Higher values
                encourage exploration of less-visited nodes.
            match_config: Budget configuration. None defaults to TIME_CONTROLLED
                with a 1000 ms limit.
            rollout_depth_limit: Maximum rollout length before returning 0.0
                (draw estimate).
            seed: Random seed for the rollout RNG. None uses a random seed.
        """
        self.c = exploration_constant
        self.match_config = match_config
        self.rollout_depth_limit = rollout_depth_limit
        self._rng = random.Random(seed)

    def choose_move(self, state: GameState) -> Move:
        """Select the best move using vanilla MCTS with UCT.

        Runs MCTS simulations until the budget is exhausted, then returns
        the root child with the highest visit count.

        Args:
            state: The current game state.

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
            return forced

        budget = SearchBudget(self.match_config, time.perf_counter_ns())

        root_state = state.copy()
        root_state.result = Board.is_terminal(
            root_state.board, root_state.n, root_state.k, root_state.last_move
        )
        root = MCTSNode(root_state)

        simulations = 0
        max_tree_depth = 0

        while not budget.exhausted(simulations, 0):
            # --- Selection ---
            node = root
            depth = 0
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child(self.c)
                depth += 1
            max_tree_depth = max(max_tree_depth, depth)

            # --- Expansion ---
            if not node.is_terminal():
                mover = node.state.current_player  # who makes the move
                node = node.expand()
                depth += 1
                max_tree_depth = max(max_tree_depth, depth)
            else:
                # Terminal node: mover is the player who just moved (parent's player)
                if node.parent is not None:
                    mover = node.parent.state.current_player
                else:
                    mover = node.state.current_player

            # --- Simulation (random rollout) ---
            result = self._rollout(node.state, mover)

            # --- Backpropagation ---
            # result = +1 if `mover` wins; backpropagate from the newly expanded node
            node.backpropagate(result)
            simulations += 1

        if not root.children:
            # No simulations ran; return first candidate
            candidates = Board.get_candidate_moves(state, radius=2)
            move = candidates[0] if candidates else (0, 0)
        else:
            move = root.most_visited_child().move

        state.nodes_visited = simulations
        state.max_depth_reached = max_tree_depth
        state.prunings = 0
        state.compute_ebf()
        return move

    def _rollout(self, state: GameState, mover: Player) -> float:
        """Random rollout from state.

        Args:
            state: State to roll out from (after expansion).
            mover: The player whose perspective determines +1/-1.

        Returns:
            +1 if mover wins, -1 if mover loses, 0 for draw or depth limit.
        """
        if state.result != Result.IN_PROGRESS:
            return self._terminal_value_for(state, mover)

        sim_state = state.copy()

        for _ in range(self.rollout_depth_limit):
            empty = Board.get_all_empty_cells(sim_state.board)
            if not empty:
                return 0.0  # Draw
            move = self._rng.choice(empty)
            sim_state = sim_state.apply_move(move)
            sim_state.result = Board.is_terminal(
                sim_state.board, sim_state.n, sim_state.k, sim_state.last_move
            )
            if sim_state.result != Result.IN_PROGRESS:
                return self._terminal_value_for(sim_state, mover)

        return 0.0  # Rollout limit reached

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
        """Return the agent's display name including the exploration constant.

        Returns:
            A string of the form "MCTS-Vanilla(c=N)".
        """
        return f"MCTS-Vanilla(c={self.c})"

    def get_tier(self) -> int:
        """Return the benchmark tier for this agent.

        Returns:
            3 — Monte Carlo search.
        """
        return 3
