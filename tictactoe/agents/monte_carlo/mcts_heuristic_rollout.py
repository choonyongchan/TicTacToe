"""MCTS with heuristic rollout policy.

Extends vanilla MCTS by replacing the purely-random simulation phase with a
domain-informed rollout policy. The four MCTS phases (Selection, Expansion,
Simulation, Backpropagation) are otherwise identical to MCTSVanilla.

Heuristic rollout policy (applied at every step of the simulation):
    1. Win immediately: if the current player has a winning move, play it.
    2. Block immediately: if the opponent has a winning move, block it.
    3. Softmax-weighted sampling: score all candidate moves with
       score_move_statically, convert scores to a probability distribution
       via softmax (temperature = 1.0), and sample one move. This biases
       the rollout towards moves that look promising without fully
       evaluating the position.

The heuristic policy produces more informative simulation results than pure
random play, particularly in the early game when tactical threats are sparse
and random play would rarely trigger them.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import math
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
from tictactoe.evaluation.move_ordering import score_move_statically


class MCTSHeuristicRollout(BaseAgent):
    """MCTS agent with heuristic (non-random) rollout policy.

    Uses UCB1 (UCT) for the Selection phase, identical to MCTSVanilla. The
    Simulation phase is replaced by a three-priority heuristic: win > block >
    softmax-weighted sampling. This produces higher-quality estimates per
    simulation at the cost of more computation per rollout step.

    The rollout depth limit is deliberately lower than MCTSVanilla (default
    50 vs 200) to compensate for the higher per-step cost.

    Attributes:
        c: UCB1 exploration constant.
        match_config: Budget configuration (time/node/depth controlled).
        rollout_depth_limit: Maximum number of moves per heuristic rollout.
        _rng: Seeded random number generator for softmax sampling.
    """

    def __init__(self, exploration_constant: float | None = None,
                 match_config: MatchConfig | None = None,
                 rollout_depth_limit: int | None = None,
                 seed: int | None = None) -> None:
        """Initialise the heuristic rollout MCTS agent.

        Args:
            exploration_constant: UCB1 exploration constant c. None reads from
                config (default 1.414 if config is not loaded).
            match_config: Budget configuration. None defaults to TIME_CONTROLLED
                with a 1000 ms limit.
            rollout_depth_limit: Maximum rollout length. None reads from config
                (default 50 if config is not loaded).
            seed: Random seed for softmax sampling. None uses a random seed.
        """
        from tictactoe.config import get_config as _cfg, ConfigError as _CE
        try:
            _c = _cfg()
            self.c = exploration_constant if exploration_constant is not None \
                else _c.mcts.exploration_constant
            self.rollout_depth_limit = rollout_depth_limit if rollout_depth_limit is not None \
                else _c.mcts.heuristic_rollout_depth
        except _CE:
            self.c = exploration_constant if exploration_constant is not None else 1.414
            self.rollout_depth_limit = rollout_depth_limit if rollout_depth_limit is not None else 50
        self.match_config = match_config
        self._rng = random.Random(seed)

    def choose_move(self, state: GameState) -> Move:
        """Select the best move using MCTS with a heuristic rollout policy.

        Runs MCTS simulations until the budget is exhausted, then returns
        the root child with the highest visit count.

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
                mover = node.state.current_player
                node = node.expand()
                depth += 1
                max_tree_depth = max(max_tree_depth, depth)
            else:
                if node.parent is not None:
                    mover = node.parent.state.current_player
                else:
                    mover = node.state.current_player

            # --- Heuristic Rollout ---
            result = self._heuristic_rollout(node.state, mover)

            # --- Backpropagation ---
            node.backpropagate(result)
            simulations += 1

        if not root.children:
            candidates = Board.get_candidate_moves(state, radius=2)
            move = candidates[0] if candidates else (0, 0)
        else:
            move = root.most_visited_child().move

        state.nodes_visited = simulations
        state.max_depth_reached = max_tree_depth
        state.prunings = 0
        state.compute_ebf()
        return move

    def _heuristic_rollout(self, state: GameState, mover: Player,
                           depth: int = 0) -> float:
        """Simulate a game from state using the heuristic rollout policy.

        At each step, the current player:
        1. Takes an immediate winning move if one exists.
        2. Blocks an immediate opponent winning move if one exists.
        3. Otherwise samples from candidate moves weighted by softmax-
           transformed score_move_statically scores.

        The result is from mover's perspective regardless of whose turn it
        is at each rollout step.

        Args:
            state: The game state to roll out from.
            mover: The player whose perspective determines the sign of the
                returned reward (+1 = mover wins).
            depth: Current rollout depth (used to enforce the depth limit).

        Returns:
            +1.0 if mover won the simulation, -1.0 if mover lost, 0.0 for
            a draw or if the depth limit was reached.
        """
        if state.result != Result.IN_PROGRESS:
            return self._terminal_value_for(state, mover)
        if depth >= self.rollout_depth_limit:
            return 0.0

        # Check immediate win
        win = Board.get_winning_move(state.board, state.n, state.k, state.current_player)
        if win is not None:
            move = win
        else:
            # Check immediate block
            block = Board.get_blocking_move(state.board, state.n, state.k, state.current_player)
            if block is not None:
                move = block
            else:
                # Softmax-weighted sampling over candidate moves
                candidates = Board.get_candidate_moves(state, radius=2)
                if not candidates:
                    return 0.0
                scores = [max(0.01, score_move_statically(state, m)) for m in candidates]
                total = sum(scores)
                probs = [s / total for s in scores]
                move = self._rng.choices(candidates, weights=probs, k=1)[0]

        next_state = state.apply_move(move)
        next_state.result = Board.is_terminal(
            next_state.board, next_state.n, next_state.k, move
        )
        return self._heuristic_rollout(next_state, mover, depth + 1)

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
            A string of the form "MCTS-HeuristicRollout(c=N, temp=1.0)".
        """
        return f"MCTS-HeuristicRollout(c={self.c}, temp=1.0)"

    def get_tier(self) -> int:
        """Return the benchmark tier for this agent.

        Returns:
            3 — Monte Carlo search.
        """
        return 3
