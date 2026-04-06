"""MCTS with RAVE (Rapid Action Value Estimation).

RAVE augments vanilla MCTS by maintaining an AMAF (All Moves As First)
table at each node. The AMAF value for move m at node s estimates how good
m is from s, using data from ALL simulations where m was played at any later
point — not just when m was the immediate next move.

Combined selection score (per child):
    combined = (1 - alpha) * UCT_score + alpha * AMAF_score

where UCT_score is the standard UCB1 value and:
    alpha = max(0, (rave_k - N) / rave_k)

As the child's visit count N increases toward rave_k, alpha decreases from
1.0 to 0.0 — smoothly transitioning from pure AMAF to pure UCT. The
parameter rave_k controls how quickly we trust the Monte Carlo statistics
over the AMAF statistics.

AMAF backpropagation:
    After each simulation, every move played by the node's player during the
    rollout updates the AMAF table of every ancestor node for that player.
    This "free" information makes RAVE converge faster than vanilla MCTS on
    games where move quality is relatively position-independent (e.g., Go),
    but can be misleading in highly tactical positions.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import math
import random
import time
from typing import NamedTuple

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents._search_budget import SearchBudget
from tictactoe.agents._shared_utils import check_forced_move
from tictactoe.agents.monte_carlo.node import MCTSNode
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Move, Player, Result
from tictactoe.benchmark.metrics import MatchConfig


class MCTSRave(BaseAgent):
    """MCTS agent augmented with RAVE (AMAF) value estimates.

    Replaces UCB1 selection with a combined UCT + AMAF score that blends
    Monte Carlo visit statistics with rapid action value estimates. Early in
    the search, AMAF dominates (alpha ≈ 1); as visit counts grow, UCT takes
    over (alpha → 0).

    Attributes:
        c: Exploration constant used in the UCT term of the combined score.
        rave_k: Crossover point — the number of visits at which alpha = 0.5,
            meaning UCT and AMAF contribute equally to the selection score.
        match_config: Budget configuration (time/node/depth controlled).
        rollout_depth_limit: Maximum moves per random rollout.
        _rng: Seeded RNG for rollout move sampling.
    """

    def __init__(self, exploration_constant: float = 0.5,
                 rave_k: float = 500.0,
                 match_config: MatchConfig | None = None,
                 rollout_depth_limit: int = 200,
                 seed: int | None = None) -> None:
        """Initialise the MCTS-RAVE agent.

        Args:
            exploration_constant: UCT exploration constant c (typically lower
                than vanilla MCTS because AMAF already encourages exploration).
            rave_k: RAVE crossover parameter. Higher values keep the AMAF
                influence active for longer. Default 500.0 is suitable for
                typical board sizes.
            match_config: Budget configuration. None defaults to TIME_CONTROLLED
                with a 1000 ms limit.
            rollout_depth_limit: Maximum rollout length.
            seed: Random seed. None uses a random seed.
        """
        self.c = exploration_constant
        self.rave_k = rave_k
        self.match_config = match_config
        self.rollout_depth_limit = rollout_depth_limit
        self._rng = random.Random(seed)

    def choose_move(self, state: GameState) -> Move:
        """Select the best move using MCTS with RAVE/AMAF.

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
            path: list[MCTSNode] = [node]

            while node.is_fully_expanded() and not node.is_terminal():
                node = self._best_child_rave(node)
                path.append(node)
                depth += 1
            max_tree_depth = max(max_tree_depth, depth)

            # --- Expansion ---
            if not node.is_terminal():
                mover = node.state.current_player
                node = node.expand()
                path.append(node)
                depth += 1
                max_tree_depth = max(max_tree_depth, depth)
            else:
                if node.parent is not None:
                    mover = node.parent.state.current_player
                else:
                    mover = node.state.current_player

            # --- Rollout (collect moves played) ---
            result, rollout_moves = self._rollout_with_moves(node.state, mover)

            # --- Backpropagation with RAVE updates ---
            self._backpropagate_rave(node, result, rollout_moves)
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

    def _best_child_rave(self, node: MCTSNode) -> MCTSNode:
        """Select the child with the highest combined UCT + RAVE score.

        The combined score is:
            combined = (1 - alpha) * UCT + alpha * AMAF
        where:
            UCT   = Q/N + c * sqrt(ln(N_parent) / N)
            AMAF  = amaf_value / amaf_visits  (from parent's AMAF table)
            alpha = max(0, (rave_k - N) / rave_k)

        Unvisited children receive a score of +infinity and are always
        selected first.

        Args:
            node: The parent node whose children are being evaluated.

        Returns:
            The child with the maximum combined score.
        """
        def combined_score(child: MCTSNode) -> float:
            if child.visits == 0:
                return math.inf
            N = child.visits
            N_parent = node.visits
            q_value = child.value / N
            uct = q_value + self.c * math.sqrt(math.log(max(N_parent, 1)) / max(N, 1))

            # AMAF score from parent's perspective
            if child.move is not None:
                amaf_n = node.amaf_visits.get(child.move, 0)
                amaf_v = node.amaf_values.get(child.move, 0.0)
                amaf_q = amaf_v / max(amaf_n, 1) if amaf_n > 0 else 0.0
            else:
                amaf_q = 0.0
                amaf_n = 0

            alpha = max(0.0, (self.rave_k - N) / self.rave_k)
            return (1.0 - alpha) * uct + alpha * amaf_q

        return max(node.children, key=combined_score)

    def _rollout_with_moves(
        self, state: GameState, mover: Player
    ) -> tuple[float, list[tuple[Player, Move]]]:
        """Random rollout that also records each (player, move) pair played.

        The recorded move list is used by _backpropagate_rave to update the
        AMAF tables of all nodes along the path from the expanded leaf to the
        root.

        Args:
            state: State from which to start the rollout.
            mover: The player whose perspective determines the reward sign.

        Returns:
            A (reward, moves_played) pair where reward is +1/-1/0 and
            moves_played is a list of (player_who_moved, move) tuples in
            the order they were played during the rollout.
        """
        if state.result != Result.IN_PROGRESS:
            return self._terminal_value_for(state, mover), []

        sim_state = state.copy()
        moves_played: list[tuple[Player, Move]] = []

        for _ in range(self.rollout_depth_limit):
            empty = Board.get_all_empty_cells(sim_state.board)
            if not empty:
                return 0.0, moves_played
            move = self._rng.choice(empty)
            player_who_moved = sim_state.current_player
            sim_state = sim_state.apply_move(move)
            sim_state.result = Board.is_terminal(
                sim_state.board, sim_state.n, sim_state.k, sim_state.last_move
            )
            moves_played.append((player_who_moved, move))
            if sim_state.result != Result.IN_PROGRESS:
                return self._terminal_value_for(sim_state, mover), moves_played

        return 0.0, moves_played

    def _backpropagate_rave(
        self,
        node: MCTSNode,
        result: float,
        rollout_moves: list[tuple[Player, Move]],
    ) -> None:
        """Backpropagate result and update AMAF tables along the path.

        Convention: result = +1 if the mover (who made the move to reach
        this node) wins. Signs flip going up the tree.
        """
        current_result = result
        current_node = node

        # Build sets of moves played in rollout per player for AMAF lookup
        rollout_by_player: dict[Player, list[Move]] = {}
        for pl, mv in rollout_moves:
            rollout_by_player.setdefault(pl, []).append(mv)

        while current_node is not None:
            current_node.visits += 1
            current_node.value += current_result

            # Update AMAF table: for each child move that appeared in the
            # rollout by the same player who would play that move from this
            # node's state, update the AMAF entry.
            # The player who moves at current_node is current_node.state.current_player
            # (the player who will make the next move from this node's state).
            # But since this node was just reached by a move from the parent,
            # the player who moved to reach this node is the parent's current_player,
            # which is current_node.state.current_player.opponent() ...
            # Actually we update AMAF for the player whose moves are stored at this
            # node. We track: "if move m was played by ANY player after this node
            # in the rollout, update AMAF".
            # Standard RAVE: update AMAF at a node for every move played by the
            # same player (the node's player to move) during the rollout.
            node_player = current_node.state.current_player
            for mv in rollout_by_player.get(node_player, []):
                current_node.amaf_visits[mv] = current_node.amaf_visits.get(mv, 0) + 1
                current_node.amaf_values[mv] = current_node.amaf_values.get(mv, 0.0) + current_result

            current_result = -current_result
            current_node = current_node.parent

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
        """Return the agent's display name including key hyperparameters.

        Returns:
            A string of the form "MCTS-RAVE(c=N, k=M)".
        """
        return f"MCTS-RAVE(c={self.c}, k={self.rave_k})"

    def get_tier(self) -> int:
        """Return the benchmark tier for this agent.

        Returns:
            3 — Monte Carlo search.
        """
        return 3
