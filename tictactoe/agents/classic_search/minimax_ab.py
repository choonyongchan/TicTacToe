"""Minimax with Alpha-Beta Pruning — exact adversarial search at fixed depth.

Minimax explores the complete game tree to a fixed depth, assuming both
players play optimally. The maximising player selects the move with the
highest score; the minimising player selects the move with the lowest score.

Alpha-beta pruning eliminates subtrees that cannot affect the final decision:
- Alpha (α): the best score the maximising player can already guarantee.
- Beta (β): the best score the minimising player can already guarantee.
Whenever α >= β at any node, the remaining siblings are skipped (pruned).

Key properties:
- Complete: always returns a move (falls back to first candidate).
- Exact: produces the true minimax value within the search depth.
- No transposition table or iterative deepening — purely depth-controlled.
- Move ordering via score_move_statically improves pruning efficiency.

Dependency chain: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import time

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents._search_budget import SearchBudget
from tictactoe.agents._shared_utils import check_forced_move
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import INF, Cell, Move, Player, Result
from tictactoe.evaluation.heuristics import evaluate_position
from tictactoe.evaluation.move_ordering import order_moves
from tictactoe.benchmark.metrics import MatchConfig


class MinimaxAB(BaseAgent):
    """Classic minimax with alpha-beta pruning at a fixed search depth.

    Uses MAX/MIN alternation via an is_maximising boolean flag. The root
    player is fixed as maximising throughout the entire search tree, and
    all evaluations are relative to that player via evaluate_position.

    No transposition table, iterative deepening, killer moves, or history
    heuristic — only the static score_move_statically ordering from
    order_moves is applied to improve pruning efficiency.

    Attributes:
        depth: The fixed search depth used when no MatchConfig is provided.
        match_config: The budget configuration controlling search termination.
    """

    def __init__(self, depth: int = 4, match_config: MatchConfig | None = None) -> None:
        """Initialise the MinimaxAB agent.

        Args:
            depth: Fixed search depth. Used when match_config is None or
                DEPTH_CONTROLLED.
            match_config: Budget configuration. When None, defaults to
                DEPTH_CONTROLLED with the given depth.
        """
        self.depth = depth
        if match_config is None:
            self.match_config = MatchConfig.depth_controlled(depth)
        else:
            self.match_config = match_config

    def choose_move(self, state: GameState) -> Move:
        """Select the best move using minimax with alpha-beta pruning.

        Args:
            state: The current game state.

        Returns:
            The chosen (row, col) move.
        """
        # Short-circuit for forced moves.
        forced = check_forced_move(state)
        if forced is not None:
            state.nodes_visited = 1
            state.max_depth_reached = 0
            state.prunings = 0
            state.compute_ebf()
            return forced

        budget = SearchBudget(self.match_config, time.perf_counter_ns())
        counters = [0, 0, 0]  # [nodes_visited, max_depth_reached, prunings]

        maximising_player = state.current_player
        candidates = Board.get_candidate_moves(state, radius=2)
        candidates = order_moves(state, candidates)

        best_move = candidates[0]  # Fallback
        best_score = -INF

        for move in candidates:
            if budget.exhausted(counters[0], 0):
                break

            child = state.apply_move(move)
            child.result = Board.is_terminal(child.board, child.n, child.k, child.last_move)

            score = self._minimax(
                child,
                budget.max_depth() - 1,
                -INF,
                INF,
                False,  # Opponent moves next (minimising)
                budget,
                maximising_player,
                counters,
                1,
            )

            if score > best_score:
                best_score = score
                best_move = move

        state.nodes_visited = counters[0]
        state.max_depth_reached = counters[1]
        state.prunings = counters[2]
        state.compute_ebf()
        return best_move

    def _minimax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        is_maximising: bool,
        budget: SearchBudget,
        maximising_player: Player,
        counters: list,
        depth_from_root: int,
    ) -> float:
        """Recursive minimax with alpha-beta pruning.

        Args:
            state: Current game state.
            depth: Remaining search depth.
            alpha: Best score the maximising player can guarantee.
            beta: Best score the minimising player can guarantee.
            is_maximising: True when the current player is maximising.
            budget: Search budget controller.
            maximising_player: The root player (fixed throughout search).
            counters: Shared [nodes_visited, max_depth_reached, prunings].
            depth_from_root: Distance from the search root.

        Returns:
            The best score achievable from this state.
        """
        counters[0] += 1
        counters[1] = max(counters[1], depth_from_root)

        # Terminal or depth limit.
        if state.result != Result.IN_PROGRESS:
            return evaluate_position(state, maximising_player)
        if depth == 0 or budget.exhausted(counters[0], depth_from_root):
            return evaluate_position(state, maximising_player)

        candidates = Board.get_candidate_moves(state, radius=2)
        if not candidates:
            return evaluate_position(state, maximising_player)

        candidates = order_moves(state, candidates)

        if is_maximising:
            best = -INF
            for move in candidates:
                child = state.apply_move(move)
                child.result = Board.is_terminal(child.board, child.n, child.k, child.last_move)

                score = self._minimax(
                    child, depth - 1, alpha, beta, False,
                    budget, maximising_player, counters, depth_from_root + 1,
                )
                best = max(best, score)
                alpha = max(alpha, best)
                if alpha >= beta:
                    counters[2] += 1
                    break
            return best
        else:
            best = INF
            for move in candidates:
                child = state.apply_move(move)
                child.result = Board.is_terminal(child.board, child.n, child.k, child.last_move)

                score = self._minimax(
                    child, depth - 1, alpha, beta, True,
                    budget, maximising_player, counters, depth_from_root + 1,
                )
                best = min(best, score)
                beta = min(beta, best)
                if alpha >= beta:
                    counters[2] += 1
                    break
            return best

    def get_name(self) -> str:
        """Return the agent's display name including the configured depth.

        Returns:
            A string of the form "MinimaxAB(depth=N)".
        """
        return f"MinimaxAB(depth={self.depth})"

    def get_tier(self) -> int:
        """Return the benchmark tier for this agent.

        Returns:
            1 — classic search, no heuristic enhancements.
        """
        return 1
