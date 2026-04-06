"""NegaMax — Tier 1 classic negamax with alpha-beta pruning.

Negamax formulation of minimax: scores are always from the perspective of
the current player, and the caller negates the returned score.

Dependency chain: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import time

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.agents._search_budget import SearchBudget
from tictactoe.agents._shared_utils import check_forced_move
from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import INF, Move, Result
from tictactoe.evaluation.heuristics import evaluate_position
from tictactoe.evaluation.move_ordering import order_moves
from tictactoe.benchmark.metrics import MatchConfig


class NegaMax(BaseAgent):
    """Classic negamax search with alpha-beta pruning.

    All scores are from the perspective of state.current_player. The parent
    negates the returned score to convert to its own perspective.
    """

    def __init__(self, depth: int = 4, match_config: MatchConfig | None = None) -> None:
        """Initialise the NegaMax agent.

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
        """Select the best move using negamax with alpha-beta pruning.

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

        candidates = Board.get_candidate_moves(state, radius=2)
        candidates = order_moves(state, candidates)

        best_move = candidates[0]  # Fallback
        best_score = -INF
        alpha = -INF
        beta = INF

        for move in candidates:
            if budget.exhausted(counters[0], 0):
                break

            child = state.apply_move(move)
            child.result = Board.is_terminal(child.board, child.n, child.k, child.last_move)

            score = -self._negamax(
                child,
                budget.max_depth() - 1,
                -beta,
                -alpha,
                budget,
                counters,
                1,
            )

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                counters[2] += 1
                break

        state.nodes_visited = counters[0]
        state.max_depth_reached = counters[1]
        state.prunings = counters[2]
        state.compute_ebf()
        return best_move

    def _negamax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        budget: SearchBudget,
        counters: list,
        depth_from_root: int,
    ) -> float:
        """Recursive negamax with alpha-beta pruning.

        Scores are from the perspective of state.current_player.
        The caller must negate the returned value.

        Args:
            state: Current game state.
            depth: Remaining search depth.
            alpha: Lower bound (current player's perspective).
            beta: Upper bound (current player's perspective).
            budget: Search budget controller.
            counters: Shared [nodes_visited, max_depth_reached, prunings].
            depth_from_root: Distance from the search root.

        Returns:
            Best score from state.current_player's perspective.
        """
        counters[0] += 1
        counters[1] = max(counters[1], depth_from_root)

        # Terminal or depth limit.
        if state.result != Result.IN_PROGRESS:
            return evaluate_position(state, state.current_player)
        if depth == 0 or budget.exhausted(counters[0], depth_from_root):
            return evaluate_position(state, state.current_player)

        candidates = Board.get_candidate_moves(state, radius=2)
        if not candidates:
            return evaluate_position(state, state.current_player)

        candidates = order_moves(state, candidates)

        best = -INF
        for move in candidates:
            child = state.apply_move(move)
            child.result = Board.is_terminal(child.board, child.n, child.k, child.last_move)

            score = -self._negamax(
                child, depth - 1, -beta, -alpha, budget, counters, depth_from_root + 1,
            )
            best = max(best, score)
            alpha = max(alpha, best)
            if alpha >= beta:
                counters[2] += 1
                break

        return best

    def get_name(self) -> str:
        return f"NegaMax(depth={self.depth})"

    def get_tier(self) -> int:
        return 1
