"""NegaScout (Principal Variation Search) — Tier 1 classic search.

NegaScout (also known as Principal Variation Search, PVS) improves upon
plain negamax by exploiting the observation that, with good move ordering,
the first child is likely to be the best move (the principal variation node).

Algorithm:
    1. The first child is searched with the full [alpha, beta] window, just
       as in standard negamax.
    2. Every subsequent child is first searched with a null (zero-width) window
       [-alpha-1, -alpha]. This scout search is very fast because it can only
       prove that the move is better or worse than alpha; it cannot compute an
       exact value.
    3. If the null-window search returns a score in (alpha, beta) — a
       "fail-high" — it means the move may actually be the best, so a full
       re-search with window [-beta, -score] is performed to determine the
       exact value.

Key properties:
- When move ordering is perfect, the null-window scout never fails high and
  NegaScout visits the same nodes as the theoretically optimal alpha-beta.
- When move ordering is imperfect, re-searches add overhead; NegaScout can
  be slower than plain alpha-beta if the first child is rarely the best move.
- No transposition table, iterative deepening, or advanced move ordering
  beyond the static score_move_statically heuristic.

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


class NegaScout(BaseAgent):
    """NegaScout (Principal Variation Search) with alpha-beta pruning.

    The first child is searched with the full [alpha, beta] window.
    Remaining children are searched with a null window [-alpha-1, -alpha].
    If the null-window scout fails high (score > alpha and score < beta),
    a re-search with the full window is performed to find the exact value.

    Attributes:
        depth: The fixed search depth used when no MatchConfig is provided.
        match_config: The budget configuration controlling search termination.
    """

    def __init__(self, depth: int = 4, match_config: MatchConfig | None = None) -> None:
        """Initialise the NegaScout agent.

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
        """Select the best move using NegaScout search.

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

        for i, move in enumerate(candidates):
            if budget.exhausted(counters[0], 0):
                break

            child = state.apply_move(move)
            child.result = Board.is_terminal(child.board, child.n, child.k, child.last_move)

            if i == 0:
                # First child: full window search.
                score = -self._negascout(
                    child,
                    budget.max_depth() - 1,
                    -beta,
                    -alpha,
                    budget,
                    counters,
                    1,
                )
            else:
                # Null window search.
                score = -self._negascout(
                    child,
                    budget.max_depth() - 1,
                    -alpha - 1,
                    -alpha,
                    budget,
                    counters,
                    1,
                )
                # Re-search if null window fails high.
                if alpha < score < beta:
                    score = -self._negascout(
                        child,
                        budget.max_depth() - 1,
                        -beta,
                        -score,
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

    def _negascout(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        budget: SearchBudget,
        counters: list,
        depth_from_root: int,
    ) -> float:
        """Recursive NegaScout search.

        First child is searched with the full window. Remaining children use
        a null window [-alpha-1, -alpha] with optional re-search on failure.

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
        for i, move in enumerate(candidates):
            child = state.apply_move(move)
            child.result = Board.is_terminal(child.board, child.n, child.k, child.last_move)

            if i == 0:
                # First child: full window.
                score = -self._negascout(
                    child, depth - 1, -beta, -alpha, budget, counters, depth_from_root + 1,
                )
            else:
                # Null window.
                score = -self._negascout(
                    child, depth - 1, -alpha - 1, -alpha, budget, counters, depth_from_root + 1,
                )
                # Re-search if null window fails high.
                if alpha < score < beta:
                    score = -self._negascout(
                        child, depth - 1, -beta, -score, budget, counters, depth_from_root + 1,
                    )

            best = max(best, score)
            alpha = max(alpha, best)
            if alpha >= beta:
                counters[2] += 1
                break

        return best

    def get_name(self) -> str:
        """Return the agent's display name including the configured depth.

        Returns:
            A string of the form "NegaScout(depth=N)".
        """
        return f"NegaScout(depth={self.depth})"

    def get_tier(self) -> int:
        """Return the benchmark tier for this agent.

        Returns:
            1 — classic search, no heuristic enhancements.
        """

        return 1
