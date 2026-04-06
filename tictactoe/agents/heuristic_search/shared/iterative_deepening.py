"""Iterative deepening wrapper for enhanced search agents.

Iterative Deepening Depth-First Search (IDDFS) combines the space efficiency
of depth-first search with the optimality properties of breadth-first search.
The board is searched to depth 1, then depth 2, and so on until the budget is
exhausted. Because earlier iterations fill the transposition table and move-
ordering tables, later iterations are dramatically faster than a single deep
search would be.

Aspiration Windows:
    On depth >= 2, the search is first tried with a narrow window
    [prev_score - delta, prev_score + delta] around the previous iteration's
    result. If the search fails (score outside the window), it is immediately
    re-tried with a full [-INF, INF] window. Aspiration windows reduce the
    effective branching factor when the score is stable across iterations.

Result Safety:
    Only the result from the last FULLY COMPLETED iteration is returned. If a
    partial iteration is abandoned mid-way due to budget exhaustion, the
    previous iteration's best move is kept — avoiding accidentally returning a
    move from a shallow, incomplete search.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

from typing import Callable

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import INF, Move, Score
from tictactoe.agents._search_budget import SearchBudget
from tictactoe.agents.heuristic_search.shared.transposition_table import TranspositionTable
from tictactoe.evaluation.move_ordering import KillerMoveTable, HistoryTable


class IterativeDeepeningWrapper:
    """Drives iterative deepening for any depth-limited search function.

    The wrapper is search-algorithm agnostic: it accepts any callable that
    conforms to the search_fn signature and calls it with increasing depth
    limits, managing the aspiration window logic and partial-iteration safety
    check itself.

    The killer move and history heuristic tables are NOT cleared between
    iterations — they accumulate information across depths, which improves
    move ordering at deeper levels.

    Returns the best move from the last FULLY COMPLETED iteration. If the
    budget expires mid-iteration, the previous iteration's result is
    returned (unless it was depth 1, in which case the partial result is
    used as a fallback).

    Attributes:
        _search_fn: The depth-limited search callable.
        _use_aspiration: Whether aspiration windows are enabled.
        _aspiration_delta: Half-width of the aspiration window.
    """

    def __init__(self, search_fn: Callable, use_aspiration: bool = True,
                 aspiration_delta: float = 50.0) -> None:
        """Initialise the wrapper.

        Args:
            search_fn: The depth-limited search callable. Signature:
                (state, depth, alpha, beta, budget, tt, killers, history, counters)
                → (score, best_move).
            use_aspiration: Whether to use aspiration windows on depth >= 2.
            aspiration_delta: Half-width of the aspiration window.
        """
        self._search_fn = search_fn
        self._use_aspiration = use_aspiration
        self._aspiration_delta = aspiration_delta

    def run(self, state: GameState, budget: SearchBudget, tt: TranspositionTable,
            killers: KillerMoveTable, history: HistoryTable,
            counters: list) -> tuple[Move, Score, int]:
        """Run iterative deepening. Returns (best_move, best_score, max_depth).

        Starts from depth 1 and increases until the budget is exhausted. The
        result from the last FULLY COMPLETED iteration is returned.

        Args:
            state: The current game state at the root.
            budget: Budget controller (time/node/depth).
            tt: Shared transposition table.
            killers: Killer move table, preserved across iterations.
            history: History heuristic table, preserved across iterations.
            counters: Shared mutable list [nodes, max_depth_reached, prunings].

        Returns:
            A (best_move, best_score, max_depth_completed) triple.
        """
        # Fallback: first candidate move
        candidates = Board.get_candidate_moves(state, radius=2)
        best_move: Move | None = candidates[0] if candidates else None
        best_score: Score = -INF
        max_depth_completed = 0
        prev_score = 0.0

        for depth in range(1, 101):
            if budget.exhausted(counters[0], depth):
                break

            # Aspiration windows
            if self._use_aspiration and depth >= 2:
                alpha = prev_score - self._aspiration_delta
                beta = prev_score + self._aspiration_delta
            else:
                alpha, beta = -INF, INF

            # Track nodes at start of this iteration
            nodes_before = counters[0]

            result_score, result_move = self._search_fn(
                state, depth, alpha, beta, budget, tt, killers, history, counters
            )

            # Handle aspiration failure: re-search with full window
            if result_score <= alpha or result_score >= beta:
                result_score, result_move = self._search_fn(
                    state, depth, -INF, INF, budget, tt, killers, history, counters
                )

            # Check if budget was exhausted during this iteration
            if budget.exhausted(counters[0], depth) and depth > 1:
                # Partial iteration: only use result if meaningful nodes were added
                nodes_added = counters[0] - nodes_before
                if nodes_added < 2:
                    break
                # If it got a result_move (non-None), use it
                if result_move is not None:
                    best_move = result_move
                    best_score = result_score
                break

            if result_move is not None:
                best_move = result_move
                best_score = result_score
            prev_score = result_score
            max_depth_completed = depth

        # best_move is always initialised to candidates[0] above; cast is safe.
        return best_move, best_score, max_depth_completed  # type: ignore[return-value]
