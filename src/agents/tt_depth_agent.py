from __future__ import annotations

from src.agents.negamax_base_agent import NegamaxBaseAgent
from src.core.state import State
from src.core.transposition_table import TranspositionTable
from src.core.types import NEGATIVE_INFINITY
from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.heuristic import Heuristic


class TTDepthAgent(NegamaxBaseAgent):
    """Base class for agents using depth-limited negamax with a transposition table.

    Provides _negamax_tt for subclasses that implement iterative-deepening or
    best-node-search strategies.

    Attributes:
        _max_depth: Maximum search depth.
        _heuristic: Heuristic called at depth-0 leaf nodes.
    """

    def __init__(self, name: str, max_depth: int) -> None:
        super().__init__(name, max_depth)
        self._max_depth = max_depth
        self._heuristic: BaseHeuristic = Heuristic()

    def _negamax_tt(
        self,
        state: State,
        alpha: float,
        beta: float,
        depth: int,
        tt: TranspositionTable,
    ) -> float:
        """Depth-limited negamax with alpha-beta pruning and TT lookup/store.

        Uses the heuristic at depth 0 and stores bounds under all 8
        symmetry-equivalent Zobrist hashes.

        Args:
            state: Current game state.
            alpha: Lower bound on the value the current player can guarantee.
            beta: Upper bound imposed by the ancestor node.
            depth: Remaining plies to search.
            tt: Transposition table shared across calls.

        Returns:
            Score from the current player's perspective.
        """
        h = state._hash

        entry = tt.lookup_at_depth(h, depth)
        if entry is not None:
            lb, ub, _ = entry
            if lb >= beta:
                return lb
            if ub <= alpha:
                return ub
            alpha = max(alpha, lb)
            beta = min(beta, ub)
            if alpha >= beta:
                return lb

        original_alpha = alpha
        original_beta = beta

        if state.is_terminal():
            g = -self._terminal_score(state)
            tt.store_symmetric(state._hashes, g, g, None, state.board.n, depth)
            return g

        if depth == 0:
            return self._heuristic.evaluate(state)

        g = NEGATIVE_INFINITY
        best_move: tuple[int, int] | None = None

        for row, col in state.board.get_candidate_cells(state.history, state.candidate_d):
            state.apply(row, col)
            score = -self._negamax_tt(state, -beta, -alpha, depth - 1, tt)
            state.undo()
            if score > g:
                g = score
                best_move = (row, col)
            if g >= beta:
                break
            if g > alpha:
                alpha = g

        existing = tt.lookup(h) if tt.depth_of(h) == depth else None
        lb = existing[0] if existing is not None else NEGATIVE_INFINITY
        ub = existing[1] if existing is not None else -NEGATIVE_INFINITY

        if g <= original_alpha:
            ub = g
        elif g >= original_beta:
            lb = g
        else:
            lb = ub = g

        tt.store_symmetric(state._hashes, lb, ub, best_move, state.board.n, depth)
        return g
