from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.state import State
from src.core.transposition_table import TranspositionTable
from src.core.types import NEGATIVE_INFINITY


class MTDfAgent(BaseAgent):
    def __init__(self, max_depth: int) -> None:
        super().__init__("MTDfAgent")
        self._epsilon = 1.0 / (max_depth + 1)

    def act(self, state: State) -> tuple[int, int]:
        tt = TranspositionTable()
        self._mtdf(state, 0.0, tt)
        best = tt.best_move(state._hash)
        if best is not None:
            return best
        # Fallback: full-window sweep (shouldn't trigger in practice)
        best_score = NEGATIVE_INFINITY
        best_move: tuple[int, int] | None = None
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = -self._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, tt)
            state.undo()
            if score > best_score:
                best_score = score
                best_move = (row, col)
        assert best_move is not None
        return best_move

    def _terminal_score(self, state: State) -> float:
        if state.winner() is None:
            return 0.0
        return 1.0 - self._epsilon * len(state.history)

    def _negamax_tt(
        self,
        state: State,
        alpha: float,
        beta: float,
        tt: TranspositionTable,
    ) -> float:
        h = state._hash

        entry = tt.lookup(h)
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

        # Capture window AFTER TT tightening — matches pseudocode "a := alpha"
        # so that fail-low/high classification is relative to the actual window used.
        original_alpha = alpha
        original_beta = beta

        if state.is_terminal():
            g = -self._terminal_score(state)
            tt.store(h, g, g, None)
            return g

        g = NEGATIVE_INFINITY
        best_move: tuple[int, int] | None = None

        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = -self._negamax_tt(state, -beta, -alpha, tt)
            state.undo()
            if score > g:
                g = score
                best_move = (row, col)
            if g >= beta:
                break
            if g > alpha:
                alpha = g

        # Merge new bound with the other existing bound (preserves prior probe's work)
        existing = tt.lookup(h)
        lb = existing[0] if existing is not None else NEGATIVE_INFINITY
        ub = existing[1] if existing is not None else -NEGATIVE_INFINITY

        if g <= original_alpha:
            ub = g
        elif g >= original_beta:
            lb = g
        else:
            lb = ub = g

        tt.store(h, lb, ub, best_move)
        return g

    def _mtdf(self, state: State, f: float, tt: TranspositionTable) -> float:
        lower = NEGATIVE_INFINITY
        upper = -NEGATIVE_INFINITY  # +inf

        while lower < upper:
            beta = f if f > lower else lower + self._epsilon
            g = self._negamax_tt(state, beta - self._epsilon, beta, tt)
            if g < beta:
                upper = g
            else:
                lower = g
            f = g

        return f
