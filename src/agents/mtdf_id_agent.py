from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.forced_move import ForcedMove
from src.core.state import State
from src.core.transposition_table import TranspositionTable
from src.core.types import NEGATIVE_INFINITY
from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.heuristic import Heuristic


class MTDfIDAgent(BaseAgent):
    def __init__(self, max_depth: int) -> None:
        super().__init__("MTDfIDAgent")
        self._epsilon = 1.0 / (max_depth + 1)
        self._max_depth = max_depth
        self._heuristic: BaseHeuristic = Heuristic()

    def act(self, state: State) -> tuple[int, int]:
        forced = ForcedMove.detect(state)
        if forced is not None:
            return forced
        tt = TranspositionTable()
        f = 0.0
        for depth in range(1, self._max_depth + 1):
            f = self._mtdf(state, f, depth, tt)
        best = tt.best_move(state._hash)
        assert best is not None
        return best

    def _terminal_score(self, state: State) -> float:
        if state.winner() is None:
            return 0.0
        return 1.0 - self._epsilon * len(state.history)

    def _negamax_tt(
        self,
        state: State,
        alpha: float,
        beta: float,
        depth: int,
        tt: TranspositionTable,
    ) -> float:
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

    def _mtdf(
        self, state: State, f: float, depth: int, tt: TranspositionTable
    ) -> float:
        lower = NEGATIVE_INFINITY
        upper = -NEGATIVE_INFINITY  # +inf

        while lower < upper:
            beta = f if f > lower else lower + self._epsilon
            g = self._negamax_tt(state, beta - self._epsilon, beta, depth, tt)
            if g < beta:
                upper = g
            else:
                lower = g
            f = g

        return f
