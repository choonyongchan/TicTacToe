from __future__ import annotations

from src.agents.tt_depth_agent import TTDepthAgent
from src.core.forced_move import ForcedMove
from src.core.state import State
from src.core.transposition_table import TranspositionTable
from src.core.types import NEGATIVE_INFINITY


class MTDfIDAgent(TTDepthAgent):
    def __init__(self, max_depth: int) -> None:
        super().__init__("MTDfIDAgent", max_depth)

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

    def _mtdf(
        self, state: State, f: float, depth: int, tt: TranspositionTable
    ) -> float:
        lower = NEGATIVE_INFINITY
        upper = -NEGATIVE_INFINITY

        while lower < upper:
            beta = f if f > lower else lower + self._epsilon
            g = self._negamax_tt(state, beta - self._epsilon, beta, depth, tt)
            if g < beta:
                upper = g
            else:
                lower = g
            f = g

        return f
