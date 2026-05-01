from __future__ import annotations

from src.agents.tt_depth_agent import TTDepthAgent
from src.core.forced_move import ForcedMove
from src.core.state import State
from src.core.transposition_table import TranspositionTable


class BNSIDAgent(TTDepthAgent):
    def __init__(self, max_depth: int) -> None:
        super().__init__("BNSIDAgent", max_depth)

    def act(self, state: State) -> tuple[int, int]:
        forced = ForcedMove.detect(state)
        if forced is not None:
            return forced
        tt = TranspositionTable()
        best_node: tuple[int, int] | None = None
        for depth in range(1, self._max_depth + 1):
            result = self._bns_tt(state, -1.0, 1.0, depth, tt)
            if result is not None:
                best_node = result
        assert best_node is not None
        return best_node

    def _bns_tt(
        self,
        state: State,
        alpha: float,
        beta: float,
        depth: int,
        tt: TranspositionTable,
    ) -> tuple[int, int] | None:
        best_node: tuple[int, int] | None = None
        while True:
            test = (alpha + beta) / 2
            better_count = 0
            for row, col in state.board.get_candidate_cells(state.history, state.candidate_d):
                state.apply(row, col)
                val = -self._negamax_tt(state, -test, -(test - self._epsilon), depth - 1, tt)
                state.undo()
                if val >= test:
                    better_count += 1
                    best_node = (row, col)
            if better_count > 0:
                alpha = test
            else:
                beta = test
            if beta - alpha < 2 * self._epsilon or better_count == 1:
                break
        return best_node
