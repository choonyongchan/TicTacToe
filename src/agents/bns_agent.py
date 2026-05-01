from __future__ import annotations

from src.agents.negamax_base_agent import NegamaxBaseAgent
from src.core.state import State
from src.core.types import NEGATIVE_INFINITY


class BNSAgent(NegamaxBaseAgent):
    def __init__(self, max_depth: int) -> None:
        super().__init__("BNSAgent", max_depth)

    def act(self, state: State) -> tuple[int, int]:
        best = self._bns(state, -1.0, 1.0)
        assert best is not None
        return best

    def _alphabeta(self, state: State, alpha: float, beta: float) -> float:
        if state.is_terminal():
            return -self._terminal_score(state)
        best = NEGATIVE_INFINITY
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = -self._alphabeta(state, -beta, -alpha)
            state.undo()
            if score > best:
                best = score
            if best >= beta:
                break
            if best > alpha:
                alpha = best
        return best

    def _heuristics(self, alpha: float, beta: float) -> float:
        return (alpha + beta) / 2

    def _bns(self, state: State, alpha: float, beta: float) -> tuple[int, int] | None:
        best_node: tuple[int, int] | None = None
        while True:
            test = self._heuristics(alpha, beta)
            better_count = 0
            for row, col in state.board.get_empty_cells():
                state.apply(row, col)
                val = -self._alphabeta(state, -test, -(test - self._epsilon))
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
