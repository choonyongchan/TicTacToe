from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.state import State
from src.core.types import NEGATIVE_INFINITY


class BNSAgent(BaseAgent):
    def __init__(self, max_depth: int) -> None:
        super().__init__("BNSAgent")
        self._epsilon = 1.0 / (max_depth + 1)

    def act(self, state: State) -> tuple[int, int]:
        best = self._bns(state, -1.0, 1.0)
        assert best is not None
        return best

    def _terminal_score(self, state: State) -> float:
        if state.winner() is None:
            return 0.0
        return 1.0 - self._epsilon * len(state.history)

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

    def _bns(self, state: State, alpha: float, beta: float) -> tuple[int, int] | None:
        best_node: tuple[int, int] | None = None
        while True:
            test = (alpha + beta) / 2
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
