from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.state import State
from src.core.types import NEGATIVE_INFINITY


class BNSAgent(BaseAgent):
    def __init__(self, max_depth: int) -> None:
        super().__init__("BNSAgent")
        self._epsilon = 1.0 / (max_depth + 1)

    def act(self, state: State) -> tuple[int, int]:
        raise NotImplementedError

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
