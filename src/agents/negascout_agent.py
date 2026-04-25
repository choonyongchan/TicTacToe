from src.agents.base_agent import BaseAgent
from src.core.state import State
from src.core.types import NEGATIVE_INFINITY


class NegascoutAgent(BaseAgent):
    def __init__(self, max_depth: int) -> None:
        super().__init__("NegascoutAgent")
        self._epsilon = 1.0 / (max_depth + 1)

    def act(self, state: State) -> tuple[int, int]:
        best_score = NEGATIVE_INFINITY
        best_move: tuple[int, int] | None = None
        alpha = NEGATIVE_INFINITY

        for i, (row, col) in enumerate(state.board.get_empty_cells()):
            state.apply(row, col)
            if i == 0:
                score = -self._negascout(state, NEGATIVE_INFINITY, -alpha)
            else:
                score = -self._negascout(state, -alpha - 1, -alpha)
                if score > alpha:
                    score = -self._negascout(state, NEGATIVE_INFINITY, -score)
            state.undo()

            if score > best_score:
                best_score = score
                best_move = (row, col)
            if best_score > alpha:
                alpha = best_score

        assert best_move is not None
        return best_move

    def _terminal_score(self, state: State) -> float:
        if state.winner() is None:
            return 0.0
        return 1.0 - self._epsilon * len(state.history)

    def _negascout(self, state: State, alpha: float, beta: float) -> float:
        if state.is_terminal():
            return -self._terminal_score(state)

        best = NEGATIVE_INFINITY
        for i, (row, col) in enumerate(state.board.get_empty_cells()):
            state.apply(row, col)
            if i == 0:
                score = -self._negascout(state, -beta, -alpha)
            else:
                score = -self._negascout(state, -alpha - 1, -alpha)
                if alpha < score < beta:
                    score = -self._negascout(state, -beta, -score)
            state.undo()

            if score > best:
                best = score
            if best >= beta:
                break
            if best > alpha:
                alpha = best

        return best
