from src.agents.base_agent import BaseAgent
from src.core.state import State
from src.core.types import INFINITY, NEGATIVE_INFINITY, Player


class NegamaxAgent(BaseAgent):
    def __init__(self, maximizer: Player, max_depth: int = 9) -> None:
        super().__init__("NegamaxAgent")
        self.maximizer = maximizer
        self._epsilon = 1.0 / (max_depth + 1)

    def act(self, state: State) -> tuple[int, int]:
        best_score = NEGATIVE_INFINITY
        best_move: tuple[int, int] | None = None
        alpha = NEGATIVE_INFINITY
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = self._negamax(state, alpha, INFINITY)
            state.undo()
            if score > best_score:
                best_score = score
                best_move = (row, col)
            alpha = max(alpha, best_score)
        assert best_move is not None
        return best_move

    def _terminal_score(self, state: State) -> float:
        winner = state.winner()
        depth = len(state.history)
        if winner == self.maximizer:
            return 1.0 - self._epsilon * depth
        if winner is not None:
            return -1.0 + self._epsilon * depth
        return 0.0

    def _negamax(self, state: State, alpha: float, beta: float) -> float:
        if state.is_terminal():
            return self._terminal_score(state)
        is_max = state.current_player == self.maximizer
        best = NEGATIVE_INFINITY if is_max else INFINITY
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = self._negamax(state, alpha, beta)
            state.undo()
            if is_max:
                if score > best:
                    best = score
                if best >= beta:
                    break
                if best > alpha:
                    alpha = best
            else:
                if score < best:
                    best = score
                if best <= alpha:
                    break
                if best < beta:
                    beta = best
        return best
