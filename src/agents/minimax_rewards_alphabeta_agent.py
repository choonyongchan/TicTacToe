from src.agents.base_agent import BaseAgent
from src.core.state import State
from src.core.types import INFINITY, NEGATIVE_INFINITY, Player


class MinimaxRewardsAlphaBetaAgent(BaseAgent):
    def __init__(self, maximizer: Player, max_depth: int) -> None:
        super().__init__("MinimaxRewardsAlphaBetaAgent")
        self.maximizer = maximizer
        # epsilon = 1/(max_depth+1) guarantees win>0 and loss<0 at any reachable depth
        self._epsilon = 1.0 / (max_depth + 1)

    def act(self, state: State) -> tuple[int, int]:
        best_score = NEGATIVE_INFINITY
        best_move: tuple[int, int] | None = None
        alpha = NEGATIVE_INFINITY
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = self._minimax(state, alpha, INFINITY)
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

    def _minimax(self, state: State, alpha: float, beta: float) -> float:
        if state.is_terminal():
            return self._terminal_score(state)
        if state.current_player == self.maximizer:
            return self._maximize(state, alpha, beta)
        return self._minimize(state, alpha, beta)

    def _maximize(self, state: State, alpha: float, beta: float) -> float:
        best = NEGATIVE_INFINITY
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = self._minimax(state, alpha, beta)
            state.undo()
            if score > best:
                best = score
            if best >= beta:
                break
            if best > alpha:
                alpha = best
        return best

    def _minimize(self, state: State, alpha: float, beta: float) -> float:
        best = INFINITY
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = self._minimax(state, alpha, beta)
            state.undo()
            if score < best:
                best = score
            if best <= alpha:
                break
            if best < beta:
                beta = best
        return best
