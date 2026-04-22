from src.agents.base_agent import BaseAgent
from src.core.board import Board
from src.core.state import State
from src.core.types import Player


class MinimaxAlphaBetaAgent(BaseAgent):
    def __init__(self, maximizer: Player) -> None:
        super().__init__("MinimaxAlphaBetaAgent")
        self.maximizer = maximizer

    def act(self, state: State) -> tuple[int, int]:
        best_score = -2
        best_move: tuple[int, int] | None = None
        alpha = -2
        for row, col in Board.get_empty_cells(state.board):
            state.apply(row, col)
            score = self._minimax(state, alpha, 2)
            state.undo()
            if score > best_score:
                best_score = score
                best_move = (row, col)
            alpha = max(alpha, best_score)
        assert best_move is not None
        return best_move

    def _terminal_score(self, state: State) -> int:
        winner = state.winner()
        if winner == self.maximizer:
            return 1
        if winner is not None:
            return -1
        return 0

    def _minimax(self, state: State, alpha: int, beta: int) -> int:
        if state.is_terminal():
            return self._terminal_score(state)
        if state.current_player == self.maximizer:
            return self._maximize(state, alpha, beta)
        return self._minimize(state, alpha, beta)

    def _maximize(self, state: State, alpha: int, beta: int) -> int:
        best = -2
        for row, col in Board.get_empty_cells(state.board):
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

    def _minimize(self, state: State, alpha: int, beta: int) -> int:
        best = 2
        for row, col in Board.get_empty_cells(state.board):
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
