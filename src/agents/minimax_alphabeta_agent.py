from src.agents.minimax_base_agent import MinimaxBaseAgent
from src.core.state import State
from src.core.types import Player


class MinimaxAlphaBetaAgent(MinimaxBaseAgent):
    """Exact minimax agent with alpha-beta pruning."""

    def __init__(self, maximizer: Player) -> None:
        super().__init__("MinimaxAlphaBetaAgent", maximizer)

    def act(self, state: State) -> tuple[int, int]:
        """Return the exact best move via alpha-beta minimax.

        Args:
            state: Current game state.

        Returns:
            (row, col) of the best move for the maximizer.
        """
        best_score = -2
        best_move: tuple[int, int] | None = None
        alpha = -2
        for row, col in state.board.get_empty_cells():
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
        """Return +1 for a maximizer win, -1 for a loss, 0 for a draw.

        Args:
            state: Terminal game state.
        """
        winner = state.winner()
        if winner == self._maximizer:
            return 1
        if winner is not None:
            return -1
        return 0

    def _minimax(self, state: State, alpha: int, beta: int) -> int:
        """Recursive alpha-beta minimax dispatch.

        Args:
            state: Current game state.
            alpha: Best score the maximizer can guarantee.
            beta: Best score the minimizer can guarantee.

        Returns:
            Score from the maximizer's perspective.
        """
        if state.is_terminal():
            return self._terminal_score(state)
        if state.current_player == self._maximizer:
            return self._maximize(state, alpha, beta)
        return self._minimize(state, alpha, beta)

    def _maximize(self, state: State, alpha: int, beta: int) -> int:
        """Return the maximum score reachable with alpha-beta pruning.

        Args:
            state: Current game state (maximizer's turn).
            alpha: Current lower bound.
            beta: Current upper bound.
        """
        best = -2
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

    def _minimize(self, state: State, alpha: int, beta: int) -> int:
        """Return the minimum score reachable with alpha-beta pruning.

        Args:
            state: Current game state (minimizer's turn).
            alpha: Current lower bound.
            beta: Current upper bound.
        """
        best = 2
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
