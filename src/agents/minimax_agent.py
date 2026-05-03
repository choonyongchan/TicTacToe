from src.agents.minimax_base_agent import MinimaxBaseAgent
from src.core.state import State
from src.core.types import Player


class MinimaxAgent(MinimaxBaseAgent):
    """Exact minimax agent without alpha-beta pruning."""

    def __init__(self, maximizer: Player) -> None:
        super().__init__("MinimaxAgent", maximizer)

    def act(self, state: State) -> tuple[int, int]:
        """Return the exact best move via full minimax search.

        Args:
            state: Current game state.

        Returns:
            (row, col) of the best move for the maximizer.
        """
        best_score = -2
        best_move: tuple[int, int] | None = None
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = self._minimax(state)
            state.undo()
            if score > best_score:
                best_score = score
                best_move = (row, col)
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

    def _minimax(self, state: State) -> int:
        """Recursive minimax dispatch.

        Args:
            state: Current game state.

        Returns:
            Score from the maximizer's perspective.
        """
        if state.is_terminal():
            return self._terminal_score(state)
        if state.current_player == self._maximizer:
            return self._maximize(state)
        return self._minimize(state)

    def _maximize(self, state: State) -> int:
        """Return the maximum score reachable from state.

        Args:
            state: Current game state (maximizer's turn).
        """
        best = -2
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = self._minimax(state)
            state.undo()
            if score > best:
                best = score
        return best

    def _minimize(self, state: State) -> int:
        """Return the minimum score reachable from state.

        Args:
            state: Current game state (minimizer's turn).
        """
        best = 2
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = self._minimax(state)
            state.undo()
            if score < best:
                best = score
        return best
