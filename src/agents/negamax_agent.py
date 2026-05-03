from src.agents.negamax_base_agent import NegamaxBaseAgent
from src.core.state import State
from src.core.types import NEGATIVE_INFINITY


class NegamaxAgent(NegamaxBaseAgent):
    """Exact negamax agent with alpha-beta pruning and no depth limit."""

    def __init__(self, max_depth: int) -> None:
        super().__init__("NegamaxAgent", max_depth)

    def act(self, state: State) -> tuple[int, int]:
        """Return the exact best move via full negamax search.

        Args:
            state: Current game state.

        Returns:
            (row, col) of the best move.
        """
        best_score = NEGATIVE_INFINITY
        best_move: tuple[int, int] | None = None
        alpha = NEGATIVE_INFINITY
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = -self._negamax(state, NEGATIVE_INFINITY, -alpha)
            state.undo()
            if score > best_score:
                best_score = score
                best_move = (row, col)
            if best_score > alpha:
                alpha = best_score
        assert best_move is not None
        return best_move

    def _negamax(self, state: State, alpha: float, beta: float) -> float:
        """Recursive negamax with alpha-beta pruning.

        Args:
            state: Current game state.
            alpha: Lower bound on the value the maximiser can guarantee.
            beta: Upper bound from the minimiser's perspective.

        Returns:
            Score from the current player's perspective.
        """
        if state.is_terminal():
            # Current player is the loser; negate the winner's positive reward.
            return -self._terminal_score(state)
        best = NEGATIVE_INFINITY
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = -self._negamax(state, -beta, -alpha)
            state.undo()
            if score > best:
                best = score
            if best >= beta:
                break
            if best > alpha:
                alpha = best
        return best
