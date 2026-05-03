from __future__ import annotations

from src.agents.negamax_base_agent import NegamaxBaseAgent
from src.core.state import State
from src.core.types import NEGATIVE_INFINITY


class BNSAgent(NegamaxBaseAgent):
    """Best Node Search agent: binary-searches the game value via alpha-beta windows."""

    def __init__(self, max_depth: int) -> None:
        super().__init__("BNSAgent", max_depth)

    def act(self, state: State) -> tuple[int, int]:
        """Return the best move found by BNS.

        Args:
            state: Current game state.

        Returns:
            (row, col) of the best move.
        """
        best = self._bns(state, -1.0, 1.0)
        assert best is not None
        return best

    def _alphabeta(self, state: State, alpha: float, beta: float) -> float:
        """Full-tree negamax with alpha-beta pruning (no depth limit).

        Args:
            state: Current game state.
            alpha: Lower bound on the value the current player can guarantee.
            beta: Upper bound imposed by the ancestor node.

        Returns:
            Score from the current player's perspective.
        """
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
        """Return the midpoint test value for the BNS binary search.

        Args:
            alpha: Current lower bound.
            beta: Current upper bound.

        Returns:
            Midpoint between alpha and beta.
        """
        return (alpha + beta) / 2

    def _bns(self, state: State, alpha: float, beta: float) -> tuple[int, int] | None:
        """Run the Best Node Search loop to find the best root move.

        Iteratively narrows [alpha, beta] by counting moves that score >= test.

        Args:
            state: Current game state.
            alpha: Initial lower bound.
            beta: Initial upper bound.

        Returns:
            (row, col) of the best move, or None if no moves exist.
        """
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
