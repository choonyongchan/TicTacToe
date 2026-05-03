from __future__ import annotations

from src.agents.tt_depth_agent import TTDepthAgent
from src.core.forced_move import ForcedMove
from src.core.state import State
from src.core.transposition_table import TranspositionTable
from src.core.types import NEGATIVE_INFINITY


class MTDfIDAgent(TTDepthAgent):
    """Iterative-deepening MTD(f) agent with forced-move detection and TT reuse."""

    def __init__(self, max_depth: int) -> None:
        super().__init__("MTDfIDAgent", max_depth)

    def act(self, state: State) -> tuple[int, int]:
        """Return the best move via iterative-deepening MTD(f).

        Short-circuits with a forced move when one exists.

        Args:
            state: Current game state.

        Returns:
            (row, col) of the best move.
        """
        forced = ForcedMove.detect(state)
        if forced is not None:
            return forced
        tt = TranspositionTable()
        f = 0.0
        for depth in range(1, self._max_depth + 1):
            f = self._mtdf(state, f, depth, tt)
        best = tt.best_move(state._hash)
        assert best is not None
        return best

    def _mtdf(
        self, state: State, f: float, depth: int, tt: TranspositionTable
    ) -> float:
        """Run the MTD(f) loop at a fixed depth.

        Args:
            state: Current game state.
            f: Initial game-value guess.
            depth: Search depth for this pass.
            tt: Transposition table shared across iterative-deepening passes.

        Returns:
            Converged game value at the given depth.
        """
        lower = NEGATIVE_INFINITY
        upper = -NEGATIVE_INFINITY

        while lower < upper:
            beta = f if f > lower else lower + self._epsilon
            g = self._negamax_tt(state, beta - self._epsilon, beta, depth, tt)
            if g < beta:
                upper = g
            else:
                lower = g
            f = g

        return f
