from __future__ import annotations

from src.agents.negamax_base_agent import NegamaxBaseAgent
from src.core.state import State
from src.core.transposition_table import TranspositionTable
from src.core.types import NEGATIVE_INFINITY


class MTDfAgent(NegamaxBaseAgent):
    """Single-pass MTD(f) agent: binary-probes a TT-backed negamax at fixed depth."""

    def __init__(self, max_depth: int) -> None:
        super().__init__("MTDfAgent", max_depth)

    def act(self, state: State) -> tuple[int, int]:
        """Return the best move found by a single MTD(f) pass.

        Args:
            state: Current game state.

        Returns:
            (row, col) of the best move.
        """
        tt = TranspositionTable()
        self._mtdf(state, 0.0, tt)
        best = tt.best_move(state._hash)
        if best is not None:
            return best
        # Fallback: full-window sweep (shouldn't trigger in practice)
        best_score = NEGATIVE_INFINITY
        best_move: tuple[int, int] | None = None
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = -self._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, tt)
            state.undo()
            if score > best_score:
                best_score = score
                best_move = (row, col)
        assert best_move is not None
        return best_move

    def _negamax_tt(
        self,
        state: State,
        alpha: float,
        beta: float,
        tt: TranspositionTable,
    ) -> float:
        """Negamax with alpha-beta and TT lookup/store (no depth limit).

        Args:
            state: Current game state.
            alpha: Lower bound on the value the current player can guarantee.
            beta: Upper bound imposed by the ancestor node.
            tt: Transposition table.

        Returns:
            Score from the current player's perspective.
        """
        h = state._hash

        entry = tt.lookup(h)
        if entry is not None:
            lb, ub, _ = entry
            if lb >= beta:
                return lb
            if ub <= alpha:
                return ub
            alpha = max(alpha, lb)
            beta = min(beta, ub)
            if alpha >= beta:
                return lb

        # Capture window AFTER TT tightening — matches pseudocode "a := alpha"
        # so that fail-low/high classification is relative to the actual window used.
        original_alpha = alpha
        original_beta = beta

        if state.is_terminal():
            g = -self._terminal_score(state)
            tt.store_symmetric(state._hashes, g, g, None, state.board.n)
            return g

        g = NEGATIVE_INFINITY
        best_move: tuple[int, int] | None = None

        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = -self._negamax_tt(state, -beta, -alpha, tt)
            state.undo()
            if score > g:
                g = score
                best_move = (row, col)
            if g >= beta:
                break
            if g > alpha:
                alpha = g

        # Merge new bound with the other existing bound (preserves prior probe's work)
        existing = tt.lookup(h)
        lb = existing[0] if existing is not None else NEGATIVE_INFINITY
        ub = existing[1] if existing is not None else -NEGATIVE_INFINITY

        if g <= original_alpha:
            ub = g
        elif g >= original_beta:
            lb = g
        else:
            lb = ub = g

        tt.store_symmetric(state._hashes, lb, ub, best_move, state.board.n)
        return g

    def _mtdf(self, state: State, f: float, tt: TranspositionTable) -> float:
        """Run the MTD(f) loop: repeatedly narrow [lower, upper] via null-window probes.

        Args:
            state: Current game state.
            f: Initial guess for the game value.
            tt: Transposition table (shared across probes).

        Returns:
            Converged game value.
        """
        lower = NEGATIVE_INFINITY
        upper = -NEGATIVE_INFINITY  # +inf

        while lower < upper:
            beta = f if f > lower else lower + self._epsilon
            g = self._negamax_tt(state, beta - self._epsilon, beta, tt)
            if g < beta:
                upper = g
            else:
                lower = g
            f = g

        return f
