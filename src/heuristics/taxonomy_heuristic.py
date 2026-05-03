from __future__ import annotations

import numpy as np

from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.heuristic_utils import DIRECTIONS, tanh_normalize
from src.core.state import State


def _taxonomy_score(grid: np.ndarray, n: int, k: int, player_val: int) -> float:
    """Weighted sum of open/half-open maximal runs for player_val.

    Args:
        grid: Raw board array.
        n: Board side length.
        k: Win-condition run length.
        player_val: Integer player value to score.

    Returns:
        Raw weighted run score (un-normalised).
    """
    total = 0.0
    for dr, dc in DIRECTIONS:
        for r in range(n):
            for c in range(n):
                # Only start a new run here if the previous cell is not player_val.
                prev_r, prev_c = r - dr, c - dc
                if 0 <= prev_r < n and 0 <= prev_c < n and grid[prev_r, prev_c] == player_val:
                    continue
                if grid[r, c] != player_val:
                    continue
                # Walk forward counting run length.
                m = 0
                rr, cc = r, c
                while 0 <= rr < n and 0 <= cc < n and grid[rr, cc] == player_val:
                    m += 1
                    rr += dr
                    cc += dc
                # Count open ends (empty, in-bounds cells just outside the run).
                open_ends = 0
                if 0 <= prev_r < n and 0 <= prev_c < n and grid[prev_r, prev_c] == 0:
                    open_ends += 1
                if 0 <= rr < n and 0 <= cc < n and grid[rr, cc] == 0:
                    open_ends += 1
                if open_ends > 0 and m < k:
                    total += open_ends * float(4 ** (m - 1))
    return total


class TaxonomyHeuristic(BaseHeuristic):
    """Heuristic based on a weighted sum of open and half-open runs, normalised via tanh."""

    def evaluate(self, state: State) -> float:
        """Return a tanh-normalised run-taxonomy score.

        Args:
            state: Current (non-terminal) game state.

        Returns:
            Score in [-1.0, 1.0]; positive favours the current player.
        """
        board = state.board
        n, k = board.n, board.k
        if k < 2:
            return 0.0
        me = int(state.current_player)
        opp = int(state.current_player.opponent())
        score_me = _taxonomy_score(board._grid, n, k, me)
        score_opp = _taxonomy_score(board._grid, n, k, opp)
        return tanh_normalize(score_me, score_opp, k)
