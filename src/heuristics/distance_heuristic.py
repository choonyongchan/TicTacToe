from __future__ import annotations

import numpy as np

from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.heuristic_utils import DIRECTIONS
from src.core.state import State


def _min_gap(grid: np.ndarray, n: int, k: int, player_val: int, opp_val: int) -> int:
    """Minimum moves player still needs to complete any unblocked k-window.

    Args:
        grid: Raw board array.
        n: Board side length.
        k: Win-condition run length.
        player_val: Integer value of the player to evaluate.
        opp_val: Integer value of the opponent (blocks windows).

    Returns:
        Minimum number of empty cells the player must fill to complete a run.
    """
    min_g = k
    for dr, dc in DIRECTIONS:
        for r in range(n):
            for c in range(n):
                end_r = r + (k - 1) * dr
                end_c = c + (k - 1) * dc
                if not (0 <= end_r < n and 0 <= end_c < n):
                    continue
                count = 0
                blocked = False
                for i in range(k):
                    cell = grid[r + i * dr, c + i * dc]
                    if cell == opp_val:
                        blocked = True
                        break
                    if cell == player_val:
                        count += 1
                if not blocked:
                    min_g = min(min_g, k - count)
    return min_g


class DistanceHeuristic(BaseHeuristic):
    """Heuristic based on minimum moves remaining to win for each player."""

    def evaluate(self, state: State) -> float:
        """Return (opponent_gap - my_gap) / (k-1), clamped to [-1, 1].

        Args:
            state: Current (non-terminal) game state.

        Returns:
            Score in [-1.0, 1.0]; positive favours the current player.
        """
        board = state.board
        n, k = board.n, board.k
        if k <= 1:
            return 0.0
        me = int(state.current_player)
        opp = int(state.current_player.opponent())
        dist_me = _min_gap(board._grid, n, k, me, opp)
        dist_opp = _min_gap(board._grid, n, k, opp, me)
        return max(-1.0, min(1.0, (dist_opp - dist_me) / (k - 1)))
