from __future__ import annotations

import numpy as np

from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.heuristic_utils import DIRECTIONS
from src.core.state import State


def _count_immediate_threats(grid: np.ndarray, n: int, k: int, player_val: int, opp_val: int) -> int:
    """Count k-windows with exactly k-1 player pieces and 1 empty cell (one move from win)."""
    count = 0
    for dr, dc in DIRECTIONS:
        for r in range(n):
            for c in range(n):
                end_r = r + (k - 1) * dr
                end_c = c + (k - 1) * dc
                if not (0 <= end_r < n and 0 <= end_c < n):
                    continue
                player_count = 0
                blocked = False
                empty = 0
                for i in range(k):
                    cell = grid[r + i * dr, c + i * dc]
                    if cell == opp_val:
                        blocked = True
                        break
                    elif cell == player_val:
                        player_count += 1
                    else:
                        empty += 1
                if not blocked and player_count == k - 1 and empty == 1:
                    count += 1
    return count


class ThreatHeuristic(BaseHeuristic):
    def evaluate(self, state: State) -> float:
        board = state.board
        n, k = board.n, board.k
        if k < 2:
            return 0.0
        me = int(state.current_player)
        opp = int(state.current_player.opponent())
        threats_me = _count_immediate_threats(board.board, n, k, me, opp)
        threats_opp = _count_immediate_threats(board.board, n, k, opp, me)
        total = threats_me + threats_opp
        if total == 0:
            return 0.0
        return (threats_me - threats_opp) / total
