from __future__ import annotations

import numpy as np

from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.heuristic_utils import DIRECTIONS, tanh_normalize
from src.core.state import State


def _score_windows(grid: np.ndarray, n: int, k: int, player_val: int, opp_val: int) -> float:
    win_threat = float(4 ** (k - 1))
    total = 0.0
    for dr, dc in DIRECTIONS:
        for r in range(n):
            for c in range(n):
                end_r = r + (k - 1) * dr
                end_c = c + (k - 1) * dc
                if not (0 <= end_r < n and 0 <= end_c < n):
                    continue
                m = 0
                e = 0
                blocked = False
                for i in range(k):
                    cell = grid[r + i * dr, c + i * dc]
                    if cell == opp_val:
                        blocked = True
                        break
                    elif cell == player_val:
                        m += 1
                    else:
                        e += 1
                if blocked or m == 0:
                    continue
                if m == k - 1 and e == 1:
                    total += win_threat
                    continue
                pr, pc = r - dr, c - dc
                nr, nc = r + k * dr, c + k * dc
                left_open = 0 <= pr < n and 0 <= pc < n and int(grid[pr, pc]) == 0
                right_open = 0 <= nr < n and 0 <= nc < n and int(grid[nr, nc]) == 0
                open_ends = int(left_open) + int(right_open)
                if open_ends == 0:
                    continue
                total += open_ends * float(4 ** (m - 1))
    return total


class WindowScorerHeuristic(BaseHeuristic):
    def evaluate(self, state: State) -> float:
        board = state.board
        n, k = board.n, board.k
        if k < 2:
            return 0.0
        me = int(state.current_player)
        opp = int(state.current_player.opponent())
        score_me = _score_windows(board._grid, n, k, me, opp)
        score_opp = _score_windows(board._grid, n, k, opp, me)
        return tanh_normalize(score_me, score_opp, k)
