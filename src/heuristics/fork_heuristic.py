from __future__ import annotations

import numpy as np

from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.heuristic_utils import DIRECTIONS
from src.core.state import State


def _count_threats_at(grid: np.ndarray, n: int, k: int, row: int, col: int, player_val: int) -> int:
    """Count directions forming a threat (run length >= k-1, >= 1 open end) through (row, col)."""
    count = 0
    for dr, dc in DIRECTIONS:
        fwd = 0
        r, c = row + dr, col + dc
        while 0 <= r < n and 0 <= c < n and grid[r, c] == player_val:
            fwd += 1
            r += dr
            c += dc
        fwd_open = 0 <= r < n and 0 <= c < n and int(grid[r, c]) == 0

        bwd = 0
        r, c = row - dr, col - dc
        while 0 <= r < n and 0 <= c < n and grid[r, c] == player_val:
            bwd += 1
            r -= dr
            c -= dc
        bwd_open = 0 <= r < n and 0 <= c < n and int(grid[r, c]) == 0

        total = 1 + fwd + bwd
        open_ends = int(fwd_open) + int(bwd_open)
        if total >= k - 1 and open_ends >= 1:
            count += 1
    return count


def _is_fork(grid: np.ndarray, n: int, k: int, row: int, col: int, player_val: int) -> bool:
    """True if placing player_val at (row, col) creates >= 2 simultaneous threats."""
    grid[row, col] = player_val
    threats = _count_threats_at(grid, n, k, row, col, player_val)
    grid[row, col] = 0
    return threats >= 2


class ForkHeuristic(BaseHeuristic):
    def evaluate(self, state: State) -> float:
        board = state.board
        empty_cells = board.get_empty_cells()
        total_empty = len(empty_cells)
        if total_empty == 0:
            return 0.0
        me = int(state.current_player)
        opp = int(state.current_player.opponent())
        grid = board.board.copy()
        n, k = board.n, board.k
        fork_me = sum(1 for r, c in empty_cells if _is_fork(grid, n, k, r, c, me))
        fork_opp = sum(1 for r, c in empty_cells if _is_fork(grid, n, k, r, c, opp))
        return (fork_me - fork_opp) / total_empty
