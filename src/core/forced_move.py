from __future__ import annotations

import numpy as np

from src.core.state import State
from src.core.types import DIRECTIONS


class ForcedMove:
    @staticmethod
    def detect(state: State) -> tuple[int, int] | None:
        grid = state.board._grid
        n = state.board.n
        k = state.board.k
        cur = int(state.current_player)
        opp = 3 - cur  # Player.X=1, Player.O=2; flip between them

        win = ForcedMove._find_threat(grid, n, k, cur, opp)
        if win is not None:
            return win
        return ForcedMove._find_threat(grid, n, k, opp, cur)

    @staticmethod
    def _find_threat(
        grid: np.ndarray, n: int, k: int, player: int, opponent: int
    ) -> tuple[int, int] | None:
        for dr, dc in DIRECTIONS:
            for r in range(n):
                for c in range(n):
                    end_r = r + dr * (k - 1)
                    end_c = c + dc * (k - 1)
                    if not (0 <= end_r < n and 0 <= end_c < n):
                        continue
                    player_count = 0
                    empty_pos: tuple[int, int] | None = None
                    blocked = False
                    for i in range(k):
                        cr, cc = r + dr * i, c + dc * i
                        val = int(grid[cr, cc])
                        if val == player:
                            player_count += 1
                        elif val == 0:
                            if empty_pos is not None:
                                blocked = True
                                break
                            empty_pos = (cr, cc)
                        elif val == opponent:
                            blocked = True
                            break
                    if not blocked and player_count == k - 1 and empty_pos is not None:
                        return empty_pos
        return None
