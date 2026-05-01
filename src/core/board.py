from __future__ import annotations

import numpy as np

from .types import Board2D, DIRECTIONS, Player


class Board:
    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k
        self.board: Board2D = np.zeros((n, n), dtype=np.uint8)

    def reset(self) -> None:
        self.board[:] = Player._

    def get(self, row: int, col: int) -> Player:
        return Player(int(self.board[row, col]))

    def set(self, row: int, col: int, player: Player) -> None:
        self.board[row, col] = int(player)

    def is_empty(self, row: int, col: int) -> bool:
        return int(self.board[row, col]) == Player._

    def is_full(self) -> bool:
        return not np.any(self.board == Player._)

    def get_empty_cells(self) -> list[tuple[int, int]]:
        rows, cols = np.where(self.board == Player._)
        return list(zip(rows.tolist(), cols.tolist()))

    def get_candidate_cells(
        self, history: list[tuple[int, int]], d: int
    ) -> list[tuple[int, int]]:
        if not history:
            return [(self.n // 2, self.n // 2)]
        if self.n * self.n - len(history) <= (2 * d + 1) ** 2:
            return self.get_empty_cells()
        candidates: set[tuple[int, int]] = set()
        for pr, pc in history:
            for dr in range(-d, d + 1):
                for dc in range(-d, d + 1):
                    r, c = pr + dr, pc + dc
                    if self.is_in_bounds(r, c) and self.is_empty(r, c):
                        candidates.add((r, c))
        return list(candidates)

    def is_in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.n and 0 <= col < self.n

    def _check_direction(
        self, row: int, col: int, player_val: int, dr: int, dc: int
    ) -> int:
        count = 0
        r, c = row + dr, col + dc
        while self.is_in_bounds(r, c) and int(self.board[r, c]) == player_val:
            count += 1
            r += dr
            c += dc
        return count

    def check_win(self, row: int, col: int) -> bool:
        player_val = int(self.board[row, col])
        if player_val == Player._:
            return False
        for dr, dc in DIRECTIONS:
            count = (
                1
                + self._check_direction(row, col, player_val, dr, dc)
                + self._check_direction(row, col, player_val, -dr, -dc)
            )
            if count >= self.k:
                return True
        return False

    def render(self, row: int | None = None, col: int | None = None) -> str:
        symbols = {Player._: ".", Player.X: "X", Player.O: "O"}
        lines = []
        for r in range(self.n):
            row_parts = []
            for c in range(self.n):
                sym = symbols[Player(int(self.board[r, c]))]
                if row is not None and (r, c) == (row, col):
                    row_parts.append(f"[{sym}]")
                else:
                    row_parts.append(f" {sym} ")
            lines.append("|".join(row_parts))
        return "\n".join(lines)
