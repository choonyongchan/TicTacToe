from __future__ import annotations

import numpy as np

from .types import Board2D, DIRECTIONS, Player


class Board:
    @staticmethod
    def create() -> Board2D:
        return np.zeros((3, 3), dtype=np.uint8)

    @staticmethod
    def reset(board: Board2D) -> None:
        board[:] = Player._

    @staticmethod
    def get(board: Board2D, row: int, col: int) -> Player:
        return Player(int(board[row, col]))

    @staticmethod
    def set(board: Board2D, row: int, col: int, player: Player) -> None:
        board[row, col] = int(player)

    @staticmethod
    def is_empty(board: Board2D, row: int, col: int) -> bool:
        return int(board[row, col]) == Player._

    @staticmethod
    def is_full(board: Board2D) -> bool:
        return not np.any(board == Player._)

    @staticmethod
    def get_empty_cells(board: Board2D) -> list[tuple[int, int]]:
        rows, cols = np.where(board == Player._)
        return list(zip(rows.tolist(), cols.tolist()))

    @staticmethod
    def is_in_bounds(row: int, col: int) -> bool:
        return 0 <= row < 3 and 0 <= col < 3

    @staticmethod
    def _check_direction(board: Board2D, row: int, col: int, player_val: int, dr: int, dc: int) -> int:
        count = 0
        r, c = row + dr, col + dc
        while Board.is_in_bounds(r, c) and int(board[r, c]) == player_val:
            count += 1
            r += dr
            c += dc
        return count

    @staticmethod
    def check_win(board: Board2D, row: int, col: int) -> bool:
        player_val = int(board[row, col])
        if player_val == Player._:
            return False
        for dr, dc in DIRECTIONS:
            count = (1
                     + Board._check_direction(board, row, col, player_val,  dr,  dc)
                     + Board._check_direction(board, row, col, player_val, -dr, -dc))
            if count >= 3:
                return True
        return False

    @staticmethod
    def render(board: Board2D, row: int | None = None, col: int | None = None) -> str:
        symbols = {Player._: ".", Player.X: "X", Player.O: "O"}
        lines = []
        for r in range(3):
            row_parts = []
            for c in range(3):
                sym = symbols[Player(int(board[r, c]))]
                if row is not None and (r, c) == (row, col):
                    row_parts.append(f"[{sym}]")
                else:
                    row_parts.append(f" {sym} ")
            lines.append("|".join(row_parts))
        return "\n".join(lines)
