from __future__ import annotations

import numpy as np

from .types import Board2D, DIRECTIONS, Player


class Board:
    """Stateless n×n game grid with win detection and candidate-cell selection.

    Attributes:
        n: Board side length.
        k: Run length required to win.
    """

    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k
        self._grid: Board2D = np.zeros((n, n), dtype=np.uint8)

    def reset(self) -> None:
        """Clear all cells to empty."""
        self._grid[:] = Player._

    def get(self, row: int, col: int) -> Player:
        """Return the occupant of a cell.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            Player occupying the cell, or Player._ if empty.
        """
        return Player(int(self._grid[row, col]))

    def set(self, row: int, col: int, player: Player) -> None:
        """Place a player's piece on a cell.

        Args:
            row: Row index.
            col: Column index.
            player: Player to place.
        """
        self._grid[row, col] = int(player)

    def is_empty(self, row: int, col: int) -> bool:
        """Return True if the cell has no piece.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            True if the cell is unoccupied.
        """
        return int(self._grid[row, col]) == Player._

    def is_full(self) -> bool:
        """Return True if every cell is occupied."""
        return not np.any(self._grid == Player._)

    def get_empty_cells(self) -> list[tuple[int, int]]:
        """Return all empty cell coordinates as (row, col) pairs."""
        rows, cols = np.where(self._grid == Player._)
        return list(zip(rows.tolist(), cols.tolist()))

    def get_candidate_cells(
        self, history: list[tuple[int, int]], d: int
    ) -> list[tuple[int, int]]:
        """Return empty cells within Chebyshev distance d of any played cell.

        Falls back to the board centre on an empty board, or to all empty cells
        when the candidate region would cover the entire board.

        Args:
            history: Sequence of (row, col) moves played so far.
            d: Chebyshev radius around each played cell.

        Returns:
            Candidate empty cells for the next move.
        """
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
        """Return True if (row, col) lies within the board.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            True if (row, col) is within the n×n grid.
        """
        return 0 <= row < self.n and 0 <= col < self.n

    def _check_direction(
        self, row: int, col: int, player_val: int, dr: int, dc: int
    ) -> int:
        """Count consecutive pieces owned by player_val in one direction.

        Args:
            row: Starting row (exclusive — search begins one step away).
            col: Starting column.
            player_val: Integer player value to match.
            dr: Row delta per step.
            dc: Column delta per step.

        Returns:
            Number of matching consecutive pieces found.
        """
        count = 0
        r, c = row + dr, col + dc
        while self.is_in_bounds(r, c) and int(self._grid[r, c]) == player_val:
            count += 1
            r += dr
            c += dc
        return count

    def check_win(self, row: int, col: int) -> bool:
        """Return True if the piece at (row, col) completes a winning run.

        Args:
            row: Row of the most recently placed piece.
            col: Column of the most recently placed piece.
        """
        player_val = int(self._grid[row, col])
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
        """Return a human-readable string of the board.

        Args:
            row: Optional row of a cell to highlight with brackets.
            col: Optional column of a cell to highlight with brackets.

        Returns:
            Multi-line string representation of the board.
        """
        symbols = {Player._: ".", Player.X: "X", Player.O: "O"}
        lines = []
        for r in range(self.n):
            row_parts = []
            for c in range(self.n):
                sym = symbols[Player(int(self._grid[r, c]))]
                if row is not None and (r, c) == (row, col):
                    row_parts.append(f"[{sym}]")
                else:
                    row_parts.append(f" {sym} ")
            lines.append("|".join(row_parts))
        return "\n".join(lines)
