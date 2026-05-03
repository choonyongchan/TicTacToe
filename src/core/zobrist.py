from __future__ import annotations

import numpy as np


class ZobristTable:
    """Precomputed random uint64 table for incremental Zobrist hashing."""

    def __init__(self, n: int) -> None:
        rng = np.random.default_rng(42)
        self._table: np.ndarray = rng.integers(
            0, np.iinfo(np.uint64).max, size=(n, n, 3), dtype=np.uint64
        )

    def hash_move(self, current_hash: int, row: int, col: int, player_val: int) -> int:
        """XOR the current hash with the entry for (row, col, player_val).

        Args:
            current_hash: Hash before this move.
            row: Row index of the cell.
            col: Column index of the cell.
            player_val: Integer player value (1 or 2).

        Returns:
            Updated hash after the move.
        """
        return current_hash ^ int(self._table[row, col, player_val])
