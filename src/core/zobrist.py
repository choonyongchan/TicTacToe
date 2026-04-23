from __future__ import annotations

import numpy as np


class ZobristTable:
    def __init__(self, n: int) -> None:
        rng = np.random.default_rng(42)
        self._table: np.ndarray = rng.integers(
            0, np.iinfo(np.uint64).max, size=(n, n, 3), dtype=np.uint64
        )

    def hash_move(self, current_hash: int, row: int, col: int, player_val: int) -> int:
        return current_hash ^ int(self._table[row, col, player_val])
