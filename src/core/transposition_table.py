from __future__ import annotations


class TranspositionTable:
    def __init__(self) -> None:
        self._table: dict[int, tuple[float, float, tuple[int, int] | None]] = {}
        self._depths: dict[int, int] = {}

    def lookup(self, key: int) -> tuple[float, float, tuple[int, int] | None] | None:
        return self._table.get(key)

    def lookup_at_depth(
        self, key: int, min_depth: int
    ) -> tuple[float, float, tuple[int, int] | None] | None:
        """Return entry only if it was stored at depth >= min_depth."""
        entry = self._table.get(key)
        if entry is None:
            return None
        stored_depth = self._depths.get(key, 0)
        if stored_depth < min_depth:
            # Return only the best_move hint; do not use bounds for cutoffs.
            return None
        return entry

    def store(
        self,
        key: int,
        lower: float,
        upper: float,
        best_move: tuple[int, int] | None,
        depth: int = 0,
    ) -> None:
        if depth < self._depths.get(key, -1):
            return
        self._table[key] = (lower, upper, best_move)
        self._depths[key] = depth

    def best_move(self, key: int) -> tuple[int, int] | None:
        entry = self._table.get(key)
        return entry[2] if entry is not None else None

    def depth_of(self, key: int) -> int:
        return self._depths.get(key, -1)

    def __len__(self) -> int:
        return len(self._table)
