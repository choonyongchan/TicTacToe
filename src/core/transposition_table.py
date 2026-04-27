from __future__ import annotations


class TranspositionTable:
    def __init__(self) -> None:
        self._table: dict[int, tuple[float, float, tuple[int, int] | None]] = {}

    def lookup(self, key: int) -> tuple[float, float, tuple[int, int] | None] | None:
        return self._table.get(key)

    def store(
        self,
        key: int,
        lower: float,
        upper: float,
        best_move: tuple[int, int] | None,
    ) -> None:
        self._table[key] = (lower, upper, best_move)

    def best_move(self, key: int) -> tuple[int, int] | None:
        entry = self._table.get(key)
        return entry[2] if entry is not None else None

    def __len__(self) -> int:
        return len(self._table)
