from __future__ import annotations

from .manipulator import Manipulator
from .state import State


class TTState(State):
    """State subclass that maintains all 8 symmetry-equivalent Zobrist hashes.

    Required by agents that use a TranspositionTable. Plain State does not
    track _hashes, so passing a plain State to a TT agent will raise AttributeError.
    """

    def __init__(self, n: int, k: int) -> None:
        super().__init__(n, k)
        self._hashes: list[int] = [0] * Manipulator.TRANSFORM_COUNT
