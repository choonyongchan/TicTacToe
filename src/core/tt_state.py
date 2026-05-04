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

    def apply(self, row: int, col: int) -> None:
        """Place a piece and update all 8 symmetry-equivalent Zobrist hashes.

        Args:
            row: Row of the cell to play.
            col: Column of the cell to play.
        """
        super().apply(row, col)
        # current_player is already flipped after super(); opponent() recovers who just played
        player_val = int(self.current_player.opponent())
        for i, (tr, tc) in enumerate(
            Manipulator.all_transform_moves((row, col), self.board.n)
        ):
            self._hashes[i] ^= int(self._zobrist._table[tr, tc, player_val])

    def undo(self) -> None:
        """Remove the last placed piece and restore all 8 symmetry-equivalent Zobrist hashes.

        XOR is self-inverse, so applying the same XOR restores the previous hash values.
        """
        row, col = self.history[-1]
        # current_player before undo is the NEXT player; opponent() is who played the move
        player_val = int(self.current_player.opponent())
        super().undo()
        for i, (tr, tc) in enumerate(
            Manipulator.all_transform_moves((row, col), self.board.n)
        ):
            self._hashes[i] ^= int(self._zobrist._table[tr, tc, player_val])
