from __future__ import annotations

from .board import Board
from .manipulator import Manipulator
from .types import Player
from .zobrist import ZobristTable


class State:
    """Mutable game state tracking board, turn, move history, and Zobrist hash.

    Attributes:
        board: The underlying Board instance.
        current_player: Player whose turn it is.
        history: Ordered list of (row, col) moves applied so far.
        candidate_d: Chebyshev radius used for candidate cell generation.
    """

    def __init__(self, n: int = 3, k: int = 3) -> None:
        self.board = Board(n, k)
        self._zobrist = ZobristTable(n)
        self.current_player: Player = Player.X
        self.history: list[tuple[int, int]] = []
        self._state_count: int = 0
        self._visited: set[int] = set()
        self._hash: int = 0
        self._hashes: list[int] = [0] * Manipulator.TRANSFORM_COUNT
        self.candidate_d: int = max(1, self.board.k - 2)

    def apply(self, row: int, col: int) -> None:
        """Place the current player's piece and advance the turn.

        Also updates the Zobrist hash and all symmetry-equivalent hashes.

        Args:
            row: Row of the cell to play.
            col: Column of the cell to play.
        """
        self.board.set(row, col, self.current_player)
        self.history.append((row, col))
        player_val = int(self.current_player)
        self._hash = self._zobrist.hash_move(self._hash, row, col, player_val)
        for i, (tr, tc) in enumerate(
            Manipulator.all_transform_moves((row, col), self.board.n)
        ):
            self._hashes[i] ^= int(self._zobrist._table[tr, tc, player_val])
        if self._hash not in self._visited:
            self._visited.add(self._hash)
            self._state_count += 1
        self.current_player = self.current_player.opponent()

    def undo(self) -> None:
        """Remove the last placed piece and revert the turn."""
        row, col = self.history.pop()
        prev_player = self.current_player.opponent()
        prev_val = int(prev_player)
        self._hash = self._zobrist.hash_move(self._hash, row, col, prev_val)
        for i, (tr, tc) in enumerate(
            Manipulator.all_transform_moves((row, col), self.board.n)
        ):
            self._hashes[i] ^= int(self._zobrist._table[tr, tc, prev_val])
        self.board.set(row, col, Player._)
        self.current_player = prev_player

    def is_terminal(self) -> bool:
        """Return True if the game has ended (win or draw)."""
        if not self.history:
            return False
        if self.board.check_win(*self.history[-1]):
            return True
        return self.board.is_full()

    def winner(self) -> Player | None:
        """Return the winning player, or None if the game is not yet won.

        Returns:
            Winning Player if the last move completed a run, else None.
        """
        if not self.history:
            return None
        if self.board.check_win(*self.history[-1]):
            return self.current_player.opponent()
        return None

    @property
    def state_count(self) -> int:
        """Total distinct board positions visited across the game."""
        return self._state_count

    def reset(self) -> None:
        """Reset all state to the beginning of a new game."""
        self.board.reset()
        self.current_player = Player.X
        self.history = []
        self._state_count = 0
        self._visited = set()
        self._hash = 0
        self._hashes = [0] * Manipulator.TRANSFORM_COUNT
