from __future__ import annotations

from .board import Board
from .manipulator import Manipulator
from .types import Player
from .zobrist import ZobristTable


class State:
    def __init__(self, n: int = 3, k: int = 3) -> None:
        self.board = Board(n, k)
        self._zobrist = ZobristTable(n)
        self.current_player: Player = Player.X
        self.history: list[tuple[int, int]] = []
        self.state_count: int = 0
        self.visited: set[int] = set()
        self._hash: int = 0
        self._hashes: list[int] = [0] * Manipulator.TRANSFORM_COUNT

    def apply(self, row: int, col: int) -> None:
        self.board.set(row, col, self.current_player)
        self.history.append((row, col))
        player_val = int(self.current_player)
        self._hash = self._zobrist.hash_move(self._hash, row, col, player_val)
        for i, (tr, tc) in enumerate(
            Manipulator.all_transform_moves((row, col), self.board.n)
        ):
            self._hashes[i] ^= int(self._zobrist._table[tr, tc, player_val])
        if self._hash not in self.visited:
            self.visited.add(self._hash)
            self.state_count += 1
        self.current_player = self.current_player.opponent()

    def undo(self) -> None:
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
        if not self.history:
            return False
        if self.board.check_win(*self.history[-1]):
            return True
        return self.board.is_full()

    def winner(self) -> Player | None:
        if not self.history:
            return None
        if self.board.check_win(*self.history[-1]):
            return self.current_player.opponent()
        return None

    def reset(self) -> None:
        self.board.reset()
        self.current_player = Player.X
        self.history = []
        self.state_count = 0
        self.visited = set()
        self._hash = 0
        self._hashes = [0] * Manipulator.TRANSFORM_COUNT
