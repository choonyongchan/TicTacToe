from __future__ import annotations

from .board import Board as _board
from .types import Board2D, Player, ZOBRIST_TABLE


class State:
    def __init__(self) -> None:
        self.board: Board2D = _board.create()
        self.current_player: Player = Player.X
        self.history: list[tuple[int, int]] = []
        self.state_count: int = 0
        self.visited: set[int] = set()
        self._hash: int = 0

    def apply(self, row: int, col: int) -> None:
        _board.set(self.board, row, col, self.current_player)
        self.history.append((row, col))
        self._hash ^= int(ZOBRIST_TABLE[row, col, int(self.current_player)])
        if self._hash not in self.visited:
            self.visited.add(self._hash)
            self.state_count += 1
        self.current_player = self.current_player.opponent()

    def undo(self) -> None:
        row, col = self.history.pop()
        prev_player = self.current_player.opponent()
        self._hash ^= int(ZOBRIST_TABLE[row, col, int(prev_player)])
        self.board[row, col] = Player._
        self.current_player = prev_player

    def is_terminal(self) -> bool:
        if not self.history:
            return False
        if _board.check_win(self.board, *self.history[-1]):
            return True
        return _board.is_full(self.board)

    def winner(self) -> Player | None:
        if not self.history:
            return None
        if _board.check_win(self.board, *self.history[-1]):
            return self.current_player.opponent()
        return None

    def reset(self) -> None:
        _board.reset(self.board)
        self.current_player = Player.X
        self.history = []
        self.state_count = 0
        self.visited = set()
        self._hash = 0
