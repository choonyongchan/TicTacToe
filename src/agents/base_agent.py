from abc import ABC, abstractmethod

from src.core.board import Board
from src.core.state import State


class BaseAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def act(self, state: State) -> tuple[int, int]: ...

    def validate(self, state: State, row: int, col: int) -> bool:
        return Board.is_in_bounds(row, col) and Board.is_empty(state.board, row, col)
