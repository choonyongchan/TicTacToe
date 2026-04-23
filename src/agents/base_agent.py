from abc import ABC, abstractmethod

from src.core.state import State


class BaseAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def act(self, state: State) -> tuple[int, int]: ...

    def validate(self, state: State, row: int, col: int) -> bool:
        return state.board.is_in_bounds(row, col) and state.board.is_empty(row, col)
