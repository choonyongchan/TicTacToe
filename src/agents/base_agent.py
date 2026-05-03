from abc import ABC, abstractmethod

from src.core.state import State


class BaseAgent(ABC):
    """Abstract base for all game-playing agents.

    Attributes:
        name: Human-readable identifier for this agent.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def act(self, state: State) -> tuple[int, int]:
        """Choose a move for the current player.

        Args:
            state: Current game state.

        Returns:
            (row, col) of the chosen move.
        """
        ...

    def validate(self, state: State, row: int, col: int) -> bool:
        """Return True if (row, col) is a legal move in state.

        Args:
            state: Current game state.
            row: Row index to validate.
            col: Column index to validate.
        """
        return state.board.is_in_bounds(row, col) and state.board.is_empty(row, col)
