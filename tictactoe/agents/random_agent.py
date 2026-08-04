import random

from tictactoe.agents.base_agent import BaseAgent
from tictactoe.core.state import State


class RandomAgent(BaseAgent):
    """Agent that selects a uniformly random legal move."""

    def __init__(self) -> None:
        super().__init__("RandomAgent")

    def act(self, state: State) -> tuple[int, int]:
        """Return a random empty cell.

        Args:
            state: Current game state.

        Returns:
            (row, col) of a randomly chosen empty cell.
        """
        empty = state.board.get_empty_cells()
        return random.choice(empty)
