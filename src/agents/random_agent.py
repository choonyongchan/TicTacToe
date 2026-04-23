import random

from src.agents.base_agent import BaseAgent
from src.core.state import State


class RandomAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("RandomAgent")

    def act(self, state: State) -> tuple[int, int]:
        empty = state.board.get_empty_cells()
        return random.choice(empty)
