from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.state import State


class BNSAgent(BaseAgent):
    def __init__(self, max_depth: int) -> None:
        super().__init__("BNSAgent")
        self._epsilon = 1.0 / (max_depth + 1)

    def act(self, state: State) -> tuple[int, int]:
        raise NotImplementedError

    def _terminal_score(self, state: State) -> float:
        if state.winner() is None:
            return 0.0
        return 1.0 - self._epsilon * len(state.history)
