from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.state import State


class NegamaxBaseAgent(BaseAgent):
    def __init__(self, name: str, max_depth: int) -> None:
        super().__init__(name)
        self._epsilon: float = 1.0 / (max_depth + 1)

    def _terminal_score(self, state: State) -> float:
        if state.winner() is None:
            return 0.0
        return 1.0 - self._epsilon * len(state.history)
