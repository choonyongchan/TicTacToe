from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.state import State
from src.core.types import Player


class MinimaxBaseAgent(BaseAgent):
    def __init__(self, name: str, maximizer: Player) -> None:
        super().__init__(name)
        self._maximizer: Player = maximizer
