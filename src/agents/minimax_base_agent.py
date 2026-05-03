from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.state import State
from src.core.types import Player


class MinimaxBaseAgent(BaseAgent):
    """Abstract base for minimax-family agents that maximise a fixed player's score.

    Attributes:
        _maximizer: The player this agent acts on behalf of.
    """

    def __init__(self, name: str, maximizer: Player) -> None:
        super().__init__(name)
        self._maximizer: Player = maximizer
