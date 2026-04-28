from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.state import State


class BaseHeuristic(ABC):
    @abstractmethod
    def evaluate(self, state: State) -> float:
        """Return a score in [-1.0, 1.0] from state.current_player's perspective.

        Positive = current player is winning. Called only on non-terminal states.
        """
        ...
