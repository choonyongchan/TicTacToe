from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.state import State


class BaseHeuristic(ABC):
    """Abstract base for board-position heuristics."""

    @abstractmethod
    def evaluate(self, state: State) -> float:
        """Return a score in [-1.0, 1.0] from state.current_player's perspective.

        Positive = current player is winning. Called only on non-terminal states.

        Args:
            state: Current (non-terminal) game state.

        Returns:
            Heuristic value in [-1.0, 1.0].
        """
        ...
