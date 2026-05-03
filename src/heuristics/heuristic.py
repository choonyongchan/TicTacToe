from __future__ import annotations

from src.core.state import State
from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.distance_heuristic import DistanceHeuristic


class Heuristic(BaseHeuristic):
    """Ensemble heuristic used as the default leaf evaluator in search agents.

    Currently delegates to DistanceHeuristic; extend by composing additional
    signals here.
    """

    def __init__(self) -> None:
        self._scorer = DistanceHeuristic()

    def evaluate(self, state: State) -> float:
        """Return the heuristic score for the current player.

        Args:
            state: Current (non-terminal) game state.

        Returns:
            Score in [-1.0, 1.0]; positive favours the current player.
        """
        return self._scorer.evaluate(state)
