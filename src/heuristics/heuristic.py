from __future__ import annotations

from src.core.state import State
from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.distance_heuristic import DistanceHeuristic


class Heuristic(BaseHeuristic):
    def __init__(self) -> None:
        self._scorer = DistanceHeuristic()

    def evaluate(self, state: State) -> float:
        return self._scorer.evaluate(state)
