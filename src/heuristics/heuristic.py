from __future__ import annotations

from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.distance_heuristic import DistanceHeuristic
from src.heuristics.taxonomy_heuristic import TaxonomyHeuristic
from src.heuristics.fork_heuristic import ForkHeuristic
from src.core.state import State


class Heuristic(BaseHeuristic):
    def __init__(self) -> None:
        self._components: list[BaseHeuristic] = [
            DistanceHeuristic(),
            TaxonomyHeuristic(),
            ForkHeuristic(),
        ]

    def evaluate(self, state: State) -> float:
        total = sum(h.evaluate(state) for h in self._components)
        return total / len(self._components)
