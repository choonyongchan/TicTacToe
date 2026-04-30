from __future__ import annotations

from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.window_scorer_heuristic import WindowScorerHeuristic
from src.core.state import State


class Heuristic(BaseHeuristic):
    def __init__(self) -> None:
        self._scorer = WindowScorerHeuristic()

    def evaluate(self, state: State) -> float:
        return self._scorer.evaluate(state)
