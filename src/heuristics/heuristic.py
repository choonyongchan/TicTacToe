from __future__ import annotations

from typing import Iterable

from src.core.state import State
from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.distance_heuristic import DistanceHeuristic
from src.heuristics.fork_heuristic import ForkHeuristic
from src.heuristics.taxonomy_heuristic import TaxonomyHeuristic
from src.heuristics.threat_heuristic import ThreatHeuristic
from src.heuristics.window_scorer_heuristic import WindowScorerHeuristic

# Registry of every available heuristic signal, keyed by a toggle name.
# Pass a subset of these keys to Heuristic(enabled=...) to evaluate which
# heuristics (or combinations of them) perform best.
HEURISTIC_REGISTRY: dict[str, type[BaseHeuristic]] = {
    "distance": DistanceHeuristic,
    "fork": ForkHeuristic,
    "taxonomy": TaxonomyHeuristic,
    "threat": ThreatHeuristic,
    "window_scorer": WindowScorerHeuristic,
}


class Heuristic(BaseHeuristic):
    """Ensemble heuristic used as the default leaf evaluator in search agents.

    Averages every heuristic in `enabled` (all of HEURISTIC_REGISTRY by
    default). Pass a smaller `enabled` set to benchmark individual
    heuristics or alternative combinations.

    Attributes:
        enabled: Names of the active heuristics, as passed to __init__.
    """

    def __init__(self, enabled: Iterable[str] | None = None) -> None:
        self.enabled = tuple(enabled) if enabled is not None else tuple(HEURISTIC_REGISTRY)
        self._scorers = [HEURISTIC_REGISTRY[name]() for name in self.enabled]

    def evaluate(self, state: State) -> float:
        """Return the mean score across all enabled heuristics.

        Args:
            state: Current (non-terminal) game state.

        Returns:
            Score in [-1.0, 1.0]; positive favours the current player.
        """
        return sum(s.evaluate(state) for s in self._scorers) / len(self._scorers)
