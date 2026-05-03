from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.state import State


class NegamaxBaseAgent(BaseAgent):
    """Base class for negamax-family agents providing depth-adjusted terminal scoring.

    Attributes:
        _epsilon: Per-ply score discount that penalises slower wins.
    """

    def __init__(self, name: str, max_depth: int) -> None:
        super().__init__(name)
        self._epsilon: float = 1.0 / (max_depth + 1)

    def _terminal_score(self, state: State) -> float:
        """Return a positive score for the player who just won, scaled by depth.

        Draws return 0. A win at depth d returns 1 - epsilon*d, ensuring the
        agent prefers faster victories and slower losses.

        Args:
            state: Terminal game state (is_terminal() must be True).

        Returns:
            Score in (0, 1] for a win, or 0.0 for a draw.
        """
        if state.winner() is None:
            return 0.0
        return 1.0 - self._epsilon * len(state.history)
