"""MCTS with informed rollouts: play a forced win/block, else the heuristic-best move."""

from __future__ import annotations

import math
from typing import Iterable

from src.agents.mc_uct_agent import MCUCTAgent
from src.core.forced_move import ForcedMove
from src.core.state import State
from src.heuristics.heuristic import Heuristic


class MCInformedAgent(MCUCTAgent):
    """MCUCTAgent whose rollout policy checks for a forced move first.

    Only _rollout_move is overridden — tree structure, UCT selection, and
    backpropagation are inherited unchanged from MCUCTAgent.

    Attributes:
        heuristics: Names of the Heuristic sub-scores guiding rollout move
            choice when no forced move exists (see HEURISTIC_REGISTRY).
    """

    def __init__(
        self,
        n_simulations: int = 200,
        c: float = math.sqrt(2),
        seed: int | None = None,
        heuristics: Iterable[str] | None = None,
    ) -> None:
        super().__init__(n_simulations, c, seed, name="MCInformedAgent")
        self._heuristic = Heuristic(heuristics)
        self.heuristics = self._heuristic.enabled

    def _rollout_move(self, state: State) -> tuple[int, int]:
        """Play an immediate win/block if one exists, else the heuristic-best move.

        Args:
            state: Current (non-terminal) game state.

        Returns:
            (row, col) of the chosen move.
        """
        forced = ForcedMove.detect(state)
        if forced is not None:
            return forced

        best_move = None
        best_score = math.inf
        for move in state.board.get_empty_cells():
            state.apply(*move)
            # A terminal result here can only be a draw (wins/blocks are
            # already handled by ForcedMove above), so it scores as neutral.
            score = 0.0 if state.is_terminal() else self._heuristic.evaluate(state)
            state.undo()
            if score < best_score:
                best_score = score
                best_move = move
        return best_move
