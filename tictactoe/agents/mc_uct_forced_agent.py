"""MCUCTAgent with forced-move-only rollouts (Q15: isolate ForcedMove.detect from heuristics)."""

from __future__ import annotations

import math

from tictactoe.agents.mc_uct_agent import MCUCTAgent
from tictactoe.core.forced_move import ForcedMove
from tictactoe.core.state import State


class MCUCTForcedAgent(MCUCTAgent):
    """MCUCTAgent whose rollout policy checks for a forced move first, else uniform random.

    Isolates ForcedMove.detect from MCInformedAgent's heuristic tie-break
    (Q11 found the heuristic contributes nothing beyond forced-move
    detection on 3x3) to test whether ForcedMove.detect alone closes the
    move-accuracy gap between MCUCTAgent and MCInformedAgent.
    """

    def __init__(
        self,
        n_simulations: int = 200,
        c: float = math.sqrt(2),
        seed: int | None = None,
    ) -> None:
        super().__init__(n_simulations, c, seed, name="MCUCTForcedAgent")

    def _rollout_move(self, state: State) -> tuple[int, int]:
        """Play an immediate win/block if one exists, else uniform random.

        Args:
            state: Current (non-terminal) game state.

        Returns:
            (row, col) of the chosen move.
        """
        forced = ForcedMove.detect(state)
        if forced is not None:
            return forced
        return super()._rollout_move(state)
