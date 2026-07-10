"""Pure (flat) Monte Carlo agent: the historical predecessor to MCTS/UCT."""

from __future__ import annotations

from src.agents.mc_shared import MonteCarloAgent
from src.core.state import State


class MCPureAgent(MonteCarloAgent):
    """Flat Monte Carlo: averages random rollouts per candidate move, no tree.

    Splits the simulation budget evenly across the root's legal moves and
    plays the move with the highest average rollout reward. No exploration
    bonus, no tree reuse across moves within a ply — this is what UCT was
    invented to improve on.

    Attributes:
        n_simulations: Total rollouts run per act() call, split across moves.
    """

    def __init__(self, n_simulations: int = 200, seed: int | None = None) -> None:
        super().__init__("MCPureAgent", n_simulations, seed)

    def act(self, state: State) -> tuple[int, int]:
        """Choose the move with the highest average random-rollout reward.

        Args:
            state: Current game state.

        Returns:
            (row, col) of the chosen move.
        """
        me = state.current_player
        moves = state.board.get_empty_cells()
        per_move = max(1, self.n_simulations // len(moves))

        best_move, best_avg = moves[0], float("-inf")
        for move in moves:
            state.apply(*move)
            total = sum(self._random_rollout(state, me)[0] for _ in range(per_move))
            state.undo()
            avg = total / per_move
            if avg > best_avg:
                best_move, best_avg = move, avg
        return best_move
