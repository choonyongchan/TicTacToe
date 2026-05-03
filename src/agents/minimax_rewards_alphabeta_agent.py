from src.agents.minimax_alphabeta_agent import MinimaxAlphaBetaAgent
from src.agents.minimax_base_agent import MinimaxBaseAgent
from src.core.state import State
from src.core.types import INFINITY, NEGATIVE_INFINITY, Player


class MinimaxRewardsAlphaBetaAgent(MinimaxAlphaBetaAgent):
    """Alpha-beta minimax with depth-adjusted floating-point rewards.

    Assigns scores in (-1, 1) so that faster wins score higher and faster
    losses score lower, encouraging decisive play.

    Attributes:
        _epsilon: Per-ply score discount; set so win > 0 and loss < 0 at any depth.
    """

    def __init__(self, maximizer: Player, max_depth: int) -> None:
        MinimaxBaseAgent.__init__(self, "MinimaxRewardsAlphaBetaAgent", maximizer)
        # epsilon = 1/(max_depth+1) guarantees win>0 and loss<0 at any reachable depth
        self._epsilon: float = 1.0 / (max_depth + 1)

    def act(self, state: State) -> tuple[int, int]:
        """Return the best move via alpha-beta with depth-adjusted rewards.

        Args:
            state: Current game state.

        Returns:
            (row, col) of the best move for the maximizer.
        """
        best_score = NEGATIVE_INFINITY
        best_move: tuple[int, int] | None = None
        alpha = NEGATIVE_INFINITY
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = self._minimax(state, alpha, INFINITY)
            state.undo()
            if score > best_score:
                best_score = score
                best_move = (row, col)
            alpha = max(alpha, best_score)
        assert best_move is not None
        return best_move

    def _terminal_score(self, state: State) -> float:
        """Return a depth-adjusted score: positive for a maximizer win, negative for a loss.

        Args:
            state: Terminal game state.

        Returns:
            Score in (-1, 1]: 1 - epsilon*depth for a win, -1 + epsilon*depth for a loss.
        """
        winner = state.winner()
        depth = len(state.history)
        if winner == self._maximizer:
            return 1.0 - self._epsilon * depth
        if winner is not None:
            return -1.0 + self._epsilon * depth
        return 0.0

    def _minimax(self, state: State, alpha: float, beta: float) -> float:
        """Recursive alpha-beta minimax dispatch with float rewards.

        Args:
            state: Current game state.
            alpha: Best score the maximizer can guarantee.
            beta: Best score the minimizer can guarantee.

        Returns:
            Score from the maximizer's perspective.
        """
        if state.is_terminal():
            return self._terminal_score(state)
        if state.current_player == self._maximizer:
            return self._maximize(state, alpha, beta)
        return self._minimize(state, alpha, beta)

    def _maximize(self, state: State, alpha: float, beta: float) -> float:
        """Return the maximum score reachable with alpha-beta pruning.

        Args:
            state: Current game state (maximizer's turn).
            alpha: Current lower bound.
            beta: Current upper bound.
        """
        best = NEGATIVE_INFINITY
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = self._minimax(state, alpha, beta)
            state.undo()
            if score > best:
                best = score
            if best >= beta:
                break
            if best > alpha:
                alpha = best
        return best

    def _minimize(self, state: State, alpha: float, beta: float) -> float:
        """Return the minimum score reachable with alpha-beta pruning.

        Args:
            state: Current game state (minimizer's turn).
            alpha: Current lower bound.
            beta: Current upper bound.
        """
        best = INFINITY
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = self._minimax(state, alpha, beta)
            state.undo()
            if score < best:
                best = score
            if best <= alpha:
                break
            if best < beta:
                beta = best
        return best
