"""Move candidate generation and ordering utilities.

Good move ordering dramatically reduces the search tree by steering
alpha-beta pruning towards the best moves first. The ordering pipeline
combines static heuristics, killer moves, and history heuristics.

All functions are pure or operate on explicit table objects so they can
be imported and used in isolation by any algorithm.
"""

from __future__ import annotations

from collections import defaultdict

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import INF, Cell, Move, MoveList
from tictactoe.evaluation.heuristics import (
    count_open_threats,
    precompute_positional_weights,
)


# ---------------------------------------------------------------------------
# Move scoring
# ---------------------------------------------------------------------------


def score_move_statically(state: GameState, move: Move) -> float:
    """Assign a priority score to a move without any recursive search.

    Priority tiers (highest first):
    - +INF        : Immediate win for the current player.
    - +1e6        : Blocks an immediate opponent win.
    - +1e4        : Creates an open (k-1) threat.
    - +9e3        : Blocks opponent's open (k-1) threat.
    - +1e2        : Creates an open (k-2) threat.
    - positional  : Positional weight bonus (small, acts as tie-breaker).

    Args:
        state: The current game state.
        move: The candidate move to score.

    Returns:
        A float priority score. Higher scores are preferred.
    """
    board = state.board
    n = state.n
    k = state.k
    player = state.current_player
    opponent = player.opponent()

    # Check immediate win.
    if Board._would_win(board, n, k, move[0], move[1], player):
        return INF

    # Check blocking opponent's immediate win.
    if Board._would_win(board, n, k, move[0], move[1], opponent):
        return 1e6

    score = 0.0

    # Simulate the move temporarily to evaluate threats created.
    board[move[0]][move[1]] = player.to_cell()

    creates_k_minus_1 = count_open_threats(board, n, k, player, k - 1) > 0
    if creates_k_minus_1:
        score += 1e4

    if k >= 2:
        creates_k_minus_2 = count_open_threats(board, n, k, player, k - 2) > 0
        if creates_k_minus_2:
            score += 1e2

    board[move[0]][move[1]] = Cell.EMPTY  # Undo.

    # Check blocking opponent's open (k-1) threat.
    board[move[0]][move[1]] = opponent.to_cell()
    blocks_k_minus_1 = count_open_threats(board, n, k, opponent, k - 1) > 0
    board[move[0]][move[1]] = Cell.EMPTY  # Undo.
    if blocks_k_minus_1:
        score += 9e3

    # Add positional bonus as a tie-breaker.
    weights = precompute_positional_weights(n, k)
    score += weights[move[0]][move[1]]

    return score


def order_moves(
    state: GameState,
    moves: MoveList,
    killers: KillerMoveTable | None = None,
    history: HistoryTable | None = None,
) -> MoveList:
    """Sort candidate moves from most to least promising.

    The full ordering pipeline:
    1. Static score via score_move_statically (wins, blocks, threats).
    2. Killer move bonus (moves that caused cutoffs at the same depth).
    3. History heuristic bonus (moves that historically caused cutoffs).

    Args:
        state: The current game state.
        moves: The list of candidate moves to order.
        killers: Optional killer move table for the current search.
        history: Optional history table accumulated over the search.

    Returns:
        A new list with moves sorted from best to worst.
    """
    current_depth = state.move_number  # Used as proxy for search depth.

    def priority(move: Move) -> float:
        base = score_move_statically(state, move)

        if killers is not None and move in killers.get(current_depth):
            base += 5e3

        if history is not None:
            base += history.score(move)

        return base

    return sorted(moves, key=priority, reverse=True)


# ---------------------------------------------------------------------------
# Killer move table
# ---------------------------------------------------------------------------


class KillerMoveTable:
    """Stores moves that caused beta cutoffs at each search depth.

    Killer moves are non-capturing moves that were good enough to cause
    a cutoff at a given depth. Storing them and trying them early in
    sibling nodes improves pruning efficiency.

    Attributes:
        _killers: Mapping from depth to a list of up to 2 killer moves.
    """

    def __init__(self) -> None:
        """Initialise an empty killer move table."""
        self._killers: dict[int, list[Move]] = defaultdict(list)

    def store(self, depth: int, move: Move) -> None:
        """Record a killer move at the given search depth.

        At most two killer moves are kept per depth. Duplicate moves are
        not re-added. When the limit is exceeded the oldest entry is dropped.

        Args:
            depth: The search depth at which the cutoff occurred.
            move: The move that caused the cutoff.
        """
        killers_at_depth = self._killers[depth]

        if move in killers_at_depth:
            return  # Already present.

        killers_at_depth.append(move)

        if len(killers_at_depth) > 2:
            killers_at_depth.pop(0)  # Drop oldest entry.

    def get(self, depth: int) -> list[Move]:
        """Return the stored killer moves for a depth.

        Args:
            depth: The search depth to look up.

        Returns:
            A list of up to 2 killer moves. Empty list if none recorded.
        """
        return list(self._killers[depth])

    def clear(self) -> None:
        """Remove all stored killer moves.

        Call between separate search invocations to avoid stale data.
        """
        self._killers.clear()


# ---------------------------------------------------------------------------
# History heuristic table
# ---------------------------------------------------------------------------


class HistoryTable:
    """Accumulates scores for moves that historically caused cutoffs.

    Moves that produce cutoffs at deeper plies receive exponentially higher
    scores, reflecting their greater reliability as ordering hints.

    Attributes:
        _scores: Mapping from Move to accumulated history score.
    """

    def __init__(self) -> None:
        """Initialise an empty history table."""
        self._scores: dict[Move, float] = defaultdict(float)

    def update(self, move: Move, depth: int) -> None:
        """Increment the score for a move that caused a cutoff at depth.

        The score increment is 2^depth, so deeper cutoffs are weighted more.

        Args:
            move: The move that caused the cutoff.
            depth: The search depth at which the cutoff occurred.
        """
        self._scores[move] += 2.0 ** depth

    def score(self, move: Move) -> float:
        """Return the accumulated history score for a move.

        Args:
            move: The move to look up.

        Returns:
            The total history score; 0.0 for unseen moves.
        """
        return self._scores[move]

    def clear(self) -> None:
        """Reset all accumulated history scores.

        Call between separate search invocations.
        """
        self._scores.clear()

    def get_top_n(self, n: int) -> list[Move]:
        """Return the n moves with the highest accumulated history scores.

        Args:
            n: The number of top moves to return.

        Returns:
            A list of up to n moves sorted by descending history score.
        """
        sorted_moves = sorted(
            self._scores.keys(),
            key=lambda move: self._scores[move],
            reverse=True,
        )
        return sorted_moves[:n]
