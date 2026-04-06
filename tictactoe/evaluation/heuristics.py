"""Static board evaluation helpers for n×n Tic-Tac-Toe.

All functions are pure (no side effects) and stateless. They can be
imported and used in isolation by any algorithm implementation.

Expensive index computations are cached via functools.lru_cache so they
are never recomputed for the same (n, k) pair across the lifetime of a
process.
"""

from __future__ import annotations

from functools import lru_cache

from tictactoe.core.state import GameState
from tictactoe.core.types import (
    WIN_SCORE,
    Board2D,
    Cell,
    Player,
    Result,
)


# ---------------------------------------------------------------------------
# Line index pre-computation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def precompute_line_indices(n: int, k: int) -> list[list[tuple[int, int]]]:
    """Return the coordinate lists of every possible winning line.

    A winning line is a contiguous sequence of exactly k cells that
    runs horizontally, vertically, diagonally, or anti-diagonally.

    Args:
        n: Board dimension.
        k: Winning line length.

    Returns:
        A list of lines, where each line is a list of (row, col) tuples.
    """
    lines: list[list[tuple[int, int]]] = []

    # Horizontal lines.
    for row in range(n):
        for start_col in range(n - k + 1):
            lines.append([(row, start_col + i) for i in range(k)])

    # Vertical lines.
    for col in range(n):
        for start_row in range(n - k + 1):
            lines.append([(start_row + i, col) for i in range(k)])

    # Diagonal (top-left → bottom-right) lines.
    for start_row in range(n - k + 1):
        for start_col in range(n - k + 1):
            lines.append([(start_row + i, start_col + i) for i in range(k)])

    # Anti-diagonal (top-right → bottom-left) lines.
    for start_row in range(n - k + 1):
        for start_col in range(k - 1, n):
            lines.append([(start_row + i, start_col - i) for i in range(k)])

    return lines


@lru_cache(maxsize=64)
def precompute_positional_weights(n: int, k: int) -> list[list[float]]:
    """Build an n×n matrix of normalised line-participation counts.

    Each cell's weight is the number of winning lines it participates in,
    normalised to the range [0, 1] by dividing by the maximum count.
    Centre cells score highest on any board, making this useful as a
    positional bonus in evaluation functions.

    Args:
        n: Board dimension.
        k: Winning line length.

    Returns:
        An n×n matrix of float weights in [0, 1].
    """
    counts: list[list[float]] = [[0.0] * n for _ in range(n)]

    for line in precompute_line_indices(n, k):
        for row, col in line:
            counts[row][col] += 1.0

    max_count = max(counts[row][col] for row in range(n) for col in range(n))
    if max_count == 0.0:
        return counts

    return [
        [counts[row][col] / max_count for col in range(n)] for row in range(n)
    ]


# ---------------------------------------------------------------------------
# Line scoring
# ---------------------------------------------------------------------------


def _count_player_cells(
    cells: list[Cell], player: Player
) -> tuple[int, int]:
    """Count cells belonging to player and to the opponent in a line.

    Args:
        cells: The cell contents of a single line.
        player: The player from whose perspective the line is evaluated.

    Returns:
        A (player_count, opponent_count) tuple.
    """
    player_cell = player.to_cell()
    opponent_cell = player.opponent().to_cell()
    player_count = sum(1 for c in cells if c is player_cell)
    opponent_count = sum(1 for c in cells if c is opponent_cell)
    return player_count, opponent_count


def score_line(cells: list[Cell], player: Player, k: int) -> float:
    """Score a single line from one player's perspective.

    A line is "dead" (worth 0) if it contains pieces from both players.
    Otherwise, exponential scoring is applied:
    - player pieces contribute +10^count.
    - opponent pieces contribute -10^count.

    Args:
        cells: The cell contents of a single line (length == k).
        player: The perspective player.
        k: Winning line length (used implicitly via the line length).

    Returns:
        A float score. Positive favours player; negative favours opponent.
    """
    player_count, opponent_count = _count_player_cells(cells, player)

    # Dead line: both players are present.
    if player_count > 0 and opponent_count > 0:
        return 0.0

    if player_count > 0:
        return 10.0 ** player_count
    if opponent_count > 0:
        return -(10.0 ** opponent_count)

    return 0.0


def _generate_all_lines(
    board: Board2D, n: int, k: int
) -> list[list[Cell]]:
    """Extract cell contents for every winning line on the board.

    Args:
        board: The current board state.
        n: Board dimension.
        k: Winning line length.

    Returns:
        A list of cell-content lists, one per winning line.
    """
    return [
        [board[row][col] for row, col in line]
        for line in precompute_line_indices(n, k)
    ]


def score_board(board: Board2D, n: int, k: int, player: Player) -> float:
    """Sum score_line over every winning line on the board.

    Args:
        board: The current board state.
        n: Board dimension.
        k: Winning line length.
        player: The perspective player.

    Returns:
        Total board score from player's point of view.
    """
    total = 0.0
    for line_cells in _generate_all_lines(board, n, k):
        total += score_line(line_cells, player, k)
    return total


# ---------------------------------------------------------------------------
# Threat counting
# ---------------------------------------------------------------------------


def count_open_threats(
    board: Board2D, n: int, k: int, player: Player, threat_size: int
) -> int:
    """Count lines where player has exactly threat_size pieces and no opponent pieces.

    An "open threat" is a line that can still be completed — it has the
    required number of player pieces and all remaining cells are empty.

    Args:
        board: The current board state.
        n: Board dimension.
        k: Winning line length.
        player: The player whose threats are counted.
        threat_size: Number of player pieces the line must contain.

    Returns:
        The number of open threat lines.
    """
    player_cell = player.to_cell()
    count = 0

    for line_cells in _generate_all_lines(board, n, k):
        player_pieces = sum(1 for c in line_cells if c is player_cell)
        opponent_pieces = sum(
            1 for c in line_cells if c is player.opponent().to_cell()
        )

        if player_pieces == threat_size and opponent_pieces == 0:
            count += 1

    return count


# ---------------------------------------------------------------------------
# Full static evaluation
# ---------------------------------------------------------------------------


def evaluate_position(state: GameState, player: Player) -> float:
    """Compute a complete static evaluation of a position.

    Checks terminal conditions first (these dominate). For non-terminal
    positions, combines three signals:
    - Line score (score_board).
    - Positional bonus (precomputed positional weights).
    - Open threat bonus (count_open_threats for sizes k-1 and k-2).

    Args:
        state: The current game state.
        player: The perspective player (positive score means player is winning).

    Returns:
        WIN_SCORE for a terminal win, -WIN_SCORE for a terminal loss,
        0.0 for a draw, or a heuristic float for non-terminal positions.
    """
    result = state.result

    if result is Result.X_WINS:
        if player is Player.X:
            return WIN_SCORE
        return -WIN_SCORE

    if result is Result.O_WINS:
        if player is Player.O:
            return WIN_SCORE
        return -WIN_SCORE

    if result is Result.DRAW:
        return 0.0

    board = state.board
    n = state.n
    k = state.k

    line_score = score_board(board, n, k, player)

    positional_weights = precompute_positional_weights(n, k)
    positional_bonus = _compute_positional_bonus(board, n, player, positional_weights)

    threat_bonus = _compute_threat_bonus(board, n, k, player)

    return line_score + positional_bonus + threat_bonus


def _compute_positional_bonus(
    board: Board2D,
    n: int,
    player: Player,
    weights: list[list[float]],
) -> float:
    """Sum positional weights for player's occupied cells, minus opponent's.

    Args:
        board: The current board state.
        n: Board dimension.
        player: The perspective player.
        weights: Pre-computed positional weight matrix.

    Returns:
        Net positional score for player.
    """
    player_cell = player.to_cell()
    opponent_cell = player.opponent().to_cell()
    bonus = 0.0

    for row in range(n):
        for col in range(n):
            cell = board[row][col]
            if cell is player_cell:
                bonus += weights[row][col]
            elif cell is opponent_cell:
                bonus -= weights[row][col]

    return bonus


def _compute_threat_bonus(
    board: Board2D, n: int, k: int, player: Player
) -> float:
    """Bonus based on open threats of size k-1 and k-2 for both players.

    Args:
        board: The current board state.
        n: Board dimension.
        k: Winning line length.
        player: The perspective player.

    Returns:
        Net threat bonus (player threats minus opponent threats).
    """
    opponent = player.opponent()
    bonus = 0.0

    if k >= 2:
        bonus += count_open_threats(board, n, k, player, k - 1) * 1e4
        bonus -= count_open_threats(board, n, k, opponent, k - 1) * 1e4

    if k >= 3:
        bonus += count_open_threats(board, n, k, player, k - 2) * 1e2
        bonus -= count_open_threats(board, n, k, opponent, k - 2) * 1e2

    return bonus
