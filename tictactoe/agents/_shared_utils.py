"""Shared utilities used across multiple agent implementations.

Provides move-selection helpers that avoid code duplication in search agents.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

from tictactoe.core.board import Board
from tictactoe.core.state import GameState
from tictactoe.core.types import Move


def check_forced_move(state: GameState) -> Move | None:
    """Return a forced move if one exists, otherwise None.

    Checks in priority order:
    1. Immediate winning move for the current player.
    2. Immediate blocking move against the opponent's win.

    This is called at the top of choose_move to short-circuit the full
    search tree when the answer is obvious.

    Args:
        state: The current game state.

    Returns:
        A winning or blocking (row, col) move, or None if no forced move
        exists.
    """
    player = state.current_player
    board = state.board
    n = state.n
    k = state.k

    # Check for an immediate win.
    win_move = Board.get_winning_move(board, n, k, player)
    if win_move is not None:
        return win_move

    # Check for an immediate block.
    block_move = Board.get_blocking_move(board, n, k, player)
    if block_move is not None:
        return block_move

    return None
