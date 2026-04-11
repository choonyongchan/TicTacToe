"""Win-probability estimation — separate feature module.

Converts the static board heuristic into an estimated win probability for
Player X using a logistic (sigmoid) function.  Terminal positions return
exact values (1.0 / 0.0 / 0.5).

This module depends only on tictactoe.core and tictactoe.evaluation.
"""
from __future__ import annotations

import math

from tictactoe.core.state import GameState
from tictactoe.core.types import Player, Result, WIN_SCORE
from tictactoe.evaluation.heuristics import evaluate_position

# Sigmoid scale factor.  WIN_SCORE / 8 places a "one near-winning threat"
# position at ~73 % and a clearly decisive position at >95 %.
_SCALE: float = WIN_SCORE / 8.0


def compute_win_probability(state: GameState) -> float:
    """Return estimated win probability for Player X in [0.0, 1.0].

    Returns:
        1.0 for X_WINS, 0.0 for O_WINS, 0.5 for DRAW or an empty board,
        sigmoid(score / _SCALE) otherwise.
    """
    result = state.result
    if result is Result.X_WINS:
        return 1.0
    if result is Result.O_WINS:
        return 0.0
    if result is Result.DRAW:
        return 0.5

    score = evaluate_position(state, Player.X)
    return 1.0 / (1.0 + math.exp(-score / _SCALE))
