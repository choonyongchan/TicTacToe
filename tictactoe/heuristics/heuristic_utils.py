from __future__ import annotations

import math

DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))


def tanh_normalize(score_me: float, score_opp: float, k: int) -> float:
    """Normalise a pair of raw scores into [-1, 1] using tanh.

    Returns 0.0 when k < 2 to avoid division by zero.

    Args:
        score_me: Raw score for the current player.
        score_opp: Raw score for the opponent.
        k: Win-condition run length.

    Returns:
        Value in [-1, 1] representing relative advantage.
    """
    if k < 2:
        return 0.0
    return math.tanh((score_me - score_opp) / (2.0 * float(4 ** (k - 2))))
