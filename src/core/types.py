from __future__ import annotations
from enum import Enum, IntEnum

import numpy as np


class Player(IntEnum):
    """Game piece identifier, also used as the empty-cell sentinel.

    Attributes:
        _: Empty cell placeholder; not a valid player.
        X: First player.
        O: Second player.
    """

    _ = 0  # Placeholder for empty cell, not a valid player.
    X = 1
    O = 2  # noqa: E741

    def opponent(self) -> Player:
        """Return the other player."""
        return Player.O if self is Player.X else Player.X


class Direction(Enum):
    """Board scan direction expressed as a (row-delta, col-delta) pair.

    Attributes:
        HORIZONTAL: Left-to-right along a row.
        VERTICAL: Top-to-bottom along a column.
        DIAGONAL: Top-left to bottom-right.
        ANTI_DIAGONAL: Top-right to bottom-left.
    """

    HORIZONTAL = (0, 1)
    VERTICAL = (1, 0)
    DIAGONAL = (1, 1)
    ANTI_DIAGONAL = (1, -1)

    def __init__(self, dr: int, dc: int) -> None:
        self.dr = dr
        self.dc = dc

    def __iter__(self):
        yield self.dr
        yield self.dc


DIRECTIONS = tuple(Direction)

Board2D = np.ndarray  # shape (n, n), dtype uint8

INFINITY: float = float("inf")
NEGATIVE_INFINITY: float = float("-inf")
