"""Core type definitions for the n×n Tic-Tac-Toe framework.

This module defines all enums, type aliases, and numeric constants used
throughout the framework. It has no dependencies on other framework modules
and should be imported by every other module that needs shared types.
"""

from __future__ import annotations

from enum import Enum, auto

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Player(Enum):
    """Represents one of the two players in the game.

    Attributes:
        X: The first player, who plays the X symbol.
        O: The second player, who plays the O symbol.
    """

    X = "X"
    O = "O"

    def opponent(self) -> Player:
        """Return the opposing player.

        Returns:
            The other Player value (X → O or O → X).
        """
        return Player.O if self is Player.X else Player.X

    def to_cell(self) -> Cell:
        """Convert this player to the corresponding Cell value.

        Returns:
            Cell.X if this player is X, Cell.O otherwise.
        """
        return Cell.X if self is Player.X else Cell.O


class Cell(Enum):
    """Represents the content of a single board cell.

    Attributes:
        EMPTY: The cell has not been claimed by either player.
        X: The cell has been claimed by Player X.
        O: The cell has been claimed by Player O.
    """

    EMPTY = 0
    X = 1
    O = 2


class Result(Enum):
    """Represents the outcome of a game or a game position.

    Attributes:
        IN_PROGRESS: The game has not yet ended.
        X_WINS: Player X has won.
        O_WINS: Player O has won.
        DRAW: The game ended with no winner.
    """

    IN_PROGRESS = auto()
    X_WINS = auto()
    O_WINS = auto()
    DRAW = auto()


class MatchMode(Enum):
    """Controls how each agent's move budget is enforced during a match.

    Attributes:
        TIME_CONTROLLED: Each agent has a wall-clock time limit per move.
        NODE_CONTROLLED: Each agent has a maximum node-visit budget per move.
        DEPTH_CONTROLLED: Each agent searches to a fixed depth per move.
    """

    TIME_CONTROLLED = auto()
    NODE_CONTROLLED = auto()
    DEPTH_CONTROLLED = auto()


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Board2D = list[list[Cell]]
"""A two-dimensional grid of Cell values representing the board state."""

Move = tuple[int, int]
"""A (row, col) pair identifying a board position."""

MoveList = list[Move]
"""An ordered sequence of Move values."""

Score = float
"""A numeric evaluation score returned by heuristic functions."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INF: float = float("inf")
"""Positive infinity, used as an initial bound in search algorithms."""

WIN_SCORE: float = 1_000_000.0
"""Score assigned to a terminal win; must dominate any heuristic value."""

DEFAULT_TIME_LIMIT_MS: float = 1_000.0
"""Default per-move wall-clock budget in milliseconds for TIME_CONTROLLED mode."""

DEFAULT_NODE_BUDGET: int = 100_000
"""Default per-move node-visit budget for NODE_CONTROLLED mode."""

RANDOM_AGENT_WIN_THRESHOLD: float = 0.85
"""Minimum win rate an agent must achieve against RandomAgent on n=3 to pass
the sanity check. An agent below this threshold is considered buggy."""
