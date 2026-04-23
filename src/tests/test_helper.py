"""Shared test helpers and puzzle fixtures for agent tests."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from src.core.state import State


def fresh_state(n: int = 3, k: int = 3) -> State:
    return State(n, k)


def state_with_moves(moves: Sequence[tuple[int, int]], n: int = 3, k: int = 3) -> State:
    s = State(n, k)
    for row, col in moves:
        s.apply(row, col)
    return s


@dataclass(frozen=True)
class Puzzle:
    n: int
    k: int
    moves: tuple[tuple[int, int], ...]
    best_move: tuple[int, int]
    description: str


# ---------------------------------------------------------------------------
# 3×3, k=3 — the classic dummy search tree
# ---------------------------------------------------------------------------
#
# Board after: X(0,0), O(0,1), X(1,1), O(2,2), X(1,0), O(2,0)
#   X | O | .
#   X | X | .
#   O | . | O
# X to move. Empty: (0,2), (1,2), (2,1).
#
# Hand-traced minimax tree (maximizer=X):
#   X(1,2) → row 1 = X,X,X → X WINS                          score =  1
#   X(0,2) → O to move:
#               O(2,1) → row 2 = O,O,O → O WINS              score = -1
#               O(1,2) → X(2,1) → full board, draw           score =  0
#             minimize(-1, 0) = -1
#   X(2,1) → O to move:
#               O(0,2) → X(1,2) → X WINS                     score =  1
#               O(1,2) → X(0,2) → full board, draw           score =  0
#             minimize(1, 0) = 0
# maximize(1, -1, 0) = 1  →  best_move = (1, 2)

PUZZLE_3X3 = Puzzle(
    n=3,
    k=3,
    moves=(
        (0, 0),
        (0, 1),
        (1, 1),
        (2, 2),
        (1, 0),
        (2, 0),
    ),
    best_move=(1, 2),
    description="3×3 k=3 dummy tree: X to move, best move (1,2) wins row 1",
)


# ---------------------------------------------------------------------------
# 4×4, k=4 — near-terminal (2 empty cells), 2-ply tree
# ---------------------------------------------------------------------------
#
# Board after 14 moves (X to move, empty: (0,3) and (1,3)):
#   X | X | X | .
#   O | O | O | .
#   X | O | X | O
#   O | X | O | X
#
# Hand-traced minimax tree (maximizer=X, k=4):
#   X(0,3) → row 0 = X,X,X,X → X WINS                       score =  1
#   X(1,3) → O plays (0,3) → row 1 = O,O,O,O → O WINS       score = -1
# maximize(1, -1) = 1  →  best_move = (0, 3)

PUZZLE_4X4 = Puzzle(
    n=4,
    k=4,
    moves=(
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 0),
        (3, 3),
        (3, 2),
    ),
    best_move=(0, 3),
    description="4×4 k=4 near-terminal: X wins row 0 with (0,3)",
)


# ---------------------------------------------------------------------------
# 5×5, k=3 — near-terminal (3 empty cells), 3-ply tree
# ---------------------------------------------------------------------------
#
# Board after 22 moves (X to move, empty: (4,2), (4,3), (4,4)):
#   X | O | X | O | X
#   X | O | X | O | X
#   O | X | O | X | O
#   O | X | O | X | O
#   X | O | . | . | .
#
# Hand-traced minimax tree (X to move, k=3):
#   X(4,3) → col 3: (2,3)=X,(3,3)=X,(4,3)=X → X WINS        score =  1
#   X(4,4) → O plays (4,3) → draw                            score =  0
#   X(4,2) → O plays (4,4) → col 4: O,O,O → O WINS          score = -1
# maximize(1, 0, -1) = 1  →  best_move = (4, 3)

PUZZLE_5X5 = Puzzle(
    n=5,
    k=3,
    moves=(
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 1),
        (1, 0),
        (1, 3),
        (1, 2),
        (2, 0),
        (1, 4),
        (2, 2),
        (2, 1),
        (2, 4),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 0),
        (4, 1),
    ),
    best_move=(4, 3),
    description="5×5 k=3 near-terminal: X wins col 3 with (4,3)",
)
