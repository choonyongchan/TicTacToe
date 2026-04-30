from __future__ import annotations

from typing import Callable

import numpy as np

from .types import Board2D


class Manipulator:
    # 8 coordinate transforms for the dihedral group of a square.
    # Each maps (r, c, n) -> (r', c') where n is board size.
    # ALL methods iterate this tuple so ordering is always consistent.
    COORD_TRANSFORMS: tuple[Callable[[int, int, int], tuple[int, int]], ...] = (
        lambda r, c, n: (r, c),           # 0: identity
        lambda r, c, n: (c, n - 1 - r),   # 1: CW 90°
        lambda r, c, n: (n - 1 - r, n - 1 - c),  # 2: 180°
        lambda r, c, n: (n - 1 - c, r),   # 3: CW 270°
        lambda r, c, n: (r, n - 1 - c),   # 4: reflect (flip LR)
        lambda r, c, n: (n - 1 - c, n - 1 - r),  # 5: reflect + CW 90°
        lambda r, c, n: (n - 1 - r, c),   # 6: reflect + 180°
        lambda r, c, n: (c, r),            # 7: reflect + CW 270°
    )
    TRANSFORM_COUNT: int = len(COORD_TRANSFORMS)  # 8

    @staticmethod
    def all_transforms(board: Board2D) -> list[Board2D]:
        """Return 8 transformed copies of board, ordered by COORD_TRANSFORMS."""
        n = board.shape[0]
        result = []
        for fn in Manipulator.COORD_TRANSFORMS:
            t = np.empty_like(board)
            for r in range(n):
                for c in range(n):
                    tr, tc = fn(r, c, n)
                    t[tr, tc] = board[r, c]
            result.append(t)
        assert len(result) == Manipulator.TRANSFORM_COUNT
        return result

    @staticmethod
    def all_transform_moves(
        move: tuple[int, int] | None, n: int
    ) -> list[tuple[int, int] | None]:
        """Return 8 transformed coordinates, ordered by COORD_TRANSFORMS."""
        if move is None:
            return [None] * Manipulator.TRANSFORM_COUNT
        r, c = move
        result = [fn(r, c, n) for fn in Manipulator.COORD_TRANSFORMS]
        assert len(result) == Manipulator.TRANSFORM_COUNT
        return result
