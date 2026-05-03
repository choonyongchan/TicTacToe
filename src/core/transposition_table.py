from __future__ import annotations

from .manipulator import Manipulator


class TranspositionTable:
    """Depth-aware transposition table storing alpha-beta bounds per Zobrist key.

    Each entry holds a (lower_bound, upper_bound, best_move) tuple.
    Shallow entries are not promoted over deeper ones.
    """

    def __init__(self) -> None:
        self._table: dict[int, tuple[float, float, tuple[int, int] | None]] = {}
        self._depths: dict[int, int] = {}

    def lookup(self, key: int) -> tuple[float, float, tuple[int, int] | None] | None:
        """Return the stored entry for key, regardless of depth.

        Args:
            key: Zobrist hash of the position.

        Returns:
            (lower, upper, best_move) tuple, or None if absent.
        """
        return self._table.get(key)

    def lookup_at_depth(
        self, key: int, min_depth: int
    ) -> tuple[float, float, tuple[int, int] | None] | None:
        """Return entry only if it was stored at depth >= min_depth.

        Args:
            key: Zobrist hash of the position.
            min_depth: Minimum search depth required to trust the entry.

        Returns:
            (lower, upper, best_move) tuple, or None if absent or too shallow.
        """
        entry = self._table.get(key)
        if entry is None:
            return None
        stored_depth = self._depths.get(key, 0)
        if stored_depth < min_depth:
            # Return only the best_move hint; do not use bounds for cutoffs.
            return None
        return entry

    def store(
        self,
        key: int,
        lower: float,
        upper: float,
        best_move: tuple[int, int] | None,
        depth: int = 0,
    ) -> None:
        """Store bounds for key, overwriting only if depth >= existing depth.

        Args:
            key: Zobrist hash of the position.
            lower: Lower bound on the position value.
            upper: Upper bound on the position value.
            best_move: Best move found, or None.
            depth: Search depth at which bounds were computed.
        """
        if depth < self._depths.get(key, -1):
            return
        self._table[key] = (lower, upper, best_move)
        self._depths[key] = depth

    def best_move(self, key: int) -> tuple[int, int] | None:
        """Return the best move stored for key, or None if absent.

        Args:
            key: Zobrist hash of the position.
        """
        entry = self._table.get(key)
        return entry[2] if entry is not None else None

    def depth_of(self, key: int) -> int:
        """Return the depth at which key was last stored, or -1 if absent.

        Args:
            key: Zobrist hash of the position.
        """
        return self._depths.get(key, -1)

    def store_symmetric(
        self,
        hashes: list[int],
        lower: float,
        upper: float,
        best_move: tuple[int, int] | None,
        n: int,
        depth: int = 0,
    ) -> None:
        """Store bounds under all 8 symmetry-equivalent hashes.

        Args:
            hashes: List of TRANSFORM_COUNT Zobrist hashes for the position.
            lower: Lower bound on the position value.
            upper: Upper bound on the position value.
            best_move: Best move in the canonical orientation, or None.
            n: Board side length (needed to transform the move).
            depth: Search depth at which bounds were computed.
        """
        assert len(hashes) == Manipulator.TRANSFORM_COUNT
        transform_moves = Manipulator.all_transform_moves(best_move, n)
        assert len(transform_moves) == Manipulator.TRANSFORM_COUNT
        for key, t_move in zip(hashes, transform_moves):
            self.store(key, lower, upper, t_move, depth)

    def __len__(self) -> int:
        return len(self._table)
