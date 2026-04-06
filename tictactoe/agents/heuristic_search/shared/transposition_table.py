"""Transposition table with two-tier replacement and Zobrist hashing.

Provides a fixed-size hash map keyed by Zobrist board hashes, used to cache
previously-evaluated positions and avoid redundant subtree searches.

Key design decisions:
- Two-tier replacement: Tier A is depth-preferred (deeper entries survive
  eviction); Tier B is always-replace (most recent entry wins). Together
  they improve both quality and recency of cached data.
- Incremental Zobrist hashing: O(1) hash updates via XOR after each move,
  avoiding O(n²) full recomputation.
- Optional symmetry reduction: canonical hash is the minimum across all 8
  dihedral symmetries of the board, halving effective table usage on
  symmetric positions.

Used by all Tier 2 enhanced search agents.

Dependency chain position: types → state → board → game → agents → benchmark.
"""
from __future__ import annotations

import random
from enum import Enum

from tictactoe.core.types import Cell, Move, Player, Score


class TTFlag(Enum):
    """Classification of a transposition table score relative to the search window.

    Attributes:
        EXACT: The stored score is the true minimax value for this position.
        LOWER_BOUND: The true value is >= stored score (beta cut-off occurred;
            the value may be higher but the window was exceeded).
        UPPER_BOUND: The true value is <= stored score (alpha was never raised;
            all moves failed low).
    """

    EXACT = 0
    LOWER_BOUND = 1
    UPPER_BOUND = 2


class TranspositionTable:
    """Fixed-size hash map with two-tier replacement strategy.

    Stores previously-evaluated board positions so that the search can avoid
    re-expanding transpositions — positions reachable by multiple move
    orderings that lead to the same board.

    Replacement policy:
        Tier A: depth-preferred — a new entry replaces an existing one only if
            the new entry was computed at a greater or equal remaining depth,
            preserving high-quality deep results.
        Tier B: always-replace — every new entry unconditionally overwrites the
            slot, keeping recently-seen positions available.

    On lookup, both tiers are checked. The first entry whose Zobrist key
    matches and whose stored depth is sufficient is returned.

    Each entry stores: (zobrist_key, depth, score, flag, best_move).

    Attributes:
        _size: Total number of slots in each tier (power of 2).
        _mask: Bitmask used to map a hash key to a slot index (_size - 1).
        _tier_a: Depth-preferred tier.
        _tier_b: Always-replace tier.
        _hits: Number of lookups that returned a usable score.
        _lookups: Total number of lookup calls.
        _rng: Seeded RNG used to generate the Zobrist random numbers.
        _use_symmetry: Whether to canonicalise positions via board symmetry.
        _zobrist_cache: Lazily-computed Zobrist tables, keyed by board size n.
    """

    def __init__(self, table_size: int = 2**20, use_symmetry: bool = False,
                 seed: int = 42) -> None:
        """Initialise the transposition table.

        Args:
            table_size: Number of slots per tier. Must be a power of 2 so
                that bitwise masking can replace the modulo operation.
            use_symmetry: If True, positions are hashed by their canonical
                (minimum-hash) representative across all 8 board symmetries.
                This increases cache utilisation but costs O(n²) per lookup.
            seed: Seed for the Zobrist random number generator. Fixing the
                seed ensures reproducible hashes across runs.
        """
        # table_size must be power of 2
        self._size = table_size
        self._mask = table_size - 1
        # Two tiers: depth-preferred and always-replace
        # Each entry: (full_key, depth, score, flag, best_move) or None
        self._tier_a: list[tuple | None] = [None] * table_size
        self._tier_b: list[tuple | None] = [None] * table_size
        self._hits = 0
        self._lookups = 0
        self._rng = random.Random(seed)
        self._use_symmetry = use_symmetry
        # Zobrist tables cached by board size n
        self._zobrist_cache: dict[int, list[list[list[int]]]] = {}

    def _get_zobrist(self, n: int) -> list[list[list[int]]]:
        """Generate or retrieve the cached Zobrist table for board size n.

        The table is a 3-D array of shape [n][n][2] where index 0 is Player.X
        and index 1 is Player.O. Each entry is a uniformly-random 64-bit integer
        used to XOR-hash the corresponding (row, col, player) triple.

        Args:
            n: Board dimension (number of rows and columns).

        Returns:
            The Zobrist table for the given board size.
        """
        if n not in self._zobrist_cache:
            table = [
                [[self._rng.getrandbits(64) for _ in range(2)]
                 for _ in range(n)]
                for _ in range(n)
            ]
            self._zobrist_cache[n] = table
        return self._zobrist_cache[n]

    def hash_board(self, board: list, n: int) -> int:
        """Compute the full Zobrist hash of an arbitrary board position.

        Iterates every cell and XORs in the corresponding random value for
        non-empty cells. O(n²) — use update_hash for incremental updates.

        Args:
            board: 2-D list of Cell values (size n×n).
            n: Board dimension.

        Returns:
            64-bit integer Zobrist hash of the position.
        """
        zobrist = self._get_zobrist(n)
        h = 0
        for r in range(n):
            for c in range(n):
                cell = board[r][c]
                if cell is Cell.X:
                    h ^= zobrist[r][c][0]
                elif cell is Cell.O:
                    h ^= zobrist[r][c][1]
        return h

    def update_hash(self, current_hash: int, row: int, col: int,
                    player: Player, n: int) -> int:
        """Incremental O(1) hash update via XOR.

        Args:
            current_hash: The current Zobrist hash before the move.
            row: Row of the cell being played.
            col: Column of the cell being played.
            player: The player making the move (current_player BEFORE apply_move).
            n: Board dimension.

        Returns:
            Updated Zobrist hash after the move.
        """
        zobrist = self._get_zobrist(n)
        player_idx = 0 if player is Player.X else 1
        return current_hash ^ zobrist[row][col][player_idx]

    def _symmetry_transforms(self, board: list, n: int) -> list[int]:
        """Return the Zobrist hashes of all 8 dihedral symmetry transformations.

        The dihedral group D4 contains 8 transformations of a square: identity,
        three rotations (90, 180, 270 degrees CW), and four reflections
        (horizontal, vertical, and the two diagonals). Two positions that are
        related by any of these symmetries are strategically equivalent.

        Args:
            board: Current board state.
            n: Board dimension.

        Returns:
            List of 8 Zobrist hashes, one per transformation.
        """
        def transform(r: int, c: int, t: int) -> tuple[int, int]:
            if t == 0:
                return r, c
            if t == 1:
                return c, n - 1 - r        # 90 deg CW
            if t == 2:
                return n - 1 - r, n - 1 - c  # 180 deg
            if t == 3:
                return n - 1 - c, r         # 270 deg CW
            if t == 4:
                return r, n - 1 - c         # flip H
            if t == 5:
                return c, r                 # flip diagonal
            if t == 6:
                return n - 1 - r, c         # flip V
            # t == 7
            return n - 1 - c, n - 1 - r    # flip anti-diagonal

        hashes = []
        zobrist = self._get_zobrist(n)
        for t in range(8):
            h = 0
            for r in range(n):
                for c in range(n):
                    nr, nc = transform(r, c, t)
                    cell = board[nr][nc]
                    if cell is Cell.X:
                        h ^= zobrist[r][c][0]
                    elif cell is Cell.O:
                        h ^= zobrist[r][c][1]
            hashes.append(h)
        return hashes

    def symmetry_canonical(self, hash_key: int, board: list, n: int) -> int:
        """Return the canonical (minimum) hash across all 8 symmetry transformations.

        When use_symmetry is True, all 8 symmetric variants of a position map
        to the same canonical hash, effectively multiplying table capacity by
        up to 8 on symmetric boards.

        Args:
            hash_key: The standard Zobrist hash of the position.
            board: Current board state (needed to compute the 8 variants).
            n: Board dimension.

        Returns:
            hash_key unchanged if use_symmetry is False; otherwise the minimum
            hash over all 8 dihedral transforms.
        """
        if not self._use_symmetry:
            return hash_key
        return min(self._symmetry_transforms(board, n))

    def lookup(self, hash_key: int, depth: int, alpha: float,
               beta: float) -> float | None:
        """Look up stored entry. Returns usable score or None.

        Args:
            hash_key: Zobrist hash of the current position.
            depth: Remaining search depth requested.
            alpha: Current alpha bound.
            beta: Current beta bound.

        Returns:
            A score that can be used directly, or None if no usable entry.
        """
        self._lookups += 1
        idx = hash_key & self._mask

        for tier in (self._tier_a[idx], self._tier_b[idx]):
            if tier is None:
                continue
            stored_key, stored_depth, stored_score, stored_flag, _ = tier
            if stored_key != hash_key:
                continue
            if stored_depth < depth:
                continue
            # Flag-compatible check
            if stored_flag == TTFlag.EXACT:
                self._hits += 1
                return stored_score
            elif stored_flag == TTFlag.LOWER_BOUND:
                if stored_score >= beta:
                    self._hits += 1
                    return stored_score
                alpha = max(alpha, stored_score)
            elif stored_flag == TTFlag.UPPER_BOUND:
                if stored_score <= alpha:
                    self._hits += 1
                    return stored_score
                beta = min(beta, stored_score)
            if alpha >= beta:
                self._hits += 1
                return stored_score
        return None

    def store(self, hash_key: int, depth: int, score: float, flag: TTFlag,
              best_move: Move | None) -> None:
        """Store entry using two-tier replacement.

        Tier A keeps the entry with the greatest depth (depth-preferred).
        Tier B always replaces (always-replace).

        Args:
            hash_key: Zobrist hash of the position.
            depth: Remaining search depth when this entry was computed.
            score: The evaluation score for this position.
            flag: EXACT, LOWER_BOUND, or UPPER_BOUND.
            best_move: The best move found at this node, or None.
        """
        idx = hash_key & self._mask
        entry = (hash_key, depth, score, flag, best_move)

        # Tier A: depth-preferred
        if self._tier_a[idx] is None:
            self._tier_a[idx] = entry
        else:
            _, existing_depth, _, _, _ = self._tier_a[idx]
            if depth >= existing_depth:
                self._tier_a[idx] = entry

        # Tier B: always-replace
        self._tier_b[idx] = entry

    def get_best_move(self, hash_key: int) -> Move | None:
        """Return stored best move regardless of depth/flag.

        Args:
            hash_key: Zobrist hash of the position.

        Returns:
            The stored best move, or None if no matching entry exists.
        """
        idx = hash_key & self._mask
        for tier in (self._tier_a[idx], self._tier_b[idx]):
            if tier is not None:
                stored_key, _, _, _, best_move = tier
                if stored_key == hash_key:
                    return best_move
        return None

    def clear(self) -> None:
        """Reset all entries and statistics.

        Clears both tiers and resets the hit/lookup counters to zero.
        Called at the start of each move to avoid stale information from
        prior positions (except in MTD(f), which intentionally keeps the TT
        between iterations).
        """
        self._tier_a = [None] * self._size
        self._tier_b = [None] * self._size
        self._hits = 0
        self._lookups = 0

    def hit_rate(self) -> float:
        """Return the fraction of lookups that yielded a usable result.

        Returns:
            hits / lookups, or 0.0 if no lookups have been made.
        """
        if self._lookups == 0:
            return 0.0
        return self._hits / self._lookups

    def size(self) -> int:
        """Return the number of occupied Tier A slots.

        Returns:
            Count of non-None entries in Tier A.
        """
        count = 0
        for i in range(self._size):
            if self._tier_a[i] is not None:
                count += 1
        return count
