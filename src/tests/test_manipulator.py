from __future__ import annotations

import numpy as np

from src.core.manipulator import Manipulator
from src.core.state import State
from src.core.transposition_table import TranspositionTable


# ---------------------------------------------------------------------------
# A. Class attributes
# ---------------------------------------------------------------------------


def test_transform_count():
    assert Manipulator.TRANSFORM_COUNT == 8
    assert len(Manipulator.COORD_TRANSFORMS) == 8


# ---------------------------------------------------------------------------
# B. Board transforms — shape and content
# ---------------------------------------------------------------------------


def test_all_transforms_shape():
    board = np.arange(9, dtype=np.uint8).reshape(3, 3)
    transforms = Manipulator.all_transforms(board)
    assert len(transforms) == 8
    for t in transforms:
        assert t.shape == (3, 3)
        assert t.dtype == np.uint8


def test_all_transforms_identity():
    board = np.arange(9, dtype=np.uint8).reshape(3, 3)
    assert np.array_equal(Manipulator.all_transforms(board)[0], board)


def test_all_transforms_cw90():
    # np.rot90(k=-1) on [[1,2],[3,4]] → [[3,1],[4,2]]
    board = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = np.array([[3, 1], [4, 2]], dtype=np.uint8)
    assert np.array_equal(Manipulator.all_transforms(board)[1], expected)


def test_all_transforms_180():
    board = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = np.array([[4, 3], [2, 1]], dtype=np.uint8)
    assert np.array_equal(Manipulator.all_transforms(board)[2], expected)


def test_all_transforms_cw270():
    # np.rot90(k=1) on [[1,2],[3,4]] → [[2,4],[1,3]]
    board = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = np.array([[2, 4], [1, 3]], dtype=np.uint8)
    assert np.array_equal(Manipulator.all_transforms(board)[3], expected)


def test_all_transforms_reflect():
    # fliplr on [[1,2],[3,4]] → [[2,1],[4,3]]
    board = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = np.array([[2, 1], [4, 3]], dtype=np.uint8)
    assert np.array_equal(Manipulator.all_transforms(board)[4], expected)


def test_all_transforms_8_distinct_for_asymmetric_board():
    board = np.arange(1, 10, dtype=np.uint8).reshape(3, 3)
    transforms = Manipulator.all_transforms(board)
    assert len({t.tobytes() for t in transforms}) == 8


# ---------------------------------------------------------------------------
# C. Move coordinate transforms
# ---------------------------------------------------------------------------


def test_all_transform_moves_none():
    assert Manipulator.all_transform_moves(None, 3) == [None] * 8


def test_all_transform_moves_identity():
    assert Manipulator.all_transform_moves((1, 2), 4)[0] == (1, 2)


def test_all_transform_moves_cw90():
    # (r, c) → (c, n-1-r); n=4, (1,2) → (2, 2)
    assert Manipulator.all_transform_moves((1, 2), 4)[1] == (2, 2)


def test_all_transform_moves_180():
    # (r, c) → (n-1-r, n-1-c); n=4, (1,2) → (2, 1)
    assert Manipulator.all_transform_moves((1, 2), 4)[2] == (2, 1)


def test_all_transform_moves_cw270():
    # (r, c) → (n-1-c, r); n=4, (1,2) → (1, 1)
    assert Manipulator.all_transform_moves((1, 2), 4)[3] == (1, 1)


def test_all_transform_moves_reflect():
    # (r, c) → (r, n-1-c); n=4, (1,2) → (1, 1)
    assert Manipulator.all_transform_moves((1, 2), 4)[4] == (1, 1)


def test_all_transform_moves_corner_maps_to_corner():
    n = 3
    corners = {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}
    for t_move in Manipulator.all_transform_moves((0, 0), n):
        assert t_move in corners


def test_all_transform_moves_index0_is_identity():
    move = (1, 2)
    n = 4
    assert Manipulator.all_transform_moves(move, n)[0] == move


# ---------------------------------------------------------------------------
# D. End-to-end alignment: incremental _hashes vs. from-scratch hashes
# ---------------------------------------------------------------------------


def _compute_hash_from_board(zobrist, board: np.ndarray) -> int:
    """Test-local helper: Zobrist hash by full board scan."""
    h = 0
    rows, cols = np.where(board != 0)
    for r, c in zip(rows.tolist(), cols.tolist()):
        h ^= int(zobrist._table[r, c, int(board[r, c])])
    return h


def test_symmetry_hashes_full_end_to_end():
    """3×3 board, 2 moves: verify all 8 incremental hashes, TT storage, and move transforms."""
    state = State(n=3, k=3)
    state.apply(0, 0)  # X at top-left
    state.apply(0, 1)  # O at top-middle (asymmetric — all 8 transforms are distinct)
    best_move = (0, 0)
    n = state.board.n

    # 1. Compute all 8 board transforms from the current board array.
    board_transforms = Manipulator.all_transforms(state.board._grid)
    assert len(board_transforms) == Manipulator.TRANSFORM_COUNT

    # 2. Compute expected hashes from scratch for each transform.
    expected_hashes = [
        _compute_hash_from_board(state._zobrist, t) for t in board_transforms
    ]

    # 3. All 8 incremental hashes must match from-scratch hashes.
    assert len(state._hashes) == Manipulator.TRANSFORM_COUNT
    for i, (incremental, expected) in enumerate(zip(state._hashes, expected_hashes)):
        assert incremental == expected, (
            f"_hashes[{i}] mismatch: incremental={incremental}, "
            f"from-scratch={expected}"
        )

    # 4. Store in TT; verify all 8 entries are present with correct bounds.
    tt = TranspositionTable()
    tt.store_symmetric(state._hashes, -0.5, 0.5, best_move, n, depth=2)
    assert len(tt) == Manipulator.TRANSFORM_COUNT

    for h in state._hashes:
        entry = tt.lookup(h)
        assert entry is not None, f"No TT entry for hash {h}"
        lb, ub, _ = entry
        assert lb == -0.5 and ub == 0.5

    # 5. Each TT entry must store the correctly transformed move.
    expected_moves = Manipulator.all_transform_moves(best_move, n)
    for h, exp_move in zip(state._hashes, expected_moves):
        assert tt.best_move(h) == exp_move, (
            f"TT best_move for hash {h}: got {tt.best_move(h)}, expected {exp_move}"
        )

    # 6. Undo must restore _hashes correctly.
    state.undo()
    board_after_undo = Manipulator.all_transforms(state.board._grid)
    expected_after_undo = [
        _compute_hash_from_board(state._zobrist, t) for t in board_after_undo
    ]
    for i, (incremental, expected) in enumerate(zip(state._hashes, expected_after_undo)):
        assert incremental == expected, (
            f"After undo, _hashes[{i}] mismatch: {incremental} vs {expected}"
        )
