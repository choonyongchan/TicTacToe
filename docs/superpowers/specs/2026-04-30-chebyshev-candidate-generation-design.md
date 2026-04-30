# Chebyshev Candidate Move Generation — Design Spec

**Date:** 2026-04-30  
**Status:** Approved

---

## Problem

`MTDfIDAgent._negamax_tt` iterates over all empty cells at every node via `state.board.get_empty_cells()`. On an n×n board with m stones played, the branching factor is n²−m regardless of where the stones are. For large boards or deep searches, most of those cells are irrelevant — far from any existing stone and incapable of contributing to a win.

---

## Goal

Limit the candidate move set to empty cells within Chebyshev distance `d = max(1, k−2)` of any played stone. This reduces the effective branching factor while preserving search correctness for all windows with ≥2 pieces (the only windows that score meaningfully).

---

## Analytical Foundation

### Chebyshev distance

`chebyshev((r1,c1), (r2,c2)) = max(|r1−r2|, |c1−c2|)`

A cell is a candidate if its Chebyshev distance to the nearest played stone is ≤ d.

### Completeness bound

A move M is strategically relevant only if it belongs to a k-window with ≥1 existing stone. The farthest M can be from the nearest stone in the window is k−1 steps (pieces at one end, M at the other). So d = k−1 is theoretically complete.

### Tighter bound: d = k−2

Windows with m=1 piece score `4^0 = 1` by the window scorer formula — negligible. For m ≥ 2, the farthest empty cell from the nearest piece is k−2 steps. So **d = k−2 captures all windows that influence the search**.

`ForcedMove.detect()` already handles the one exception: a winning/blocking cell that is the sole empty cell in a full k-window (m = k−1). That cell is always adjacent (d=1) to an existing piece, so it's covered.

### Forking moves

A fork at F creates ≥2 simultaneous threats. Each threat requires ≥k−2 pieces within k−1 steps of F in that direction. Therefore F is always within Chebyshev distance ≤ k−2 of existing pieces in each threat direction. Fork cells are never isolated — they're geometrically anchored to nearby stones.

For k=4: d = k−2 = 2. For k=5: d = k−2 = 3. For k=3: d = max(1, 1) = 1.

### Why d=1 is insufficient

Bridging move example: stones at (0,0) and (0,2). The bridge cell (0,1) is d=1 — captured. But the extension cell (0,4) is d=2 from (0,2) — missed by d=1 whenever there's no stone at (0,3).

---

## Design

### `Board.get_candidate_cells(history, d)`

New method on `Board`:

```python
def get_candidate_cells(self, history: list[tuple[int, int]], d: int) -> list[tuple[int, int]]:
    if not history:
        return [(self.n // 2, self.n // 2)]
    candidates: set[tuple[int, int]] = set()
    for pr, pc in history:
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
                r, c = pr + dr, pc + dc
                if self.is_in_bounds(r, c) and self.is_empty(r, c):
                    candidates.add((r, c))
    return list(candidates)
```

**Opening rule:** when `history` is empty, return `[(n//2, n//2)]` — the center cell. Consistent with Gomoku convention; avoids the "no stones → no candidates" degeneracy.

**Complexity:** O(|history| × (2d+1)²). For a 15×15 board mid-game (~50 stones, d=2): 50 × 25 = 1250 iterations. The set deduplicates overlapping neighborhoods automatically.

### `State.__init__` addition

```python
self.candidate_d: int = max(1, self.board.k - 2)
```

One line. No other `State` methods change.

### Call site change

`MTDfIDAgent._negamax_tt`, one line:

```python
# before
for row, col in state.board.get_empty_cells():

# after
for row, col in state.board.get_candidate_cells(state.history, state.candidate_d):
```

No other files change. Heuristics scan k-windows (not empty cells) and are unaffected.

---

## Files Changed

| Action | Path | Change |
|--------|------|--------|
| Modify | `src/core/board.py` | Add `get_candidate_cells(history, d)` |
| Modify | `src/core/state.py` | Add `candidate_d` field in `__init__` |
| Modify | `src/agents/mtdf_id_agent.py` | Replace `get_empty_cells()` with `get_candidate_cells(...)` in `_negamax_tt` |
| Modify | `src/tests/test_core.py` | Add 4 tests for `get_candidate_cells` |

---

## Tests (`test_core.py`)

| Test | What it checks |
|------|----------------|
| Empty board → `[(n//2, n//2)]` | Opening rule |
| After 1 stone at (r,c), candidate set = empty cells in Chebyshev-d ring around (r,c) | Basic correctness |
| No cell in candidate set is occupied | Safety invariant |
| `set(get_candidate_cells(...)) ⊆ set(get_empty_cells())` | Subset invariant (no phantom cells) |

---

## Correctness Invariant

The candidate set is always a subset of `get_empty_cells()`. The search remains correct because:

1. `ForcedMove.detect()` handles immediate win/block before the search loop runs.
2. All windows with m ≥ 2 pieces have their empty cells within d = k−2 of some stone.
3. Windows with m = 1 score negligibly and do not affect the search outcome.

---

## Complexity Impact

| Board | k | d | Empty cells (mid-game) | Candidates (approx) |
|-------|---|---|------------------------|---------------------|
| 4×4   | 4 | 2 | ~8                     | ~6 (small board, near-full overlap) |
| 10×10 | 5 | 3 | ~75                    | ~30−40 |
| 15×15 | 5 | 3 | ~175                   | ~50−70 |

Branching factor reduction is most significant on larger boards in the early/mid game.
