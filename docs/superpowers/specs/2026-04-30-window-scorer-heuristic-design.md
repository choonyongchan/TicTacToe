# Window Scorer Heuristic — Design Spec

**Date:** 2026-04-30  
**Status:** Approved

---

## Problem

MTDfIDAgent is significantly slower than plain MTDfAgent because it runs a 4-heuristic ensemble (`DistanceHeuristic + TaxonomyHeuristic + ForkHeuristic + ThreatHeuristic`) at every depth-0 leaf node. `ForkHeuristic` dominates at O(n³) — it probes every empty cell in every direction. The ensemble's cost compounds across iterative deepening depths and all board sizes.

The heuristic serves leaf evaluation, which guides move ordering through iterative deepening. Strong move ordering → more alpha-beta cutoffs → faster search overall.

---

## Goal

Replace the 4-heuristic ensemble with a single, cheap `WindowScorerHeuristic` that:
- Runs in O(n² · k) for any board size n and win-length k
- Captures the full threat gradient (not just immediate threats)
- Correctly handles gap threats like `X _ X` (connecting two segments)
- Scores positions with the classic open/half-open/blocked weighting

---

## Design

### Core Algorithm

Scan all k-windows in 4 directions `(0,1), (1,0), (1,1), (1,-1)`.

For each window starting at `(r, c)` in direction `(dr, dc)`:

1. **Classify the window**: count player pieces (`m`), empty cells inside (`e`), and check for any opponent piece (blocked).
2. **Skip** if blocked or `m = 0`.
3. **One-move win check**: if `m == k-1` and `e == 1`, score = `WIN_THREAT_SCORE = 4^(k-1)`. This handles both end-empty (`X X _`) and gap threats (`X _ X`). **Stop here — skip openness check.**
4. **Check openness**: examine the cell just before the window start and just after the window end.
   - `left_open = True` if that cell is in-bounds and empty
   - `right_open = True` if that cell is in-bounds and empty
   - `open_ends = left_open + right_open`
5. **Skip** if `open_ends == 0` (dead run — can never reach k).
6. **Score**: `open_ends * base^(m - 1)` where `base = 4`.

Sum scores for `player` and `opponent` separately.

### Normalization

```
raw = score_me - score_opp
scale = 2 * 4^(k - 2)   # one open run of k-1 ≈ ±0.5
return tanh(raw / scale)
```

Compresses output to `[-1.0, 1.0]` with saturation for decisive positions.

### Why `base = 4`

Each additional piece in a window quadruples its score. This matches the branching factor intuition: a run of length m covers 4× as many "paths to victory" as a run of m-1. The ratio between open k-1 and open k-2 is 4:1, and between open and half-open of the same length is 2:1 — consistent with classical Gomoku pattern tables.

### Gap Threats (One-Move Wins via Internal Gaps)

A window like `[X, _, X]` with k=3 has m=2 player pieces and 1 empty cell *inside* the window. Placing in the gap wins immediately — this is a definite win for the player to move.

The openness formula fails here: if both outside cells are blocked (edge or opponent), `open_ends = 0` and the formula yields score = 0, incorrectly dismissing a winning position.

**Special case:** When a window has exactly `m = k-1` player pieces and exactly 1 empty cell inside (regardless of position — end or gap), score it as `WIN_THREAT_SCORE = 4^(k-1)`. This is higher than any non-winning window score and independent of openness.

This check takes priority over the openness formula and applies to all patterns:
- `[X, X, _]` with k=3: standard end-empty threat — scored as WIN_THREAT_SCORE
- `[X, _, X]` with k=3: gap threat — also scored as WIN_THREAT_SCORE
- `[X, X, X, X, _]` with k=5: end-empty four — WIN_THREAT_SCORE
- `[X, X, _, X, X]` with k=5: split four (double gap counts as 2 empty → not k-1, handled by formula)

TaxonomyHeuristic would miss `[X, _, X]` since it walks contiguous runs and stops at the gap.

---

## Pattern Examples (k=5)

`WIN_THREAT_SCORE = 4^(k-1) = 4^4 = 256` for k=5.

| Window content | m | empty inside | open_ends | Score |
|---|---|---|---|---|
| `X X X X _` | 4 | 1 | any | **256** (WIN_THREAT_SCORE) |
| `X _ X X X` | 4 | 1 | any | **256** (WIN_THREAT_SCORE, gap threat) |
| `_ X X X X _` | 4 | 0 | 2 | 2 × 4³ = 128 (open, no gap) |
| `_ X X X X` (edge) | 4 | 0 | 1 | 1 × 4³ = 64 (half-open) |
| `_ X X X _` | 3 | 0 | 2 | 2 × 4² = 32 |
| `_ X X _` | 2 | 0 | 2 | 2 × 4¹ = 8 |
| `_ X _` | 1 | 0 | 2 | 2 × 4⁰ = 2 |

WIN_THREAT_SCORE > any non-winning window score (256 > 128), ensuring one-move wins are always ranked highest. The ratios for non-winning windows match the intent of the classic table (open four >> half-open four >> open three).

---

## Implementation

### File to create

`src/heuristics/window_scorer_heuristic.py`

```
WindowScorerHeuristic(BaseHeuristic)
  - _score_windows(grid, n, k, player_val, opp_val) -> float
  - evaluate(state) -> float
```

### File to modify

`src/heuristics/heuristic.py` — replace the 4-component ensemble with a single `WindowScorerHeuristic` instance.

### Files to delete (after verifying no other imports)

- `src/heuristics/distance_heuristic.py`
- `src/heuristics/taxonomy_heuristic.py`
- `src/heuristics/fork_heuristic.py`
- `src/heuristics/threat_heuristic.py`

Or leave them in place and simply stop using them in the ensemble (safer first step).

---

## Complexity

| | Before | After |
|---|---|---|
| ForkHeuristic | O(n³) | — |
| DistanceHeuristic | O(n² · k) | — |
| TaxonomyHeuristic | O(n²) | — |
| ThreatHeuristic | O(n² · k) | — |
| **WindowScorerHeuristic** | — | **O(n² · k)** |
| **Total** | **O(n³)** | **O(n² · k)** |

For a 4×4 board (n=4, k=3): reduction from ~64 to ~48 operations per evaluation.  
For a 10×10 board (n=10, k=5): reduction from ~1000 to ~200 operations per evaluation.

---

## Verification

1. Run existing tests: `source .venv/bin/activate && pytest src/tests/ -v`
2. Benchmark: time `python main.py` before and after, compare move count and wall time.
3. Spot-check `evaluate()` on known positions:
   - Empty board → 0.0
   - Player has k-1 in a row (open) → positive, close to 1.0
   - Opponent has k-1 in a row (open) → negative, close to -1.0
   - Both players equal threats → near 0.0
   - Player has `X _ X` with k=3 at edge (no open ends outside) → positive (WIN_THREAT_SCORE applied, not 0.0)
