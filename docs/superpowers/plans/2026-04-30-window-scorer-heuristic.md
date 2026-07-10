# Window Scorer Heuristic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 4-heuristic ensemble in `Heuristic` with a single `WindowScorerHeuristic` that scores all k-windows by (run length, openness), dropping ForkHeuristic's O(n³) cost to O(n²·k).

**Architecture:** Scan all k-windows in 4 directions; for each unblocked window with ≥1 player piece, apply a WIN_THREAT_SCORE for one-move wins (m=k-1, e=1) or an openness-weighted formula `open_ends * 4^(m-1)` otherwise. Normalize with `tanh`. `Heuristic` becomes a thin wrapper delegating to `WindowScorerHeuristic`.

**Tech Stack:** Python, numpy (existing), pytest (existing).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/heuristics/window_scorer_heuristic.py` | `_score_windows()` + `WindowScorerHeuristic` class |
| Modify | `src/heuristics/heuristic.py` | Replace 4-component ensemble with `WindowScorerHeuristic` |
| Modify | `src/tests/test_heuristics.py` | Add `TestWindowScorerHeuristic`; update `TestHeuristic` |

---

### Task 1: Write failing tests for `WindowScorerHeuristic`

**Files:**
- Modify: `src/tests/test_heuristics.py`

- [ ] **Step 1: Add the import and test class to `test_heuristics.py`**

Append to the bottom of `src/tests/test_heuristics.py`:

```python
from src.heuristics.window_scorer_heuristic import WindowScorerHeuristic


class TestWindowScorerHeuristic:
    def test_empty_board_returns_zero(self):
        h = WindowScorerHeuristic()
        assert h.evaluate(fresh_state(3, 3)) == pytest.approx(0.0)

    def test_result_in_bounds(self):
        state = state_with_moves([(0, 0), (0, 1), (1, 0), (1, 1)], n=5, k=3)
        h = WindowScorerHeuristic()
        assert -1.0 <= h.evaluate(state) <= 1.0

    def test_current_player_threat_returns_positive(self):
        # X:(0,0),(0,1) → row window has k-1=2 X pieces + 1 empty. O at corners. Current=X.
        state = state_with_moves([(0, 0), (3, 3), (0, 1), (3, 0)], n=4, k=3)
        h = WindowScorerHeuristic()
        assert h.evaluate(state) > 0.0

    def test_opponent_threat_returns_negative(self):
        # O:(0,0),(0,1) row threat; X at far corners. Current=X.
        state = state_with_moves([(3, 3), (0, 0), (3, 0), (0, 1)], n=4, k=3)
        h = WindowScorerHeuristic()
        assert h.evaluate(state) < 0.0

    def test_gap_threat_at_edge_scores_positive(self):
        # X at (0,0) and (0,2), k=3: window [X,_,X] has m=2, e=1 → WIN_THREAT_SCORE.
        # Even though left outside is out-of-bounds (open_ends formula would score 0),
        # WIN_THREAT_SCORE path fires regardless of openness.
        # O at bottom corners. Current=X.
        state = state_with_moves([(0, 0), (3, 3), (0, 2), (3, 0)], n=4, k=3)
        h = WindowScorerHeuristic()
        assert h.evaluate(state) > 0.0

    def test_implements_base_heuristic(self):
        from src.heuristics.base_heuristic import BaseHeuristic
        assert isinstance(WindowScorerHeuristic(), BaseHeuristic)
```

- [ ] **Step 2: Run tests to confirm they fail with ImportError**

```bash
source .venv/bin/activate && pytest src/tests/test_heuristics.py::TestWindowScorerHeuristic -v
```

Expected: `ERRORS` — `ModuleNotFoundError: No module named 'src.heuristics.window_scorer_heuristic'`

---

### Task 2: Implement `WindowScorerHeuristic`

**Files:**
- Create: `src/heuristics/window_scorer_heuristic.py`

- [ ] **Step 1: Create the file**

```python
from __future__ import annotations

import math

import numpy as np

from src.heuristics.base_heuristic import BaseHeuristic
from src.core.state import State

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _score_windows(grid: np.ndarray, n: int, k: int, player_val: int, opp_val: int) -> float:
    win_threat = float(4 ** (k - 1))
    total = 0.0
    for dr, dc in _DIRECTIONS:
        for r in range(n):
            for c in range(n):
                end_r = r + (k - 1) * dr
                end_c = c + (k - 1) * dc
                if not (0 <= end_r < n and 0 <= end_c < n):
                    continue
                m = 0
                e = 0
                blocked = False
                for i in range(k):
                    cell = grid[r + i * dr, c + i * dc]
                    if cell == opp_val:
                        blocked = True
                        break
                    elif cell == player_val:
                        m += 1
                    else:
                        e += 1
                if blocked or m == 0:
                    continue
                if m == k - 1 and e == 1:
                    total += win_threat
                    continue
                pr, pc = r - dr, c - dc
                nr, nc = r + k * dr, c + k * dc
                left_open = 0 <= pr < n and 0 <= pc < n and int(grid[pr, pc]) == 0
                right_open = 0 <= nr < n and 0 <= nc < n and int(grid[nr, nc]) == 0
                open_ends = int(left_open) + int(right_open)
                if open_ends == 0:
                    continue
                total += open_ends * float(4 ** (m - 1))
    return total


class WindowScorerHeuristic(BaseHeuristic):
    def evaluate(self, state: State) -> float:
        board = state.board
        n, k = board.n, board.k
        if k < 2:
            return 0.0
        me = int(state.current_player)
        opp = int(state.current_player.opponent())
        score_me = _score_windows(board.board, n, k, me, opp)
        score_opp = _score_windows(board.board, n, k, opp, me)
        raw = score_me - score_opp
        scale = 2.0 * float(4 ** (k - 2))
        return math.tanh(raw / scale)
```

- [ ] **Step 2: Run the new tests to verify they pass**

```bash
source .venv/bin/activate && pytest src/tests/test_heuristics.py::TestWindowScorerHeuristic -v
```

Expected: 6 tests PASSED.

- [ ] **Step 3: Commit**

```bash
git add src/heuristics/window_scorer_heuristic.py src/tests/test_heuristics.py
git commit -m "feat: implement WindowScorerHeuristic with gap-threat detection"
```

---

### Task 3: Update `Heuristic` to use `WindowScorerHeuristic`

**Files:**
- Modify: `src/heuristics/heuristic.py`
- Modify: `src/tests/test_heuristics.py` (fix `TestHeuristic`)

- [ ] **Step 1: Replace `heuristic.py` contents**

```python
from __future__ import annotations

from src.heuristics.base_heuristic import BaseHeuristic
from src.heuristics.window_scorer_heuristic import WindowScorerHeuristic
from src.core.state import State


class Heuristic(BaseHeuristic):
    def __init__(self) -> None:
        self._scorer = WindowScorerHeuristic()

    def evaluate(self, state: State) -> float:
        return self._scorer.evaluate(state)
```

- [ ] **Step 2: Remove the stale `test_components_count` test from `TestHeuristic` in `test_heuristics.py`**

Delete this method from `TestHeuristic`:

```python
    def test_components_count(self):
        h = Heuristic()
        assert len(h._components) == 4
```

- [ ] **Step 3: Run all heuristic tests**

```bash
source .venv/bin/activate && pytest src/tests/test_heuristics.py -v
```

Expected: all tests PASSED. The individual `TestDistanceHeuristic`, `TestTaxonomyHeuristic`, `TestForkHeuristic`, `TestThreatHeuristic` tests still pass because those files are unchanged; `TestHeuristic` and `TestWindowScorerHeuristic` pass with the new implementation.

- [ ] **Step 4: Run full test suite**

```bash
source .venv/bin/activate && pytest src/tests/ -v
```

Expected: all tests PASSED. In particular `TestMTDfIDAgentHeuristicIntegration` must pass — it calls `agent._negamax_tt(..., depth=0, ...)` and expects a positive score and correct move selection, which `WindowScorerHeuristic` provides.

- [ ] **Step 5: Commit**

```bash
git add src/heuristics/heuristic.py src/tests/test_heuristics.py
git commit -m "refactor: replace 4-heuristic ensemble with WindowScorerHeuristic"
```
