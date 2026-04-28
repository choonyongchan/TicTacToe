# MTD(f) Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `TranspositionTable` in `src/core/` and `MTDfAgent` in `src/agents/`, making all 38 pre-written tests pass.

**Architecture:** `TranspositionTable` stores `(lowerbound, upperbound, best_move)` per Zobrist hash key — no flag enum, matching Plaat's original pseudocode. `MTDfAgent` runs MTD(f) via repeated null-window `_negamax_tt` probes (window width = `_epsilon`), retrieves the best move from the TT root entry after convergence.

**Tech Stack:** Python 3.10+, pytest, src.core.types.NEGATIVE_INFINITY

---

## File Map

| File | Action |
|------|--------|
| `src/core/transposition_table.py` | Create |
| `src/tests/test_transposition_table.py` | Create |
| `src/agents/mtdf_agent.py` | Create |
| `src/tests/test_mtdf_agent.py` | Modify (add import, swap `dict` for `TranspositionTable`) |

---

### Task 1: TranspositionTable

**Files:**
- Create: `src/core/transposition_table.py`
- Create: `src/tests/test_transposition_table.py`

- [ ] **Step 1: Write the failing test file**

Create `src/tests/test_transposition_table.py`:

```python
import pytest

from src.core.transposition_table import TranspositionTable
from src.core.types import NEGATIVE_INFINITY

INF = -NEGATIVE_INFINITY  # float("inf")


class TestInit:
    def test_empty_on_creation(self):
        assert len(TranspositionTable()) == 0


class TestStore:
    def test_len_increases_on_new_key(self):
        tt = TranspositionTable()
        tt.store(1, NEGATIVE_INFINITY, INF, None)
        assert len(tt) == 1

    def test_len_two_distinct_keys(self):
        tt = TranspositionTable()
        tt.store(1, NEGATIVE_INFINITY, INF, None)
        tt.store(2, NEGATIVE_INFINITY, INF, None)
        assert len(tt) == 2

    def test_overwrite_same_key_no_len_change(self):
        tt = TranspositionTable()
        tt.store(1, NEGATIVE_INFINITY, INF, None)
        tt.store(1, 0.5, 0.5, (0, 0))
        assert len(tt) == 1


class TestLookup:
    def test_miss_returns_none(self):
        assert TranspositionTable().lookup(42) is None

    def test_hit_returns_full_tuple(self):
        tt = TranspositionTable()
        tt.store(7, -0.5, 0.8, (1, 2))
        assert tt.lookup(7) == (-0.5, 0.8, (1, 2))

    def test_lower_bound_stored(self):
        tt = TranspositionTable()
        tt.store(1, 0.3, INF, None)
        lb, ub, _ = tt.lookup(1)
        assert lb == pytest.approx(0.3)
        assert ub == INF

    def test_upper_bound_stored(self):
        tt = TranspositionTable()
        tt.store(1, NEGATIVE_INFINITY, 0.3, None)
        lb, ub, _ = tt.lookup(1)
        assert lb == NEGATIVE_INFINITY
        assert ub == pytest.approx(0.3)

    def test_exact_stored(self):
        tt = TranspositionTable()
        tt.store(1, 0.3, 0.3, (0, 1))
        lb, ub, bm = tt.lookup(1)
        assert lb == pytest.approx(0.3)
        assert ub == pytest.approx(0.3)
        assert bm == (0, 1)

    def test_none_best_move_round_trips(self):
        tt = TranspositionTable()
        tt.store(5, 0.0, 0.0, None)
        assert tt.lookup(5)[2] is None


class TestBestMove:
    def test_miss_returns_none(self):
        assert TranspositionTable().best_move(99) is None

    def test_hit_returns_move(self):
        tt = TranspositionTable()
        tt.store(3, 0.0, 0.5, (1, 2))
        assert tt.best_move(3) == (1, 2)

    def test_none_best_move_returns_none(self):
        tt = TranspositionTable()
        tt.store(3, 0.0, 0.5, None)
        assert tt.best_move(3) is None


class TestOverwrite:
    def test_overwrite_updates_lower(self):
        tt = TranspositionTable()
        tt.store(1, NEGATIVE_INFINITY, 0.5, None)
        tt.store(1, 0.3, 0.5, None)
        lb, _, _ = tt.lookup(1)
        assert lb == pytest.approx(0.3)

    def test_overwrite_updates_upper(self):
        tt = TranspositionTable()
        tt.store(1, NEGATIVE_INFINITY, 0.5, None)
        tt.store(1, NEGATIVE_INFINITY, 0.3, None)
        _, ub, _ = tt.lookup(1)
        assert ub == pytest.approx(0.3)

    def test_overwrite_updates_best_move(self):
        tt = TranspositionTable()
        tt.store(1, 0.0, 0.0, (0, 0))
        tt.store(1, 0.0, 0.0, (1, 1))
        assert tt.best_move(1) == (1, 1)
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
source .venv/bin/activate && python -m pytest src/tests/test_transposition_table.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.core.transposition_table'`

- [ ] **Step 3: Implement TranspositionTable**

Create `src/core/transposition_table.py`:

```python
from __future__ import annotations


class TranspositionTable:
    def __init__(self) -> None:
        self._table: dict[int, tuple[float, float, tuple[int, int] | None]] = {}

    def lookup(self, key: int) -> tuple[float, float, tuple[int, int] | None] | None:
        return self._table.get(key)

    def store(
        self,
        key: int,
        lower: float,
        upper: float,
        best_move: tuple[int, int] | None,
    ) -> None:
        self._table[key] = (lower, upper, best_move)

    def best_move(self, key: int) -> tuple[int, int] | None:
        entry = self._table.get(key)
        return entry[2] if entry is not None else None

    def __len__(self) -> int:
        return len(self._table)
```

- [ ] **Step 4: Run tests — expect all 16 pass**

```bash
python -m pytest src/tests/test_transposition_table.py -v
```

Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add src/core/transposition_table.py src/tests/test_transposition_table.py
git commit -m "feat: add TranspositionTable to src/core with 16 tests"
```

---

### Task 2: MTDfAgent

**Files:**
- Create: `src/agents/mtdf_agent.py`
- Modify: `src/tests/test_mtdf_agent.py`

- [ ] **Step 1: Update test_mtdf_agent.py imports and TT construction**

In `src/tests/test_mtdf_agent.py`:

Add to the imports block (after existing imports):
```python
from src.core.transposition_table import TranspositionTable
```

Replace every occurrence of `tt: dict = {}` with `tt = TranspositionTable()`.
There are 10 occurrences — they all live in `TestMtdf`, `TestNegamaxTt`, and one in `TestMtdf.test_tt_populated_after_search`. Use find-and-replace; no logic changes.

- [ ] **Step 2: Run updated tests — expect ImportError**

```bash
python -m pytest src/tests/test_mtdf_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.mtdf_agent'`

- [ ] **Step 3: Implement MTDfAgent**

Create `src/agents/mtdf_agent.py`:

```python
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.core.state import State
from src.core.transposition_table import TranspositionTable
from src.core.types import NEGATIVE_INFINITY


class MTDfAgent(BaseAgent):
    def __init__(self, max_depth: int) -> None:
        super().__init__("MTDfAgent")
        self._epsilon = 1.0 / (max_depth + 1)

    def act(self, state: State) -> tuple[int, int]:
        tt = TranspositionTable()
        self._mtdf(state, 0.0, tt)
        best = tt.best_move(state._hash)
        if best is not None:
            return best
        # Fallback: full-window sweep (shouldn't trigger in practice)
        best_score = NEGATIVE_INFINITY
        best_move: tuple[int, int] | None = None
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = -self._negamax_tt(state, NEGATIVE_INFINITY, -NEGATIVE_INFINITY, tt)
            state.undo()
            if score > best_score:
                best_score = score
                best_move = (row, col)
        assert best_move is not None
        return best_move

    def _terminal_score(self, state: State) -> float:
        if state.winner() is None:
            return 0.0
        return 1.0 - self._epsilon * len(state.history)

    def _negamax_tt(
        self,
        state: State,
        alpha: float,
        beta: float,
        tt: TranspositionTable,
    ) -> float:
        h = state._hash

        entry = tt.lookup(h)
        if entry is not None:
            lb, ub, _ = entry
            if lb >= beta:
                return lb
            if ub <= alpha:
                return ub
            alpha = max(alpha, lb)
            beta = min(beta, ub)
            if alpha >= beta:
                return lb

        # Capture window AFTER TT tightening — matches pseudocode "a := alpha"
        # so that fail-low/high classification is relative to the actual window used.
        original_alpha = alpha
        original_beta = beta

        if state.is_terminal():
            g = -self._terminal_score(state)
            tt.store(h, g, g, None)
            return g

        g = NEGATIVE_INFINITY
        best_move: tuple[int, int] | None = None

        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            score = -self._negamax_tt(state, -beta, -alpha, tt)
            state.undo()
            if score > g:
                g = score
                best_move = (row, col)
            if g >= beta:
                break
            if g > alpha:
                alpha = g

        # Merge new bound with the other existing bound (preserves prior probe's work)
        existing = tt.lookup(h)
        lb = existing[0] if existing is not None else NEGATIVE_INFINITY
        ub = existing[1] if existing is not None else -NEGATIVE_INFINITY

        if g <= original_alpha:
            ub = g
        elif g >= original_beta:
            lb = g
        else:
            lb = ub = g

        tt.store(h, lb, ub, best_move)
        return g

    def _mtdf(self, state: State, f: float, tt: TranspositionTable) -> float:
        lower = NEGATIVE_INFINITY
        upper = -NEGATIVE_INFINITY  # +inf

        while lower < upper:
            beta = f if f > lower else lower + self._epsilon
            g = self._negamax_tt(state, beta - self._epsilon, beta, tt)
            if g < beta:
                upper = g
            else:
                lower = g
            f = g

        return f
```

- [ ] **Step 4: Run all MTD(f) tests — expect all 22 pass**

```bash
python -m pytest src/tests/test_mtdf_agent.py -v
```

Expected: 22 passed.

- [ ] **Step 5: Run full test suite — expect no regressions**

```bash
python -m pytest src/tests/ -v
```

Expected: all tests pass (38 new + all prior tests).

- [ ] **Step 6: Commit**

```bash
git add src/agents/mtdf_agent.py src/tests/test_mtdf_agent.py
git commit -m "feat: implement MTDfAgent with TranspositionTable (null-window MTD(f))"
```
