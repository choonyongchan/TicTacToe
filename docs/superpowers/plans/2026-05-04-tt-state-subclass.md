# TTState Subclass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `TTState(State)` so that symmetry-hash machinery (`_hashes`, `Manipulator` loops) runs only when a Transposition Table agent is in use.

**Architecture:** `State` is stripped of `_hashes` and the `Manipulator` import. A new `TTState` subclass overrides `apply()`, `undo()`, and `reset()` to maintain the 8-element `_hashes` list. All TT agents and their tests receive `TTState`; non-TT agents use the leaner `State`.

**Tech Stack:** Python 3.11+, pytest

---

## File Map

| File | Change |
|---|---|
| `src/core/state.py` | Remove `_hashes`, manipulator loops, `Manipulator` import |
| `src/core/tt_state.py` | **Create** — `TTState(State)` with `_hashes` |
| `src/tests/test_core.py` | Add `TTState` tests; add `State-has-no-_hashes` test |
| `src/tests/test_helper.py` | Import + use `TTState` instead of `State` |
| `main.py` | Import `TTState`; use it for TT agents |

---

## Task 1: Create `TTState` with `_hashes` initialisation

**Files:**
- Create: `src/core/tt_state.py`
- Modify: `src/tests/test_core.py`

- [ ] **Step 1: Write the failing test**

Append to the bottom of `src/tests/test_core.py`:

```python
# ---------------------------------------------------------------------------
# tt_state.py
# ---------------------------------------------------------------------------

from src.core.tt_state import TTState
from src.core.manipulator import Manipulator


class TestTTStateInit:
    def test_hashes_length(self):
        s = TTState(3, 3)
        assert len(s._hashes) == Manipulator.TRANSFORM_COUNT

    def test_hashes_all_zero(self):
        s = TTState(3, 3)
        assert s._hashes == [0] * Manipulator.TRANSFORM_COUNT

    def test_inherits_state_behaviour(self):
        s = TTState(3, 3)
        assert s.current_player is Player.X
        assert s.history == []
        assert s._hash == 0
```

- [ ] **Step 2: Run test to verify it fails**

```
source .venv/bin/activate && pytest src/tests/test_core.py::TestTTStateInit -v
```

Expected: `ModuleNotFoundError: No module named 'src.core.tt_state'`

- [ ] **Step 3: Create `src/core/tt_state.py`**

```python
from __future__ import annotations

from .manipulator import Manipulator
from .state import State


class TTState(State):
    """State subclass that maintains all 8 symmetry-equivalent Zobrist hashes.

    Required by agents that use a TranspositionTable. Plain State does not
    track _hashes, so passing a plain State to a TT agent will raise AttributeError.
    """

    def __init__(self, n: int, k: int) -> None:
        super().__init__(n, k)
        self._hashes: list[int] = [0] * Manipulator.TRANSFORM_COUNT
```

- [ ] **Step 4: Run test to verify it passes**

```
source .venv/bin/activate && pytest src/tests/test_core.py::TestTTStateInit -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/core/tt_state.py src/tests/test_core.py
git commit -m "feat: add TTState subclass with _hashes initialisation"
```

---

## Task 2: Implement `TTState.apply()` — symmetry hash updates

**Files:**
- Modify: `src/core/tt_state.py`
- Modify: `src/tests/test_core.py`

- [ ] **Step 1: Write the failing tests**

Append to `TestTTStateInit` class in `src/tests/test_core.py` (or add a new class below it):

```python
class TestTTStateApply:
    def test_apply_changes_hashes(self):
        s = TTState(3, 3)
        s.apply(0, 0)
        assert s._hashes != [0] * Manipulator.TRANSFORM_COUNT

    def test_identity_hash_matches_main_hash(self):
        # Transform index 0 is the identity; its hash must equal _hash
        s = TTState(3, 3)
        s.apply(1, 1)
        assert s._hashes[0] == s._hash

    def test_apply_two_moves_hashes_nonzero(self):
        s = TTState(3, 3)
        s.apply(0, 0)
        s.apply(1, 1)
        assert any(h != 0 for h in s._hashes)

    def test_apply_same_position_same_hashes(self):
        s1 = TTState(3, 3)
        s1.apply(0, 0)
        s2 = TTState(3, 3)
        s2.apply(0, 0)
        assert s1._hashes == s2._hashes
```

- [ ] **Step 2: Run tests to verify they fail**

```
source .venv/bin/activate && pytest src/tests/test_core.py::TestTTStateApply -v
```

Expected: FAILED — `_hashes` not changed by `apply()` (still all zeros)

- [ ] **Step 3: Implement `TTState.apply()`**

Add to `src/core/tt_state.py`:

```python
    def apply(self, row: int, col: int) -> None:
        super().apply(row, col)
        # current_player is already flipped after super(); opponent() recovers who just played
        player_val = int(self.current_player.opponent())
        for i, (tr, tc) in enumerate(
            Manipulator.all_transform_moves((row, col), self.board.n)
        ):
            self._hashes[i] ^= int(self._zobrist._table[tr, tc, player_val])
```

- [ ] **Step 4: Run tests to verify they pass**

```
source .venv/bin/activate && pytest src/tests/test_core.py::TestTTStateApply src/tests/test_core.py::TestTTStateInit -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add src/core/tt_state.py src/tests/test_core.py
git commit -m "feat: TTState.apply() updates symmetry hashes via Manipulator"
```

---

## Task 3: Implement `TTState.undo()` — symmetry hash restoration

**Files:**
- Modify: `src/core/tt_state.py`
- Modify: `src/tests/test_core.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/tests/test_core.py`:

```python
class TestTTStateUndo:
    def test_undo_restores_hashes_to_zero(self):
        s = TTState(3, 3)
        s.apply(0, 0)
        s.undo()
        assert s._hashes == [0] * Manipulator.TRANSFORM_COUNT

    def test_undo_two_moves(self):
        s = TTState(3, 3)
        s.apply(0, 0)
        s.apply(1, 1)
        after_first = list(s._hashes)  # snapshot after first apply only would differ, but we test undo
        s.undo()
        s.undo()
        assert s._hashes == [0] * Manipulator.TRANSFORM_COUNT

    def test_undo_restores_identity_hash(self):
        s = TTState(3, 3)
        s.apply(1, 1)
        s.undo()
        assert s._hashes[0] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
source .venv/bin/activate && pytest src/tests/test_core.py::TestTTStateUndo -v
```

Expected: FAILED — `_hashes` not restored after `undo()`

- [ ] **Step 3: Implement `TTState.undo()`**

Add to `src/core/tt_state.py`:

```python
    def undo(self) -> None:
        row, col = self.history[-1]          # peek before super() pops history
        prev_val = int(self.current_player.opponent())  # who made the last move
        super().undo()
        for i, (tr, tc) in enumerate(
            Manipulator.all_transform_moves((row, col), self.board.n)
        ):
            self._hashes[i] ^= int(self._zobrist._table[tr, tc, prev_val])
```

- [ ] **Step 4: Run tests to verify they pass**

```
source .venv/bin/activate && pytest src/tests/test_core.py::TestTTStateUndo src/tests/test_core.py::TestTTStateApply src/tests/test_core.py::TestTTStateInit -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add src/core/tt_state.py src/tests/test_core.py
git commit -m "feat: TTState.undo() restores symmetry hashes via Manipulator"
```

---

## Task 4: Implement `TTState.reset()` — zero out `_hashes`

**Files:**
- Modify: `src/core/tt_state.py`
- Modify: `src/tests/test_core.py`

- [ ] **Step 1: Write the failing test**

Append to `src/tests/test_core.py`:

```python
class TestTTStateReset:
    def test_reset_zeros_hashes(self):
        s = TTState(3, 3)
        s.apply(0, 0)
        s.reset()
        assert s._hashes == [0] * Manipulator.TRANSFORM_COUNT

    def test_reset_preserves_hashes_attribute(self):
        s = TTState(3, 3)
        s.reset()
        assert hasattr(s, "_hashes")
```

- [ ] **Step 2: Run tests to verify they fail**

```
source .venv/bin/activate && pytest src/tests/test_core.py::TestTTStateReset -v
```

Expected: FAILED — `reset()` calls `State.reset()` which tries to reset `self._hashes` (that line will raise `AttributeError` on plain `State` after Task 5, but TTState inherits it and currently State still has it — still useful to add the override)

Actually at this point `State` still has `_hashes`, so `State.reset()` currently zeros `TTState._hashes` correctly and the test will PASS. That is expected — the test serves as a guard for after Task 5.

Run anyway to confirm they pass:

```
source .venv/bin/activate && pytest src/tests/test_core.py::TestTTStateReset -v
```

Expected: PASSED (because State.reset() still zeros _hashes)

- [ ] **Step 3: Add `TTState.reset()` override**

This override is needed so that after Task 5 strips `_hashes` from `State.reset()`, `TTState` still zeros its own `_hashes`. Add to `src/core/tt_state.py`:

```python
    def reset(self) -> None:
        super().reset()
        self._hashes = [0] * Manipulator.TRANSFORM_COUNT
```

- [ ] **Step 4: Verify tests still pass**

```
source .venv/bin/activate && pytest src/tests/test_core.py::TestTTStateReset -v
```

Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add src/core/tt_state.py src/tests/test_core.py
git commit -m "feat: TTState.reset() re-zeros symmetry hashes"
```

---

## Task 5: Update `test_helper.py` to return `TTState`

**Files:**
- Modify: `src/tests/test_helper.py`

This must happen before stripping `_hashes` from `State`, otherwise TT agent tests (which call `fresh_state()` / `state_with_moves()`) will break.

- [ ] **Step 1: Run TT agent tests to confirm they currently pass**

```
source .venv/bin/activate && pytest src/tests/test_mtdf_agent.py src/tests/test_mtdf_id_agent.py src/tests/test_bns_id_agent.py -v --tb=no -q
```

Expected: all PASSED

- [ ] **Step 2: Update `src/tests/test_helper.py`**

Replace the file content:

```python
"""Shared test helpers and puzzle fixtures for agent tests."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from src.core.tt_state import TTState


def fresh_state(n: int = 3, k: int = 3) -> TTState:
    return TTState(n, k)


def state_with_moves(moves: Sequence[tuple[int, int]], n: int = 3, k: int = 3) -> TTState:
    s = TTState(n, k)
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

# ---------------------------------------------------------------------------
# 4×4, k=3 — O must block X's column-0 threat
# ---------------------------------------------------------------------------
#
# Board after 3 moves (X,O,X), O to move:
#   X | . | . | .
#   X | O | . | .
#   . | . | . | .
#   . | . | . | .
#
# X at (0,0),(1,0) threatens (2,0) for a 3-in-column win.
# O at (1,1) has no immediate win.
# O must block at (2,0).

PUZZLE_4X4_BLOCK = Puzzle(
    n=4,
    k=3,
    moves=((0, 0), (1, 1), (1, 0)),
    best_move=(2, 0),
    description="4×4 k=3 blocking: O to move, must block X's column-0 win at (2,0)",
)


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
```

- [ ] **Step 3: Run TT agent tests to confirm they still pass**

```
source .venv/bin/activate && pytest src/tests/test_mtdf_agent.py src/tests/test_mtdf_id_agent.py src/tests/test_bns_id_agent.py -v --tb=short -q
```

Expected: all PASSED

- [ ] **Step 4: Commit**

```bash
git add src/tests/test_helper.py
git commit -m "refactor: test_helper returns TTState so TT agent tests get symmetry hashes"
```

---

## Task 6: Strip `_hashes` and `Manipulator` from `State`

**Files:**
- Modify: `src/core/state.py`
- Modify: `src/tests/test_core.py`

- [ ] **Step 1: Write the guard test**

Append to `src/tests/test_core.py`:

```python
class TestStateNoHashesAttribute:
    def test_plain_state_has_no_hashes(self):
        assert not hasattr(State(3, 3), "_hashes")
```

- [ ] **Step 2: Run guard test to verify it fails (State still has _hashes now)**

```
source .venv/bin/activate && pytest src/tests/test_core.py::TestStateNoHashesAttribute -v
```

Expected: FAILED — `State` currently has `_hashes`

- [ ] **Step 3: Edit `src/core/state.py`**

Remove the `Manipulator` import line:
```python
from .manipulator import Manipulator   # DELETE this line
```

In `__init__`, remove:
```python
self._hashes: list[int] = [0] * Manipulator.TRANSFORM_COUNT   # DELETE
```

In `apply()`, remove the docstring line mentioning symmetry hashes and remove the for-loop:
```python
        for i, (tr, tc) in enumerate(
            Manipulator.all_transform_moves((row, col), self.board.n)
        ):
            self._hashes[i] ^= int(self._zobrist._table[tr, tc, player_val])
```

In `undo()`, remove the for-loop:
```python
        for i, (tr, tc) in enumerate(
            Manipulator.all_transform_moves((row, col), self.board.n)
        ):
            self._hashes[i] ^= int(self._zobrist._table[tr, tc, prev_val])
```

In `reset()`, remove:
```python
self._hashes = [0] * Manipulator.TRANSFORM_COUNT   # DELETE
```

After editing, `src/core/state.py` should look like:

```python
from __future__ import annotations

from .board import Board
from .types import Player
from .zobrist import ZobristTable


class State:
    """Mutable game state tracking board, turn, move history, and Zobrist hash.

    Attributes:
        board: The underlying Board instance.
        current_player: Player whose turn it is.
        history: Ordered list of (row, col) moves applied so far.
        candidate_d: Chebyshev radius used for candidate cell generation.
    """

    def __init__(self, n: int = 3, k: int = 3) -> None:
        self.board = Board(n, k)
        self._zobrist = ZobristTable(n)
        self.current_player: Player = Player.X
        self.history: list[tuple[int, int]] = []
        self._state_count: int = 0
        self._visited: set[int] = set()
        self._hash: int = 0
        self.candidate_d: int = max(1, self.board.k - 2)

    def apply(self, row: int, col: int) -> None:
        """Place the current player's piece and advance the turn.

        Args:
            row: Row of the cell to play.
            col: Column of the cell to play.
        """
        self.board.set(row, col, self.current_player)
        self.history.append((row, col))
        player_val = int(self.current_player)
        self._hash = self._zobrist.hash_move(self._hash, row, col, player_val)
        if self._hash not in self._visited:
            self._visited.add(self._hash)
            self._state_count += 1
        self.current_player = self.current_player.opponent()

    def undo(self) -> None:
        """Remove the last placed piece and revert the turn."""
        row, col = self.history.pop()
        prev_player = self.current_player.opponent()
        prev_val = int(prev_player)
        self._hash = self._zobrist.hash_move(self._hash, row, col, prev_val)
        self.board.set(row, col, Player._)
        self.current_player = prev_player

    def is_terminal(self) -> bool:
        """Return True if the game has ended (win or draw)."""
        if not self.history:
            return False
        if self.board.check_win(*self.history[-1]):
            return True
        return self.board.is_full()

    def winner(self) -> Player | None:
        """Return the winning player, or None if the game is not yet won.

        Returns:
            Winning Player if the last move completed a run, else None.
        """
        if not self.history:
            return None
        if self.board.check_win(*self.history[-1]):
            return self.current_player.opponent()
        return None

    @property
    def state_count(self) -> int:
        """Total distinct board positions visited across the game."""
        return self._state_count

    def reset(self) -> None:
        """Reset all state to the beginning of a new game."""
        self.board.reset()
        self.current_player = Player.X
        self.history = []
        self._state_count = 0
        self._visited = set()
        self._hash = 0
```

- [ ] **Step 4: Run guard test and full test suite**

```
source .venv/bin/activate && pytest src/tests/test_core.py::TestStateNoHashesAttribute -v
```
Expected: PASSED

```
source .venv/bin/activate && pytest src/tests/ -v --tb=short -q
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add src/core/state.py src/tests/test_core.py
git commit -m "refactor: strip _hashes and Manipulator from State; symmetry hashing moved to TTState"
```

---

## Task 7: Update `main.py` to use `TTState` for TT agents

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update `main.py`**

Add `TTState` import alongside `State`:

```python
from src.core.state import State
from src.core.tt_state import TTState
```

Define which agent keys require symmetry hashes:

```python
_TT_AGENTS = {"mtdf", "mtdf_id", "bns_id"}
```

In `Main.__init__`, replace the state creation line:

```python
self._state = State(n=n, k=k)
```

with:

```python
self._state = TTState(n=n, k=k) if agent in _TT_AGENTS else State(n=n, k=k)
```

After editing, the relevant section of `main.py` looks like:

```python
from src.core.state import State
from src.core.tt_state import TTState
from src.core.types import Player

_TT_AGENTS = {"mtdf", "mtdf_id", "bns_id"}

# ... (AGENTS dict unchanged) ...

class Main:
    def __init__(self, n: int, k: int, agent: str, verbose: bool) -> None:
        self._n = n
        self._k = k
        max_depth = n * n
        self._agents: dict[str, BaseAgent] = {
            p: AGENTS[agent](p, max_depth) for p in ("X", "O")
        }
        self._state = TTState(n=n, k=k) if agent in _TT_AGENTS else State(n=n, k=k)
        self._verbose = verbose
```

- [ ] **Step 2: Run a smoke test for a TT agent**

```
source .venv/bin/activate && python main.py -n 3 -k 3 -agt mtdf_id -v
```

Expected: game plays to completion, prints winner/draw and time.

- [ ] **Step 3: Run a smoke test for a non-TT agent**

```
source .venv/bin/activate && python main.py -n 3 -k 3 -agt negamax -v
```

Expected: game plays to completion.

- [ ] **Step 4: Run full test suite**

```
source .venv/bin/activate && pytest src/tests/ -v --tb=short
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: main.py uses TTState for TT agents, plain State for others"
```
