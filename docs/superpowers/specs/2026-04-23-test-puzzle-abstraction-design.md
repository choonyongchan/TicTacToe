# Test Puzzle Abstraction Design

**Date:** 2026-04-23  
**Status:** Approved

## Goal

Eliminate duplicated test helpers across five test files and introduce a shared `Puzzle` abstraction with three concrete board positions (3×3 k=3, 4×4 k=4, 5×5 k=3) that any agent test can import to verify correctness on multiple board configurations.

---

## Section 1 — `src/tests/test_helper.py` (new file)

Single source of truth for shared test utilities and puzzle fixtures.

### Generalized helpers

```python
def fresh_state(n: int = 3, k: int = 3) -> State:
    return State(n, k)

def state_with_moves(moves: list[tuple[int, int]], n: int = 3, k: int = 3) -> State:
    s = State(n, k)
    for row, col in moves:
        s.apply(row, col)
    return s
```

These replace the identical functions duplicated in every existing test file. Default values preserve backward compatibility for callers that omit `n`/`k`.

### `Puzzle` dataclass

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Puzzle:
    n: int                          # board side length
    k: int                          # k-in-a-row to win
    moves: list[tuple[int, int]]    # move sequence to reach the position
    best_move: tuple[int, int]      # expected optimal move from this position
    description: str                # board diagram + hand-traced minimax comment
```

`frozen=True` prevents accidental mutation. `description` is a multiline string containing the ASCII board diagram and hand-trace, matching the existing documentation style in the test files.

### Three puzzle constants

**`PUZZLE_3X3`** — existing 3×3 k=3 position, extracted from current `DUMMY_TREE_MOVES`.

```
Board after: X(0,0), O(0,1), X(1,1), O(2,2), X(1,0), O(2,0)
  X | O | .
  X | X | .
  O | . | O
X to move. best_move = (1, 2)  →  row 1 = X,X,X → X wins
```

**`PUZZLE_4X4`** — 4×4 k=4 near-terminal position, 2 empty cells, X to move.

```
Board after 14 moves:
  X | X | X | .
  O | O | O | .
  X | O | X | O
  O | X | O | X
Empty: (0,3), (1,3). X to move.

Hand-trace:
  X(0,3) → row 0 = X,X,X,X → X wins → score = 1
  X(1,3) → O plays (0,3) → row 1 = O,O,O,O → O wins → score = -1
X maximises: best_move = (0, 3)
```

Move sequence: `[(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,0),(2,1),(2,2),(2,3),(3,1),(3,0),(3,3),(3,2)]`

**`PUZZLE_5X5`** — 5×5 k=3 near-terminal position, X to move.

```
Board: a position where X has two in a row and (r,c) completes a 3-in-a-row,
while O has a near-win that X must ignore because the immediate win scores higher.
best_move to be finalized with hand-trace during implementation.
```

The exact 5×5 move sequence and `best_move` will be determined and hand-verified during implementation, following the same hand-trace documentation convention as the 3×3 and 4×4 puzzles.

---

## Section 2 — Updates to existing test files

Each of the five test files replaces local helpers and `DUMMY_TREE_MOVES` with imports:

```python
from src.tests.test_helper import fresh_state, state_with_moves, PUZZLE_3X3
```

The constant `DUMMY_TREE_MOVES` references are replaced with `PUZZLE_3X3.moves`. No test logic changes — only the source of the data changes.

Files affected:
- `src/tests/test_agents.py`
- `src/tests/test_minimax_agent.py`
- `src/tests/test_minimax_alphabeta_agent.py`
- `src/tests/test_minimax_rewards_alphabeta_agent.py`
- `src/tests/test_core.py`

---

## Section 3 — New agent test sections using PUZZLE_4X4 and PUZZLE_5X5

Each of the three agent test files (`test_minimax_agent.py`, `test_minimax_alphabeta_agent.py`, `test_minimax_rewards_alphabeta_agent.py`) gets a new test class using the larger puzzles:

```python
class TestActLargerBoards:
    def test_4x4_picks_best_move(self):
        state = state_with_moves(PUZZLE_4X4.moves, PUZZLE_4X4.n, PUZZLE_4X4.k)
        agent = <AgentClass>(Player.X)
        assert agent.act(state) == PUZZLE_4X4.best_move

    def test_5x5_picks_best_move(self):
        state = state_with_moves(PUZZLE_5X5.moves, PUZZLE_5X5.n, PUZZLE_5X5.k)
        agent = <AgentClass>(Player.X)
        assert agent.act(state) == PUZZLE_5X5.best_move
```

`test_agents.py` (RandomAgent) is excluded — RandomAgent has no concept of optimal play.

---

## Implementation Order

1. Create `src/tests/test_helper.py` with helpers, `Puzzle` dataclass, and all three puzzle constants.
2. Update `test_core.py` to import from `test_helper.py`.
3. Update the four agent test files to import from `test_helper.py` and replace `DUMMY_TREE_MOVES`.
4. Add `TestActLargerBoards` to the three minimax agent test files.
5. Run full test suite to confirm no regressions.
