# TTState Subclass Design

**Date:** 2026-05-04

## Context

`State.apply()` and `State.undo()` currently always compute all 8 symmetry-equivalent Zobrist hashes via `Manipulator.all_transform_moves()` and maintain `_hashes: list[int]`. This work is only useful to agents that exploit a Transposition Table (TT). Non-TT agents (e.g., `RandomAgent`, `NegamaxAgent`) pay the cost unnecessarily. The goal is to gate this work behind a subclass so only TT agents pay for it.

## Design

### `State` (cleaned up, `src/core/state.py`)

Remove all symmetry-hash machinery:

- `__init__`: remove `self._hashes = [0] * Manipulator.TRANSFORM_COUNT`
- `apply()`: remove the `for i, (tr, tc) in enumerate(Manipulator.all_transform_moves(...))` loop
- `undo()`: same removal
- `reset()`: remove `self._hashes = [0] * Manipulator.TRANSFORM_COUNT`
- Remove `from .manipulator import Manipulator` import (no longer used)

Everything else stays: `_hash`, `_zobrist`, `_visited`, `_state_count`, `history`, `board`, `current_player`.

### `TTState(State)` (new file, `src/core/tt_state.py`)

Subclass that adds symmetry-hash tracking on top of `State`:

```python
class TTState(State):
    def __init__(self, n: int, k: int) -> None:
        super().__init__(n, k)
        self._hashes: list[int] = [0] * Manipulator.TRANSFORM_COUNT

    def apply(self, row: int, col: int) -> None:
        super().apply(row, col)
        # current_player is already flipped; opponent() recovers who just played
        player_val = int(self.current_player.opponent())
        for i, (tr, tc) in enumerate(
            Manipulator.all_transform_moves((row, col), self.board.n)
        ):
            self._hashes[i] ^= int(self._zobrist._table[tr, tc, player_val])

    def undo(self) -> None:
        row, col = self.history[-1]           # peek before super pops
        prev_val = int(self.current_player.opponent())  # who made the last move
        super().undo()
        for i, (tr, tc) in enumerate(
            Manipulator.all_transform_moves((row, col), self.board.n)
        ):
            self._hashes[i] ^= int(self._zobrist._table[tr, tc, prev_val])

    def reset(self) -> None:
        super().reset()
        self._hashes = [0] * Manipulator.TRANSFORM_COUNT
```

### Call-site changes

| File | Change |
|---|---|
| `src/core/__init__.py` | Export `TTState` |
| `main.py` | `from src.core.tt_state import TTState`; pass `TTState(n, k)` when using `MTDfIDAgent` / `BNSIDAgent` |
| `src/agents/tt_depth_agent.py` | Docstring: note that `State` must be a `TTState` |
| `src/agents/mtdf_agent.py` | Same docstring note |
| `src/agents/mtdf_id_agent.py` | Same docstring note |
| `src/agents/bns_id_agent.py` | Same docstring note |
| `src/tests/test_transposition_table.py` | Replace `State(...)` with `TTState(...)` where `_hashes` is accessed |

### Why `undo()` peeks before calling `super()`

`State.undo()` pops `history` immediately on entry. `TTState.undo()` must read `history[-1]` and derive `prev_val` before delegating, otherwise the move is gone.

### Why `player_val` is derived via `.opponent()` in `apply()`

`super().apply()` flips `current_player` before returning, so after the call `self.current_player` is the *next* player. The player who just moved is `self.current_player.opponent()`.

## Verification

1. Run full test suite: `source .venv/bin/activate && pytest src/tests/ -v`
2. Run a game: `python main.py` — confirm normal play with MTDfIDAgent
3. Check that `state._hashes` raises `AttributeError` on plain `State` (confirming removal)
4. Check that `TTState(3)._hashes` is `[0, 0, 0, 0, 0, 0, 0, 0]` (8 zeros) at init
5. Existing TT tests pass without modification (after swapping `State` → `TTState` in those tests)
