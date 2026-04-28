# MTD(f) Agent Design

## Context

All prior agents in `src/agents/` lack a transposition table. MTD(f) is the
natural next step: it reuses results across repeated null-window probes via a
per-search TT, so total work is only slightly more than a single full-window
alpha-beta call. The test suite at `src/tests/test_mtdf_agent.py` is already
written and defines the complete interface.

---

## Files

| File | Change |
|------|--------|
| `src/core/transposition_table.py` | New — `TranspositionTable` class |
| `src/tests/test_transposition_table.py` | New — TT unit tests |
| `src/agents/mtdf_agent.py` | New — `MTDfAgent` |
| `src/tests/test_mtdf_agent.py` | Update — replace `tt: dict = {}` with `tt = TranspositionTable()` |

---

## TranspositionTable (`src/core/transposition_table.py`)

No `TTFlag` enum. Each entry stores two bounds directly, matching Plaat's
original pseudocode (`n.lowerbound`, `n.upperbound`).

```
Entry: (lowerbound: float, upperbound: float, best_move: tuple[int,int] | None)
Key:   int  (state._hash — incremental Zobrist, maintained by State)
```

### API

```python
class TranspositionTable:
    def lookup(self, key: int) -> tuple[float, float, tuple[int,int] | None] | None
    def store(self, key: int, lower: float, upper: float,
              best_move: tuple[int,int] | None) -> None
    def best_move(self, key: int) -> tuple[int,int] | None
    def __len__(self) -> int
```

- `lookup` returns `None` on a miss.
- `store` overwrites any existing entry at `key`.
- `best_move` is a convenience accessor used by `act`.
- `__len__` supports `assert len(tt) > 0` in tests.

---

## TranspositionTable Tests (`src/tests/test_transposition_table.py`)

```
TestInit
  test_empty_on_creation              len == 0

TestStore
  test_len_increases_on_new_key       store one entry → len == 1
  test_len_two_distinct_keys          store two keys → len == 2
  test_overwrite_same_key_no_len_change  store twice at same key → len stays 1

TestLookup
  test_miss_returns_none              unseen key → None
  test_hit_returns_full_tuple         (lower, upper, best_move) round-trips
  test_lower_bound_stored             lower=-0.5, upper=+inf → retrieved exactly
  test_upper_bound_stored             lower=-inf, upper=0.5 → retrieved exactly
  test_exact_stored                   lower==upper → both retrieved correctly
  test_none_best_move_round_trips     best_move=None stored and retrieved as None

TestBestMove
  test_miss_returns_none              unseen key → None
  test_hit_returns_move               store (1,2) → best_move() == (1,2)
  test_none_best_move_returns_none    store None best_move → best_move() returns None

TestOverwrite
  test_overwrite_updates_lower        second store wins on lookup (lower field)
  test_overwrite_updates_upper        second store wins on lookup (upper field)
  test_overwrite_updates_best_move    best_move updated on overwrite
```

---

## MTDfAgent (`src/agents/mtdf_agent.py`)

Extends `BaseAgent`. Constructor, `_terminal_score`, and `_epsilon` are
identical to `NegamaxAgent`.

### `_negamax_tt(state, alpha, beta, tt) → float`

Negamax with TT, following Plaat's `AlphaBetaWithMemory` adapted to negamax
convention (single recursive function; scores from current player's view).

```
h = state._hash
entry = tt.lookup(h)
if entry:
    lb, ub, _ = entry
    if lb >= beta:   return lb          # fail-high cutoff
    if ub <= alpha:  return ub          # fail-low cutoff
    alpha = max(alpha, lb)              # tighten window from stored bounds
    beta  = min(beta,  ub)

original_alpha, original_beta = alpha, beta

if state.is_terminal():
    g = -_terminal_score(state)
    tt.store(h, g, g, None)            # exact
    return g

g = -∞; best_move = None
for each empty cell (row, col):
    state.apply(row, col)
    score = -_negamax_tt(state, -beta, -alpha, tt)
    state.undo()
    if score > g:
        g = score; best_move = (row, col)
    alpha = max(alpha, g)
    if g >= beta: break                # cutoff

# Store bounds (mirrors pseudocode store logic)
lb, ub = tt.lookup(h)[:2] if tt.lookup(h) else (-∞, +∞)
if g <= original_alpha: ub = g
elif g >= original_beta: lb = g
else: lb = ub = g
tt.store(h, lb, ub, best_move)

return g
```

### `_mtdf(state, f, tt) → float`

Null-window step = `self._epsilon` (Option A: all scores are multiples of
`_epsilon`, so no score can fall strictly inside the window).

```
lower = -∞; upper = +∞
for _ in range(50):
    beta = f if f > lower else lower + self._epsilon
    g = _negamax_tt(state, beta - self._epsilon, beta, tt)
    if g < beta: upper = g
    else:        lower = g
    f = g
    if lower >= upper: break
return f
```

### `act(state) → (row, col)`

```
tt = TranspositionTable()
_mtdf(state, 0.0, tt)
best = tt.best_move(state._hash)
if best is not None: return best
# Fallback (shouldn't trigger):
sweep empty cells with _negamax_tt(child, -∞, +∞, tt); return best
```

### `test_mtdf_agent.py` changes

Add import: `from src.core.transposition_table import TranspositionTable`  
Replace every `tt: dict = {}` with `tt = TranspositionTable()`  
No logic changes — `len(tt)` works via `__len__`.

---

## Verification

```bash
source .venv/bin/activate
python -m pytest src/tests/test_transposition_table.py -v   # 16 tests
python -m pytest src/tests/test_mtdf_agent.py -v            # 22 tests
```
