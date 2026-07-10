# Refactor: Shared Abstractions, Test Fixtures, and Google Docstrings

**Date:** 2026-05-01
**Status:** Approved

---

## Context

The codebase has grown to 11 agent implementations and 5 heuristic evaluators. Over that time, several small but concrete pieces of logic were copy-pasted across files rather than shared:

- `_DIRECTIONS` and `tanh_normalize` appear verbatim in every heuristic file.
- `_epsilon` and `_terminal_score` are copy-pasted into 4 negamax-family agent files (`NegamaxAgent`, `MTDfAgent`, `BNSAgent`, `NegascoutAgent`), in addition to living in `TTDepthAgent`.
- The `MinimaxAlphaBetaAgent` alpha-beta `_maximize`/`_minimize` skeleton is duplicated wholesale into `MinimaxRewardsAlphaBetaAgent`.
- Some agent tests build board states inline when named puzzle fixtures already exist in `test_helper.py`.
- No production class or method carries a docstring.
- Several attributes that are internal implementation details are exposed as public names (no underscore prefix), and one helper uses Python's name-mangling double-underscore (`__heuristics` in `BNSAgent`) where a single underscore is correct.

This refactor extracts the shared logic into proper shared homes, tightens test fixtures, enforces consistent public/private naming, and adds complete Google docstrings to all production code.

---

## Architecture

### 1. Heuristic Utilities (`src/heuristics/heuristic_utils.py`) — new file

| Symbol | Type | Description |
|---|---|---|
| `DIRECTIONS` | `tuple[tuple[int,int],...]` | The 4 search directions: `((0,1),(1,0),(1,1),(1,-1))`. Replaces the local `_DIRECTIONS` in every heuristic. |
| `tanh_normalize(score_me, score_opp, k)` | helper function | Returns `tanh((score_me - score_opp) / (2 * 4^(k-2)))`. Replaces the inline `tanh(...)` expression in `TaxonomyHeuristic` and `WindowScorerHeuristic`. Returns `0.0` when `k < 2`. |

All five existing heuristic files import `DIRECTIONS` and `tanh_normalize` from this module and remove their local definitions. No behaviour changes.

---

### 2. `NegamaxBaseAgent` (`src/agents/negamax_base_agent.py`) — new file

Extends `BaseAgent`. Provides the shared epsilon/terminal-score logic for every negamax-family agent.

**Attributes:**
- `_epsilon: float` — depth penalty coefficient, set to `1 / (max_depth + 1)` in `__init__`.

**Methods:**
- `__init__(name, max_depth)` — stores name and computes `_epsilon`.
- `_terminal_score(state) -> float` — returns `1.0 - self._epsilon * len(state.history)` on win; `0.0` on draw.

**Inheritors** (each removes its own copy of `_epsilon` and `_terminal_score`):
`NegamaxAgent`, `MTDfAgent`, `BNSAgent`, `NegascoutAgent`, `TTDepthAgent`

---

### 3. `MinimaxBaseAgent` (`src/agents/minimax_base_agent.py`) — new file

Extends `BaseAgent`. Provides the shared maximizer-player attribute and the abstract terminal-score contract for the classic minimax family.

**Attributes:**
- `_maximizer: Player` — the player who maximizes score.

**Methods:**
- `__init__(name, maximizer)` — stores name and maximizer.
- `_terminal_score(state) -> float` — abstract; each subclass defines its own scoring scale.

**Inheritors:**
- `MinimaxAgent(MinimaxBaseAgent)` — `_terminal_score` returns `int` ±1.
- `MinimaxAlphaBetaAgent(MinimaxBaseAgent)` — `_terminal_score` returns `int` ±1; adds alpha-beta `_maximize`/`_minimize`.
- `MinimaxRewardsAlphaBetaAgent(MinimaxAlphaBetaAgent)` — inherits alpha-beta methods unchanged; `__init__` adds `_epsilon`; `_terminal_score` returns float with depth penalty.

---

### Resulting Inheritance Hierarchy

```
BaseAgent
├── RandomAgent
├── NegamaxBaseAgent
│   ├── NegamaxAgent
│   ├── MTDfAgent
│   ├── BNSAgent
│   ├── NegascoutAgent
│   └── TTDepthAgent
│       ├── MTDfIDAgent
│       └── BNSIDAgent
└── MinimaxBaseAgent
    ├── MinimaxAgent
    └── MinimaxAlphaBetaAgent
        └── MinimaxRewardsAlphaBetaAgent
```

---

### 4. Public / Private Visibility

The rule: a name carries a leading `_` if and only if it is an internal implementation detail that callers outside the class (or module) should not depend on.

**Changes required:**

| Symbol | Current | Change | Reason |
|---|---|---|---|
| `Board.board` (numpy array) | public | rename to `Board._grid` | Internal storage; all reads/writes already go through public methods. All call-sites (`state.py`, heuristics, `forced_move.py`) updated to `._grid`. |
| `State.state_count` | public | rename to `State._state_count` | Internal monotonic counter; not part of the `State` contract. |
| `State.visited` | public | rename to `State._visited` | Internal Zobrist hash set; callers have no reason to inspect it. |
| `MinimaxAgent.maximizer` | public | rename to `_maximizer` | Internal config; absorbed into `MinimaxBaseAgent._maximizer`. |
| `MinimaxAlphaBetaAgent.maximizer` | public | → `MinimaxBaseAgent._maximizer` | Same. |
| `MinimaxRewardsAlphaBetaAgent.maximizer` | public | → `MinimaxBaseAgent._maximizer` | Same. |
| `BNSAgent.__heuristics` | double-underscore | rename to `_heuristics` | Double-underscore triggers Python name-mangling; single underscore is the correct Python convention for "private". |

**Attributes that stay public** (external callers legitimately use them):

| Symbol | Rationale |
|---|---|
| `Board.n`, `Board.k` | Board dimensions used by agents and heuristics. |
| `State.board` | `Board` object accessed by agents: `state.board.get_candidate_cells(...)`. |
| `State.current_player` | Agents read this every turn. |
| `State.history` | Agents pass it to `get_candidate_cells`; also used in terminal score. |
| `State.candidate_d` | Agents pass it to `get_candidate_cells`. |
| `BaseAgent.name` | Display / logging. |
| `Manipulator.COORD_TRANSFORMS`, `TRANSFORM_COUNT` | Class-level constants used by `State`. |

**Test updates:** Any test that directly asserts `.maximizer`, `.state_count`, or `.visited` must be updated to use the new `_`-prefixed names. Tests in `test_core.py` that verify `state_count` behaviour (there are ~5) will reference `_state_count`; the one that checks `len(s.visited)` will reference `_visited`. Tests that assert `agent.maximizer` in the three minimax test files will reference `agent._maximizer`.

---

### 5. Test Fixtures

All agent test files must source board positions exclusively from the named puzzle fixtures in `src/tests/test_helper.py`:

| Fixture | Board | Purpose |
|---|---|---|
| `PUZZLE_3X3` | 3×3 k=3, 6 moves in | Primary correctness check, 3-ply tree |
| `PUZZLE_4X4` | 4×4 k=4, 14 moves in | Immediate win threat, 2-ply |
| `PUZZLE_4X4_BLOCK` | 4×4 k=3, 3 moves in | Blocking / defensive logic |
| `PUZZLE_5X5` | 5×5 k=3, 22 moves in | Larger board, 3-ply |

Any inline `state_with_moves(...)` call that replicates one of these positions is replaced with the corresponding named puzzle. `state_with_moves` remains available for truly ad-hoc positions not covered by a puzzle.

---

### 6. Google Docstrings

Every class, class-level attribute block, and method in the following directories receives a complete Google-style docstring:

- `src/core/` (6 files)
- `src/agents/` (13 files including the 2 new base classes)
- `src/heuristics/` (7 files including the new utils module)

Format:
```python
def method(self, arg: Type) -> ReturnType:
    """One-line summary.

    Args:
        arg: Description.

    Returns:
        Description.
    """
```

Test files are excluded.

---

## Data Flow

No data-flow changes. This refactor is purely structural: the same computations happen in the same order; only their physical location changes (extracted to shared bases/helpers).

---

## Error Handling

No new error handling is introduced. Existing validation (e.g. `validate` in `BaseAgent`, bounds checking in `Board`) is unchanged.

---

## Verification

1. `source .venv/bin/activate && pytest src/tests/ -v` — full suite must pass with zero regressions.
2. Confirm `_DIRECTIONS` / `_epsilon` / `_terminal_score` no longer appear duplicated across files (`grep -r "_epsilon" src/agents/`).
3. Confirm no public `.maximizer`, `.state_count`, `.visited`, or `.board` (numpy array) attributes remain (`grep -rn "\.maximizer\b\|\.state_count\b\|\.visited\b\|board\.board\b" src/`).
4. Confirm every agent test file imports from `test_helper` and uses named puzzle fixtures for its correctness assertions.
5. Spot-check three docstrings — one from `src/core`, one from `src/agents`, one from `src/heuristics` — for correct Google format.
