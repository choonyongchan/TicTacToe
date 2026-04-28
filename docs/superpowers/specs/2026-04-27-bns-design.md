# BestNodeSearch (BNS) Agent Design

**Date:** 2026-04-27

## Overview

Implement `BNSAgent`, a TicTacToe AI agent using the Best Node Search algorithm. BNS finds the best *move* (not the exact game value) by iteratively narrowing an α/β window with null-window alpha-beta probes, terminating when exactly one candidate move remains or the window is too narrow to distinguish adjacent scores.

## Score System

Scores are floats in `[-1.0, 1.0]`:
- Win: `1.0 - epsilon * len(history)` (depth-discounted; earlier wins score higher)
- Draw: `0.0`
- Loss: negated from opponent's perspective
- `epsilon = 1 / (max_depth + 1)`

Minimum gap between distinct scores is `epsilon`. Adjacent scores differ by at least `epsilon`.

## Architecture

`BNSAgent` extends `BaseAgent`. Constructor takes `max_depth: int`. No `TranspositionTable`.

```
BNSAgent
├── act(state) → tuple[int, int]          # public entry point
├── _bns(state, alpha, beta) → tuple      # outer BNS loop
├── _alphabeta(state, alpha, beta) → float # standard negamax alpha-beta
└── _terminal_score(state) → float        # identical to all other agents
```

## Core Algorithm

### `nextGuess` heuristic: midpoint bisection

```
test = (alpha + beta) / 2
```

**Rationale:** TicTacToe scores are trimodal (win ≈ +1, draw = 0, loss ≈ -1). Midpoint naturally separates winning moves from draw/loss moves on the first probe (test = 0 for a full [-1, 1] window). Binary search halves the window each iteration, which is optimal when the score distribution is non-uniform. `subtreeCount` is unused — it only matters for the standard BNS formula, which assumes uniform distribution.

### `_bns` loop

```python

def _nextguess(self, alpha, beta):
    return (alpha + beta) / 2

def _bns(self, state, alpha, beta):
    best_node = None
    while True:
        test = self._nextguess(alpha, beta)
        better_count = 0
        for row, col in state.board.get_empty_cells():
            state.apply(row, col)
            val = -self._alphabeta(state, -test, -(test - self._epsilon))
            state.undo()
            if val >= test:
                better_count += 1
                best_node = (row, col)
        if better_count > 0:
            alpha = test
        else:
            beta = test
        if beta - alpha < 2 * self._epsilon or better_count == 1:
            break
    return best_node
```

**Null-window probe:** `alphabeta(child, -(test), -(test - epsilon))` — width `epsilon` matches the minimum score granularity (analogous to the integer `test - 1` step in the original pseudocode).

**Termination:** `beta - alpha < 2 * epsilon` (window too narrow to distinguish adjacent float scores, analogous to `β − α < 2` for integer scores) OR `better_count == 1` (exactly one candidate remains).

**Alpha/beta update:**
- `better_count > 0` → `alpha = test` (best score is ≥ test)
- `better_count == 0` → `beta = test` (best score is < test)

### `act()`

Calls `_bns(state, -1.0, 1.0)` — safe full window since all scores are in `[-1.0, 1.0]`.

### `_alphabeta`

Standard negamax alpha-beta, no TT, no PVS re-search:

```python
def _alphabeta(self, state, alpha, beta):
    if state.is_terminal():
        return -self._terminal_score(state)
    best = NEGATIVE_INFINITY
    for row, col in state.board.get_empty_cells():
        state.apply(row, col)
        score = -self._alphabeta(state, -beta, -alpha)
        state.undo()
        if score > best:
            best = score
        if best >= beta:
            break
        if best > alpha:
            alpha = best
    return best
```

## Testing

File: `src/tests/test_bns_agent.py`

| Class | What it covers |
|---|---|
| `TestInit` | name, epsilon values |
| `TestTerminalScore` | draw=0, win>0, depth-discounted, earlier > later |
| `TestAlphabeta` | full-window exact value, fail-low, fail-high on PUZZLE_3X3 |
| `TestBns` | correct best node on PUZZLE_3X3; alpha update when child passes; beta update when none pass |
| `TestActWinningMove` | takes immediate win (row and column) |
| `TestActBlockingMove` | blocks opponent win |
| `TestActSmallTree` | PUZZLE_3X3 best move |
| `TestActLargerBoards` | PUZZLE_4X4, PUZZLE_5X5 |
| `TestActValidMove` | in-bounds, empty cell on fresh state |
| `TestAgreesWithNegamax` | identical moves to NegamaxAgent on PUZZLE_3X3, empty board, PUZZLE_4X4 |
